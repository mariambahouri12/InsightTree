from __future__ import annotations

import re

from application.ports.llm_port import LLMPort
from config.settings import TreeSearchSettings
from domain.entities import Evidence
from domain.tree import Branch, EvidenceTree


_INITIAL_QUESTIONS_SYSTEM = """
Break down the complex question into independent investigation questions
that are directly useful for answering it.

IMPORTANT:
- The questions must cover the different aspects necessary
  to answer the original question.
- Do not create generic questions.
- Do not create questions that require information absent
  from the context if that information is not necessary.
- Each question must be testable through evidence.
- For a question involving a numerical trend and its causes,
  cover in particular:
  1. the relevant numerical values;
  2. the factors explaining the trend;
  3. the elements that help distinguish a structural cause
     from a simple mechanical effect when the document allows it.

Generate a maximum of 4 questions.

Strict JSON:
{"questions":["..."]}
""".strip()


_CHILD_QUESTIONS_SYSTEM = """
Based on the newly discovered evidence in ONE branch,
identify only the information needs that remain genuinely unresolved.

Do NOT generate a new question if the available evidence
already allows the branch to be answered.

Never repeat a search that has already been performed.

A child question must be directly justified by missing information
or a contradiction discovered in the evidence.

Strict JSON:
{"questions":["..."]}
""".strip()


_FILTER_SYSTEM = """
Filter the candidate questions.

Keep only questions that are:
- genuinely necessary;
- non-redundant;
- sufficiently precise;
- evidence-oriented;
- useful for answering the original question.

Remove generic, vague, or redundant questions.

Strict JSON:
{"kept_questions":["..."]}
""".strip()


class BranchExplorationService:
    def __init__(
        self,
        llm: LLMPort,
        settings: TreeSearchSettings,
    ) -> None:
        self._llm = llm
        self._settings = settings

    def generate_initial_branch_questions(
        self,
        original_question: str,
        initial_evidence: list[Evidence],
    ) -> list[str]:

        result = self._llm.generate_json(
            f"Original question:\n"
            f"{original_question}\n\n"
            f"Initial evidence:\n"
            f"{self._format_evidence(initial_evidence)}",
            system=_INITIAL_QUESTIONS_SYSTEM,
        )

        candidates = self._normalize(
            result.get("questions")
        )

        candidates = candidates[
            : self._settings.max_initial_branches
        ]

        if len(candidates) > 1:
            candidates = self._filter_questions(
                candidates,
                [],
            )

        return candidates[
            : self._settings.max_initial_branches
        ]

    def generate_child_questions(
        self,
        branch: Branch,
        tree: EvidenceTree,
    ) -> list[str]:

        if branch.depth >= self._settings.max_tree_depth:
            return []

        if not branch.evidence:
            return []

        result = self._llm.generate_json(
            f"Branch question:\n"
            f"{branch.question}\n\n"
            f"Context:\n"
            f"{branch.context_summary or '(none)'}\n\n"
            f"Evidence:\n"
            f"{self._format_evidence(branch.evidence[-8:])}",
            system=_CHILD_QUESTIONS_SYSTEM,
        )

        candidates = self._normalize(
            result.get("questions")
        )

        candidates = candidates[
            : self._settings.max_children_per_branch
        ]

        return self._filter_questions(
            candidates,
            tree.all_questions(),
        )

    def priority(
        self,
        relevance: float,
        information_gain: float,
    ) -> float:

        return (
            0.6 * max(
                0.0,
                min(1.0, relevance),
            )
            + 0.4 * max(
                0.0,
                min(1.0, information_gain),
            )
        )

    def _filter_questions(
        self,
        candidates: list[str],
        existing: list[str],
    ) -> list[str]:

        existing_norm = {
            self._normalize_question(q)
            for q in existing
        }

        local_seen: set[str] = set()
        unique: list[str] = []

        for question in candidates:

            key = self._normalize_question(
                question
            )

            if (
                key
                and key not in existing_norm
                and key not in local_seen
            ):
                local_seen.add(key)
                unique.append(question)

        if not unique:
            return []

        result = self._llm.generate_json(
            f"Candidate questions:\n"
            f"{chr(10).join('- ' + q for q in unique)}\n\n"
            f"Existing questions:\n"
            f"{chr(10).join('- ' + q for q in existing) or '(none)'}",
            system=_FILTER_SYSTEM,
        )

        kept = self._normalize(
            result.get("kept_questions")
        )

        kept_norm = {
            self._normalize_question(q)
            for q in kept
        }

        filtered = [
            q
            for q in unique
            if self._normalize_question(q)
            in kept_norm
        ]

        return filtered or unique

    @staticmethod
    def _normalize(values) -> list[str]:

        if not isinstance(values, list):
            return []

        result = []
        seen = set()

        for value in values:

            text = str(value).strip()

            key = (
                BranchExplorationService
                ._normalize_question(text)
            )

            if text and key not in seen:
                seen.add(key)
                result.append(text)

        return result

    @staticmethod
    def _normalize_question(
        question: str,
    ) -> str:

        return re.sub(
            r"\s+",
            " ",
            question.strip().lower(),
        )

    @staticmethod
    def _format_evidence(
        evidence_list: list[Evidence],
    ) -> str:

        if not evidence_list:
            return "(none)"

        return "\n---\n".join(
            f"[{e.citation_label()}] "
            f"{e.text[:700]}"
            for e in evidence_list
        )
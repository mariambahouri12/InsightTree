from __future__ import annotations

import re

from application.ports.llm_port import LLMPort
from domain.entities import Hypothesis
from domain.tree import EvidenceTree


_GAPS_SYSTEM = """
Analyze the entire investigation and identify only the information gaps
that still prevent answering the original question.

Information already present in the evidence must not be considered a gap.

If there are no significant gaps, return an empty list.

Strict JSON:
{"gaps":["..."]}
""".strip()


_NEW_QUESTIONS_SYSTEM = """
Based on the genuinely identified information gaps, generate new
investigation questions.

Rules:
- a question must address a specific gap;
- do not repeat any question that has already been explored;
- do not generate generic questions;
- do not generate a question if the answer is already present in
  the available evidence.

Strict JSON:
{"questions":["..."]}
""".strip()


class CrossBranchReasoningService:
    def __init__(
        self,
        llm: LLMPort,
    ) -> None:
        self._llm = llm

    def identify_information_gaps(
        self,
        original_question: str,
        tree: EvidenceTree,
        hypotheses: list[Hypothesis],
    ) -> list[str]:

        branches = "\n".join(
            f"- {b.question}: "
            f"{b.conclusion or '(no conclusion)'}"
            for b in tree.nodes.values()
            if b.depth > 0
        )

        hypotheses_text = "\n".join(
            f"- {h.statement} | "
            f"confidence={h.confidence:.2f} | "
            f"uncertainty={h.remaining_uncertainty:.2f}"
            for h in hypotheses
        ) or "(none)"

        result = self._llm.generate_json(
            f"Original question:\n"
            f"{original_question}\n\n"
            f"Hypotheses:\n"
            f"{hypotheses_text}\n\n"
            f"Branches:\n"
            f"{branches or '(none)'}\n\n"
            f"Discovered facts:\n"
            f"{chr(10).join(tree.discovered_facts) or '(none)'}\n\n"
            f"Global evidence:\n"
            f"{self._format_global_evidence(tree)}",
            system=_GAPS_SYSTEM,
        )

        return self._clean(
            result.get("gaps")
        )

    def generate_new_questions(
        self,
        original_question: str,
        tree: EvidenceTree,
        gaps: list[str],
    ) -> list[str]:

        if not gaps:
            return []

        existing = tree.all_questions()

        result = self._llm.generate_json(
            f"Original question:\n"
            f"{original_question}\n\n"
            f"Information gaps:\n"
            f"{chr(10).join('- ' + g for g in gaps)}\n\n"
            f"Previously explored questions:\n"
            f"{chr(10).join('- ' + q for q in existing)}\n\n"
            f"Available evidence:\n"
            f"{self._format_global_evidence(tree)}",
            system=_NEW_QUESTIONS_SYSTEM,
        )

        existing_norm = {
            self._norm(q)
            for q in existing
        }

        return [
            q
            for q in self._clean(
                result.get("questions")
            )
            if self._norm(q) not in existing_norm
        ]

    @staticmethod
    def _clean(values) -> list[str]:

        if not isinstance(values, list):
            return []

        seen = set()
        result = []

        for value in values:

            text = str(value).strip()
            key = CrossBranchReasoningService._norm(
                text
            )

            if text and key not in seen:
                seen.add(key)
                result.append(text)

        return result

    @staticmethod
    def _norm(
        value: str,
    ) -> str:

        return re.sub(
            r"\s+",
            " ",
            value.strip().lower(),
        )

    @staticmethod
    def _format_global_evidence(
        tree: EvidenceTree,
    ) -> str:

        if not tree.global_evidence_pool:
            return "(none)"

        return "\n---\n".join(
            f"[{e.citation_label()}] "
            f"{e.text[:700]}"
            for e in tree.global_evidence_pool[-12:]
        )
from __future__ import annotations

import re

from application.ports.llm_port import LLMPort
from domain.entities import Hypothesis
from domain.tree import EvidenceTree


_GAPS_SYSTEM = """
Analyse l'investigation entière et identifie uniquement les lacunes
d'information qui empêchent encore de répondre à la question originale.

Une information déjà présente dans les preuves ne doit pas être
considérée comme une lacune.

S'il n'existe aucune lacune importante, retourne une liste vide.

JSON strict:
{"gaps":["..."]}
""".strip()


_NEW_QUESTIONS_SYSTEM = """
À partir des lacunes réellement identifiées, génère de nouvelles
questions d'investigation.

Règles:
- une question doit résoudre une lacune précise;
- ne répète aucune question déjà explorée;
- ne génère pas de question générique;
- ne génère pas de question si la réponse est déjà présente dans
  les preuves disponibles.

JSON strict:
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
            f"{b.conclusion or '(sans conclusion)'}"
            for b in tree.nodes.values()
            if b.depth > 0
        )

        hypotheses_text = "\n".join(
            f"- {h.statement} | "
            f"confidence={h.confidence:.2f} | "
            f"uncertainty={h.remaining_uncertainty:.2f}"
            for h in hypotheses
        ) or "(aucune)"

        result = self._llm.generate_json(
            f"Question originale:\n"
            f"{original_question}\n\n"
            f"Hypothèses:\n"
            f"{hypotheses_text}\n\n"
            f"Branches:\n"
            f"{branches or '(aucune)'}\n\n"
            f"Faits découverts:\n"
            f"{chr(10).join(tree.discovered_facts) or '(aucun)'}\n\n"
            f"Preuves globales:\n"
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
            f"Question originale:\n"
            f"{original_question}\n\n"
            f"Lacunes:\n"
            f"{chr(10).join('- ' + g for g in gaps)}\n\n"
            f"Questions déjà explorées:\n"
            f"{chr(10).join('- ' + q for q in existing)}\n\n"
            f"Preuves disponibles:\n"
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
            return "(aucune)"

        return "\n---\n".join(
            f"[{e.citation_label()}] "
            f"{e.text[:700]}"
            for e in tree.global_evidence_pool[-12:]
        )
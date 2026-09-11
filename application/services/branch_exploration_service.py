from __future__ import annotations

import re

from application.ports.llm_port import LLMPort
from config.settings import TreeSearchSettings
from domain.entities import Evidence
from domain.tree import Branch, EvidenceTree


_INITIAL_QUESTIONS_SYSTEM = """
Décompose la question complexe en questions d'investigation
indépendantes et directement utiles à la réponse.

IMPORTANT:
- Les questions doivent couvrir les différents aspects nécessaires
  pour répondre à la question originale.
- Ne crée pas de questions génériques.
- Ne crée pas de questions qui demandent des informations absentes
  du contexte si elles ne sont pas nécessaires.
- Chaque question doit pouvoir être testée par des preuves.
- Pour une question portant sur une évolution chiffrée et ses causes,
  couvre notamment:
  1. les valeurs numériques concernées;
  2. les facteurs expliquant l'évolution;
  3. les éléments permettant de distinguer une cause durable
     d'un simple effet mécanique lorsque le document le permet.

Génère au maximum 4 questions.

JSON strict:
{"questions":["..."]}
""".strip()


_CHILD_QUESTIONS_SYSTEM = """
À partir des preuves nouvellement découvertes dans UNE branche,
identifie uniquement les besoins d'information réellement non résolus.

Ne génère PAS une nouvelle question si les preuves disponibles
permettent déjà de répondre à la branche.

Ne répète jamais une recherche déjà effectuée.

Une question enfant doit être directement justifiée par une information
manquante ou une contradiction découverte dans les preuves.

JSON strict:
{"questions":["..."]}
""".strip()


_FILTER_SYSTEM = """
Filtre les questions candidates.

Conserve uniquement les questions:
- réellement nécessaires;
- non redondantes;
- suffisamment précises;
- orientées vers des preuves;
- utiles pour répondre à la question originale.

Supprime les questions génériques, vagues ou redondantes.

JSON strict:
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
            f"Question originale:\n"
            f"{original_question}\n\n"
            f"Preuves initiales:\n"
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
            f"Question de branche:\n"
            f"{branch.question}\n\n"
            f"Contexte:\n"
            f"{branch.context_summary or '(aucun)'}\n\n"
            f"Preuves:\n"
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
            f"Questions candidates:\n"
            f"{chr(10).join('- ' + q for q in unique)}\n\n"
            f"Questions existantes:\n"
            f"{chr(10).join('- ' + q for q in existing) or '(aucune)'}",
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
            return "(aucune)"

        return "\n---\n".join(
            f"[{e.citation_label()}] "
            f"{e.text[:700]}"
            for e in evidence_list
        )
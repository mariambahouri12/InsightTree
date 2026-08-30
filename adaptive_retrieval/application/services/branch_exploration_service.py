"""Gère la mécanique d'exploration de l'arbre : génération de questions
candidates, questions de suivi issues des preuves d'une branche, dédoublonnage
et calcul de priorité d'exploration."""
from __future__ import annotations

from application.ports.llm_port import LLMPort
from config.settings import TreeSearchSettings
from domain.entities import Evidence
from domain.tree import Branch, EvidenceTree

_INITIAL_BRANCHES_SYSTEM = (
    "Tu es un analyste qui décompose une question complexe en hypothèses "
    "d'investigation. À partir de la question originale et des premières "
    "preuves trouvées, propose des questions de suivi indépendantes et "
    "complémentaires à explorer. Réponds UNIQUEMENT en JSON strict: "
    '{"questions": ["...", "..."]}'
)

_CHILD_QUESTIONS_SYSTEM = (
    "Tu es un analyste d'investigation. À partir des preuves NOUVELLEMENT "
    "découvertes dans une branche d'analyse, génère les prochaines questions "
    "à explorer, UNIQUEMENT si elles découlent logiquement de ces preuves "
    "(pas de questions génériques). Réponds UNIQUEMENT en JSON strict: "
    '{"questions": ["...", "..."]}'
)

_DEDUP_SYSTEM = (
    "Tu reçois une liste de questions candidates et une liste de questions "
    "déjà posées dans l'arbre d'exploration. Retourne uniquement les "
    "questions candidates qui apportent une piste réellement nouvelle "
    "(pas de paraphrase de questions existantes). Réponds UNIQUEMENT en "
    'JSON strict: {"kept_questions": ["...", "..."]}'
)


class BranchExplorationService:
    def __init__(self, llm: LLMPort, settings: TreeSearchSettings) -> None:
        self._llm = llm
        self._settings = settings

    # ------------------------------------------------------------------
    # Génération des branches candidates initiales (à partir de Q0 + E0)
    # ------------------------------------------------------------------
    def generate_initial_branch_questions(self, original_question: str, initial_evidence: list[Evidence]) -> list[str]:
        excerpt = self._format_evidence(initial_evidence)
        prompt = (
            f"Question originale:\n{original_question}\n\n"
            f"Preuves initiales:\n{excerpt}\n\nQuestions de suivi candidates:"
        )
        result = self._llm.generate_json(prompt, system=_INITIAL_BRANCHES_SYSTEM)
        return self._extract_questions(result)

    # ------------------------------------------------------------------
    # Génération des questions enfants à partir des preuves d'une branche
    # ------------------------------------------------------------------
    def generate_child_questions(self, branch: Branch, tree: EvidenceTree) -> list[str]:
        if branch.depth >= self._settings.max_tree_depth:
            return []

        excerpt = self._format_evidence(branch.evidence[-6:])
        prompt = (
            f"Question de la branche:\n{branch.question}\n\n"
            f"Contexte de la branche:\n{branch.context_summary}\n\n"
            f"Preuves découvertes:\n{excerpt}\n\nProchaines questions candidates:"
        )
        result = self._llm.generate_json(prompt, system=_CHILD_QUESTIONS_SYSTEM)
        candidates = self._extract_questions(result)
        candidates = self._deduplicate(candidates, tree)
        return candidates[: self._settings.max_children_per_branch]

    def _deduplicate(self, candidates: list[str], tree: EvidenceTree) -> list[str]:
        if not candidates:
            return []
        existing_questions = [b.question for b in tree.nodes.values()]
        prompt = (
            f"Questions déjà présentes dans l'arbre:\n"
            + "\n".join(f"- {q}" for q in existing_questions)
            + "\n\nQuestions candidates:\n"
            + "\n".join(f"- {q}" for q in candidates)
            + "\n\nQuestions candidates réellement nouvelles:"
        )
        result = self._llm.generate_json(prompt, system=_DEDUP_SYSTEM)
        kept = result.get("kept_questions")
        if isinstance(kept, list) and kept:
            return [str(q) for q in kept]
        return candidates

    # ------------------------------------------------------------------
    # Priorisation des branches actives
    # ------------------------------------------------------------------
    def recalculate_priority(self, branch: Branch) -> None:
        depth_penalty = 0.1 * branch.depth
        branch.priority = max(
            (branch.evidence_strength * 0.5)
            + (branch.information_gain * 0.5)
            - depth_penalty,
            0.0,
        )

    @staticmethod
    def _format_evidence(evidence_list: list[Evidence]) -> str:
        return "\n---\n".join(
            f"[{e.citation_label()}] {e.text[:500]}" for e in evidence_list
        )

    @staticmethod
    def _extract_questions(result: dict) -> list[str]:
        questions = result.get("questions")
        if not isinstance(questions, list):
            return []
        # Nettoyage + dédoublonnage simple préservant l'ordre
        seen: set[str] = set()
        cleaned: list[str] = []
        for q in questions:
            q_str = str(q).strip()
            key = q_str.lower()
            if q_str and key not in seen:
                seen.add(key)
                cleaned.append(q_str)
        return cleaned

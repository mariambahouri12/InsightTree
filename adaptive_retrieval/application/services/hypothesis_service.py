"""Maintient et fait évoluer les hypothèses explicatives au fil de
l'exploration de l'arbre, et détecte les contradictions entre branches."""
from __future__ import annotations

from application.ports.llm_port import LLMPort
from domain.entities import Evidence, Hypothesis

_UPDATE_HYPOTHESES_SYSTEM = (
    "Tu maintiens une liste d'hypothèses explicatives pour une question "
    "d'analyse. À partir des hypothèses existantes et des nouvelles preuves, "
    "mets à jour la liste : garde les hypothèses toujours pertinentes, "
    "ajoute-en de nouvelles si nécessaire, ajuste leur confiance (0..1). "
    "Réponds UNIQUEMENT en JSON strict: "
    '{"hypotheses": [{"statement": "...", "confidence": <0..1>}]}'
)

_CONTRADICTIONS_SYSTEM = (
    "Tu analyses une liste d'hypothèses pour détecter des contradictions "
    "logiques entre elles. Réponds UNIQUEMENT en JSON strict: "
    '{"contradictions": [{"hypothesis_a": "...", "hypothesis_b": "...", "explanation": "..."}]}'
)


class HypothesisService:
    def __init__(self, llm: LLMPort) -> None:
        self._llm = llm

    def update_hypotheses(
        self,
        original_question: str,
        existing_hypotheses: list[Hypothesis],
        new_evidence: list[Evidence],
    ) -> list[Hypothesis]:
        if not new_evidence:
            return existing_hypotheses

        existing_repr = "\n".join(
            f"- {h.statement} (confiance actuelle: {h.confidence:.2f})" for h in existing_hypotheses
        ) or "(aucune)"
        evidence_repr = "\n---\n".join(f"[{e.citation_label()}] {e.text[:500]}" for e in new_evidence)

        prompt = (
            f"Question originale:\n{original_question}\n\n"
            f"Hypothèses existantes:\n{existing_repr}\n\n"
            f"Nouvelles preuves:\n{evidence_repr}\n\nHypothèses mises à jour:"
        )
        result = self._llm.generate_json(prompt, system=_UPDATE_HYPOTHESES_SYSTEM)
        raw_hypotheses = result.get("hypotheses", [])
        if not isinstance(raw_hypotheses, list):
            return existing_hypotheses

        by_statement = {h.statement.lower(): h for h in existing_hypotheses}
        updated: list[Hypothesis] = []
        for item in raw_hypotheses:
            statement = str(item.get("statement", "")).strip()
            if not statement:
                continue
            confidence = float(item.get("confidence", 0.0))
            existing = by_statement.get(statement.lower())
            if existing:
                existing.confidence = confidence
                existing.supporting_evidence_ids.extend(e.id for e in new_evidence)
                updated.append(existing)
            else:
                updated.append(
                    Hypothesis(
                        statement=statement,
                        confidence=confidence,
                        supporting_evidence_ids=[e.id for e in new_evidence],
                    )
                )
        return updated

    def check_contradictions(self, hypotheses: list[Hypothesis]) -> list[dict[str, str]]:
        if len(hypotheses) < 2:
            return []
        hypotheses_repr = "\n".join(f"- {h.statement}" for h in hypotheses)
        prompt = f"Hypothèses:\n{hypotheses_repr}\n\nContradictions détectées:"
        result = self._llm.generate_json(prompt, system=_CONTRADICTIONS_SYSTEM)
        contradictions = result.get("contradictions", [])
        return contradictions if isinstance(contradictions, list) else []

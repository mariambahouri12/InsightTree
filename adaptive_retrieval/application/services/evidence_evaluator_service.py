"""Évaluation des preuves récupérées : pertinence, force, suffisance et
gain d'information. Combine des heuristiques légères (rapides) et un
jugement LLM (qualitatif) pour les décisions structurantes."""
from __future__ import annotations

from application.ports.llm_port import LLMPort
from domain.entities import Evidence

_RELEVANCE_SYSTEM = (
    "Tu évalues si des extraits de documents sont pertinents pour répondre à "
    "une question. Réponds UNIQUEMENT en JSON strict au format: "
    '{"relevant": true|false, "relevance_score": <0..1>, "reason": "..."}'
)

_SUFFICIENCY_SYSTEM = (
    "Tu évalues si les preuves rassemblées suffisent pour répondre de façon "
    "fiable et complète à une question. Réponds UNIQUEMENT en JSON strict: "
    '{"sufficient": true|false, "confidence": <0..1>, "missing_aspects": ["..."]}'
)

_INFO_GAIN_SYSTEM = (
    "Tu évalues si un nouvel extrait de preuve apporte une information "
    "réellement NOUVELLE par rapport aux preuves déjà connues, ou s'il est "
    "redondant. Réponds UNIQUEMENT en JSON strict: "
    '{"information_gain": <0..1>, "reason": "..."}'
)


class EvidenceEvaluatorService:
    def __init__(self, llm: LLMPort, relevance_threshold: float, info_gain_threshold: float) -> None:
        self._llm = llm
        self._relevance_threshold = relevance_threshold
        self._info_gain_threshold = info_gain_threshold

    # ------------------------------------------------------------------
    # Pertinence
    # ------------------------------------------------------------------
    def is_relevant(self, question: str, evidence_list: list[Evidence]) -> bool:
        if not evidence_list:
            return False
        excerpt = self._format_evidence(evidence_list)
        prompt = f"Question:\n{question}\n\nExtraits récupérés:\n{excerpt}\n\nÉvaluation:"
        result = self._llm.generate_json(prompt, system=_RELEVANCE_SYSTEM)
        score = float(result.get("relevance_score", 0.0))
        return bool(result.get("relevant", False)) and score >= self._relevance_threshold

    # ------------------------------------------------------------------
    # Suffisance (arrêt anticipé sur Q0, ou fin de branche)
    # ------------------------------------------------------------------
    def is_sufficient(self, question: str, evidence_list: list[Evidence]) -> tuple[bool, float]:
        if not evidence_list:
            return False, 0.0
        excerpt = self._format_evidence(evidence_list)
        prompt = f"Question:\n{question}\n\nPreuves rassemblées:\n{excerpt}\n\nÉvaluation:"
        result = self._llm.generate_json(prompt, system=_SUFFICIENCY_SYSTEM)
        confidence = float(result.get("confidence", 0.0))
        return bool(result.get("sufficient", False)), confidence

    # ------------------------------------------------------------------
    # Gain d'information
    # ------------------------------------------------------------------
    def information_gain(self, new_evidence: list[Evidence], existing_pool: list[Evidence]) -> float:
        if not new_evidence:
            return 0.0
        if not existing_pool:
            return 1.0

        new_ids = {e.chunk_id for e in new_evidence}
        existing_ids = {e.chunk_id for e in existing_pool}
        overlap_ratio = len(new_ids & existing_ids) / max(len(new_ids), 1)
        if overlap_ratio > 0.8:
            return 0.0  # quasi-doublon exact -> pas la peine d'appeler le LLM

        new_excerpt = self._format_evidence(new_evidence)
        existing_excerpt = self._format_evidence(existing_pool[-8:])
        prompt = (
            f"Preuves déjà connues:\n{existing_excerpt}\n\n"
            f"Nouvelles preuves:\n{new_excerpt}\n\nÉvaluation du gain d'information:"
        )
        result = self._llm.generate_json(prompt, system=_INFO_GAIN_SYSTEM)
        return float(result.get("information_gain", 0.0))

    def is_information_gain_too_low(self, gain: float) -> bool:
        return gain < self._info_gain_threshold

    # ------------------------------------------------------------------
    # Force globale des preuves (heuristique simple, bornée [0,1])
    # ------------------------------------------------------------------
    @staticmethod
    def evidence_strength(evidence_list: list[Evidence]) -> float:
        if not evidence_list:
            return 0.0
        top_scores = sorted((e.score for e in evidence_list), reverse=True)[:5]
        avg = sum(top_scores) / len(top_scores)
        count_bonus = min(len(evidence_list) / 5, 1.0) * 0.2
        return min(avg + count_bonus, 1.0)

    @staticmethod
    def _format_evidence(evidence_list: list[Evidence]) -> str:
        return "\n---\n".join(
            f"[{e.citation_label()}] {e.text[:600]}" for e in evidence_list
        )

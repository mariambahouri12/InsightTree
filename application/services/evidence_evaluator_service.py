from __future__ import annotations

from application.ports.llm_port import LLMPort
from domain.entities import Evidence


_VALIDITY_SYSTEM = """
Évalue la qualité des preuves fournies pour répondre au besoin
d'information.

Une preuve est pertinente si elle contient directement des informations
utiles pour répondre au besoin, même si elle ne répond pas à toute la
question.

NE rejette PAS une preuve simplement parce qu'elle ne contient qu'une
partie de la réponse.

Retourne JSON strict:
{
  "valid": true|false,
  "relevance_score": 0..1,
  "source_reliability": 0..1,
  "reason": "..."
}
""".strip()


_SUFFICIENCY_SYSTEM = """
Évalue si les preuves fournies permettent de répondre de façon fiable
et suffisamment complète à la question.

Important:
- Une question peut être suffisamment répondue par plusieurs preuves
  complémentaires.
- Ne considère pas une preuve insuffisante simplement parce qu'elle
  ne contient qu'un seul élément de la réponse.
- Pour une question factuelle simple, quelques preuves directes peuvent
  suffire.
- Si les chiffres et les facteurs explicatifs sont présents dans les
  preuves, considère la question comme suffisamment couverte.
- Ne demande pas de nouvelles recherches uniquement pour obtenir
  davantage de détails non nécessaires.

Retourne JSON strict:
{
  "sufficient": true|false,
  "confidence": 0..1,
  "missing_aspects": ["..."]
}
""".strip()


_INFO_GAIN_SYSTEM = """
Évalue le gain d'information réellement nouveau apporté par les nouvelles
preuves par rapport aux preuves existantes.

Retourne JSON strict:
{
  "information_gain": 0..1,
  "reason": "..."
}
""".strip()


class EvidenceEvaluatorService:
    def __init__(
        self,
        llm: LLMPort,
        relevance_threshold: float,
        info_gain_threshold: float,
    ) -> None:
        self._llm = llm
        self._relevance_threshold = relevance_threshold
        self._info_gain_threshold = info_gain_threshold

    def is_relevant(
        self,
        information_need: str,
        evidence_list: list[Evidence],
    ) -> bool:

        if not evidence_list:
            return False

        result = self._llm.generate_json(
            f"Besoin:\n{information_need}\n\n"
            f"Preuves:\n"
            f"{self._format_evidence(evidence_list)}",
            system=_VALIDITY_SYSTEM,
        )

        try:
            reliability = float(
                result.get("source_reliability", 0.5)
            )
        except (TypeError, ValueError):
            reliability = 0.5

        try:
            relevance = float(
                result.get("relevance_score", 0.0)
            )
        except (TypeError, ValueError):
            relevance = 0.0

        reliability = max(
            0.0,
            min(1.0, reliability),
        )

        relevance = max(
            0.0,
            min(1.0, relevance),
        )

        for evidence in evidence_list:
            evidence.source_reliability = reliability

        return (
            bool(result.get("valid", False))
            and relevance >= self._relevance_threshold
        )

    def is_sufficient(
        self,
        question: str,
        evidence_list: list[Evidence],
    ) -> tuple[bool, float]:

        if not evidence_list:
            return False, 0.0

        result = self._llm.generate_json(
            f"Question:\n{question}\n\n"
            f"Preuves disponibles:\n"
            f"{self._format_evidence(evidence_list)}",
            system=_SUFFICIENCY_SYSTEM,
        )

        try:
            confidence = float(
                result.get("confidence", 0.0)
            )
        except (TypeError, ValueError):
            confidence = 0.0

        confidence = max(
            0.0,
            min(1.0, confidence),
        )

        return (
            bool(result.get("sufficient", False)),
            confidence,
        )

    def information_gain(
        self,
        new_evidence: list[Evidence],
        existing_pool: list[Evidence],
    ) -> float:

        if not new_evidence:
            return 0.0

        if not existing_pool:
            return 1.0

        existing_keys = {
            (
                e.chunk_id,
                e.text.strip(),
            )
            for e in existing_pool
        }

        new_keys = {
            (
                e.chunk_id,
                e.text.strip(),
            )
            for e in new_evidence
        }

        if new_keys and new_keys.issubset(existing_keys):
            return 0.0

        result = self._llm.generate_json(
            f"Preuves existantes:\n"
            f"{self._format_evidence(existing_pool[-8:])}\n\n"
            f"Nouvelles preuves:\n"
            f"{self._format_evidence(new_evidence)}",
            system=_INFO_GAIN_SYSTEM,
        )

        try:
            gain = float(
                result.get("information_gain", 0.0)
            )
        except (TypeError, ValueError):
            gain = 0.0

        return max(
            0.0,
            min(1.0, gain),
        )

    def is_information_gain_too_low(
        self,
        gain: float,
    ) -> bool:
        return gain < self._info_gain_threshold

    @staticmethod
    def evidence_strength(
        evidence_list: list[Evidence],
    ) -> float:

        if not evidence_list:
            return 0.0

        weighted = [
            max(0.0, min(1.0, e.score))
            * max(
                0.0,
                min(1.0, e.source_reliability),
            )
            for e in evidence_list
        ]

        top = sorted(
            weighted,
            reverse=True,
        )[:5]

        avg = sum(top) / len(top)

        count_bonus = (
            min(len(evidence_list) / 5.0, 1.0)
            * 0.2
        )

        return min(
            avg + count_bonus,
            1.0,
        )

    @staticmethod
    def _format_evidence(
        evidence_list: list[Evidence],
    ) -> str:

        return "\n---\n".join(
            f"[{e.citation_label()} | "
            f"score={e.score:.2f} | "
            f"reliability={e.source_reliability:.2f}] "
            f"{e.text[:800]}"
            for e in evidence_list
        )
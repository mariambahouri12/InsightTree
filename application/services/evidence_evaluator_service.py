from __future__ import annotations

from application.ports.llm_port import LLMPort
from domain.entities import Evidence
from infrastructure.logging.verbose_logger import (
    vlog,
    vlog_evidence,
    vlog_evidence_list,
    vlog_section,
    vlog_subsection,
    vlog_text,
)


_VALIDITY_SYSTEM = """
Evaluate the quality of the evidence provided to address the information
need.

An evidence item is relevant if it directly contains useful information
for addressing the information need, even if it does not answer the entire
question.

Do NOT reject an evidence item simply because it contains only part of
the answer.

Return strict JSON:
{
  "valid": true|false,
  "relevance_score": 0..1,
  "source_reliability": 0..1,
  "reason": "..."
}
""".strip()


_SUFFICIENCY_SYSTEM = """
Evaluate whether the provided evidence allows the question to be answered
reliably and sufficiently completely.

Important:

- A question may be sufficiently answered by several complementary pieces
  of evidence.
- Do not consider an evidence item insufficient simply because it contains
  only one part of the answer.
- For a simple factual question, a few direct pieces of evidence may be
  sufficient.
- If the figures and explanatory factors are present in the evidence,
  consider the question sufficiently covered.
- Do not request additional research only to obtain unnecessary details.
- If all raw values required for a calculation are present, consider the
  information sufficiently covered even though a calculator or another
  tool may still be required to compute the final answer.

Return strict JSON:
{
  "sufficient": true|false,
  "confidence": 0..1,
  "missing_aspects": ["..."]
}
""".strip()


_INFO_GAIN_SYSTEM = """
Evaluate the amount of genuinely new information provided by the new
evidence compared with the existing evidence.

Return strict JSON:
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

    # ------------------------------------------------------------------
    # RELEVANCE
    # ------------------------------------------------------------------

    def is_relevant(
        self,
        information_need: str,
        evidence_list: list[Evidence],
    ) -> bool:
        vlog_section("[RELEVANCE EVALUATION] CHECK")

        vlog(f"[Relevance Evaluation] Information need: {information_need}")
        vlog(
            f"[Relevance Evaluation] Evidence count: "
            f"{len(evidence_list)}"
        )

        if not evidence_list:
            vlog("[Relevance Evaluation] No evidence provided.")
            vlog("[Relevance Evaluation] DECISION: REJECT")
            return False

        self._log_evidence_list(
            evidence_list,
            title="[Relevance Evaluation] INPUT EVIDENCE",
        )

        vlog("[Relevance Evaluation] Calling LLM...")

        result = self._llm.generate_json(
            f"Information need:\n{information_need}\n\n"
            f"Evidence:\n{self._format_evidence(evidence_list)}",
            system=_VALIDITY_SYSTEM,
        )

        vlog_subsection("[Relevance Evaluation] LLM RESULT")

        vlog(
            f"valid: {result.get('valid', False)}"
        )

        vlog(
            f"relevance_score: "
            f"{result.get('relevance_score', 0.0)}"
        )

        vlog(
            f"source_reliability: "
            f"{result.get('source_reliability', 0.5)}"
        )

        vlog(
            f"reason: "
            f"{result.get('reason', 'No reason provided.')}"
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

        valid = bool(result.get("valid", False))

        vlog_subsection("[Relevance Evaluation] NORMALIZED VALUES")

        vlog(f"valid: {valid}")
        vlog(f"relevance_score: {relevance:.4f}")
        vlog(f"source_reliability: {reliability:.4f}")
        vlog(
            f"threshold: {self._relevance_threshold:.4f}"
        )

        # Update source reliability on the evidence objects.
        for evidence in evidence_list:
            evidence.source_reliability = reliability

        accepted = (
            valid
            and relevance >= self._relevance_threshold
        )

        vlog_subsection("[Relevance Evaluation] DECISION")

        if accepted:
            vlog(
                "DECISION: ACCEPT "
                f"(valid={valid}, "
                f"relevance={relevance:.4f} >= "
                f"threshold={self._relevance_threshold:.4f})"
            )
        else:
            if not valid:
                vlog(
                    "DECISION: REJECT — "
                    "LLM marked the evidence as invalid."
                )
            elif relevance < self._relevance_threshold:
                vlog(
                    "DECISION: REJECT — "
                    f"relevance score {relevance:.4f} is below "
                    f"threshold {self._relevance_threshold:.4f}."
                )
            else:
                vlog(
                    "DECISION: REJECT — "
                    "relevance criteria not satisfied."
                )

        return accepted

    # ------------------------------------------------------------------
    # SUFFICIENCY
    # ------------------------------------------------------------------

    def is_sufficient(
        self,
        question: str,
        evidence_list: list[Evidence],
    ) -> tuple[bool, float]:
        vlog_section("[SUFFICIENCY EVALUATION] CHECK")

        vlog(
            f"[Sufficiency Evaluation] Question: {question}"
        )

        vlog(
            f"[Sufficiency Evaluation] Evidence count: "
            f"{len(evidence_list)}"
        )

        if not evidence_list:
            vlog(
                "[Sufficiency Evaluation] No evidence provided."
            )
            vlog(
                "[Sufficiency Evaluation] "
                "DECISION: INSUFFICIENT"
            )
            return False, 0.0

        self._log_evidence_list(
            evidence_list,
            title="[Sufficiency Evaluation] INPUT EVIDENCE",
        )

        vlog(
            "[Sufficiency Evaluation] "
            "Calling LLM..."
        )

        result = self._llm.generate_json(
            f"Question:\n{question}\n\n"
            f"Available evidence:\n"
            f"{self._format_evidence(evidence_list)}",
            system=_SUFFICIENCY_SYSTEM,
        )

        vlog_subsection(
            "[Sufficiency Evaluation] LLM RESULT"
        )

        sufficient = bool(
            result.get("sufficient", False)
        )

        vlog(
            f"sufficient: {sufficient}"
        )

        vlog(
            f"confidence: "
            f"{result.get('confidence', 0.0)}"
        )

        missing_aspects = result.get(
            "missing_aspects",
            [],
        )

        vlog(
            f"missing_aspects: {missing_aspects}"
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

        vlog_subsection(
            "[Sufficiency Evaluation] NORMALIZED RESULT"
        )

        vlog(
            f"sufficient: {sufficient}"
        )

        vlog(
            f"confidence: {confidence:.4f}"
        )

        if missing_aspects:
            vlog(
                "missing_aspects:"
            )

            for aspect in missing_aspects:
                vlog(
                    f"  - {aspect}"
                )
        else:
            vlog(
                "missing_aspects: none"
            )

        vlog_subsection(
            "[Sufficiency Evaluation] DECISION"
        )

        if sufficient:
            vlog(
                "DECISION: SUFFICIENT"
            )
        else:
            vlog(
                "DECISION: INSUFFICIENT"
            )

            if missing_aspects:
                vlog(
                    "Reason: additional information is "
                    "missing for the question."
                )

        return sufficient, confidence

    # ------------------------------------------------------------------
    # INFORMATION GAIN
    # ------------------------------------------------------------------

    def information_gain(
        self,
        new_evidence: list[Evidence],
        existing_pool: list[Evidence],
    ) -> float:
        vlog_section(
            "[INFORMATION GAIN] EVALUATION"
        )

        vlog(
            f"[Information Gain] New evidence count: "
            f"{len(new_evidence)}"
        )

        vlog(
            f"[Information Gain] Existing pool count: "
            f"{len(existing_pool)}"
        )

        if not new_evidence:
            vlog(
                "[Information Gain] "
                "No new evidence."
            )
            vlog(
                "[Information Gain] gain=0.0000"
            )
            return 0.0

        self._log_evidence_list(
            new_evidence,
            title="[Information Gain] NEW EVIDENCE",
        )

        if not existing_pool:
            vlog(
                "[Information Gain] "
                "Existing pool is empty."
            )

            vlog(
                "[Information Gain] "
                "Any evidence is considered completely new."
            )

            vlog(
                "[Information Gain] gain=1.0000"
            )

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

        vlog(
            f"[Information Gain] Existing unique keys: "
            f"{len(existing_keys)}"
        )

        vlog(
            f"[Information Gain] New unique keys: "
            f"{len(new_keys)}"
        )

        if new_keys and new_keys.issubset(
            existing_keys
        ):
            vlog(
                "[Information Gain] "
                "All new evidence items are already present "
                "in the existing pool."
            )

            vlog(
                "[Information Gain] "
                "gain=0.0000"
            )

            return 0.0

        vlog(
            "[Information Gain] "
            "Calling LLM to estimate genuinely new information..."
        )

        result = self._llm.generate_json(
            f"Existing evidence:\n"
            f"{self._format_evidence(existing_pool[-8:])}\n\n"
            f"New evidence:\n"
            f"{self._format_evidence(new_evidence)}",
            system=_INFO_GAIN_SYSTEM,
        )

        vlog_subsection(
            "[Information Gain] LLM RESULT"
        )

        vlog(
            f"information_gain: "
            f"{result.get('information_gain', 0.0)}"
        )

        vlog(
            f"reason: "
            f"{result.get('reason', 'No reason provided.')}"
        )

        try:
            gain = float(
                result.get(
                    "information_gain",
                    0.0,
                )
            )
        except (TypeError, ValueError):
            gain = 0.0

        gain = max(
            0.0,
            min(1.0, gain),
        )

        vlog_subsection(
            "[Information Gain] NORMALIZED RESULT"
        )

        vlog(
            f"gain: {gain:.4f}"
        )

        vlog(
            f"threshold: "
            f"{self._info_gain_threshold:.4f}"
        )

        too_low = self.is_information_gain_too_low(
            gain
        )

        if too_low:
            vlog(
                "DECISION: LOW INFORMATION GAIN"
            )

            vlog(
                f"Reason: gain {gain:.4f} < "
                f"threshold {self._info_gain_threshold:.4f}"
            )
        else:
            vlog(
                "DECISION: ACCEPTABLE INFORMATION GAIN"
            )

            vlog(
                f"Reason: gain {gain:.4f} >= "
                f"threshold {self._info_gain_threshold:.4f}"
            )

        return gain

    # ------------------------------------------------------------------
    # INFORMATION GAIN THRESHOLD
    # ------------------------------------------------------------------

    def is_information_gain_too_low(
        self,
        gain: float,
    ) -> bool:
        return gain < self._info_gain_threshold

    # ------------------------------------------------------------------
    # EVIDENCE STRENGTH
    # ------------------------------------------------------------------

    @staticmethod
    def evidence_strength(
        evidence_list: list[Evidence],
    ) -> float:
        if not evidence_list:
            vlog(
                "[Evidence Strength] "
                "No evidence -> strength=0.0000"
            )
            return 0.0

        weighted = [
            max(
                0.0,
                min(1.0, e.score),
            )
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
            min(
                len(evidence_list) / 5.0,
                1.0,
            )
            * 0.2
        )

        strength = min(
            avg + count_bonus,
            1.0,
        )

        vlog_subsection(
            "[Evidence Strength]"
        )

        vlog(
            f"evidence_count: {len(evidence_list)}"
        )

        vlog(
            f"top_weighted_scores: "
            f"{[round(x, 4) for x in top]}"
        )

        vlog(
            f"average_top_scores: {avg:.4f}"
        )

        vlog(
            f"count_bonus: {count_bonus:.4f}"
        )

        vlog(
            f"final_strength: {strength:.4f}"
        )

        return strength

    # ------------------------------------------------------------------
    # FORMATTING FOR LLM
    # ------------------------------------------------------------------

    @staticmethod
    def _format_evidence(
        evidence_list: list[Evidence],
    ) -> str:
        return "\n---\n".join(
            (
                f"[{e.citation_label()} "
                f"| score={e.score:.2f} "
                f"| reliability={e.source_reliability:.2f}] "
                f"{e.text[:800]}"
            )
            for e in evidence_list
        )

    # ------------------------------------------------------------------
    # VERBOSE LOGGING HELPERS
    # ------------------------------------------------------------------

    @staticmethod
    def _log_evidence(
        evidence: Evidence,
        index: int,
    ) -> None:
        vlog(
            f"  Evidence #{index}"
        )

        vlog(
            f"    citation: "
            f"{evidence.citation_label()}"
        )

        vlog(
            f"    chunk_id: "
            f"{evidence.chunk_id}"
        )

        vlog(
            f"    evidence_id: "
            f"{getattr(evidence, 'evidence_id', 'unknown')}"
        )

        # IMPORTANT:
        # Evidence uses `retrieval_method`, not `method`.
        retrieval_method = getattr(
            evidence,
            "retrieval_method",
            "unknown",
        )

        vlog(
            f"    method: "
            f"{retrieval_method}"
        )

        vlog(
            f"    score: "
            f"{evidence.score:.4f}"
        )

        vlog(
            f"    reliability: "
            f"{evidence.source_reliability:.4f}"
        )

        source_metadata = getattr(
            evidence,
            "source_metadata",
            {},
        )

        if source_metadata:
            vlog(
                "    source_metadata:"
            )

            for key, value in source_metadata.items():
                vlog(
                    f"      {key}: {value}"
                )

        vlog(
            "    text:"
        )

        text = evidence.text.strip()

        if len(text) > 1200:
            vlog(
                f"      {text[:1200]}..."
            )
        else:
            vlog(
                f"      {text}"
            )

    @classmethod
    def _log_evidence_list(
        cls,
        evidence_list: list[Evidence],
        title: str,
    ) -> None:
        vlog_subsection(title)

        if not evidence_list:
            vlog(
                "  No evidence."
            )
            return

        for index, evidence in enumerate(
            evidence_list,
            start=1,
        ):
            cls._log_evidence(
                evidence,
                index,
            )
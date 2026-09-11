from __future__ import annotations

import re

from application.ports.llm_port import LLMPort
from domain.entities import Evidence, FinalAnswer, Hypothesis
from domain.tree import Branch


_SYNTHESIS_SYSTEM = """
You are the final analyst of an evidence-based investigation system.

Answer the original question using only the provided evidence.

RULES:
1. Do not create any information that is absent from the evidence.
2. Provide exact figures whenever they are available.
3. Explain factors that are explicitly present in the evidence.
4. When there is a contradiction between two sources, explicitly report it
   instead of arbitrarily choosing one version.
5. Clearly distinguish between:
   - observed facts;
   - interpretations;
   - uncertainties or contradictions.
6. For a temporal trend, correctly calculate the absolute and/or relative
   change when the necessary figures are available.
7. Calculations must be verifiable.
8. Never present a hypothesis as a fact.

Strict JSON:
{"answer":"...", "confidence":0..1}
""".strip()


class SynthesisService:
    def __init__(
        self,
        llm: LLMPort,
    ) -> None:
        self._llm = llm

    def synthesize(
        self,
        original_question: str,
        strongest_branches: list[Branch],
        hypotheses: list[Hypothesis],
        fallback_evidence: list[Evidence] | None = None,
    ) -> FinalAnswer:

        evidence = self._select_strongest_evidence(
            strongest_branches,
            fallback_evidence or [],
        )

        evidence_by_id = {
            e.id: e
            for e in evidence
        }

        hypothesis_blocks = []

        for hypothesis in hypotheses:

            supporting = [
                evidence_by_id[eid]
                for eid in hypothesis.supporting_evidence_ids
                if eid in evidence_by_id
            ]

            contradicting = [
                evidence_by_id[eid]
                for eid in hypothesis.contradicting_evidence_ids
                if eid in evidence_by_id
            ]

            hypothesis_blocks.append(
                f"Hypothesis: {hypothesis.statement}\n"
                f"Confidence: {hypothesis.confidence:.2f}\n"
                f"Remaining uncertainty: "
                f"{hypothesis.remaining_uncertainty:.2f}\n"
                f"Supporting evidence:\n"
                f"{self._format_evidence(supporting) or '(none)'}\n"
                f"Contradicting evidence:\n"
                f"{self._format_evidence(contradicting) or '(none)'}"
            )

        branch_blocks = []

        for branch in strongest_branches:

            branch_blocks.append(
                f"Branch: {branch.question}\n"
                f"Conclusion: "
                f"{branch.conclusion or '(none)'}\n"
                f"Strength: {branch.evidence_strength:.2f}\n"
                f"Priority: {branch.priority:.2f}\n"
                f"Evidence:\n"
                f"{self._format_evidence(branch.evidence[:8])}"
            )

        prompt = (
            f"Original question:\n"
            f"{original_question}\n\n"

            f"Strong branches:\n"
            f"{chr(10).join(branch_blocks) or '(none)'}\n\n"

            f"Hypotheses:\n"
            f"{chr(10).join(hypothesis_blocks) or '(none)'}\n\n"

            f"Selected evidence:\n"
            f"{self._format_evidence(evidence)}"
        )

        result = self._llm.generate_json(
            prompt,
            system=_SYNTHESIS_SYSTEM,
        )

        answer = str(
            result.get("answer", "")
        ).strip()

        confidence = self._bounded(
            result.get("confidence"),
            0.0,
        )

        if not answer:
            answer = self._fallback_answer(
                original_question,
                evidence,
            )

        return FinalAnswer(
            text=answer,
            hypotheses=hypotheses,
            citations=evidence,
            confidence=confidence,
        )

    @staticmethod
    def _select_strongest_evidence(
        branches: list[Branch],
        fallback: list[Evidence],
        limit: int = 12,
    ) -> list[Evidence]:

        candidates = []

        for branch in branches:
            candidates.extend(
                branch.evidence
            )

        if not candidates:
            candidates = list(fallback)

        unique = {}

        for evidence in candidates:

            key = (
                evidence.chunk_id
                or evidence.text.strip()
            )

            if (
                key not in unique
                or evidence.score
                > unique[key].score
            ):
                unique[key] = evidence

        return sorted(
            unique.values(),
            key=lambda e: (
                e.score
                * e.source_reliability
            ),
            reverse=True,
        )[:limit]

    @staticmethod
    def _format_evidence(
        evidence: list[Evidence],
    ) -> str:

        if not evidence:
            return "(none)"

        return "\n---\n".join(
            f"[{e.citation_label()} | "
            f"score={e.score:.2f} | "
            f"reliability={e.source_reliability:.2f}] "
            f"{e.text[:700]}"
            for e in evidence
        )

    @staticmethod
    def _bounded(
        value,
        default,
    ):

        try:
            return max(
                0.0,
                min(1.0, float(value)),
            )
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _fallback_answer(
        question: str,
        evidence: list[Evidence],
    ) -> str:

        if not evidence:
            return (
                "No sufficient evidence was found "
                f"to answer: {question}"
            )

        return (
            "The available evidence includes: "
            + " ".join(
                f"[{e.citation_label()}] "
                f"{e.text[:250]}"
                for e in evidence[:3]
            )
        )
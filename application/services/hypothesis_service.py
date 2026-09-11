from __future__ import annotations

from application.ports.llm_port import LLMPort
from domain.entities import Evidence, Hypothesis
from domain.tree import Branch


_INITIAL_SYSTEM = """
Formulate distinct and testable explanatory hypotheses based on
the question and the initial evidence.

The hypotheses must be consistent with the available evidence.
Do not turn generic possibilities into facts.

Strict JSON:
{"hypotheses":["..."]}
""".strip()


_UPDATE_SYSTEM = """
For ONE hypothesis, classify the evidence as supporting or contradicting.

Consider an evidence item as supporting or contradicting only if it
is genuinely relevant to the hypothesis.

Then evaluate the confidence and the remaining uncertainty.

Strict JSON:
{
  "supports":[0],
  "contradicts":[1],
  "confidence":0..1,
  "remaining_uncertainty":0..1
}
""".strip()


class HypothesisService:
    def __init__(
        self,
        llm: LLMPort,
    ) -> None:
        self._llm = llm

    def generate_initial_hypotheses(
        self,
        original_question: str,
        initial_evidence: list[Evidence],
    ) -> list[Hypothesis]:

        result = self._llm.generate_json(
            f"Question:\n"
            f"{original_question}\n\n"
            f"Evidence:\n"
            f"{self._format_evidence(initial_evidence)}",
            system=_INITIAL_SYSTEM,
        )

        values = result.get(
            "hypotheses",
            [],
        )

        if not isinstance(values, list):
            return []

        return [
            Hypothesis(
                statement=str(value).strip()
            )
            for value in values
            if str(value).strip()
        ]

    def update_confidence(
        self,
        hypotheses: list[Hypothesis],
        branch: Branch,
    ) -> None:

        if not hypotheses or not branch.evidence:
            return

        excerpt = "\n".join(
            f"[{i}] {e.text[:600]}"
            for i, e in enumerate(
                branch.evidence
            )
        )

        for hypothesis in hypotheses:

            result = self._llm.generate_json(
                f"Hypothesis:\n"
                f"{hypothesis.statement}\n\n"
                f"Evidence:\n"
                f"{excerpt}",
                system=_UPDATE_SYSTEM,
            )

            for index in self._as_int_list(
                result.get("supports")
            ):

                if 0 <= index < len(branch.evidence):

                    evidence = branch.evidence[index]

                    if (
                        evidence.id
                        not in hypothesis.supporting_evidence_ids
                    ):
                        hypothesis.supporting_evidence_ids.append(
                            evidence.id
                        )

            for index in self._as_int_list(
                result.get("contradicts")
            ):

                if 0 <= index < len(branch.evidence):

                    evidence = branch.evidence[index]

                    if (
                        evidence.id
                        not in hypothesis.contradicting_evidence_ids
                    ):
                        hypothesis.contradicting_evidence_ids.append(
                            evidence.id
                        )

            hypothesis.confidence = self._bounded_float(
                result.get("confidence"),
                hypothesis.confidence,
            )

            hypothesis.remaining_uncertainty = (
                self._bounded_float(
                    result.get("remaining_uncertainty"),
                    1.0 - hypothesis.confidence,
                )
            )

    @staticmethod
    def _bounded_float(
        value,
        default: float,
    ) -> float:

        try:
            return max(
                0.0,
                min(1.0, float(value)),
            )
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _as_int_list(
        values,
    ) -> list[int]:

        if not isinstance(values, list):
            return []

        result = []

        for value in values:
            try:
                result.append(int(value))
            except (TypeError, ValueError):
                pass

        return result

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
"""
Extracts a calculation intent from an information need
and the retrieved evidence.

Responsibility:

LLM -> understanding / extraction

This service does NOT perform any calculations.

It only returns:
- operation
- values

The actual calculation is performed afterward by CalculatorTool.
"""


from __future__ import annotations

import re
from typing import Any

from application.ports.llm_port import LLMPort
from domain.entities import Evidence


_EXTRACTION_SYSTEM = """
You are a calculation intent extractor.

Your task is to identify, in TWO mandatory steps:

STEP 1 — For each numeric value required by the calculation, first
identify its exact LABEL as it appears in the evidence, then the
numeric value directly associated with THAT specific label. Verify
that the label and the value are actually adjacent in the source text
before keeping them. Never confuse a label with a different, nearby
label in the text.

STEP 2 — Build the operation using ONLY the values identified in
step 1, in the logical order required by the question.

DO NOT CALCULATE the result.

DO NOT generate a mathematical expression.

DO NOT generate a computed value.

DO NOT take numbers that are not necessary to answer the question.

For operations where order matters, strictly respect the order in
which the values must be used.

Allowed operations:
- "add": addition
- "subtract": subtraction
- "multiply": multiplication
- "divide": division
- "modulo": modulo
- "average": average
- "percent_change": percentage change

Important rules:
- "values" must be EXACTLY the values from "matched_labels", in the same order;
- use only numbers explicitly present in the evidence;
- do not use numbers from other parts of the evidence if they are not needed;
- preserve the order of values;
- return strict JSON only;
- return no additional text.

Required format:

{
  "matched_labels": [
    {"label": "...", "value": number},
    {"label": "...", "value": number}
  ],
  "operation": "add|subtract|multiply|divide|modulo|average|percent_change",
  "values": [number, number, ...]
}
""".strip()


def _validate_matched_labels(
    matched_labels: Any,
    evidence_text: str,
) -> bool:
    """
    Vérifie que chaque (label, value) rapportée par le LLM apparaît
    bien accolée dans le texte source, pour détecter une mauvaise
    association label/valeur avant d'exécuter un calcul.
    """
    if not isinstance(matched_labels, list) or not matched_labels:
        return False

    normalized_text = re.sub(r"\s+", "", evidence_text)

    for item in matched_labels:
        if not isinstance(item, dict):
            return False

        label = str(item.get("label", "")).strip()
        value = item.get("value")

        if not label or value is None:
            return False

        try:
            value_str = (
                str(int(value))
                if float(value).is_integer()
                else str(value)
            )
        except (TypeError, ValueError):
            return False

        normalized_label = re.sub(r"\s+", "", label)

        pattern = (
            re.escape(normalized_label)
            + r".{0,10}?"
            + re.escape(value_str)
        )

        if not re.search(pattern, normalized_text):
            return False

    return True


class CalculationIntentExtractorService:

    def __init__(self, llm: LLMPort) -> None:
        self._llm = llm

    def extract(
        self,
        information_need: str,
        evidence: list[Evidence],
    ) -> dict[str, Any]:

        if not evidence:
            raise ValueError(
                "Cannot extract calculation intent without evidence."
            )

        evidence_text = "\n".join(
            f"[{index}] {item.text}"
            for index, item in enumerate(evidence)
        )

        prompt = (
            f"Besoin d'information:\n"
            f"{information_need}\n\n"
            f"Preuves disponibles:\n"
            f"{evidence_text}\n\n"
            f"Intention de calcul:"
        )

        result = self._llm.generate_json(
            prompt,
            system=_EXTRACTION_SYSTEM,
        )

        if not isinstance(result, dict):
            raise ValueError(
                "Calculation extractor returned an invalid JSON object."
            )

        # ------------------------------------------------------------
        # Validation label/valeur : rejette le résultat si le LLM a
        # associé une valeur à un label qui ne correspond pas au
        # texte source (ex: "T1" collé à une valeur qui appartient
        # en réalité à "T3").
        # ------------------------------------------------------------
        matched_labels = result.get("matched_labels")

        if not _validate_matched_labels(matched_labels, evidence_text):
            raise ValueError(
                "Calculation extractor produced values that could not "
                "be verified against the source evidence "
                f"(matched_labels={matched_labels!r})."
            )

        operation = str(
            result.get("operation", "")
        ).strip().lower()

        values = result.get("values")

        allowed_operations = {
            "add",
            "subtract",
            "multiply",
            "divide",
            "modulo",
            "average",
            "percent_change",
        }

        if operation not in allowed_operations:
            raise ValueError(
                f"Unsupported calculation operation: {operation}"
            )

        if not isinstance(values, list):
            raise ValueError(
                "Calculation extractor must return a list of values."
            )

        if not values:
            raise ValueError(
                "Calculation extractor returned no values."
            )

        normalized_values: list[float | int] = []

        for value in values:
            if isinstance(value, bool):
                raise ValueError(
                    "Boolean values are not valid calculation operands."
                )

            if not isinstance(value, (int, float)):
                raise ValueError(
                    f"Invalid calculation value: {value!r}"
                )

            if abs(float(value)) > 1e12:
                raise ValueError(
                    "Numeric value too large."
                )

            normalized_values.append(value)

        self._validate_operand_count(
            operation,
            normalized_values,
        )

        return {
            "operation": operation,
            "values": normalized_values,
        }

    @staticmethod
    def _validate_operand_count(
        operation: str,
        values: list[float | int],
    ) -> None:

        count = len(values)

        if operation in {
            "subtract",
            "divide",
            "modulo",
            "percent_change",
        }:
            if count != 2:
                raise ValueError(
                    f"Operation '{operation}' requires exactly "
                    f"2 values, received {count}."
                )

        elif operation in {
            "add",
            "multiply",
            "average",
        }:
            if count < 2:
                raise ValueError(
                    f"Operation '{operation}' requires at least "
                    f"2 values, received {count}."
                )
"""
Calculatrice déterministe.

Responsabilité :
    recevoir une intention de calcul déjà extraite
    puis effectuer le calcul de manière déterministe.

Le LLM n'est PAS utilisé ici.

Entrée attendue :

{
    "operation": "subtract",
    "values": [1250, 1180]
}

Sortie :

{
    "expression": "1250 - 1180",
    "value": 70.0
}
"""

from __future__ import annotations

import ast
import operator
from typing import Any, Optional

from application.ports.tool_port import ToolPort
from domain.entities import Evidence, ToolResult


_ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


class CalculatorTool(ToolPort):

    name = "calculator"

    description = (
        "Performs deterministic arithmetic calculations from "
        "a structured operation and numeric values."
    )

    # Le calculator a besoin des preuves afin que l'Engine puisse
    # récupérer les valeurs avant l'extraction de l'intention.
    requires_data = True

    def __init__(self) -> None:
        pass

    def execute(
        self,
        information_need: str,
        input_data: Optional[list[Evidence]] = None,
        tool_input: Optional[dict[str, Any]] = None,
    ) -> ToolResult:

        del information_need
        del input_data

        if not isinstance(tool_input, dict):
            return ToolResult(
                success=False,
                output_text="",
                error="Calculator requires a structured tool_input.",
            )

        operation = str(
            tool_input.get("operation", "")
        ).strip().lower()

        values = tool_input.get("values")

        if not operation:
            return ToolResult(
                success=False,
                output_text="",
                error="Missing calculation operation.",
            )

        if not isinstance(values, list):
            return ToolResult(
                success=False,
                output_text="",
                error="Calculation values must be provided as a list.",
            )

        try:
            normalized_values = self._validate_values(values)

            self._validate_operation(
                operation,
                normalized_values,
            )

            expression = self._build_expression(
                operation,
                normalized_values,
            )

            value = self._safe_eval(expression)

        except Exception as exc:
            return ToolResult(
                success=False,
                output_text="",
                error=f"Calculation error: {exc}",
            )

        return ToolResult(
            success=True,
            output_text=(
                f"Calculation: {expression} = "
                f"{self._format_number(value)}"
            ),
            data={
                "operation": operation,
                "values": normalized_values,
                "expression": expression,
                "value": value,
            },
        )

    @staticmethod
    def _validate_values(
        values: list[Any],
    ) -> list[float | int]:

        normalized: list[float | int] = []

        for value in values:

            if isinstance(value, bool):
                raise ValueError(
                    "Boolean values are not valid operands."
                )

            if not isinstance(value, (int, float)):
                raise ValueError(
                    f"Invalid numeric value: {value!r}"
                )

            if abs(float(value)) > 1e12:
                raise ValueError(
                    "Numeric value too large."
                )

            normalized.append(value)

        if not normalized:
            raise ValueError(
                "At least one numeric value is required."
            )

        return normalized

    @staticmethod
    def _validate_operation(
        operation: str,
        values: list[float | int],
    ) -> None:

        count = len(values)

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
                f"Unsupported operation: {operation}"
            )

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

        if operation in {
            "divide",
            "modulo",
        } and float(values[1]) == 0:
            raise ValueError(
                "Division or modulo by zero is not allowed."
            )

        if operation == "percent_change" and float(values[0]) == 0:
            raise ValueError(
                "Percent change cannot use zero as the initial value."
            )

    @classmethod
    def _build_expression(
        cls,
        operation: str,
        values: list[float | int],
    ) -> str:

        formatted = [
            cls._format_number(value)
            for value in values
        ]

        if operation == "add":
            return " + ".join(formatted)

        if operation == "subtract":
            return f"{formatted[0]} - {formatted[1]}"

        if operation == "multiply":
            return " * ".join(formatted)

        if operation == "divide":
            return f"{formatted[0]} / {formatted[1]}"

        if operation == "modulo":
            return f"{formatted[0]} % {formatted[1]}"

        if operation == "average":
            return (
                f"({' + '.join(formatted)}) / "
                f"{len(formatted)}"
            )

        if operation == "percent_change":
            return (
                f"(({formatted[0]} - {formatted[1]}) "
                f"/ {formatted[0]}) * 100"
            )

        raise ValueError(
            f"Unsupported operation: {operation}"
        )

    @staticmethod
    def _format_number(
        value: float | int,
    ) -> str:

        if isinstance(value, int):
            return str(value)

        if float(value).is_integer():
            return str(int(value))

        return str(value)

    @staticmethod
    def _safe_eval(
        expression: str,
    ) -> float:

        tree = ast.parse(
            expression,
            mode="eval",
        )

        return float(
            CalculatorTool._eval_node(tree.body)
        )

    @staticmethod
    def _eval_node(
        node: ast.AST,
    ) -> float:

        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, (int, float))
            and not isinstance(node.value, bool)
        ):
            if abs(float(node.value)) > 1e12:
                raise ValueError(
                    "Numeric literal too large."
                )

            return float(node.value)

        if (
            isinstance(node, ast.UnaryOp)
            and type(node.op) in _ALLOWED_OPERATORS
        ):
            return float(
                _ALLOWED_OPERATORS[type(node.op)](
                    CalculatorTool._eval_node(node.operand)
                )
            )

        if (
            isinstance(node, ast.BinOp)
            and type(node.op) in _ALLOWED_OPERATORS
        ):
            left = CalculatorTool._eval_node(
                node.left
            )

            right = CalculatorTool._eval_node(
                node.right
            )

            if (
                isinstance(node.op, ast.Mult)
                and abs(left * right) > 1e15
            ):
                raise ValueError(
                    "Result too large."
                )

            if (
                isinstance(node.op, ast.Div)
                and right == 0
            ):
                raise ValueError(
                    "Division by zero."
                )

            if (
                isinstance(node.op, ast.Mod)
                and right == 0
            ):
                raise ValueError(
                    "Modulo by zero."
                )

            return float(
                _ALLOWED_OPERATORS[type(node.op)](
                    left,
                    right,
                )
            )

        raise ValueError(
            f"Unsupported expression node: {ast.dump(node)}"
        )
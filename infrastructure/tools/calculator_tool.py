from __future__ import annotations

import ast
import operator
from typing import Optional

from application.ports.llm_port import LLMPort
from application.ports.tool_port import ToolPort
from domain.entities import Evidence, ToolResult

_ALLOWED = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

_SYSTEM = """
Extrait une expression arithmétique simple nécessaire au calcul.
Uniquement JSON:
{"expression":"(120-150)/150*100"}.
N'utilise pas de variables, fonctions, appels, attributs ou puissances.
""".strip()


class CalculatorTool(ToolPort):
    name = "calculator"
    description = "Évalue des expressions arithmétiques simples."
    requires_data = False

    def __init__(self, llm: LLMPort) -> None:
        self._llm = llm

    def execute(
        self,
        information_need: str,
        input_data: Optional[list[Evidence]] = None,
    ) -> ToolResult:
        result = self._llm.generate_json(
            f"Besoin:\n{information_need}",
            system=_SYSTEM,
        )
        expression = str(result.get("expression", "")).strip()
        if not expression:
            return ToolResult(False, "", error="No expression extracted.")

        try:
            value = self._safe_eval(expression)
        except Exception as exc:
            return ToolResult(False, "", error=f"Evaluation error: {exc}")

        return ToolResult(
            True,
            f"Calcul: {expression} = {value}",
            {"expression": expression, "value": value},
        )

    @staticmethod
    def _safe_eval(expression: str) -> float:
        tree = ast.parse(expression, mode="eval")
        return float(CalculatorTool._eval_node(tree.body))

    @staticmethod
    def _eval_node(node: ast.AST) -> float:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            if abs(float(node.value)) > 1e12:
                raise ValueError("Numeric literal too large.")
            return float(node.value)

        if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED:
            return _ALLOWED[type(node.op)](CalculatorTool._eval_node(node.operand))

        if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED:
            left = CalculatorTool._eval_node(node.left)
            right = CalculatorTool._eval_node(node.right)
            if isinstance(node.op, ast.Mult) and abs(left * right) > 1e15:
                raise ValueError("Result too large.")
            return _ALLOWED[type(node.op)](left, right)

        raise ValueError(f"Unsupported expression: {ast.dump(node)}")

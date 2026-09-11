import pytest

from infrastructure.tools.calculator_tool import CalculatorTool


def test_safe_calculator():
    assert CalculatorTool._safe_eval("(120-150)/150*100") == pytest.approx(-20.0)


def test_rejects_calls():
    with pytest.raises(ValueError):
        CalculatorTool._safe_eval("__import__('os').system('echo bad')")

"""Tests for the calculator tool's safe expression evaluation."""
import pytest

from app.agents.tools.calculator import CalculatorTool, safe_eval


def test_basic_arithmetic():
    assert safe_eval("2 + 3 * 4") == 14


def test_percentage_of_a_number():
    assert safe_eval("850 * 17.5 / 100") == pytest.approx(148.75)


def test_percentage_improvement():
    assert safe_eval("(84 - 72) / 72 * 100") == pytest.approx(16.6667, rel=1e-3)


def test_parentheses_and_power():
    assert safe_eval("(2 + 3) ** 2") == 25


def test_rejects_function_calls():
    with pytest.raises(ValueError):
        safe_eval("__import__('os').system('echo hi')")


def test_rejects_names():
    with pytest.raises(ValueError):
        safe_eval("x + 1")


def test_rejects_invalid_syntax():
    with pytest.raises(ValueError):
        safe_eval("2 +* 3")


def test_tool_run_returns_result_string():
    result = CalculatorTool().run(expression="10 / 2")
    assert result.output == "10 / 2 = 5.0"
    assert result.error is None


def test_tool_run_handles_division_by_zero():
    result = CalculatorTool().run(expression="1 / 0")
    assert result.error is not None
    assert "1 / 0" in result.output


def test_tool_run_handles_missing_expression():
    result = CalculatorTool().run()
    assert result.error == "missing_expression"

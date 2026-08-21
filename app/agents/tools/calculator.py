"""Calculator tool — evaluates a plain arithmetic expression safely.

Never uses eval()/exec(). Parses the expression into an AST and walks
it, allowing only numeric literals and a fixed set of arithmetic
operators — anything else (names, calls, attribute access, imports...)
raises before any code could run.
"""
import ast
import operator

from app.agents.tools.base import BaseTool, ToolResult

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.USub: operator.neg, ast.UAdd: operator.pos}


def safe_eval(expression: str) -> float:
    try:
        node = ast.parse(expression, mode="eval").body
    except SyntaxError as exc:
        raise ValueError(f"Not a valid arithmetic expression: {exc}") from exc
    return _eval_node(node)


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(
        node.value, bool
    ):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        return _BIN_OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_node(node.operand))
    raise ValueError(f"Unsupported expression element: {ast.dump(node)}")


class CalculatorTool(BaseTool):
    name = "calculator_tool"
    description = (
        "Evaluates a plain arithmetic expression. Use this for any math — "
        "percentages, differences, sums, etc. — by first converting the "
        "question into a pure arithmetic expression yourself, "
        "e.g. \"17.5% of 850\" -> \"850 * 17.5 / 100\", or a percentage "
        "improvement from 72 to 84 -> \"(84 - 72) / 72 * 100\"."
    )
    parameters = {
        "expression": "A plain arithmetic expression using only numbers, "
        "+ - * / ** % and parentheses. No words, no variables."
    }

    def run(self, expression: str = "", **kwargs) -> ToolResult:
        if not expression:
            return ToolResult(output="No expression provided.", error="missing_expression")
        try:
            value = safe_eval(expression)
        except (ValueError, ZeroDivisionError, TypeError, RecursionError) as exc:
            return ToolResult(output=f"Could not evaluate '{expression}': {exc}", error=str(exc))
        return ToolResult(output=f"{expression} = {value}")

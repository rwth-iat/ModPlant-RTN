from __future__ import annotations

import ast
from typing import Any


CONDITION_LANGUAGE = "urn:modplant:condition:ast:v1"


class UnsupportedConditionLanguage(ValueError):
    pass


_COMPARE_TO_OP = {
    ast.Eq: "eq",
    ast.NotEq: "ne",
    ast.Lt: "lt",
    ast.LtE: "le",
    ast.Gt: "gt",
    ast.GtE: "ge",
}


def parse_condition(text: str, language: str = CONDITION_LANGUAGE) -> dict[str, Any]:
    if language != CONDITION_LANGUAGE:
        raise UnsupportedConditionLanguage(language)
    normalized = text.strip()
    if not normalized:
        raise ValueError("A condition must not be empty")
    normalized = normalized.replace("&&", " and ").replace("||", " or ")
    try:
        root = ast.parse(normalized, mode="eval").body
    except SyntaxError as exc:
        raise ValueError(f"Invalid condition syntax: {text!r}") from exc
    return _from_python_ast(root)


def _from_python_ast(node: ast.AST) -> dict[str, Any]:
    if isinstance(node, ast.Constant) and isinstance(node.value, (bool, int, float, str, type(None))):
        return {"op": "literal", "value": node.value}
    if isinstance(node, ast.Name):
        if node.id.lower() == "true":
            return {"op": "literal", "value": True}
        if node.id.lower() == "false":
            return {"op": "literal", "value": False}
        return {"op": "ref", "name": node.id}
    if isinstance(node, ast.Attribute):
        parts: list[str] = []
        current: ast.AST = node
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        if not isinstance(current, ast.Name):
            raise ValueError("Only dotted variable references are allowed")
        parts.append(current.id)
        return {"op": "ref", "name": ".".join(reversed(parts))}
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return {"op": "not", "arg": _from_python_ast(node.operand)}
    if isinstance(node, ast.BoolOp) and isinstance(node.op, (ast.And, ast.Or)):
        return {
            "op": "and" if isinstance(node.op, ast.And) else "or",
            "args": [_from_python_ast(value) for value in node.values],
        }
    if isinstance(node, ast.Compare) and len(node.ops) == 1 and len(node.comparators) == 1:
        operation = _COMPARE_TO_OP.get(type(node.ops[0]))
        if operation is None:
            raise ValueError("Unsupported comparison operator")
        return {
            "op": operation,
            "left": _from_python_ast(node.left),
            "right": _from_python_ast(node.comparators[0]),
        }
    raise ValueError(f"Unsupported condition expression: {ast.dump(node, include_attributes=False)}")


def evaluate_condition(condition: dict[str, Any], context: dict[str, Any]) -> Any:
    op = condition.get("op")
    if op == "literal":
        return condition.get("value")
    if op == "ref":
        current: Any = context
        for part in str(condition["name"]).split("."):
            if not isinstance(current, dict) or part not in current:
                raise KeyError(f"Condition variable is missing: {condition['name']}")
            current = current[part]
        return current
    if op == "not":
        return not bool(evaluate_condition(condition["arg"], context))
    if op == "and":
        return all(bool(evaluate_condition(item, context)) for item in condition["args"])
    if op == "or":
        return any(bool(evaluate_condition(item, context)) for item in condition["args"])
    left = evaluate_condition(condition["left"], context)
    right = evaluate_condition(condition["right"], context)
    operations = {
        "eq": lambda: left == right,
        "ne": lambda: left != right,
        "lt": lambda: left < right,
        "le": lambda: left <= right,
        "gt": lambda: left > right,
        "ge": lambda: left >= right,
    }
    if op not in operations:
        raise ValueError(f"Unknown condition operation: {op}")
    return operations[op]()


def render_condition(condition: dict[str, Any]) -> str:
    op = condition.get("op")
    if op == "literal":
        value = condition.get("value")
        if isinstance(value, bool):
            return "True" if value else "False"
        return repr(value)
    if op == "ref":
        return str(condition["name"])
    if op == "not":
        return f"not ({render_condition(condition['arg'])})"
    if op in {"and", "or"}:
        return f" {op} ".join(f"({render_condition(item)})" for item in condition["args"])
    symbols = {"eq": "==", "ne": "!=", "lt": "<", "le": "<=", "gt": ">", "ge": ">="}
    if op in symbols:
        return f"{render_condition(condition['left'])} {symbols[op]} {render_condition(condition['right'])}"
    raise ValueError(f"Unknown condition operation: {op}")


"""
pine_lite.validator — static pass over parsed [SCRIPT] statements and
condition-block expressions that collects EVERY unsupported construct
before any evaluation is attempted.

Design (per explicit requirement): never silently skip an unrecognized
piece and never stop at the first problem — collect everything so the
whole list can be handed back to fix the script in one pass, and never run
a partial/best-effort backtest of a script that isn't fully understood.
"""
from __future__ import annotations

from . import builtins as B
from .parser import (
    Assign, MultiAssign, FuncDef, CallStmt, Call, BinOp, UnaryOp, Ternary,
    Index, Var, Num, Str,
)


def _walk_expr(node, known_funcs: set, issues: list, line_no, raw_text: str):
    if node is None or isinstance(node, (Num, Str, Var)):
        return
    if isinstance(node, Index):
        _walk_expr(node.base, known_funcs, issues, line_no, raw_text)
        return
    if isinstance(node, BinOp):
        _walk_expr(node.left, known_funcs, issues, line_no, raw_text)
        _walk_expr(node.right, known_funcs, issues, line_no, raw_text)
        return
    if isinstance(node, UnaryOp):
        _walk_expr(node.operand, known_funcs, issues, line_no, raw_text)
        return
    if isinstance(node, Ternary):
        _walk_expr(node.cond, known_funcs, issues, line_no, raw_text)
        _walk_expr(node.then, known_funcs, issues, line_no, raw_text)
        _walk_expr(node.otherwise, known_funcs, issues, line_no, raw_text)
        return
    if isinstance(node, Call):
        matched_prefix = False
        for prefix, reason in B.EXPLICITLY_UNSUPPORTED_PREFIXES.items():
            if node.name == prefix or node.name.startswith(prefix):
                issues.append({"line": line_no, "text": raw_text, "construct": node.name, "message": reason})
                matched_prefix = True
                break
        if not matched_prefix and node.name not in B.ALL_KNOWN_CALLS and node.name not in known_funcs:
            issues.append({
                "line": line_no, "text": raw_text, "construct": node.name,
                "message": f"'{node.name}' is not a recognized built-in or user-defined function",
            })
        # NOOP_FUNCS (plot/plotshape/indicator/...) are display/declaration
        # calls that never get evaluated (see evaluator.CallStmt handling)
        # -- their arguments commonly reference decorative namespaces
        # (color.new, shape.triangleup, location.bottom, size.tiny) that
        # aren't part of the trading-logic whitelist and don't need to be,
        # since nothing about them affects backtest signals. Skip walking
        # into them entirely rather than flagging every decorative call.
        if node.name in B.NOOP_FUNCS:
            return
        for a in node.args:
            _walk_expr(a, known_funcs, issues, line_no, raw_text)
        for a in node.kwargs.values():
            _walk_expr(a, known_funcs, issues, line_no, raw_text)
        return
    issues.append({
        "line": line_no, "text": raw_text,
        "message": f"internal: unrecognized AST node type {type(node).__name__}",
    })


def validate_script(statements_with_meta: list, condition_exprs: list) -> list:
    """
    statements_with_meta: list of (Statement|None, error_dict|None, line_no, raw_text)
        — from parser.parse_statement_line, one entry per non-blank [SCRIPT] line.
    condition_exprs: list of (Node|None, error_dict|None, block_name, line_no, raw_text)
        — from parser.parse_expression_text, one entry per non-empty condition block.

    Returns the combined, ordered issues list (parse errors + unsupported
    constructs + semantic checks). Empty list means the script is fully
    understood and safe to evaluate.
    """
    issues: list = []
    known_funcs: set = set()

    for stmt, err, line_no, raw_text in statements_with_meta:
        if err is None and isinstance(stmt, FuncDef):
            known_funcs.add(stmt.name)

    for stmt, err, line_no, raw_text in statements_with_meta:
        if err is not None:
            issues.append(err)
            continue
        if isinstance(stmt, Assign):
            _walk_expr(stmt.expr, known_funcs, issues, line_no, raw_text)
        elif isinstance(stmt, FuncDef):
            _walk_expr(stmt.body, known_funcs, issues, line_no, raw_text)
        elif isinstance(stmt, CallStmt):
            _walk_expr(stmt.call, known_funcs, issues, line_no, raw_text)
        elif isinstance(stmt, MultiAssign):
            if not (isinstance(stmt.expr, Call) and stmt.expr.name == "ta.macd"):
                issues.append({
                    "line": line_no, "text": raw_text,
                    "message": "multiple-assignment targets ([a,b,c] = ...) are only supported for ta.macd() in pine_lite v1",
                })
            else:
                _walk_expr(stmt.expr, known_funcs, issues, line_no, raw_text)

    for expr, err, block_name, line_no, raw_text in condition_exprs:
        if err is not None:
            issues.append({**err, "block": block_name})
            continue
        if expr is not None:
            block_issues: list = []
            _walk_expr(expr, known_funcs, block_issues, line_no, raw_text)
            for iss in block_issues:
                iss["block"] = block_name
            issues.extend(block_issues)

    return issues

"""
pine_lite.evaluator — evaluates a VALIDATED (zero-issues) [SCRIPT] block and
condition-block expressions into full-length series aligned to the candle
list: list[Optional[float]] for numeric expressions, list[bool] for
condition blocks.

Only ever called after validator.validate_script() returns an empty issues
list — every Call name is guaranteed to be a known builtins.py entry or a
user-defined FuncDef by that point, so this module doesn't re-check names,
only evaluates.
"""
from __future__ import annotations

from typing import Optional

from tradingview_mcp.core.services import indicators_calc as ind
from . import builtins as B
from .parser import (
    Assign, MultiAssign, FuncDef, CallStmt, Call, BinOp, UnaryOp, Ternary,
    Index, Var, Num, Str,
)


class EvalError(Exception):
    pass


class ScriptEnv:
    """Holds the evaluated series for every [SCRIPT]-defined variable plus
    the base OHLCV series, and the set of user function definitions."""

    def __init__(self, candles: list[dict]):
        self.n = len(candles)
        self.candles = candles
        self.vars: dict[str, list] = {
            "open": [c["open"] for c in candles],
            "high": [c["high"] for c in candles],
            "low": [c["low"] for c in candles],
            "close": [c["close"] for c in candles],
            "volume": [c.get("volume", 0.0) for c in candles],
        }
        self.funcs: dict[str, FuncDef] = {}

    def run_script(self, statements: list) -> None:
        for stmt in statements:
            if isinstance(stmt, FuncDef):
                self.funcs[stmt.name] = stmt
            elif isinstance(stmt, Assign):
                self.vars[stmt.name] = self._eval(stmt.expr)
            elif isinstance(stmt, MultiAssign):
                call = stmt.expr  # validator guarantees this is ta.macd(...)
                close = self._eval(call.args[0])
                fast, slow, signal = (self._scalar(a) for a in call.args[1:4])
                out = B.MULTI_OUTPUT_FUNCS[call.name](close, fast, slow, signal)
                keys = ["macd", "signal", "histogram"]
                for name, key in zip(stmt.names, keys):
                    self.vars[name] = out[key]
            elif isinstance(stmt, CallStmt):
                continue  # NOOP_FUNCS calls (only kind that survives validation) have no series effect

    def eval_condition(self, expr) -> list[bool]:
        raw = self._eval(expr)
        return [False if v is None else bool(v) for v in raw]

    # ---- core recursive evaluator: returns a list of length self.n ----

    def _eval(self, node) -> list:
        if isinstance(node, Num):
            return [node.value] * self.n
        if isinstance(node, Str):
            return [node.value] * self.n
        if isinstance(node, Var):
            if node.name == "true":
                return [1.0] * self.n
            if node.name == "false":
                return [0.0] * self.n
            if node.name == "na":
                return [None] * self.n
            if node.name not in self.vars:
                raise EvalError(f"undefined variable '{node.name}' (used before assignment, or a typo)")
            return self.vars[node.name]
        if isinstance(node, Index):
            base = self._eval(node.base)
            off = node.offset
            return [
                base[i - off] if 0 <= i - off < self.n else None
                for i in range(self.n)
            ]
        if isinstance(node, UnaryOp):
            operand = self._eval(node.operand)
            if node.op == "not":
                return [None if v is None else (not bool(v)) for v in operand]
            sign = -1 if node.op == "-" else 1
            return [None if v is None else sign * v for v in operand]
        if isinstance(node, BinOp):
            return self._eval_binop(node)
        if isinstance(node, Ternary):
            cond = self._eval(node.cond)
            then = self._eval(node.then)
            other = self._eval(node.otherwise)
            return [
                None if c is None else (then[i] if bool(c) else other[i])
                for i, c in enumerate(cond)
            ]
        if isinstance(node, Call):
            return self._eval_call(node)
        raise EvalError(f"cannot evaluate node of type {type(node).__name__}")

    def _eval_binop(self, node: BinOp) -> list:
        if node.op == "and":
            left = self._eval(node.left)
            right = self._eval(node.right)
            return [None if (l is None or r is None) else (bool(l) and bool(r)) for l, r in zip(left, right)]
        if node.op == "or":
            left = self._eval(node.left)
            right = self._eval(node.right)
            return [None if (l is None or r is None) else (bool(l) or bool(r)) for l, r in zip(left, right)]
        left = self._eval(node.left)
        right = self._eval(node.right)
        ops = {
            "+": lambda a, b: a + b,
            "-": lambda a, b: a - b,
            "*": lambda a, b: a * b,
            "/": lambda a, b: (a / b if b != 0 else None),
            "%": lambda a, b: (a % b if b != 0 else None),
            ">": lambda a, b: a > b,
            "<": lambda a, b: a < b,
            ">=": lambda a, b: a >= b,
            "<=": lambda a, b: a <= b,
            "==": lambda a, b: a == b,
            "!=": lambda a, b: a != b,
        }
        fn = ops[node.op]
        return [None if (l is None or r is None) else fn(l, r) for l, r in zip(left, right)]

    def _eval_call(self, node: Call) -> list:
        if node.name in self.funcs:
            fd = self.funcs[node.name]
            if len(node.args) != len(fd.params):
                raise EvalError(f"'{node.name}' expects {len(fd.params)} argument(s), got {len(node.args)}")
            saved = {}
            for pname, arg_node in zip(fd.params, node.args):
                saved[pname] = self.vars.get(pname)
                self.vars[pname] = self._eval(arg_node)
            result = self._eval(fd.body)
            for pname, old in saved.items():
                if old is None:
                    self.vars.pop(pname, None)
                else:
                    self.vars[pname] = old
            return result

        if node.name in B.SERIES_FUNCS:
            n_series, fn = B.SERIES_FUNCS[node.name]
            series_args = [self._eval(a) for a in node.args[:n_series]]
            scalar_args = [self._scalar(a) for a in node.args[n_series:]]
            return fn(*series_args, *scalar_args)

        if node.name in B.MULTI_OUTPUT_FUNCS:
            # Pine disallows assigning ta.macd() to a single variable, but as
            # a lenient convenience a bare (non-multi) assignment here just
            # returns the MACD line; use [m, s, h] = ta.macd(...) for all three.
            close = self._eval(node.args[0])
            fast, slow, signal = (self._scalar(a) for a in node.args[1:4])
            out = B.MULTI_OUTPUT_FUNCS[node.name](close, fast, slow, signal)
            return out["macd"]

        if node.name == "ta.atr":
            high, low, close = self.vars["high"], self.vars["low"], self.vars["close"]
            if len(node.args) == 1:
                period = int(self._scalar(node.args[0]))
            elif len(node.args) >= 4:
                period = int(self._scalar(node.args[3]))
            else:
                raise EvalError("ta.atr expects ta.atr(length)")
            return ind.calc_atr(high, low, close, period)

        if node.name in B.SCALAR_MATH_FUNCS:
            fn = B.SCALAR_MATH_FUNCS[node.name]
            arg_series = [self._eval(a) for a in node.args]
            return [
                None if any(v is None for v in vals) else fn(*vals)
                for vals in zip(*arg_series)
            ]

        if node.name == "nz":
            vals = self._eval(node.args[0])
            default = self._scalar(node.args[1]) if len(node.args) > 1 else 0.0
            return [default if v is None else v for v in vals]

        if node.name in B.NOOP_FUNCS:
            return [None] * self.n

        raise EvalError(f"'{node.name}' has no evaluator (should have been caught by validation)")

    def _scalar(self, node) -> float:
        """Evaluate a node expected to be a constant-ish argument (e.g. the
        `20` in ta.ema(close, 20)) — takes the first non-None value of its
        evaluated series, since the fixed-parameter arguments this subset
        supports are always literal numbers."""
        vals = self._eval(node)
        for v in vals:
            if v is not None:
                return v
        raise EvalError("expected a constant numeric argument (e.g. a period length) but got no value")

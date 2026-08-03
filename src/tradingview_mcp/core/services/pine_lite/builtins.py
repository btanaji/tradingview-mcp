"""
pine_lite.builtins — whitelist of supported ta.*/math.* functions, plus the
recognized no-op display/declaration functions (plot, indicator, strategy
header calls, etc.) that real copy-pasted Pine scripts commonly include but
which have no bearing on backtest signals.

All series-valued functions operate on FULL list[Optional[float]] series
(same shape convention as indicators_calc.py), reusing that module's math
rather than reimplementing it, and always vectorized over the whole series
rather than recomputed per-bar — matching Pine's own "series" semantics.
"""
from __future__ import annotations

import math as _math
from typing import Optional

from tradingview_mcp.core.services import indicators_calc as ind


def _with_valid_prefix(series: list, fn, *fn_args) -> list:
    """Trim the leading None-run from `series`, apply `fn` (an
    indicators_calc-style function expecting list[float]) to the remainder,
    then re-pad with None. Every indicators_calc.py function only ever
    produces a leading None run (warmup), never internal gaps, so this is
    safe for any chain of these functions (e.g. ta.ema(ta.rsi(close,14),9))."""
    n = len(series)
    first_valid = next((i for i, v in enumerate(series) if v is not None), None)
    if first_valid is None:
        return [None] * n
    valid = [v if v is not None else 0.0 for v in series[first_valid:]]
    computed = fn(valid, *fn_args)
    out = [None] * first_valid + list(computed)
    if len(out) < n:
        out += [None] * (n - len(out))
    return out[:n]


def _rolling(series: list, length, pick) -> list:
    length = int(length)
    n = len(series)
    out: list = [None] * n
    for i in range(length - 1, n):
        window = [v for v in series[i - length + 1 : i + 1] if v is not None]
        out[i] = pick(window) if window else None
    return out


def _crossover(a: list, b: list) -> list:
    n = len(a)
    out = [False] * n
    for i in range(1, n):
        if a[i - 1] is None or b[i - 1] is None or a[i] is None or b[i] is None:
            continue
        out[i] = a[i - 1] <= b[i - 1] and a[i] > b[i]
    return out


def _crossunder(a: list, b: list) -> list:
    n = len(a)
    out = [False] * n
    for i in range(1, n):
        if a[i - 1] is None or b[i - 1] is None or a[i] is None or b[i] is None:
            continue
        out[i] = a[i - 1] >= b[i - 1] and a[i] < b[i]
    return out


def _change(s: list) -> list:
    """ta.change(x): x[i] - x[i-1]."""
    n = len(s)
    out: list = [None] * n
    for i in range(1, n):
        if s[i] is None or s[i - 1] is None:
            continue
        out[i] = s[i] - s[i - 1]
    return out


# name -> (n_series_args, python_fn(series_args..., *scalar_args) -> series)
SERIES_FUNCS = {
    "ta.sma": (1, lambda s, length: _with_valid_prefix(s, ind.calc_sma, int(length))),
    "ta.ema": (1, lambda s, length: _with_valid_prefix(s, ind.calc_ema, int(length))),
    "ta.wma": (1, lambda s, length: _with_valid_prefix(s, ind.calc_wma, int(length))),
    "ta.rsi": (1, lambda s, length: _with_valid_prefix(s, ind.calc_rsi, int(length))),
    "ta.stdev": (1, lambda s, length: _with_valid_prefix(s, ind.calc_stdev, int(length))),
    "ta.highest": (1, lambda s, length: _rolling(s, length, max)),
    "ta.lowest": (1, lambda s, length: _rolling(s, length, min)),
    "ta.crossover": (2, lambda a, b: _crossover(a, b)),
    "ta.crossunder": (2, lambda a, b: _crossunder(a, b)),
    "ta.change": (1, lambda s: _change(s)),
}

# multi-output funcs handled specially by the evaluator (see evaluator.py's
# MultiAssign handling); returns a dict of series keyed "macd"/"signal"/"histogram"
MULTI_OUTPUT_FUNCS = {
    "ta.macd": lambda close, fast, slow, signal: ind.calc_macd(close, int(fast), int(slow), int(signal)),
}

# ta.atr's real Pine signature is ta.atr(length) — it implicitly uses the
# script's own high/low/close series, so it needs evaluator context rather
# than fitting the plain SERIES_FUNCS shape.
CONTEXT_FUNCS = {"ta.atr"}

SCALAR_MATH_FUNCS = {
    "math.abs": abs,
    "math.max": max,
    "math.min": min,
    "math.round": round,
    "math.sqrt": lambda x: x ** 0.5,
    "math.log": _math.log,
    "math.log10": _math.log10,
    "math.exp": _math.exp,
    "math.pow": _math.pow,
}

# `nz` is a bare (non-namespaced) function whose whole purpose is
# na-substitution, so unlike SCALAR_MATH_FUNCS it must accept None input
# rather than skip it — handled specially in evaluator.py.
SPECIAL_FUNCS = {"nz"}

# Recognized but ignored: real Pine scripts commonly include these for
# display/declaration purposes; none of them affect backtest signals.
NOOP_FUNCS = {
    "plot", "plotshape", "plotchar", "plotcandle", "plotbar", "plotarrow",
    "bgcolor", "fill", "hline", "alert", "alertcondition",
    "indicator", "strategy", "label.new", "line.new", "box.new", "table.new",
    "barcolor",
}

ALL_KNOWN_CALLS = (
    set(SERIES_FUNCS) | set(MULTI_OUTPUT_FUNCS) | CONTEXT_FUNCS
    | set(SCALAR_MATH_FUNCS) | NOOP_FUNCS | SPECIAL_FUNCS
)

# Constructs recognized by name and reported as an explicit, actionable
# "not supported" issue rather than a generic syntax error — common enough
# in real Pine scripts to deserve a clearer message than the parser alone
# can give.
EXPLICITLY_UNSUPPORTED_PREFIXES = {
    "request.security": "multi-timeframe scripts (request.security) are out of scope for pine_lite v1",
    "array.": "arrays are not supported by pine_lite v1",
    "matrix.": "matrices are not supported by pine_lite v1",
    "strategy.entry": "use this file's [Buy Condition]/[Sell Condition]/[TP]/[SL] blocks instead of strategy.entry",
    "strategy.exit": "use this file's [Buy Exit]/[Sell Exit]/[TP]/[SL] blocks instead of strategy.exit",
    "strategy.close": "use this file's [Buy Exit]/[Sell Exit] blocks instead of strategy.close",
    "input.": "script inputs (input.*) are not supported — hardcode the value directly in [SCRIPT]",
    "ta.sar": "parabolic SAR is not in the pine_lite v1 whitelist",
    "ta.supertrend": "supertrend is not in the pine_lite v1 whitelist (available separately via indicators_calc.calc_supertrend if this server needs to add it)",
}

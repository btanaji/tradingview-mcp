"""
Pine Strategy Service — user extension (parallel to custom_strategy_service.py)

Reads .txt files pairing a Pine Script [SCRIPT] block (indicator math) with
plain-English rule blocks ([Trend Filter]/[Buy Condition]/[Sell Condition]/
[Buy Exit]/[Sell Exit]/[SL]/[TP]), transpiles the (constrained) Pine subset
via pine_lite, and backtests the resulting signals against the same local
CSV data used elsewhere in this server — reusing custom_strategy_service's
CSV loader and backtest_service's cost/metrics functions so results are in
the same shape as every other backtest tool.

REQUIRES: CSV files under ~/Claude/data (TV_MCP_CSV_DATA_DIR), same as
custom_strategy_service.py. Strategy files live under ~/Claude/pine_strategies
(TV_MCP_PINE_STRATEGIES_DIR).

IMPORTANT — read before trusting results: pine_lite is a CONSTRAINED Pine
Script subset transpiler, not a full Pine interpreter (see
pine_lite/__init__.py for exactly what is/isn't supported). A script with
any unsupported construct never runs — parse_pine_strategy_file /
validate_pine_strategy return every issue found so the script can be fixed,
rather than silently skipping pieces or running a partial/wrong backtest.
"""
from __future__ import annotations

import bisect
import glob
import os
import re
from datetime import datetime, timezone
from typing import Optional

from tradingview_mcp.core.services.custom_strategy_service import (
    _fetch_csv_ohlcv, _VALID_PERIODS, _VALID_INTERVALS,
)
from tradingview_mcp.core.services.backtest_service import (
    _apply_costs, _calc_metrics, _buy_and_hold_return, _validate_numeric_inputs,
)
from tradingview_mcp.core.services import indicators_calc as ind
from tradingview_mcp.core.services import ict_detectors as ictd
from tradingview_mcp.core.services.pine_lite.parser import (
    parse_statement_line, parse_expression_text, Assign, MultiAssign, FuncDef, CallStmt,
)
from tradingview_mcp.core.services.pine_lite.validator import validate_script
from tradingview_mcp.core.services.pine_lite.evaluator import ScriptEnv, EvalError

# ─── Config ────────────────────────────────────────────────────────────────────

PINE_STRATEGIES_DIR = os.environ.get(
    "TV_MCP_PINE_STRATEGIES_DIR",
    os.path.expanduser("~/Claude/pine_strategies"),
)

_SL_MODES = {"fixed_pct", "atr_mult"}
_TP_MODES = {"fixed_pct", "atr_mult", "rr_multiple", "signal_reversal"}
_CONDITION_BLOCKS = ["Trend Filter", "Buy Condition", "Sell Condition", "Buy Exit", "Sell Exit"]

_SECTION_HEADER_RE = re.compile(r"^\[([^\]]+)\]\s*$", re.MULTILINE)


# ─── Daily / weekly bias bridge ────────────────────────────────────────────────
# Exposes ict_detectors.classify_daily_bias_v2 / classify_weekly_profile (built
# earlier for the ICT registry engine) as extra per-bar Pine variables --
# dailyBiasBullish / dailyBiasBearish / weeklyBiasBullish / weeklyBiasBearish
# -- alongside open/high/low/close/volume, so a Pine-lite [Trend Filter] or
# Buy/Sell Condition can reference them directly. Not real Pine syntax; this
# is a server-specific extension, documented in run_pine_backtest's docstring.
# Fail-CLOSED (both flags 0.0) when bias is undefined/unclassified, unlike the
# ICT registry's fail-open gating -- a trend filter's whole purpose here is to
# trade LESS during ambiguous regimes, not to pass through when unsure.

def _weekly_bias_per_day(daily_candles: list[dict]) -> list[Optional[bool]]:
    """Per-daily-candle weekly bullish flag, from classify_weekly_profile of
    the Mon-Fri week containing that day. None where the week isn't
    classifiable (insufficient data, or profile == 'unclassified')."""
    n = len(daily_candles)
    out: list[Optional[bool]] = [None] * n
    i = 0
    while i < n:
        dt0 = datetime.strptime(daily_candles[i]["date"][:10], "%Y-%m-%d")
        week_key = dt0.toordinal() - dt0.weekday()
        j = i
        while j < n:
            dtj = datetime.strptime(daily_candles[j]["date"][:10], "%Y-%m-%d")
            if dtj.toordinal() - dtj.weekday() != week_key:
                break
            j += 1
        result = ictd.classify_weekly_profile(daily_candles[i:j])
        bullish = result.get("bullish") if result.get("profile") not in (None, "insufficient_data", "unclassified") else None
        for k in range(i, j):
            out[k] = bullish
        i = j
    return out


def _bias_series_for(candles: list[dict], daily_candles: list[dict]) -> dict[str, list[float]]:
    """Build dailyBias*/weeklyBias* float series (0.0/1.0) aligned to
    `candles` (any intraday interval), looked up as-of each bar's date."""
    n = len(candles)
    daily_atr = ictd.calc_atr(daily_candles)
    weekly_flags = _weekly_bias_per_day(daily_candles)
    daily_dates = [c["date"][:10] for c in daily_candles]

    daily_bull = [0.0] * n
    daily_bear = [0.0] * n
    weekly_bull = [0.0] * n
    weekly_bear = [0.0] * n

    for i, c in enumerate(candles):
        bias = ictd.daily_bias_asof(daily_candles, c["date"], daily_atr)
        if bias in ("bullish_continuation", "bullish_reversal"):
            daily_bull[i] = 1.0
        elif bias in ("bearish_continuation", "bearish_reversal"):
            daily_bear[i] = 1.0

        idx = bisect.bisect_right(daily_dates, c["date"][:10]) - 1
        wk = weekly_flags[idx] if idx >= 0 else None
        if wk is True:
            weekly_bull[i] = 1.0
        elif wk is False:
            weekly_bear[i] = 1.0

    return {
        "dailyBiasBullish": daily_bull, "dailyBiasBearish": daily_bear,
        "weeklyBiasBullish": weekly_bull, "weeklyBiasBearish": weekly_bear,
    }


_STRING_LITERAL_RE = re.compile(r'"[^"]*"|\'[^\']*\'')


def _merge_continuation_lines(script_text: str) -> list[tuple[str, int]]:
    """Join a Pine call's arguments across physical lines (very common —
    plot()/indicator()/plotshape() calls are routinely wrapped for
    readability) into one logical line before parsing, tracking each
    logical line's STARTING line number for diagnostics. Paren depth is
    computed with string literals blanked out first so a `(` inside a
    plotted title string doesn't miscount."""
    merged: list[tuple[str, int]] = []
    buf: list[str] = []
    buf_start: Optional[int] = None
    depth = 0
    for idx, raw_line in enumerate(script_text.splitlines(), start=1):
        code = raw_line.split("//", 1)[0]
        if not code.strip() and depth == 0:
            continue
        if buf_start is None:
            buf_start = idx
        buf.append(code)
        depth += _STRING_LITERAL_RE.sub("", code).count("(") - _STRING_LITERAL_RE.sub("", code).count(")")
        if depth <= 0:
            merged.append((" ".join(buf), buf_start))
            buf, buf_start, depth = [], None, 0
    if buf:
        merged.append((" ".join(buf), buf_start))
    return merged


# ─── File discovery ────────────────────────────────────────────────────────────

def list_pine_strategy_files() -> list[str]:
    if not os.path.isdir(PINE_STRATEGIES_DIR):
        return []
    return sorted(glob.glob(os.path.join(PINE_STRATEGIES_DIR, "*.txt")))


# ─── .txt parser ───────────────────────────────────────────────────────────────

def _field(text: str, name: str, default: str = "") -> str:
    m = re.search(rf"^{name}:\s*(.+)$", text, re.MULTILINE)
    return m.group(1).strip() if m else default


def _split_sections(text: str) -> dict[str, str]:
    """Split into named sections keyed by their `[Header]` line. NAME:/
    SYMBOL_DEFAULT:/etc top-level fields are extracted separately via
    _field(), same as custom_strategy_service.parse_strategy_file."""
    sections: dict[str, list[str]] = {}
    current: Optional[str] = None
    for line in text.splitlines():
        m = _SECTION_HEADER_RE.match(line)
        if m:
            current = m.group(1).strip()
            sections[current] = []
            continue
        if current is not None:
            sections[current].append(line)
    return {k: "\n".join(v) for k, v in sections.items()}


def _parse_sl_tp(section_text: str, valid_modes: set) -> Optional[dict]:
    """[SL]/[TP] body is a single `mode: value` line (or bare `signal_reversal`
    with no value, for [TP] only)."""
    line = next(
        (ln.strip() for ln in section_text.splitlines() if ln.strip() and not ln.strip().startswith("//")),
        "",
    )
    if not line:
        return {"error": "empty block"}
    if ":" in line:
        mode, _, val = line.partition(":")
        mode = mode.strip()
        if mode not in valid_modes:
            return {"error": f"unrecognized mode '{mode}'. Expected one of: {sorted(valid_modes)}"}
        try:
            return {"mode": mode, "value": float(val.strip())}
        except ValueError:
            return {"error": f"'{mode}' requires a numeric value, got '{val.strip()}'"}
    mode = line.strip()
    if mode not in valid_modes:
        return {"error": f"unrecognized mode '{mode}'. Expected one of: {sorted(valid_modes)}"}
    if mode != "signal_reversal":
        return {"error": f"'{mode}' requires a numeric value like '{mode}: 1.5'"}
    return {"mode": mode, "value": None}


def parse_pine_strategy_file(path: str) -> dict:
    """Parse one Pine-lite strategy file into {name, symbol_default,
    interval, period, statements, conditions, sl, tp}, or
    {"error": ..., "issues": [...]} listing EVERY problem found (parse
    errors, unsupported constructs, missing/invalid SL/TP blocks) — never a
    partial result."""
    if not os.path.isfile(path):
        return {"error": f"Strategy file not found: {path}"}

    text = open(path, "r", encoding="utf-8").read()
    name = _field(text, "NAME", os.path.splitext(os.path.basename(path))[0])
    symbol = _field(text, "SYMBOL_DEFAULT", "")
    interval = _field(text, "INTERVAL", "")
    period = _field(text, "PERIOD", "")

    sections = _split_sections(text)
    if "SCRIPT" not in sections or not sections["SCRIPT"].strip():
        return {"error": "Missing required [SCRIPT] section.", "file": path, "name": name}
    if "Buy Condition" not in sections or not sections["Buy Condition"].strip():
        return {"error": "Missing required [Buy Condition] section.", "file": path, "name": name}

    statements_meta = []  # (Statement|None, err|None, line_no, raw_text)
    for merged_line, line_no in _merge_continuation_lines(sections["SCRIPT"]):
        stmt, err = parse_statement_line(merged_line, line_no)
        if stmt is None and err is None:
            continue  # blank/comment
        statements_meta.append((stmt, err, line_no, merged_line.strip()))

    condition_exprs = []  # (Node|None, err|None, block_name, line_no, raw_text)
    parsed_conditions: dict = {}
    for block in _CONDITION_BLOCKS:
        if block not in sections or not sections[block].strip():
            continue
        expr, err = parse_expression_text(sections[block], 1)
        condition_exprs.append((expr, err, block, 1, sections[block].strip()))
        if err is None and expr is not None:
            parsed_conditions[block] = expr

    issues = validate_script(statements_meta, condition_exprs)

    sl_raw = _parse_sl_tp(sections.get("SL", ""), _SL_MODES) if "SL" in sections else {"error": "missing"}
    tp_raw = _parse_sl_tp(sections.get("TP", ""), _TP_MODES) if "TP" in sections else {"error": "missing"}
    if "error" in sl_raw:
        issues.append({"line": None, "block": "SL", "message": f"[SL] block problem: {sl_raw['error']}"})
        sl_raw = None
    if "error" in tp_raw:
        issues.append({"line": None, "block": "TP", "message": f"[TP] block problem: {tp_raw['error']}"})
        tp_raw = None

    if issues:
        return {
            "error": f"{len(issues)} issue(s) found — fix these before backtesting.",
            "issues": issues, "file": path, "name": name,
        }

    statements = [s for s, e, ln, rt in statements_meta if e is None]

    return {
        "name": name, "file": path,
        "symbol_default": symbol or None,
        "interval": interval or None,
        "period": period or None,
        "statements": statements,
        "conditions": parsed_conditions,
        "sl": sl_raw, "tp": tp_raw,
    }


# ─── Public API: list / validate ───────────────────────────────────────────────

def list_pine_strategies() -> dict:
    files = list_pine_strategy_files()
    out = []
    for f in files:
        parsed = parse_pine_strategy_file(f)
        out.append({
            "strategy_name": os.path.splitext(os.path.basename(f))[0],
            "name": parsed.get("name", os.path.splitext(os.path.basename(f))[0]),
            "status": "ok" if "error" not in parsed else "has_issues",
            "issue_count": len(parsed.get("issues", [])) if "error" in parsed else 0,
        })
    return {"strategies_dir": PINE_STRATEGIES_DIR, "strategies": out}


def validate_pine_strategy(strategy_name: str) -> dict:
    """Diagnostics-only dry run — parses and validates without loading any
    CSV data or running a backtest. Use this to iterate on a script."""
    path = os.path.join(PINE_STRATEGIES_DIR, f"{strategy_name}.txt")
    parsed = parse_pine_strategy_file(path)
    if "error" in parsed:
        return parsed
    var_names = [s.name for s in parsed["statements"] if hasattr(s, "name")]
    var_names += [n for s in parsed["statements"] if hasattr(s, "names") for n in s.names]
    return {
        "status": "ok", "name": parsed["name"],
        "message": "Script parses clean — no unsupported constructs found.",
        "variables_defined": var_names,
        "conditions_present": list(parsed["conditions"].keys()),
        "sl": parsed["sl"], "tp": parsed["tp"],
    }


# ─── Simulation ────────────────────────────────────────────────────────────────

def _sl_distance(entry_price: float, sl_cfg: dict, atr_at_i: Optional[float]) -> Optional[float]:
    if sl_cfg["mode"] == "fixed_pct":
        return entry_price * sl_cfg["value"] / 100.0
    if sl_cfg["mode"] == "atr_mult":
        return atr_at_i * sl_cfg["value"] if atr_at_i else None
    return None


def _tp_distance(risk: Optional[float], tp_cfg: dict, entry_price: float, atr_at_i: Optional[float]) -> Optional[float]:
    if tp_cfg["mode"] == "fixed_pct":
        return entry_price * tp_cfg["value"] / 100.0
    if tp_cfg["mode"] == "atr_mult":
        return atr_at_i * tp_cfg["value"] if atr_at_i else None
    if tp_cfg["mode"] == "rr_multiple":
        return risk * tp_cfg["value"] if risk else None
    return None  # signal_reversal -- no fixed target


def _simulate(
    candles: list[dict], buy_cond, sell_cond, buy_exit, sell_exit, trend_filter,
    sl_cfg: dict, tp_cfg: dict, atr: list,
) -> list[dict]:
    """Single-position bar-by-bar simulation, signal-driven rather than
    zone-tap-driven (this file's own engine, not custom_strategy_service's
    zone engine — the entry trigger here is a boolean condition series, not
    a price zone). When tp_cfg is signal_reversal, an opposite-direction
    signal closes the current position and immediately opens the reverse
    position in the same bar (always-in-market, per the user's own
    clarification of how many indicator-driven systems actually trade)."""
    trades: list[dict] = []
    position: Optional[dict] = None
    n = len(candles)
    reversal_mode = tp_cfg["mode"] == "signal_reversal"

    def trend_ok(i: int) -> bool:
        if trend_filter is None:
            return True
        return bool(trend_filter[i])

    def open_position(direction: str, i: int) -> dict:
        entry_price = candles[i]["close"]
        a = atr[i]
        risk = _sl_distance(entry_price, sl_cfg, a)
        sl_val = None
        if risk is not None:
            sl_val = (entry_price - risk) if direction == "long" else (entry_price + risk)
        tp_dist = _tp_distance(risk, tp_cfg, entry_price, a)
        tp_val = None
        if tp_dist is not None:
            tp_val = (entry_price + tp_dist) if direction == "long" else (entry_price - tp_dist)
        return {"direction": direction, "entry_date": candles[i]["date"], "entry_price": entry_price,
                "sl": sl_val, "tp": tp_val}

    def _try_open(direction: str, i: int) -> Optional[dict]:
        """[SL] is a mandatory hard risk cutoff by file-format design (see
        module docstring) -- but atr_mult mode can't produce a value during
        ATR(14)'s warmup window (first ~14 bars), and rr_multiple TP depends
        on SL. Without this guard, a signal firing during warmup opens a
        position with sl=None and tp=None -- an un-exitable position that
        silently blocks every subsequent signal for the REST of the
        backtest (this produced a real 0-trades-across-4-timeframes bug,
        found by checking why a signal count didn't match a trade count).
        Skip the signal instead of opening a position with no way out."""
        pos = open_position(direction, i)
        return pos if pos["sl"] is not None else None

    for i in range(1, n):
        c = candles[i]

        if position is not None:
            direction = position["direction"]
            exit_price = None

            if position["sl"] is not None:
                if direction == "long" and c["low"] <= position["sl"]:
                    exit_price = position["sl"]
                elif direction == "short" and c["high"] >= position["sl"]:
                    exit_price = position["sl"]

            if exit_price is None and position["tp"] is not None:
                if direction == "long" and c["high"] >= position["tp"]:
                    exit_price = position["tp"]
                elif direction == "short" and c["low"] <= position["tp"]:
                    exit_price = position["tp"]

            if exit_price is None:
                if direction == "long" and buy_exit is not None and buy_exit[i]:
                    exit_price = c["close"]
                elif direction == "short" and sell_exit is not None and sell_exit[i]:
                    exit_price = c["close"]

            reversal_signal = None
            if exit_price is None and reversal_mode:
                if direction == "long" and sell_cond is not None and sell_cond[i] and trend_ok(i):
                    exit_price = c["close"]
                    reversal_signal = "short"
                elif direction == "short" and buy_cond[i] and trend_ok(i):
                    exit_price = c["close"]
                    reversal_signal = "long"

            if exit_price is not None:
                trades.append({
                    "entry_date": position["entry_date"], "entry_price": position["entry_price"],
                    "exit_date": c["date"], "exit_price": exit_price,
                })
                position = _try_open(reversal_signal, i) if reversal_signal else None
            continue

        if buy_cond[i] and trend_ok(i):
            position = _try_open("long", i)
        elif sell_cond is not None and sell_cond[i] and trend_ok(i):
            position = _try_open("short", i)

    return trades


# ─── Shared prep: parse + load data + evaluate script/conditions once ─────────
# (SL/TP choice never affects script/condition evaluation, only _simulate --
# so backtest and optimize both reuse this instead of re-running the
# transpiler/evaluator per SL/TP combination.)

def _prepare(
    strategy_name: str, symbol: Optional[str], period: Optional[str], interval: Optional[str],
    initial_capital: float, commission_pct: float, slippage_pct: float,
):
    """Returns either {"error": ...} or a dict with everything needed to run
    one or many _simulate() calls: candles, conditions, atr, parsed."""
    path = os.path.join(PINE_STRATEGIES_DIR, f"{strategy_name}.txt")
    parsed = parse_pine_strategy_file(path)
    if "error" in parsed:
        return parsed

    symbol = symbol or parsed["symbol_default"]
    if not symbol:
        return {"error": "No symbol given and no SYMBOL_DEFAULT in strategy file."}
    period = (period or parsed["period"] or "1y").lower().strip()
    interval = (interval or parsed["interval"] or "1h").lower().strip()

    if period not in _VALID_PERIODS:
        return {"error": f"Invalid period '{period}'. Choose: {sorted(_VALID_PERIODS)}"}
    if interval not in _VALID_INTERVALS:
        return {"error": f"Invalid interval '{interval}'. Choose: {sorted(_VALID_INTERVALS)}"}

    num_err = _validate_numeric_inputs(initial_capital, commission_pct, slippage_pct)
    if num_err:
        return {"error": num_err}

    try:
        candles = _fetch_csv_ohlcv(symbol, period, interval)
    except Exception as e:
        return {"error": f"Failed to fetch data for '{symbol}': {e}"}

    min_bars = 30 if interval == "1d" else 50
    if len(candles) < min_bars:
        return {"error": f"Not enough data ({len(candles)} bars). Try a longer period."}

    env = ScriptEnv(candles)
    if interval != "1d":
        try:
            daily_candles = _fetch_csv_ohlcv(symbol, period, "1d")
            if len(daily_candles) >= 3:
                env.vars.update(_bias_series_for(candles, daily_candles))
        except Exception:
            pass  # daily bias vars simply won't be available; only an error if the script actually references them
    try:
        env.run_script(parsed["statements"])
        conditions = {block: env.eval_condition(expr) for block, expr in parsed["conditions"].items()}
    except EvalError as e:
        return {"error": f"Runtime error evaluating the script: {e}"}

    if "Buy Condition" not in conditions:
        return {"error": "[Buy Condition] is required."}

    atr = ind.calc_atr(
        [c["high"] for c in candles], [c["low"] for c in candles], [c["close"] for c in candles], 14,
    )

    return {
        "parsed": parsed, "symbol": symbol, "period": period, "interval": interval,
        "candles": candles, "conditions": conditions, "atr": atr,
    }


# ─── Public API: backtest_pine_strategy ────────────────────────────────────────

def run_pine_backtest(
    strategy_name: str,
    symbol: Optional[str] = None,
    period: Optional[str] = None,
    interval: Optional[str] = None,
    initial_capital: float = 10_000.0,
    commission_pct: float = 0.1,
    slippage_pct: float = 0.05,
) -> dict:
    prep = _prepare(strategy_name, symbol, period, interval, initial_capital, commission_pct, slippage_pct)
    if "error" in prep:
        return prep
    parsed, candles, conditions, atr = prep["parsed"], prep["candles"], prep["conditions"], prep["atr"]
    symbol, period, interval = prep["symbol"], prep["period"], prep["interval"]

    trend_filter = conditions.get("Trend Filter")
    buy_cond = conditions["Buy Condition"]
    sell_cond = conditions.get("Sell Condition")
    buy_exit = conditions.get("Buy Exit")
    sell_exit = conditions.get("Sell Exit")
    sl_cfg, tp_cfg = parsed["sl"], parsed["tp"]

    raw_trades = _simulate(candles, buy_cond, sell_cond, buy_exit, sell_exit, trend_filter, sl_cfg, tp_cfg, atr)
    trades = _apply_costs(raw_trades, commission_pct, slippage_pct)
    metrics = _calc_metrics(trades, initial_capital, interval)
    bnh = _buy_and_hold_return(candles)

    return {
        "symbol": symbol.upper(), "strategy_name": parsed["name"],
        "interpreted_rules": {
            "trend_filter_active": trend_filter is not None,
            "sell_condition_active": sell_cond is not None,
            "sl_mode": sl_cfg["mode"], "sl_value": sl_cfg["value"],
            "tp_mode": tp_cfg["mode"], "tp_value": tp_cfg["value"],
            "engine": "pine_lite",
        },
        "period": period, "interval": interval,
        "candles_analyzed": len(candles),
        "date_from": candles[0]["date"], "date_to": candles[-1]["date"],
        "initial_capital": round(initial_capital, 2),
        **metrics,
        "buy_and_hold_return_pct": bnh,
        "vs_buy_and_hold_pct": round(metrics["total_return_pct"] - bnh, 2),
        "recent_trades": trades[-5:],
        "data_source": f"CSV ({symbol.upper()})",
        "disclaimer": (
            "pine_lite v1: a constrained Pine Script subset transpiler, not a full Pine "
            "interpreter — call validate_pine_strategy for exactly what is/isn't supported."
        ),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ─── Public API: optimize_pine_strategy ────────────────────────────────────────
# Pine-lite files don't declare entry/SL/TP OPTIONS lists like the plain-
# English .txt strategies do -- there's just one [SL]/[TP] mode+value. So
# "optimize" here sweeps a small SL/TP mode*value grid instead, seeded from
# the file's OWN declared values (0.5x/1x/1.5x/2x) rather than an arbitrary
# global grid, keeping the sweep specific to what the script's author judged
# reasonable. Same walk-forward-always-on philosophy as optimize_custom_strategy.

def _sl_candidates(sl_cfg: dict) -> list:
    base = sl_cfg["value"]
    out = []
    for mode in ("fixed_pct", "atr_mult"):
        for mult in (0.5, 1.0, 1.5, 2.0):
            out.append({"mode": mode, "value": round(base * mult, 4)})
    # de-dupe identical (mode, value) pairs, keep the file's own setting first
    seen, uniq = set(), []
    ordered = [sl_cfg] + out
    for c in ordered:
        key = (c["mode"], c["value"])
        if key not in seen:
            seen.add(key)
            uniq.append(c)
    return uniq


def _tp_candidates(tp_cfg: dict) -> list:
    out = [{"mode": "signal_reversal", "value": None}]
    base = tp_cfg["value"] if tp_cfg["mode"] != "signal_reversal" else None
    for mode in ("fixed_pct", "atr_mult"):
        for mult in (0.5, 1.0, 1.5, 2.0):
            v = round((base or 2.0) * mult, 4)
            out.append({"mode": mode, "value": v})
    for rr in (1.0, 1.5, 2.0, 3.0):
        out.append({"mode": "rr_multiple", "value": rr})
    seen, uniq = set(), []
    ordered = [tp_cfg] + out
    for c in ordered:
        key = (c["mode"], c["value"])
        if key not in seen:
            seen.add(key)
            uniq.append(c)
    return uniq


def optimize_pine_strategy(
    strategy_name: str,
    symbol: Optional[str] = None,
    period: Optional[str] = None,
    interval: Optional[str] = None,
    initial_capital: float = 10_000.0,
    commission_pct: float = 0.1,
    slippage_pct: float = 0.05,
    rank_by: str = "sharpe_ratio",
    n_splits: int = 3,
    train_ratio: float = 0.7,
    max_combos: int = 40,
) -> dict:
    """Sweep an SL/TP mode*value grid (seeded from the file's own declared
    values) and rank by `rank_by`. Walk-forward robustness is always
    computed alongside the ranking, same as optimize_custom_strategy.
    """
    prep = _prepare(strategy_name, symbol, period, interval, initial_capital, commission_pct, slippage_pct)
    if "error" in prep:
        return prep
    parsed, candles, conditions, atr = prep["parsed"], prep["candles"], prep["conditions"], prep["atr"]
    symbol, period, interval = prep["symbol"], prep["period"], prep["interval"]

    trend_filter = conditions.get("Trend Filter")
    buy_cond = conditions["Buy Condition"]
    sell_cond = conditions.get("Sell Condition")
    buy_exit = conditions.get("Buy Exit")
    sell_exit = conditions.get("Sell Exit")

    sl_candidates = _sl_candidates(parsed["sl"])
    tp_candidates = _tp_candidates(parsed["tp"])
    combos = [(sl, tp) for sl in sl_candidates for tp in tp_candidates][:max_combos]

    n = len(candles)
    fold_size = n // n_splits
    results = []

    for sl_cfg, tp_cfg in combos:
        raw_full = _simulate(candles, buy_cond, sell_cond, buy_exit, sell_exit, trend_filter, sl_cfg, tp_cfg, atr)
        trades_full = _apply_costs(raw_full, commission_pct, slippage_pct)
        m_full = _calc_metrics(trades_full, initial_capital, interval)

        oos_returns = []
        for f in range(n_splits):
            start = f * fold_size
            end = n if f == n_splits - 1 else (f + 1) * fold_size
            fold_slice = slice(start, end)
            fold_candles = candles[fold_slice]
            if len(fold_candles) < 20:
                continue
            split = int(len(fold_candles) * train_ratio)
            test_start = start + split
            if end - test_start < 10:
                continue
            fold_buy = buy_cond[test_start:end]
            fold_sell = sell_cond[test_start:end] if sell_cond is not None else None
            fold_buy_exit = buy_exit[test_start:end] if buy_exit is not None else None
            fold_sell_exit = sell_exit[test_start:end] if sell_exit is not None else None
            fold_trend = trend_filter[test_start:end] if trend_filter is not None else None
            fold_atr = atr[test_start:end]
            test_candles = candles[test_start:end]
            raw_oos = _simulate(test_candles, fold_buy, fold_sell, fold_buy_exit, fold_sell_exit,
                                 fold_trend, sl_cfg, tp_cfg, fold_atr)
            trades_oos = _apply_costs(raw_oos, commission_pct, slippage_pct)
            m_oos = _calc_metrics(trades_oos, initial_capital, interval)
            oos_returns.append(m_oos["total_return_pct"])

        if oos_returns:
            avg_oos = sum(oos_returns) / len(oos_returns)
            consistency = sum(1 for r in oos_returns if r > 0) / len(oos_returns)
        else:
            avg_oos, consistency = 0.0, 0.0

        if m_full["total_return_pct"] == 0:
            verdict = "WEAK"
        elif avg_oos <= 0:
            verdict = "OVERFITTED"
        elif consistency >= 0.66 and avg_oos >= 0.5 * m_full["total_return_pct"]:
            verdict = "ROBUST"
        else:
            verdict = "MODERATE"

        results.append({
            "sl_mode": sl_cfg["mode"], "sl_value": sl_cfg["value"],
            "tp_mode": tp_cfg["mode"], "tp_value": tp_cfg["value"],
            "total_return_pct": m_full["total_return_pct"],
            "sharpe_ratio": m_full["sharpe_ratio"],
            "profit_factor": m_full["profit_factor"],
            "win_rate_pct": m_full["win_rate_pct"],
            "total_trades": m_full["total_trades"],
            "max_drawdown_pct": m_full["max_drawdown_pct"],
            "walk_forward_avg_oos_return_pct": round(avg_oos, 2),
            "walk_forward_consistency": round(consistency, 2),
            "walk_forward_verdict": verdict,
        })

    valid_rank_keys = {"sharpe_ratio", "total_return_pct", "profit_factor", "calmar_ratio"}
    if rank_by not in valid_rank_keys:
        rank_by = "sharpe_ratio"
    results.sort(key=lambda r: r.get(rank_by, 0), reverse=True)
    for i, r in enumerate(results):
        r["rank"] = i + 1

    bnh = _buy_and_hold_return(candles)

    return {
        "symbol": symbol.upper(), "strategy_name": parsed["name"],
        "engine": "pine_lite",
        "period": period, "interval": interval,
        "candles_analyzed": len(candles),
        "combinations_tested": len(combos),
        "ranked_by": rank_by,
        "buy_and_hold_return_pct": bnh,
        "results": results,
        "best_by_rank": results[0] if results else None,
        "most_robust": max(results, key=lambda r: r["walk_forward_consistency"]) if results else None,
        "warning": (
            "'best_by_rank' is ranked purely on this period's fit — check "
            "'walk_forward_verdict' on the same row before trusting it. "
            "SL/TP grid is seeded from the file's own declared values "
            "(0.5x-2x), not an exhaustive search."
        ),
        "disclaimer": (
            "pine_lite v1: a constrained Pine Script subset transpiler, not a full Pine "
            "interpreter — call validate_pine_strategy for exactly what is/isn't supported."
        ),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

"""
Custom Strategy Service for tradingview-mcp — user extension

Reads plain-English .txt strategy files from a folder, interprets their
ZONES / ENTRY_CONFIRMATION / STOP_LOSS / TAKE_PROFIT / FILTERS sections,
and backtests them against OHLC loaded from CSV files you provide (one per
symbol+timeframe), reusing the same cost/Sharpe/Calmar metrics math as the
built-in backtest_service.py — so results are in the same format as the 9
built-in strategies. No live connection (MT5, Yahoo, etc.) is required at
call time; everything reads from disk.

REQUIRES: CSV files placed under ~/Claude/data (or TV_MCP_CSV_DATA_DIR),
named like 'XAUUSD_15m.csv', with columns date/open/high/low/close
(volume optional). See _fetch_csv_ohlcv's docstring for accepted formats.

IMPORTANT — read before trusting results:
This is a *generic proxy engine* for supply/demand-style strategies, not a
line-for-line port of your SD_CVD_Flow_EA.mq5 / MQL5 logic. It implements:
  - swing-based zone detection (impulse candle + swing pivot)
  - a small set of named entry-confirmation modes
  - a small set of named SL modes
  - a small set of named TP modes
It does NOT implement your MTF MSS confirmation, CVD flow, or your exact
IDM/POI/BOS/CHOCH state machine. Treat outputs as "does this general shape
of rule survive contact with data" — not a stand-in for your MQL5 backtest.
Every result includes `interpreted_rules` so you can see exactly what was
run and catch any misreading of your .txt file before trusting the numbers.
"""
from __future__ import annotations

import csv
import glob
import itertools
import os
import re
import statistics
import math
from datetime import datetime, timedelta, timezone
from typing import Optional

from tradingview_mcp.core.services.backtest_service import (
    _apply_costs, _calc_metrics, _buy_and_hold_return, _validate_numeric_inputs,
)

# ─── Config ────────────────────────────────────────────────────────────────────

STRATEGIES_DIR = os.environ.get(
    "TV_MCP_CUSTOM_STRATEGIES_DIR",
    os.path.expanduser("~/Claude/strategies"),
)

# Data source is CSV files you provide — a separate path from the Yahoo-
# based built-in tools, since Yahoo cannot deliver granular (M1/M5/M15)
# intraday history at any real depth, and MT5 requires a live terminal
# connection which proved unworkable in practice. No live connection of
# any kind is required here; everything is read from disk.

_VALID_PERIODS  = {"1mo", "3mo", "6mo", "1y", "2y", "3y", "4y", "5y", "all"}
_PERIOD_TO_DAYS = {"1mo": 30, "3mo": 90, "6mo": 182, "1y": 365, "2y": 730, "3y": 1095, "4y": 1460, "5y": 1825}

_VALID_INTERVALS = {"1m", "5m", "15m", "30m", "1h", "4h", "1d"}
_ANNUALIZATION   = {
    "1m": 252 * 6.5 * 60, "5m": 252 * 6.5 * 12, "15m": 252 * 6.5 * 4,
    "30m": 252 * 6.5 * 2, "1h": 252 * 6, "4h": 252 * 1.5, "1d": 252,
}

CSV_DATA_DIR = os.environ.get(
    "TV_MCP_CSV_DATA_DIR",
    os.path.expanduser("~/Claude/data"),
)


def _fix_annualization(metrics: dict, interval: str) -> dict:
    """The imported _calc_metrics() only has correct Sharpe-annualization
    factors for '1d'/'1h' (it falls back to 252 for anything else). For our
    additional M1/M5/M15/M30/H4 intervals, rescale sharpe_ratio and
    calmar_ratio to the correct per-bar annualization factor rather than
    silently reporting a wrong number.
    """
    if interval in ("1d", "1h") or metrics.get("total_trades", 0) == 0:
        return metrics
    correct_ann = _ANNUALIZATION.get(interval, 252)
    fallback_ann = 252  # what _calc_metrics used internally for unknown intervals
    scale = math.sqrt(correct_ann / fallback_ann)
    metrics = dict(metrics)
    metrics["sharpe_ratio"] = round(metrics["sharpe_ratio"] * scale, 2)
    return metrics


def _find_csv_file(symbol: str, interval: str) -> Optional[str]:
    """Look for a CSV matching this symbol+interval under CSV_DATA_DIR.
    Tries a few common naming conventions so you don't have to rename files
    to match exactly:
        XAUUSD_15m.csv, XAUUSD-15m.csv, XAUUSD15m.csv,
        xauusd_15m.csv (case-insensitive), XAUUSD_M15.csv (MT5-style)
    """
    if not os.path.isdir(CSV_DATA_DIR):
        return None
    mt5_style = {"1m": "M1", "5m": "M5", "15m": "M15", "30m": "M30",
                 "1h": "H1", "4h": "H4", "1d": "D1"}.get(interval, interval)
    candidates = [
        f"{symbol}_{interval}.csv", f"{symbol}-{interval}.csv", f"{symbol}{interval}.csv",
        f"{symbol}_{mt5_style}.csv", f"{symbol}-{mt5_style}.csv", f"{symbol}{mt5_style}.csv",
    ]
    lower_map = {f.lower(): f for f in os.listdir(CSV_DATA_DIR) if f.lower().endswith(".csv")}
    for c in candidates:
        if c.lower() in lower_map:
            return os.path.join(CSV_DATA_DIR, lower_map[c.lower()])
    return None


_DATE_FORMATS = (
    "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y.%m.%d %H:%M:%S",
    "%Y.%m.%d %H:%M", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d", "%Y.%m.%d",
)


def _parse_csv_date(raw: str) -> Optional[datetime]:
    raw = raw.strip()
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(raw, fmt)
        except ValueError:
            continue
    return None


def _detect_date_format(raw: str) -> Optional[str]:
    raw = raw.strip()
    for fmt in _DATE_FORMATS:
        try:
            datetime.strptime(raw, fmt)
            return fmt
        except ValueError:
            continue
    return None


# Rough bars/day used to size the tail-read window; deliberately generous
# (headroom multiplier applied on top) since underestimating just costs a
# retry with a bigger window, not incorrect results.
_BARS_PER_DAY = {"1m": 1440, "5m": 288, "15m": 96, "30m": 48, "1h": 24, "4h": 6, "1d": 1}


def _read_tail_lines(path: str, min_lines: int, chunk_size: int = 1 << 20) -> list[str]:
    """Read complete lines from the end of a text file until at least
    `min_lines` have been collected (or the whole file has been consumed),
    without loading files that are far larger than the requested window into
    memory. Returns lines in original file order (oldest of the tail first),
    not including any partial line at the very start of the read region.
    """
    with open(path, "rb") as f:
        f.seek(0, os.SEEK_END)
        file_size = f.tell()
        pos = file_size
        newline_count = 0
        blocks: list[bytes] = []
        while pos > 0 and newline_count <= min_lines:
            read_size = min(chunk_size, pos)
            pos -= read_size
            f.seek(pos)
            block = f.read(read_size)
            blocks.append(block)
            newline_count += block.count(b"\n")
        data = b"".join(reversed(blocks))
    text = data.decode("utf-8-sig", errors="ignore")
    lines = text.split("\n")
    if lines and pos > 0:
        lines = lines[1:]  # drop the partial first line unless we read from byte 0
    return [ln.rstrip("\r") for ln in lines]


def _fetch_csv_ohlcv(symbol: str, period: str, interval: str) -> list[dict]:
    """Load OHLCV history from a CSV file you provide under CSV_DATA_DIR
    (default ~/Claude/data, override with TV_MCP_CSV_DATA_DIR).

    Expected columns (header required, case-insensitive, any order):
        date/datetime/time, open, high, low, close, volume (volume optional)
    Date formats accepted: 'YYYY-MM-DD HH:MM[:SS]', 'YYYY.MM.DD HH:MM[:SS]'
    (MT5 export style), or 'YYYY-MM-DD' for daily-only files.

    `period` filters to the trailing N days from the file's latest date
    (same period tokens as before: '1mo','3mo','6mo','1y','2y') — pass
    period='all' to use the entire file.

    For a bounded period, only the tail of the file is read/parsed (via
    _read_tail_lines), not the whole thing — large M1/M5 CSVs (100MB+) would
    otherwise take 30-60s+ just to strptime-parse every row before throwing
    almost all of it away. If the estimated tail window undershoots (e.g. the
    file has denser bars than `_BARS_PER_DAY` assumes), the window doubles
    and retries before falling back to a full-file parse.
    """
    path = _find_csv_file(symbol, interval)
    if path is None:
        raise RuntimeError(
            f"No CSV found for symbol='{symbol}' interval='{interval}' under {CSV_DATA_DIR}. "
            f"Expected a file named like '{symbol}_{interval}.csv'. "
            f"Files present: {sorted(os.listdir(CSV_DATA_DIR)) if os.path.isdir(CSV_DATA_DIR) else '(directory does not exist)'}"
        )

    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        header_line = f.readline()
    if not header_line:
        raise RuntimeError(f"CSV file '{path}' is empty.")
    header = next(csv.reader([header_line]))
    header_lower = [h.strip().lower() for h in header]

    def col(*names):
        for n in names:
            if n in header_lower:
                return header_lower.index(n)
        return None

    idx_date = col("date", "datetime", "time")
    idx_open = col("open")
    idx_high = col("high")
    idx_low = col("low")
    idx_close = col("close")
    idx_vol = col("volume", "tickvol", "tick_volume", "vol")

    missing = [name for name, idx in
               [("date", idx_date), ("open", idx_open), ("high", idx_high),
                ("low", idx_low), ("close", idx_close)] if idx is None]
    if missing:
        raise RuntimeError(
            f"CSV file '{path}' is missing required column(s): {missing}. "
            f"Header found: {header}"
        )

    days = None if period == "all" else _PERIOD_TO_DAYS.get(period)

    def _parse_lines(lines: list[str]) -> list[dict]:
        out = []
        date_fmt: Optional[str] = None
        min_len = max(idx_date, idx_open, idx_high, idx_low, idx_close) + 1
        header_stripped = header_line.rstrip("\r\n")
        for line in lines:
            if not line.strip() or line == header_stripped:
                continue
            row = next(csv.reader([line]))
            if len(row) < min_len:
                continue
            raw_date = row[idx_date]
            if date_fmt is None:
                date_fmt = _detect_date_format(raw_date)
            dt = None
            if date_fmt is not None:
                try:
                    dt = datetime.strptime(raw_date.strip(), date_fmt)
                except ValueError:
                    dt = _parse_csv_date(raw_date)
            else:
                dt = _parse_csv_date(raw_date)
            if dt is None:
                continue
            try:
                out.append({
                    "date": dt.strftime("%Y-%m-%d") if dt.hour == 0 and dt.minute == 0 and interval == "1d"
                            else dt.strftime("%Y-%m-%d %H:%M"),
                    "open": float(row[idx_open]), "high": float(row[idx_high]),
                    "low": float(row[idx_low]), "close": float(row[idx_close]),
                    "volume": float(row[idx_vol]) if idx_vol is not None and row[idx_vol] else 0.0,
                    "_dt": dt,
                })
            except ValueError:
                continue
        return out

    candles: list[dict] = []
    if days is not None:
        bars_per_day = _BARS_PER_DAY.get(interval, 1440)
        est_rows = int(days * bars_per_day * 1.2) + 100
        max_attempts = 4
        for attempt in range(max_attempts):
            lines = _read_tail_lines(path, est_rows)
            candles = _parse_lines(lines)
            if not candles:
                est_rows *= 4
                continue
            candles.sort(key=lambda c: c["_dt"])
            cutoff = candles[-1]["_dt"] - timedelta(days=days)
            undershot = candles[0]["_dt"] > cutoff and len(lines) >= est_rows
            if not undershot or attempt == max_attempts - 1:
                break
            est_rows *= 4  # window was too small to reach the cutoff — retry wider
        candles = [c for c in candles if c["_dt"] >= cutoff]
    else:
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            f.readline()  # skip header
            lines = f.read().split("\n")
        candles = _parse_lines(lines)
        candles.sort(key=lambda c: c["_dt"])

    if not candles:
        raise RuntimeError(f"CSV file '{path}' had no parseable rows.")

    for c in candles:
        del c["_dt"]
    return candles

_ENTRY_MODES = {"close_beyond", "engulfing", "wick_reject", "midpoint_close"}
_SL_MODES    = {"swing_based", "opposing_candle", "fixed_atr"}
_TP_MODES    = {"partial_1_5R", "fixed_2R", "trail_only"}

_SWING_LOOKBACK = 3          # bars each side to confirm a swing pivot
_ZONE_MAX_TAPS  = 2          # matches the FILTERS convention in the sample file
_ATR_PERIOD     = 14


# ─── .txt Parser ───────────────────────────────────────────────────────────────

def list_strategy_files() -> list[str]:
    if not os.path.isdir(STRATEGIES_DIR):
        return []
    return sorted(glob.glob(os.path.join(STRATEGIES_DIR, "*.txt")))


def _extract_section(text: str, header: str) -> str:
    """Grab everything under `HEADER:` up to the next ALL_CAPS header line."""
    pattern = rf"^{header}:\s*\[?OPTIONS\]?\s*(.*?)(?=^[A-Z_]+:|\Z)"
    m = re.search(pattern, text, re.MULTILINE | re.DOTALL)
    return m.group(1).strip() if m else ""


def _extract_options(section_text: str, valid_modes: set[str]) -> list[str]:
    """Pull recognized mode keywords (e.g. `- swing_based: ...`) out of a section."""
    found = []
    for line in section_text.splitlines():
        line = line.strip().lstrip("- ").strip()
        if not line:
            continue
        key = line.split(":")[0].strip()
        if key in valid_modes:
            found.append(key)
    return found


def parse_strategy_file(path: str) -> dict:
    """Parse one .txt strategy file into a structured dict.

    Returns {"error": ...} if the file is missing required sections, rather
    than silently guessing — an ambiguous or malformed rule file should fail
    loudly, not produce a backtest of the wrong strategy.
    """
    if not os.path.isfile(path):
        return {"error": f"Strategy file not found: {path}"}

    text = open(path, "r", encoding="utf-8").read()

    def field(name: str, default: str = "") -> str:
        m = re.search(rf"^{name}:\s*(.+)$", text, re.MULTILINE)
        return m.group(1).strip() if m else default

    name    = field("NAME", os.path.splitext(os.path.basename(path))[0])
    symbol  = field("SYMBOL_DEFAULT", "")
    tf_raw  = field("TIMEFRAMES", "")
    timeframes = [t.strip() for t in tf_raw.split(",") if t.strip()]

    entry_section = _extract_section(text, "ENTRY_CONFIRMATION")
    sl_section    = _extract_section(text, "STOP_LOSS")
    tp_section    = _extract_section(text, "TAKE_PROFIT")

    entry_opts = _extract_options(entry_section, _ENTRY_MODES)
    sl_opts    = _extract_options(sl_section, _SL_MODES)
    tp_opts    = _extract_options(tp_section, _TP_MODES)

    errors = []
    if not entry_opts:
        errors.append(
            f"No recognized ENTRY_CONFIRMATION options found. "
            f"Expected one or more of: {sorted(_ENTRY_MODES)}"
        )
    if not sl_opts:
        errors.append(
            f"No recognized STOP_LOSS options found. "
            f"Expected one or more of: {sorted(_SL_MODES)}"
        )
    if not tp_opts:
        errors.append(
            f"No recognized TAKE_PROFIT options found. "
            f"Expected one or more of: {sorted(_TP_MODES)}"
        )
    if errors:
        return {"error": "; ".join(errors), "file": path}

    max_taps_match = re.search(r"tapped more than (\d+)", text)
    max_taps = int(max_taps_match.group(1)) if max_taps_match else _ZONE_MAX_TAPS

    return {
        "name": name,
        "file": path,
        "symbol_default": symbol or None,
        "timeframes": timeframes,
        "entry_options": entry_opts,
        "sl_options": sl_opts,
        "tp_options": tp_opts,
        "max_zone_taps": max_taps,
        "raw_zones_text": _extract_section(text, "ZONES"),
    }


# ─── Zone detection (shared across all entry/SL/TP combinations) ──────────────

def _find_swings(candles: list[dict]) -> list[dict]:
    """Simple pivot-based swing high/low detector -> supply/demand zone seeds.

    A swing high/low is a bar whose high/low is the extreme within
    +/- _SWING_LOOKBACK bars. The zone boundary is that single impulse
    candle's own high/low (per your "only the initial momentum candle"
    rule), not the full subsequent rally.
    """
    swings = []
    n = len(candles)
    lb = _SWING_LOOKBACK
    for i in range(lb, n - lb):
        window = candles[i - lb:i + lb + 1]
        hi = candles[i]["high"]
        lo = candles[i]["low"]
        if hi == max(c["high"] for c in window):
            swings.append({"index": i, "type": "supply", "top": hi, "bottom": candles[i]["low"]})
        if lo == min(c["low"] for c in window):
            swings.append({"index": i, "type": "demand", "top": candles[i]["high"], "bottom": lo})
    return swings


def _calc_atr_series(candles: list[dict], period: int = _ATR_PERIOD) -> list[Optional[float]]:
    trs = [None]
    for i in range(1, len(candles)):
        h, l, pc = candles[i]["high"], candles[i]["low"], candles[i - 1]["close"]
        trs.append(max(h - l, abs(h - pc), abs(l - pc)))
    atr = [None] * len(candles)
    vals = [t for t in trs if t is not None]
    if len(vals) < period:
        return atr
    running = sum(trs[1:period + 1]) / period
    atr[period] = running
    for i in range(period + 1, len(candles)):
        running = (running * (period - 1) + trs[i]) / period
        atr[i] = running
    return atr


def _confirms_entry(mode: str, candle: dict, prev_candle: dict, zone: dict) -> Optional[str]:
    """Return 'long' / 'short' if `candle` confirms entry into `zone` under `mode`, else None."""
    direction = "long" if zone["type"] == "demand" else "short"
    top, bottom = zone["top"], zone["bottom"]

    touched = candle["low"] <= top and candle["high"] >= bottom

    if mode == "close_beyond":
        if direction == "long" and candle["close"] > top:
            return "long"
        if direction == "short" and candle["close"] < bottom:
            return "short"

    elif mode == "wick_reject":
        if not touched:
            return None
        if direction == "long" and candle["close"] > candle["open"] and candle["close"] > bottom:
            return "long"
        if direction == "short" and candle["close"] < candle["open"] and candle["close"] < top:
            return "short"

    elif mode == "midpoint_close":
        mid = (top + bottom) / 2
        if direction == "long" and candle["close"] > mid:
            return "long"
        if direction == "short" and candle["close"] < mid:
            return "short"

    elif mode == "engulfing":
        body      = abs(candle["close"] - candle["open"])
        prev_body = abs(prev_candle["close"] - prev_candle["open"])
        engulfs   = body > prev_body and (
            (candle["close"] > candle["open"] and candle["close"] >= prev_candle["open"]
             and candle["open"] <= prev_candle["close"])
            or
            (candle["close"] < candle["open"] and candle["close"] <= prev_candle["open"]
             and candle["open"] >= prev_candle["close"])
        )
        if not engulfs:
            return None
        if direction == "long" and candle["close"] > candle["open"]:
            return "long"
        if direction == "short" and candle["close"] < candle["open"]:
            return "short"

    return None


def _run_zone_strategy_with_zones(
    zones: list[dict], candles: list[dict], entry_mode: str, sl_mode: str, tp_mode: str,
    max_zone_taps: int = 2,
) -> list[dict]:
    """Same tap/confirm/SL/TP mechanics as _run_zone_strategy, but the zone
    list is supplied externally (from an ict_strategies.py provider) instead
    of being derived from generic swing-pivot detection. Used for the 18-file
    ICT rulebook, where each strategy has its own zone-definition logic.
    """
    atr = _calc_atr_series(candles)
    zone_taps = {id(z): 0 for z in zones}
    trades: list[dict] = []
    position: Optional[dict] = None

    for i in range(1, len(candles)):
        candle, prev_candle = candles[i], candles[i - 1]

        if position is not None:
            direction = position["direction"]
            exit_price = None
            if direction == "long":
                if candle["low"] <= position["sl"]:
                    exit_price = position["sl"]
                elif position.get("tp") and candle["high"] >= position["tp"]:
                    exit_price = position["tp"]
                elif tp_mode == "trail_only" and candle["low"] > position["entry_price"]:
                    position["sl"] = max(position["sl"], candle["low"])
            else:
                if candle["high"] >= position["sl"]:
                    exit_price = position["sl"]
                elif position.get("tp") and candle["low"] <= position["tp"]:
                    exit_price = position["tp"]
                elif tp_mode == "trail_only" and candle["high"] < position["entry_price"]:
                    position["sl"] = min(position["sl"], candle["high"])
            if exit_price is not None:
                trades.append({
                    "entry_date": position["entry_date"], "entry_price": position["entry_price"],
                    "exit_date": candle["date"], "exit_price": exit_price,
                })
                position = None
            continue

        active_zones = [z for z in zones if z["index"] < i and zone_taps[id(z)] < max_zone_taps]
        for zone in active_zones:
            direction = _confirms_entry(entry_mode, candle, prev_candle, zone)
            if not direction:
                continue
            entry_price = candle["close"]
            zone_taps[id(zone)] += 1

            if sl_mode == "swing_based":
                sl_val = zone["bottom"] if direction == "long" else zone["top"]
            elif sl_mode == "opposing_candle":
                sl_val = candle["low"] if direction == "long" else candle["high"]
            else:  # fixed_atr
                a = atr[i] or (candle["high"] - candle["low"])
                sl_val = entry_price - 1.5 * a if direction == "long" else entry_price + 1.5 * a

            risk = abs(entry_price - sl_val)
            if risk <= 0:
                continue
            if tp_mode == "partial_1_5R":
                tp_val = entry_price + 1.5 * risk if direction == "long" else entry_price - 1.5 * risk
            elif tp_mode == "fixed_2R":
                tp_val = entry_price + 2.0 * risk if direction == "long" else entry_price - 2.0 * risk
            else:
                tp_val = None

            position = {"direction": direction, "entry_date": candle["date"],
                        "entry_price": entry_price, "sl": sl_val, "tp": tp_val}
            break

    return trades


def _parse_timeframe_tokens(timeframes_field: str) -> list[str]:
    """Pull recognized MT5-style timeframe tokens (1m,5m,15m,30m,1h,4h,1d) out
    of a messy free-text TIMEFRAMES line, in the order they appear.

    Hour/day tokens are matched case-insensitively (strategy files aren't
    consistent about '1h' vs '1H', and a lone 'D'/'d' as in "HTF = 15m, 30m,
    1H, 4H, D" is treated as daily). Minute tokens ('1m','5m','15m','30m')
    are matched LOWERCASE-ONLY and deliberately do not use IGNORECASE:
    uppercase 'M' conventionally means MONTH, not minute (e.g. "1M, 1W
    (quarter placement), 1d..." in 12_Quarterly_Theory_Swing.txt) — an
    earlier version of this function used a single case-insensitive regex
    for everything, which made '1M' match as '1m' (1-minute) and land first
    in the token list, silently resolving that file's HTF to 1-minute data
    (hundreds of thousands of candles) and hanging the backtest.
    """
    # Single pass (no blanket IGNORECASE) so tokens stay in original
    # left-to-right order — callers pick htf/ltf by token position, so a
    # multi-pass concatenation would silently reorder every other file's
    # tokens and break their htf/ltf inference.
    tokens = re.findall(r"\b(1m|5m|15m|30m|1[Hh]|4[Hh]|1[Dd]|[Dd])\b", timeframes_field)
    seen, out = set(), []
    for t in tokens:
        norm = "1d" if t.lower() == "d" else t.lower()
        if norm not in seen:
            seen.add(norm)
            out.append(norm)
    return out


_TF_RANK = {"1m": 1, "5m": 2, "15m": 3, "30m": 4, "1h": 5, "4h": 6, "1d": 7}


def _resolve_htf_ltf_tokens(tf_line: str) -> tuple[Optional[str], Optional[str]]:
    """If the TIMEFRAMES line explicitly labels HTF/LTF roles (e.g.
    'HTF (bias/OB) = 15m, 30m, 1H, 4H, D (selectable) | LTF (confirmation) =
    1m, 5m, 15m'), parse each role's token list separately instead of relying
    on first-token/last-token position. A labeled list like this is a MENU of
    acceptable choices per role, not an ordered HTF->LTF payload — picking by
    raw token position previously chose whatever timeframe happened to be
    listed first inside the HTF menu (15m, here) as if it were the intended
    HTF, silently running the backtest on a much larger/slower candle set
    than intended (this is exactly what made 20_HTF_OB_Tap_LTF_MSS ~40x
    slower than comparable strategies). When labels are present, the largest
    token in the HTF list and the smallest in the LTF list are used instead.
    Returns (None, None) if no HTF/LTF labels are found, so callers fall
    back to the existing first/last-token heuristic for normally-formatted
    TIMEFRAMES lines.
    """
    if not re.search(r"\bHTF\b", tf_line, re.IGNORECASE) or not re.search(r"\bLTF\b", tf_line, re.IGNORECASE):
        return None, None
    parts = re.split(r"\bLTF\b", tf_line, maxsplit=1, flags=re.IGNORECASE)
    if len(parts) != 2:
        return None, None
    htf_tokens = _parse_timeframe_tokens(parts[0])
    ltf_tokens = _parse_timeframe_tokens(parts[1])
    if not htf_tokens or not ltf_tokens:
        return None, None
    htf = max(htf_tokens, key=lambda t: _TF_RANK.get(t, 0))
    ltf = min(ltf_tokens, key=lambda t: _TF_RANK.get(t, 99))
    return htf, ltf


def _slice_htf_for_ltf_range(htf_candles: list[dict], ltf_slice: list[dict], lookback: int = 100) -> list[dict]:
    """Restrict htf_candles to the date range covered by ltf_slice (plus a
    lookback buffer of prior bars for swing/OB context), so walk-forward
    folds don't leak zone information from outside the fold's own window.
    """
    if not ltf_slice or not htf_candles:
        return htf_candles
    start_date, end_date = ltf_slice[0]["date"], ltf_slice[-1]["date"]
    idx_start = len(htf_candles)
    idx_end = -1
    for i, c in enumerate(htf_candles):
        if c["date"] >= start_date and idx_start == len(htf_candles):
            idx_start = i
        if c["date"] <= end_date:
            idx_end = i
    if idx_end < 0:
        return htf_candles
    idx_start = max(0, min(idx_start, idx_end) - lookback)
    return htf_candles[idx_start:idx_end + 1]


def _resolve_ict_provider(parsed: dict):
    """If this strategy's NAME has hand-coded ICT detection logic registered,
    return (provider_fn, requires_daily_htf). Otherwise (None, False), meaning
    callers should fall back to the generic swing-pivot proxy.
    """
    from tradingview_mcp.core.services.ict_strategies import STRATEGY_REGISTRY, REQUIRES_DAILY_HTF
    provider = STRATEGY_REGISTRY.get(parsed["name"])
    if provider is None:
        return None, False
    return provider, parsed["name"] in REQUIRES_DAILY_HTF


def _get_ict_zones(
    parsed: dict, provider, requires_daily: bool, symbol: str, period: str,
    ltf_interval: Optional[str] = None, htf_interval: Optional[str] = None,
    broker_utc_offset_hours: float = 0.0,
):
    """Resolve HTF/LTF intervals from the file's TIMEFRAMES field (unless
    overridden), load CSV candles for both, and run the registered zone
    provider. Shared by run_ict_backtest and the ICT-aware paths of
    run_custom_backtest / optimize_custom_strategy so all three dispatch
    through the exact same per-strategy detection logic instead of the
    generic proxy.
    """
    from tradingview_mcp.core.services.ict_strategies import DAILY_BIAS_GATED_STRATEGIES

    tf_line = ", ".join(parsed.get("timeframes") or [])
    labeled_htf, labeled_ltf = _resolve_htf_ltf_tokens(tf_line)
    tf_tokens = _parse_timeframe_tokens(tf_line) or ["1h", "15m"]

    ltf_interval = ltf_interval or labeled_ltf or tf_tokens[-1]
    htf_interval = htf_interval or labeled_htf or (tf_tokens[0] if len(tf_tokens) > 1 else tf_tokens[-1])

    ltf_candles = _fetch_csv_ohlcv(symbol, period, ltf_interval)
    htf_candles = _fetch_csv_ohlcv(symbol, period, htf_interval) if htf_interval != ltf_interval else ltf_candles
    bias_gated = parsed["name"] in DAILY_BIAS_GATED_STRATEGIES
    daily_candles = (
        _fetch_csv_ohlcv(symbol, period, "1d")
        if (requires_daily or bias_gated) and "1d" not in (ltf_interval, htf_interval)
        else None
    )
    htf_arg = daily_candles if requires_daily and daily_candles else htf_candles

    params = {"broker_utc_offset_hours": broker_utc_offset_hours}
    if bias_gated and daily_candles:
        params["daily_candles"] = daily_candles
    elif bias_gated and "1d" in (ltf_interval, htf_interval):
        params["daily_candles"] = htf_candles if htf_interval == "1d" else ltf_candles
    zones = provider(ltf_candles, htf_arg, params)
    return zones, ltf_candles, htf_candles, ltf_interval, htf_interval


def run_ict_backtest(
    strategy_name: str,
    symbol: Optional[str] = None,
    entry: Optional[str] = None,
    sl: Optional[str] = None,
    tp: Optional[str] = None,
    period: str = "3mo",
    ltf_interval: Optional[str] = None,
    htf_interval: Optional[str] = None,
    broker_utc_offset_hours: float = 0.0,
    initial_capital: float = 10_000.0,
    commission_pct: float = 0.1,
    slippage_pct: float = 0.05,
) -> dict:
    """Backtest one of the 18 ICT-rulebook strategies (registry-matched by
    NAME) using its own hand-implemented detection logic, fetching HTF and
    LTF candles from local CSV files.

    ltf_interval / htf_interval: override the file's own TIMEFRAMES tokens
    if you want a specific pair; otherwise the first token in TIMEFRAMES is
    used as HTF and the last as LTF.
    """
    files = list_strategy_files()
    matched_path = None
    for f in files:
        parsed = parse_strategy_file(f)
        if "error" in parsed:
            continue
        if parsed["name"] == strategy_name or os.path.splitext(os.path.basename(f))[0] == strategy_name:
            matched_path = f
            break
    if not matched_path:
        return {"error": f"No strategy file found with NAME or filename '{strategy_name}' in {STRATEGIES_DIR}"}

    parsed = parse_strategy_file(matched_path)
    if "error" in parsed:
        return parsed

    provider, requires_daily = _resolve_ict_provider(parsed)
    if provider is None:
        from tradingview_mcp.core.services.ict_strategies import STRATEGY_REGISTRY
        return {
            "error": f"'{parsed['name']}' has no registered ICT detection logic. "
                     f"Registered strategies: {sorted(STRATEGY_REGISTRY.keys())}. "
                     f"Use backtest_custom_strategy for the generic proxy engine instead."
        }

    entry = entry or parsed["entry_options"][0]
    sl = sl or parsed["sl_options"][0]
    tp = tp or parsed["tp_options"][0]
    if entry not in parsed["entry_options"]:
        return {"error": f"'{entry}' not declared. Options: {parsed['entry_options']}"}
    if sl not in parsed["sl_options"]:
        return {"error": f"'{sl}' not declared. Options: {parsed['sl_options']}"}
    if tp not in parsed["tp_options"]:
        return {"error": f"'{tp}' not declared. Options: {parsed['tp_options']}"}

    symbol = symbol or parsed["symbol_default"]
    if not symbol:
        return {"error": "No symbol given and no SYMBOL_DEFAULT in strategy file."}

    if period not in _VALID_PERIODS:
        return {"error": f"Invalid period '{period}'. Choose: {sorted(_VALID_PERIODS)}"}
    for iv in (ltf_interval, htf_interval):
        if iv is not None and iv not in _VALID_INTERVALS:
            return {"error": f"Invalid interval '{iv}'. Choose: {sorted(_VALID_INTERVALS)}"}

    try:
        zones, ltf_candles, htf_candles, ltf_interval, htf_interval = _get_ict_zones(
            parsed, provider, requires_daily, symbol, period,
            ltf_interval, htf_interval, broker_utc_offset_hours,
        )
    except Exception as e:
        return {"error": f"CSV data load or zone provider for '{parsed['name']}' failed: {e}"}

    if len(ltf_candles) < 50:
        return {"error": f"Not enough LTF data ({len(ltf_candles)} bars). Try a longer period."}

    raw_trades = _run_zone_strategy_with_zones(zones, ltf_candles, entry, sl, tp, parsed["max_zone_taps"])
    trades = _apply_costs(raw_trades, commission_pct, slippage_pct)
    metrics = _fix_annualization(_calc_metrics(trades, initial_capital, ltf_interval), ltf_interval)
    bnh = _buy_and_hold_return(ltf_candles)

    return {
        "symbol": symbol.upper(), "strategy_name": parsed["name"],
        "interpreted_rules": {
            "entry_confirmation": entry, "stop_loss_mode": sl, "take_profit_mode": tp,
            "ltf_interval": ltf_interval, "htf_interval": htf_interval,
            "zones_detected": len(zones),
        },
        "period": period, "ltf_candles_analyzed": len(ltf_candles),
        "date_from": ltf_candles[0]["date"], "date_to": ltf_candles[-1]["date"],
        "initial_capital": round(initial_capital, 2),
        **metrics,
        "buy_and_hold_return_pct": bnh,
        "vs_buy_and_hold_pct": round(metrics["total_return_pct"] - bnh, 2),
        "recent_trades": trades[-5:],
        "data_source": f"CSV ({symbol.upper()})",
        "disclaimer": (
            "Hand-implemented per this file's own definitions; see ict_strategies.py "
            "module docstring for exactly which strategies are fully implemented vs. "
            "parameterized variations of another. Not independently verified against "
            "real market data — verify 'zones_detected' and 'recent_trades' before trusting results."
        ),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def list_ict_strategies() -> dict:
    from tradingview_mcp.core.services.ict_strategies import STRATEGY_REGISTRY
    files = list_strategy_files()
    out = []
    for f in files:
        parsed = parse_strategy_file(f)
        if "error" in parsed:
            out.append({"file": f, "error": parsed["error"]})
            continue
        out.append({
            "name": parsed["name"],
            "filename": os.path.splitext(os.path.basename(f))[0],
            "has_ict_logic": parsed["name"] in STRATEGY_REGISTRY,
            "entry_options": parsed["entry_options"],
            "sl_options": parsed["sl_options"],
            "tp_options": parsed["tp_options"],
        })
    return {"strategies_dir": STRATEGIES_DIR, "strategies": out}


def _run_zone_strategy(
    candles: list[dict],
    entry_mode: str,
    sl_mode: str,
    tp_mode: str,
    max_zone_taps: int,
) -> list[dict]:
    """Core simulation shared by backtest_custom_strategy and the optimizer."""
    swings = _find_swings(candles)
    atr    = _calc_atr_series(candles)
    zone_taps = {id(z): 0 for z in swings}

    trades: list[dict] = []
    position: Optional[dict] = None

    for i in range(_SWING_LOOKBACK + 1, len(candles)):
        candle, prev_candle = candles[i], candles[i - 1]

        # ── manage open position ──
        if position is not None:
            direction = position["direction"]
            exit_price = None

            if direction == "long":
                if candle["low"] <= position["sl"]:
                    exit_price = position["sl"]
                elif position.get("tp") and candle["high"] >= position["tp"]:
                    exit_price = position["tp"]
                elif tp_mode == "trail_only" and candle["low"] > position["entry_price"]:
                    position["sl"] = max(position["sl"], candle["low"])
            else:
                if candle["high"] >= position["sl"]:
                    exit_price = position["sl"]
                elif position.get("tp") and candle["low"] <= position["tp"]:
                    exit_price = position["tp"]
                elif tp_mode == "trail_only" and candle["high"] < position["entry_price"]:
                    position["sl"] = min(position["sl"], candle["high"])

            if exit_price is not None:
                trades.append({
                    "entry_date": position["entry_date"], "entry_price": position["entry_price"],
                    "exit_date": candle["date"], "exit_price": exit_price,
                })
                position = None
            continue

        # ── look for a new entry ──
        active_zones = [z for z in swings if z["index"] < i and zone_taps[id(z)] < max_zone_taps]
        for zone in active_zones:
            direction = _confirms_entry(entry_mode, candle, prev_candle, zone)
            if not direction:
                continue

            entry_price = candle["close"]
            zone_taps[id(zone)] += 1

            if sl_mode == "swing_based":
                sl = zone["bottom"] if direction == "long" else zone["top"]
            elif sl_mode == "opposing_candle":
                sl = candle["low"] if direction == "long" else candle["high"]
            else:  # fixed_atr
                a = atr[i] or (candle["high"] - candle["low"])
                sl = entry_price - 1.5 * a if direction == "long" else entry_price + 1.5 * a

            risk = abs(entry_price - sl)
            if risk <= 0:
                continue

            if tp_mode == "partial_1_5R":
                tp = entry_price + 1.5 * risk if direction == "long" else entry_price - 1.5 * risk
            elif tp_mode == "fixed_2R":
                tp = entry_price + 2.0 * risk if direction == "long" else entry_price - 2.0 * risk
            else:  # trail_only
                tp = None

            position = {
                "direction": direction, "entry_date": candle["date"],
                "entry_price": entry_price, "sl": sl, "tp": tp,
            }
            break

    return trades


# ─── Public API: backtest_custom_strategy ─────────────────────────────────────

def run_custom_backtest(
    strategy_name: str,
    symbol: Optional[str] = None,
    entry: Optional[str] = None,
    sl: Optional[str] = None,
    tp: Optional[str] = None,
    period: str = "1y",
    interval: str = "1d",
    initial_capital: float = 10_000.0,
    commission_pct: float = 0.1,
    slippage_pct: float = 0.05,
) -> dict:
    path = os.path.join(STRATEGIES_DIR, f"{strategy_name}.txt")
    parsed = parse_strategy_file(path)
    if "error" in parsed:
        return parsed

    entry = entry or parsed["entry_options"][0]
    sl    = sl or parsed["sl_options"][0]
    tp    = tp or parsed["tp_options"][0]

    if entry not in parsed["entry_options"]:
        return {"error": f"'{entry}' not declared in {path}. Options: {parsed['entry_options']}"}
    if sl not in parsed["sl_options"]:
        return {"error": f"'{sl}' not declared in {path}. Options: {parsed['sl_options']}"}
    if tp not in parsed["tp_options"]:
        return {"error": f"'{tp}' not declared in {path}. Options: {parsed['tp_options']}"}

    symbol = symbol or parsed["symbol_default"]
    if not symbol:
        return {"error": "No symbol given and no SYMBOL_DEFAULT in strategy file."}

    period, interval = period.lower().strip(), interval.lower().strip()
    if period not in _VALID_PERIODS:
        return {"error": f"Invalid period '{period}'. Choose: {sorted(_VALID_PERIODS)}"}
    if interval not in _VALID_INTERVALS:
        return {"error": f"Invalid interval '{interval}'. Choose: {sorted(_VALID_INTERVALS)}"}

    num_err = _validate_numeric_inputs(initial_capital, commission_pct, slippage_pct)
    if num_err:
        return {"error": num_err}

    provider, requires_daily = _resolve_ict_provider(parsed)
    used_ict_logic = provider is not None
    htf_interval = None

    try:
        if used_ict_logic:
            zones, candles, htf_candles, interval, htf_interval = _get_ict_zones(
                parsed, provider, requires_daily, symbol, period, interval, None,
            )
        else:
            candles = _fetch_csv_ohlcv(symbol, period, interval)
    except Exception as e:
        return {"error": f"Failed to fetch/derive data for '{symbol}': {e}"}

    min_bars = 30 if interval == "1d" else 50
    if len(candles) < min_bars:
        return {"error": f"Not enough data ({len(candles)} bars). Try a longer period."}

    if used_ict_logic:
        raw_trades = _run_zone_strategy_with_zones(zones, candles, entry, sl, tp, parsed["max_zone_taps"])
    else:
        raw_trades = _run_zone_strategy(candles, entry, sl, tp, parsed["max_zone_taps"])
    trades     = _apply_costs(raw_trades, commission_pct, slippage_pct)
    metrics    = _fix_annualization(_calc_metrics(trades, initial_capital, interval), interval)
    bnh        = _buy_and_hold_return(candles)

    return {
        "symbol": symbol.upper(),
        "strategy_name": parsed["name"],
        "interpreted_rules": {
            "entry_confirmation": entry,
            "stop_loss_mode": sl,
            "take_profit_mode": tp,
            "max_zone_taps": parsed["max_zone_taps"],
            "zone_source": "ict_registry" if used_ict_logic else "generic_swing_proxy",
            **({"zones_detected": len(zones), "htf_interval": htf_interval} if used_ict_logic else {}),
            "note": (
                "Hand-implemented per this file's own NAME-matched detection logic (ict_strategies.py)."
                if used_ict_logic else
                "Generic zone-reaction proxy — see module docstring for what this does NOT model."
            ),
        },
        "period": period, "interval": interval,
        "candles_analyzed": len(candles),
        "date_from": candles[0]["date"], "date_to": candles[-1]["date"],
        "initial_capital": round(initial_capital, 2),
        **metrics,
        "buy_and_hold_return_pct": bnh,
        "vs_buy_and_hold_pct": round(metrics["total_return_pct"] - bnh, 2),
        "recent_trades": trades[-5:],
        "data_source": "CSV (" + symbol.upper() + ")",
        "disclaimer": "Educational use only. Not a validated port of your MQL5/Pine strategy.",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ─── Public API: optimize_custom_strategy ─────────────────────────────────────

def optimize_custom_strategy(
    strategy_name: str,
    symbol: Optional[str] = None,
    period: str = "2y",
    interval: str = "1d",
    initial_capital: float = 10_000.0,
    commission_pct: float = 0.1,
    slippage_pct: float = 0.05,
    rank_by: str = "sharpe_ratio",
    n_splits: int = 3,
    train_ratio: float = 0.7,
) -> dict:
    """Sweep every declared entry/SL/TP combination and rank by `rank_by`.

    Walk-forward robustness is always computed (not optional) — a
    combination that only looks good on one full-period fit is exactly the
    overfitting failure mode this guards against.
    """
    path = os.path.join(STRATEGIES_DIR, f"{strategy_name}.txt")
    parsed = parse_strategy_file(path)
    if "error" in parsed:
        return parsed

    symbol = symbol or parsed["symbol_default"]
    if not symbol:
        return {"error": "No symbol given and no SYMBOL_DEFAULT in strategy file."}

    period, interval = period.lower().strip(), interval.lower().strip()
    if period not in _VALID_PERIODS:
        return {"error": f"Invalid period '{period}'. Choose: {sorted(_VALID_PERIODS)}"}
    if interval not in _VALID_INTERVALS:
        return {"error": f"Invalid interval '{interval}'. Choose: {sorted(_VALID_INTERVALS)}"}

    from tradingview_mcp.core.services.ict_strategies import DAILY_BIAS_GATED_STRATEGIES
    provider, requires_daily = _resolve_ict_provider(parsed)
    used_ict_logic = provider is not None
    bias_gated = used_ict_logic and parsed["name"] in DAILY_BIAS_GATED_STRATEGIES
    htf_candles = None
    daily_candles = None
    htf_interval = None

    try:
        if used_ict_logic:
            zones, candles, htf_candles, interval, htf_interval = _get_ict_zones(
                parsed, provider, requires_daily, symbol, period, interval, None,
            )
            if (requires_daily or bias_gated) and "1d" not in (interval, htf_interval):
                daily_candles = _fetch_csv_ohlcv(symbol, period, "1d")
        else:
            candles = _fetch_csv_ohlcv(symbol, period, interval)
    except Exception as e:
        return {"error": f"Failed to fetch/derive data for '{symbol}': {e}"}

    min_bars = 30 if interval == "1d" else 50
    if len(candles) < min_bars:
        return {"error": f"Not enough data ({len(candles)} bars). Try a longer period."}

    combos = list(itertools.product(parsed["entry_options"], parsed["sl_options"], parsed["tp_options"]))
    if len(combos) > 60:
        return {"error": f"{len(combos)} combinations is too many for one call — narrow the .txt options first."}

    n = len(candles)
    fold_size = n // n_splits
    results = []

    def _zone_trades(ltf_slice, htf_full, daily_full, e, s, t):
        """Run the correct engine for this strategy on a candle slice — the
        registered ICT provider (re-run so its zones are derived only from
        data available up to this slice, not leaked from the full period)
        or the generic swing proxy."""
        if used_ict_logic:
            htf_slice = _slice_htf_for_ltf_range(htf_full, ltf_slice) if htf_full is not None else ltf_slice
            daily_slice = _slice_htf_for_ltf_range(daily_full, ltf_slice) if daily_full is not None else None
            htf_arg = daily_slice if (requires_daily and daily_slice) else htf_slice
            p = {"broker_utc_offset_hours": 0.0}
            if bias_gated and daily_slice:
                p["daily_candles"] = daily_slice
            try:
                z = provider(ltf_slice, htf_arg, p)
            except Exception:
                z = []
            return _run_zone_strategy_with_zones(z, ltf_slice, e, s, t, parsed["max_zone_taps"])
        return _run_zone_strategy(ltf_slice, e, s, t, parsed["max_zone_taps"])

    for entry, sl, tp in combos:
        raw_full = _zone_trades(candles, htf_candles, daily_candles, entry, sl, tp)
        trades_full = _apply_costs(raw_full, commission_pct, slippage_pct)
        m_full = _fix_annualization(_calc_metrics(trades_full, initial_capital, interval), interval)

        # walk-forward: n_splits folds, each split train/test by train_ratio
        oos_returns = []
        for f in range(n_splits):
            start = f * fold_size
            end   = n if f == n_splits - 1 else (f + 1) * fold_size
            fold  = candles[start:end]
            if len(fold) < 20:
                continue
            split = int(len(fold) * train_ratio)
            test_fold = fold[split:]
            if len(test_fold) < 10:
                continue
            raw_oos = _zone_trades(test_fold, htf_candles, daily_candles, entry, sl, tp)
            trades_oos = _apply_costs(raw_oos, commission_pct, slippage_pct)
            m_oos = _fix_annualization(_calc_metrics(trades_oos, initial_capital, interval), interval)
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
            "entry_confirmation": entry, "stop_loss_mode": sl, "take_profit_mode": tp,
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
        "zone_source": "ict_registry" if used_ict_logic else "generic_swing_proxy",
        "period": period, "interval": interval,
        "htf_interval": htf_interval if used_ict_logic else None,
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
            "ROBUST/MODERATE picks generalized in walk-forward folds; "
            "OVERFITTED picks did not."
        ),
        "disclaimer": "Educational use only. Not a validated port of your MQL5/Pine strategy.",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def list_available_strategies() -> dict:
    files = list_strategy_files()
    out = []
    for f in files:
        parsed = parse_strategy_file(f)
        if "error" in parsed:
            out.append({"file": f, "error": parsed["error"]})
        else:
            out.append({
                "strategy_name": os.path.splitext(os.path.basename(f))[0],
                "name": parsed["name"],
                "symbol_default": parsed["symbol_default"],
                "timeframes": parsed["timeframes"],
                "entry_options": parsed["entry_options"],
                "sl_options": parsed["sl_options"],
                "tp_options": parsed["tp_options"],
            })
    return {"strategies_dir": STRATEGIES_DIR, "strategies": out}

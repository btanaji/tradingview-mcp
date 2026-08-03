"""
ict_detectors.py — shared detection primitives for the ICT-style strategy
rulebook (01-18, 20). Each function implements one specific, named rule from
the .txt files so strategy-specific code in ict_strategies.py can compose
them, rather than re-deriving the same logic 18 times.

IMPORTANT ARCHITECTURE NOTE:
The .txt files' definition sections (LIQUIDITY_POOLS, ORDER_BLOCKS,
FAIR_VALUE_GAPS, etc.) are NOT parsed at runtime — free-form English rules
("a candle whose high is higher than the 2 candles before and after it")
cannot be reliably turned into code by a generic parser. Instead, each rule
was read and hand-implemented here as a named function. The .txt file's
runtime job is limited to declaring which ENTRY_CONFIRMATION / STOP_LOSS /
TAKE_PROFIT options are enabled (parsed as before) — the definitional logic
lives in this module and in ict_strategies.py's per-NAME registry, and must
be updated by hand if a .txt file's definition changes.
"""
from __future__ import annotations
import bisect
import math
from datetime import datetime
from typing import Optional


# ─── Fractal swings (Liq_Sweep_Reversal's LIQUIDITY_POOLS definition) ─────────

def find_fractal_swings(candles: list[dict], side_bars: int = 2) -> list[dict]:
    """5-bar fractal: high(low) strictly greater(less) than `side_bars` candles
    on each side. Returns swings in index order.
    """
    swings = []
    n = len(candles)
    for i in range(side_bars, n - side_bars):
        window = candles[i - side_bars:i + side_bars + 1]
        hi, lo = candles[i]["high"], candles[i]["low"]
        if hi == max(c["high"] for c in window) and hi > max(
            c["high"] for c in window if c is not candles[i]
        ):
            swings.append({"index": i, "type": "high", "price": hi})
        if lo == min(c["low"] for c in window) and lo < min(
            c["low"] for c in window if c is not candles[i]
        ):
            swings.append({"index": i, "type": "low", "price": lo})
    return swings


# ─── Liquidity pools + sweeps (Liq_Sweep_Reversal) ────────────────────────────

def find_sweeps(candles: list[dict], swings: list[dict]) -> list[dict]:
    """A pool (unbroken swing high/low) is 'swept' when a later candle's wick
    trades through its level (touch, not necessarily close). Once swept, the
    pool is retired (not reused) — matches LIQUIDITY_POOLS' 'remove from
    active pool list' rule.
    """
    sweeps = []
    active_pools = {"high": None, "low": None}
    pool_swing_index = {"high": None, "low": None}

    swings_sorted = sorted(swings, key=lambda s: s["index"])
    swing_ptr = 0

    for i, c in enumerate(candles):
        while swing_ptr < len(swings_sorted) and swings_sorted[swing_ptr]["index"] <= i:
            s = swings_sorted[swing_ptr]
            active_pools[s["type"]] = s["price"]
            pool_swing_index[s["type"]] = s["index"]
            swing_ptr += 1

        if active_pools["high"] is not None and c["high"] > active_pools["high"]:
            sweeps.append({
                "index": i, "pool_type": "high", "pool_price": active_pools["high"],
                "pre_sweep_swing_index": pool_swing_index["high"],
            })
            active_pools["high"] = None
        if active_pools["low"] is not None and c["low"] < active_pools["low"]:
            sweeps.append({
                "index": i, "pool_type": "low", "pool_price": active_pools["low"],
                "pre_sweep_swing_index": pool_swing_index["low"],
            })
            active_pools["low"] = None

    return sweeps


def find_confirmation_after_sweep(
    candles: list[dict], sweep: dict, swings: list[dict], max_bars: int = 20,
) -> Optional[dict]:
    """CONFIRMATION_SEQUENCE: find the pre-sweep swing point of the OPPOSITE
    type that formed before the sweep candle, then require a close beyond it
    within `max_bars` candles of the sweep. Returns a zone-shaped dict usable
    by the shared tap/confirm engine, or None if no valid confirmation.
    """
    direction = "long" if sweep["pool_type"] == "low" else "short"
    ref_type = "low" if direction == "long" else "high"

    ref_candidates = [s for s in swings if s["type"] == ref_type and s["index"] < sweep["index"]]
    if not ref_candidates:
        return None
    ref = max(ref_candidates, key=lambda s: s["index"])

    for i in range(sweep["index"] + 1, min(sweep["index"] + 1 + max_bars, len(candles))):
        c = candles[i]
        if direction == "long" and c["close"] > ref["price"]:
            return {"type": "demand", "top": ref["price"], "bottom": ref["price"],
                    "index": sweep["index"], "confirmed_index": i,
                    "sweep_wick": min(candles[j]["low"] for j in range(sweep["index"], i + 1))}
        if direction == "short" and c["close"] < ref["price"]:
            return {"type": "supply", "top": ref["price"], "bottom": ref["price"],
                    "index": sweep["index"], "confirmed_index": i,
                    "sweep_wick": max(candles[j]["high"] for j in range(sweep["index"], i + 1))}
    return None


# ─── Fair Value Gaps (FVG_Rebalance) ──────────────────────────────────────────

def find_fvgs(candles: list[dict]) -> list[dict]:
    """3-candle gap: bullish if candle[i-2].high < candle[i].low; bearish if
    candle[i-2].low > candle[i].high. Tracks fill status: filled once a later
    candle CLOSES all the way through the gap's far edge.
    """
    gaps = []
    n = len(candles)
    for i in range(2, n):
        c1, c3 = candles[i - 2], candles[i]
        if c1["high"] < c3["low"]:
            gap = {"type": "bullish", "top": c3["low"], "bottom": c1["high"],
                   "origin_index": i, "filled_at": None}
        elif c1["low"] > c3["high"]:
            gap = {"type": "bearish", "top": c1["low"], "bottom": c3["high"],
                   "origin_index": i, "filled_at": None}
        else:
            continue
        for j in range(i + 1, n):
            if gap["type"] == "bullish" and candles[j]["close"] < gap["bottom"]:
                gap["filled_at"] = j
                break
            if gap["type"] == "bearish" and candles[j]["close"] > gap["top"]:
                gap["filled_at"] = j
                break
        gaps.append(gap)
    return gaps


def find_inverse_fvgs(candles: list[dict]) -> list[dict]:
    """IFVG: once an FVG is invalidated (a later candle CLOSES fully through
    its far edge — find_fvgs' existing 'filled_at' definition), it flips
    polarity: the same price band becomes a zone favoring the OPPOSITE
    direction (a bullish FVG that gets closed through becomes resistance/
    supply, and vice versa). Source: algostorm.com ICT/SMC entry models
    write-up ("Inverse Fair Value Gaps").
    """
    gaps = find_fvgs(candles)
    out = []
    for g in gaps:
        if g["filled_at"] is None:
            continue
        flipped_type = "supply" if g["type"] == "bullish" else "demand"
        out.append({"type": flipped_type, "top": g["top"], "bottom": g["bottom"], "index": g["filled_at"]})
    return out


def find_bpr_overlaps(candles: list[dict], min_overlap_pct: float = 0.30) -> list[dict]:
    """BPR_FVG_Overlap: overlap of a bullish and bearish FVG at the same
    price range, formed by two opposite fast moves in close succession.
    Overlap must be >= min_overlap_pct of the smaller gap's range.
    """
    gaps = find_fvgs(candles)
    bprs = []
    for i, g1 in enumerate(gaps):
        for g2 in gaps[i + 1:]:
            if g1["type"] == g2["type"]:
                continue
            if abs(g1["origin_index"] - g2["origin_index"]) > 10:
                continue
            top = min(g1["top"], g2["top"])
            bottom = max(g1["bottom"], g2["bottom"])
            if top <= bottom:
                continue
            smaller_range = min(g1["top"] - g1["bottom"], g2["top"] - g2["bottom"])
            if smaller_range <= 0:
                continue
            overlap_pct = (top - bottom) / smaller_range
            if overlap_pct >= min_overlap_pct:
                direction = "demand" if g1["type"] == "bullish" else "supply"
                bprs.append({
                    "type": direction, "top": top, "bottom": bottom,
                    "index": max(g1["origin_index"], g2["origin_index"]),
                })
    return bprs


# ─── Order Blocks + Breaker/Mitigation classification (OB_Continuation etc.) ──

def find_order_blocks(candles: list[dict], swings: list[dict], trend: str) -> list[dict]:
    """Candidate OB = last opposite-colored candle before a break of structure
    in `trend` direction, valid only if (1) a swing was swept in the 3 candles
    before it, and (2) the impulsive move leaves a 3-candle FVG in trend
    direction. Each returned OB also carries a 'status' of 'valid' initially;
    callers classify breaker/mitigation by checking later candles.
    """
    sweeps = find_sweeps(candles, swings)
    swept_indices = {s["index"] for s in sweeps}
    fvgs = find_fvgs(candles)
    fvg_by_origin = {g["origin_index"]: g for g in fvgs}

    obs = []
    swings_by_index = {s["index"]: s for s in swings}
    n = len(candles)

    for i in range(3, n - 3):
        bull_bos = trend == "bullish" and any(
            s["type"] == "high" and s["index"] < i and candles[i]["close"] > s["price"]
            for s in swings if s["index"] < i and i - s["index"] < 20
        )
        bear_bos = trend == "bearish" and any(
            s["type"] == "low" and s["index"] < i and candles[i]["close"] < s["price"]
            for s in swings if s["index"] < i and i - s["index"] < 20
        )
        if not (bull_bos or bear_bos):
            continue

        ob_candle_idx = i - 1
        ob_candle = candles[ob_candle_idx]
        is_bearish_candle = ob_candle["close"] < ob_candle["open"]
        is_bullish_candle = ob_candle["close"] > ob_candle["open"]
        if bull_bos and not is_bearish_candle:
            continue
        if bear_bos and not is_bullish_candle:
            continue

        swept_recently = any(ob_candle_idx - 3 <= s_idx <= ob_candle_idx for s_idx in swept_indices)
        if not swept_recently:
            continue

        has_fvg = any(abs(origin - i) <= 2 for origin in fvg_by_origin)
        if not has_fvg:
            continue

        obs.append({
            "type": "demand" if bull_bos else "supply",
            "top": ob_candle["high"], "bottom": ob_candle["low"],
            "index": ob_candle_idx, "status": "valid",
        })
    return obs


def classify_ob_retest(candles: list[dict], ob: dict, from_index: int) -> Optional[dict]:
    """Scan forward from `from_index` for the first retest of `ob`.
    Returns {'event': 'mitigation'} if it holds (no body close through the
    far extreme), {'event': 'breaker'} if a body closes through the far
    extreme (per BREAKER_DEFINITION/MITIGATION_DEFINITION).
    """
    direction = ob["type"]
    far_extreme = ob["bottom"] if direction == "demand" else ob["top"]
    for j in range(from_index, len(candles)):
        c = candles[j]
        touched = c["low"] <= ob["top"] and c["high"] >= ob["bottom"]
        if not touched:
            continue
        body_beyond = (c["close"] < far_extreme) if direction == "demand" else (c["close"] > far_extreme)
        if body_beyond:
            return {"event": "breaker", "index": j}
        return {"event": "mitigation", "index": j}
    return None


# ─── HTF trend (OB_Continuation's trend test, reused everywhere) ─────────────

def htf_trend(swings: list[dict]) -> str:
    """Bullish if the two most recent swings form higher-high/higher-low,
    bearish if lower-low/lower-high, else 'undefined'."""
    highs = sorted([s for s in swings if s["type"] == "high"], key=lambda s: s["index"])
    lows = sorted([s for s in swings if s["type"] == "low"], key=lambda s: s["index"])
    if len(highs) < 2 or len(lows) < 2:
        return "undefined"
    if highs[-1]["price"] > highs[-2]["price"] and lows[-1]["price"] > lows[-2]["price"]:
        return "bullish"
    if lows[-1]["price"] < lows[-2]["price"] and highs[-1]["price"] < highs[-2]["price"]:
        return "bearish"
    return "undefined"


# ─── Premium / discount split (Time_Price_Confluence, Daily_Bias_PDHPDL) ─────

def premium_discount_zone(top: float, bottom: float, price: float) -> str:
    mid = (top + bottom) / 2
    return "discount" if price < mid else "premium"


# ─── Session windows (Killzone_Intraday, Silver_Bullet, Session_Liquidity_ALN) ─

_SESSION_WINDOWS = {
    "london":       (2, 0, 5, 0),
    "new_york_fx":  (7, 0, 10, 0),
    "new_york_idx": (8, 30, 11, 0),
    "new_york_pm":  (13, 30, 16, 0),  # indices PM session (Killzones-TTrades)
    "asian":        (20, 0, 0, 0),   # wraps midnight
    "london_close": (10, 0, 12, 0),
}
_SILVER_BULLET_WINDOWS = [(3, 0, 4, 0), (10, 0, 11, 0), (14, 0, 15, 0)]


def _parse_est_time(date_str: str) -> Optional[tuple[int, int]]:
    """Extract (hour, minute) from a candle's date string.

    ASSUMPTION FLAGGED: this assumes the datetime string is already in EST.
    MT5 candle timestamps are in the BROKER's server time, which is usually
    NOT EST (commonly EET/UTC+2 or UTC+3) — you must supply the correct
    broker UTC offset via `broker_utc_offset_hours` in the strategy call, or
    session-window filters will silently check the wrong hours.
    """
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(date_str, fmt)
            return dt.hour, dt.minute
        except ValueError:
            continue
    return None


def in_session_window(date_str: str, window_name: str, broker_utc_offset_hours: float = 0.0) -> bool:
    parsed = _parse_est_time(date_str)
    if parsed is None:
        return True  # daily-only data has no intraday time; don't filter it out
    hour, minute = parsed
    est_hour = (hour - broker_utc_offset_hours - (-5)) % 24  # broker_UTC -> UTC -> EST(UTC-5)
    start_h, start_m, end_h, end_m = _SESSION_WINDOWS[window_name]
    t = est_hour + minute / 60
    start, end = start_h + start_m / 60, end_h + end_m / 60
    if window_name == "asian":
        return t >= start or t < end
    return start <= t < end


def in_any_silver_bullet_window(date_str: str, broker_utc_offset_hours: float = 0.0) -> bool:
    parsed = _parse_est_time(date_str)
    if parsed is None:
        return True
    hour, minute = parsed
    est_hour = (hour - broker_utc_offset_hours - (-5)) % 24
    t = est_hour + minute / 60
    return any(s <= t < e for s, _m, e, _m2 in [(s, m, e, m2) for (s, m, e, m2) in _SILVER_BULLET_WINDOWS])


# ─── IPDA rolling ranges (IPDA_Arrays) ────────────────────────────────────────

def ipda_ranges(daily_candles: list[dict], as_of_index: int) -> dict:
    """Trailing 20/40/60 trading-day high-low ranges, ending at as_of_index."""
    out = {}
    for n_days in (20, 40, 60):
        start = max(0, as_of_index - n_days + 1)
        window = daily_candles[start:as_of_index + 1]
        if len(window) < n_days:
            out[n_days] = None
            continue
        hi = max(c["high"] for c in window)
        lo = min(c["low"] for c in window)
        out[n_days] = {"high": hi, "low": lo, "mid": (hi + lo) / 2}
    return out


# ─── OTE Fibonacci zone (OTE_Fib_Entry) ───────────────────────────────────────

def ote_zone(swing_from: dict, swing_to: dict) -> dict:
    """62%-79% retracement zone of the swing_from -> swing_to leg (body-based
    anchors are the caller's responsibility — pass body high/low as prices).
    """
    lo, hi = sorted([swing_from["price"], swing_to["price"]])
    rng = hi - lo
    direction = "demand" if swing_to["price"] > swing_from["price"] else "supply"
    if direction == "demand":
        top = hi - 0.62 * rng
        bottom = hi - 0.79 * rng
    else:
        top = lo + 0.79 * rng
        bottom = lo + 0.62 * rng
    return {"type": direction, "top": top, "bottom": bottom}


# ─── Daily bias PDH/PDL classification (Daily_Bias_PDHPDL) ───────────────────

def classify_daily_bias(daily_candles: list[dict], as_of_index: int) -> str:
    if as_of_index < 2:
        return "undefined"
    a, b = daily_candles[as_of_index - 1], daily_candles[as_of_index]
    prev2 = daily_candles[as_of_index - 2]
    if a["high"] > prev2["high"] and a["low"] > prev2["low"]:
        return "bullish"
    if a["low"] < prev2["low"] and a["high"] < prev2["high"]:
        return "bearish"
    return "undefined"


def classify_pdhpdl_outcome(candle_a: dict, candle_b: dict) -> str:
    """Returns one of: 'no_sweep', 'case1', 'case2', 'continuation', 'outside_day'."""
    swept_high = candle_b["high"] > candle_a["high"]
    swept_low = candle_b["low"] < candle_a["low"]
    if swept_high and swept_low:
        return "outside_day"
    if not swept_high and not swept_low:
        return "no_sweep"
    if swept_high:
        if candle_b["close"] < candle_a["low"]:
            return "case2"
        if candle_b["close"] < candle_a["high"]:
            return "case1"
        return "continuation"
    if swept_low:
        if candle_b["close"] > candle_a["high"]:
            return "case2"
        if candle_b["close"] > candle_a["low"]:
            return "case1"
        return "continuation"


# ─── Displacement / MSS vs Liquidity Grab ─────────────────────────────────────
# Source: "MSS vs Liquidity Grab" (TTrades) — displacement is a full-bodied
# candle typically leaving an FVG; MSS = displacement THROUGH a swing point
# with a body CLOSE beyond it; Liquidity Grab = wick pierces the swing point
# but the body fails to close beyond it (no displacement). The PDFs never
# give a numeric displacement threshold, so `min_body_atr` (candle body >=
# this many ATRs) is an explicit implementation judgment call, not sourced.

def calc_atr(candles: list[dict], period: int = 14) -> list[Optional[float]]:
    """Simple ATR (average true range), same-length list aligned to candles;
    first `period` entries are None (insufficient warmup)."""
    trs: list[float] = []
    atr: list[Optional[float]] = []
    for i, c in enumerate(candles):
        if i == 0:
            tr = c["high"] - c["low"]
        else:
            prev_close = candles[i - 1]["close"]
            tr = max(c["high"] - c["low"], abs(c["high"] - prev_close), abs(c["low"] - prev_close))
        trs.append(tr)
        if i < period:
            atr.append(None)
        elif i == period:
            atr.append(sum(trs[1:period + 1]) / period)
        else:
            atr.append((atr[-1] * (period - 1) + tr) / period)
    return atr


def is_displacement(candles: list[dict], index: int, atr: list[Optional[float]], min_body_atr: float = 1.0) -> bool:
    """A large-body candle: body size >= min_body_atr * ATR at this index.
    Judgment call — no numeric threshold given in source material."""
    a = atr[index] if index < len(atr) else None
    if not a:
        return False
    c = candles[index]
    return abs(c["close"] - c["open"]) >= min_body_atr * a


def classify_mss_or_grab(
    candles: list[dict], swing: dict, break_index: int, atr: list[Optional[float]],
) -> str:
    """At `break_index`, classify the interaction with `swing` (a fractal
    swing dict from find_fractal_swings) as one of:
      'mss'            — body closes beyond the swing AND the candle displaces
      'liquidity_grab' — wick pierces the swing but body fails to close beyond it
      'none'           — swing not touched at this index
    """
    c = candles[break_index]
    level = swing["price"]
    if swing["type"] == "high":
        wicked = c["high"] > level
        body_beyond = c["close"] > level
    else:
        wicked = c["low"] < level
        body_beyond = c["close"] < level
    if not wicked:
        return "none"
    if body_beyond and is_displacement(candles, break_index, atr):
        return "mss"
    if not body_beyond:
        return "liquidity_grab"
    return "none"  # body closed beyond but candle isn't a displacement candle — ambiguous, not classified


# ─── Weekly range profile classifier ──────────────────────────────────────────
# Sources: "The weekly profile guide" vol.1-3 (amtrades), ICT_Weeklt_Range_
# Profiles.pdf (LumiTraders, 12 named sub-profiles). The 3 top-level families
# implemented here (classic_expansion / consolidation_reversal / midweek_
# reversal) are the amtrades framework; the 12 LumiTraders sub-types are
# refinements of these (see module docstring note below) and are NOT
# separately coded — day "quality ratings" and the exact accumulation-vs-
# retracement distinction are explicitly non-formulaic in the source PDFs,
# so the day-index heuristics below (which weekday sets the week's high/low,
# and how much of the range Mon-Wed consolidates) are an implementation
# judgment call, not a literal transcription.

def classify_weekly_profile(week_daily_candles: list[dict]) -> dict:
    """Classify a Mon-Fri (or partial) sequence of DAILY candles into one of
    the weekly range profile families. Expects `week_daily_candles` to be
    exactly that week's daily candles in order (index 0 = Monday).
    """
    if len(week_daily_candles) < 4:
        return {"profile": "insufficient_data"}

    highs = [c["high"] for c in week_daily_candles]
    lows = [c["low"] for c in week_daily_candles]
    hi_day = highs.index(max(highs))
    lo_day = lows.index(min(lows))
    week_open = week_daily_candles[0]["open"]
    week_close = week_daily_candles[-1]["close"]
    bullish = week_close > week_open

    full_range = max(highs) - min(lows)
    mon_wed = week_daily_candles[:3]
    mon_wed_range = max(c["high"] for c in mon_wed) - min(c["low"] for c in mon_wed)
    consolidation_ratio = (mon_wed_range / full_range) if full_range else 1.0

    extreme_day = hi_day if not bullish else lo_day  # the day that set the DIRECTIONAL extreme (low for bullish week, high for bearish week)

    if extreme_day <= 1:
        # low (bullish) or high (bearish) of the week formed Mon/Tue -> classic expansion,
        # rest of week (Wed-Thu) expands toward the opposite extreme.
        profile = "classic_expansion"
    elif consolidation_ratio < 0.5 and extreme_day == 3:
        # Mon-Wed consolidated tightly, Thu manipulated/reversed out of that range.
        profile = "consolidation_reversal"
    elif extreme_day == 2:
        # extreme formed on Wednesday.
        profile = "midweek_reversal"
    else:
        profile = "unclassified"

    return {
        "profile": profile,
        "bullish": bullish,
        "week_high_day": hi_day,
        "week_low_day": lo_day,
        "consolidation_ratio": round(consolidation_ratio, 3),
    }


# ─── Daily bias via failure-to-displace (TTrades Daily_Bias) ──────────────────
# Source: Daily_Bias-TTrades_edu.pdf — reversal framed off PDH/PDL specifically
# on a FAILURE TO DISPLACE beyond that level; the opposite level becomes the
# new draw on liquidity. Numeric displacement threshold not given in source
# (see is_displacement's judgment-call note); reused here.

def classify_daily_bias_v2(
    daily_candles: list[dict], as_of_index: int, atr: list[Optional[float]],
) -> str:
    """Returns one of: 'bullish_continuation', 'bearish_continuation',
    'bullish_reversal', 'bearish_reversal', 'undefined'.

    Continuation = today's candle displaces (closes) beyond yesterday's
    high/low with a displacement-sized body. Reversal = today wicks beyond
    yesterday's high/low but fails to close beyond it (failure to displace) —
    the level held, so bias flips toward the opposite side (PDL becomes the
    new draw on liquidity after a failed PDH break, and vice versa).
    """
    if as_of_index < 1:
        return "undefined"
    prev, cur = daily_candles[as_of_index - 1], daily_candles[as_of_index]

    displaced_up = cur["close"] > prev["high"] and is_displacement(daily_candles, as_of_index, atr)
    displaced_down = cur["close"] < prev["low"] and is_displacement(daily_candles, as_of_index, atr)
    if displaced_up:
        return "bullish_continuation"
    if displaced_down:
        return "bearish_continuation"

    failed_up = cur["high"] > prev["high"] and cur["close"] <= prev["high"]
    failed_down = cur["low"] < prev["low"] and cur["close"] >= prev["low"]
    if failed_up:
        return "bearish_reversal"
    if failed_down:
        return "bullish_reversal"
    return "undefined"


def daily_bias_asof(daily_candles: list[dict], date_str: str, atr: Optional[list[Optional[float]]] = None) -> str:
    """classify_daily_bias_v2's result as of the most recent daily candle at
    or before `date_str` — lets an LTF/HTF event (given its own date string)
    look up "what was the daily bias when this happened" for gating other
    strategies (e.g. only take OB_Continuation/Liq_Sweep_Reversal zones that
    agree with the prevailing daily bias).
    """
    if not daily_candles:
        return "undefined"
    if atr is None:
        atr = calc_atr(daily_candles)
    dates = [c["date"] for c in daily_candles]
    i = bisect.bisect_right(dates, date_str) - 1
    if i < 1:
        return "undefined"
    return classify_daily_bias_v2(daily_candles, i, atr)


# ─── HTF POI qualification via stop-raid confirmation (Son's Model) ──────────
# Source: Sons_Model_HTF-TTrades_edu.pdf — an HTF draw-on-liquidity zone only
# becomes a tradeable POI once a middle timeframe confirms a stop raid at/
# through it (top-down: identify DOL on HTF -> confirm stop raid on mid TF ->
# time entry on LTF). "Stop raid" here reuses the liquidity-grab definition
# from classify_mss_or_grab (wick through, body fails to close beyond).

def find_htf_poi_with_stop_raid(
    htf_candles: list[dict], mid_candles: list[dict], mid_dates: Optional[list[str]] = None,
) -> list[dict]:
    """HTF swings (potential DOL) confirmed by a mid-timeframe stop raid.
    Returns zone dicts with 'index' mapped into mid_candles (so callers can
    feed these into the shared tap/confirm engine against mid or LTF data
    sharing the same date axis as mid_candles).
    """
    htf_swings = find_fractal_swings(htf_candles)
    mid_atr = calc_atr(mid_candles)
    pois = []
    for s in htf_swings:
        level = s["price"]
        for j, c in enumerate(mid_candles):
            if s["type"] == "high" and c["high"] > level and c["close"] < level:
                pois.append({
                    "type": "supply", "top": c["high"], "bottom": level, "index": j,
                    "htf_swing_index": s["index"],
                })
                break
            if s["type"] == "low" and c["low"] < level and c["close"] > level:
                pois.append({
                    "type": "demand", "top": level, "bottom": c["low"], "index": j,
                    "htf_swing_index": s["index"],
                })
                break
    return pois


# ─── Internal liquidity / Inducement (IDM) ────────────────────────────────────
# Source: algostorm.com ICT/SMC entry models write-up, Model 2 ("FVG +
# Internal Liquidity / Inducement"). Distinct from find_sweeps' EXTERNAL
# liquidity pools: these are the minor pullback swing points formed WITHIN a
# displacement leg itself. The model requires one of these internal points to
# be swept (the "inducement") before an FVG retracement entry is considered
# valid — the idea being that a raid of minor internal stops precedes the
# "real" move into the FVG. No numeric minimum swing size is specified in
# source material; side_bars=1 (a 3-bar fractal) is used since displacement
# legs are often too short for a standard 5-bar fractal to find any internal
# pivot at all — a documented judgment call, not sourced.

def find_internal_liquidity_sweep(
    candles: list[dict], leg_start: int, leg_end: int, direction: str,
) -> list[dict]:
    """Minor fractal swings formed inside [leg_start, leg_end] (a displacement
    leg). direction='bullish' -> internal LOWS are returned (the retracement
    pivots expected to get swept on the pullback before continuation up);
    direction='bearish' -> internal HIGHS.
    """
    if leg_end - leg_start < 4:
        return []
    leg = candles[leg_start:leg_end + 1]
    swings = find_fractal_swings(leg, side_bars=1)
    want_type = "low" if direction == "bullish" else "high"
    return [
        {"type": s["type"], "price": s["price"], "index": leg_start + s["index"]}
        for s in swings if s["type"] == want_type
    ]


def confirm_idm_sweep(
    candles: list[dict], idm_levels: list[dict], from_index: int, max_bars: int = 20,
) -> Optional[dict]:
    """First candle at/after from_index whose wick sweeps any of idm_levels.
    Returns {'index', 'level', 'type'} for the first hit, or None."""
    for j in range(from_index, min(from_index + max_bars, len(candles))):
        c = candles[j]
        for lvl in idm_levels:
            if lvl["type"] == "low" and c["low"] < lvl["price"]:
                return {"index": j, "level": lvl["price"], "type": "low"}
            if lvl["type"] == "high" and c["high"] > lvl["price"]:
                return {"index": j, "level": lvl["price"], "type": "high"}
    return None

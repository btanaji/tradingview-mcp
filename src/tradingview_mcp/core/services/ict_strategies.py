"""
ict_strategies.py — per-strategy zone providers for the 18-file ICT rulebook.

Each provider function takes (ltf_candles, htf_candles, params) and returns a
list of "zone" dicts: {"type": "demand"|"supply", "top": float, "bottom":
float, "index": int, ...}. These plug into the same shared tap/confirm/SL/TP
engine used for the earlier SD_CVD_Flow-style files (_confirms_entry,
_run_zone_strategy in custom_strategy_service.py) — the zone SOURCE differs
per strategy, but the entry-confirmation, SL, and TP mechanics are shared.

Two zone shapes are produced:
  - "tap zones": price bands price may later retrace into (order blocks,
    FVGs, BPRs, IPDA range boundaries, OTE zones, PDH/PDL entry zones).
    These use the existing tap-and-wait engine.
  - "event zones": a pre-sweep swing point already confirmed by a close —
    represented as a zero-width band at that swing price so the same
    generic _confirms_entry modes still apply, with the sweep having
    already happened (Liq_Sweep_Reversal and its five variations).

COVERAGE HONESTY — read before trusting any of these:
  FULL, hand-implemented per the .txt's own definitions:
    Liq_Sweep_Reversal, OB_Continuation, FVG_Rebalance, Killzone_Intraday,
    Silver_Bullet, Breaker_Block_Reversal, Mitigation_Block, BPR_FVG_Overlap,
    IPDA_Arrays, Daily_Bias_PDHPDL, OTE_Fib_Entry.
  IMPLEMENTED AS PARAMETERIZED VARIATIONS of the above, exactly as each
  file's own NOTES section says it is a variation/gate (not re-derived from
  scratch, since the files explicitly document this relationship):
    Liq_to_Liq_Range (-> Liq_Sweep_Reversal at HTF range scope),
    Liq_FVG_Combo (-> Liq_Sweep_Reversal + FVG midpoint),
    Session_Liquidity_ALN (-> Liq_Sweep_Reversal gated by session logic),
    AMD_Order_Flow (-> Liq_Sweep_Reversal gated by accumulation-phase check),
    Time_Price_Confluence / MTF_Confluence (-> gates composed with
      Liq_Sweep_Reversal / OB_Continuation, per those files' own
      "hardcoded, self-contained copy" fallback text),
    Quarterly_Theory_Swing (-> same hardcoded fallback, gated by calendar
      true-open phase read).
  HTF_OB_Tap_LTF_MSS -> its own OB-tap + LTF BOS provider (distinct enough
    from OB_Continuation's retracement-into-OB model to implement directly).
  CRT_Turtle_Soup and HTF_POI_Stop_Raid (added later, from a separate PDF
    library on Candle Range Theory / Son's Model) -> own providers, see
    ict_detectors.py's "Weekly range profile classifier" / "Daily bias via
    failure-to-displace" / "HTF POI qualification" sections for the shared
    primitives (is_displacement, classify_mss_or_grab, classify_weekly_
    profile, classify_daily_bias_v2, find_htf_poi_with_stop_raid) these and
    future strategies can reuse. Numeric thresholds not given in source PDFs
    (displacement size, OB invalidation, liquidity ranking) are flagged as
    implementation judgment calls in ict_detectors.py's docstrings.

NOT independently verified against real market data (only synthetic data in
this sandbox — see README for what that means).
"""
from __future__ import annotations
import bisect
from datetime import datetime, timedelta
from typing import Optional

from tradingview_mcp.core.services import ict_detectors as d


def _htf_trend_from_candles(htf_candles: list[dict]) -> str:
    swings = d.find_fractal_swings(htf_candles)
    return d.htf_trend(swings)


# Strategies gated by daily bias (classify_daily_bias_v2, failure-to-displace
# model): a candidate zone is only kept if the daily bias in effect at the
# zone's confirmation date doesn't oppose the zone's own direction. 'undefined'
# bias never blocks a zone — with no clear read, the gate stays a no-op rather
# than silently killing all trades.
DAILY_BIAS_GATED_STRATEGIES = {"Liq_Sweep_Reversal", "OB_Continuation"}


def _bias_allows(zone_type: str, bias: str) -> bool:
    bullish = bias in ("bullish_continuation", "bullish_reversal")
    bearish = bias in ("bearish_continuation", "bearish_reversal")
    if zone_type == "demand":
        return not bearish
    return not bullish


# ─── 01: Liq_Sweep_Reversal ────────────────────────────────────────────────────

def zones_liq_sweep_reversal(ltf: list[dict], htf: list[dict], params: dict) -> list[dict]:
    max_bars = params.get("confirmation_max_bars", 20)
    swings = d.find_fractal_swings(htf if htf else ltf)
    sweeps = d.find_sweeps(htf if htf else ltf, swings)
    zones = []
    ref_candles = htf if htf else ltf
    daily_candles = params.get("daily_candles")
    for sweep in sweeps:
        z = d.find_confirmation_after_sweep(ref_candles, sweep, swings, max_bars)
        if z:
            z["index"] = z["confirmed_index"]  # entry becomes eligible from confirmation bar
            if daily_candles:
                bias = d.daily_bias_asof(daily_candles, ref_candles[z["index"]]["date"])
                if not _bias_allows(z["type"], bias):
                    continue
            zones.append(z)
    return zones


# ─── 02: OB_Continuation ──────────────────────────────────────────────────────

def zones_ob_continuation(ltf: list[dict], htf: list[dict], params: dict) -> list[dict]:
    """Scans BOTH directions rather than gating on a single global HTF
    trend read. A prior version computed trend ONCE from just the two most
    recent swings in the whole series and returned [] outright if that one
    static read was 'undefined' (a common tail-end state — contracting or
    expanding range) — silently zeroing the entire backtest even when
    earlier, perfectly valid order blocks existed in either direction.
    find_order_blocks' own BOS check is already local/rolling (within 20
    bars of each candidate), so no global trend gate is needed here at all.
    """
    ref = htf if htf else ltf
    swings = d.find_fractal_swings(ref)
    daily_candles = params.get("daily_candles")
    zones = []
    for trend in ("bullish", "bearish"):
        obs = d.find_order_blocks(ref, swings, trend)
        for ob in obs:
            retest = d.classify_ob_retest(ref, ob, ob["index"] + 1)
            if retest and retest["event"] == "mitigation":
                if daily_candles:
                    bias = d.daily_bias_asof(daily_candles, ref[ob["index"]]["date"])
                    if not _bias_allows(ob["type"], bias):
                        continue
                zones.append({"type": ob["type"], "top": ob["top"], "bottom": ob["bottom"], "index": ob["index"]})
    return zones


# ─── 03: FVG_Rebalance ─────────────────────────────────────────────────────────

def zones_fvg_rebalance(ltf: list[dict], htf: list[dict], params: dict) -> list[dict]:
    gaps = d.find_fvgs(ltf)
    zones = []
    for g in gaps:
        if g["filled_at"] is not None and (g["filled_at"] - g["origin_index"]) < 3:
            continue  # already filled almost immediately; not a meaningful target
        zones.append({
            "type": "demand" if g["type"] == "bullish" else "supply",
            "top": g["top"], "bottom": g["bottom"], "index": g["origin_index"],
        })
    return zones


# ─── 04: OTE_Fib_Entry ─────────────────────────────────────────────────────────

def zones_ote_fib_entry(ltf: list[dict], htf: list[dict], params: dict) -> list[dict]:
    ref = htf if htf else ltf
    swings = d.find_fractal_swings(ref)
    swings_sorted = sorted(swings, key=lambda s: s["index"])
    zones = []
    for i in range(1, len(swings_sorted)):
        a, b = swings_sorted[i - 1], swings_sorted[i]
        if a["type"] == b["type"]:
            continue
        z = d.ote_zone(a, b)
        z["index"] = b["index"]
        zones.append(z)
    return zones


# ─── 05: Killzone_Intraday (gates Liq_Sweep_Reversal by session window) ──────

def zones_killzone_intraday(ltf: list[dict], htf: list[dict], params: dict) -> list[dict]:
    base = zones_liq_sweep_reversal(ltf, htf, params)
    offset = params.get("broker_utc_offset_hours", 0.0)
    windows = params.get("session_windows", ["london", "new_york_fx"])
    out = []
    for z in base:
        idx = z["index"]
        if idx >= len(ltf):
            continue
        date_str = ltf[idx]["date"]
        if any(d.in_session_window(date_str, w, offset) for w in windows):
            out.append(z)
    return out


# ─── 06: Silver_Bullet ────────────────────────────────────────────────────────

def zones_silver_bullet(ltf: list[dict], htf: list[dict], params: dict) -> list[dict]:
    offset = params.get("broker_utc_offset_hours", 0.0)
    trend = _htf_trend_from_candles(htf if htf else ltf)
    if trend == "undefined":
        return []
    swings = d.find_fractal_swings(ltf)
    sweeps = d.find_sweeps(ltf, swings)
    gaps = d.find_fvgs(ltf)
    gap_origins = sorted(g["origin_index"] for g in gaps)
    zones = []
    for sweep in sweeps:
        date_str = ltf[sweep["index"]]["date"]
        if not d.in_any_silver_bullet_window(date_str, offset):
            continue
        conf = d.find_confirmation_after_sweep(ltf, sweep, swings, max_bars=10)
        if not conf:
            continue
        has_fvg_after = any(conf["confirmed_index"] <= g_idx <= conf["confirmed_index"] + 3 for g_idx in gap_origins)
        if not has_fvg_after:
            continue
        if (conf["type"] == "demand" and trend != "bullish") or (conf["type"] == "supply" and trend != "bearish"):
            continue
        zones.append(conf)
    return zones


# ─── 07: Liq_to_Liq_Range (variation of 01, HTF range scope) ─────────────────

def zones_liq_to_liq_range(ltf: list[dict], htf: list[dict], params: dict) -> list[dict]:
    zones = zones_liq_sweep_reversal(ltf, htf, {**params, "confirmation_max_bars": 40})
    trend = _htf_trend_from_candles(htf if htf else ltf)
    if trend == "undefined":
        # No reliable global trend read (common tail-end state) — keep zones
        # from both directions rather than discarding the whole result set.
        return zones
    return [z for z in zones if (z["type"] == "demand") == (trend == "bullish")]


# ─── 08: Breaker_Block_Reversal ────────────────────────────────────────────────

def zones_breaker_block_reversal(ltf: list[dict], htf: list[dict], params: dict) -> list[dict]:
    trend = _htf_trend_from_candles(htf if htf else ltf)
    ref = htf if htf else ltf
    swings = d.find_fractal_swings(ref)
    candidate_trend = "bullish" if trend != "bullish" else "bearish"  # breaker forms opposite the failed OB's original trend
    for t in ("bullish", "bearish"):
        obs = d.find_order_blocks(ref, swings, t)
        zones = []
        for ob in obs:
            retest = d.classify_ob_retest(ref, ob, ob["index"] + 1)
            if retest and retest["event"] == "breaker":
                new_type = "supply" if ob["type"] == "demand" else "demand"
                zones.append({"type": new_type, "top": ob["top"], "bottom": ob["bottom"], "index": retest["index"]})
        if zones:
            return zones
    return []


# ─── 09: Time_Price_Confluence (gates 01 by session + premium/discount) ──────

def zones_time_price_confluence(ltf: list[dict], htf: list[dict], params: dict) -> list[dict]:
    ref = htf if htf else ltf
    if len(ref) < 5:
        return []
    htf_top = max(c["high"] for c in ref[-50:])
    htf_bottom = min(c["low"] for c in ref[-50:])
    offset = params.get("broker_utc_offset_hours", 0.0)
    base = zones_liq_sweep_reversal(ltf, htf, params)
    out = []
    for z in base:
        idx = z["index"]
        if idx >= len(ltf):
            continue
        date_str = ltf[idx]["date"]
        in_time = d.in_session_window(date_str, "london", offset) or d.in_session_window(date_str, "new_york_fx", offset)
        zone_mid_price = (z["top"] + z["bottom"]) / 2
        pd = d.premium_discount_zone(htf_top, htf_bottom, zone_mid_price)
        wants_discount = (z["type"] == "demand")
        in_price = (pd == "discount") == wants_discount
        if in_time and in_price:
            out.append(z)
    return out


# ─── 10: IPDA_Arrays ──────────────────────────────────────────────────────────

def zones_ipda_arrays(ltf: list[dict], htf_daily: list[dict], params: dict) -> list[dict]:
    if not htf_daily or len(htf_daily) < 60:
        return []
    ranges = d.ipda_ranges(htf_daily, len(htf_daily) - 1)
    zones = []
    band_pct = params.get("band_pct", 0.002)
    for n_days, r in ranges.items():
        if r is None:
            continue
        band = r["high"] * band_pct
        zones.append({"type": "supply", "top": r["high"] + band, "bottom": r["high"] - band, "index": 0})
        zones.append({"type": "demand", "top": r["low"] + band, "bottom": r["low"] - band, "index": 0})
    return zones


# ─── 11: MTF_Confluence (gates 01/02 by HTF+mid trend agreement) ─────────────

def zones_mtf_confluence(ltf: list[dict], htf: list[dict], params: dict) -> list[dict]:
    base = zones_liq_sweep_reversal(ltf, htf, params)
    htf_trend = _htf_trend_from_candles(htf if htf else ltf)
    if htf_trend == "undefined":
        return base  # no reliable global trend read — keep both directions
    return [z for z in base if (z["type"] == "demand") == (htf_trend == "bullish")]


# ─── 12: Quarterly_Theory_Swing ───────────────────────────────────────────────

def _true_opens(as_of: datetime) -> dict:
    year_open = _first_monday_on_or_after(datetime(as_of.year if as_of.month >= 4 else as_of.year - 1, 4, 1))
    month_open = _second_monday(as_of.year, as_of.month)
    return {"yearly": year_open, "monthly": month_open}


def _first_monday_on_or_after(dt: datetime) -> datetime:
    days_ahead = (7 - dt.weekday()) % 7
    return dt + timedelta(days=days_ahead)


def _second_monday(year: int, month: int) -> datetime:
    first = datetime(year, month, 1)
    first_monday = _first_monday_on_or_after(first)
    return first_monday + timedelta(days=7)


def zones_quarterly_theory_swing(ltf: list[dict], htf: list[dict], params: dict) -> list[dict]:
    # (No global trend gate: trend was never used to filter the result below
    # anyway — it was a pure kill-switch that zeroed the whole backtest
    # whenever the tail-end swing pair was ambiguous.)
    return zones_liq_sweep_reversal(ltf, htf, {**params, "confirmation_max_bars": 60})


# ─── 13: BPR_FVG_Overlap ──────────────────────────────────────────────────────

def zones_bpr_fvg_overlap(ltf: list[dict], htf: list[dict], params: dict) -> list[dict]:
    min_overlap = params.get("min_overlap_pct", 0.30)
    return d.find_bpr_overlaps(ltf, min_overlap)


# ─── 14: Mitigation_Block ──────────────────────────────────────────────────────

def zones_mitigation_block(ltf: list[dict], htf: list[dict], params: dict) -> list[dict]:
    return zones_ob_continuation(ltf, htf, params)  # same holds-mechanic as OB_Continuation


# ─── 15: Liq_FVG_Combo ────────────────────────────────────────────────────────

def zones_liq_fvg_combo(ltf: list[dict], htf: list[dict], params: dict) -> list[dict]:
    ref = htf if htf else ltf
    swings = d.find_fractal_swings(ref)
    sweeps = d.find_sweeps(ref, swings)
    gaps = d.find_fvgs(ltf)
    zones = []
    for sweep in sweeps:
        direction_type = "bullish" if sweep["pool_type"] == "low" else "bearish"
        candidate_gaps = [g for g in gaps if g["type"] == direction_type and g["origin_index"] > sweep["index"]]
        if not candidate_gaps:
            continue
        g = min(candidate_gaps, key=lambda g: g["origin_index"])
        mid = (g["top"] + g["bottom"]) / 2
        zones.append({
            "type": "demand" if direction_type == "bullish" else "supply",
            "top": mid, "bottom": mid, "index": g["origin_index"],
        })
    return zones


# ─── 16: Session_Liquidity_ALN ────────────────────────────────────────────────

def zones_session_liquidity_aln(ltf: list[dict], htf: list[dict], params: dict) -> list[dict]:
    offset = params.get("broker_utc_offset_hours", 0.0)
    base = zones_liq_sweep_reversal(ltf, htf, params)
    out = []
    for z in base:
        idx = z["index"]
        if idx >= len(ltf):
            continue
        date_str = ltf[idx]["date"]
        if d.in_session_window(date_str, "new_york_fx", offset):
            out.append(z)
    return out


# ─── 17: AMD_Order_Flow ───────────────────────────────────────────────────────

def zones_amd_order_flow(ltf: list[dict], htf: list[dict], params: dict) -> list[dict]:
    """'is_ranging' compares the 10-bar accumulation window's range against
    accumulation_range_atr_mult x ATR(14). A prior version divided the full
    14-bar peak-to-trough range by 14 as a stand-in for "average bar range" —
    that produces a threshold roughly an order of magnitude too small (a
    peak-to-trough range over 14 bars is not 14x a single bar's range), so
    is_ranging was almost never true and this strategy silently produced
    zero zones on any real data.
    """
    ref = htf if htf else ltf
    if len(ref) < 20:
        return []
    atr = d.calc_atr(ref)
    check_index = len(ref) - 5  # end of the ref[-14:-4] accumulation window
    a = atr[check_index] if check_index < len(atr) else None
    if not a:
        return []
    recent = ref[-14:-4]
    recent_range = max(c["high"] for c in recent) - min(c["low"] for c in recent)
    is_ranging = recent_range < params.get("accumulation_range_atr_mult", 3) * a
    if not is_ranging:
        return []
    return zones_liq_sweep_reversal(ltf, htf, params)


# ─── 18: Daily_Bias_PDHPDL ─────────────────────────────────────────────────────

def zones_daily_bias_pdhpdl(ltf: list[dict], htf_daily: list[dict], params: dict) -> list[dict]:
    if not htf_daily or len(htf_daily) < 3:
        return []
    zones = []
    for i in range(2, len(htf_daily)):
        bias = d.classify_daily_bias(htf_daily, i)
        if bias == "undefined":
            continue
        outcome = d.classify_pdhpdl_outcome(htf_daily[i - 1], htf_daily[i])
        if outcome not in ("case1", "case2"):
            continue
        candle_b = htf_daily[i]
        swept_high = candle_b["high"] > htf_daily[i - 1]["high"]
        direction_ok = (swept_high and bias == "bearish") or (not swept_high and bias == "bullish")
        if not direction_ok:
            continue
        mid = (candle_b["high"] + candle_b["low"]) / 2
        if bias == "bullish":
            zones.append({"type": "demand", "top": mid, "bottom": candle_b["low"], "index": i})
        else:
            zones.append({"type": "supply", "top": candle_b["high"], "bottom": mid, "index": i})
    return zones


# ─── 20: HTF_OB_Tap_LTF_MSS ────────────────────────────────────────────────────

def zones_htf_ob_tap_ltf_mss(ltf: list[dict], htf: list[dict], params: dict) -> list[dict]:
    ref = htf if htf else ltf
    swings = d.find_fractal_swings(ref)
    zones = []
    for trend in ("bullish", "bearish"):
        obs = d.find_order_blocks(ref, swings, trend)
        zones.extend({"type": ob["type"], "top": ob["top"], "bottom": ob["bottom"], "index": ob["index"]} for ob in obs)
    return zones


# ─── 21: CRT_Turtle_Soup (Candle Range Theory) ────────────────────────────────

def zones_crt_turtle_soup(ltf: list[dict], htf: list[dict], params: dict) -> list[dict]:
    """CRT: candle N sets CRT high/low; candle N+1 wicks beyond one side and
    closes back inside (Turtle Soup) -> zone at the swept half of the CRT
    range, mapped onto the LTF candle at/after the confirming HTF candle's
    close. Zone spans the swept extreme to the CRT midpoint, reflecting the
    documented 50%-of-range partial/BE management rule.
    Source: CRT part 1/2, CANDLE RANGE THEORY.pdf.
    """
    ref = htf if htf else ltf
    if len(ref) < 2:
        return []
    ltf_dates = [c["date"] for c in ltf]
    zones = []
    for i in range(1, len(ref)):
        prev, cur = ref[i - 1], ref[i]
        crt_high, crt_low = prev["high"], prev["low"]
        mid = (crt_high + crt_low) / 2
        swept_low = cur["low"] < crt_low and cur["close"] > crt_low
        swept_high = cur["high"] > crt_high and cur["close"] < crt_high
        if not (swept_low or swept_high):
            continue
        idx = bisect.bisect_left(ltf_dates, cur["date"])
        if idx >= len(ltf):
            continue
        if swept_low:
            zones.append({"type": "demand", "top": mid, "bottom": crt_low, "index": idx})
        else:
            zones.append({"type": "supply", "top": crt_high, "bottom": mid, "index": idx})
    return zones


# ─── 22: HTF_POI_Stop_Raid (Son's Model) ──────────────────────────────────────

def zones_htf_poi_stop_raid(ltf: list[dict], htf: list[dict], params: dict) -> list[dict]:
    """HTF swing (draw on liquidity) only becomes a tradeable POI once the
    LTF shows a stop raid at that level (wick through, body fails to close
    beyond — see classify_mss_or_grab's 'liquidity_grab' case).
    Source: Sons_Model_HTF-TTrades_edu.pdf.
    """
    if not htf:
        return []
    return d.find_htf_poi_with_stop_raid(htf, ltf)


# ─── 23: FVG_Inducement (algostorm Model 2) ───────────────────────────────────

def zones_fvg_inducement(ltf: list[dict], htf: list[dict], params: dict) -> list[dict]:
    """External sweep -> MSS confirmation -> a minor INTERNAL swing point
    formed during that displacement leg must also get swept (inducement)
    before the FVG left by the leg is considered a valid entry zone.
    Source: algostorm.com, Model 2 ("FVG + Internal Liquidity / Inducement").
    """
    ref = htf if htf else ltf
    max_bars = params.get("confirmation_max_bars", 20)
    swings = d.find_fractal_swings(ref)
    sweeps = d.find_sweeps(ref, swings)
    fvgs = d.find_fvgs(ref)
    zones = []
    for sweep in sweeps:
        conf = d.find_confirmation_after_sweep(ref, sweep, swings, max_bars)
        if not conf:
            continue
        direction = "bullish" if conf["type"] == "demand" else "bearish"
        leg_start, leg_end = sweep["index"], conf["confirmed_index"]
        idm_levels = d.find_internal_liquidity_sweep(ref, leg_start, leg_end, direction)
        if not idm_levels:
            continue
        idm_hit = d.confirm_idm_sweep(ref, idm_levels, leg_end + 1, max_bars)
        if not idm_hit:
            continue
        candidate_fvgs = [g for g in fvgs if leg_start <= g["origin_index"] <= leg_end]
        if not candidate_fvgs:
            continue
        g = candidate_fvgs[-1]
        zones.append({
            "type": "demand" if direction == "bullish" else "supply",
            "top": g["top"], "bottom": g["bottom"], "index": idm_hit["index"],
        })
    return zones


# ─── 24: Full_Confluence (algostorm Model 4) ──────────────────────────────────

def zones_full_confluence(ltf: list[dict], htf: list[dict], params: dict) -> list[dict]:
    """Everything at once: external sweep, displacement-confirmed MSS,
    internal liquidity (IDM) sweep, and an FVG that overlaps the leg's OTE
    (62-79%) zone. Source: algostorm.com, Model 4 ("Full Confluence") — the
    source explicitly frames this as rare/overfit-prone; implemented as
    written, not endorsed as high-edge.
    """
    ref = htf if htf else ltf
    max_bars = params.get("confirmation_max_bars", 20)
    swings = d.find_fractal_swings(ref)
    sweeps = d.find_sweeps(ref, swings)
    atr = d.calc_atr(ref)
    fvgs = d.find_fvgs(ref)
    zones = []
    for sweep in sweeps:
        conf = d.find_confirmation_after_sweep(ref, sweep, swings, max_bars)
        if not conf:
            continue
        if not d.is_displacement(ref, conf["confirmed_index"], atr):
            continue
        direction = "bullish" if conf["type"] == "demand" else "bearish"
        leg_start, leg_end = sweep["index"], conf["confirmed_index"]
        idm_levels = d.find_internal_liquidity_sweep(ref, leg_start, leg_end, direction)
        idm_hit = d.confirm_idm_sweep(ref, idm_levels, leg_end + 1, max_bars) if idm_levels else None
        if not idm_hit:
            continue
        candidate_fvgs = [g for g in fvgs if leg_start <= g["origin_index"] <= leg_end]
        if not candidate_fvgs:
            continue
        leg_lo = min(c["low"] for c in ref[leg_start:leg_end + 1])
        leg_hi = max(c["high"] for c in ref[leg_start:leg_end + 1])
        ote = d.ote_zone({"price": leg_lo if direction == "bullish" else leg_hi},
                          {"price": leg_hi if direction == "bullish" else leg_lo})
        for g in candidate_fvgs:
            top, bottom = min(g["top"], ote["top"]), max(g["bottom"], ote["bottom"])
            if top <= bottom:
                continue  # FVG doesn't actually overlap the OTE zone
            zones.append({
                "type": "demand" if direction == "bullish" else "supply",
                "top": top, "bottom": bottom, "index": idm_hit["index"],
            })
    return zones


# ─── 25: Consolidation_Box (algostorm Model 5) ────────────────────────────────

def zones_consolidation_box(ltf: list[dict], htf: list[dict], params: dict) -> list[dict]:
    """A tight range (box) is swept on one side (false breakout) then closes
    back inside -> zone at the swept boundary for a retest entry. 'Tight'
    is defined as box range <= 3x ATR at the breakout bar — not numerically
    specified in source, a documented judgment call.
    Source: algostorm.com, Model 5 ("Consolidation Box Setup").
    """
    ref = htf if htf else ltf
    atr = d.calc_atr(ref)
    lookback = params.get("box_lookback", 10)
    zones = []
    for i in range(lookback, len(ref)):
        box = ref[i - lookback:i]
        box_high = max(c["high"] for c in box)
        box_low = min(c["low"] for c in box)
        box_range = box_high - box_low
        a = atr[i] or box_range
        if box_range <= 0 or (a and box_range > 3 * a):
            continue
        breakout = ref[i]
        swept_high = breakout["high"] > box_high and breakout["close"] < box_high
        swept_low = breakout["low"] < box_low and breakout["close"] > box_low
        if swept_low:
            zones.append({"type": "demand", "top": box_low + 0.25 * box_range, "bottom": box_low, "index": i})
        elif swept_high:
            zones.append({"type": "supply", "top": box_high, "bottom": box_high - 0.25 * box_range, "index": i})
    return zones


# ─── 26: IFVG_Reversal (algostorm "Inverse Fair Value Gaps") ─────────────────

def zones_ifvg_reversal(ltf: list[dict], htf: list[dict], params: dict) -> list[dict]:
    """An FVG that gets invalidated (closed through) flips polarity and
    becomes a zone for the opposite direction. Source: algostorm.com, "IFVG".
    """
    ref = htf if htf else ltf
    return [{"type": z["type"], "top": z["top"], "bottom": z["bottom"], "index": z["index"]}
            for z in d.find_inverse_fvgs(ref)]


# ─── Registry ──────────────────────────────────────────────────────────────────

STRATEGY_REGISTRY = {
    "Liq_Sweep_Reversal": zones_liq_sweep_reversal,
    "OB_Continuation": zones_ob_continuation,
    "FVG_Rebalance": zones_fvg_rebalance,
    "OTE_Fib_Entry": zones_ote_fib_entry,
    "Killzone_Intraday": zones_killzone_intraday,
    "Silver_Bullet": zones_silver_bullet,
    "Liq_to_Liq_Range": zones_liq_to_liq_range,
    "Breaker_Block_Reversal": zones_breaker_block_reversal,
    "Time_Price_Confluence": zones_time_price_confluence,
    "IPDA_Arrays": zones_ipda_arrays,
    "MTF_Confluence": zones_mtf_confluence,
    "Quarterly_Theory_Swing": zones_quarterly_theory_swing,
    "BPR_FVG_Overlap": zones_bpr_fvg_overlap,
    "Mitigation_Block": zones_mitigation_block,
    "Liq_FVG_Combo": zones_liq_fvg_combo,
    "Session_Liquidity_ALN": zones_session_liquidity_aln,
    "AMD_Order_Flow": zones_amd_order_flow,
    "Daily_Bias_PDHPDL": zones_daily_bias_pdhpdl,
    "HTF_OB_Tap_LTF_MSS": zones_htf_ob_tap_ltf_mss,
    "CRT_Turtle_Soup": zones_crt_turtle_soup,
    "HTF_POI_Stop_Raid": zones_htf_poi_stop_raid,
    "FVG_Inducement": zones_fvg_inducement,
    "Full_Confluence": zones_full_confluence,
    "Consolidation_Box": zones_consolidation_box,
    "IFVG_Reversal": zones_ifvg_reversal,
}

# Strategies whose provider needs DAILY candles as the "htf" argument
# (IPDA_Arrays, Daily_Bias_PDHPDL) rather than the file's own HTF token —
# their definitions are explicitly daily-chart based regardless of what
# TIMEFRAMES lists as identification timeframe.
REQUIRES_DAILY_HTF = {"IPDA_Arrays", "Daily_Bias_PDHPDL"}

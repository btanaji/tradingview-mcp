"""
Live signal evaluator — the "LIVE AGENT SYSTEM" layer from CLAUDE.md's
architecture plan: MT5-exported CSVs -> per-symbol scan -> live signal
evaluator -> risk engine -> (optional) paper execution.

Design principle: reuse the SAME zone-provider / ICT-detector / pine_lite
code paths as the three backtest engines rather than duplicating detection
logic. A "live signal" here is detected by running the normal
run_ict_backtest / run_custom_backtest / run_pine_backtest over a short
trailing window and checking whether the most recent trade in its
recent_trades log opened exactly on the latest available bar.

Caveat (kept honest rather than papered over): none of the three engines'
trade dicts expose the stop-loss/take-profit price actually used at entry
(_apply_costs only keeps entry/exit price+date), so this module cannot
recover a strategy's real internal SL. It instead computes a fallback
ATR(14)-based stop and a naive 5-bar direction bias — clearly labeled as
such — good enough to size a position, not a substitute for the strategy's
own rule. Also: core/portfolio.py has no short-selling support, so only
long ("BUY") signals can be auto-executed; short signals are reported only.

No live network/broker connection exists anywhere in this codebase — like
the backtest engines, this reads whatever is currently on disk under
TV_MCP_CSV_DATA_DIR. "Live" means "as fresh as the CSV export," not a
streaming feed.
"""
from __future__ import annotations

from typing import Any, Literal, Optional

from tradingview_mcp.core.services.custom_strategy_service import (
    run_ict_backtest,
    run_custom_backtest,
    _fetch_csv_ohlcv,
)
from tradingview_mcp.core.services.pine_strategy_service import run_pine_backtest
from tradingview_mcp.core.services.indicators_calc import calc_atr
from tradingview_mcp.core.errors import ErrorCode, make_error

_ENGINES = {
    "ict": run_ict_backtest,
    "custom": run_custom_backtest,
    "pine": run_pine_backtest,
}

_SUMMARY_FIELDS = ("total_trades", "win_rate_pct", "total_return_pct", "sharpe_ratio")


def evaluate_live_signal(
    engine: Literal["ict", "custom", "pine"],
    strategy_name: str,
    symbol: Optional[str] = None,
    period: str = "1mo",
    atr_period: int = 14,
    atr_stop_multiplier: float = 1.5,
    **engine_kwargs: Any,
) -> dict[str, Any]:
    """Check whether `strategy_name` has a fresh entry signal on the most
    recent bar of local CSV data.

    Args:
        engine: 'ict' (ict_strategies registry), 'custom' (generic
            swing-pivot proxy), or 'pine' (pine_lite).
        strategy_name: the strategy file's NAME field or filename.
        symbol: overrides the file's SYMBOL_DEFAULT if given.
        period: how much trailing history to run the underlying backtest
            over — short is fine ('1mo'/'3mo'); only the last bar matters.
        atr_period / atr_stop_multiplier: fallback stop-loss sizing when a
            fresh signal is found (see module docstring — this is NOT the
            strategy's own internal SL, which isn't exposed by any engine's
            trade log).
        **engine_kwargs: forwarded verbatim to the matching run_*_backtest
            function (e.g. interval= for custom/pine, ltf_interval=/
            htf_interval=/broker_utc_offset_hours= for ict).
    """
    if engine not in _ENGINES:
        return make_error(ErrorCode.INVALID_PARAMETER, f"engine must be one of {sorted(_ENGINES)}")

    try:
        result = _ENGINES[engine](strategy_name, symbol, period=period, **engine_kwargs)
    except TypeError as e:
        return make_error(ErrorCode.INVALID_PARAMETER, f"bad kwargs for engine={engine!r}: {e}")

    if "error" in result:
        return result

    resolved_symbol = result["symbol"]
    date_to = result["date_to"]
    recent_trades = result.get("recent_trades", [])
    last_trade = recent_trades[-1] if recent_trades else None
    fresh_signal = bool(last_trade) and last_trade["entry_date"] == date_to

    signal: dict[str, Any] = {
        "fresh_signal": fresh_signal,
        "engine": engine,
        "strategy_name": result["strategy_name"],
        "symbol": resolved_symbol,
        "as_of": date_to,
        "entry_price": None,
        "suggested_direction": None,
        "suggested_stop_price": None,
        "backtest_summary": {k: result[k] for k in _SUMMARY_FIELDS if k in result},
    }

    if not fresh_signal:
        return signal

    entry_price = last_trade["entry_price"]
    signal["entry_price"] = entry_price
    signal["stop_note"] = (
        f"ATR({atr_period})x{atr_stop_multiplier} fallback stop + a naive short-lookback "
        "direction bias — NOT the strategy's actual internal SL rule (not exposed by any "
        "engine's trade log). Verify before sizing/executing."
    )

    interval = (
        result.get("interpreted_rules", {}).get("ltf_interval")
        or result.get("interval")
        or "1h"
    )
    try:
        candles = _fetch_csv_ohlcv(resolved_symbol, "1mo", interval)
    except Exception:
        candles = []

    if len(candles) > atr_period + 5:
        highs = [c["high"] for c in candles]
        lows = [c["low"] for c in candles]
        closes = [c["close"] for c in candles]
        atr_series = calc_atr(highs, lows, closes, atr_period)
        atr_val = next((a for a in reversed(atr_series) if a is not None), None)
        lookback = min(5, len(closes) - 1)
        direction = "long" if closes[-1] >= closes[-1 - lookback] else "short"
        signal["suggested_direction"] = direction
        if atr_val:
            stop = entry_price - atr_val * atr_stop_multiplier if direction == "long" \
                else entry_price + atr_val * atr_stop_multiplier
            signal["suggested_stop_price"] = round(stop, 6)

    return signal


def scan_watchlist(
    watchlist: list[dict[str, Any]],
    user_id: Optional[str] = None,
    auto_paper_trade: bool = False,
    risk_pct: float = 1.0,
    max_single_symbol_pct: Optional[float] = None,
    max_total_exposure_pct: Optional[float] = None,
    max_daily_loss_pct: Optional[float] = None,
    max_consecutive_losses: Optional[int] = None,
    send_telegram_alerts: bool = False,
) -> dict[str, Any]:
    """Scan a list of strategy/symbol pairs for fresh signals in one call —
    intended to be invoked on a schedule (cron/APScheduler) outside the MCP
    request/response loop, per the "keep the LLM out of the execution hot
    path" design principle: this function is deterministic, an external
    scheduler decides when to call it, not an LLM per-tick.

    Args:
        watchlist: list of dicts, each at minimum
            {"engine": "ict"|"custom"|"pine", "strategy_name": "..."},
            plus optional "symbol"/"period"/engine-specific kwargs — see
            evaluate_live_signal.
        user_id: paper-trading user to execute against; required if
            auto_paper_trade is True.
        auto_paper_trade: if True, automatically BUYs (long only — see
            module docstring) every fresh long signal via
            core.portfolio.execute_trade, sized by risk_pct against the
            suggested ATR stop, subject to the exposure/circuit-breaker
            limits below. Short signals are always reported, never traded.
        risk_pct: % of paper-portfolio equity to risk per auto-trade.
        max_single_symbol_pct / max_total_exposure_pct / max_daily_loss_pct /
            max_consecutive_losses: forwarded to execute_trade's risk checks
            (None = skip that check).
        send_telegram_alerts: if True, sends one Telegram message per fresh
            signal via alerting_service (requires TV_MCP_TELEGRAM_BOT_TOKEN /
            TV_MCP_TELEGRAM_CHAT_ID env vars; a missing/failed send is
            reported per-signal in the result, never raised).
    """
    if auto_paper_trade and not user_id:
        return make_error(ErrorCode.INVALID_PARAMETER, "user_id is required when auto_paper_trade is True")

    signals = []
    for entry in watchlist:
        entry = dict(entry)
        engine = entry.pop("engine", None)
        strategy_name = entry.pop("strategy_name", None)
        if not engine or not strategy_name:
            signals.append(make_error(
                ErrorCode.INVALID_PARAMETER,
                "watchlist entry missing 'engine' or 'strategy_name'", entry=entry,
            ))
            continue
        signals.append(evaluate_live_signal(engine, strategy_name, **entry))

    fresh = [s for s in signals if s.get("fresh_signal")]
    executed = []
    skipped_short = []

    if auto_paper_trade:
        from tradingview_mcp.core.portfolio import execute_trade
        for sig in fresh:
            if sig.get("suggested_direction") != "long" or not sig.get("suggested_stop_price"):
                if sig.get("suggested_direction") == "short":
                    skipped_short.append(sig["symbol"])
                continue
            trade = execute_trade(
                user_id, sig["symbol"], 0.0, sig["entry_price"], "BUY",
                risk_pct=risk_pct, stop_price=sig["suggested_stop_price"],
                max_single_symbol_pct=max_single_symbol_pct,
                max_total_exposure_pct=max_total_exposure_pct,
                max_daily_loss_pct=max_daily_loss_pct,
                max_consecutive_losses=max_consecutive_losses,
            )
            executed.append({"symbol": sig["symbol"], "strategy_name": sig["strategy_name"], "result": trade})

    alert_results = []
    if send_telegram_alerts and fresh:
        from tradingview_mcp.core.services.alerting_service import alert_fresh_signals
        alert_results = alert_fresh_signals(fresh)

    return {
        "scanned": len(watchlist),
        "fresh_signals": fresh,
        "all_signals": signals,
        "auto_paper_trade": auto_paper_trade,
        "executed_trades": executed,
        "short_signals_skipped": skipped_short,
        "telegram_alerts_sent": alert_results,
    }

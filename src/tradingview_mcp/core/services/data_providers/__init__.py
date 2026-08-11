"""
Unified data-provider abstraction layer — the OpenBB-pattern piece of the
architecture in CLAUDE.md, built with the free-substitution table instead of
OpenBB's paid providers:

    Polygon/Intrinio (equities/forex)  -> local MT5-exported CSV + Yahoo chart API
    FMP/Intrinio (fundamentals)        -> SEC EDGAR (edgar_provider.py)
    Macro/economic vendors             -> FRED (fred_provider.py)
    Benzinga News                      -> RSS + Marketaux (news_service.py / marketaux_service.py, pre-existing)
    Real-time WS feeds / crypto        -> tradingview-screener (screener_service.py, pre-existing)

get_ohlcv() is the one genuinely new piece here: a single call that routes to
whichever of the two existing (separately-built) OHLCV fetchers applies,
rather than callers having to know that FX/metals come from local CSV while
equities/crypto come from Yahoo's chart API. Nothing about either underlying
fetcher changes — this only adds a dispatch layer in front of them.
"""
from __future__ import annotations

from typing import Any, Literal, Optional

from tradingview_mcp.core.errors import ErrorCode, make_error
from tradingview_mcp.core.services.data_providers.fred_provider import get_fred_series
from tradingview_mcp.core.services.data_providers.edgar_provider import get_company_fundamentals

__all__ = ["get_ohlcv", "get_fred_series", "get_company_fundamentals"]

_SOURCES = ("csv", "yahoo")


def get_ohlcv(
    symbol: str,
    source: Literal["csv", "yahoo"],
    period: str = "3mo",
    interval: str = "1d",
) -> dict[str, Any]:
    """Fetch OHLCV candles from whichever backing provider `source` names.

    Args:
        symbol: instrument symbol. For source='csv' this must match a local
            file under TV_MCP_CSV_DATA_DIR (typically FX/metals exported
            from MT5, e.g. 'XAUUSD'). For source='yahoo' this is a Yahoo
            Finance ticker (equities/crypto/indices, e.g. 'AAPL', 'BTC-USD').
        source: 'csv' (local MT5-exported files) or 'yahoo' (Yahoo Finance
            chart API — no key required, direct HTTP with a proxy fallback).
        period: trailing window, e.g. '1mo'/'3mo'/'6mo'/'1y'/'2y' (csv also
            accepts '3y'/'4y'/'5y'/'all').
        interval: candle interval, e.g. '1d'/'1h' (csv also accepts
            '1m'/'5m'/'15m'/'30m'/'4h').
    """
    if source not in _SOURCES:
        return make_error(ErrorCode.INVALID_PARAMETER, f"source must be one of {_SOURCES}")

    try:
        if source == "csv":
            from tradingview_mcp.core.services.custom_strategy_service import _fetch_csv_ohlcv
            candles = _fetch_csv_ohlcv(symbol, period, interval)
        else:
            from tradingview_mcp.core.services.backtest_service import _fetch_ohlcv
            candles = _fetch_ohlcv(symbol, period, interval)
    except Exception as e:
        return make_error(ErrorCode.UPSTREAM_ERROR, f"OHLCV fetch failed (source={source}): {e}")

    if not candles:
        return make_error(ErrorCode.NO_DATA, f"no candles returned for {symbol!r} (source={source})")

    return {
        "symbol": symbol.upper(),
        "source": source,
        "period": period,
        "interval": interval,
        "n_candles": len(candles),
        "date_from": candles[0]["date"],
        "date_to": candles[-1]["date"],
        "candles": candles,
    }

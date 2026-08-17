"""
TradingView MCP Server — routing layer only.

Each @mcp.tool() handler is responsible for:
  1. Validating / sanitising parameters
  2. Delegating to the appropriate service module
  3. Returning the result

No business logic lives here. All computation is in core/services/*.
"""
from __future__ import annotations

import argparse
import asyncio
import os
from typing import Optional

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

# ── Service imports ────────────────────────────────────────────────────────────
from tradingview_mcp.core.services.coinlist import load_symbols
from tradingview_mcp.core.services.screener_service import (
    fetch_bollinger_analysis,
    fetch_trending_analysis,
    analyze_coin,
    scan_consecutive_candles,
    scan_advanced_candle_patterns_single_tf,
    fetch_multi_timeframe_patterns,
    run_multi_timeframe_analysis,
)
from tradingview_mcp.core.services.scanner_service import (
    volume_breakout_scan,
    volume_confirmation_analyze,
    smart_volume_scan,
)
from tradingview_mcp.core.services.multi_agent_service import run_multi_agent_analysis
from tradingview_mcp.core.services.egx_service import (
    get_egx_market_overview,
    scan_egx_sector,
    run_egx_sector_scanner,
    analyze_egx_index,
    screen_egx_stocks,
    generate_egx_trade_plan,
    analyze_egx_fibonacci,
)
from tradingview_mcp.core.services.marketaux_service import (
    analyze_sentiment,
    fetch_news_summary,
)
from tradingview_mcp.core.services.yahoo_finance_service import (
    get_price,
    get_price_async,
    get_market_snapshot,
)
from tradingview_mcp.core.services.bitcoin_market_service import get_bitcoin_market_pulse
from tradingview_mcp.core.services.extended_hours_service import (
    get_extended_hours_price,
    get_extended_hours_price_async,
)
from tradingview_mcp.core.services.options_service import (
    get_options_chain,
    get_unusual_options_activity,
)
from tradingview_mcp.core.services.futures_service import (
    get_futures_overview,
    get_futures_movers,
    get_futures_category_snapshot,
    get_futures_watchlist,
)
from tradingview_mcp.core.services.stock_screener_service import (
    EXAMPLE_MARKETS,
    fetch_stock_prices,
    screen_stocks,
)
from tradingview_mcp.core.services.backtest_service import (
    run_backtest,
    compare_strategies as _compare_strategies,
    walk_forward_backtest,
)
from tradingview_mcp.core.services.custom_strategy_service import (
    run_custom_backtest, optimize_custom_strategy as _optimize_custom_strategy,
    list_available_strategies,
)
from tradingview_mcp.core.services.custom_strategy_service import (
     list_ict_strategies as list_ict_strategies_impl, run_ict_backtest,
)
from tradingview_mcp.core.services.pine_strategy_service import (
    list_pine_strategies as list_pine_strategies_impl,
    validate_pine_strategy as _validate_pine_strategy,
    run_pine_backtest,
    optimize_pine_strategy as _optimize_pine_strategy,
)
from tradingview_mcp.core.services.risk_service import (
    calc_position_size,
    calc_var,
    check_exposure_limits,
    check_circuit_breaker,
    calc_option_greeks,
)
from tradingview_mcp.core.portfolio import (
    execute_trade as _execute_paper_trade,
    get_portfolio as _get_paper_portfolio,
    check_pretrade_risk as _check_pretrade_risk,
)
from tradingview_mcp.core.services.live_signal_service import (
    evaluate_live_signal as _evaluate_live_signal,
    scan_watchlist as _scan_watchlist,
)
from tradingview_mcp.core.services.alerting_service import (
    send_telegram_alert as _send_telegram_alert,
)
from tradingview_mcp.core.services.data_providers import (
    get_ohlcv as _get_ohlcv,
    get_fred_series as _get_fred_series,
    get_company_fundamentals as _get_company_fundamentals,
)
from tradingview_mcp.core.services.ml_factor_service import (
    run_alpha_factor_analysis as _run_alpha_factor_analysis,
)
from tradingview_mcp.core.services.rl_service import (
    train_rl_trading_agent as _train_rl_trading_agent,
)
from tradingview_mcp.core.utils.validators import (
    sanitize_timeframe,
    sanitize_exchange,
    normalize_tradingview_symbol,
    normalize_yahoo_symbol,
)
from tradingview_mcp.core.errors import (
    ErrorCode,
    exception_to_envelope,
    make_error,
)

try:
    import tradingview_screener  # noqa: F401
    TRADINGVIEW_SCREENER_AVAILABLE = True
except ImportError:
    TRADINGVIEW_SCREENER_AVAILABLE = False


# ── MCP server instance ────────────────────────────────────────────────────────

mcp = FastMCP(
    name="TradingView Multi-Market Screener",
    instructions=(
        "Multi-market screener backed by TradingView. "
        "Supports crypto exchanges (KuCoin, Binance, Bybit, MEXC, etc.), stock markets "
        "(EGX, BIST, NASDAQ, NYSE, Bursa Malaysia, HKEX, SSE, SZSE, TWSE, TPEX), "
        "and futures markets (CME, COMEX, NYMEX, CBOT — equity index, energy, metals, "
        "agriculture, rates, forex, crypto futures). "
        "Tools: top_gainers, top_losers, bollinger_scan, coin_analysis, multi_agent_analysis, "
        "volume_breakout_scanner, futures_market_overview, futures_top_movers, "
        "futures_category_snapshot, futures_watchlist, egx_market_overview, and more."
    ),
)


# ── Screener tools ─────────────────────────────────────────────────────────────

@mcp.tool(annotations=ToolAnnotations(title="Top Gainers Screener", readOnlyHint=True, destructiveHint=False, openWorldHint=True))
async def top_gainers(exchange: str = "KUCOIN", timeframe: str = "15m", limit: int = 25) -> list[dict] | dict:
    """Return top gainers for an exchange and timeframe using Bollinger Band analysis.

    Args:
        exchange: Exchange name — crypto: KUCOIN, BINANCE, BYBIT, MEXC; stocks: EGX, BIST, NASDAQ, NYSE, BURSA, HKEX, SSE, SZSE, TWSE, TPEX
        timeframe: One of 5m, 15m, 1h, 4h, 1D, 1W, 1M
        limit: Number of rows to return (max 50)

    Returns:
        list[dict] on success. On ANY failure returns a structured error
        envelope ``{"error": {"code": ..., "retryable": ...}}`` — never a
        raw exception string.
    """
    exchange = sanitize_exchange(exchange, "KUCOIN")
    timeframe = sanitize_timeframe(timeframe, "15m")
    limit = max(1, min(limit, 50))
    try:
        # Underlying tradingview-screener is sync (uses urllib). Push to a
        # worker thread so the event loop is free for other concurrent
        # tool calls.
        rows = await asyncio.to_thread(
            fetch_trending_analysis, exchange, timeframe=timeframe, limit=limit
        )
    except Exception as e:
        return exception_to_envelope(e, context="top_gainers")
    return [{"symbol": r["symbol"], "changePercent": r["changePercent"], "indicators": dict(r["indicators"])} for r in rows]


@mcp.tool(annotations=ToolAnnotations(title="Top Losers Screener", readOnlyHint=True, destructiveHint=False, openWorldHint=True))
def top_losers(exchange: str = "KUCOIN", timeframe: str = "15m", limit: int = 25) -> list[dict] | dict:
    """Return top losers for an exchange and timeframe. Supports crypto (KUCOIN, BINANCE, MEXC) and stocks (EGX, BIST, NASDAQ).

    Returns ``list[dict]`` on success. On ANY failure returns a structured
    error envelope ``{"error": {"code": ..., "retryable": ...}}``.
    """
    exchange = sanitize_exchange(exchange, "KUCOIN")
    timeframe = sanitize_timeframe(timeframe, "15m")
    limit = max(1, min(limit, 50))
    try:
        rows = fetch_trending_analysis(exchange, timeframe=timeframe, limit=limit)
    except Exception as e:
        return exception_to_envelope(e, context="top_losers")
    rows.sort(key=lambda x: x["changePercent"])
    return [{"symbol": r["symbol"], "changePercent": r["changePercent"], "indicators": dict(r["indicators"])} for r in rows[:limit]]


@mcp.tool(annotations=ToolAnnotations(title="Bollinger Squeeze Scanner", readOnlyHint=True, destructiveHint=False, openWorldHint=True))
def bollinger_scan(exchange: str = "KUCOIN", timeframe: str = "4h", bbw_threshold: float = 0.04, limit: int = 50) -> list[dict] | dict:
    """Scan for assets with low Bollinger Band Width (squeeze detection). Works with crypto and stocks.

    This scans a whole EXCHANGE for squeezes (canonical name is exactly
    `bollinger_scan`; there is no "get_bollinger_band_analysis" tool). For
    the Bollinger read of ONE symbol, call `coin_analysis` instead.

    Example: bollinger_scan(exchange="BINANCE", timeframe="15m", bbw_threshold=0.008)

    Args:
        exchange: Exchange — crypto: KUCOIN, BINANCE, BYBIT, MEXC; stocks: EGX, BIST, NASDAQ, NYSE, BURSA, HKEX, SSE, SZSE, TWSE, TPEX
        timeframe: One of 5m, 15m, 1h, 4h, 1D, 1W, 1M. Typical squeeze thresholds: 15m→0.008, 1h→0.02, 4h→0.04, 1D→0.12
        bbw_threshold: Maximum BBW value to filter (default 0.04)
        limit: Number of rows to return (max 100)

    Returns ``list[dict]`` on success. On ANY failure returns a structured
    error envelope ``{"error": {"code": ..., "retryable": ...}}``.
    """
    exchange = sanitize_exchange(exchange, "KUCOIN")
    timeframe = sanitize_timeframe(timeframe, "4h")
    limit = max(1, min(limit, 100))
    try:
        rows = fetch_bollinger_analysis(exchange, timeframe=timeframe, bbw_filter=bbw_threshold, limit=limit)
    except Exception as e:
        return exception_to_envelope(e, context="bollinger_scan")
    return [{"symbol": r["symbol"], "changePercent": r["changePercent"], "indicators": dict(r["indicators"])} for r in rows]


@mcp.tool(annotations=ToolAnnotations(title="Bollinger Rating Filter", readOnlyHint=True, destructiveHint=False, openWorldHint=True))
def rating_filter(exchange: str = "KUCOIN", timeframe: str = "5m", rating: int = 2, limit: int = 25) -> list[dict] | dict:
    """Filter coins by Bollinger Band rating.

    Args:
        exchange: Exchange name like KUCOIN, BINANCE, BYBIT, MEXC, etc.
        timeframe: One of 5m, 15m, 1h, 4h, 1D, 1W, 1M
        rating: BB rating (-3 to +3): -3=Strong Sell, -2=Sell, -1=Weak Sell, 1=Weak Buy, 2=Buy, 3=Strong Buy
        limit: Number of rows to return (max 50)

    Returns ``list[dict]`` on success. On ANY failure returns a structured
    error envelope ``{"error": {"code": ..., "retryable": ...}}``.
    """
    exchange = sanitize_exchange(exchange, "KUCOIN")
    timeframe = sanitize_timeframe(timeframe, "5m")
    rating = max(-3, min(3, rating))
    limit = max(1, min(limit, 50))
    try:
        rows = fetch_trending_analysis(exchange, timeframe=timeframe, filter_type="rating", rating_filter=rating, limit=limit)
    except Exception as e:
        return exception_to_envelope(e, context="rating_filter")
    return [{"symbol": r["symbol"], "changePercent": r["changePercent"], "indicators": dict(r["indicators"])} for r in rows]


# ── Coin / asset analysis ──────────────────────────────────────────────────────

@mcp.tool(annotations=ToolAnnotations(title="Full Technical Analysis", readOnlyHint=True, destructiveHint=False, openWorldHint=True))
def coin_analysis(symbol: str, exchange: str = "KUCOIN", timeframe: str = "15m") -> dict:
    """Get detailed analysis for a specific asset (coin or stock) on specified exchange and timeframe.

    This is the canonical single-symbol technical readout (there is no
    "get_technical_analysis" or "get_technical_summary" tool — use THIS one).
    Use `multi_timeframe_analysis` instead when you need trend alignment
    across several timeframes, and `combined_analysis` when you also want
    news sentiment + headlines in the same call.

    Example: coin_analysis(symbol="BTCUSDT", exchange="BINANCE", timeframe="1h")

    Args:
        symbol: Bare ticker, no exchange prefix — crypto: "BTCUSDT", "ETHUSDT"; stocks: "COMI" (EGX), "THYAO" (BIST), "600519" (SSE), "300251" (SZSE), "2330" (TWSE), "3105" (TPEX)
        exchange: Exchange — crypto: KUCOIN, BINANCE, MEXC; stocks: EGX, BIST, NASDAQ, NYSE, BURSA, HKEX, SSE, SZSE, TWSE, TPEX. If the symbol isn't listed there, the error's `listed_on` field names exchanges that do list it.
        timeframe: Time interval (5m, 15m, 1h, 4h, 1D, 1W, 1M)

    Returns:
        Detailed analysis with all indicators and metrics
    """
    exchange = sanitize_exchange(exchange, "KUCOIN")
    timeframe = sanitize_timeframe(timeframe, "15m")
    return analyze_coin(symbol, exchange, timeframe)


# ── Candle pattern tools ───────────────────────────────────────────────────────

@mcp.tool(annotations=ToolAnnotations(title="Consecutive Candles Scanner", readOnlyHint=True, destructiveHint=False, openWorldHint=True))
def consecutive_candles_scan(
    exchange: str = "KUCOIN",
    timeframe: str = "15m",
    pattern_type: str = "bullish",
    candle_count: int = 3,
    min_growth: float = 2.0,
    limit: int = 20,
) -> dict:
    """Scan for coins with consecutive growing/shrinking candles pattern.

    Args:
        exchange: Exchange name (BINANCE, KUCOIN, etc.)
        timeframe: Time interval (5m, 15m, 1h, 4h)
        pattern_type: "bullish" (growing candles) or "bearish" (shrinking candles)
        candle_count: Number of consecutive candles to check (2-5)
        min_growth: Minimum growth percentage for each candle
        limit: Maximum number of results to return
    """
    exchange = sanitize_exchange(exchange, "KUCOIN")
    timeframe = sanitize_timeframe(timeframe, "15m")
    candle_count = max(2, min(5, candle_count))
    min_growth = max(0.5, min(20.0, min_growth))
    limit = max(1, min(50, limit))
    return scan_consecutive_candles(exchange, timeframe, pattern_type, candle_count, min_growth, limit)


@mcp.tool(annotations=ToolAnnotations(title="Candlestick Pattern Analysis", readOnlyHint=True, destructiveHint=False, openWorldHint=True))
def advanced_candle_pattern(
    exchange: str = "KUCOIN",
    base_timeframe: str = "15m",
    pattern_length: int = 3,
    min_size_increase: float = 10.0,
    limit: int = 15,
) -> dict:
    """Advanced candle pattern analysis using multi-timeframe data.

    Args:
        exchange: Exchange name (BINANCE, KUCOIN, etc.)
        base_timeframe: Base timeframe for analysis (5m, 15m, 1h, 4h)
        pattern_length: Number of consecutive periods to analyse (2-4)
        min_size_increase: Minimum percentage increase in candle size
        limit: Maximum number of results to return
    """
    exchange = sanitize_exchange(exchange, "KUCOIN")
    base_timeframe = sanitize_timeframe(base_timeframe, "15m")
    pattern_length = max(2, min(4, pattern_length))
    min_size_increase = max(5.0, min(50.0, min_size_increase))
    limit = max(1, min(30, limit))

    symbols = load_symbols(exchange)
    if not symbols:
        return {"error": f"No symbols found for exchange: {exchange}", "exchange": exchange}
    symbols = symbols[: min(limit * 2, 100)]

    if TRADINGVIEW_SCREENER_AVAILABLE:
        try:
            results = fetch_multi_timeframe_patterns(exchange, symbols, base_timeframe, pattern_length, min_size_increase)
            return {
                "exchange": exchange,
                "base_timeframe": base_timeframe,
                "pattern_length": pattern_length,
                "min_size_increase": min_size_increase,
                "method": "multi-timeframe",
                "total_found": len(results),
                "data": results[:limit],
            }
        except Exception:
            pass  # Fall through to single-timeframe fallback

    return scan_advanced_candle_patterns_single_tf(exchange, symbols, base_timeframe, pattern_length, min_size_increase, limit)


# ── Volume scanner tools ───────────────────────────────────────────────────────

@mcp.tool(annotations=ToolAnnotations(title="Volume Breakout Scanner", readOnlyHint=True, destructiveHint=False, openWorldHint=True))
async def volume_breakout_scanner(
    exchange: str = "KUCOIN",
    timeframe: str = "15m",
    volume_multiplier: float = 2.0,
    price_change_min: float = 3.0,
    limit: int = 25,
) -> list[dict] | dict:
    """Detect coins with volume breakout + price breakout.

    Args:
        exchange: Exchange name like KUCOIN, BINANCE, BYBIT, MEXC, etc.
        timeframe: One of 5m, 15m, 1h, 4h, 1D, 1W, 1M
        volume_multiplier: How many times the volume should be above normal level (default 2.0)
        price_change_min: Minimum price change percentage (default 3.0)
        limit: Number of rows to return (max 50)

    Returns ``list[dict]`` on success, or an error envelope on total upstream
    failure (``{"error": {"code": "ALL_BATCHES_FAILED", ...}}``). The empty
    list now strictly means "no matches today"; rate-limit cliffs surface
    explicitly.
    """
    exchange = sanitize_exchange(exchange, "KUCOIN")
    timeframe = sanitize_timeframe(timeframe, "15m")
    volume_multiplier = max(1.5, min(10.0, volume_multiplier))
    price_change_min = max(1.0, min(20.0, price_change_min))
    limit = max(1, min(limit, 50))
    try:
        # volume_breakout_scan iterates many batches against the screener
        # endpoint (sync urllib). Off-load to a worker thread to keep the
        # event loop responsive for other concurrent tool calls.
        return await asyncio.to_thread(
            volume_breakout_scan,
            exchange, timeframe, volume_multiplier, price_change_min, limit,
        )
    except Exception as e:
        return exception_to_envelope(e, context="volume_breakout_scanner")


@mcp.tool(annotations=ToolAnnotations(title="Volume Confirmation Analysis", readOnlyHint=True, destructiveHint=False, openWorldHint=True))
def volume_confirmation_analysis(symbol: str, exchange: str = "KUCOIN", timeframe: str = "15m") -> dict:
    """Detailed volume confirmation analysis for a specific coin.

    Args:
        symbol: Coin symbol (e.g., BTCUSDT)
        exchange: Exchange name
        timeframe: Time frame for analysis
    """
    exchange = sanitize_exchange(exchange, "KUCOIN")
    timeframe = sanitize_timeframe(timeframe, "15m")
    return volume_confirmation_analyze(symbol, exchange, timeframe)


@mcp.tool(annotations=ToolAnnotations(title="Smart Volume Scanner", readOnlyHint=True, destructiveHint=False, openWorldHint=True))
def smart_volume_scanner(
    exchange: str = "KUCOIN",
    min_volume_ratio: float = 2.0,
    min_price_change: float = 2.0,
    rsi_range: str = "any",
    limit: int = 20,
) -> list[dict] | dict:
    """Smart volume + technical analysis combination scanner.

    Args:
        exchange: Exchange name
        min_volume_ratio: Minimum volume multiplier (default 2.0)
        min_price_change: Minimum price change percentage (default 2.0)
        rsi_range: "oversold" (<30), "overbought" (>70), "neutral" (30-70), "any"
        limit: Number of results (max 30)

    Returns ``list[dict]`` on success. On ANY failure returns a structured
    error envelope ``{"error": {"code": ..., "retryable": ...}}``.
    """
    exchange = sanitize_exchange(exchange, "KUCOIN")
    min_volume_ratio = max(1.2, min(10.0, min_volume_ratio))
    min_price_change = max(0.5, min(20.0, min_price_change))
    limit = max(1, min(limit, 30))
    try:
        return smart_volume_scan(exchange, min_volume_ratio, min_price_change, rsi_range, limit)
    except Exception as e:
        return exception_to_envelope(e, context="smart_volume_scanner")


# ── Multi-agent analysis ───────────────────────────────────────────────────────

@mcp.tool(annotations=ToolAnnotations(title="Multi-Agent Market Debate", readOnlyHint=True, destructiveHint=False, openWorldHint=True))
def multi_agent_analysis(symbol: str, exchange: str = "KUCOIN", timeframe: str = "15m") -> dict:
    """Run a multi-agent debate (Technical, Sentiment, Risk) for a specific symbol.

    Args:
        symbol: Symbol — crypto: "BTCUSDT"; stocks: "COMI" (EGX), "THYAO" (BIST), "600519" (SSE), "300251" (SZSE), "2330" (TWSE), "3105" (TPEX), "GDX" (AMEX)
        exchange: Exchange — crypto: KUCOIN, BINANCE, MEXC; stocks: EGX, BIST, NASDAQ, NYSE, AMEX, NYSEARCA, PCX, SSE, SZSE, TWSE, TPEX
        timeframe: Time interval (5m, 15m, 1h, 4h, 1D, 1W)

    Returns:
        A structured debate between 3 AI agents culminating in a final trading decision.
    """
    exchange = sanitize_exchange(exchange, "KUCOIN")
    timeframe = sanitize_timeframe(timeframe, "15m")
    full_symbol = normalize_tradingview_symbol(symbol, exchange)
    return run_multi_agent_analysis(full_symbol, exchange, timeframe)


# ── EGX market tools ───────────────────────────────────────────────────────────

@mcp.tool(annotations=ToolAnnotations(title="EGX Market Overview", readOnlyHint=True, destructiveHint=False, openWorldHint=True))
def egx_market_overview(timeframe: str = "1D", limit: int = 10) -> dict:
    """Get a comprehensive overview of the Egyptian Exchange (EGX) market.

    Args:
        timeframe: One of 5m, 15m, 1h, 4h, 1D, 1W, 1M (default 1D for stocks)
        limit: Number of stocks per category (max 20)
    """
    timeframe = sanitize_timeframe(timeframe, "1D")
    limit = max(1, min(limit, 20))
    return get_egx_market_overview(timeframe, limit)


@mcp.tool(annotations=ToolAnnotations(title="EGX Sector Scan", readOnlyHint=True, destructiveHint=False, openWorldHint=True))
def egx_sector_scan(sector: str = "", timeframe: str = "1D", limit: int = 20) -> dict:
    """Scan EGX stocks by sector. Shows available sectors if none specified.

    Args:
        sector: Sector name (banks, healthcare_and_pharma, real_estate, etc.)
                Leave empty to list all sectors.
        timeframe: One of 5m, 15m, 1h, 4h, 1D, 1W, 1M
        limit: Max results per sector (max 50)
    """
    timeframe = sanitize_timeframe(timeframe, "1D")
    limit = max(1, min(limit, 50))
    return scan_egx_sector(sector, timeframe, limit)


@mcp.tool(annotations=ToolAnnotations(title="EGX Sector Rotation Scanner", readOnlyHint=True, destructiveHint=False, openWorldHint=True))
def egx_sector_scanner(
    timeframe: str = "1D",
    top_n_sectors: int = 5,
    top_n_stocks: int = 3,
    min_stock_score: int = 60,
) -> dict:
    """Sector rotation scanner for EGX — identifies hot/cold sectors and top picks.

    Args:
        timeframe: One of 5m, 15m, 1h, 4h, 1D, 1W, 1M (default 1D)
        top_n_sectors: Number of top sectors to show stock picks for (1-18, default 5)
        top_n_stocks: Number of top stocks per highlighted sector (1-10, default 3)
        min_stock_score: Minimum stock score for picks (0-100, default 60)
    """
    timeframe = sanitize_timeframe(timeframe, "1D")
    top_n_sectors = max(1, min(18, top_n_sectors))
    top_n_stocks = max(1, min(10, top_n_stocks))
    min_stock_score = max(0, min(100, min_stock_score))
    return run_egx_sector_scanner(timeframe, top_n_sectors, top_n_stocks, min_stock_score)


@mcp.tool(annotations=ToolAnnotations(title="EGX Index Analysis", readOnlyHint=True, destructiveHint=False, openWorldHint=True))
def egx_index_analysis(index: str = "EGX30", timeframe: str = "1D", limit: int = 30) -> dict:
    """Analyse an EGX index showing constituent performance with full indicators.

    Args:
        index: EGX30, EGX70, EGX100, SHARIAH33, EGX35LV, TAMAYUZ
        timeframe: One of 5m, 15m, 1h, 4h, 1D, 1W, 1M (default 1D)
        limit: Number of stocks to show in detail (max 100)
    """
    timeframe = sanitize_timeframe(timeframe, "1D")
    limit = max(1, min(limit, 100))
    return analyze_egx_index(index, timeframe, limit)


@mcp.tool(annotations=ToolAnnotations(title="EGX Stock Screener", readOnlyHint=True, destructiveHint=False, openWorldHint=True))
def egx_stock_screener(
    timeframe: str = "1D",
    min_score: int = 55,
    index_filter: str = "",
    limit: int = 20,
) -> dict:
    """Production stock ranking engine for EGX — finds strong stocks with actionable setups.

    Args:
        timeframe: One of 5m, 15m, 1h, 4h, 1D, 1W, 1M (default 1D)
        min_score: Minimum stock score to include (0-100, default 55)
        index_filter: Filter by index — EGX30, EGX70, EGX100, SHARIAH33, EGX35LV, TAMAYUZ
        limit: Number of results (max 50)
    """
    timeframe = sanitize_timeframe(timeframe, "1D")
    min_score = max(0, min(100, min_score))
    limit = max(1, min(50, limit))
    return screen_egx_stocks(timeframe, min_score, index_filter, limit)


@mcp.tool(annotations=ToolAnnotations(title="EGX Trade Plan", readOnlyHint=True, destructiveHint=False, openWorldHint=True))
def egx_trade_plan(symbol: str, timeframe: str = "1D") -> dict:
    """Generate a full trade plan for a specific EGX stock.

    Args:
        symbol: EGX stock symbol (e.g., "COMI", "TMGH", "FWRY")
        timeframe: One of 5m, 15m, 1h, 4h, 1D, 1W, 1M (default 1D)
    """
    timeframe = sanitize_timeframe(timeframe, "1D")
    return generate_egx_trade_plan(symbol, timeframe)


@mcp.tool(annotations=ToolAnnotations(title="EGX Fibonacci Retracement", readOnlyHint=True, destructiveHint=False, openWorldHint=True))
def egx_fibonacci_retracement(symbol: str, lookback: str = "52W", timeframe: str = "1D") -> dict:
    """Fibonacci retracement analysis for EGX stocks.

    Args:
        symbol: EGX stock symbol (e.g., "COMI", "TMGH", "FWRY")
        lookback: Period for swing high/low — "1M", "3M", "6M", "52W", "ALL" (default 52W)
        timeframe: Analysis timeframe (5m, 15m, 1h, 4h, 1D, 1W, 1M — default 1D)
    """
    timeframe = sanitize_timeframe(timeframe, "1D")
    lookback = lookback.strip().upper()
    return analyze_egx_fibonacci(symbol, lookback, timeframe)


# ── Multi-timeframe analysis ───────────────────────────────────────────────────

@mcp.tool(annotations=ToolAnnotations(title="Multi-Timeframe Analysis", readOnlyHint=True, destructiveHint=False, openWorldHint=True))
async def multi_timeframe_analysis(symbol: str, exchange: str = "KUCOIN") -> dict:
    """Multi-timeframe alignment analysis (Weekly → Daily → 4H → 1H → 15m).

    Canonical name is exactly `multi_timeframe_analysis` (there is no
    "get_multi_timeframe_analysis" tool). Use this for cross-timeframe trend
    alignment on ONE symbol; for a single-timeframe deep dive use
    `coin_analysis`; for TA + sentiment + news use `combined_analysis`.

    Example: multi_timeframe_analysis(symbol="SOLUSDT", exchange="BINANCE")

    Args:
        symbol: Bare ticker, no exchange prefix — crypto: "BTCUSDT"; stocks: "COMI" (EGX), "THYAO" (BIST), "600519" (SSE), "300251" (SZSE), "2330" (TWSE), "3105" (TPEX), "GDX" (AMEX)
        exchange: Exchange — crypto: KUCOIN, BINANCE, MEXC; stocks: EGX, BIST, NASDAQ, NYSE, AMEX, NYSEARCA, PCX, SSE, SZSE, TWSE, TPEX
    """
    exchange = sanitize_exchange(exchange, "KUCOIN")
    full_symbol = normalize_tradingview_symbol(symbol, exchange)
    # tradingview_ta backs this with a single threading.Semaphore-gated
    # request; pushing the whole call to a thread keeps the event loop
    # free while we wait on the upstream HTTP.
    return await asyncio.to_thread(run_multi_timeframe_analysis, full_symbol, exchange)


# ── Sentiment & news tools ─────────────────────────────────────────────────────

@mcp.tool(annotations=ToolAnnotations(title="Market News Sentiment", readOnlyHint=True, destructiveHint=False, openWorldHint=True))
def market_sentiment(symbol: str, category: str = "all", limit: int = 20) -> dict:
    """News sentiment for stocks and crypto (licensed Marketaux entity sentiment).

    Args:
        symbol: Asset symbol ("AAPL", "BTC", "ETH", "TSLA")
        category: News group to search ("crypto", "stocks", "all")
        limit: Max articles to analyse
    """
    return analyze_sentiment(symbol, category, limit)


@mcp.tool(annotations=ToolAnnotations(title="Financial News Feed", readOnlyHint=True, destructiveHint=False, openWorldHint=True))
async def financial_news(symbol: str = None, category: str = "stocks", limit: int = 10) -> dict:
    """Real-time financial news via Marketaux (licensed).

    Args:
        symbol: Optional symbol filter ("AAPL", "BTC"). None = all news.
        category: News category ("crypto", "stocks", "all")
        limit: Max number of news items
    """
    # The Marketaux fetch is sync (urllib) — offload so parallel news
    # requests don't block the event loop.
    return await asyncio.to_thread(fetch_news_summary, symbol, category, limit)


@mcp.tool(annotations=ToolAnnotations(title="Combined TA + Sentiment + News", readOnlyHint=True, destructiveHint=False, openWorldHint=True))
async def combined_analysis(symbol: str, exchange: str = "NASDAQ", timeframe: str = "1D") -> dict:
    """POWER TOOL: TradingView technical analysis + news sentiment + financial news.

    Use this when you want TA AND sentiment AND news for one symbol in a
    single call. For indicators only, `coin_analysis` is faster; for
    cross-timeframe trend alignment use `multi_timeframe_analysis`.

    Example: combined_analysis(symbol="NVDA", exchange="NASDAQ", timeframe="1D")

    Args:
        symbol: Bare ticker, no exchange prefix ("AAPL", "BTCUSDT", "THYAO", "GDX")
        exchange: Exchange (NASDAQ, NYSE, AMEX, NYSEARCA, PCX, BINANCE, KUCOIN, MEXC, BIST, EGX, TWSE, TPEX)
        timeframe: Analysis timeframe (5m, 15m, 1h, 4h, 1D, 1W)
    """
    exchange_clean = sanitize_exchange(exchange, "NASDAQ")
    timeframe_clean = sanitize_timeframe(timeframe, "1D")
    cat = "crypto" if exchange_clean.upper() in ["BINANCE", "KUCOIN", "BYBIT", "MEXC"] else "stocks"

    # The three sub-calls hit independent upstreams (TradingView TA,
    # Marketaux news/sentiment) so fan them out in parallel. Pre-P4 this was
    # ~3x sequential wall-clock; now bounded by the slowest single call.
    # The TA throttle (threading.Semaphore in screener_provider) still
    # caps in-flight TV calls correctly because asyncio.to_thread runs
    # the sync call in a worker thread that respects the semaphore.
    tech, sentiment, news = await asyncio.gather(
        asyncio.to_thread(analyze_coin, symbol, exchange_clean, timeframe_clean),
        asyncio.to_thread(analyze_sentiment, symbol, cat),
        asyncio.to_thread(fetch_news_summary, symbol, cat, 5),
    )

    tech_momentum = tech.get("market_sentiment", {}).get("momentum", "") if isinstance(tech, dict) else ""
    tech_bullish = tech_momentum == "Bullish"
    sent_bullish = sentiment.get("sentiment_score", 0) > 0.1
    signals_agree = tech_bullish == sent_bullish
    confidence = "HIGH" if signals_agree else "MIXED"
    tech_signal = tech.get("market_sentiment", {}).get("buy_sell_signal", "N/A") if isinstance(tech, dict) else "N/A"

    return {
        "symbol": symbol,
        "exchange": exchange_clean,
        "timeframe": timeframe_clean,
        "technical": tech,
        "sentiment": sentiment,
        "news": {"count": news.get("count", 0), "latest": news.get("items", [])[:3]},
        "confluence": {
            "signals_agree": signals_agree,
            "confidence": confidence,
            "recommendation": (
                f"Technical {tech_signal} "
                f"{'confirmed by' if signals_agree else 'conflicts with'} "
                f"{sentiment.get('sentiment_label', 'Neutral')} news sentiment "
                f"({sentiment.get('posts_analyzed', 0)} articles analyzed)"
            ),
        },
    }


# ── Backtest tools ─────────────────────────────────────────────────────────────

@mcp.tool(annotations=ToolAnnotations(title="Strategy Backtest", readOnlyHint=True, destructiveHint=False, openWorldHint=True))
def backtest_strategy(
    symbol: str,
    strategy: str,
    period: str = "1y",
    initial_capital: float = 10000.0,
    commission_pct: float = 0.1,
    slippage_pct: float = 0.05,
    interval: str = "1d",
    include_trade_log: bool = False,
    include_equity_curve: bool = False,
) -> dict:
    """Backtest a trading strategy on historical data with institutional-grade metrics.

    Args:
        symbol: Yahoo Finance symbol (AAPL, BTC-USD, THYAO.IS, ^GSPC)
        strategy: rsi | bollinger | macd | ema_cross | supertrend | donchian
                  | rsi_pullback | keltner_breakout | triple_ema
                  (rsi_pullback and triple_ema need period >= '1y' for SMA200 warmup)
        period: '1mo', '3mo', '6mo', '1y', '2y'
        initial_capital: Starting capital in USD (default $10,000)
        commission_pct: Per-trade commission % (default 0.1%)
        slippage_pct: Per-trade slippage % (default 0.05%)
        interval: '1d' (daily) or '1h' (hourly)
        include_trade_log: Include full per-trade log (default False)
        include_equity_curve: Include equity curve data points (default False)
    """
    return run_backtest(
        symbol, strategy, period, initial_capital,
        commission_pct, slippage_pct, interval,
        include_trade_log, include_equity_curve,
    )


@mcp.tool(annotations=ToolAnnotations(title="Strategy Comparison Race", readOnlyHint=True, destructiveHint=False, openWorldHint=True))
def compare_strategies(
    symbol: str,
    period: str = "1y",
    initial_capital: float = 10000.0,
    interval: str = "1d",
) -> dict:
    """Run all 9 strategies (RSI, Bollinger, MACD, EMA Cross, Supertrend, Donchian, RSI Pullback, Keltner Breakout, Triple EMA) and return a ranked leaderboard.

    Args:
        symbol: Yahoo Finance symbol (AAPL, BTC-USD, SPY…)
        period: '1mo', '3mo', '6mo', '1y', '2y'
                (period >= '1y' recommended so rsi_pullback and triple_ema can
                 complete SMA200 warmup; otherwise they contribute zero trades)
        initial_capital: Starting capital in USD (default $10,000)
        interval: '1d' (daily) or '1h' (hourly)
    """
    return _compare_strategies(symbol, period, initial_capital, interval=interval)


@mcp.tool(annotations=ToolAnnotations(title="Walk-Forward Backtest", readOnlyHint=True, destructiveHint=False, openWorldHint=True))
def walk_forward_backtest_strategy(
    symbol: str,
    strategy: str,
    period: str = "2y",
    initial_capital: float = 10000.0,
    commission_pct: float = 0.1,
    slippage_pct: float = 0.05,
    n_splits: int = 3,
    train_ratio: float = 0.7,
    interval: str = "1d",
) -> dict:
    """Walk-forward backtest to detect overfitting — validates strategy on unseen data.

    Args:
        symbol: Yahoo Finance symbol (AAPL, BTC-USD, SPY…)
        strategy: rsi | bollinger | macd | ema_cross | supertrend | donchian
                  | keltner_breakout
                  (rsi_pullback and triple_ema not supported here — SMA200 warmup
                   exceeds typical fold size; use run_backtest with period='2y')
        period: '1mo', '3mo', '6mo', '1y', '2y' (recommend '2y')
        initial_capital: Starting capital per fold in USD (default $10,000)
        commission_pct: Per-trade commission % (default 0.1%)
        slippage_pct: Per-trade slippage % (default 0.05%)
        n_splits: Number of walk-forward folds (default 3, max 10)
        train_ratio: Fraction of each fold used for training (default 0.7)
        interval: '1d' (daily) or '1h' (hourly)
    """
    return walk_forward_backtest(
        symbol, strategy, period, initial_capital,
        commission_pct, slippage_pct, n_splits, train_ratio, interval,
    )


# ── Risk management tools ──────────────────────────────────────────────────────

@mcp.tool(annotations=ToolAnnotations(title="Position Size Calculator", readOnlyHint=True, destructiveHint=False, openWorldHint=False))
def calculate_position_size(
    account_equity: float,
    risk_pct: float,
    entry_price: float,
    stop_price: float,
    contract_size: float = 1.0,
    max_position_pct: float = 100.0,
) -> dict:
    """Size a position so a stop-out risks exactly `risk_pct` of account equity,
    the fractional/risk-based sizing none of the backtest engines do internally
    (they compound equity by raw % price move, not by capital actually risked).

    Args:
        account_equity: current account/portfolio equity in account currency.
        risk_pct: % of equity to risk if the stop is hit (e.g. 1.0 = risk 1%).
        entry_price/stop_price: planned entry and stop-loss price, same units.
        contract_size: units per 1.0 quantity (1.0 for shares/coins; the lot's
            unit size for FX/futures contracts).
        max_position_pct: hard cap on notional as % of equity, applied even if
            the stop is close enough that risk-based sizing would exceed it.
    """
    return calc_position_size(account_equity, risk_pct, entry_price, stop_price,
                               contract_size, max_position_pct)


@mcp.tool(annotations=ToolAnnotations(title="Portfolio Value at Risk", readOnlyHint=True, destructiveHint=False, openWorldHint=False))
def calculate_portfolio_var(
    returns_pct: list[float],
    confidence: float = 95.0,
    method: str = "historical",
    capital: float = 0.0,
) -> dict:
    """Value at Risk (and CVaR/Expected Shortfall) from a series of per-period
    % returns — feed it trade returns from a backtest's trade_log or daily
    equity-curve returns.

    Args:
        returns_pct: per-period returns in percent, e.g. [-1.2, 0.8, 2.1, ...].
        confidence: confidence level, e.g. 95 or 99.
        method: 'historical' (empirical percentile) or 'parametric' (Gaussian).
        capital: if > 0, also expresses VaR/CVaR in currency.
    """
    return calc_var(returns_pct, confidence, method, capital or None)


@mcp.tool(annotations=ToolAnnotations(title="Exposure Limit Check", readOnlyHint=True, destructiveHint=False, openWorldHint=False))
def check_risk_limits(
    open_positions: list[dict],
    new_position_notional: float,
    new_position_symbol: str,
    account_equity: float,
    max_single_symbol_pct: float = 20.0,
    max_total_exposure_pct: float = 100.0,
) -> dict:
    """Check whether adding a new position would breach per-symbol or total
    exposure caps before it's sent to execution.

    Args:
        open_positions: list of {"symbol": str, "notional": float}.
        new_position_notional: notional of the proposed new position.
        new_position_symbol: symbol of the proposed new position.
        account_equity: current equity, used as the denominator for both caps.
        max_single_symbol_pct: cap on any one symbol's total notional / equity.
        max_total_exposure_pct: cap on total notional across positions / equity.
    """
    return check_exposure_limits(open_positions, new_position_notional,
                                  new_position_symbol, account_equity,
                                  max_single_symbol_pct, max_total_exposure_pct)


@mcp.tool(annotations=ToolAnnotations(title="Circuit Breaker Check", readOnlyHint=True, destructiveHint=False, openWorldHint=False))
def check_circuit_breaker_status(
    daily_pnl_pct: float,
    consecutive_losses: int = 0,
    max_daily_loss_pct: float = 3.0,
    max_consecutive_losses: int = 5,
    max_drawdown_pct: float = 0.0,
    current_drawdown_pct: float = 0.0,
) -> dict:
    """Should live/paper trading halt right now? Checks daily loss limit,
    consecutive-loss streak, and optional drawdown ceiling.

    Args:
        daily_pnl_pct: today's P&L as % of starting equity (negative = loss).
        consecutive_losses: count of consecutive losing trades today.
        max_daily_loss_pct: halt if daily loss exceeds this (positive number).
        max_consecutive_losses: halt after this many consecutive losers.
        max_drawdown_pct: optional hard drawdown ceiling from peak equity
            (0 disables this check).
        current_drawdown_pct: current drawdown from peak equity (positive number).
    """
    return check_circuit_breaker(daily_pnl_pct, consecutive_losses, max_daily_loss_pct,
                                  max_consecutive_losses, max_drawdown_pct or None,
                                  current_drawdown_pct)


@mcp.tool(annotations=ToolAnnotations(title="Option Greeks Calculator", readOnlyHint=True, destructiveHint=False, openWorldHint=False))
def calculate_option_greeks(
    spot: float,
    strike: float,
    volatility_pct: float,
    expiry_days: int,
    risk_free_rate_pct: float = 0.0,
    dividend_yield_pct: float = 0.0,
    option_type: str = "call",
) -> dict:
    """European option price + Greeks (delta/gamma/theta/vega/rho) via
    QuantLib's Black-Scholes-Merton engine. Requires QuantLib installed
    (pip install QuantLib); returns a DEPENDENCY_MISSING error envelope
    if it isn't.

    Args:
        spot/strike: underlying price and option strike.
        volatility_pct: annualized implied volatility, e.g. 25.0 for 25%.
        expiry_days: calendar days to expiry.
        risk_free_rate_pct: annualized risk-free rate, e.g. 4.5 for 4.5%.
        dividend_yield_pct: annualized dividend yield, e.g. 1.5 for 1.5%.
        option_type: 'call' or 'put'.
    """
    return calc_option_greeks(spot, strike, volatility_pct, expiry_days,
                               risk_free_rate_pct, dividend_yield_pct, option_type)

# ── Yahoo Finance tools ────────────────────────────────────────────────────────

@mcp.tool(annotations=ToolAnnotations(title="Real-Time Price Quote", readOnlyHint=True, destructiveHint=False, openWorldHint=True))
async def yahoo_price(symbol: str) -> dict:
    """Real-time price quote from Yahoo Finance for any stock, crypto, ETF or index.

    Args:
        symbol: Yahoo Finance symbol — e.g. AAPL, BTC-USD, SPY, ^GSPC, EURUSD=X, THYAO.IS
    """
    return await get_price_async(normalize_yahoo_symbol(symbol))


@mcp.tool(annotations=ToolAnnotations(title="Global Market Snapshot", readOnlyHint=True, destructiveHint=False, openWorldHint=True))
def market_snapshot() -> dict:
    """Global market overview: major indices, top crypto, FX rates, and key ETFs.
    Powered by Yahoo Finance.
    """
    return get_market_snapshot()


@mcp.tool(annotations=ToolAnnotations(title="Bitcoin Market Pulse", readOnlyHint=True, destructiveHint=False, openWorldHint=True))
def bitcoin_market_pulse() -> dict:
    """Single-call BTC macro context: price, dominance, total market cap + risk assessment.

    Use this WHENEVER analyzing any cryptocurrency (altcoin or BTC itself) to
    get the broader market frame in one shot. A SOL/ETH/whatever setup looks
    very different when BTC is dumping with rising dominance vs. when alts
    are leading. Calling this once gives Claude the macro context to provide
    Bitcoin-aware commentary alongside the per-coin analysis - without
    chaining 2-3 separate yahoo_price + manual reasoning calls.

    Returns:
      - bitcoin: price, 24h change %, volume, market cap
      - dominance: BTC and ETH market-cap share of total crypto
      - total_market: total crypto mcap + 24h change + active coin count
      - assessment: label (HIGH_RISK / ALT_RISK / ALT_FAVORABLE / OPPORTUNITY_WITH_CAUTION / NEUTRAL) + 1-paragraph reasoning
    """
    return get_bitcoin_market_pulse()


@mcp.tool(annotations=ToolAnnotations(title="Extended-Hours Stock Price", readOnlyHint=True, destructiveHint=False, openWorldHint=True))
async def stock_extended_hours(symbol: str) -> dict:
    """Real-time pre-market and after-hours prices for a US stock symbol.

    Use this when the user asks about a stock outside the regular 9:30am-4pm
    ET session — earnings reactions, overnight news, "what is X doing in
    after-hours?", "how did Y open in pre-market?". Returns the most recent
    valid print from each session window (pre-market, regular, post-market)
    along with computed % changes vs. the previous close and the regular
    close, respectively.

    During the regular session, post_market will be null (no data yet).
    On weekends/holidays, returns whatever's most recent in each window.

    Args:
        symbol: US stock symbol — AAPL, NVDA, TSLA, SPY, ^GSPC, etc.

    Returns:
        - pre_market: {price, as_of_utc, change_vs_previous_close_pct} or null
        - regular: {price, as_of_utc, change_pct} (consolidated tape close)
        - post_market: {price, as_of_utc, change_vs_regular_close_pct} or null
        - previous_close, currency, exchange, market_state for context
    """
    return await get_extended_hours_price_async(symbol)


@mcp.tool(annotations=ToolAnnotations(title="Options Chain", readOnlyHint=True, destructiveHint=False, openWorldHint=True))
def stock_options_chain(symbol: str, expiry: Optional[str] = None) -> dict:
    """Full options chain (calls + puts) for a US stock symbol and one expiry.

    Use this when the user asks "what's the options chain for X?", "show me
    AAPL puts expiring next Friday", or wants to inspect bid/ask/IV/volume on
    a specific strike. If no expiry is provided, returns the nearest expiry
    so Claude can quote it back and ask "want a different one?".

    Args:
        symbol: US stock symbol — AAPL, NVDA, TSLA, SPY, etc.
        expiry: Optional ISO date (YYYY-MM-DD). Must match one of the
            `available_expiries` Yahoo returns; otherwise returns an error
            with the list of valid dates.

    Returns:
        - underlying_price, underlying_change_pct
        - requested_expiry, available_expiries (list of YYYY-MM-DD)
        - call_count, put_count
        - calls: list of {strike, last_price, bid, ask, volume,
          open_interest, implied_volatility, in_the_money, expiration}
        - puts: same shape as calls
    """
    return get_options_chain(symbol, expiry)


@mcp.tool(annotations=ToolAnnotations(title="Unusual Options Activity", readOnlyHint=True, destructiveHint=False, openWorldHint=True))
def stock_options_unusual_activity(
    symbol: str,
    top_n: int = 10,
    min_volume: int = 100,
    expiries: int = 4,
) -> dict:
    """Top strikes by volume / open-interest ratio — institutional positioning signal.

    Use this when the user asks "any unusual options activity on X?", "where
    is the smart money positioned on NVDA before earnings?", or wants a
    V/OI screener for a ticker. A V/OI ratio > 1 means today's volume already
    exceeds standing open interest, which classically flags fresh institutional
    positioning on a specific strike in a specific direction (call vs put).

    Scans the soonest few expirations, filters out illiquid strikes (under
    `min_volume`), and returns the top-N sorted by V/OI descending. Also
    returns aggregate call vs put volume so Claude can comment on the
    overall directional bias.

    Args:
        symbol: US stock symbol — AAPL, NVDA, TSLA, SPY, META, etc.
        top_n: How many strikes to return. Default 10.
        min_volume: Filter floor for today's volume — prevents noise from
            illiquid strikes with high V/OI ratios. Default 100.
        expiries: Number of soonest expirations to scan. Default 4
            (typically covers ~1 month of weeklies + monthlies).

    Returns:
        - underlying_price
        - expiries_scanned (list of YYYY-MM-DD)
        - total_call_volume, total_put_volume, put_call_volume_ratio
        - unusual: list of top-N contracts sorted by V/OI desc, each with
          {strike, side (call|put), expiration, volume, open_interest,
          v_oi_ratio, last_price, implied_volatility, in_the_money,
          strike_vs_spot_pct (moneyness)}
    """
    return get_unusual_options_activity(symbol, top_n, min_volume, expiries)


# ── Futures tools ─────────────────────────────────────────────────────────────

@mcp.tool(annotations=ToolAnnotations(title="Futures Market Overview", readOnlyHint=True, destructiveHint=False, openWorldHint=True))
def futures_market_overview(
    category: str = "all",
    exchanges: str = "us",
    limit: int = 30,
    volume_min: int = 0,
) -> dict:
    """Top futures contracts sorted by trading volume.

    Args:
        category:   all | equity_index | energy | metals | agriculture | rates | forex | crypto_futures
        exchanges:  us (CME, COMEX, NYMEX, CBOT) | global (adds ICE, EUREX)
        limit:      max contracts to return (default 30)
        volume_min: minimum volume filter (0 = no filter)

    Returns:
        Dict with total_available count and list of contracts with OHLCV + % change.
    """
    try:
        return get_futures_overview(
            category=category,
            exchanges=exchanges,
            limit=limit,
            volume_min=volume_min,
        )
    except Exception as exc:
        # NOTE: was ErrorCode.SERVICE_ERROR — a member that never existed, so
        # this handler itself raised AttributeError instead of returning.
        return make_error(ErrorCode.UPSTREAM_ERROR, f"Futures overview failed: {exc}", retryable=True)


@mcp.tool(annotations=ToolAnnotations(title="Futures Top Movers", readOnlyHint=True, destructiveHint=False, openWorldHint=True))
def futures_top_movers(
    direction: str = "gainers",
    exchanges: str = "us",
    limit: int = 20,
    volume_min: int = 10,
) -> dict:
    """Futures contracts with the biggest percentage moves today.

    Args:
        direction:  gainers | losers
        exchanges:  us | global
        limit:      max results
        volume_min: minimum volume filter (default 10, filters illiquid contracts)

    Returns:
        List of futures ranked by % change with OHLCV data.
    """
    direction = direction.lower()
    if direction not in ("gainers", "losers"):
        direction = "gainers"
    try:
        return get_futures_movers(
            direction=direction,
            exchanges=exchanges,
            limit=limit,
            volume_min=volume_min,
        )
    except Exception as exc:
        return make_error(ErrorCode.UPSTREAM_ERROR, f"Futures movers failed: {exc}", retryable=True)


@mcp.tool(annotations=ToolAnnotations(title="Futures Category Snapshot", readOnlyHint=True, destructiveHint=False, openWorldHint=True))
def futures_category_snapshot(category: str = "energy") -> dict:
    """Quote all major front-month contracts in a specific futures category.

    Args:
        category: equity_index | energy | metals | agriculture | rates | forex | crypto_futures

    Returns:
        OHLCV quotes for the standard watchlist of contracts in that category.
        Example symbols: ES1! NQ1! (equity_index), CL1! NG1! (energy), GC1! SI1! (metals).
    """
    return get_futures_category_snapshot(category)


@mcp.tool(annotations=ToolAnnotations(title="Futures Watchlist", readOnlyHint=True, destructiveHint=False, openWorldHint=True))
def futures_watchlist() -> dict:
    """Return the full categorized list of well-known front-month futures symbols.

    Categories: equity_index, energy, metals, agriculture, rates, forex, crypto_futures.
    Use these symbols with futures_category_snapshot or coin_analysis for deeper analysis.
    """
    return get_futures_watchlist()


@mcp.tool(annotations=ToolAnnotations(title="US Stock Screener", readOnlyHint=True, destructiveHint=False, openWorldHint=True))
async def stock_screener(
    country: str = "america",
    stock_type: str = "common",
    limit: int = 50,
    exclude_otc: bool = True,
    compact: bool = False,
    sort_by: str = "market_cap",
) -> dict:
    """Screen stocks by share type — the API twin of TradingView's
    "Common stock" / "Preferred stock" symbol-search filter.

    Args:
        country: TradingView market name — e.g. america, korea, germany,
            brazil, japan, uk, india, turkey, canada, australia, france, hongkong
        stock_type: common | preferred
        limit: rows to return (max 2000, single upstream request), ranked by market cap descending
        exclude_otc: default True — drop OTC listings (foreign companies traded
            over-the-counter); "america" otherwise means "US venue", not "US company"
        compact: default False — True returns only ticker/symbol/price/currency/
            change_percent per row (light payload for bulk price feeds)
        sort_by: market_cap (default) | dividend_yield | change | price —
            server-side descending sort over the WHOLE market, so e.g.
            sort_by=dividend_yield with limit=20 is the market's true top-20
            dividend payers, not just the biggest companies re-sorted

    Returns:
        Envelope dict: total_matches (market-wide count), returned, and rows
        of {ticker, symbol, description, exchange, price, open, high, low,
        currency, change_percent, dividend_yield, market_cap} — price is the
        current/last close; open/high/low are the current session's daily bar. Prices are in the
        market's local currency (e.g. KRW for korea).
    """
    try:
        # tradingview-screener is sync (urllib) — off-load to a worker thread
        # so the event loop stays free for concurrent tool calls.
        return await asyncio.to_thread(
            screen_stocks, country, stock_type, limit, exclude_otc, compact, sort_by
        )
    except ValueError as e:
        return make_error(ErrorCode.INVALID_PARAMETER, str(e))
    except Exception as e:
        return make_error(
            ErrorCode.UPSTREAM_ERROR,
            f"scan failed for market {country!r}: {e}",
            known_good_markets=list(EXAMPLE_MARKETS),
        )


@mcp.tool(annotations=ToolAnnotations(title="Multi-Symbol Stock Prices", readOnlyHint=True, destructiveHint=False, openWorldHint=True))
async def stock_prices(tickers: str) -> dict:
    """Current price + daily % change for specific stock symbols.

    Args:
        tickers: comma-separated EXCHANGE:SYMBOL list (max 2000 — one
            upstream request even at full size), e.g.
            "NASDAQ:NVDA, NASDAQ:TSLA, KRX:005930". The exchange prefix is
            required — the scanner's direct-ticker lookup is exchange-scoped.

    Returns:
        Envelope dict: rows of {ticker, symbol, description, exchange, price,
        open, high, low, currency, change_percent} plus a not_found list naming any requested
        ticker the scanner didn't recognize.
    """
    try:
        return await asyncio.to_thread(fetch_stock_prices, tickers)
    except ValueError as e:
        return make_error(ErrorCode.INVALID_PARAMETER, str(e))
    except Exception as e:
        return make_error(ErrorCode.UPSTREAM_ERROR, f"price lookup failed: {e}")


# ── Paper trading / portfolio tools ────────────────────────────────────────────
#
# SQLite-backed simulated portfolio (core/portfolio.py) — no real money or
# broker connection involved. `execute_paper_trade` is the first stateful
# tool in this server; it is the only place risk_service's exposure and
# circuit-breaker checks are actually enforced (as opposed to just computed
# and reported back).

@mcp.tool(annotations=ToolAnnotations(title="Execute Paper Trade", readOnlyHint=False, destructiveHint=True, idempotentHint=False, openWorldHint=False))
def execute_paper_trade(
    user_id: str,
    symbol: str,
    side: str,
    quantity: float = 0.0,
    current_price: float = 0.0,
    risk_pct: float = 0.0,
    stop_price: float = 0.0,
    max_single_symbol_pct: float = 0.0,
    max_total_exposure_pct: float = 0.0,
    max_daily_loss_pct: float = 0.0,
    max_consecutive_losses: int = 0,
) -> dict:
    """Execute a simulated BUY/SELL against a per-user SQLite paper portfolio
    (starts each new user_id at $10,000 cash). No real broker or funds are
    involved — this is for testing strategies/risk rules end-to-end before
    any live wiring exists.

    Args:
        user_id: arbitrary identifier — a new one is created with $10,000 on first use.
        symbol: instrument symbol (stored uppercased; not validated against any exchange).
        side: 'BUY' or 'SELL'.
        quantity: units to trade. Ignored on a BUY if risk_pct and stop_price are both set
            (quantity is computed instead). Required (> 0) for SELL and for a BUY without
            risk-based sizing.
        current_price: execution price for this trade.
        risk_pct: if set with stop_price, size the BUY so a stop-out risks exactly this %
            of current equity (cash + open positions marked at average price).
        stop_price: planned stop-loss price, used only for risk_pct sizing.
        max_single_symbol_pct / max_total_exposure_pct: if set (> 0), reject the BUY if it
            would push that symbol's or the whole portfolio's notional past this % of equity.
        max_daily_loss_pct / max_consecutive_losses: if set (> 0), reject the BUY if today's
            realized loss or the current losing-trade streak has already tripped this limit.
        Risk checks (all optional, 0 = skip) only ever gate new BUYs — closing a position
        via SELL always goes through.
    """
    return _execute_paper_trade(
        user_id, symbol, quantity, current_price, side,
        risk_pct=risk_pct or None,
        stop_price=stop_price or None,
        max_single_symbol_pct=max_single_symbol_pct or None,
        max_total_exposure_pct=max_total_exposure_pct or None,
        max_daily_loss_pct=max_daily_loss_pct or None,
        max_consecutive_losses=max_consecutive_losses or None,
    )


@mcp.tool(annotations=ToolAnnotations(title="View Paper Portfolio", readOnlyHint=True, destructiveHint=False, openWorldHint=False))
def get_paper_portfolio(user_id: str) -> dict:
    """View a paper-trading user's current cash balance and open positions
    (symbol, quantity, average entry price). Creates the user with $10,000
    cash on first call if they don't exist yet.
    """
    return _get_paper_portfolio(user_id)


@mcp.tool(annotations=ToolAnnotations(title="Pre-Trade Risk Check", readOnlyHint=True, destructiveHint=False, openWorldHint=False))
def check_paper_trade_risk(
    user_id: str,
    symbol: str,
    notional: float,
    max_single_symbol_pct: float = 0.0,
    max_total_exposure_pct: float = 0.0,
    max_daily_loss_pct: float = 0.0,
    max_consecutive_losses: int = 0,
) -> dict:
    """Dry-run exposure and circuit-breaker checks against a paper-trading
    user's current state, without placing any trade — use this to preview
    whether execute_paper_trade would allow a BUY before sending it.

    Args:
        user_id: the paper-trading user to check against.
        symbol: instrument symbol for the proposed new position.
        notional: proposed new position's notional value (quantity * price).
        max_single_symbol_pct / max_total_exposure_pct: exposure caps to check (0 = skip).
        max_daily_loss_pct / max_consecutive_losses: circuit-breaker limits to check (0 = skip).
    """
    return _check_pretrade_risk(
        user_id, symbol, notional,
        max_single_symbol_pct or None,
        max_total_exposure_pct or None,
        max_daily_loss_pct or None,
        max_consecutive_losses or None,
    )

# ── Live signal evaluator ───────────────────────────────────────────────────────
#
# Reuses the same run_ict_backtest/run_custom_backtest/run_pine_backtest code
# paths (no duplicated detection logic) over a short trailing window to check
# whether a fresh entry signal exists on the latest available bar of local
# CSV data. There is no live broker/network feed in this codebase — "live"
# here means "as fresh as whatever is on disk." Intended to be invoked on a
# schedule by an external process (cron/APScheduler), not per-tick by an LLM.

@mcp.tool(annotations=ToolAnnotations(title="Evaluate Live Signal", readOnlyHint=True, destructiveHint=False, openWorldHint=False))
def evaluate_live_signal(
    engine: str,
    strategy_name: str,
    symbol: str = "",
    period: str = "1mo",
    atr_period: int = 14,
    atr_stop_multiplier: float = 1.5,
    interval: str = "",
    ltf_interval: str = "",
    htf_interval: str = "",
    broker_utc_offset_hours: float = 0.0,
) -> dict:
    """Check whether a strategy has a fresh entry signal on the most recent
    bar of local CSV data, by running the matching backtest engine over a
    short window and checking if its most recent trade opened on the latest
    bar. Also returns a fallback ATR-based stop suggestion — NOT the
    strategy's own internal SL (no engine's trade log exposes that).

    Args:
        engine: 'ict' (ict_strategies registry), 'custom' (generic
            swing-pivot proxy engine), or 'pine' (pine_lite).
        strategy_name: the strategy file's NAME field or filename.
        symbol: overrides the file's SYMBOL_DEFAULT if given.
        period: trailing window for the underlying backtest (e.g. '1mo', '3mo').
        atr_period / atr_stop_multiplier: fallback stop-loss sizing on a fresh signal.
        interval: candle interval — used by 'custom'/'pine' engines only.
        ltf_interval / htf_interval / broker_utc_offset_hours: used by the 'ict' engine only.
    """
    kwargs: dict = {}
    if engine == "ict":
        if ltf_interval:
            kwargs["ltf_interval"] = ltf_interval
        if htf_interval:
            kwargs["htf_interval"] = htf_interval
        if broker_utc_offset_hours:
            kwargs["broker_utc_offset_hours"] = broker_utc_offset_hours
    elif interval:
        kwargs["interval"] = interval
    return _evaluate_live_signal(
        engine, strategy_name, symbol or None, period,
        atr_period, atr_stop_multiplier, **kwargs,
    )


@mcp.tool(annotations=ToolAnnotations(title="Scan Watchlist For Signals", readOnlyHint=False, destructiveHint=True, idempotentHint=False, openWorldHint=False))
def scan_watchlist(
    watchlist: list[dict],
    user_id: str = "",
    auto_paper_trade: bool = False,
    risk_pct: float = 1.0,
    max_single_symbol_pct: float = 0.0,
    max_total_exposure_pct: float = 0.0,
    max_daily_loss_pct: float = 0.0,
    max_consecutive_losses: int = 0,
    send_telegram_alerts: bool = False,
) -> dict:
    """Scan multiple strategy/symbol pairs for fresh signals in one call —
    meant to be invoked on a schedule by an external process (cron/
    APScheduler), keeping the LLM out of the per-tick execution loop.

    Args:
        watchlist: list of dicts, each at minimum
            {"engine": "ict"|"custom"|"pine", "strategy_name": "..."}, plus
            optional "symbol"/"period"/"interval"/"ltf_interval"/
            "htf_interval"/"broker_utc_offset_hours" per evaluate_live_signal.
        user_id: paper-trading user to execute against; required if auto_paper_trade is True.
        auto_paper_trade: if True, auto-BUYs every fresh LONG signal via the paper
            portfolio, sized by risk_pct against the suggested ATR stop. Short
            signals are always reported, never auto-traded (no short-selling
            support in the paper portfolio).
        risk_pct: % of paper-portfolio equity to risk per auto-trade.
        max_single_symbol_pct / max_total_exposure_pct / max_daily_loss_pct /
            max_consecutive_losses: risk limits applied to each auto-trade (0 = skip).
        send_telegram_alerts: if True, sends one Telegram message per fresh signal
            (requires TV_MCP_TELEGRAM_BOT_TOKEN / TV_MCP_TELEGRAM_CHAT_ID env vars).
    """
    return _scan_watchlist(
        watchlist, user_id or None, auto_paper_trade, risk_pct,
        max_single_symbol_pct or None, max_total_exposure_pct or None,
        max_daily_loss_pct or None, max_consecutive_losses or None,
        send_telegram_alerts,
    )


@mcp.tool(annotations=ToolAnnotations(title="Send Telegram Alert", readOnlyHint=False, destructiveHint=True, idempotentHint=False, openWorldHint=True))
def send_telegram_alert(text: str, bot_token: str = "", chat_id: str = "") -> dict:
    """Send a Telegram message via the Bot API — mainly for testing your
    TV_MCP_TELEGRAM_BOT_TOKEN / TV_MCP_TELEGRAM_CHAT_ID setup before relying
    on scan_watchlist's automatic alerts.

    Args:
        text: message body (Markdown formatting supported).
        bot_token / chat_id: override the TV_MCP_TELEGRAM_BOT_TOKEN /
            TV_MCP_TELEGRAM_CHAT_ID env vars for this call.
    """
    return _send_telegram_alert(text, bot_token or None, chat_id or None)

# ── Data abstraction layer ──────────────────────────────────────────────────────
#
# OpenBB-pattern unified provider abstraction, free-provider substitution
# per CLAUDE.md's plan (see core/services/data_providers/__init__.py for the
# full substitution table). get_ohlcv routes to the existing CSV/Yahoo
# fetchers; FRED and SEC EDGAR are genuinely new free official sources.

@mcp.tool(annotations=ToolAnnotations(title="Unified OHLCV Fetch", readOnlyHint=True, destructiveHint=False, openWorldHint=True))
def get_ohlcv(
    symbol: str,
    source: str,
    period: str = "3mo",
    interval: str = "1d",
) -> dict:
    """Fetch OHLCV candles from either backing data source through one call.

    Args:
        symbol: instrument symbol. For source='csv': a local MT5-exported
            file under TV_MCP_CSV_DATA_DIR (FX/metals, e.g. 'XAUUSD'). For
            source='yahoo': a Yahoo Finance ticker (equities/crypto/indices,
            e.g. 'AAPL', 'BTC-USD').
        source: 'csv' (local files) or 'yahoo' (Yahoo Finance chart API).
        period: trailing window, e.g. '1mo'/'3mo'/'6mo'/'1y'/'2y' (csv also
            accepts '3y'/'4y'/'5y'/'all').
        interval: candle interval, e.g. '1d'/'1h' (csv also accepts
            '1m'/'5m'/'15m'/'30m'/'4h').
    """
    return _get_ohlcv(symbol, source, period, interval)


@mcp.tool(annotations=ToolAnnotations(title="FRED Macro Series", readOnlyHint=True, destructiveHint=False, openWorldHint=True))
def get_macro_series(
    series_id: str,
    start_date: str = "",
    end_date: str = "",
    limit: int = 1000,
) -> dict:
    """Fetch a macroeconomic time series from FRED (St. Louis Fed) — free,
    official, no substitute needed. Requires FRED_API_KEY env var (free
    signup at https://fred.stlouisfed.org/docs/api/api_key.html).

    Args:
        series_id: FRED series ID, e.g. 'GDP', 'CPIAUCSL' (CPI), 'DFF' (fed
            funds rate), 'UNRATE' (unemployment), 'DGS10' (10y treasury yield).
        start_date / end_date: 'YYYY-MM-DD', optional (defaults to full history).
        limit: max observations returned (default/cap 1000 — narrow the date
            range instead of raising this for long series).
    """
    return _get_fred_series(series_id, start_date or None, end_date or None, limit=limit)


@mcp.tool(annotations=ToolAnnotations(title="SEC EDGAR Company Fundamentals", readOnlyHint=True, destructiveHint=False, openWorldHint=True))
def get_company_fundamentals(ticker: str, concept: str = "Revenues") -> dict:
    """Fetch one XBRL us-gaap concept's full reported history for a
    US-listed company from SEC EDGAR — free, official, no paid tier, no
    substitute needed. Requires SEC_EDGAR_USER_AGENT env var set to a
    descriptive identifier (e.g. 'Your Name your@email.com') per SEC's
    fair-use policy.

    Args:
        ticker: stock ticker, e.g. 'AAPL'.
        concept: an XBRL us-gaap taxonomy tag, e.g. 'Revenues' (default),
            'Assets', 'NetIncomeLoss', 'EarningsPerShareBasic',
            'StockholdersEquity'. Browse a company's actual filings at
            https://www.sec.gov/cgi-bin/browse-edgar to find tags it reports.
    """
    return _get_company_fundamentals(ticker, concept)

# ── ML alpha-factor research ────────────────────────────────────────────────────
#
# Qlib-style Data Handler -> Feature Engineering -> Model -> Analysis
# workflow (research layer only — not wired into any backtest engine's
# entries/exits). Features reuse indicators_calc.py; model is LightGBM if
# it actually loads in this environment, else a pure-stdlib logistic
# regression fallback — see ml_factor_service.py's module docstring.

@mcp.tool(annotations=ToolAnnotations(title="ML Alpha Factor Analysis", readOnlyHint=True, destructiveHint=False, openWorldHint=True))
def run_alpha_factor_analysis(
    symbol: str,
    source: str = "yahoo",
    period: str = "2y",
    interval: str = "1d",
    horizon: int = 5,
    train_frac: float = 0.7,
    model: str = "auto",
) -> dict:
    """Train a directional alpha model on engineered technical factors
    (RSI/MACD/Bollinger %B/ATR%/EMA distance/lagged returns/volatility) and
    report its out-of-sample predictive skill: hit rate vs. baseline and
    Information Coefficient (Spearman correlation between predicted score
    and realized forward return). Research-layer output only — not wired
    into any backtest engine's entries/exits.

    Args:
        symbol: instrument symbol (see get_ohlcv for source-specific format).
        source: 'csv' (local MT5-exported files) or 'yahoo' (Yahoo Finance).
        period / interval: how much history to fetch and at what granularity
            (e.g. period='2y', interval='1d' — need enough bars for a
            meaningful train/test split; more history is better here).
        horizon: bars ahead the label predicts (forward % return).
        train_frac: fraction of bars (chronological split, not shuffled)
            used for training; the rest is held out as the test set.
        model: 'lightgbm' (falls back to logistic if the compiled library
            fails to load — common on Windows without its MSVC runtime),
            'logistic' (pure stdlib, always available), or 'auto' (try
            lightgbm, silently fall back).
    """
    return _run_alpha_factor_analysis(symbol, source, period, interval, horizon, train_frac, model)


# ── RL trading agent (FinRL-inspired) ───────────────────────────────────────────
#
# Lowest-priority, purely additive strategy source per CLAUDE.md's build
# order. Wraps the same feature pipeline as run_alpha_factor_analysis into a
# long/flat Gymnasium environment (no shorting — matches the paper
# portfolio). PPO via Stable-Baselines3 if installed and it actually trains,
# else a pure-stdlib tabular Q-learning fallback — see rl_service.py.

@mcp.tool(annotations=ToolAnnotations(title="Train RL Trading Agent", readOnlyHint=True, destructiveHint=False, openWorldHint=True))
def train_rl_trading_agent(
    symbol: str,
    source: str = "yahoo",
    period: str = "2y",
    interval: str = "1d",
    train_frac: float = 0.7,
    episodes: int = 200,
    algo: str = "auto",
) -> dict:
    """Train a long/flat RL trading agent on engineered technical features
    and evaluate it out-of-sample against buy-and-hold. Research-layer
    output only — not wired into any backtest engine's entries/exits.

    Args:
        symbol: instrument symbol (see get_ohlcv for source-specific format).
        source: 'csv' (local MT5-exported files) or 'yahoo' (Yahoo Finance).
        period / interval: how much history to fetch and at what granularity.
        train_frac: fraction of bars (chronological, not shuffled) used for
            training; the rest is the out-of-sample evaluation window.
        episodes: training passes over the data (Q-learning) or a timesteps
            multiplier (PPO: episodes * n_train_bars timesteps).
        algo: 'ppo' (Stable-Baselines3; falls back to qlearning if not
            installed or training fails), 'qlearning' (pure stdlib, always
            available), or 'auto' (try ppo, silently fall back).
    """
    return _train_rl_trading_agent(symbol, source, period, interval, train_frac, episodes, algo)

# ── Resource ───────────────────────────────────────────────────────────────────

@mcp.resource("exchanges://list")
def exchanges_list() -> str:
    """List available exchanges from the coinlist directory."""
    try:
        current_dir = os.path.dirname(__file__)
        coinlist_dir = os.path.join(current_dir, "coinlist")
        if os.path.exists(coinlist_dir):
            exchanges = [
                f[:-4].upper()
                for f in os.listdir(coinlist_dir)
                if f.endswith(".txt")
            ]
            if exchanges:
                return f"Available exchanges: {', '.join(sorted(exchanges))}"
    except Exception:
        pass
    return "Common exchanges: KUCOIN, BINANCE, BYBIT, MEXC, BITGET, OKX, COINBASE, GATEIO, HUOBI, BITFINEX, KRAKEN, BITSTAMP, BIST, EGX, NASDAQ, TWSE, TPEX"


# ── Entry point ────────────────────────────────────────────────────────────────

# ---------------------------------------------------------------------------
# Blanket async offload for every still-synchronous tool.
#
# mcp 1.12.x calls sync tool functions BARE on the event loop
# (func_metadata.call_fn_with_arg_validation ends in `return fn(**kwargs)`),
# so one slow scan stalls every concurrent session. Seven hot paths were
# hand-converted to async (#49), which left an arbitrary split — top_gainers
# offloaded while its twin top_losers blocked (issue #77). Rather than
# maintaining that list tool-by-tool, wrap whatever is still sync at import
# time in asyncio.to_thread. Tools added later are covered automatically the
# moment this module loads. Argument validation is untouched: FastMCP
# validates against fn_metadata (built from the ORIGINAL signature via
# functools.wraps) and then calls fn(**parsed_kwargs), which the wrapper
# forwards verbatim.
# ---------------------------------------------------------------------------
def _offload_sync_tools() -> int:
    import functools

    converted = 0
    for _tool in mcp._tool_manager.list_tools():
        if _tool.is_async:
            continue
        _sync_fn = _tool.fn

        @functools.wraps(_sync_fn)
        async def _threaded(*args, __sync_fn=_sync_fn, **kwargs):
            return await asyncio.to_thread(__sync_fn, *args, **kwargs)

        _tool.fn = _threaded
        _tool.is_async = True
        converted += 1
    return converted


_OFFLOADED_TOOL_COUNT = _offload_sync_tools()


def main() -> None:
    parser = argparse.ArgumentParser(description="TradingView Screener MCP server")
    parser.add_argument(
        "transport",
        choices=["stdio", "streamable-http"],
        default="stdio",
        nargs="?",
        help="Transport (default stdio)",
    )
    parser.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))
    args = parser.parse_args()

    if os.environ.get("DEBUG_MCP"):
        import sys
        print(f"[DEBUG_MCP] pkg cwd={os.getcwd()} argv={sys.argv} file={__file__}", file=sys.stderr, flush=True)

    if args.transport == "stdio":
        mcp.run()
    else:
        try:
            mcp.settings.host = args.host
            mcp.settings.port = args.port
        except Exception:
            pass
        mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()

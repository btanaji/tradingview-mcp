"""
Telegram alerting — free, no paid tier (a bot token from @BotFather is free).

Used by scan_watchlist callers (the MCP tool, or scripts/live_scanner.py)
to push a message when evaluate_live_signal finds a fresh signal. Kept as a
thin wrapper around the plain Bot API (via `requests`, already a hard
dependency) rather than pulling in python-telegram-bot — sendMessage is one
POST, no need for that library's polling/webhook machinery here.

Env:
  TV_MCP_TELEGRAM_BOT_TOKEN   bot token from @BotFather
  TV_MCP_TELEGRAM_CHAT_ID     destination chat/channel/user id
Both can be overridden per-call; if neither the argument nor the env var is
set, send_telegram_alert returns a DEPENDENCY_MISSING envelope rather than
raising, consistent with this codebase's error-envelope convention.
"""
from __future__ import annotations

import os
from typing import Any, Optional

import requests

from tradingview_mcp.core.errors import ErrorCode, make_error

_API_BASE = "https://api.telegram.org/bot{token}/sendMessage"


def send_telegram_alert(
    text: str,
    bot_token: Optional[str] = None,
    chat_id: Optional[str] = None,
    parse_mode: str = "Markdown",
    timeout_s: float = 10.0,
) -> dict[str, Any]:
    """Send a Telegram message via the Bot API's sendMessage endpoint.

    Args:
        text: message body (Markdown by default — escape/avoid stray
            `*_[]()` in symbol names if switching parse_mode).
        bot_token / chat_id: override TV_MCP_TELEGRAM_BOT_TOKEN /
            TV_MCP_TELEGRAM_CHAT_ID for this call.
        parse_mode: 'Markdown', 'MarkdownV2', 'HTML', or '' for plain text.
        timeout_s: request timeout.
    """
    token = bot_token or os.environ.get("TV_MCP_TELEGRAM_BOT_TOKEN", "")
    chat = chat_id or os.environ.get("TV_MCP_TELEGRAM_CHAT_ID", "")
    if not token or not chat:
        return make_error(
            ErrorCode.DEPENDENCY_MISSING,
            "Telegram not configured: set TV_MCP_TELEGRAM_BOT_TOKEN and "
            "TV_MCP_TELEGRAM_CHAT_ID (or pass bot_token/chat_id).",
        )

    try:
        resp = requests.post(
            _API_BASE.format(token=token),
            json={"chat_id": chat, "text": text, "parse_mode": parse_mode},
            timeout=timeout_s,
        )
        payload = resp.json()
    except requests.exceptions.Timeout:
        return make_error(ErrorCode.UPSTREAM_TIMEOUT, "Telegram API request timed out", retryable=True)
    except Exception as e:
        return make_error(ErrorCode.UPSTREAM_ERROR, f"Telegram send failed: {e}", retryable=True)

    if not resp.ok or not payload.get("ok"):
        return make_error(
            ErrorCode.UPSTREAM_ERROR,
            f"Telegram API rejected message: {payload.get('description', resp.text)}",
            status_code=resp.status_code,
        )

    return {
        "status": "sent",
        "message_id": payload["result"]["message_id"],
        "chat_id": payload["result"]["chat"]["id"],
    }


def format_signal_alert(signal: dict[str, Any]) -> str:
    """Render one evaluate_live_signal() result as a Telegram message."""
    direction = (signal.get("suggested_direction") or "?").upper()
    lines = [
        f"*Fresh {direction} signal* — {signal.get('strategy_name')} ({signal.get('engine')})",
        f"Symbol: `{signal.get('symbol')}`  |  As of: {signal.get('as_of')}",
    ]
    if signal.get("entry_price") is not None:
        lines.append(f"Entry: {signal['entry_price']}")
    if signal.get("suggested_stop_price") is not None:
        lines.append(f"Suggested stop (ATR fallback, not the strategy's own rule): {signal['suggested_stop_price']}")
    summary = signal.get("backtest_summary") or {}
    if summary:
        lines.append(
            f"Backtest: {summary.get('total_trades', '?')} trades, "
            f"{summary.get('win_rate_pct', '?')}% win rate, "
            f"{summary.get('total_return_pct', '?')}% return, "
            f"Sharpe {summary.get('sharpe_ratio', '?')}"
        )
    return "\n".join(lines)


def alert_fresh_signals(
    signals: list[dict[str, Any]],
    bot_token: Optional[str] = None,
    chat_id: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Send one Telegram alert per fresh signal in `signals` (as returned by
    scan_watchlist's "fresh_signals" list). Returns the per-signal send
    results in the same order; a failed send for one signal doesn't stop the
    rest."""
    results = []
    for sig in signals:
        text = format_signal_alert(sig)
        results.append(send_telegram_alert(text, bot_token, chat_id))
    return results

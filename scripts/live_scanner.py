#!/usr/bin/env python
"""
Standalone live-watchlist scanner — the external scheduler for
live_signal_service.scan_watchlist, run outside the MCP request/response
loop per CLAUDE.md's design principle: deterministic scheduled Python does
the per-tick work, the LLM/MCP layer is for judgment calls, not execution.

Not an MCP tool. Run it directly:

    python scripts/live_scanner.py --watchlist watchlist.json --interval 60

or install the package and run `python -m tradingview_mcp` style — this
script imports the installed tradingview_mcp package, so `pip install -e .`
(or being inside the repo with src/ on PYTHONPATH) is a prerequisite.

Deliberately a plain sleep loop, not APScheduler — one dependency-free file
that does exactly one thing (call scan_watchlist on an interval, log the
result, keep going after errors). Swap in APScheduler/cron/systemd-timer
later if you need overlapping jobs, misfire policies, or persistence; this
loop doesn't need any of that for a single watchlist.

Config (all also settable via env, CLI flags win):
  TV_MCP_WATCHLIST_FILE           path to a JSON file: list of watchlist
                                   entries per live_signal_service.scan_watchlist
                                   (each needs at least engine/strategy_name)
  TV_MCP_SCAN_INTERVAL_SECONDS    seconds between scans (default 300)
  TV_MCP_PAPER_USER_ID            user_id for auto-paper-trading
  TV_MCP_AUTO_PAPER_TRADE         "1"/"true" to enable auto paper trading
  TV_MCP_RISK_PCT                 % equity risked per auto-trade (default 1.0)
  TV_MCP_TELEGRAM_BOT_TOKEN / TV_MCP_TELEGRAM_CHAT_ID   for alerts (see alerting_service.py)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tradingview_mcp.core.services.live_signal_service import scan_watchlist  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("live_scanner")

_STOP = False


def _handle_signal(signum, frame):
    global _STOP
    log.info("received signal %s, stopping after current scan", signum)
    _STOP = True


def _str2bool(v: str) -> bool:
    return v.strip().lower() in ("1", "true", "yes", "on")


def load_watchlist(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or not all(isinstance(e, dict) for e in data):
        raise ValueError(f"{path} must contain a JSON array of watchlist-entry objects")
    return data


def run_once(watchlist: list[dict], args: argparse.Namespace) -> dict:
    result = scan_watchlist(
        watchlist,
        user_id=args.user_id,
        auto_paper_trade=args.auto_paper_trade,
        risk_pct=args.risk_pct,
        max_single_symbol_pct=args.max_single_symbol_pct,
        max_total_exposure_pct=args.max_total_exposure_pct,
        max_daily_loss_pct=args.max_daily_loss_pct,
        max_consecutive_losses=args.max_consecutive_losses,
        send_telegram_alerts=args.telegram_alerts,
    )

    if "error" in result:
        log.error("scan_watchlist failed: %s", result["error"])
        return result

    n_fresh = len(result["fresh_signals"])
    log.info(
        "scanned %d, %d fresh signal(s), %d trade(s) executed, %d alert(s) sent",
        result["scanned"], n_fresh, len(result["executed_trades"]),
        sum(1 for a in result["telegram_alerts_sent"] if a.get("status") == "sent"),
    )
    for sig in result["fresh_signals"]:
        log.info(
            "  FRESH %s %s/%s @ %s entry=%s stop=%s",
            (sig.get("suggested_direction") or "?").upper(),
            sig["engine"], sig["strategy_name"], sig["symbol"],
            sig.get("entry_price"), sig.get("suggested_stop_price"),
        )
    for trade in result["executed_trades"]:
        outcome = trade["result"]
        if outcome.get("status") == "success":
            log.info("  EXECUTED BUY %s qty=%s @ %s", trade["symbol"], outcome["quantity"], outcome["price"])
        else:
            log.warning("  TRADE REJECTED %s: %s", trade["symbol"], outcome.get("error") or outcome.get("reasons"))
    for symbol in result.get("short_signals_skipped", []):
        log.info("  short signal skipped (no short-selling support): %s", symbol)
    for alert in result["telegram_alerts_sent"]:
        if alert.get("status") != "sent":
            log.warning("  telegram alert failed: %s", alert.get("error"))

    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--watchlist", default=os.environ.get("TV_MCP_WATCHLIST_FILE"),
                   help="path to a JSON watchlist file (required)")
    p.add_argument("--interval", type=float,
                   default=float(os.environ.get("TV_MCP_SCAN_INTERVAL_SECONDS", "300")),
                   help="seconds between scans (default 300)")
    p.add_argument("--user-id", default=os.environ.get("TV_MCP_PAPER_USER_ID"),
                   help="paper-trading user_id (required if --auto-paper-trade)")
    p.add_argument("--auto-paper-trade", action="store_true",
                   default=_str2bool(os.environ.get("TV_MCP_AUTO_PAPER_TRADE", "0")))
    p.add_argument("--risk-pct", type=float, default=float(os.environ.get("TV_MCP_RISK_PCT", "1.0")))
    p.add_argument("--max-single-symbol-pct", type=float, default=None)
    p.add_argument("--max-total-exposure-pct", type=float, default=None)
    p.add_argument("--max-daily-loss-pct", type=float, default=None)
    p.add_argument("--max-consecutive-losses", type=int, default=None)
    p.add_argument("--telegram-alerts", action="store_true", default=False,
                   help="send a Telegram alert per fresh signal (needs TV_MCP_TELEGRAM_* env vars)")
    p.add_argument("--once", action="store_true", help="run a single scan and exit (no loop)")
    args = p.parse_args()

    if not args.watchlist:
        p.error("--watchlist (or TV_MCP_WATCHLIST_FILE) is required")
    if args.auto_paper_trade and not args.user_id:
        p.error("--user-id (or TV_MCP_PAPER_USER_ID) is required with --auto-paper-trade")

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    log.info("watchlist=%s interval=%ss auto_paper_trade=%s telegram_alerts=%s",
              args.watchlist, args.interval, args.auto_paper_trade, args.telegram_alerts)

    if args.once:
        watchlist = load_watchlist(args.watchlist)
        run_once(watchlist, args)
        return

    while not _STOP:
        try:
            watchlist = load_watchlist(args.watchlist)
            run_once(watchlist, args)
        except Exception as e:
            # A bad scan (missing CSV, malformed watchlist entry, etc.) must
            # not kill the loop — log it and try again next interval.
            log.exception("scan iteration failed: %s", e)

        for _ in range(int(args.interval)):
            if _STOP:
                break
            time.sleep(1)

    log.info("stopped")


if __name__ == "__main__":
    main()

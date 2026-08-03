# Custom + ICT Strategy Tools — CSV Data Edition

**MT5 is no longer used anywhere in this setup.** All backtesting now reads OHLC from CSV files you provide — no live connection, no "must be open" requirement, nothing that can silently hang or time out waiting on an external process.

Adds 5 tools to your local tradingview-mcp fork:
- `list_custom_strategies`, `backtest_custom_strategy`, `optimize_custom_strategy` — generic proxy engine
- `list_ict_strategies`, `backtest_ict_strategy` — the 18-file ICT rulebook engine

## What changed from the MT5 version
- `_fetch_mt5_ohlcv()` is gone, replaced by `_fetch_csv_ohlcv()` — reads a CSV file from disk instead of calling into a running terminal
- Also fixed a real bug found during this change: `run_custom_backtest`/`optimize_custom_strategy` were calling a `_dispatch_engine` function that didn't exist (leftover from an earlier draft) — now correctly call `_run_zone_strategy` directly. Verified working end-to-end.
- No `MetaTrader5` package needed anymore — skip that install step entirely

## CSV file requirements
Place files under `~/Claude/data` (override with the `TV_MCP_CSV_DATA_DIR` environment variable), named like:
```
XAUUSD_15m.csv
XAUUSD_4h.csv
XAUUSD_1d.csv
```
Naming is flexible — `XAUUSD_15m.csv`, `XAUUSD-15m.csv`, `XAUUSD15m.csv`, or MT5-style `XAUUSD_M15.csv` all work, case-insensitive.

**Required columns** (header row required, any order): `date` (or `datetime`/`time`), `open`, `high`, `low`, `close`. `volume` is optional.

**Accepted date formats**: `YYYY-MM-DD HH:MM[:SS]`, `YYYY.MM.DD HH:MM[:SS]` (MT5 export style), or `YYYY-MM-DD` for daily-only files.

**Which timeframes you need depends on the strategy**: check each `.txt` file's `TIMEFRAMES` line — most ICT strategies need at least two files (an HTF one for bias/structure, an LTF one for entries). `list_ict_strategies` won't tell you which files are missing; that only surfaces when you actually call `backtest_ict_strategy` and it errors naming the exact missing file.

## Files in this package
```
custom_strategy_service.py   — parser + both engines, now CSV-based
ict_detectors.py              — shared detection primitives (unchanged)
ict_strategies.py             — per-strategy zone-provider registry (unchanged)
server_additions.py           — the 5 @mcp.tool registrations (docstrings updated for CSV)
strategies/                   — your 19 .txt files + sd_cvd_flow.txt sample
```

## Step 1 — Copy the updated Python module
Your fork already has `ict_detectors.py`, `ict_strategies.py`, and the tool registrations in `server.py` from before — **only `custom_strategy_service.py` needs replacing**. Overwrite:
```
C:\Users\Klaus\tradingview-mcp-custom\src\tradingview_mcp\core\services\custom_strategy_service.py
```
with the version in this package.

## Step 2 — server.py needs NO changes
The tool signatures didn't change (still `symbol`, `period`, `interval`, etc.) — only what happens internally. Your existing `server.py` edits from before are still correct as-is. (`server_additions.py` in this package is provided for reference/docstring text only — you don't need to re-paste anything unless you want the updated parameter descriptions.)

## Step 3 — Place your CSV files
```
mkdir C:\Users\Klaus\Claude\data
```
Drop your CSV files there, one per symbol+timeframe you plan to backtest.

## Step 4 — Restart Claude Desktop, test
No MT5 needed at all now — you can test with MT5 fully closed.
```
List my custom strategies
Backtest sd_cvd_flow with entry=engulfing, sl=swing_based, tp=partial_1_5R on XAUUSD, 15m
Backtest Liq_Sweep_Reversal on XAUUSD, ltf_interval=15m, htf_interval=4h, broker_utc_offset_hours=2
```
If a file is missing, the error message tells you exactly which filename it expected and lists what it actually found in your data folder.

## What was verified before packaging this time
- Created realistic synthetic CSV files (15m/4h/1d, ~3000/750/400 rows) matching the exact format you'd export
- Ran `_fetch_csv_ohlcv()` directly — confirmed correct row count, date parsing, and period-filtering
- Ran the full `run_ict_backtest()` end-to-end on `Liq_Sweep_Reversal` against these real CSVs — produced 34 trades with complete metrics, zero errors
- **Caught and fixed** the `_dispatch_engine` bug in `run_custom_backtest`/`optimize_custom_strategy` by actually running them against real CSVs, not just reading the code — both now confirmed working (backtest produced valid metrics, optimizer completed a 36-combination sweep with walk-forward verdicts)

## What's still not verified
- Your actual CSV export format — I tested against files I generated myself matching the documented spec. If your real export uses a different date format, decimal separator, or column order, the loader may error (with a message showing exactly which columns it found and which it needed) or need a small tweak. Send me the first few lines of a real file if it doesn't load and I'll adjust the parser.
- The ICT detection logic itself is still only tested against synthetic data — see the earlier README's coverage notes (fully vs. parameterized-variation implementations per strategy) for what that does and doesn't mean for trustworthiness.

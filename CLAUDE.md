# tradingview-mcp-custom — Project State

This file is auto-loaded by Claude Code at the start of any session rooted
here. It summarizes the current architecture, what's been built, known gaps,
and where a follow-up session should pick up. A companion PDF
(`TradingView_MCP_Session_Knowledge.pdf`, delivered to the user separately)
covers the ICT/SMC concepts and case studies in more depth with diagrams —
this file is the engineering/state summary.

## Architecture: 3 independent backtest engines, one shared cost/metrics layer

All strategy files are `.txt`. All engines read local CSV OHLCV data only —
no live network/broker calls (config: `TV_MCP_CSV_DATA_DIR`, default
`~/Claude/data`; symbol currently used throughout: XAUUSD, data from
2020-01-02 to present across M1/M5/M15/M30/H1/H4/D1).

1. **Generic proxy** (`custom_strategy_service.py`) — swing-pivot zone
   detector (`_find_swings`), used only when a `.txt` file's `NAME` doesn't
   match the ICT registry.
2. **ICT registry** (`ict_strategies.py` + `ict_detectors.py`) — 26
   hand-coded Python zone providers (18 original + 8 added this session:
   CRT_Turtle_Soup, HTF_POI_Stop_Raid, FVG_Inducement, Full_Confluence,
   Consolidation_Box, IFVG_Reversal), matched by file `NAME` via
   `STRATEGY_REGISTRY`. Files live in `~/Claude/strategies/`.
3. **pine_lite** (`pine_lite/` + `pine_strategy_service.py`) — a
   CONSTRAINED Pine Script subset transpiler (tokenizer → parser →
   validator → evaluator). Files live in `~/Claude/pine_strategies/`
   (config: `TV_MCP_PINE_STRATEGIES_DIR`). NOT a full Pine interpreter —
   no `request.security`, `for`/`while`, `var`/`varip`, arrays, user-defined
   types/methods, native `strategy.entry/exit`. The validator collects
   EVERY unsupported construct before any evaluation is attempted — never
   partial/silent execution. See `pine_lite/__init__.py` for the exact
   supported subset.

Shared foundation: `backtest_service.py`'s `_apply_costs`/`_calc_metrics`/
`_buy_and_hold_return` — same metrics shape from every tool.

**Critical architectural fact**: no engine does fractional/risk-based
position sizing. Equity compounds by each trade's raw % price move — a
"1.5% stop" means the instrument moved 1.5%, not that 1.5% of the account
was risked. Fine for backtest metrics; would need a dedicated risk module
before any live use (see "Next steps" below).

## Key files

| File | Role |
|---|---|
| `server.py` | MCP tool registration (thin delegation, no business logic) |
| `core/services/custom_strategy_service.py` | Generic-proxy engine, ICT dispatch (`_resolve_ict_provider`), CSV loader (`_fetch_csv_ohlcv` — tail-reads only the needed period, doesn't parse whole files) |
| `core/services/ict_strategies.py` | 26 zone-provider functions + `STRATEGY_REGISTRY` |
| `core/services/ict_detectors.py` | Reusable detection primitives: MSS/liquidity-grab, order blocks, daily bias (`classify_daily_bias_v2`, `daily_bias_asof`), weekly profile (`classify_weekly_profile`), IDM (inducement), inverse FVG, HTF POI/stop-raid |
| `core/services/backtest_service.py` | Shared cost/metrics + the 9 built-in indicator strategies (rsi/bollinger/macd/ema_cross/supertrend/donchian/rsi_pullback/keltner_breakout/triple_ema) — these fetch from Yahoo Finance by default, NOT local CSV |
| `core/services/pine_lite/` | tokenizer.py, parser.py, builtins.py, validator.py, evaluator.py |
| `core/services/pine_strategy_service.py` | pine_lite orchestration + `_simulate` (signal-driven, not zone-tap) |
| `core/services/indicators_calc.py` | Pure-stdlib indicator math (SMA/EMA/RSI/MACD/ATR/Bollinger/Donchian/Supertrend/WMA/stdev) |
| `core/portfolio.py` | SQLite paper-trading module — **built but currently unused/unwired**, natural starting point for live portfolio tracking |

## MCP tools available

`backtest_custom_strategy`, `optimize_custom_strategy`, `list_custom_strategies`,
`backtest_ict_strategy`, `list_ict_strategies`,
`backtest_pine_strategy`, `optimize_pine_strategy`, `validate_pine_strategy`, `list_pine_strategies`,
`backtest_strategy`, `compare_strategies`, `walk_forward_backtest_strategy` (the 9 Yahoo-based built-ins).

## Bugs found & fixed this session (see PDF Chapter 02 for full detail)

1. `optimize_custom_strategy`/`backtest_custom_strategy` silently bypassed
   the ICT registry (always ran the generic proxy) — fixed via
   `_resolve_ict_provider`. Every result now reports `zone_source`.
2. CSV loader hung 60s+ on large timeframes (parsed whole file before
   trimming) — fixed with tail-read + single date-format detection.
3. A single ambiguous global trend read zeroed 6 entire ICT strategies —
   fixed by scanning both directions per-candidate instead of one global
   pre-gate.
4. `AMD_Order_Flow`'s ranging-threshold formula was ~10x too tight (divided
   a 14-bar range by 14) — replaced with a real ATR(14) comparison.
5. Timeframe-token parsing: case sensitivity vs. minute/month collision
   (`1M` vs `1m`) — fixed with a single order-preserving regex pass that
   keeps minute tokens case-sensitive, hour/day case-insensitive.
6. `pine_lite`'s `_simulate` could open a position during ATR(14) warmup
   with `sl=None`/`tp=None` — an un-exitable position silently blocking
   ALL further trading for the rest of the backtest. Fixed via `_try_open`,
   which skips a signal rather than opening an un-exitable position.
7. `_VALID_PERIODS` was capped at 2y (leftover from the Yahoo-fetch built-ins'
   original design) even though local CSVs have 6.5 years of data — extended
   to 3y/4y/5y (shared constant, so this applies to all three engines).

## Case studies run this session (full detail + interactive tables in the PDF and 2 published artifacts)

- **AP Hero Flow Pressure** (`03_AP_Hero.txt`, pine_lite): adapted from a
  real LuxAlgo/community indicator. Volume weighting had to be dropped
  (XAUUSD CSVs have no volume column). Daily-bias trend filter
  (`dailyBiasBullish`/`dailyBiasBearish`) beat an EMA-50 filter decisively.
  Walk-forward verdict: only 4h ever showed genuine robustness, and even
  there the durable edge was ~2-5% over 2 years, not the 20-40% headline
  full-period numbers.
- **Adv ICT Structure Lite** (`04_Adv_ICT_Structure_Lite.txt`): simplified
  rewrite of LuxAlgo's "Pure Price Action Structures" (the original has 142
  pine_lite validation issues — full OOP/type system, out of scope).
  20-bar rolling swing breakout + same daily-bias filter. Walk-forward:
  1h was the standout (10/40 combos genuinely ROBUST, best +6.26% return),
  better than either strategy's 4h result.
- **9 built-in strategies on local XAUUSD data**: on 1h (real sample sizes),
  6 of 9 lose significantly; only `ema_cross` (+10.71%, Sharpe 2.10) and
  `triple_ema` (+15.63%, Sharpe 4.49) show real edge — consistent with the
  session-wide pattern that trend-following beats mean-reversion here.

**Cross-cutting lesson**: every strategy that looked strong on a single
full-period backtest failed to fully hold up under walk-forward validation.
Never trust a single fit — `optimize_*_strategy`'s walk-forward check
(`walk_forward_verdict`: ROBUST/MODERATE/OVERFITTED/WEAK) is not optional.

## Next steps discussed (not yet built) — full target architecture

User wants to evolve this into an agent-based live system: feature
engineering, strategy building, live watchlist monitoring (per-minute scan),
alerting, risk management, execution, portfolio management — using ONLY
open source / free resources. This section is the complete plan to resume
from in a fresh session; nothing below is implemented yet.

### Design principle #1: keep the LLM out of the execution hot path

Deterministic scheduled Python for live scanning/risk/execution/portfolio
(fast, repeatable, auditable). Claude/MCP layer stays for the judgment-heavy,
non-time-critical work: strategy design, optimization, walk-forward review,
graduation decisions (backtest → paper → live), reporting. Never call an
LLM per-symbol per-scan for a real-money decision.

### Design principle #2: adopt 3 major open-source projects wholesale, replace only what's genuinely paid

The user asked for the "full features" of QuantLib, Qlib, FinRL, and OpenBB,
with paid parts replaced by our own free logic. Analysis:

| Project | What it provides | License/cost | Plan |
|---|---|---|---|
| **QuantLib** (lballabio/QuantLib, via QuantLib-Python) | Derivatives pricing, yield curve bootstrapping, day-count/calendar conventions, Monte Carlo, Greeks/VaR | 100% free (BSD-3), no paid tier | **Adopt directly as a dependency** — use for the Risk Management module's VaR/Greeks math instead of ad-hoc % rules. Do not reimplement. |
| **Qlib** (microsoft/qlib) | Data handler → alpha/feature engineering → ML model (LightGBM etc.) → portfolio strategy → backtest → analysis; point-in-time data handling (avoids look-ahead bias) | Free (MIT), no paid tier | **Adopt the workflow pattern** — feed Qlib-style ML alpha factors into our existing `indicators_calc.py`/`ict_detectors.py` factor set as additional strategy inputs. |
| **FinRL** (AI4Finance-Foundation/FinRL) | RL trading framework: gym environments, PPO/DDPG/SAC agents via Stable-Baselines3, paper/live via Alpaca/CCXT | Free (MIT), no paid tier | **Wrap our existing bar-by-bar simulation engines as a Gymnasium env**, train RL agents as one more strategy source alongside plain-English/ICT/pine_lite — additive, not a priority, sequence last. |
| **OpenBB** (OpenBB-finance/OpenBB) | Unified data SDK (equities/crypto/forex/macro/news/options/fundamentals) + terminal UI | Core SDK free, but most useful data providers behind it are paid (Polygon.io, Intrinio, Benzinga News, FMP premium) + OpenBB Hub/Terminal Pro cloud subscription | **This is the one to actually rebuild.** See free-substitution table below. |

### OpenBB paid-provider → free substitution table (the actual "avoid cost" work)

| OpenBB paid provider | What it does | Free replacement |
|---|---|---|
| Polygon.io / Intrinio (real-time/deep historical equities & forex) | Live quotes, tick data | Existing MT5 bridge (FX/metals) + `yfinance` (equities, delayed but sufficient) |
| FMP premium / Intrinio (fundamentals) | Financial statements, ratios | **SEC EDGAR** (official, free, full XBRL filings) + `yfinance` fundamentals |
| Benzinga News (paid news/sentiment) | Headlines + sentiment tags | Free RSS + `feedparser` + open-source **FinBERT** (Hugging Face, run locally) for our own sentiment scoring |
| Real-time WebSocket feeds | Streaming quotes | MT5 package (already free, already used) + `ccxt` free public WebSocket endpoints for crypto |
| Macro/economic data vendors | GDP, rates, inflation | **FRED** (St. Louis Fed) — official, fully free, no substitute needed |
| Options chain data (OPRA-derived, paid) | Chains, greeks, OI | Free-tier broker APIs (Tradier sandbox, IBKR paper) for chains; compute Greeks ourselves via **QuantLib** |
| OpenBB Hub / Terminal Pro (cloud UI, sync) | Hosted dashboard | Self-hosted **Streamlit** or **Plotly Dash** (free) — no cloud sync needed for a single-user system |

### Unified layered architecture (top to bottom)

```
RESEARCH & STRATEGY LAYER (Qlib-inspired)
  Data Handler -> Alpha/Feature Engineering -> ML Model (LightGBM etc.)
  -> Portfolio Strategy -> Backtest -> Analysis
  (feeds additional alpha factors into indicators_calc.py/ict_detectors.py)
        |
PRICING & RISK MATH LAYER (QuantLib, adopted directly)
  Yield curves, day-count/calendar conventions, options pricing,
  Monte Carlo VaR, Greeks -> powers the Risk Management module
        |
DATA LAYER (OpenBB pattern, paid providers replaced per table above)
  Unified provider abstraction: MT5 (FX/metals) . yfinance (equities)
  . ccxt (crypto) . FRED (macro) . SEC EDGAR (fundamentals)
  . free RSS/FinBERT (news+sentiment)
        |
EXECUTION & LEARNING LAYER (FinRL-inspired)
  Existing 3 backtest engines wrapped as a Gymnasium env; RL agents
  (PPO/SAC via Stable-Baselines3) as one more strategy source
        |
LIVE AGENT SYSTEM (from the prior session's design, unchanged)
  MT5 live feed -> APScheduler (per-minute watchlist loop)
  -> live signal evaluator (SAME zone-provider/ScriptEnv code as backtest)
  -> Risk engine (now QuantLib-powered: % equity risk sizing, max
     exposure, circuit breakers -- currently the biggest actual gap,
     since no engine in this repo does fractional position sizing today)
  -> Telegram alert (python-telegram-bot, free) + paper/live execution
     (MT5/ccxt)
  -> Portfolio (resurrect core/portfolio.py, SQLite, schema already exists)
```

### Recommended build order

1. **Data abstraction layer** (OpenBB pattern + free providers) — needed by
   everything else, including the risk module.
2. **Risk-management module**, QuantLib-powered — biggest concrete gap
   identified this session (no fractional/risk-based position sizing
   exists anywhere in the codebase today).
3. **Live signal evaluator** — reuses existing zone-provider/`ScriptEnv`
   code verbatim; prove the full pipeline (scan → alert → risk-check →
   paper-fill → portfolio update) via `core/portfolio.py` before any real
   execution.
4. **Qlib-style ML alpha factors** — additive research layer once the
   above is stable.
5. **FinRL-style RL agents** — lowest priority, purely additive strategy
   source, sequence last.

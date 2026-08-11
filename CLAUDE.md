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

**Critical architectural fact**: no *backtest* engine does fractional/
risk-based position sizing. Equity compounds by each trade's raw % price
move — a "1.5% stop" means the instrument moved 1.5%, not that 1.5% of the
account was risked. This is intentionally unchanged (a backtest-metrics
concern) — but a real risk-based sizing/execution path now exists
alongside it for **paper trading**, via `core/services/risk_service.py` and
`core/portfolio.py`'s risk-aware `execute_trade`; see "Live agent system"
below for the full pipeline built this session.

## Key files

| File | Role |
|---|---|
| `server.py` | MCP tool registration (thin delegation, no business logic) — 62 tools total |
| `core/services/custom_strategy_service.py` | Generic-proxy engine, ICT dispatch (`_resolve_ict_provider`), CSV loader (`_fetch_csv_ohlcv` — tail-reads only the needed period, doesn't parse whole files) |
| `core/services/ict_strategies.py` | 26 zone-provider functions + `STRATEGY_REGISTRY` |
| `core/services/ict_detectors.py` | Reusable detection primitives: MSS/liquidity-grab, order blocks, daily bias (`classify_daily_bias_v2`, `daily_bias_asof`), weekly profile (`classify_weekly_profile`), IDM (inducement), inverse FVG, HTF POI/stop-raid |
| `core/services/backtest_service.py` | Shared cost/metrics + the 9 built-in indicator strategies (rsi/bollinger/macd/ema_cross/supertrend/donchian/rsi_pullback/keltner_breakout/triple_ema) — these fetch from Yahoo Finance by default, NOT local CSV |
| `core/services/pine_lite/` | tokenizer.py, parser.py, builtins.py, validator.py, evaluator.py |
| `core/services/pine_strategy_service.py` | pine_lite orchestration + `_simulate` (signal-driven, not zone-tap) |
| `core/services/indicators_calc.py` | Pure-stdlib indicator math (SMA/EMA/RSI/MACD/ATR/Bollinger/Donchian/Supertrend/WMA/stdev) |
| `core/portfolio.py` | SQLite paper-trading module — **now wired** (see "Live agent system" below): risk-aware `execute_trade`, `check_pretrade_risk`, exposed via `execute_paper_trade`/`get_paper_portfolio`/`check_paper_trade_risk` |
| `core/services/risk_service.py` | Position sizing (`calc_position_size`), VaR/CVaR (`calc_var`, own inverse-normal-CDF, no scipy), exposure & circuit-breaker checks, QuantLib-powered option Greeks (`calc_option_greeks`) |
| `core/services/live_signal_service.py` | `evaluate_live_signal` / `scan_watchlist` — reuses `run_ict_backtest`/`run_custom_backtest`/`run_pine_backtest` verbatim to detect a fresh entry on the latest bar; ATR fallback stop (NOT the strategy's real internal SL — not exposed by any trade log) |
| `core/services/alerting_service.py` | Telegram Bot API alerts (`requests`, no `python-telegram-bot` dependency) |
| `scripts/live_scanner.py` | Standalone external scheduler (plain sleep loop, not APScheduler) driving `scan_watchlist` on an interval — NOT an MCP tool, run directly |
| `core/services/data_providers/` | Unified OHLCV facade (`get_ohlcv`, routes to CSV or Yahoo) + `fred_provider.py` (FRED macro) + `edgar_provider.py` (SEC EDGAR fundamentals) |
| `core/services/ml_factor_service.py` | Qlib-style alpha factor research: features from `indicators_calc.py`, LightGBM-or-pure-stdlib-logistic-regression model, IC/hit-rate analysis — research layer only, not wired into entries/exits |
| `core/services/rl_service.py` | FinRL-style long/flat Gymnasium env + PPO-or-pure-stdlib-Q-learning agent — research layer only, not wired into entries/exits |

## MCP tools available (62 total)

**Backtesting**: `backtest_custom_strategy`, `optimize_custom_strategy`, `list_custom_strategies`,
`backtest_ict_strategy`, `list_ict_strategies`,
`backtest_pine_strategy`, `optimize_pine_strategy`, `validate_pine_strategy`, `list_pine_strategies`,
`backtest_strategy`, `compare_strategies`, `walk_forward_backtest_strategy` (the 9 Yahoo-based built-ins).

**Risk management**: `calculate_position_size`, `calculate_portfolio_var`, `check_risk_limits`,
`check_circuit_breaker_status`, `calculate_option_greeks`.

**Paper trading** (first stateful tools in this server): `execute_paper_trade`, `get_paper_portfolio`,
`check_paper_trade_risk`.

**Live signal evaluator**: `evaluate_live_signal`, `scan_watchlist`, `send_telegram_alert`.

**Data abstraction layer**: `get_ohlcv`, `get_macro_series` (FRED), `get_company_fundamentals` (SEC EDGAR).

**ML/RL research layer**: `run_alpha_factor_analysis`, `train_rl_trading_agent`.

Plus the ~30 pre-existing screener/market-data tools (top gainers/losers, EGX, futures, options, news,
sentiment, Yahoo prices) unrelated to this session's backtest/live-agent work.

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

## Live agent system — built this session, all 5 build-order steps done

User wanted this evolved into an agent-based live system: feature
engineering, strategy building, live watchlist monitoring, alerting, risk
management, execution, portfolio management — open source / free resources
only. All 5 steps of the recommended build order below are now built and
verified against real XAUUSD CSV data. **Everything is local and
uncommitted** on branch `feat/ict-pine-lite-engines` — the user explicitly
said not to push/pull GitHub, work stays local.

1. **Data abstraction layer** — `get_ohlcv` (routes to existing CSV/Yahoo
   fetchers), `get_macro_series` (FRED), `get_company_fundamentals` (SEC
   EDGAR). Crypto/`ccxt` and RSS+FinBERT were *not* rebuilt — the
   pre-existing `tradingview-screener`-backed crypto tools and
   `marketaux_service.py`/`news_service.py` already cover that ground, so
   building `ccxt`/FinBERT from scratch would have been redundant.
2. **Risk-management module** (`risk_service.py`) — QuantLib installs
   cleanly via pip on this Windows box and powers `calc_option_greeks`;
   everything else (position sizing, VaR, exposure/circuit-breaker checks)
   is pure stdlib and works without QuantLib. Wired into `core/portfolio.py`
   so `execute_trade` can risk-size a BUY and reject it against exposure/
   circuit-breaker limits — the fractional-sizing gap flagged in the
   "Critical architectural fact" above is now closed **for paper trading**;
   the three backtest engines themselves are unchanged (still raw %
   price-move compounding, by design — that's a backtest-metrics concern,
   not a live-execution one).
3. **Live signal evaluator + paper trading + alerting + scheduler** — full
   pipeline proven end-to-end: `scan_watchlist` reuses
   `run_ict_backtest`/`run_custom_backtest`/`run_pine_backtest` verbatim to
   detect a fresh entry on the latest bar, sizes it via `risk_service`,
   executes through the risk-aware `execute_paper_trade`, and can alert via
   Telegram (`alerting_service.py`, plain `requests`, no
   `python-telegram-bot` dependency). `scripts/live_scanner.py` is the
   external scheduler — a dependency-free sleep loop, not APScheduler,
   driving `scan_watchlist` on an interval; verified with `--once` against
   real data and against a malformed watchlist entry (logs and continues,
   doesn't crash). **Caveat kept honest, not papered over**: no backtest
   engine's trade log exposes the strategy's actual internal stop-loss
   price, so `evaluate_live_signal` computes a clearly-labeled ATR(14)
   fallback stop instead — not a reproduction of the real rule.
4. **Qlib-style ML alpha factors** (`ml_factor_service.py`) — features from
   `indicators_calc.py` (RSI/MACD/Bollinger %B/ATR%/EMA distance/returns/
   volatility), chronological (no-shuffle) train/test split, IC + hit-rate
   analysis. **LightGBM's Windows wheel installs and imports fine but its
   compiled `lib_lightgbm.dll` fails at `LoadLibrary`** (missing MSVC
   runtime dependency chain) — this is a real, observed environment
   failure, not hypothetical. The module tries LightGBM per-call and falls
   back to a pure-stdlib logistic regression (plain gradient descent, no
   numpy) when it fails, reporting `model_used` either way.
5. **FinRL-style RL agent** (`rl_service.py`) — a Gymnasium long/flat env
   (`TradingEnv`, no shorting — matches `core/portfolio.py`) wrapping the
   same feature pipeline as step 4. `stable-baselines3`/`torch` were **not
   installed this session** (multi-hundred-MB dependency chain, judged not
   worth forcing given the LightGBM DLL precedent) — PPO is attempted and
   falls back to a pure-stdlib tabular Q-learning agent (4 of 10 features
   discretized into tertiles) when unavailable. The Q-learning path is
   what's actually been run and verified.

Both ML/RL tools are explicitly **research-layer only** — outputs are not
wired into any backtest engine's entries/exits, and both disclaimers apply
the same single-fold skepticism CLAUDE.md's case-study section applies
everywhere else (a promising `information_coefficient`/RL result on one
chronological split is not proof of an edge — see "Cross-cutting lesson"
above).

### Environment notes for a fresh session

- **QuantLib**: works. `pip install QuantLib` (or `pip install ".[risk]"`).
- **LightGBM**: `pip install lightgbm` succeeds and `import lightgbm`
  succeeds, but constructing a `Booster` throws `FileNotFoundError` loading
  `lib_lightgbm.dll` — needs the MSVC redistributable installed on this
  machine to actually use it; `ml_factor_service.py` degrades gracefully
  without that.
- **stable-baselines3 / torch**: not installed, not attempted. `pip install
  ".[rl-full]"` to add it — `rl_service.py`'s PPO path is otherwise
  untested (only the qlearning fallback has been run for real).
- **Telegram / FRED / SEC EDGAR**: all need env vars
  (`TV_MCP_TELEGRAM_BOT_TOKEN`/`TV_MCP_TELEGRAM_CHAT_ID`, `FRED_API_KEY`,
  `SEC_EDGAR_USER_AGENT`) not set in this session — SEC EDGAR was still
  live-tested successfully by passing `user_agent` directly to the
  function (bypassing the env var) since it needs no API key, only a
  descriptive UA string.

### Design principles (still governing any further work here)

#### Design principle #1: keep the LLM out of the execution hot path

Deterministic scheduled Python for live scanning/risk/execution/portfolio
(fast, repeatable, auditable). Claude/MCP layer stays for the judgment-heavy,
non-time-critical work: strategy design, optimization, walk-forward review,
graduation decisions (backtest → paper → live), reporting. Never call an
LLM per-symbol per-scan for a real-money decision.

#### Design principle #2: adopt 3 major open-source projects wholesale, replace only what's genuinely paid

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

### Unified layered architecture (top to bottom) — as planned vs. as built

```
RESEARCH & STRATEGY LAYER (Qlib-inspired)               [BUILT: ml_factor_service.py]
  Data Handler -> Alpha/Feature Engineering -> ML Model (LightGBM or
  pure-stdlib logistic fallback) -> Analysis (IC + hit-rate)
  (feeds indicators_calc.py's factor set — Portfolio Strategy/Backtest
  integration NOT built: research-layer output only)
        |
PRICING & RISK MATH LAYER (QuantLib, adopted directly)   [BUILT: risk_service.py]
  calc_option_greeks via QuantLib; position sizing/VaR/exposure/
  circuit-breaker checks are pure stdlib -> powers core/portfolio.py
        |
DATA LAYER (OpenBB pattern, paid providers replaced)     [BUILT: data_providers/]
  Unified provider abstraction: MT5-exported CSV / yahoo (equities)
  . FRED (macro) . SEC EDGAR (fundamentals)
  (crypto and RSS/sentiment NOT rebuilt — pre-existing
  tradingview-screener + marketaux_service.py/news_service.py already
  cover that ground)
        |
EXECUTION & LEARNING LAYER (FinRL-inspired)              [BUILT: rl_service.py]
  3 backtest engines' feature pipeline wrapped as a Gymnasium long/flat
  env; PPO (untested, stable-baselines3 not installed) or pure-stdlib
  Q-learning (verified) as one more strategy source
        |
LIVE AGENT SYSTEM                                        [BUILT]
  Local CSV (as fresh as the last MT5 export) -> scripts/live_scanner.py
  (plain sleep loop, external to MCP) -> scan_watchlist (SAME
  run_ict_backtest/run_custom_backtest/run_pine_backtest code as backtest)
  -> risk_service (% equity risk sizing, exposure caps, circuit breakers)
  -> Telegram alert (alerting_service.py, plain requests) + paper
     execution (execute_paper_trade)
  -> Portfolio (core/portfolio.py, SQLite, now wired and risk-aware)
```

### Recommended build order — all 5 steps complete

1. **Data abstraction layer** — done (`data_providers/`).
2. **Risk-management module** — done (`risk_service.py`, wired into
   `core/portfolio.py`). This was the concrete gap flagged at the top of
   this file; it's now closed for paper trading (the 3 backtest engines
   are intentionally unchanged — that's a metrics concern, not this fix's
   scope).
3. **Live signal evaluator** — done, full pipeline (scan → alert →
   risk-check → paper-fill → portfolio update) proven end-to-end against
   real XAUUSD CSV data and via `scripts/live_scanner.py`.
4. **Qlib-style ML alpha factors** — done (`ml_factor_service.py`),
   research layer only.
5. **FinRL-style RL agents** — done (`rl_service.py`), research layer
   only, Q-learning path verified / PPO path untested (dependency not
   installed).

### What's still open for a future session

- **Real broker/live execution** — everything above is CSV-driven paper
  trading; no live MT5/broker socket connection exists anywhere in this
  codebase (by design, matching the "no live network/broker calls" fact
  at the top of this file).
- **ccxt / real-time crypto feeds, RSS+FinBERT sentiment** — deliberately
  not rebuilt (see data-layer note above); revisit only if the existing
  `tradingview-screener`/`marketaux_service.py` coverage proves
  insufficient.
- **stable-baselines3/torch** — install with `pip install ".[rl-full]"` to
  actually exercise `rl_service.py`'s PPO path.
- **LightGBM on this machine** — needs the MSVC redistributable installed
  for `ml_factor_service.py` to use it over the logistic fallback.
- **Wiring ML/RL research output into an actual strategy** — both are
  intentionally research-only right now (no entries/exits driven by
  `run_alpha_factor_analysis`/`train_rl_trading_agent`); doing so would be
  a deliberate next step, not an oversight.
- **OpenBB Hub-style dashboard, options-chain sandbox integration** — from
  the original substitution table, not attempted this session (lowest
  priority items on that table).

"""
pine_lite — a constrained Pine Script (v5-ish) subset transpiler.

NOT a general Pine Script interpreter. Supports:
  - variable assignment: `ema20 = ta.ema(close, 20)`
  - multiple-assignment for ta.macd only: `[macdLine, sig, hist] = ta.macd(close, 12, 26, 9)`
  - single-expression function definitions: `f(a, b) => a + b`
  - arithmetic, comparison, logical and/or/not, ternary `cond ? a : b`
  - historical reference: `close[1]`
  - a whitelisted set of ta.*/math.* built-ins (see builtins.py), reusing
    core.services.indicators_calc.py's existing indicator math
  - common display/declaration calls (plot, indicator, strategy, bgcolor,
    ...) recognized and no-op'd, since real copy-pasted scripts include them

Explicitly NOT supported (flagged by validator.py with a clear message,
never silently ignored or partially executed): request.security (multi-
timeframe), for/while loops, var/varip persistent state, arrays/matrices,
user-defined types, native strategy.entry/exit (use this server's own
[Buy Condition]/[SL]/[TP] blocks instead), script inputs (input.*).

Pipeline: tokenizer -> parser -> validator (must return zero issues before
any evaluation is attempted) -> evaluator. See pine_strategy_service.py for
the orchestration layer that ties this to file parsing and backtesting.
"""

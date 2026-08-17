"""
Qlib-style alpha-factor research: Data Handler -> Feature Engineering ->
ML Model -> Analysis (the research-layer slice of CLAUDE.md's architecture;
Portfolio Strategy/Backtest integration is a follow-up, not built here).

Feature engineering reuses indicators_calc.py (RSI/MACD/Bollinger/ATR/EMA) —
same "Qlib-style factors feed the existing indicator set" plan, not a
separate factor library.

Model: LightGBM if it actually loads, else a pure-stdlib logistic
regression fallback (gradient descent, no numpy). Both paths were built
because LightGBM's Windows wheel imported fine in this dev environment but
its compiled `lib_lightgbm.dll` failed to load at runtime (missing MSVC
runtime dependency chain — a known Windows wheel issue, not something pip
can fix). Rather than block this feature on that, the module tries
LightGBM lazily per-call and transparently drops to the logistic model,
reporting which one actually ran in the output. Import failure alone
would only need a try/except ImportError; here the failure surfaces at
first *use* (constructing a Booster), so the try/except wraps the actual
training call, not the import.

Point-in-time discipline: features at bar i use only data up to and
including bar i; labels are the forward return from i to i+horizon. The
train/test split is chronological (no shuffling) so the test set is always
strictly after the training set — the same anti-lookahead discipline
Qlib's point-in-time data handling is designed to enforce.
"""
from __future__ import annotations

import math
import random
import statistics
from typing import Any, Literal, Optional

from tradingview_mcp.core.services.indicators_calc import (
    calc_rsi, calc_macd, calc_bollinger, calc_atr, calc_ema,
)
from tradingview_mcp.core.services.data_providers import get_ohlcv
from tradingview_mcp.core.errors import ErrorCode, make_error

_FEATURE_NAMES = [
    "rsi14", "macd_hist", "bb_pct_b", "atr_pct",
    "ema20_dist_pct", "ema50_dist_pct", "ret_5", "ret_10", "ret_20", "volatility_20",
]


def _build_features_and_labels(candles: list[dict], horizon: int) -> dict[str, Any]:
    closes = [c["close"] for c in candles]
    highs = [c["high"] for c in candles]
    lows = [c["low"] for c in candles]
    n = len(closes)

    rsi = calc_rsi(closes, 14)
    macd = calc_macd(closes)
    bb = calc_bollinger(closes, 20, 2.0)
    atr = calc_atr(highs, lows, closes, 14)
    ema20 = calc_ema(closes, 20)
    ema50 = calc_ema(closes, 50)

    rows: list[list[float]] = []
    labels_reg: list[float] = []
    labels_cls: list[int] = []
    dates: list[str] = []

    # i needs 50 bars of warmup behind it (ema50) and `horizon` bars ahead
    # of it for the label — both bounds keep every row point-in-time-valid.
    for i in range(50, n - horizon):
        if None in (rsi[i], macd["histogram"][i], bb["upper"][i], bb["lower"][i],
                     atr[i], ema20[i], ema50[i]):
            continue
        bb_range = bb["upper"][i] - bb["lower"][i]
        bb_pct_b = (closes[i] - bb["lower"][i]) / bb_range if bb_range > 0 else 0.5
        ret_5 = (closes[i] - closes[i - 5]) / closes[i - 5] if i >= 5 else 0.0
        ret_10 = (closes[i] - closes[i - 10]) / closes[i - 10] if i >= 10 else 0.0
        ret_20 = (closes[i] - closes[i - 20]) / closes[i - 20] if i >= 20 else 0.0
        window_closes = closes[i - 20:i + 1]
        vol_20 = statistics.stdev(window_closes) / statistics.mean(window_closes) if len(window_closes) > 1 else 0.0

        rows.append([
            rsi[i],
            macd["histogram"][i],
            bb_pct_b,
            atr[i] / closes[i] * 100,
            (closes[i] - ema20[i]) / ema20[i] * 100,
            (closes[i] - ema50[i]) / ema50[i] * 100,
            ret_5 * 100, ret_10 * 100, ret_20 * 100,
            vol_20 * 100,
        ])
        fwd_ret = (closes[i + horizon] - closes[i]) / closes[i] * 100
        labels_reg.append(fwd_ret)
        labels_cls.append(1 if fwd_ret > 0 else 0)
        dates.append(candles[i]["date"])

    return {"X": rows, "y_reg": labels_reg, "y_cls": labels_cls, "dates": dates}


def _standardize(train_X: list[list[float]], apply_X: list[list[float]]) -> list[list[float]]:
    n_feat = len(train_X[0])
    means = [statistics.mean(r[j] for r in train_X) for j in range(n_feat)]
    stds = [statistics.pstdev(r[j] for r in train_X) or 1.0 for j in range(n_feat)]
    return [[(r[j] - means[j]) / stds[j] for j in range(n_feat)] for r in apply_X]


class _LogisticRegression:
    """Plain-gradient-descent binary logistic regression — no numpy.
    Fallback model when LightGBM isn't usable in the current environment."""

    def __init__(self, n_features: int, lr: float = 0.1, l2: float = 0.001):
        self.weights = [0.0] * n_features
        self.bias = 0.0
        self.lr = lr
        self.l2 = l2

    def fit(self, X: list[list[float]], y: list[int], epochs: int = 300) -> None:
        n = len(X)
        for _ in range(epochs):
            grad_w = [0.0] * len(self.weights)
            grad_b = 0.0
            for xi, yi in zip(X, y):
                z = self.bias + sum(w * x for w, x in zip(self.weights, xi))
                pred = 1.0 / (1.0 + math.exp(-max(-60, min(60, z))))
                err = pred - yi
                for j, x in enumerate(xi):
                    grad_w[j] += err * x
                grad_b += err
            for j in range(len(self.weights)):
                self.weights[j] -= self.lr * (grad_w[j] / n + self.l2 * self.weights[j])
            self.bias -= self.lr * (grad_b / n)

    def predict_proba(self, X: list[list[float]]) -> list[float]:
        out = []
        for xi in X:
            z = self.bias + sum(w * x for w, x in zip(self.weights, xi))
            out.append(1.0 / (1.0 + math.exp(-max(-60, min(60, z)))))
        return out

    def feature_importances(self) -> list[float]:
        return [abs(w) for w in self.weights]


def _try_lightgbm(train_X, train_y, test_X, feature_names) -> Optional[dict[str, Any]]:
    try:
        import lightgbm as lgb
        train_set = lgb.Dataset(train_X, label=train_y, feature_name=feature_names)
        booster = lgb.train(
            {"objective": "binary", "verbosity": -1, "num_leaves": 15, "min_data_in_leaf": 10},
            train_set, num_boost_round=100,
        )
        proba = list(booster.predict(test_X))
        importances = list(booster.feature_importance(importance_type="gain"))
        return {"proba": proba, "importances": [float(v) for v in importances], "model": "lightgbm"}
    except Exception:
        return None


def _spearman_ic(predicted: list[float], actual: list[float]) -> float:
    """Information Coefficient — Spearman rank correlation between
    predicted score and realized forward return, computed via stdlib rank
    + Pearson correlation on the ranks (no scipy)."""
    def rank(vals: list[float]) -> list[float]:
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        ranks = [0.0] * len(vals)
        for pos, idx in enumerate(order):
            ranks[idx] = pos
        return ranks

    if len(predicted) < 3:
        return 0.0
    rp, ra = rank(predicted), rank(actual)
    try:
        return round(statistics.correlation(rp, ra), 4)
    except (statistics.StatisticsError, ZeroDivisionError):
        return 0.0


def run_alpha_factor_analysis(
    symbol: str,
    source: Literal["csv", "yahoo"] = "yahoo",
    period: str = "2y",
    interval: str = "1d",
    horizon: int = 5,
    train_frac: float = 0.7,
    model: Literal["auto", "lightgbm", "logistic"] = "auto",
) -> dict[str, Any]:
    """Train a directional alpha model on engineered technical factors and
    report its out-of-sample predictive skill — Qlib's
    Data Handler -> Feature Engineering -> Model -> Analysis workflow,
    reusing indicators_calc.py for the factor set.

    Args:
        symbol: instrument symbol (see get_ohlcv for source-specific format).
        source: 'csv' (local MT5-exported files) or 'yahoo' (Yahoo Finance).
        period / interval: how much history to fetch and at what granularity.
        horizon: bars ahead the label predicts (forward % return).
        train_frac: fraction of bars (chronological, not shuffled) used for
            training; the rest is the out-of-sample test set.
        model: 'lightgbm' (falls back to logistic if it fails to load in
            this environment), 'logistic' (pure stdlib, always available),
            or 'auto' (try lightgbm, silently fall back).

    Returns:
        model_used, n_train/n_test, hit_rate_pct (test-set directional
        accuracy), information_coefficient (Spearman rank correlation
        between predicted score and realized forward return — near 0 means
        no edge, |IC| > ~0.03-0.05 is considered notable in equity alpha
        research), and feature_importances.
    """
    if not (0.3 <= train_frac <= 0.9):
        return make_error(ErrorCode.INVALID_PARAMETER, "train_frac must be between 0.3 and 0.9")
    if horizon < 1:
        return make_error(ErrorCode.INVALID_PARAMETER, "horizon must be >= 1")

    fetch = get_ohlcv(symbol, source, period, interval)
    if "error" in fetch:
        return fetch
    candles = fetch["candles"]

    data = _build_features_and_labels(candles, horizon)
    X, y_cls, y_reg, dates = data["X"], data["y_cls"], data["y_reg"], data["dates"]
    if len(X) < 60:
        return make_error(
            ErrorCode.NO_DATA,
            f"only {len(X)} usable rows after feature/label warmup — need at least 60; try a longer period",
        )

    split = int(len(X) * train_frac)
    train_X_raw, test_X_raw = X[:split], X[split:]
    train_y_cls, test_y_cls = y_cls[:split], y_cls[split:]
    test_y_reg = y_reg[split:]

    train_X = _standardize(train_X_raw, train_X_raw)
    test_X = _standardize(train_X_raw, test_X_raw)

    used_model = None
    proba = None
    importances = None

    if model in ("auto", "lightgbm"):
        result = _try_lightgbm(train_X, train_y_cls, test_X, _FEATURE_NAMES)
        if result is not None:
            proba, importances, used_model = result["proba"], result["importances"], result["model"]
        elif model == "lightgbm":
            return make_error(
                ErrorCode.DEPENDENCY_MISSING,
                "LightGBM is installed but failed to load (common on Windows without the "
                "MSVC runtime it was compiled against) — try model='logistic' instead.",
            )

    if used_model is None:
        clf = _LogisticRegression(len(_FEATURE_NAMES))
        clf.fit(train_X, train_y_cls)
        proba = clf.predict_proba(test_X)
        importances = clf.feature_importances()
        used_model = "logistic"

    predicted_direction = [1 if p >= 0.5 else 0 for p in proba]
    hits = sum(1 for p, a in zip(predicted_direction, test_y_cls) if p == a)
    hit_rate = round(hits / len(test_y_cls) * 100, 2)
    ic = _spearman_ic(proba, test_y_reg)

    importance_ranked = sorted(
        zip(_FEATURE_NAMES, importances), key=lambda t: t[1], reverse=True
    )

    return {
        "symbol": fetch["symbol"], "source": source, "horizon_bars": horizon,
        "model_used": used_model,
        "n_train": len(train_X), "n_test": len(test_X),
        "hit_rate_pct": hit_rate,
        "baseline_hit_rate_pct": round(sum(test_y_cls) / len(test_y_cls) * 100, 2),
        "information_coefficient": ic,
        "feature_importances": [{"feature": f, "importance": round(v, 4)} for f, v in importance_ranked],
        "test_period": {"from": dates[split] if split < len(dates) else None, "to": dates[-1]},
        "disclaimer": (
            "Diagnostic view of the classifier — for the same model driving real trades, see "
            "backtest_strategy/walk_forward_backtest_strategy with strategy='ml_alpha'. "
            "hit_rate_pct above baseline_hit_rate_pct (the test set's actual up-move frequency) "
            "and a non-trivial |information_coefficient| are the signals to look for; a hit rate "
            "near baseline or IC near 0 means this feature set has no edge at this horizon for "
            "this symbol/period. Chronological (non-shuffled) train/test split; no walk-forward "
            "re-training across multiple windows yet — treat like a single-fold backtest, i.e. "
            "with the same skepticism CLAUDE.md's session notes apply to any single-fit result."
        ),
    }

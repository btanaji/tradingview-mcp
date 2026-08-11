"""
FinRL-style RL trading agent — lowest-priority, purely additive strategy
source per CLAUDE.md's build order (sequenced last on purpose).

Wraps the same OHLCV + indicators_calc.py feature pipeline as
ml_factor_service.py into a Gymnasium environment (long/flat, no shorting —
matching core/portfolio.py's lack of short-selling support), rather than
building a separate FinRL-style env stack. Reward is the chosen position's
realized 1-bar return, penalized on every position flip.

Model: Stable-Baselines3 PPO if it's importable AND actually trains
without error, else a pure-stdlib tabular Q-learning fallback (dict-based
Q-table over discretized state, no numpy/torch). This project did not
attempt to install stable-baselines3/torch in this session — it's a
multi-hundred-MB dependency chain and, per the lightgbm/QuantLib precedent
in this codebase, the right call is graceful degradation rather than
blocking the feature on a heavy install succeeding in every environment.
The Q-learning path is what's actually been run and verified here.

Research-layer / additive strategy source only, same caveat as
ml_factor_service.py: not wired into any backtest engine's entries/exits.
"""
from __future__ import annotations

import random
import statistics
from typing import Any, Literal, Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from tradingview_mcp.core.services.ml_factor_service import _build_features_and_labels, _FEATURE_NAMES
from tradingview_mcp.core.services.data_providers import get_ohlcv
from tradingview_mcp.core.errors import ErrorCode, make_error

_TRANSACTION_COST_PCT = 0.05  # per position flip, in the same % units as the reward


class TradingEnv(gym.Env):
    """Long/flat trading environment over a fixed sequence of (features,
    next-bar-return) pairs. Action 0 = flat, 1 = long. No short-selling
    (matches core/portfolio.py). One episode = one full pass through the
    provided rows."""

    metadata = {"render_modes": []}

    def __init__(self, features: list[list[float]], step_returns_pct: list[float]):
        super().__init__()
        assert len(features) == len(step_returns_pct)
        self.features = np.array(features, dtype=np.float32)
        self.step_returns_pct = step_returns_pct
        self.n = len(features)
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            low=-50.0, high=50.0, shape=(self.features.shape[1],), dtype=np.float32
        )
        self._i = 0
        self._position = 0

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self._i = 0
        self._position = 0
        return self.features[0], {}

    def step(self, action: int):
        reward = self.step_returns_pct[self._i] if action == 1 else 0.0
        if action != self._position:
            reward -= _TRANSACTION_COST_PCT
        self._position = action
        self._i += 1
        terminated = self._i >= self.n - 1
        obs = self.features[min(self._i, self.n - 1)]
        return obs, reward, terminated, False, {}


class _QLearningAgent:
    """Tabular Q-learning over a small discretized state space — the
    pure-stdlib fallback model. No numpy/torch needed for the Q-table
    itself (dict keyed by discretized state tuples)."""

    def __init__(self, actions: tuple[int, ...] = (0, 1), lr: float = 0.1,
                 gamma: float = 0.9, epsilon: float = 0.15):
        self.q: dict[tuple, dict[int, float]] = {}
        self.actions = actions
        self.lr, self.gamma, self.epsilon = lr, gamma, epsilon

    def _row(self, state: tuple) -> dict[int, float]:
        return self.q.setdefault(state, {a: 0.0 for a in self.actions})

    def act(self, state: tuple, greedy: bool = False) -> int:
        qs = self._row(state)
        if not greedy and random.random() < self.epsilon:
            return random.choice(self.actions)
        return max(qs, key=qs.get)

    def update(self, state: tuple, action: int, reward: float, next_state: tuple) -> None:
        qs = self._row(state)
        next_max = max(self._row(next_state).values())
        qs[action] += self.lr * (reward + self.gamma * next_max - qs[action])


# Only the 4 most informative features (per ml_factor_service's importance
# ranking pattern) are discretized for Q-learning — using all 10 would blow
# up the state space (3^10 vs 3^4) for a model with no function
# approximation to generalize across nearby states.
_QL_FEATURE_IDX = [0, 1, 2, 3]  # rsi14, macd_hist, bb_pct_b, atr_pct


def _tertile_edges(train_X: list[list[float]], idx: int) -> tuple[float, float]:
    vals = sorted(row[idx] for row in train_X)
    n = len(vals)
    return vals[n // 3], vals[2 * n // 3]


def _discretize(row: list[float], edges: list[tuple[float, float]]) -> tuple:
    key = []
    for pos, j in enumerate(_QL_FEATURE_IDX):
        lo, hi = edges[pos]
        v = row[j]
        key.append(0 if v < lo else (1 if v < hi else 2))
    return tuple(key)


def _train_qlearning(train_X, train_returns, episodes: int) -> tuple[_QLearningAgent, list[tuple]]:
    edges = [_tertile_edges(train_X, j) for j in _QL_FEATURE_IDX]
    states = [_discretize(row, edges) for row in train_X]
    agent = _QLearningAgent()
    n = len(states)
    for _ in range(episodes):
        position = 0
        for i in range(n - 1):
            action = agent.act(states[i])
            reward = train_returns[i] if action == 1 else 0.0
            if action != position:
                reward -= _TRANSACTION_COST_PCT
            position = action
            agent.update(states[i], action, reward, states[i + 1])
    return agent, edges


def _try_ppo(env: TradingEnv, total_timesteps: int):
    try:
        from stable_baselines3 import PPO
        model = PPO("MlpPolicy", env, verbose=0)
        model.learn(total_timesteps=total_timesteps)
        return model
    except Exception:
        return None


def train_rl_trading_agent(
    symbol: str,
    source: Literal["csv", "yahoo"] = "yahoo",
    period: str = "2y",
    interval: str = "1d",
    train_frac: float = 0.7,
    episodes: int = 200,
    algo: Literal["auto", "ppo", "qlearning"] = "auto",
) -> dict[str, Any]:
    """Train a long/flat RL trading agent on engineered technical features
    and evaluate it out-of-sample against buy-and-hold — the FinRL-inspired
    additive strategy source. No short-selling (matches the paper
    portfolio). Research-layer output only, not wired into any backtest
    engine's entries/exits.

    Args:
        symbol: instrument symbol (see get_ohlcv for source-specific format).
        source: 'csv' (local MT5-exported files) or 'yahoo' (Yahoo Finance).
        period / interval: how much history to fetch and at what granularity.
        train_frac: fraction of bars (chronological, not shuffled) used for
            training; the rest is the out-of-sample evaluation window.
        episodes: training passes over the training data (Q-learning) or a
            timesteps multiplier (PPO: episodes * n_train_bars timesteps).
        algo: 'ppo' (Stable-Baselines3; falls back to qlearning if the
            library isn't installed or fails to train), 'qlearning' (pure
            stdlib, always available), or 'auto' (try ppo, silently fall back).
    """
    if not (0.3 <= train_frac <= 0.9):
        return make_error(ErrorCode.INVALID_PARAMETER, "train_frac must be between 0.3 and 0.9")
    if episodes < 1:
        return make_error(ErrorCode.INVALID_PARAMETER, "episodes must be >= 1")

    fetch = get_ohlcv(symbol, source, period, interval)
    if "error" in fetch:
        return fetch
    candles = fetch["candles"]

    data = _build_features_and_labels(candles, horizon=1)
    X, y_reg, dates = data["X"], data["y_reg"], data["dates"]
    if len(X) < 80:
        return make_error(
            ErrorCode.NO_DATA,
            f"only {len(X)} usable rows after feature warmup — need at least 80; try a longer period",
        )

    split = int(len(X) * train_frac)
    train_X, test_X = X[:split], X[split:]
    train_returns, test_returns = y_reg[:split], y_reg[split:]

    used_algo = None
    agent = edges = None
    ppo_model = None

    if algo in ("auto", "ppo"):
        env = TradingEnv(train_X, train_returns)
        ppo_model = _try_ppo(env, total_timesteps=episodes * len(train_X))
        if ppo_model is not None:
            used_algo = "ppo"
        elif algo == "ppo":
            return make_error(
                ErrorCode.DEPENDENCY_MISSING,
                "stable-baselines3 is not installed (or failed to train) in this environment — "
                "try algo='qlearning' instead. pip install stable-baselines3 to enable PPO.",
            )

    if used_algo is None:
        agent, edges = _train_qlearning(train_X, train_returns, episodes)
        used_algo = "qlearning"

    # ─ Out-of-sample evaluation, either policy ─
    actions_taken: list[int] = []
    if used_algo == "ppo":
        test_env = TradingEnv(test_X, test_returns)
        obs, _ = test_env.reset()
        for _ in range(len(test_X) - 1):
            action, _ = ppo_model.predict(obs, deterministic=True)
            actions_taken.append(int(action))
            obs, _, terminated, _, _ = test_env.step(int(action))
            if terminated:
                break
        actions_taken.append(actions_taken[-1] if actions_taken else 0)
    else:
        test_states = [_discretize(row, edges) for row in test_X]
        actions_taken = [agent.act(s, greedy=True) for s in test_states]

    equity = 100.0
    peak = equity
    max_dd = 0.0
    position = 0
    returns_series = []
    for action, ret in zip(actions_taken, test_returns):
        r = ret if action == 1 else 0.0
        if action != position:
            r -= _TRANSACTION_COST_PCT
        position = action
        equity *= (1 + r / 100)
        peak = max(peak, equity)
        max_dd = max(max_dd, (peak - equity) / peak * 100)
        returns_series.append(r)

    total_return_pct = round(equity - 100.0, 2)
    bnh_return_pct = round(sum(test_returns), 2)  # buy-and-hold ≈ sum of per-bar % returns held throughout
    sharpe = 0.0
    if len(returns_series) > 1 and statistics.pstdev(returns_series) > 0:
        sharpe = round(statistics.mean(returns_series) / statistics.pstdev(returns_series) * (252 ** 0.5), 2)
    time_in_market_pct = round(sum(actions_taken) / len(actions_taken) * 100, 2) if actions_taken else 0.0

    return {
        "symbol": fetch["symbol"], "source": source,
        "algo_used": used_algo,
        "n_train": len(train_X), "n_test": len(test_X),
        "total_return_pct": total_return_pct,
        "buy_and_hold_return_pct": bnh_return_pct,
        "vs_buy_and_hold_pct": round(total_return_pct - bnh_return_pct, 2),
        "max_drawdown_pct": round(max_dd, 2),
        "sharpe_ratio": sharpe,
        "time_in_market_pct": time_in_market_pct,
        "test_period": {"from": dates[split] if split < len(dates) else None, "to": dates[-1]},
        "disclaimer": (
            "Research-layer output only — not wired into any backtest engine's entries/exits. "
            "One chronological train/test split, no walk-forward re-training across multiple "
            "windows — treat with the same single-fold skepticism CLAUDE.md applies everywhere "
            "else. The qlearning fallback discretizes only 4 of the 10 available features into "
            "3 bins each (state-space size constraint for a tabular method with no function "
            "approximation) — it is a much cruder policy than PPO would be if stable-baselines3 "
            "were available."
        ),
    }

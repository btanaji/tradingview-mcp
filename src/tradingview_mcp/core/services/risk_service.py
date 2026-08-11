"""
Risk management: position sizing, portfolio VaR, exposure/circuit-breaker
limits, and option Greeks.

Fills the gap noted across all three backtest engines: none of them size
positions by risk — equity compounds by each trade's raw % price move, not
by how much of the account was actually put at stake. Nothing here changes
that (backtests stay as they are); this module is what a live/paper
execution path would call *before* sending an order.

Only `calc_option_greeks` needs QuantLib (real Black-Scholes machinery is
not worth reimplementing). Everything else — position sizing, VaR,
exposure/circuit-breaker checks — is pure stdlib, consistent with
indicators_calc.py, so the module works even without QuantLib installed.
"""
from __future__ import annotations

import math
import statistics
from typing import Any, Literal

from tradingview_mcp.core.errors import ErrorCode, make_error

try:
    import QuantLib as ql
    QUANTLIB_AVAILABLE = True
except ImportError:
    QUANTLIB_AVAILABLE = False


# ─── Position sizing ───────────────────────────────────────────────────────

def calc_position_size(
    account_equity: float,
    risk_pct: float,
    entry_price: float,
    stop_price: float,
    contract_size: float = 1.0,
    max_position_pct: float = 100.0,
) -> dict[str, Any]:
    """Size a position so a stop-out risks exactly `risk_pct` of equity.

    Args:
        account_equity: current account/portfolio equity in account currency.
        risk_pct: fraction of equity to risk if the stop is hit (e.g. 1.0 = 1%).
        entry_price / stop_price: same units (e.g. price per unit/share/coin/lot).
        contract_size: units represented by one "quantity" (1.0 for shares/coins;
            set to the lot's unit size for FX/futures contracts).
        max_position_pct: hard cap on notional as a % of equity, applied even if
            the risk-based size would exceed it (protects against a stop set too
            close to entry blowing up position size).
    """
    if account_equity <= 0:
        return make_error(ErrorCode.INVALID_PARAMETER, "account_equity must be > 0")
    if not (0 < risk_pct <= 100):
        return make_error(ErrorCode.INVALID_PARAMETER, "risk_pct must be between 0 and 100")
    if entry_price <= 0 or stop_price <= 0:
        return make_error(ErrorCode.INVALID_PARAMETER, "entry_price and stop_price must be > 0")
    if entry_price == stop_price:
        return make_error(ErrorCode.INVALID_PARAMETER, "entry_price and stop_price must differ")

    stop_distance = abs(entry_price - stop_price)
    stop_distance_pct = round(stop_distance / entry_price * 100, 4)

    risk_amount = account_equity * (risk_pct / 100)
    raw_quantity = risk_amount / (stop_distance * contract_size)

    notional = raw_quantity * contract_size * entry_price
    max_notional = account_equity * (max_position_pct / 100)
    capped = notional > max_notional

    quantity = raw_quantity
    if capped:
        quantity = max_notional / (contract_size * entry_price)

    # Round DOWN (never up) to 6dp: execute_trade recomputes cost from this
    # exact quantity, and rounding up here can push cost a fraction of a
    # cent past account_equity/max_notional, causing a correctly-sized
    # trade to be rejected as "insufficient balance" downstream.
    quantity = math.floor(quantity * 1_000_000) / 1_000_000
    notional = quantity * contract_size * entry_price

    actual_risk_amount = quantity * contract_size * stop_distance
    actual_risk_pct = round(actual_risk_amount / account_equity * 100, 4)

    return {
        "quantity": quantity,
        "notional": round(notional, 2),
        "notional_pct_of_equity": round(notional / account_equity * 100, 4),
        "stop_distance_pct": stop_distance_pct,
        "requested_risk_pct": risk_pct,
        "actual_risk_pct": actual_risk_pct,
        "actual_risk_amount": round(actual_risk_amount, 2),
        "capped_by_max_position_pct": capped,
    }


# ─── Portfolio VaR ─────────────────────────────────────────────────────────

def _inverse_normal_cdf(p: float) -> float:
    """Acklam's rational approximation of the inverse standard normal CDF.

    Pure-stdlib z-score lookup (accurate to ~1e-9) so parametric VaR doesn't
    need QuantLib or scipy for a single number.
    """
    if not (0 < p < 1):
        raise ValueError("p must be in (0, 1)")

    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]

    p_low, p_high = 0.02425, 1 - 0.02425
    if p < p_low:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p <= p_high:
        q = p - 0.5
        r = q * q
        return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
               (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    q = math.sqrt(-2 * math.log(1 - p))
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)


def calc_var(
    returns_pct: list[float],
    confidence: float = 95.0,
    method: Literal["historical", "parametric"] = "historical",
    capital: float | None = None,
) -> dict[str, Any]:
    """Value at Risk from a series of per-period % returns (trade returns or
    daily equity-curve returns).

    Args:
        returns_pct: per-period returns in percent (e.g. [-1.2, 0.8, 2.1, ...]).
        confidence: confidence level, e.g. 95 or 99.
        method: 'historical' (empirical percentile) or 'parametric' (Gaussian,
            using sample mean/stdev).
        capital: if given, also express VaR/CVaR in currency.
    """
    if len(returns_pct) < 2:
        return make_error(ErrorCode.INVALID_PARAMETER, "need at least 2 returns")
    if not (50 < confidence < 100):
        return make_error(ErrorCode.INVALID_PARAMETER, "confidence must be between 50 and 100")

    alpha = 1 - confidence / 100
    sorted_r = sorted(returns_pct)

    if method == "historical":
        idx = max(0, min(len(sorted_r) - 1, int(round(alpha * len(sorted_r))) - 1))
        var_pct = -sorted_r[idx]
        tail = [r for r in sorted_r if r <= sorted_r[idx]]
        cvar_pct = -statistics.mean(tail) if tail else var_pct
    elif method == "parametric":
        mean_r = statistics.mean(returns_pct)
        std_r = statistics.stdev(returns_pct)
        z = _inverse_normal_cdf(alpha)
        var_pct = -(mean_r + z * std_r)
        phi_z = math.exp(-z * z / 2) / math.sqrt(2 * math.pi)
        cvar_pct = -(mean_r - std_r * phi_z / alpha)
    else:
        return make_error(ErrorCode.INVALID_PARAMETER, "method must be 'historical' or 'parametric'")

    result = {
        "method": method,
        "confidence_pct": confidence,
        "n_observations": len(returns_pct),
        "var_pct": round(var_pct, 4),
        "cvar_pct": round(cvar_pct, 4),
    }
    if capital is not None:
        result["var_amount"] = round(capital * var_pct / 100, 2)
        result["cvar_amount"] = round(capital * cvar_pct / 100, 2)
    return result


# ─── Exposure & circuit-breaker checks ─────────────────────────────────────

def check_exposure_limits(
    open_positions: list[dict[str, Any]],
    new_position_notional: float,
    new_position_symbol: str,
    account_equity: float,
    max_single_symbol_pct: float = 20.0,
    max_total_exposure_pct: float = 100.0,
) -> dict[str, Any]:
    """Check whether adding a new position would breach exposure caps.

    Args:
        open_positions: list of {"symbol": str, "notional": float}.
        new_position_notional: notional of the proposed new position.
        new_position_symbol: symbol of the proposed new position.
        account_equity: current equity used as the denominator for both caps.
        max_single_symbol_pct: cap on any one symbol's total notional / equity.
        max_total_exposure_pct: cap on total notional across all positions / equity.
    """
    if account_equity <= 0:
        return make_error(ErrorCode.INVALID_PARAMETER, "account_equity must be > 0")

    existing_symbol_notional = sum(
        p["notional"] for p in open_positions if p["symbol"] == new_position_symbol
    )
    existing_total_notional = sum(p["notional"] for p in open_positions)

    projected_symbol_notional = existing_symbol_notional + new_position_notional
    projected_total_notional = existing_total_notional + new_position_notional

    projected_symbol_pct = round(projected_symbol_notional / account_equity * 100, 4)
    projected_total_pct = round(projected_total_notional / account_equity * 100, 4)

    breaches = []
    if projected_symbol_pct > max_single_symbol_pct:
        breaches.append(
            f"{new_position_symbol} exposure would be {projected_symbol_pct}% "
            f"of equity (limit {max_single_symbol_pct}%)"
        )
    if projected_total_pct > max_total_exposure_pct:
        breaches.append(
            f"total exposure would be {projected_total_pct}% of equity "
            f"(limit {max_total_exposure_pct}%)"
        )

    return {
        "allowed": len(breaches) == 0,
        "breaches": breaches,
        "projected_symbol_exposure_pct": projected_symbol_pct,
        "projected_total_exposure_pct": projected_total_pct,
        "max_single_symbol_pct": max_single_symbol_pct,
        "max_total_exposure_pct": max_total_exposure_pct,
    }


def check_circuit_breaker(
    daily_pnl_pct: float,
    consecutive_losses: int = 0,
    max_daily_loss_pct: float = 3.0,
    max_consecutive_losses: int = 5,
    max_drawdown_pct: float | None = None,
    current_drawdown_pct: float = 0.0,
) -> dict[str, Any]:
    """Should live/paper trading halt right now?

    Args:
        daily_pnl_pct: today's realized+unrealized P&L as % of starting equity
            (negative = loss).
        consecutive_losses: count of consecutive losing trades so far today.
        max_daily_loss_pct: halt if daily loss exceeds this (positive number).
        max_consecutive_losses: halt after this many consecutive losers.
        max_drawdown_pct: optional hard drawdown ceiling from peak equity.
        current_drawdown_pct: current drawdown from peak equity (positive number).
    """
    triggers = []
    if daily_pnl_pct <= -abs(max_daily_loss_pct):
        triggers.append(f"daily loss {daily_pnl_pct}% breached -{max_daily_loss_pct}% limit")
    if consecutive_losses >= max_consecutive_losses:
        triggers.append(
            f"{consecutive_losses} consecutive losses reached limit of {max_consecutive_losses}"
        )
    if max_drawdown_pct is not None and current_drawdown_pct >= abs(max_drawdown_pct):
        triggers.append(f"drawdown {current_drawdown_pct}% breached {max_drawdown_pct}% limit")

    return {
        "halt_trading": len(triggers) > 0,
        "triggers": triggers,
        "daily_pnl_pct": daily_pnl_pct,
        "consecutive_losses": consecutive_losses,
        "current_drawdown_pct": current_drawdown_pct,
    }


# ─── Option Greeks (QuantLib) ──────────────────────────────────────────────

def calc_option_greeks(
    spot: float,
    strike: float,
    volatility_pct: float,
    expiry_days: int,
    risk_free_rate_pct: float = 0.0,
    dividend_yield_pct: float = 0.0,
    option_type: Literal["call", "put"] = "call",
) -> dict[str, Any]:
    """European option price + Greeks via QuantLib's Black-Scholes-Merton engine.

    Args:
        spot: underlying price.
        strike: option strike.
        volatility_pct: annualized implied volatility, e.g. 25.0 for 25%.
        expiry_days: calendar days to expiry.
        risk_free_rate_pct: annualized risk-free rate, e.g. 4.5 for 4.5%.
        dividend_yield_pct: annualized dividend yield, e.g. 1.5 for 1.5%.
        option_type: 'call' or 'put'.
    """
    if not QUANTLIB_AVAILABLE:
        return make_error(
            ErrorCode.DEPENDENCY_MISSING,
            "QuantLib is not installed. Run: pip install QuantLib",
        )
    if spot <= 0 or strike <= 0:
        return make_error(ErrorCode.INVALID_PARAMETER, "spot and strike must be > 0")
    if expiry_days <= 0:
        return make_error(ErrorCode.INVALID_PARAMETER, "expiry_days must be > 0")
    if volatility_pct <= 0:
        return make_error(ErrorCode.INVALID_PARAMETER, "volatility_pct must be > 0")

    calendar = ql.NullCalendar()
    day_count = ql.Actual365Fixed()
    today = ql.Date.todaysDate()
    ql.Settings.instance().evaluationDate = today
    expiry_date = today + int(expiry_days)

    opt_type = ql.Option.Call if option_type == "call" else ql.Option.Put
    payoff = ql.PlainVanillaPayoff(opt_type, strike)
    exercise = ql.EuropeanExercise(expiry_date)
    option = ql.VanillaOption(payoff, exercise)

    spot_handle = ql.QuoteHandle(ql.SimpleQuote(spot))
    rate_handle = ql.YieldTermStructureHandle(
        ql.FlatForward(today, risk_free_rate_pct / 100, day_count)
    )
    div_handle = ql.YieldTermStructureHandle(
        ql.FlatForward(today, dividend_yield_pct / 100, day_count)
    )
    vol_handle = ql.BlackVolTermStructureHandle(
        ql.BlackConstantVol(today, calendar, volatility_pct / 100, day_count)
    )
    process = ql.BlackScholesMertonProcess(spot_handle, div_handle, rate_handle, vol_handle)
    option.setPricingEngine(ql.AnalyticEuropeanEngine(process))

    return {
        "option_type": option_type,
        "price": round(option.NPV(), 6),
        "delta": round(option.delta(), 6),
        "gamma": round(option.gamma(), 6),
        "theta": round(option.theta() / 365, 6),
        "vega": round(option.vega() / 100, 6),
        "rho": round(option.rho() / 100, 6),
        "inputs": {
            "spot": spot, "strike": strike, "volatility_pct": volatility_pct,
            "expiry_days": expiry_days, "risk_free_rate_pct": risk_free_rate_pct,
            "dividend_yield_pct": dividend_yield_pct,
        },
    }

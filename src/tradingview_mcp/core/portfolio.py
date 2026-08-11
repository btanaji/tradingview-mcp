import sqlite3
import os
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone

from tradingview_mcp.core.services.risk_service import (
    calc_position_size,
    check_exposure_limits,
    check_circuit_breaker,
)

# Store the DB in the user's home directory or current directory
DB_DIR = os.path.expanduser("~/.tradingview_mcp_data")
DB_PATH = os.path.join(DB_DIR, "portfolio.db")

def init_db():
    if not os.path.exists(DB_DIR):
        os.makedirs(DB_DIR)
        
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Table for users and their paper trading balance
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        user_id TEXT PRIMARY KEY,
        balance REAL NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # Table for active positions
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS positions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT NOT NULL,
        symbol TEXT NOT NULL,
        quantity REAL NOT NULL,
        average_price REAL NOT NULL,
        side TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users (user_id)
    )
    ''')
    
    # Table for trade history/logs
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS trade_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT NOT NULL,
        symbol TEXT NOT NULL,
        quantity REAL NOT NULL,
        price REAL NOT NULL,
        side TEXT NOT NULL,  -- 'BUY' or 'SELL'
        realized_pnl REAL DEFAULT 0,
        executed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    conn.commit()
    conn.close()

def get_or_create_user(user_id: str, initial_balance: float = 10000.0) -> float:
    """Returns the current balance of the user. Creates the user with 10k if they don't exist."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("SELECT balance FROM users WHERE user_id = ?", (user_id,))
    row = cursor.fetchone()
    
    if row is None:
        cursor.execute("INSERT INTO users (user_id, balance) VALUES (?, ?)", (user_id, initial_balance))
        conn.commit()
        balance = initial_balance
    else:
        balance = row[0]
        
    conn.close()
    return balance

def _get_open_positions(user_id: str) -> List[Dict[str, Any]]:
    """Open positions valued at their average entry price (no live-price feed
    is wired into this module — callers with fresher marks should override
    the traded symbol's notional themselves before calling risk checks)."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT symbol, quantity, average_price FROM positions WHERE user_id = ?", (user_id,))
    rows = cursor.fetchall()
    conn.close()
    return [{"symbol": s, "quantity": q, "average_price": p, "notional": q * p} for s, q, p in rows]


def _get_equity(user_id: str, balance: float) -> float:
    """Cash + mark-to-average-price value of open positions."""
    return balance + sum(p["notional"] for p in _get_open_positions(user_id))


def _get_consecutive_losses(user_id: str) -> int:
    """Count consecutive losing SELLs (realized_pnl < 0) working back from the
    most recent closed trade, stopping at the first winner."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT realized_pnl FROM trade_history WHERE user_id = ? AND side = 'SELL' "
        "ORDER BY executed_at DESC, id DESC",
        (user_id,),
    )
    rows = cursor.fetchall()
    conn.close()
    streak = 0
    for (pnl,) in rows:
        if pnl < 0:
            streak += 1
        else:
            break
    return streak


def _get_daily_pnl_pct(user_id: str, equity: float) -> float:
    """Today's realized P&L (UTC calendar day) as % of current equity."""
    if equity <= 0:
        return 0.0
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT COALESCE(SUM(realized_pnl), 0) FROM trade_history "
        "WHERE user_id = ? AND side = 'SELL' AND date(executed_at) = ?",
        (user_id, today),
    )
    realized_today = cursor.fetchone()[0]
    conn.close()
    return round(realized_today / equity * 100, 4)


def check_pretrade_risk(
    user_id: str,
    symbol: str,
    notional: float,
    max_single_symbol_pct: Optional[float] = None,
    max_total_exposure_pct: Optional[float] = None,
    max_daily_loss_pct: Optional[float] = None,
    max_consecutive_losses: Optional[int] = None,
) -> Dict[str, Any]:
    """Run exposure and circuit-breaker checks before a BUY. Any limit left
    as None is skipped. Returns {"allowed": bool, "reasons": [...]} plus the
    underlying risk_service payloads for inspection."""
    balance = get_or_create_user(user_id)
    equity = _get_equity(user_id, balance)
    reasons: List[str] = []

    exposure_result = None
    if max_single_symbol_pct is not None or max_total_exposure_pct is not None:
        exposure_result = check_exposure_limits(
            _get_open_positions(user_id), notional, symbol.upper(), equity,
            max_single_symbol_pct if max_single_symbol_pct is not None else 20.0,
            max_total_exposure_pct if max_total_exposure_pct is not None else 100.0,
        )
        if "error" in exposure_result:
            return exposure_result
        if not exposure_result["allowed"]:
            reasons.extend(exposure_result["breaches"])

    breaker_result = None
    if max_daily_loss_pct is not None or max_consecutive_losses is not None:
        breaker_result = check_circuit_breaker(
            _get_daily_pnl_pct(user_id, equity),
            _get_consecutive_losses(user_id),
            max_daily_loss_pct if max_daily_loss_pct is not None else 3.0,
            max_consecutive_losses if max_consecutive_losses is not None else 5,
        )
        if breaker_result["halt_trading"]:
            reasons.extend(breaker_result["triggers"])

    return {
        "allowed": len(reasons) == 0,
        "reasons": reasons,
        "equity": round(equity, 2),
        "exposure_check": exposure_result,
        "circuit_breaker_check": breaker_result,
    }


def execute_trade(
    user_id: str,
    symbol: str,
    quantity: float,
    current_price: float,
    side: str,
    risk_pct: Optional[float] = None,
    stop_price: Optional[float] = None,
    max_single_symbol_pct: Optional[float] = None,
    max_total_exposure_pct: Optional[float] = None,
    max_daily_loss_pct: Optional[float] = None,
    max_consecutive_losses: Optional[int] = None,
) -> Dict[str, Any]:
    """Execute a simulated trade (BUY or SELL) for a user.

    Optional risk controls (all skipped unless passed, so existing callers
    are unaffected):
      - risk_pct + stop_price: on a BUY, `quantity` is ignored and instead
        computed via risk_service.calc_position_size so the stop risks
        exactly risk_pct of equity.
      - max_single_symbol_pct / max_total_exposure_pct: reject the BUY if it
        would breach per-symbol or total exposure caps.
      - max_daily_loss_pct / max_consecutive_losses: reject the BUY if the
        account's circuit breaker is currently tripped.
    Risk checks only gate new BUYs — closing a position via SELL is never
    blocked.
    """
    symbol = symbol.upper()
    side = side.upper()

    if side not in ['BUY', 'SELL']:
        return {"error": "Side must be 'BUY' or 'SELL'"}

    # Initialize user if they don't exist
    balance = get_or_create_user(user_id)

    if side == 'BUY':
        if risk_pct is not None and stop_price is not None:
            equity = _get_equity(user_id, balance)
            sizing = calc_position_size(equity, risk_pct, current_price, stop_price)
            if "error" in sizing:
                return sizing
            quantity = sizing["quantity"]

        if quantity <= 0:
            return {"error": "Quantity must be greater than 0"}

        if any(v is not None for v in (
            max_single_symbol_pct, max_total_exposure_pct,
            max_daily_loss_pct, max_consecutive_losses,
        )):
            risk_check = check_pretrade_risk(
                user_id, symbol, quantity * current_price,
                max_single_symbol_pct, max_total_exposure_pct,
                max_daily_loss_pct, max_consecutive_losses,
            )
            if "error" in risk_check:
                return risk_check
            if not risk_check["allowed"]:
                return {"error": "Trade rejected by risk checks", "reasons": risk_check["reasons"]}
    elif quantity <= 0:
        return {"error": "Quantity must be greater than 0"}

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    try:
        if side == 'BUY':
            cost = quantity * current_price
            if balance < cost:
                return {"error": f"Insufficient balance. Required: ${cost:.2f}, Available: ${balance:.2f}"}
            
            # Deduct balance
            new_balance = balance - cost
            cursor.execute("UPDATE users SET balance = ? WHERE user_id = ?", (new_balance, user_id))
            
            # Check if position exists to average down, else create new
            cursor.execute("SELECT id, quantity, average_price FROM positions WHERE user_id = ? AND symbol = ?", (user_id, symbol))
            pos = cursor.fetchone()
            
            if pos:
                pos_id, existing_qty, existing_avg_price = pos
                new_qty = existing_qty + quantity
                new_avg_price = ((existing_qty * existing_avg_price) + (quantity * current_price)) / new_qty
                cursor.execute("UPDATE positions SET quantity = ?, average_price = ? WHERE id = ?", (new_qty, new_avg_price, pos_id))
            else:
                cursor.execute("INSERT INTO positions (user_id, symbol, quantity, average_price, side) VALUES (?, ?, ?, ?, ?)", 
                               (user_id, symbol, quantity, current_price, "LONG"))
                
            # Log history
            cursor.execute("INSERT INTO trade_history (user_id, symbol, quantity, price, side) VALUES (?, ?, ?, ?, ?)",
                           (user_id, symbol, quantity, current_price, 'BUY'))
            
            conn.commit()
            return {
                "status": "success", 
                "action": "BUY", 
                "symbol": symbol,
                "quantity": quantity,
                "price": current_price,
                "total_cost": cost,
                "remaining_balance": new_balance
            }
            
        elif side == 'SELL':
            # Check position
            cursor.execute("SELECT id, quantity, average_price FROM positions WHERE user_id = ? AND symbol = ?", (user_id, symbol))
            pos = cursor.fetchone()
            
            if not pos:
                return {"error": f"You do not own any {symbol}"}
                
            pos_id, existing_qty, existing_avg_price = pos
            
            if quantity > existing_qty:
                return {"error": f"Cannot sell {quantity} of {symbol}. You only own {existing_qty}."}
                
            revenue = quantity * current_price
            realized_pnl = (current_price - existing_avg_price) * quantity
            
            # Add to balance
            new_balance = balance + revenue
            cursor.execute("UPDATE users SET balance = ? WHERE user_id = ?", (new_balance, user_id))
            
            # Update or remove position
            new_qty = existing_qty - quantity
            if new_qty <= 0.00001:  # Floating point safety
                cursor.execute("DELETE FROM positions WHERE id = ?", (pos_id,))
            else:
                cursor.execute("UPDATE positions SET quantity = ? WHERE id = ?", (new_qty, pos_id))
                
            # Log history
            cursor.execute("INSERT INTO trade_history (user_id, symbol, quantity, price, side, realized_pnl) VALUES (?, ?, ?, ?, ?, ?)",
                           (user_id, symbol, quantity, current_price, 'SELL', realized_pnl))
            
            conn.commit()
            return {
                "status": "success", 
                "action": "SELL", 
                "symbol": symbol,
                "quantity": quantity,
                "price": current_price,
                "revenue": revenue,
                "realized_pnl": realized_pnl,
                "new_balance": new_balance
            }

    except Exception as e:
        conn.rollback()
        return {"error": f"Database error during trade: {str(e)}"}
    finally:
        conn.close()

def get_portfolio(user_id: str) -> Dict[str, Any]:
    """Retrieve the user's current portfolio (balance and open positions)."""
    balance = get_or_create_user(user_id)
    
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute("SELECT symbol, quantity, average_price FROM positions WHERE user_id = ?", (user_id,))
    rows = cursor.fetchall()
    
    positions = []
    for row in rows:
        positions.append({
            "symbol": row["symbol"],
            "quantity": row["quantity"],
            "average_price": row["average_price"]
        })
        
    conn.close()
    
    return {
        "user_id": user_id,
        "balance": balance,
        "positions": positions
    }

# Initialize DB when module is imported
init_db()

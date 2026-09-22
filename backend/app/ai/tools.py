from sqlalchemy.orm import Session

from app.services.stock_service import StockService
from app.services.price_attribution import get_price_move_attribution
from app.services.technical_analysis import get_stock_indicators


def get_stock_quote(db: Session, symbol: str) -> dict:
    return StockService(db).get_quote(symbol)


def get_stock_history(db: Session, symbol: str, days: int = 20) -> list[dict]:
    return StockService(db).get_history(symbol, min(max(days, 1), 252))


def get_52_week_data(db: Session, symbol: str) -> dict:
    history = get_stock_history(db, symbol, 252)
    if not history:
        raise ValueError("No historical data")
    high = max(row["high"] for row in history)
    low = min(row["low"] for row in history)
    current = history[-1]["close"]
    return {
        "symbol": symbol.strip().upper(), "current_price": current,
        "week_52_high": high, "week_52_low": low,
        "distance_from_high_pct": ((current - high) / high * 100) if high else 0.0,
        "distance_from_low_pct": ((current - low) / low * 100) if low else 0.0,
    }


def get_technical_indicators(db: Session, symbol: str, period: int = 14) -> dict:
    return get_stock_indicators(db, symbol, period)


def get_stock_move_attribution(db: Session, symbol: str, window_days: int = 1) -> dict:
    return get_price_move_attribution(db, symbol, window_days)

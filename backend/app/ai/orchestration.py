"""Query routing and evidence assembly for the FinPilot AI service."""

from __future__ import annotations

import json
import re
from datetime import date, datetime
from typing import Any

from sqlalchemy.orm import Session

from app.ai.providers import configured_provider
from app.ai.tools import (
    get_52_week_data,
    get_stock_history,
    get_stock_move_attribution,
    get_stock_quote,
    get_technical_indicators,
)
from app.models.research import Document, DocumentChunk
from app.models.stock import Stock
from app.services.rag import retrieve_documents as hybrid_retrieve_documents


def _json_default(value: Any):
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def route_query(message: str) -> str:
    text = message.lower()
    if any(word in text for word in ("document", "filing", "report", "management", "transcript")):
        return "documents"
    if any(word in text for word in ("why", "move", "fell", "rose", "dropped", "jumped")):
        return "attribution"
    if any(word in text for word in ("rsi", "macd", "overbought", "oversold", "technical")):
        return "technical_analysis"
    if any(word in text for word in ("52-week", "52 week", "high", "low")):
        return "market_data"
    return "market_data"


def _symbol_from_message(db: Session, message: str, current_symbol: str | None) -> str | None:
    if current_symbol:
        return current_symbol.strip().upper()
    lowered = message.lower()
    for stock in db.query(Stock).all():
        if stock.ticker.lower() in lowered or stock.company_name.lower() in lowered:
            return stock.ticker
    match = re.search(r"\b[A-Z][A-Z0-9]{1,14}\b", message)
    return match.group(0) if match else None


def retrieve_documents(db: Session, query: str, user_id: int | None = None, limit: int = 5) -> list[dict]:
    statement = db.query(DocumentChunk, Document).join(Document, Document.id == DocumentChunk.document_id)
    if user_id is not None:
        statement = statement.filter(Document.user_id == user_id)
    matches = statement.filter(DocumentChunk.text.ilike(f"%{query.strip()}%")).limit(limit).all()
    return [
        {
            "document_id": document.id,
            "document": document.name,
            "chunk": chunk.chunk_index,
            "text": chunk.text,
        }
        for chunk, document in matches
    ]


async def build_evidence(db: Session, message: str, current_symbol: str | None = None, user_id: int | None = None) -> dict:
    route = route_query(message)
    symbol = _symbol_from_message(db, message, current_symbol)
    evidence: dict[str, Any] = {"route": route, "symbol": symbol, "facts": [], "citations": []}
    if route == "documents":
        matches = await hybrid_retrieve_documents(db, message, user_id)
        evidence["facts"] = matches
        evidence["citations"] = [{"type": "document", "document_id": item["document_id"], "title": item["document"]} for item in matches]
        return evidence
    if not symbol:
        evidence["facts"].append({"message": "No stock symbol was detected; provide a ticker or current_symbol."})
        return evidence
    if route == "attribution":
        result = get_stock_move_attribution(db, symbol)
    elif route == "technical_analysis":
        result = get_technical_indicators(db, symbol)
    elif "52" in message.lower() or "high" in message.lower() or "low" in message.lower():
        result = get_52_week_data(db, symbol)
    elif "history" in message.lower() or "return" in message.lower():
        result = {"symbol": symbol, "history": get_stock_history(db, symbol, 20)}
    else:
        result = get_stock_quote(db, symbol)
    evidence["facts"].append(result)
    evidence["citations"].append({"type": "market_database", "symbol": symbol})
    return evidence


def deterministic_answer(message: str, evidence: dict) -> str:
    facts = evidence["facts"]
    if not facts:
        return "I need a stock symbol or an indexed document query to answer that."
    if evidence["route"] == "documents":
        if not facts:
            return "I could not find matching indexed document evidence."
        return "I found indexed document evidence, but an LLM is required to synthesize a detailed answer."
    fact = facts[0]
    if evidence["route"] == "attribution":
        move = fact["move"]
        return f"{fact['ticker']} moved {move['return_percent']:.2f}% over the selected window. The evidence pack includes measured price, volume, and volatility facts; news items are correlation evidence only."
    if "price" in fact:
        return f"{fact['ticker']} ({fact['symbol']}) is at {fact['price']:.2f}, changing {fact['change_percent']:.2f}% from the previous stored close."
    if "week_52_high" in fact:
        return f"{fact['symbol']} has a calculated 52-week high of {fact['week_52_high']:.2f} and low of {fact['week_52_low']:.2f}."
    return json.dumps(fact, default=_json_default)


async def answer_query(db: Session, message: str, current_symbol: str | None = None, user_id: int | None = None) -> dict:
    evidence = await build_evidence(db, message, current_symbol, user_id)
    provider = configured_provider()
    context = json.dumps(evidence, default=_json_default)
    if provider and evidence["facts"]:
        # Use the Responses API tool loop for configured LLMs. The model can
        # request additional deterministic tools instead of answering from memory.
        from app.ai.agent import ask_finpilot

        answer = await ask_finpilot(db, message, current_symbol, user_id=user_id)
        provider_name = provider.name
    else:
        answer = deterministic_answer(message, evidence)
        provider_name = "deterministic"
    return {"answer": answer, "route": evidence["route"], "evidence": evidence["facts"], "citations": evidence["citations"], "provider": provider_name}
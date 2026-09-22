"""Responses API orchestration; financial facts are always supplied by tools."""

import json
from datetime import date, datetime
from typing import Any

from sqlalchemy.orm import Session

from app.ai import tools as market_tools
from app.database import settings

TOOLS = [
    {
        "type": "function", "name": "get_stock_quote",
        "description": "Get the latest stored market quote for an NSE stock.",
        "parameters": {"type": "object", "properties": {"symbol": {"type": "string", "description": "NSE ticker, e.g. RELIANCE"}}, "required": ["symbol"], "additionalProperties": False},
        "strict": True,
    },
    {
        "type": "function", "name": "get_stock_history",
        "description": "Get stored daily OHLCV history for an NSE stock.",
        "parameters": {"type": "object", "properties": {"symbol": {"type": "string"}, "days": {"type": "integer", "minimum": 1, "maximum": 252}}, "required": ["symbol", "days"], "additionalProperties": False},
        "strict": True,
    },
    {
        "type": "function", "name": "get_52_week_data",
        "description": "Get current price plus calculated 52-week high and low for an NSE stock.",
        "parameters": {"type": "object", "properties": {"symbol": {"type": "string"}}, "required": ["symbol"], "additionalProperties": False},
        "strict": True,
    },
    {
        "type": "function", "name": "get_technical_indicators",
        "description": "Calculate deterministic RSI, moving averages, MACD, ATR, Bollinger Bands, and volume ratio for an NSE stock.",
        "parameters": {"type": "object", "properties": {"symbol": {"type": "string"}, "period": {"type": "integer", "minimum": 2, "maximum": 100}}, "required": ["symbol", "period"], "additionalProperties": False},
        "strict": True,
    },
    {
        "type": "function", "name": "get_stock_move_attribution",
        "description": "Build a measured price, volume, volatility, and cached-news evidence pack for a recent NSE stock move. News is correlation evidence, not proof of causation.",
        "parameters": {"type": "object", "properties": {"symbol": {"type": "string"}, "window_days": {"type": "integer", "minimum": 1, "maximum": 20}}, "required": ["symbol", "window_days"], "additionalProperties": False},
        "strict": True,
    },
    {
        "type": "function", "name": "retrieve_document_evidence",
        "description": "Retrieve matching indexed document chunks for evidence-based financial research.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}, "user_id": {"type": ["integer", "null"], "minimum": 1}}, "required": ["query", "user_id"], "additionalProperties": False},
        "strict": True,
    },
]


def _json_default(value: Any):
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def execute_tool(db: Session, tool_name: str, arguments: dict) -> dict | list[dict]:
    if tool_name == "get_stock_quote":
        return market_tools.get_stock_quote(db, arguments["symbol"])
    if tool_name == "get_stock_history":
        return market_tools.get_stock_history(db, arguments["symbol"], arguments.get("days", 20))
    if tool_name == "get_52_week_data":
        return market_tools.get_52_week_data(db, arguments["symbol"])
    if tool_name == "get_technical_indicators":
        return market_tools.get_technical_indicators(db, arguments["symbol"], arguments.get("period", 14))
    if tool_name == "get_stock_move_attribution":
        return market_tools.get_stock_move_attribution(db, arguments["symbol"], arguments.get("window_days", 1))
    if tool_name == "retrieve_document_evidence":
        from app.ai.orchestration import retrieve_documents
        return retrieve_documents(db, arguments["query"], arguments.get("user_id"))
    raise ValueError(f"Unknown tool: {tool_name}")


def _chat_completion_tools() -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["parameters"],
            },
        }
        for tool in TOOLS
    ]


async def ask_finpilot(db: Session, message: str, current_symbol: str | None = None, user_id: int | None = None) -> str:
    api_key = settings.LLM_API_KEY or settings.OPENAI_API_KEY
    if not api_key:
        raise ValueError("LLM_API_KEY is not configured")
    try:
        from openai import AsyncOpenAI
    except ImportError as error:
        raise RuntimeError("OpenAI SDK is not installed. Run pip install -r requirements.txt") from error

    context = ""
    if current_symbol:
        context = f"The user is viewing {current_symbol.strip().upper()}. Resolve references such as 'it' or 'this stock' to that symbol unless they name another stock."
    instructions = f"""You are FinPilot, an AI financial research assistant.
{context}
Never invent live market data or calculate market values from memory. Use tools for current prices, historical data, returns, 52-week levels, technical indicators, price-move attribution, and indexed documents. Clearly distinguish tool-provided facts from interpretation; do not present correlation as causation. Be concise."""
    client = AsyncOpenAI(
        api_key=api_key,
        base_url=settings.LLM_BASE_URL or None,
    )
    input_items: list[Any] = [{"role": "user", "content": message}]

    provider = settings.LLM_PROVIDER.strip().lower() or "openai"
    model = settings.LLM_MODEL or settings.OPENAI_MODEL

    if provider != "openai" or settings.LLM_BASE_URL:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": instructions},
            {"role": "user", "content": message},
        ]
        for _ in range(4):
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                tools=_chat_completion_tools(),
                tool_choice="auto",
            )
            choice = response.choices[0]
            assistant = choice.message
            if not assistant.tool_calls:
                return assistant.content or "The model returned an empty answer."
            messages.append({
                "role": "assistant",
                "content": assistant.content,
                "tool_calls": [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {"name": call.function.name, "arguments": call.function.arguments},
                    }
                    for call in assistant.tool_calls
                ],
            })
            for call in assistant.tool_calls:
                arguments = json.loads(call.function.arguments)
                if call.function.name == "retrieve_document_evidence" and arguments.get("user_id") is None:
                    arguments["user_id"] = user_id
                try:
                    result = execute_tool(db, call.function.name, arguments)
                    output = json.dumps(result, default=_json_default)
                except Exception as error:
                    output = json.dumps({"error": str(error)})
                messages.append({"role": "tool", "tool_call_id": call.id, "content": output})
        raise RuntimeError("The AI requested too many tool calls")

    # Bound the loop so a malformed tool exchange cannot hold an HTTP request forever.
    for _ in range(4):
        response = await client.responses.create(model=model, instructions=instructions, tools=TOOLS, input=input_items)
        tool_calls = [item for item in response.output if item.type == "function_call"]
        if not tool_calls:
            return response.output_text
        input_items.extend(response.output)
        for call in tool_calls:
            try:
                arguments = json.loads(call.arguments)
                if call.name == "retrieve_document_evidence" and arguments.get("user_id") is None:
                    arguments["user_id"] = user_id
                result = execute_tool(db, call.name, arguments)
                output = json.dumps(result, default=_json_default)
            except Exception as error:
                output = json.dumps({"error": str(error)})
            input_items.append({"type": "function_call_output", "call_id": call.call_id, "output": output})
    raise RuntimeError("The AI requested too many tool calls")

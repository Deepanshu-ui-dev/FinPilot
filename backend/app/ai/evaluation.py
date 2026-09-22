"""Small, dependency-free regression checks for query routing."""

from app.ai.orchestration import route_query


CASES = [
    ("What is RELIANCE current price?", "market_data"),
    ("What is the 52-week high?", "market_data"),
    ("Is RELIANCE overbought?", "technical_analysis"),
    ("Why did RELIANCE fall today?", "attribution"),
    ("What did management say in the filing?", "documents"),
]


def run_route_evaluation() -> dict:
    results = [{"query": query, "expected": expected, "actual": route_query(query), "passed": route_query(query) == expected} for query, expected in CASES]
    return {"passed": sum(item["passed"] for item in results), "total": len(results), "results": results}
"""LLM provider boundary. Add another provider here without changing API routes."""

from __future__ import annotations

from typing import Protocol

from app.database import settings


class LLMProvider(Protocol):
    name: str

    async def answer(self, prompt: str, context: str) -> str:
        ...


class OpenAIProvider:
    name = "openai"

    async def answer(self, prompt: str, context: str) -> str:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(
            api_key=settings.LLM_API_KEY or settings.OPENAI_API_KEY,
            base_url=settings.LLM_BASE_URL or None,
        )
        response = await client.responses.create(
            model=settings.LLM_MODEL or settings.OPENAI_MODEL,
            instructions=(
                "You are FinPilot, a financial research assistant. Use only the supplied evidence. "
                "Separate facts from interpretation and never claim correlation proves causation."
            ),
            input=f"Evidence:\n{context}\n\nUser question:\n{prompt}",
        )
        return response.output_text


def configured_provider() -> LLMProvider | None:
    """Return the configured provider, or None for deterministic local mode."""
    if not (settings.LLM_API_KEY or settings.OPENAI_API_KEY):
        return None
    provider = settings.LLM_PROVIDER.strip().lower() or "openai"
    instance = OpenAIProvider()
    instance.name = provider
    return instance
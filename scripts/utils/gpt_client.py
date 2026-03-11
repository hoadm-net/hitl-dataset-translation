"""
OpenAI API wrapper for LLM-based translation (Phase 3).

Provides a retry-enabled client for calling GPT models with few-shot prompting.
Uses the official openai Python SDK v1.x.
"""

from __future__ import annotations

import os
import time
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI, RateLimitError, APIError

load_dotenv()

_client: OpenAI | None = None


def get_client() -> OpenAI:
    """Lazily initialize and return the OpenAI client."""
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY is not set. Copy .env.example to .env and fill it in.")
        _client = OpenAI(api_key=api_key)
    return _client


def chat_completion(
    messages: list[dict[str, str]],
    model: str | None = None,
    temperature: float = 0.3,
    max_retries: int = 5,
    retry_delay: float = 10.0,
) -> str:
    """
    Call the OpenAI chat completion endpoint with automatic retry on rate limits.

    Args:
        messages: List of {"role": ..., "content": ...} dicts.
        model: Model name override; falls back to OPENAI_MODEL env var or gpt-4o-mini.
        temperature: Sampling temperature.
        max_retries: Maximum number of retry attempts on transient errors.
        retry_delay: Initial delay between retries (seconds); doubles on each retry.

    Returns:
        The assistant's reply string.

    Raises:
        RuntimeError: If all retries are exhausted.
    """
    model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    client = get_client()

    delay = retry_delay
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
            )
            return response.choices[0].message.content.strip()
        except RateLimitError:
            if attempt == max_retries:
                raise
            print(f"[GPT] Rate limit hit. Retrying in {delay:.0f}s (attempt {attempt}/{max_retries})...")
            time.sleep(delay)
            delay = min(delay * 2, 120.0)
        except APIError as e:
            if attempt == max_retries:
                raise RuntimeError(f"OpenAI API error after {max_retries} attempts: {e}") from e
            print(f"[GPT] API error: {e}. Retrying in {delay:.0f}s...")
            time.sleep(delay)
            delay = min(delay * 2, 120.0)

    raise RuntimeError("Unreachable")  # pragma: no cover

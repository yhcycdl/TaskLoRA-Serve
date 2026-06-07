from __future__ import annotations

import time
from typing import Any

import httpx


class VLLMBackend:
    def __init__(self, base_url: str, timeout_seconds: float = 120):
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    async def chat_completions(
        self,
        model: str,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
        stream: bool = False,
    ) -> tuple[dict[str, Any], float]:
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream,
        }
        start = time.perf_counter()
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            response = await client.post(f"{self.base_url}/v1/chat/completions", json=payload)
            response.raise_for_status()
            data = response.json()
        latency_ms = (time.perf_counter() - start) * 1000
        return data, latency_ms


def extract_total_tokens(response: dict[str, Any]) -> int:
    usage = response.get("usage") or {}
    total = usage.get("total_tokens")
    if isinstance(total, int):
        return total
    prompt = usage.get("prompt_tokens") or 0
    completion = usage.get("completion_tokens") or 0
    if isinstance(prompt, int) and isinstance(completion, int):
        return prompt + completion
    return 0


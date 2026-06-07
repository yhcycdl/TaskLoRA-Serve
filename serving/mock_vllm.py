from __future__ import annotations

import argparse
import time
import uuid
from typing import Any

KNOWN_MODELS = {
    "Qwen/Qwen2.5-1.5B-Instruct",
    "code-lora",
    "math-lora",
}


def estimate_tokens(messages: list[dict[str, str]], completion: str) -> tuple[int, int, int]:
    prompt_text = " ".join(message.get("content", "") for message in messages)
    prompt_tokens = max(1, len(prompt_text.split()))
    completion_tokens = max(1, len(completion.split()))
    return prompt_tokens, completion_tokens, prompt_tokens + completion_tokens


def build_mock_completion(model: str, messages: list[dict[str, str]]) -> str:
    user_messages = [message.get("content", "") for message in messages if message.get("role") == "user"]
    prompt = user_messages[-1] if user_messages else ""
    prompt_preview = prompt.strip().replace("\n", " ")[:160]

    if model == "code-lora":
        return (
            "Mock code-lora response:\n"
            "```python\n"
            "def solution(*args):\n"
            "    # Replace this mock body with the real vLLM adapter output.\n"
            "    return None\n"
            "```\n"
            f"Prompt preview: {prompt_preview}"
        )
    if model == "math-lora":
        return (
            "Mock math-lora response:\n"
            "We solve the problem step by step. This local mock does not run a model, "
            "but it verifies routing and benchmark plumbing.\n"
            "#### 42\n"
            f"Prompt preview: {prompt_preview}"
        )
    return (
        "Mock base-model response: this verifies the general task path through the Gateway. "
        f"Prompt preview: {prompt_preview}"
    )


def make_chat_completion(payload: dict[str, Any]) -> dict[str, Any]:
    model = str(payload.get("model") or "")
    messages = payload.get("messages") or []
    if model not in KNOWN_MODELS:
        raise ValueError(f"Unknown mock model: {model}")
    if payload.get("stream"):
        raise ValueError("The mock backend only supports non-streaming responses")
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty list")

    content = build_mock_completion(model, messages)
    prompt_tokens, completion_tokens, total_tokens = estimate_tokens(messages, content)
    return {
        "id": f"chatcmpl-mock-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        },
    }


def create_app():
    try:
        from fastapi import FastAPI, HTTPException
    except ImportError as exc:  # pragma: no cover - exercised only in missing-dependency envs.
        raise RuntimeError(
            "FastAPI is required to run the mock backend. Install `requirements-gateway.txt`."
        ) from exc

    app = FastAPI(title="TaskLoRA Mock vLLM Backend", version="0.1.0")

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {"ok": True, "service": "mock-vllm", "models": sorted(KNOWN_MODELS)}

    @app.get("/v1/models")
    async def models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {"id": model, "object": "model", "owned_by": "tasklora-mock"}
                for model in sorted(KNOWN_MODELS)
            ],
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            return make_chat_completion(payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return app


try:
    app = create_app()
except RuntimeError:
    app = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a local mock vLLM-compatible backend.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8001)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError("uvicorn is required. Install `requirements-gateway.txt`.") from exc
    if app is None:
        create_app()
    uvicorn.run("serving.mock_vllm:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()

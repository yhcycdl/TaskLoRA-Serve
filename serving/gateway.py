from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response

from observability.metrics import GatewayMetrics
from registry.model_registry import ModelRegistry
from serving.request_schema import TaskChatRequest
from serving.router import TaskRouter
from serving.vllm_backend import VLLMBackend, extract_total_tokens

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "serving.yaml"


def resolve_project_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def load_config(path: Path | None = None) -> dict[str, Any]:
    config_path = path or resolve_project_path(
        os.environ.get("TASKLORA_SERVING_CONFIG", "configs/serving.yaml")
    )
    with config_path.open("r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}
    if os.environ.get("TASKLORA_VLLM_BASE_URL"):
        loaded["vllm_base_url"] = os.environ["TASKLORA_VLLM_BASE_URL"]
    if os.environ.get("TASKLORA_REQUEST_TIMEOUT_SECONDS"):
        loaded["request_timeout_seconds"] = float(os.environ["TASKLORA_REQUEST_TIMEOUT_SECONDS"])
    return loaded


config = load_config()
registry = ModelRegistry(resolve_project_path(config.get("registry_path", "registry/adapters.yaml")))
router = TaskRouter(
    registry=registry,
    task_routes=config.get("task_routes", {}),
    base_model_alias=config.get("base_model_alias", "base"),
    base_model_serving_name=config.get("base_model_serving_name", config.get("base_model")),
)
backend = VLLMBackend(
    base_url=config.get("vllm_base_url", "http://localhost:8001"),
    timeout_seconds=float(config.get("request_timeout_seconds", 120)),
)
metrics = GatewayMetrics()

app = FastAPI(title="TaskLoRA-Serve Gateway", version="0.1.0")


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "ok": True,
        "service": "tasklora-gateway",
        "base_model": config.get("base_model"),
        "base_model_serving_name": config.get("base_model_serving_name", config.get("base_model")),
        "vllm_base_url": config.get("vllm_base_url"),
    }


@app.get("/adapters")
async def adapters() -> dict[str, Any]:
    return registry.to_dict()


@app.get("/metrics")
async def prometheus_metrics() -> Response:
    return Response(content=metrics.render(), media_type="text/plain; version=0.0.4")


@app.post("/v1/task/chat")
async def task_chat(request: TaskChatRequest) -> dict[str, Any]:
    try:
        decision = router.route(request.task)
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    adapter = decision.adapter
    try:
        backend_response, latency_ms = await backend.chat_completions(
            model=decision.serving_model,
            messages=[message.model_dump() for message in request.messages],
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            stream=request.stream,
        )
    except Exception as exc:  # vLLM/network errors should be visible to clients and metrics.
        metrics.observe_error(request.task, adapter, type(exc).__name__)
        raise HTTPException(status_code=502, detail=f"Backend request failed: {exc}") from exc

    total_tokens = extract_total_tokens(backend_response)
    metrics.observe_success(request.task, adapter, latency_ms, total_tokens)
    return {
        "ok": True,
        "task": request.task,
        "adapter": adapter,
        "model": decision.serving_model,
        "latency_ms": latency_ms,
        "response": backend_response,
    }

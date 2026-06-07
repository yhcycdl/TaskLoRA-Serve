from __future__ import annotations

import argparse
import asyncio
import json
import random
import time
import uuid
from pathlib import Path
from typing import Any

import httpx
import yaml

from benchmark.analyze_results import summarize_records


def load_config(path: str) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def choose_workload(config: dict[str, Any], rng: random.Random) -> tuple[str, str, str]:
    workloads = config["workloads"]
    total = sum(float(workload["ratio"]) for workload in workloads)
    draw = rng.random() * total
    cumulative = 0.0
    selected = workloads[-1]
    for workload in workloads:
        cumulative += float(workload["ratio"])
        if draw <= cumulative:
            selected = workload
            break
    prompt = rng.choice(selected["prompts"])
    return selected["name"], selected["task"], prompt


def extract_tokens(gateway_response: dict[str, Any]) -> int:
    response = gateway_response.get("response") or {}
    usage = response.get("usage") or {}
    total = usage.get("total_tokens")
    if isinstance(total, int):
        return total
    prompt = usage.get("prompt_tokens") or 0
    completion = usage.get("completion_tokens") or 0
    return prompt + completion if isinstance(prompt, int) and isinstance(completion, int) else 0


async def send_one(
    client: httpx.AsyncClient,
    url: str,
    config: dict[str, Any],
    rng: random.Random,
) -> dict[str, Any]:
    workload, task, prompt = choose_workload(config, rng)
    request_id = str(uuid.uuid4())
    payload = {
        "task": task,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": int(config.get("max_tokens", 256)),
        "temperature": float(config.get("temperature", 0.2)),
    }
    started_at = time.time()
    started = time.perf_counter()
    try:
        response = await client.post(url, json=payload)
        latency_ms = (time.perf_counter() - started) * 1000
        ended_at = time.time()
        status_code = response.status_code
        data = response.json()
        ok = response.is_success and bool(data.get("ok"))
        return {
            "request_id": request_id,
            "started_at": started_at,
            "ended_at": ended_at,
            "ok": ok,
            "status_code": status_code,
            "task": task,
            "workload": workload,
            "adapter": data.get("adapter"),
            "model": data.get("model"),
            "latency_ms": latency_ms,
            "backend_latency_ms": data.get("latency_ms"),
            "tokens": extract_tokens(data),
            "error": None if ok else data.get("detail") or data,
        }
    except Exception as exc:
        latency_ms = (time.perf_counter() - started) * 1000
        ended_at = time.time()
        return {
            "request_id": request_id,
            "started_at": started_at,
            "ended_at": ended_at,
            "ok": False,
            "status_code": None,
            "task": task,
            "workload": workload,
            "adapter": None,
            "model": None,
            "latency_ms": latency_ms,
            "backend_latency_ms": None,
            "tokens": 0,
            "error": f"{type(exc).__name__}: {exc}",
        }


async def worker(
    worker_id: int,
    client: httpx.AsyncClient,
    url: str,
    config: dict[str, Any],
    end_time: float,
    records: list[dict[str, Any]],
) -> None:
    rng = random.Random(int(config.get("seed", 42)) + worker_id)
    while time.perf_counter() < end_time:
        record = await send_one(client, url, config, rng)
        record["worker_id"] = worker_id
        records.append(record)


async def run_loadgen(config: dict[str, Any], url: str, concurrency: int, duration: int) -> list[dict[str, Any]]:
    timeout = float(config.get("request_timeout_seconds", 120))
    records: list[dict[str, Any]] = []
    end_time = time.perf_counter() + duration
    async with httpx.AsyncClient(timeout=timeout) as client:
        await asyncio.gather(
            *[
                worker(i, client, url, config, end_time, records)
                for i in range(concurrency)
            ]
        )
    return records


def write_records(path: str, records: list[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Async load generator for TaskLoRA Gateway.")
    parser.add_argument("--config", default="configs/benchmark.yaml")
    parser.add_argument("--url", default=None)
    parser.add_argument("--concurrency", type=int, default=None)
    parser.add_argument("--duration", type=int, default=None)
    parser.add_argument("--output", default="benchmark/results/loadgen.jsonl")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    url = args.url or config.get("url", "http://localhost:8000/v1/task/chat")
    concurrency = args.concurrency or int(config.get("concurrency", 4))
    duration = args.duration or int(config.get("duration_seconds", 60))
    records = asyncio.run(run_loadgen(config, url, concurrency, duration))
    write_records(args.output, records)
    summary = summarize_records(records)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Wrote {len(records)} records to {args.output}")


if __name__ == "__main__":
    main()

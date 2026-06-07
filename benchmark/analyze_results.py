from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    if p <= 0:
        return min(values)
    if p >= 100:
        return max(values)
    ordered = sorted(values)
    k = (len(ordered) - 1) * (p / 100)
    lower = int(k)
    upper = min(lower + 1, len(ordered) - 1)
    weight = k - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    ok_records = [record for record in records if record.get("ok")]
    latencies = [float(record["latency_ms"]) for record in ok_records if record.get("latency_ms") is not None]
    tokens = sum(int(record.get("tokens") or 0) for record in ok_records)

    mean_latency_ms = sum(latencies) / len(latencies) if latencies else 0.0
    started_values = [float(record["started_at"]) for record in records if record.get("started_at")]
    ended_values = [float(record["ended_at"]) for record in records if record.get("ended_at")]
    if started_values and ended_values:
        wall_seconds = max(max(ended_values) - min(started_values), 1e-9)
    else:
        wall_seconds = max(sum(latencies) / 1000, 1e-9)

    adapter_counts = Counter(record.get("adapter") or "unknown" for record in records)
    task_counts = Counter(record.get("task") or "unknown" for record in records)
    error_counts = Counter(record.get("error") or "none" for record in records if not record.get("ok"))

    return {
        "requests": total,
        "ok": len(ok_records),
        "errors": total - len(ok_records),
        "error_rate": (total - len(ok_records)) / total if total else 0.0,
        "mean_latency_ms": mean_latency_ms,
        "p50_latency_ms": percentile(latencies, 50),
        "p95_latency_ms": percentile(latencies, 95),
        "p99_latency_ms": percentile(latencies, 99),
        "tokens": tokens,
        "wall_seconds": wall_seconds,
        "requests_per_sec": len(ok_records) / wall_seconds,
        "tokens_per_sec": tokens / wall_seconds,
        "adapter_counts": dict(adapter_counts),
        "task_counts": dict(task_counts),
        "error_counts": dict(error_counts),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze TaskLoRA loadgen JSONL results.")
    parser.add_argument("path")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = read_jsonl(args.path)
    summary = summarize_records(records)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

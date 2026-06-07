from __future__ import annotations

try:
    from prometheus_client import Counter, Histogram, generate_latest
except ImportError:  # pragma: no cover - used only when optional dependency is absent
    Counter = Histogram = None
    generate_latest = None


class GatewayMetrics:
    def __init__(self) -> None:
        self.enabled = Counter is not None and Histogram is not None
        if not self.enabled:
            return
        self.request_total = Counter(
            "llm_request_total",
            "Total LLM gateway requests",
            ["task", "adapter", "status"],
        )
        self.request_latency = Histogram(
            "llm_request_latency_seconds",
            "LLM gateway request latency",
            ["task", "adapter"],
        )
        self.request_errors = Counter(
            "llm_request_errors_total",
            "Total LLM gateway request errors",
            ["task", "adapter", "error_type"],
        )
        self.tokens_generated = Counter(
            "llm_tokens_generated_total",
            "Total tokens reported by backend responses",
            ["task", "adapter"],
        )
        self.adapter_requests = Counter(
            "llm_adapter_requests_total",
            "Adapter-level request distribution",
            ["adapter"],
        )

    def observe_success(self, task: str, adapter: str, latency_ms: float, total_tokens: int) -> None:
        if not self.enabled:
            return
        self.request_total.labels(task, adapter, "ok").inc()
        self.request_latency.labels(task, adapter).observe(latency_ms / 1000)
        self.tokens_generated.labels(task, adapter).inc(total_tokens)
        self.adapter_requests.labels(adapter).inc()

    def observe_error(self, task: str, adapter: str, error_type: str) -> None:
        if not self.enabled:
            return
        self.request_total.labels(task, adapter, "error").inc()
        self.request_errors.labels(task, adapter, error_type).inc()

    def render(self) -> bytes:
        if not self.enabled or generate_latest is None:
            return b"# prometheus_client is not installed\n"
        return generate_latest()


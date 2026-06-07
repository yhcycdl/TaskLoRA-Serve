# Architecture

TaskLoRA-Serve V1 is a small AI Infra project for multi-task LoRA serving.

## Goal

Show an end-to-end LLM engineering path:

1. Convert public datasets into chat-format SFT data.
2. Train task-specific QLoRA adapters.
3. Evaluate base vs adapter quality.
4. Serve adapters through vLLM.
5. Route requests through a Gateway.
6. Benchmark latency, throughput, and adapter distribution.
7. Expose Prometheus metrics.

## Request Flow

```text
Client
-> FastAPI Gateway /v1/task/chat
-> TaskRouter maps task to adapter
-> VLLMBackend calls /v1/chat/completions
-> vLLM serves base model or LoRA adapter
-> Gateway records Prometheus metrics
-> Client receives wrapped response
```

## Design Choices

- **Code + Math instead of code-review/bugfix**: public datasets are easier to use and evaluation is more objective.
- **Qwen2.5-1.5B-Instruct**: low-cost enough for a single GPU, while still relevant for coding and math.
- **QLoRA**: makes adapter training feasible on limited VRAM.
- **vLLM OpenAI API**: realistic serving interface and LoRA module support.
- **FastAPI Gateway**: keeps routing, metrics, and backend details outside clients.

## V1 Boundaries

V1 does not include SGLang, Kubernetes, DPO/RLHF, prefix-cache-aware routing, or large-scale HumanEval. Those are V1.5/V2 extensions after the core chain is proven.


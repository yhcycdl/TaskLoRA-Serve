from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.analyze_results import percentile, summarize_records
from evaluation.eval_utils import extract_code_block, extract_gsm8k_answer
from pydantic import ValidationError
from registry.model_registry import ModelRegistry
from serving.mock_vllm import make_chat_completion
from serving.request_schema import TaskChatRequest
from serving.router import TaskRouter
from training.build_dataset import normalize_codealpaca, normalize_gsm8k, split_records
from training.train_utils import validate_messages


def test_dataset_normalizers() -> None:
    code = normalize_codealpaca(
        {
            "instruction": "Write add(a, b).",
            "input": "",
            "output": "def add(a, b):\n    return a + b",
        }
    )
    assert code is not None
    assert code["task"] == "code"
    assert validate_messages(code)

    math = normalize_gsm8k(
        {"question": "What is 2+2?", "answer": "2+2=4\n#### 4"},
        split="train",
    )
    assert math is not None
    assert math["task"] == "math"
    assert validate_messages(math)

    train, valid, test = split_records([{"i": i} for i in range(100)], seed=42)
    assert len(train) == 90
    assert len(valid) == 5
    assert len(test) == 5


def test_eval_helpers() -> None:
    assert extract_gsm8k_answer("Reasoning...\n#### 1,234") == "1234"
    assert extract_gsm8k_answer("The final answer is -7.") == "-7"
    assert extract_code_block("```python\ndef f():\n    return 1\n```").startswith("def f")


def test_registry_and_router() -> None:
    registry = ModelRegistry("registry/adapters.yaml")
    router = TaskRouter(
        registry=registry,
        task_routes={"code": "code-lora", "math": "math-lora", "general": "base"},
        base_model_alias="base",
        base_model_serving_name="Qwen/Qwen2.5-1.5B-Instruct",
    )
    assert router.route("code").serving_model == "code-lora"
    assert router.route("math").serving_model == "math-lora"
    assert router.route("general").serving_model == "Qwen/Qwen2.5-1.5B-Instruct"


def test_request_schema() -> None:
    request = TaskChatRequest(
        task="math",
        messages=[{"role": "user", "content": "What is 2+2?"}],
    )
    assert request.max_tokens == 512

    try:
        TaskChatRequest(
            task="math",
            messages=[{"role": "user", "content": "What is 2+2?"}],
            stream=True,
        )
    except ValidationError:
        pass
    else:
        raise AssertionError("stream=True should be rejected in V1")


def test_benchmark_summary() -> None:
    assert percentile([1, 2, 3, 4], 50) == 2.5
    summary = summarize_records(
        [
            {
                "ok": True,
                "task": "code",
                "adapter": "code-lora",
                "latency_ms": 100,
                "tokens": 10,
                "started_at": 1.0,
                "ended_at": 1.1,
            },
            {
                "ok": False,
                "task": "math",
                "adapter": "math-lora",
                "latency_ms": 200,
                "tokens": 0,
                "error": "timeout",
                "started_at": 1.0,
                "ended_at": 1.2,
            },
        ]
    )
    assert summary["requests"] == 2
    assert summary["ok"] == 1
    assert summary["error_rate"] == 0.5


def test_mock_vllm_response_shape() -> None:
    response = make_chat_completion(
        {
            "model": "math-lora",
            "messages": [{"role": "user", "content": "What is 6 times 7?"}],
            "max_tokens": 64,
        }
    )
    assert response["model"] == "math-lora"
    assert response["choices"][0]["message"]["role"] == "assistant"
    assert response["usage"]["total_tokens"] > 0


if __name__ == "__main__":
    test_dataset_normalizers()
    test_eval_helpers()
    test_registry_and_router()
    test_request_schema()
    test_benchmark_summary()
    test_mock_vllm_response_shape()
    print("smoke tests passed")

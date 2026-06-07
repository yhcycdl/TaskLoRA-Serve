from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from tqdm import tqdm

from evaluation.eval_utils import (
    extract_code_block,
    generate_response,
    load_model_and_tokenizer,
    load_yaml,
    write_jsonl,
    write_summary,
)

CODE_SYSTEM = (
    "You are an expert Python programming assistant. Return only the Python code needed "
    "to solve the task."
)


def load_mbpp(dataset_id: str, config: str, split: str, limit: int | None) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install the 'datasets' package to run evaluation.") from exc
    dataset = load_dataset(dataset_id, config, split=split)
    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))
    return list(dataset)


def run_python_tests(code: str, tests: list[str], timeout_seconds: int) -> tuple[bool, str]:
    script = "\n".join(
        [
            "from typing import *",
            "import math",
            "import itertools",
            "import collections",
            code,
            *tests,
        ]
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = Path(tmpdir) / "candidate.py"
        script_path.write_text(script, encoding="utf-8")
        completed = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    passed = completed.returncode == 0
    detail = completed.stdout.strip() or completed.stderr.strip()
    return passed, detail


def run_mbpp_eval(
    config_path: str,
    model_name: str,
    adapter_path: str | None,
    limit: int | None,
    output_path: str | None,
    load_in_4bit: bool,
) -> dict[str, Any]:
    config = load_yaml(config_path)
    code_cfg = config["code"]
    records = load_mbpp(
        code_cfg["dataset"],
        code_cfg.get("dataset_config", "sanitized"),
        code_cfg.get("split", "test"),
        limit if limit is not None else code_cfg.get("limit"),
    )
    model, tokenizer = load_model_and_tokenizer(
        config["base_model"],
        adapter_path=adapter_path,
        load_in_4bit=load_in_4bit,
    )

    results: list[dict[str, Any]] = []
    passed_count = 0
    timeout_seconds = int(code_cfg.get("timeout_seconds", 8))
    for item in tqdm(records, desc=f"MBPP eval: {model_name}"):
        prompt = (
            f"Write a Python function for the following task.\n\nTask:\n{item['text']}\n\n"
            "Return only code."
        )
        prediction_text = generate_response(
            model,
            tokenizer,
            [{"role": "system", "content": CODE_SYSTEM}, {"role": "user", "content": prompt}],
            max_new_tokens=int(config.get("max_new_tokens", 512)),
            temperature=float(config.get("temperature", 0.0)),
        )
        code = extract_code_block(prediction_text)
        tests = list(item.get("test_list") or [])
        try:
            passed, detail = run_python_tests(code, tests, timeout_seconds)
        except subprocess.TimeoutExpired:
            passed, detail = False, "timeout"
        passed_count += int(passed)
        results.append(
            {
                "model": model_name,
                "task_id": item.get("task_id"),
                "text": item["text"],
                "passed": passed,
                "detail": detail,
                "prediction_text": prediction_text,
                "code": code,
                "tests": tests,
            }
        )

    pass_at_1 = passed_count / len(records) if records else 0.0
    out = output_path or code_cfg.get("output", "report/results/mbpp_eval.jsonl")
    out_path = Path(out)
    if out_path.suffix == ".jsonl":
        model_out = out_path.with_name(f"{out_path.stem}_{model_name}{out_path.suffix}")
    else:
        model_out = out_path
    write_jsonl(model_out, results)
    summary = {
        "task": "code",
        "model": model_name,
        "adapter_path": adapter_path,
        "count": len(records),
        "passed": passed_count,
        "pass_at_1": pass_at_1,
        "results_path": str(model_out),
    }
    write_summary(model_out.with_suffix(".summary.json"), summary)
    print(summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate MBPP pass@1.")
    parser.add_argument("--config", default="configs/eval.yaml")
    parser.add_argument("--model", default="base")
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--load-in-4bit", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_mbpp_eval(
        config_path=args.config,
        model_name=args.model,
        adapter_path=args.adapter_path,
        limit=args.limit,
        output_path=args.output,
        load_in_4bit=args.load_in_4bit,
    )


if __name__ == "__main__":
    main()


from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from tqdm import tqdm

from evaluation.eval_utils import (
    exact_match,
    extract_gsm8k_answer,
    generate_response,
    load_model_and_tokenizer,
    load_yaml,
    write_jsonl,
    write_summary,
)

MATH_SYSTEM = (
    "You are a careful math reasoning assistant. Solve the problem step by step and "
    "include the final answer."
)


def load_gsm8k(dataset_id: str, config: str, split: str, limit: int | None) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install the 'datasets' package to run evaluation.") from exc
    dataset = load_dataset(dataset_id, config, split=split)
    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))
    return list(dataset)


def run_math_eval(
    config_path: str,
    model_name: str,
    adapter_path: str | None,
    limit: int | None,
    output_path: str | None,
    load_in_4bit: bool,
) -> dict[str, Any]:
    config = load_yaml(config_path)
    math_cfg = config["math"]
    records = load_gsm8k(
        math_cfg["dataset"],
        math_cfg.get("dataset_config", "main"),
        math_cfg.get("split", "test"),
        limit if limit is not None else math_cfg.get("limit"),
    )
    model, tokenizer = load_model_and_tokenizer(
        config["base_model"],
        adapter_path=adapter_path,
        load_in_4bit=load_in_4bit,
    )

    results: list[dict[str, Any]] = []
    correct = 0
    for item in tqdm(records, desc=f"GSM8K eval: {model_name}"):
        messages = [
            {"role": "system", "content": MATH_SYSTEM},
            {"role": "user", "content": item["question"]},
        ]
        prediction_text = generate_response(
            model,
            tokenizer,
            messages,
            max_new_tokens=int(config.get("max_new_tokens", 512)),
            temperature=float(config.get("temperature", 0.0)),
        )
        prediction = extract_gsm8k_answer(prediction_text)
        target = extract_gsm8k_answer(item["answer"])
        is_correct = exact_match(prediction, target)
        correct += int(is_correct)
        results.append(
            {
                "model": model_name,
                "question": item["question"],
                "target": target,
                "prediction": prediction,
                "correct": is_correct,
                "prediction_text": prediction_text,
            }
        )

    accuracy = correct / len(records) if records else 0.0
    out = output_path or math_cfg.get("output", "report/results/gsm8k_eval.jsonl")
    out_path = Path(out)
    if out_path.suffix == ".jsonl":
        model_out = out_path.with_name(f"{out_path.stem}_{model_name}{out_path.suffix}")
    else:
        model_out = out_path
    write_jsonl(model_out, results)
    summary = {
        "task": "math",
        "model": model_name,
        "adapter_path": adapter_path,
        "count": len(records),
        "correct": correct,
        "accuracy": accuracy,
        "results_path": str(model_out),
    }
    write_summary(model_out.with_suffix(".summary.json"), summary)
    print(summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate GSM8K exact match.")
    parser.add_argument("--config", default="configs/eval.yaml")
    parser.add_argument("--model", default="base")
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--load-in-4bit", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_math_eval(
        config_path=args.config,
        model_name=args.model,
        adapter_path=args.adapter_path,
        limit=args.limit,
        output_path=args.output,
        load_in_4bit=args.load_in_4bit,
    )


if __name__ == "__main__":
    main()


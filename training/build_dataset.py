from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from training.train_utils import validate_messages, write_jsonl

CODE_SYSTEM = (
    "You are an expert programming assistant. Write correct, concise code and explain "
    "important implementation details when useful."
)

MATH_SYSTEM = (
    "You are a careful math reasoning assistant. Solve the problem step by step and "
    "include the final answer."
)


def normalize_codealpaca(record: dict[str, Any]) -> dict[str, Any] | None:
    instruction = str(record.get("instruction") or "").strip()
    input_text = str(record.get("input") or "").strip()
    output = str(record.get("output") or "").strip()
    if not instruction or not output:
        return None

    user = instruction
    if input_text:
        user = f"{instruction}\n\nInput:\n{input_text}"

    result = {
        "task": "code",
        "source": "sahil2801/CodeAlpaca-20k",
        "messages": [
            {"role": "system", "content": CODE_SYSTEM},
            {"role": "user", "content": user},
            {"role": "assistant", "content": output},
        ],
        "metadata": {
            "instruction_len": len(instruction),
            "input_len": len(input_text),
            "output_len": len(output),
        },
    }
    return result if validate_messages(result) else None


def normalize_gsm8k(record: dict[str, Any], split: str) -> dict[str, Any] | None:
    question = str(record.get("question") or "").strip()
    answer = str(record.get("answer") or "").strip()
    if not question or not answer:
        return None

    result = {
        "task": "math",
        "source": "openai/gsm8k",
        "messages": [
            {"role": "system", "content": MATH_SYSTEM},
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ],
        "metadata": {"split": split, "answer_len": len(answer)},
    }
    return result if validate_messages(result) else None


def split_records(
    records: list[dict[str, Any]],
    seed: int,
    train_ratio: float = 0.9,
    valid_ratio: float = 0.05,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1")
    if not 0 <= valid_ratio < 1:
        raise ValueError("valid_ratio must be between 0 and 1")
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    if len(shuffled) >= 3:
        train_end = max(1, int(len(shuffled) * train_ratio))
        valid_count = max(1, int(len(shuffled) * valid_ratio))
        if train_end + valid_count >= len(shuffled):
            train_end = len(shuffled) - 2
            valid_count = 1
        valid_end = train_end + valid_count
    else:
        train_end = len(shuffled)
        valid_end = len(shuffled)
    return shuffled[:train_end], shuffled[train_end:valid_end], shuffled[valid_end:]


def take_limit(records: list[dict[str, Any]], limit: int | None) -> list[dict[str, Any]]:
    if limit is None:
        return records
    return records[: max(0, limit)]


def load_local_records(path: str | Path) -> list[dict[str, Any]]:
    local_path = Path(path)
    if not local_path.exists():
        raise FileNotFoundError(f"Local dataset file does not exist: {local_path}")

    if local_path.suffix == ".jsonl":
        records: list[dict[str, Any]] = []
        with local_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    if local_path.suffix == ".json":
        with local_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for key in ("train", "data", "records"):
                value = data.get(key)
                if isinstance(value, list):
                    return value
        raise ValueError(f"Unsupported JSON dataset shape in {local_path}")

    raise ValueError(f"Unsupported local dataset extension: {local_path.suffix}")


def load_hf_dataset(dataset_id: str, config: str | None = None):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "The 'datasets' package is required. Install dependencies with "
            "`pip install -r requirements.txt`."
        ) from exc

    if config:
        return load_dataset(dataset_id, config)
    return load_dataset(dataset_id)


def build_code_dataset(args: argparse.Namespace) -> dict[str, int]:
    if args.code_local_file:
        raw_train = load_local_records(args.code_local_file)
    else:
        dataset = load_hf_dataset(args.code_dataset)
        raw_train = list(dataset["train"])
    raw_train = take_limit(raw_train, args.code_limit)
    records = [r for item in raw_train if (r := normalize_codealpaca(item))]
    train, valid, test = split_records(records, seed=args.seed)

    output_dir = Path(args.output_dir)
    counts = {
        "code_train": write_jsonl(output_dir / "code_train.jsonl", train),
        "code_valid": write_jsonl(output_dir / "code_valid.jsonl", valid),
        "code_test": write_jsonl(output_dir / "code_test.jsonl", test),
    }
    return counts


def build_math_dataset(args: argparse.Namespace) -> dict[str, int]:
    if args.math_local_train_file:
        raw_train = load_local_records(args.math_local_train_file)
        if args.math_local_test_file:
            raw_test = load_local_records(args.math_local_test_file)
        else:
            _, _, raw_test = split_records(raw_train, seed=args.seed, train_ratio=0.9, valid_ratio=0.05)
    else:
        dataset = load_hf_dataset(args.math_dataset, args.math_config)
        raw_train = list(dataset["train"])
        raw_test = list(dataset["test"])

    raw_train = take_limit(raw_train, args.math_limit)
    raw_test = take_limit(raw_test, args.math_limit)

    train_records = [r for item in raw_train if (r := normalize_gsm8k(item, "train"))]
    test_records = [r for item in raw_test if (r := normalize_gsm8k(item, "test"))]
    train, valid, _ = split_records(train_records, seed=args.seed, train_ratio=0.9, valid_ratio=0.1)

    output_dir = Path(args.output_dir)
    counts = {
        "math_train": write_jsonl(output_dir / "math_train.jsonl", train),
        "math_valid": write_jsonl(output_dir / "math_valid.jsonl", valid),
        "math_test": write_jsonl(output_dir / "math_test.jsonl", test_records),
    }
    return counts


def print_summary(counts: dict[str, int]) -> None:
    task_counts = Counter()
    for name, count in counts.items():
        task_counts[name] = count
    print("Dataset build complete:")
    for name, count in sorted(task_counts.items()):
        print(f"  {name}: {count}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build TaskLoRA chat-format datasets.")
    parser.add_argument("--task", choices=["all", "code", "math"], default="all")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--code-dataset", default="sahil2801/CodeAlpaca-20k")
    parser.add_argument("--code-local-file", default=None)
    parser.add_argument("--code-limit", type=int, default=None)
    parser.add_argument("--math-dataset", default="openai/gsm8k")
    parser.add_argument("--math-config", default="main")
    parser.add_argument("--math-local-train-file", default=None)
    parser.add_argument("--math-local-test-file", default=None)
    parser.add_argument("--math-limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    counts: dict[str, int] = {}
    if args.task in {"all", "code"}:
        counts.update(build_code_dataset(args))
    if args.task in {"all", "math"}:
        counts.update(build_math_dataset(args))
    print_summary(counts)


if __name__ == "__main__":
    main()

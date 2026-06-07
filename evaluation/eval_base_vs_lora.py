from __future__ import annotations

import argparse
from pathlib import Path

from evaluation.eval_math import run_math_eval
from evaluation.eval_mbpp import run_mbpp_eval
from evaluation.eval_utils import load_yaml, write_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run configured base-vs-LoRA evals.")
    parser.add_argument("--config", default="configs/eval.yaml")
    parser.add_argument("--load-in-4bit", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    summaries = []

    summaries.append(
        run_math_eval(
            config_path=args.config,
            model_name="base",
            adapter_path=None,
            limit=config["math"].get("limit"),
            output_path=config["math"].get("output"),
            load_in_4bit=args.load_in_4bit,
        )
    )
    summaries.append(
        run_math_eval(
            config_path=args.config,
            model_name="math-lora",
            adapter_path=config["models"]["math-lora"]["adapter_path"],
            limit=config["math"].get("limit"),
            output_path=config["math"].get("output"),
            load_in_4bit=args.load_in_4bit,
        )
    )
    summaries.append(
        run_mbpp_eval(
            config_path=args.config,
            model_name="base",
            adapter_path=None,
            limit=config["code"].get("limit"),
            output_path=config["code"].get("output"),
            load_in_4bit=args.load_in_4bit,
        )
    )
    summaries.append(
        run_mbpp_eval(
            config_path=args.config,
            model_name="code-lora",
            adapter_path=config["models"]["code-lora"]["adapter_path"],
            limit=config["code"].get("limit"),
            output_path=config["code"].get("output"),
            load_in_4bit=args.load_in_4bit,
        )
    )

    output = Path("report/results/eval_summary.json")
    write_summary(output, {"summaries": summaries})
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()


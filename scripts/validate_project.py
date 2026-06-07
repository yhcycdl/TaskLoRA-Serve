from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]

REQUIRED_FILES = [
    "README.md",
    "PROJECT_CODE_GUIDE.md",
    "RUNBOOK.md",
    "requirements.txt",
    "requirements-gateway.txt",
    "configs/code_lora.yaml",
    "configs/math_lora.yaml",
    "configs/eval.yaml",
    "configs/serving.yaml",
    "registry/adapters.yaml",
    "training/build_dataset.py",
    "training/train_qlora.py",
    "evaluation/eval_math.py",
    "evaluation/eval_mbpp.py",
    "serving/gateway.py",
    "serving/mock_vllm.py",
    "benchmark/loadgen.py",
    "observability/metrics.py",
    "tests/smoke_test.py",
]


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def check_required_files() -> list[str]:
    errors: list[str] = []
    for relative in REQUIRED_FILES:
        if not (PROJECT_ROOT / relative).exists():
            errors.append(f"missing required file: {relative}")
    return errors


def check_registry() -> list[str]:
    errors: list[str] = []
    registry = load_yaml(PROJECT_ROOT / "registry/adapters.yaml")
    adapters = registry.get("adapters") or {}
    for adapter_name in ("code-lora", "math-lora"):
        adapter = adapters.get(adapter_name)
        if not adapter:
            errors.append(f"registry missing adapter: {adapter_name}")
            continue
        if adapter.get("serving_name") != adapter_name:
            errors.append(f"{adapter_name} serving_name should be {adapter_name}")
        if adapter.get("task") not in {"code", "math"}:
            errors.append(f"{adapter_name} has unexpected task: {adapter.get('task')}")
    return errors


def check_serving_config() -> list[str]:
    errors: list[str] = []
    serving = load_yaml(PROJECT_ROOT / "configs/serving.yaml")
    routes = serving.get("task_routes") or {}
    expected = {"code": "code-lora", "math": "math-lora", "general": "base"}
    for task, adapter in expected.items():
        if routes.get(task) != adapter:
            errors.append(f"task_routes.{task} should be {adapter}")
    if not serving.get("base_model_serving_name"):
        errors.append("serving config missing base_model_serving_name")
    return errors


def check_training_configs() -> list[str]:
    errors: list[str] = []
    for relative in ("configs/code_lora.yaml", "configs/math_lora.yaml"):
        config = load_yaml(PROJECT_ROOT / relative)
        lora = config.get("lora") or {}
        qlora = config.get("qlora") or {}
        if config.get("base_model") != "Qwen/Qwen2.5-1.5B-Instruct":
            errors.append(f"{relative} uses unexpected base_model")
        if int(lora.get("r", 0)) <= 0:
            errors.append(f"{relative} lora.r must be positive")
        if not qlora.get("enabled", False):
            errors.append(f"{relative} qlora.enabled should be true for V1")
    return errors


def check_adapter_outputs(require_outputs: bool) -> list[str]:
    if not require_outputs:
        return []
    errors: list[str] = []
    for relative in ("outputs/code-lora/adapter_config.json", "outputs/math-lora/adapter_config.json"):
        if not (PROJECT_ROOT / relative).exists():
            errors.append(f"missing trained adapter artifact: {relative}")
    return errors


def run_checks(require_outputs: bool) -> dict[str, Any]:
    errors: list[str] = []
    errors.extend(check_required_files())
    errors.extend(check_registry())
    errors.extend(check_serving_config())
    errors.extend(check_training_configs())
    errors.extend(check_adapter_outputs(require_outputs))
    return {
        "ok": not errors,
        "python": sys.version.split()[0],
        "project_root": str(PROJECT_ROOT),
        "require_outputs": require_outputs,
        "errors": errors,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate TaskLoRA-Serve project structure/configs.")
    parser.add_argument("--require-outputs", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_checks(require_outputs=args.require_outputs)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"Project root: {result['project_root']}")
        print(f"Python: {result['python']}")
        if result["ok"]:
            print("Project validation passed.")
        else:
            print("Project validation failed:")
            for error in result["errors"]:
                print(f"  - {error}")
    raise SystemExit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()


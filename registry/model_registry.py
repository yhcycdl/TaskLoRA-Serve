from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class AdapterInfo:
    name: str
    serving_name: str
    task: str
    path: str
    base_model: str
    status: str
    dataset: str | None = None
    eval_score: float | None = None


class ModelRegistry:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.raw = self._load()
        self.base_models = self.raw.get("base_models", {})
        self.adapters = {
            name: self._adapter_from_config(name, config)
            for name, config in self.raw.get("adapters", {}).items()
        }

    def _load(self) -> dict[str, Any]:
        with self.path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    @staticmethod
    def _adapter_from_config(name: str, config: dict[str, Any]) -> AdapterInfo:
        return AdapterInfo(
            name=config.get("name", name),
            serving_name=config.get("serving_name", config.get("name", name)),
            task=config["task"],
            path=config["path"],
            base_model=config["base_model"],
            status=config.get("status", "pending"),
            dataset=config.get("dataset"),
            eval_score=config.get("eval_score"),
        )

    def list_adapters(self) -> list[AdapterInfo]:
        return list(self.adapters.values())

    def get_adapter(self, name: str) -> AdapterInfo:
        try:
            return self.adapters[name]
        except KeyError as exc:
            raise KeyError(f"Unknown adapter: {name}") from exc

    def get_adapter_by_task(self, task: str) -> AdapterInfo:
        matches = [adapter for adapter in self.adapters.values() if adapter.task == task]
        if not matches:
            raise KeyError(f"No adapter registered for task: {task}")
        ready = [adapter for adapter in matches if adapter.status in {"ready", "pending"}]
        return ready[0] if ready else matches[0]

    def to_dict(self) -> dict[str, Any]:
        return {
            "base_models": self.base_models,
            "adapters": [adapter.__dict__ for adapter in self.list_adapters()],
        }


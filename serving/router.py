from __future__ import annotations

from dataclasses import dataclass

from registry.model_registry import AdapterInfo, ModelRegistry


@dataclass(frozen=True)
class RouteDecision:
    task: str
    adapter: str
    serving_model: str
    reason: str


class TaskRouter:
    def __init__(
        self,
        registry: ModelRegistry,
        task_routes: dict[str, str],
        base_model_alias: str,
        base_model_serving_name: str,
    ):
        self.registry = registry
        self.task_routes = task_routes
        self.base_model_alias = base_model_alias
        self.base_model_serving_name = base_model_serving_name

    def route(self, task: str) -> RouteDecision:
        adapter_name = self.task_routes.get(task)
        if adapter_name is None:
            raise KeyError(f"Unknown task route: {task}")
        if adapter_name == self.base_model_alias or task == "general":
            return RouteDecision(
                task=task,
                adapter=self.base_model_alias,
                serving_model=self.base_model_serving_name,
                reason="general task uses base model",
            )

        adapter: AdapterInfo = self.registry.get_adapter(adapter_name)
        return RouteDecision(
            task=task,
            adapter=adapter.name,
            serving_model=adapter.serving_name,
            reason=f"task={task} mapped to adapter={adapter.name}",
        )

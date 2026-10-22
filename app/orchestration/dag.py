from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Task:
    name: str
    fn: Callable[[dict[str, Any]], Any]
    depends_on: tuple[str, ...] = ()
    retries: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


class DAGError(RuntimeError):
    pass


class DAG:
    def __init__(self, tasks: Iterable[Task] = ()):
        self.tasks = {task.name: task for task in tasks}

    def add(self, task: Task) -> DAG:
        if task.name in self.tasks:
            raise DAGError(f"Duplicate task: {task.name}")
        self.tasks[task.name] = task
        return self

    def validate(self) -> None:
        for task in self.tasks.values():
            for dep in task.depends_on:
                if dep not in self.tasks:
                    raise DAGError(f"Missing dependency: {task.name}->{dep}")
        visiting, visited = set(), set()

        def visit(name: str) -> None:
            if name in visited:
                return
            if name in visiting:
                raise DAGError(f"Cycle detected at {name}")
            visiting.add(name)
            for dep in self.tasks[name].depends_on:
                visit(dep)
            visiting.remove(name)
            visited.add(name)

        for name in self.tasks:
            visit(name)

    def topological(self) -> list[str]:
        self.validate()
        done: set[str] = set()
        order: list[str] = []
        while len(order) < len(self.tasks):
            before = len(order)
            for name, task in self.tasks.items():
                if name not in done and set(task.depends_on) <= done:
                    order.append(name)
                    done.add(name)
            if len(order) == before:
                raise DAGError("Unable to schedule DAG")
        return order

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from .dag import DAG


class DAGRunner:
    def run(self, dag: DAG, context: dict[str, Any] | None = None) -> dict[str, Any]:
        ctx = dict(context or {})
        results = {}
        events = []
        started = datetime.now(UTC)
        run_id = uuid4().hex
        for name in dag.topological():
            task = dag.tasks[name]
            attempts = 0
            while True:
                attempts += 1
                try:
                    value = task.fn(ctx)
                    results[name] = value
                    ctx[name] = value
                    events.append({"task": name, "status": "completed", "attempts": attempts})
                    break
                except Exception as exc:
                    events.append(
                        {
                            "task": name,
                            "status": "retry" if attempts <= task.retries else "failed",
                            "attempts": attempts,
                            "error": str(exc),
                        }
                    )
                    if attempts > task.retries:
                        raise
        return {
            "run_id": run_id,
            "started_at": started.isoformat(),
            "finished_at": datetime.now(UTC).isoformat(),
            "tasks": results,
            "events": events,
        }

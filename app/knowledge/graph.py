from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class Edge:
    source: str
    relation: str
    target: str


class KnowledgeGraph:
    def __init__(self):
        self.nodes: set[str] = set()
        self.edges: list[Edge] = []

    def add_node(self, node: str) -> None:
        self.nodes.add(node)

    def add_edge(self, source: str, relation: str, target: str) -> None:
        self.nodes.update((source, target))
        self.edges.append(Edge(source, relation, target))

    def neighbors(self, node: str) -> list[dict[str, str]]:
        return [{"relation": e.relation, "target": e.target} for e in self.edges if e.source == node]

    def shortest_path(self, source: str, target: str) -> list[str]:
        q = deque([(source, [source])])
        seen = {source}
        while q:
            node, path = q.popleft()
            if node == target:
                return path
            for edge in self.edges:
                if edge.source == node and edge.target not in seen:
                    seen.add(edge.target)
                    q.append((edge.target, path + [edge.target]))
        return []

    def manifest(self) -> dict[str, Any]:
        return {
            "nodes": sorted(self.nodes),
            "edges": [{"source": e.source, "relation": e.relation, "target": e.target} for e in self.edges],
        }


def lineage_graph(records: list[dict[str, Any]]) -> KnowledgeGraph:
    """Grafo de linhagem montado a partir dos registros reais do registry de modelos."""
    graph = KnowledgeGraph()
    for record in records:
        model = f"model:{record['alias']}"
        metadata = record.get("metadata") or {}
        if metadata.get("dataset_hash"):
            graph.add_edge(f"dataset:{metadata['dataset_hash'][:12]}", "trained", model)
        if metadata.get("training_run_id") is not None:
            graph.add_edge(f"training_run:{metadata['training_run_id']}", "produced", model)
        graph.add_edge(model, "in_stage", f"stage:{record['stage']}")
    return graph

"""Dependency graph across migration repo objects.

Dependencies are inferred by matching an object's input tables against other
objects' output tables — i.e. object B depends on object A if A writes a
table that B reads. This is what drives topological conversion order: leaf
objects (no dependencies) convert first, so callers get already-converted
context.
"""

from __future__ import annotations

import networkx as nx

from repo.metadata import ObjectMetadata


def infer_dependencies(metadatas: list[ObjectMetadata]) -> dict[str, list[str]]:
    """Return {object_name: [names of objects it depends on]}."""
    owner_of_table: dict[str, str] = {}
    for m in metadatas:
        for table in m.output_tables:
            owner_of_table[table] = m.name

    depends_on: dict[str, list[str]] = {}
    for m in metadatas:
        deps = {
            owner_of_table[table]
            for table in m.input_tables
            if table in owner_of_table and owner_of_table[table] != m.name
        }
        depends_on[m.name] = sorted(deps)
    return depends_on


class CyclicDependencyError(ValueError):
    pass


class DependencyGraph:
    """Wraps a networkx DiGraph of migration-repo objects.

    Edge direction: dependency -> dependent (A -> B means B depends on A),
    which matches networkx's topological_sort producing dependencies first.
    """

    def __init__(self, metadatas: list[ObjectMetadata]) -> None:
        self._graph: nx.DiGraph = nx.DiGraph()
        depends_on = infer_dependencies(metadatas)

        for m in metadatas:
            self._graph.add_node(m.name)
        for name, deps in depends_on.items():
            for dep in deps:
                self._graph.add_edge(dep, name)

        if not nx.is_directed_acyclic_graph(self._graph):
            cycles = list(nx.simple_cycles(self._graph))
            raise CyclicDependencyError(f"Dependency graph has cycles: {cycles}")

    def topological_order(self) -> list[str]:
        """Object names ordered so every dependency precedes its dependents."""
        return list(nx.topological_sort(self._graph))

    def dependencies_of(self, name: str) -> list[str]:
        return sorted(self._graph.predecessors(name))

    def dependents_of(self, name: str) -> list[str]:
        return sorted(self._graph.successors(name))

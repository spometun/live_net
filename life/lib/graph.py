import pytest
from typing import List
import abc

from life.lib.simple_log import LOG


class GraphNode:

    @abc.abstractmethod
    def get_adjacent_nodes(self) -> List["GraphNode"]: ...

    def visit(self, function_name: str):
        visited_ids = set()
        visited_nodes = list()  # make sure that nodes (potentially) removed during visit are still referenced,
        # which guarantees unique ids with (potentially) created other nodes during visit
        self._visit(function_name, visited_ids=visited_ids, visited_nodes=visited_nodes)
        del visited_ids
        del visited_nodes

    def _visit(self, function_name: str, visited_ids: set, visited_nodes: set):
        if id(self) in visited_ids:
            return
        try:
            exec(f"self.{function_name}()")
        except AttributeError:
            pass
        visited_ids.add(id(self))
        visited_nodes.append(self)
        for node in self.get_adjacent_nodes():
            node._visit(function_name, visited_ids=visited_ids, visited_nodes=visited_nodes)


class NodesHolder(GraphNode):
    def __init__(self, nodes: List[GraphNode]):
        self.nodes = nodes

    def get_adjacent_nodes(self) -> List[GraphNode]:
        return self.nodes


def test_graph():
    class GL(GraphNode):
        _call_counter = 0

        def __init__(self, nodes, num):
            super().__init__()
            self.nodes = nodes
            self.num = num

        def print(self):
            print(f"{self.num} {super()._global_counter}")

        @staticmethod
        def func():
            GL._call_counter += 1

        def get_adjacent_nodes(self) -> List["GraphNode"]:
            return self.nodes

    n1 = GL([], 1)
    n2 = GL([], 2)
    n3 = GL([n1, n2], 3)
    n4 = GL([n1, n2], 4)
    n5 = GL([n3, n4], 5)
    n5.visit("func")
    assert GL._call_counter == 5
    n5.visit("func")
    assert GL._call_counter == 10

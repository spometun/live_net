import pytest
from typing import List
import abc


class GraphNode:
    _global_counter = 0

    def __init__(self):
        self._counter = 0

    @abc.abstractmethod
    def get_adjacent_nodes(self) -> List["GraphNode"]: ...

    def visit(self, function_name: str):
        GraphNode._global_counter += 1
        self._visit(function_name)

    def _visit(self, function_name: str):
        assert self._counter == GraphNode._global_counter or self._counter + 1 == GraphNode._global_counter, \
               "Internal error (this may happen because of nested call of GraphNode.visit() which is not allowed"
        if self._counter == GraphNode._global_counter:
            return
        for node in self.get_adjacent_nodes():
            node._visit(function_name)
        exec(f"self.{function_name}()")
        self._counter = GraphNode._global_counter


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




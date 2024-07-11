import pytest
from typing import List, Callable
import abc

from ai_libs.simple_log import LOG, LOGD


class GraphNode(abc.ABC):

    @abc.abstractmethod
    def get_adjacent_nodes(self) -> List["GraphNode"]: ...

    # visited function must comply one of the following:
    # A. Do not modify graph structure
    # or
    # B. Visited function may disconnect visited node (and may disconnect some or all of its children)
    # but siblings of visited node must be intact
    # or
    # C. ... what about creation ?
    # TODO: remove visit in favor of apply_func? then may be rename -> apply_func -> visit
    def visit(self, function_name: str, *args):
        visited_ids = set()
        self._visit(function_name, args, visited_ids=visited_ids)
        del visited_ids

    def _visit(self, function_name: str, args: tuple, visited_ids: set):
        try:
            name = self.name
        except AttributeError:
            name = "NoName"
        if id(self) in visited_ids:
            # LOGD(f"{name} already visited")
            return

        # LOGD(f"visiting {name}")
        function = getattr(self, function_name, None)
        if function is not None:
            function(*args)

        visited_ids.add(id(self))
        # make a copy because self.get_adjacent_nodes() may change
        # (some visited children may be disconnected - and removed from adjacent during the visit)
        adjacent_nodes = [node for node in self.get_adjacent_nodes()]
        for node in adjacent_nodes:
            node._visit(function_name, args, visited_ids=visited_ids)

    def apply_func(self, function: Callable):
        # "function" must accept single argument - node object (which is derived from GraphNode)
        visited_ids = set()
        self._apply_func(function, visited_ids=visited_ids)
        del visited_ids

    def _apply_func(self, function: Callable, visited_ids: set):
        if id(self) in visited_ids:
            return
        function(self)
        visited_ids.add(id(self))
        # make a copy because self.get_adjacent_nodes() may change
        # (some visited children may be disconnected - and removed from adjacent during the visit)
        adjacent_nodes = [node for node in self.get_adjacent_nodes()]
        for node in adjacent_nodes:
            node._apply_func(function, visited_ids=visited_ids)


class NodesHolder(GraphNode):
    def __init__(self, name, nodes: List[GraphNode]):
        self.nodes = nodes
        self.name = name

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

        def func(self):
            GL._call_counter += 1

        def summator(self, v):
            v[0] += self.num

        def get_adjacent_nodes(self) -> List["GraphNode"]:
            return self.nodes

    n1 = GL([], 1)
    n2 = GL([], 2)
    n3 = GL([n1, n2], 3)
    n4 = GL([n1, n2], 4)
    n5 = GL([n3, n4], 5)

    def call_func_member(obj):
        obj.func()

    n5.apply_func(call_func_member)
    assert GL._call_counter == 5
    n5.apply_func(call_func_member)
    assert GL._call_counter == 10

    v = [0]

    def call_summator(obj):
        obj.summator(v)

    n5.apply_func(call_summator)
    assert v[0] == 15


def test_except():
    class N(GraphNode):
        def get_adjacent_nodes(self) -> List["GraphNode"]:
            return []
        def f(self):
            v = self.__dict__["ku43"]

    node = N()
    with pytest.raises(KeyError):
        node.visit("f")

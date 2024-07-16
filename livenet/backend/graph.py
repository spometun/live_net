import keyword
import typing

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

    @typing.final
    def visit_func(self, function: Callable):
        # "function" must accept single argument - node object (is derived from GraphNode)
        visited_ids = set()
        self._apply_func(function, visited_ids=visited_ids)
        del visited_ids

    @typing.final
    def visit_member(self, member_name: str, *args, **kwargs):
        # calls member_name(*args, **kwargs) on all objects which have member_name defined
        func = get_call_member_func_if_exists(member_name, *args, **kwargs)
        self.visit_func(func)

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


def get_call_member_func_if_exists(member_name: str, *args, **kwargs):
    assert member_name.isidentifier() and not keyword.iskeyword(member_name), f"Invalid input member name {member_name}"

    def impl(obj):
        function = getattr(obj, member_name, None)
        if function is not None:
            function(*args, **kwargs)

    return impl


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

    n5.visit_func(call_func_member)
    assert GL._call_counter == 5
    n5.visit_func(get_call_member_func_if_exists("func"))
    assert GL._call_counter == 10

    v = [0]

    def call_summator(obj):
        obj.summator(v)

    n5.visit_member("summator", v)
    assert v[0] == 15


def test_except():
    class N(GraphNode):
        def get_adjacent_nodes(self) -> List["GraphNode"]:
            return []
        def f(self):
            v = self.ku43
        def g(self):
            pass

    node = N()
    node.visit_member("g")
    with pytest.raises(AttributeError):
        node.visit_member("f")

    with pytest.raises(AssertionError):
        node.visit_member("mu.bu")

    with pytest.raises(AssertionError):
        node.visit_member("for")

    with pytest.raises(AssertionError):
        node.visit_member("3ku")


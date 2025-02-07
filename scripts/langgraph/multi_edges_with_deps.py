from langgraph.graph import StateGraph, START, END
from typing import TypedDict
import operator
from collections.abc import Sequence
from typing import Annotated, Any, TypedDict

class State(TypedDict):
        my_val: Annotated[int, operator.add]


def check1(state):
     pass
def check2(state):
     pass
def check3(state):
     pass
def check4(state):
     pass
def check_end(state):
     pass
def pp1(state):
     pass


class ABC:

    @staticmethod
    def crisscross_simple():

        graph = StateGraph(State)
        # print(help(graph.add_node))
        # print(help(graph.add_edge))

        graph.add_node("check1", check1)
        graph.add_node("check2", check2)
        graph.add_node("check3", check3)

        graph.add_edge(START, "check1")
        graph.add_edge("check1", "check2")

        # https://github.com/langchain-ai/langgraph/issues/954
        graph.add_edge(["check1", "check2"], "check3")

        xx = graph.compile().invoke(
             State(
                  my_val=0),
                  debug=True
             )
ABC.crisscross_simple()

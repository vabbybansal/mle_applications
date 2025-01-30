from langgraph.graph import StateGraph, START, END
from typing import TypedDict
import operator
from collections.abc import Sequence
from typing import Annotated, Any, TypedDict

class State(TypedDict):
        my_val: Annotated[int, operator.add]


def xx1(state):
     input_val = state["my_val"]
     return {
          "my_val": 1
     }

def xx2(state):
     input_val = state["my_val"]
     return {
            "my_val": 2
        }
def xx3(state):
     input_val = state["my_val"]
     return {
          "my_val": 3
     }

def xx4(state):
     input_val = state["my_val"]
     return {
          "my_val": 4
     }

def xx5(state):
     input_val = state["my_val"]
     return {
          "my_val": 5
     }

def xx6(state):
     input_val = state["my_val"]
     return {
          "my_val": 6
     }

class ABC:

    @staticmethod
    def single_starts_multiple_edges():

        graph = StateGraph(State)

        graph.add_node("x1", xx1)
        graph.add_node("x2", xx2)
        graph.add_node("x3", xx3)
        graph.add_node("x4", xx4)
        graph.add_node("x5", xx5)
        graph.add_node("x6", xx6)

        graph.add_edge(START, "x1")
        # graph.add_edge(START, "x2")
        graph.add_edge("x1", "x2")
        graph.add_edge("x1", "x3")
        graph.add_edge("x2", "x4")
        graph.add_edge("x4", "x5")
        graph.add_edge("x5", "x6")
        graph.add_edge("x3", "x6")
        graph.add_edge("x6", END)
        # print(help(graph.compile().invoke))

        xx = graph.compile().invoke(
             State(
                  my_val=0),
                  debug=True
             )

    @staticmethod
    def crisscross():

        graph = StateGraph(State)

        graph.add_node("x1", xx1)
        graph.add_node("x2", xx2)
        graph.add_node("x3", xx3)
        graph.add_node("x4", xx4)

        graph.add_edge(START, "x1")
        # graph.add_edge(START, "x2")
        graph.add_edge("x1", "x2")
        graph.add_edge("x1", "x3")
        graph.add_edge("x2", "x4")
        graph.add_edge("x2", "x3")
        graph.add_edge("x3", "x4")
        graph.add_edge("x4", END)
        # print(help(graph.compile().invoke))

        xx = graph.compile().invoke(
             State(
                  my_val=0),
                  debug=True
             )

    @staticmethod
    def crisscross_complex():

        graph = StateGraph(State)

        graph.add_node("x1", xx1)
        graph.add_node("x2", xx2)
        graph.add_node("x3", xx3)
        graph.add_node("x4", xx4)
        graph.add_node("x5", xx5)

        graph.add_edge(START, "x1")
        # graph.add_edge(START, "x2")
        graph.add_edge("x1", "x2")
        graph.add_edge("x1", "x3")
        graph.add_edge("x2", "x4")
        graph.add_edge("x2", "x3")
        graph.add_edge("x3", "x4")
        graph.add_edge("x1", "x5")
        graph.add_edge("x3", "x5")
        graph.add_edge("x5", "x4")
        graph.add_edge("x4", END)
        # print(help(graph.compile().invoke))

        xx = graph.compile().invoke(
             State(
                  my_val=0),
                  debug=True
             )
ABC.crisscross_complex()

# How it works: BFS. Catch -> any repeating node at the same time step is deduped.
# For crisscross_complex:
    # 1 executes. It's neighbors [2,3,5] are added to the stack -> state: 1
    # 2,3,5 execute. state: 11. deduped neighbors added for all three nodes [3,4,5]
    # [3,4,5] execute. state: 23. deduped neighbors added for all three nodes [4,5]
    # [4,5] execute. state: 32. deduped neighbors added for all three nodes [4]
    # [4] executes. state: 36. ends

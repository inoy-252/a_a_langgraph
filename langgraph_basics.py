from typing import TypedDict

from langgraph.graph import END, START, StateGraph


class SimpleState(TypedDict, total=False):
    user_name: str
    greeting: str


def welcome_node(state: SimpleState) -> dict:
    name = state.get("user_name", "INOY")
    return {"greeting": f"Yokoso, {name}! Welcome to my Soul Society "}


if __name__ == "__main__":
    builder = StateGraph(SimpleState)
    builder.add_node("greeter", welcome_node)

    builder.add_edge(START, "greeter")
    builder.add_edge("greeter", END)

    graph = builder.compile()

    result = graph.invoke({"user_name": "Yasir"})
    print("Final state in Notebook")
    print(result)

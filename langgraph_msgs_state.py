from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, MessagesState, StateGraph

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")


def chatbot_node(state: MessagesState) -> dict:
    """Reads the messages from the notebook, calls Gemini, and returns the reply."""

    conversation = state["messages"]

    response = llm.invoke(conversation)

    return {"messages": [response]}


if __name__ == "__main__":
    builder = StateGraph(MessagesState)
    builder.add_node("chatbot", chatbot_node)
    builder.add_edge(START, "chatbot")
    builder.add_edge("chatbot", END)

    graph = builder.compile()

    user_input = (
        "Hi! My name is Yasir. Tell me a 1-sentence motivational quote about coding."
    )
    print(f"[User]: {user_input}\n")

    initial_state = {"messages": [HumanMessage(content=user_input)]}
    final_output = graph.invoke(initial_state)
    print("[Final State Messages Count]:", len(final_output["messages"]))
    print("\n[Gemini Reply]:")
    print(final_output["messages"][-1].content)

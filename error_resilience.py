from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()


@tool
def safe_divide(a: float, b: float) -> str:
    """Safely divides number a by number b."""
    try:
        return f"The result is {a / b}"
    except ZeroDivisionError:
        # Instead of crashing Python, return an explanation for Gemini!
        return "Error: Division by zero is impossible. Please explain this to the user."


tools = [safe_divide]
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")
llm_with_tools = llm.bind_tools(tools)


def chatbot_node(state: MessagesState) -> dict:
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

builder = StateGraph(MessagesState)
builder.add_node("chatbot", chatbot_node)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "chatbot")
builder.add_conditional_edges("chatbot", tools_condition)
builder.add_edge("tools", "chatbot")
agent = builder.compile()

if __name__ == "__main__":
    prompt = "Can you divide 100 by 0 for me?"
    print(f"[User]: {prompt}\n")
    result = agent.invoke({"messages": [HumanMessage(content=prompt)]})
    print("[Final Agent Answer]:")
    print(result["messages"][-1].content)
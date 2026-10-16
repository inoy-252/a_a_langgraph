from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()


@tool
def get_current_weather(city: str) -> str:
    """Get the current weather forecast for a given city."""
    mock_weather = {
        "tokyo": "19°C, Cloudy with gentle breeze",
        "paris": "15°C, Sunny and crisp",
        "london": "12°C, Light drizzle",
    }
    return mock_weather.get(city.lower(), "20°C, Clear skies")


@tool
def multiply_numbers(a: float, b: float) -> float:
    """Multiplies two numbers together and returns the result."""
    return a * b


tools = [get_current_weather, multiply_numbers]

llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")
llm_with_tools = llm.bind_tools(tools)


def chatbot_node(state: MessagesState) -> dict:
    """Takes the conversation history, calls Gemini with tools, and returns the response."""
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


if __name__ == "__main__":
    builder = StateGraph(MessagesState)

    builder.add_node("chatbot", chatbot_node)
    builder.add_node("tools", ToolNode(tools))

    builder.add_edge(START, "chatbot")

    builder.add_conditional_edges("chatbot", tools_condition)

    builder.add_edge("tools", "chatbot")

    agent = builder.compile()

    user_prompt = (
        "What is the weather in London right now, and what is 24 multiplied by 7?"
    )
    print(f"[User]: {user_prompt}\n")

    initial_input = {"messages": [HumanMessage(content=user_prompt)]}
    result = agent.invoke(initial_input)

    print("=" * 60)
    print("THE AUTONOMOUS REACT AGENT FLOW:")
    print("=" * 60)

    for i, msg in enumerate(result["messages"], start=1):
        msg_type = type(msg).__name__
        print(f"\nStep{i} [{msg_type}]:")
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            print(f"Gemini requested tools: {[tc['name'] for tc in msg.tool_calls]}")
        else:
            print(f" Content: {msg.content}")

    print("\n" + "=" * 60)
    print("[FINAL ANSWER]:")
    print(result["messages"][-1].content)
    print("=" * 60)

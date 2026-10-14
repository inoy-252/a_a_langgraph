from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

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


if __name__ == "__main__":
    tools = [get_current_weather, multiply_numbers]

    llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")

    llm_with_tools = llm.bind_tools(tools)

    query = "What is the weather in Paris, and what is 15 multiplied by 8?"
    print(f"[User]: {query}\n")

    ai_response = llm_with_tools.invoke([HumanMessage(content=query)])

    print("[AI Direct Text]:", repr(ai_response.content))
    print(f"\n[AI Requested Tool Calls] ({len(ai_response.tool_calls)} calls):")
    for tc in ai_response.tool_calls:
        print(f"   -Name: {tc['name']}")
        print(f"    Args: {tc['args']}")
        print(f"    ID:   {tc['id']}\n")
    tool_map = {t.name: t for t in tools}
    tool_messages = []

    print("[Executing Tools Locally...]")
    for tc in ai_response.tool_calls:
        tool_fn = tool_map[tc["name"]]
        result = tool_fn.invoke(tc["args"])
        print(f"   -Ran {tc['name']} -> Got: {result}")

        tool_messages.append(
            ToolMessage(
                content=str(result),
                tool_call_id=tc["id"],
            )
        )

    conversation_history = [HumanMessage(content=query), ai_response] + tool_messages

    final_response = llm_with_tools.invoke(conversation_history)

    print("\n[Gemini's Final Answer to User]:")
    print(final_response.content)

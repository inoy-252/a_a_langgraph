from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()


def get_current_weather(city: str) -> str:
    """Get the current weather forecast for a given city."""
    mock_weather_db = {
        "tokyo": "19°C, Cloudy with gentle breeze",
        "paris": "15°C, Sunny and crisp",
        "london": "12°C, Light drizzle",
    }
    return mock_weather_db.get(city.lower(), "20°C, Clear skies")


if __name__ == "__main__":
    client = genai.Client()
    user_query = "What is the weather like in Tokyo right now?"
    print(f"User: {user_query}\n")

    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=user_query,
        config=types.GenerateContentConfig(
            tools=[get_current_weather],
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                disable=True
            ),
        ),
    )
    print("What Gemini decided to return:")
    print(response.function_calls)

    if response.function_calls and response.candidates:
        tool_call = response.function_calls[0]

        tool_args = dict(tool_call.args) if tool_call.args else {}
        tool_name = str(tool_call.name)

        tool_output = get_current_weather(**tool_args)
        print(f"\n[Tool Execution]: Ran {tool_call.name} -> Got '{tool_output}'\n")

        tool_response_part = types.Part.from_function_response(
            name=tool_name,
            response={"result": tool_output},
        )

        user_content = types.Content(
            role="user", parts=[types.Part.from_text(text=user_query)]
        )
        model_tool_call_content = response.candidates[0].content
        tool_content = types.Content(role="user", parts=[tool_response_part])

        final_response = client.models.generate_content(
            model="gemini-3.5-flash-lite`",
            contents=[user_content, model_tool_call_content, tool_content],
            config=types.GenerateContentConfig(
                tools=[get_current_weather],
                automatic_function_calling=types.AutomaticFunctionCallingConfig(
                    disable=True
                ),
            ),
        )
        print("[Gemini Final Answer]:")
        print(final_response.text)
    else:
        print("[Gemini Direct Answer]:")
        print(response.text)

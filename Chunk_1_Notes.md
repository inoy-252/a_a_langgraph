# Project 3 — Chunk 1: Agent Concepts & Gemini Function Calling

## 1. The Core Mental Model: What is an AI Agent?
A regular LLM is just a **text predictor** (a brain with no hands or real-time senses). 
An **AI Agent** is an LLM equipped with **tools** (calculator, weather lookup, web search) running in an iterative loop called **ReAct** (**Reason + Act**):

1. **Perceive:** Receives the user question + conversation state.
2. **Reason:** Thinks: *"Do I know this answer, or do I need an external tool?"*
3. **Act:** Emits a structured request to call a specific Python function with extracted inputs.
4. **Observe:** Your Python program runs the function and gives the result back.
5. **Synthesize:** The LLM digests the result and writes the final natural answer.

---

## 2. The Golden Secret of Function Calling
> **The LLM NEVER executes your Python code.**
- The model only inspects the **tool's description (docstring)** and **parameter types** (`city: str`).
- The model uses natural language processing to extract values from messy human sentences.
- It returns structured JSON: `{"name": "get_current_weather", "args": {"city": "Paris"}}`.
- **Your Python program** executes the actual code and sends the answer back.

---

## 3. Why `temperature = 0.0` for AI Agents?
- **High temperature (0.7 - 1.0):** Creative, random, takes risks (great for stories/poetry).
- **Zero temperature (0.0):** Strict, deterministic, mathematically precise.
- **For Agents:** Always use `0.0`. You want 100% precision when extracting parameters and calling tools, never "creative" hallucinations.

---

## 4. The Two Approaches We Built

### Approach A: Raw Google SDK (`gemini_function_calling.py`)
- Low-level plumbing using the official `google-genai` SDK.
- Catching `response.function_calls` manually by setting `automatic_function_calling disable=True`.
- Packaging local function results into `types.Part.from_function_response(...)`.
- Role for tool outputs in Gemini is `role="user"` (sent from client side).

### Approach B: LangChain / LangGraph (`langchain_tool_binding.py`)
Replaces manual plumbing with standard abstractions:
1. **`@tool` Decorator:**
   Turns any Python function into an agent tool. Automatically builds schemas from type hints and docstrings.
2. **`.bind_tools()`:**
   Attaches the tool list permanently to the model (`llm.bind_tools(tools)`).
3. **The 3 Universal Message Types:**
   - **`HumanMessage`**: What the user said.
   - **`AIMessage`**: What Gemini replies with (contains `ai_response.tool_calls`).
   - **`ToolMessage`**: The execution result, tagged with `tool_call_id` so Gemini knows which question was answered.

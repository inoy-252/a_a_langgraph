# Project 3 — Chunk 4: Real-World Tools & Error Resilience

## 1. The Core Principle: "Never Let Tools Crash the Agent"
In production systems, tools interact with external APIs, databases, networks, and unpredictable math.
- **The Bad Pattern:** Letting exceptions like `ZeroDivisionError`, `KeyError`, or `ConnectionError` bubble up and crash the Python runtime.
- **The Senior Pattern (Self-Healing):** Wrap tool logic in `try / except`. Instead of crashing, return a descriptive error message string to the LLM.

---

## 2. How the Self-Healing Loop Works
```
1. User Prompt: "Divide 100 by 0"
          │
          ▼
2. Gemini Tool Call: safe_divide(a=100, b=0)
          │
          ▼
3. Python executes try/except:
   Catches ZeroDivisionError -> returns: "Error: Division by zero is impossible."
          │
          ▼
4. LangGraph feeds error back to Gemini as ToolMessage
          │
          ▼
5. Gemini reflects and politely explains:
   "I cannot divide 100 by 0 because division by zero is mathematically undefined."
```

---

## 3. Code Implementation Pattern
```python
@tool
def safe_divide(a: float, b: float) -> str:
    """Safely divides number a by number b."""
    try:
        return f"The result is {a / b}"
    except ZeroDivisionError:
        return "Error: Division by zero is impossible. Please explain this to the user."
```

---

## 4. Key Interview Questions

**Q: How do you handle tool execution failures in LangGraph?**
> **A:** Rather than raising uncaught exceptions that crash the server, tools should catch anticipated errors and return structured error messages back to the agent. This allows the model to self-heal: either explaining the constraint to the user or selecting an alternative tool.

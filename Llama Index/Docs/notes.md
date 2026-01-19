# Agentic AI Concepts

Definition for Agents work is a continuous cycle of: thinking (Thought) → acting (Act) and observing (Observe).

Let's break down these actions together:

## Thought-Action-Observation Cycle

- **Thought**: The LLM part of the Agent decides what the next step should be.
- **Action**: The agent takes an action by calling the tools with the associated arguments.
- **Observation**: The model reflects on the response from the tool.

### System Message Example

```
You are an AI assistant designed to help users efficiently and accurately. Your
primary goal is to provide helpful, precise, and clear responses.

You have access to the following tools:
Tool Name: calculator, Description: Multiply two integers., Arguments: a: int, b: int, Outputs: int

You should think step by step in order to fulfill the objective with a reasoning divided into
Thought/Action/Observation steps that can be repeated multiple times if needed. You should first
reflect on the current situation using 'Thought: {your_thoughts}', then (if necessary), call a
tool with the proper JSON formatting 'Action: {JSON_BLOB}', or print your final answer starting
with the prefix 'Final Answer:'
```

## Tool Usage Example

Based on its reasoning and the fact that Alfred knows about a `get_weather` tool, Alfred prepares a JSON-formatted command that calls the weather API tool:

**Thought**: I need to check the current weather for New York.

```json
{
  "action": "get_weather",
  "action_input": {
    "location": "New York"
  }
}
```

### Observation

**Feedback from the Environment:**

After the tool call, Alfred receives an observation. This might be the raw weather data from the API such as:

> "Current weather in New York: partly cloudy, 15°C, 60% humidity."

### Updated Thought

**Reflecting:**

With the observation in hand, Alfred updates its internal reasoning:

> "Now that I have the weather data for New York, I can compile an answer for the user."

---

## Internal Reasoning: Chain-of-Thought vs ReAct

Thoughts represent the Agent's internal reasoning and planning processes to solve the task.

### Chain-of-Thought (CoT)

Chain-of-Thought is a reasoning technique where the model:
- Breaks a problem into steps
- Reasons internally step-by-step
- Arrives at a final answer

| Feature | CoT |
|---------|-----|
| Internal reasoning | ✅ |
| No external tools | ✅ |
| No environment interaction | ✅ |
| Deterministic logic | ✅ |
| Cannot fetch data | ❌ |
| Cannot take actions | ❌ |

### ReAct (Reason + Act)

ReAct combines "Reasoning" (Think) with "Acting" (Act).

ReAct combines:
- **Reasoning** (Thought)
- **Action** (Tool usage)
- **Observation** (Result from tool)

Think of it as: "Think → Do something → Observe → Think again → Decide"

#### Example (ReAct)

```
Thought: I need to find the latest weather in Paris.
Action: Search["weather in Paris"]
Observation: It's 18°C and cloudy.
Thought: Now that I know the weather...
Action: Finish["It's 18°C and cloudy in Paris."]
```

| Feature | ReAct |
|---------|-------|
| Uses tools (APIs, DBs, files, calculators) | ✅ |
| Interacts with external systems | ✅ |
| Multi-step workflows | ✅ |
| More complex | ❌ |
| Slower | ❌ |
| Needs tool definitions | ❌ |

---
title: LangGraph Examples
emoji: "\U0001F99C"
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "4.0.0"
app_file: app.py
pinned: false
license: mit
tags:
  - langgraph
  - langchain
  - agents
  - llm
  - state-machines
  - workflows
---

# LangGraph Examples

A collection of practical examples demonstrating [LangGraph](https://github.com/langchain-ai/langgraph) - a library for building stateful, multi-actor applications with LLMs.

## Overview

This repository contains progressively complex examples showing how to build agentic workflows using LangGraph's StateGraph pattern.

## Examples

| Example | Description | Key Features |
|---------|-------------|--------------|
| [Basic Graph](examples/01_basic_graph/) | Introduction to state machines with conditional routing | `StateGraph`, `START/END`, conditional edges |
| [Image Tool Agent](examples/02_image_tool_agent/) | Multimodal document analysis with vision capabilities | Tool calling, vision LLM, `ToolNode` |
| [Email Classification](examples/03_email_classification/) | Spam detection and response drafting workflow | Multi-node workflow, LLM routing, two provider implementations |

## Architecture

All examples follow the LangGraph StateGraph pattern:

```
┌─────────┐     ┌─────────┐     ┌─────────┐
│  START  │────▶│  Node   │────▶│   END   │
└─────────┘     └────┬────┘     └─────────┘
                     │
              ┌──────┴──────┐
              ▼             ▼
         ┌────────┐   ┌────────┐
         │ Node A │   │ Node B │
         └────────┘   └────────┘
```

## Installation

```bash
# Clone the repository
git clone https://github.com/LordMaduz/AI-Agentic.git
cd LangGraph

# Install dependencies
pip install -r requirements.txt
```

## Environment Variables

Create a `.env` file in the root directory:

```env
# Required for Anthropic examples
ANTHROPIC_API_KEY=your_anthropic_api_key

# Required for OpenAI examples
OPENAI_API_KEY=your_openai_api_key

# Optional: Langfuse tracing
LANGFUSE_PUBLIC_KEY=your_langfuse_public_key
LANGFUSE_SECRET_KEY=your_langfuse_secret_key
LANGFUSE_HOST=https://cloud.langfuse.com
```

## Quick Start

### Basic Graph Example

```python
from examples.basic_graph.graph import graph

result = graph.invoke({"graph_state": "Hello!"})
print(result)
# Output: {'graph_state': 'Hello! I am happy!'} or {'graph_state': 'Hello! I am sad!'}
```

### Email Classification Example

```python
from examples.email_classification.spam_email_agent_openai import compiled_graph

email = {
    "sender": "john@example.com",
    "subject": "Meeting Request",
    "body": "Would you be available for a call next week?"
}

result = compiled_graph.invoke({
    "email": email,
    "is_spam": None,
    "spam_reason": None,
    "email_category": None,
    "email_draft": None,
    "messages": []
})
```

## Project Structure

```
LangGraph/
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
├── .env.example
│
└── examples/
    ├── 01_basic_graph/
    │   ├── graph.py
    │   └── README.md
    │
    ├── 02_image_tool_agent/
    │   ├── image_tool_agent.py
    │   └── README.md
    │
    └── 03_email_classification/
        ├── spam_email_agent_claude.py
        ├── spam_email_agent_openai.py
        └── README.md
```

## Key Concepts Demonstrated

### 1. State Management
All examples use `TypedDict` for type-safe state definitions:

```python
class State(TypedDict):
    graph_state: str
```

### 2. Conditional Routing
Decision-making based on state or LLM outputs:

```python
builder.add_conditional_edges("node", routing_function, {"option_a": "node_a", "option_b": "node_b"})
```

### 3. Tool Integration
Using LangGraph's prebuilt `ToolNode` for function calling:

```python
from langgraph.prebuilt import ToolNode, tools_condition
builder.add_node("tools", ToolNode(tools))
```

### 4. Observability
Integration with Langfuse for tracing and monitoring:

```python
from langfuse.langchain import CallbackHandler
langfuse_handler = CallbackHandler()
```

## Requirements

- Python 3.10+
- See `requirements.txt` for full dependency list

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Resources

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangChain Documentation](https://python.langchain.com/)
- [Langfuse Documentation](https://langfuse.com/docs)

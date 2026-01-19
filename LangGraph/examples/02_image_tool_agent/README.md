# Image Tool Agent

This example demonstrates a multimodal agent that can analyze images and perform tool calls using Claude's vision capabilities.

## Overview

An "Alfred the Butler" themed agent that can:
- Extract text from images using vision LLM
- Perform mathematical calculations
- Route between direct responses and tool execution

## Graph Structure

```
┌─────────┐
│  START  │
└────┬────┘
     │
     ▼
┌───────────┐
│ assistant │◄────┐
└─────┬─────┘     │
      │           │
      ▼           │
  ┌───────┐       │
  │ tools │───────┘
  │ needed?│
  └───┬───┘
      │ No
      ▼
┌─────────┐
│   END   │
└─────────┘
```

## Key Features

### 1. Multimodal Vision

Extract text from images using Claude's vision capabilities:

```python
def extract_text(img_path: str) -> str:
    """Extract text from an image file using a multimodal model."""
    with open(img_path, "rb") as image_file:
        image_bytes = image_file.read()
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")
    # ... invoke vision LLM
```

### 2. Tool Integration

Uses LangGraph's prebuilt `ToolNode` for automatic tool execution:

```python
from langgraph.prebuilt import ToolNode, tools_condition

tools = [divide, extract_text]
llm_with_tools = llm.bind_tools(tools, parallel_tool_calls=False)

builder.add_node("tools", ToolNode(tools))
builder.add_conditional_edges("assistant", tools_condition)
```

### 3. State Management

Tracks both input files and conversation messages:

```python
class AgentState(TypedDict):
    input_file: Optional[str]  # Contains file path (PDF/PNG)
    messages: Annotated[list[AnyMessage], add_messages]
```

### 4. Observability

Integrated with Langfuse for tracing:

```python
from langfuse.langchain import CallbackHandler
langfuse_handler = CallbackHandler()
```

## Usage

### Simple Calculation

```python
from image_tool_agent import react_graph

messages = [HumanMessage(content="Divide 6790 by 5")]
result = react_graph.invoke(
    input={"messages": messages, "input_file": None}
)
```

### Image Text Extraction

```python
messages = [HumanMessage(content="Extract text from the document")]
result = react_graph.invoke(
    input={"messages": messages, "input_file": "/path/to/image.png"}
)
```

## Environment Variables

```bash
ANTHROPIC_API_KEY=your_key
LANGFUSE_PUBLIC_KEY=your_key      # Optional
LANGFUSE_SECRET_KEY=your_key      # Optional
LANGFUSE_HOST=https://cloud.langfuse.com
```

## Dependencies

```bash
pip install langgraph langchain-anthropic langchain-core langfuse
```

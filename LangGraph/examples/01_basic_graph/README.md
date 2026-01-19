# Basic Graph Example

This example demonstrates the fundamentals of LangGraph's StateGraph pattern with conditional routing.

## Overview

A simple 3-node graph that shows:
- How to define state using `TypedDict`
- How to create nodes as functions
- How to add edges between nodes
- How to implement conditional routing

## Graph Structure

```
┌─────────┐
│  START  │
└────┬────┘
     │
     ▼
┌─────────┐
│ node_1  │  "Hi" → "Hi, I am"
└────┬────┘
     │
     ▼ (50/50 random)
┌────┴────┐
│         │
▼         ▼
┌───────┐ ┌───────┐
│node_2 │ │node_3 │
│"happy"│ │"sad!" │
└───┬───┘ └───┬───┘
    │         │
    └────┬────┘
         ▼
    ┌─────────┐
    │   END   │
    └─────────┘
```

## Key Concepts

### State Definition

```python
from typing_extensions import TypedDict

class State(TypedDict):
    graph_state: str
```

### Node Functions

Each node receives state and returns state updates:

```python
def node_1(state):
    return {"graph_state": state['graph_state'] + " I am"}
```

### Conditional Edges

Route to different nodes based on logic:

```python
def decide_mood(state) -> Literal["node_2", "node_3"]:
    if random.random() < 0.5:
        return "node_2"
    return "node_3"

builder.add_conditional_edges("node_1", decide_mood)
```

## Usage

```python
from graph import graph

result = graph.invoke({"graph_state": "Hi, this is Lance."})
print(result)
# Possible outputs:
# {'graph_state': 'Hi, this is Lance. I am happy!'}
# {'graph_state': 'Hi, this is Lance. I am sad!'}
```

## Dependencies

```bash
pip install langgraph typing-extensions ipython
```

## Visualization

The example includes code to visualize the graph using Mermaid:

```python
from IPython.display import Image, display
display(Image(graph.get_graph().draw_mermaid_png()))
```

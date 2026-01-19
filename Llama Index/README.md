---
title: LlamaIndex Examples
emoji: 🦙
colorFrom: purple
colorTo: blue
sdk: static
pinned: false
license: mit
tags:
  - llamaindex
  - rag
  - agents
  - huggingface
  - ai-agents
---

# LlamaIndex Examples

A collection of practical examples demonstrating LlamaIndex capabilities including RAG, multi-agent workflows, and agentic systems.

## Examples

| File | Description |
|------|-------------|
| `examples/01_intro.py` | Basic LlamaIndex setup with HuggingFace Inference API |
| `examples/02_rag.py` | RAG implementation with ChromaDB and document ingestion |
| `examples/03_agents.py` | Multi-agent workflow with calculator and query agents |
| `examples/04_agentic_workflows.py` | Advanced workflows with context state management |

## Tech Stack

- **LLM**: Qwen/Qwen3-Next-80B-A3B-Thinking (via HuggingFace Inference API)
- **Embeddings**: BAAI/bge-small-en-v1.5
- **Vector Store**: ChromaDB
- **Framework**: LlamaIndex

## Installation

```bash
git clone https://huggingface.co/spaces/YOUR_USERNAME/llama-index-examples
cd llama-index-examples
pip install -r requirements.txt
```

## Setup

1. Create a `.env` file based on `.env.example`:
```bash
cp .env.example .env
```

2. Add your HuggingFace token to `.env`:
```
HF_TOKEN=your_huggingface_token_here
```

## Usage

Run any example:
```bash
python examples/01_intro.py
python examples/02_rag.py
python examples/03_agents.py
python examples/04_agentic_workflows.py
```

## Key Concepts Covered

- **RAG (Retrieval-Augmented Generation)**: Document ingestion, vector embeddings, semantic search
- **ReAct Agents**: Reasoning and action cycles for problem-solving
- **Multi-Agent Orchestration**: Coordinating specialized agents
- **State Management**: Persistent context across workflow execution

## Project Structure

```
llama-index-examples/
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
├── LICENSE
├── examples/
│   ├── 01_intro.py
│   ├── 02_rag.py
│   ├── 03_agents.py
│   └── 04_agentic_workflows.py
└── docs/
    └── notes.md
```

## License

MIT

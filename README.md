# AI-Agentic

A collection of practical examples and projects exploring AI agent frameworks and agentic workflows.

## Projects

| Directory | Description | Framework |
|-----------|-------------|-----------|
| [Chapter 1](Chapter%201/) | Introduction to AI agents and basic tool usage | - |
| [LangGraph](LangGraph/) | Stateful multi-actor applications with conditional routing and tool integration | LangGraph |
| [Llama Index](Llama%20Index/) | RAG, multi-agent workflows, and agentic systems | LlamaIndex |
| [Small Agents](Small%20Agents/) | Tool calling, multi-agent orchestration, RAG, and MCP integration | smolagents |

## Key Concepts Covered

- **Agent Types**: ToolCallingAgent, CodeAgent, ReAct agents
- **RAG (Retrieval-Augmented Generation)**: Document ingestion, vector embeddings, semantic search
- **Multi-Agent Orchestration**: Hierarchical and coordinated agent systems
- **Tool Integration**: Custom tools, MCP protocol, function calling
- **State Management**: StateGraph patterns, conditional routing, context persistence
- **Observability**: Langfuse tracing and monitoring

## Tech Stack

- **Frameworks**: LangGraph, LlamaIndex, smolagents
- **LLM Providers**: Anthropic (Claude), OpenAI, HuggingFace Inference API, Together AI
- **Vector Stores**: ChromaDB
- **Embeddings**: BAAI/bge-small-en-v1.5
- **Search**: SearXNG, DuckDuckGo

## Getting Started

Each project has its own README with specific setup instructions. General requirements:

```bash
# Python 3.10+
python -m venv venv
source venv/bin/activate

# Install project-specific dependencies
cd <project-directory>
pip install -r requirements.txt
```

### Environment Variables

Projects may require API keys in a `.env` file:

```env
ANTHROPIC_API_KEY=your_key
OPENAI_API_KEY=your_key
HF_TOKEN=your_huggingface_token
```

## License

MIT

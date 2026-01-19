# smolagents-examples

Practical examples exploring smolagents framework features - from basic tool calling to multi-agent orchestration, RAG, and MCP integration.

## Overview

This repository contains hands-on examples demonstrating [smolagents](https://github.com/huggingface/smolagents) framework capabilities. Each example builds on previous concepts, providing a progressive learning path.

## Examples

| # | File | Description |
|---|------|-------------|
| 01 | [basic_tool_calling.py](examples/01_basic_tool_calling.py) | Simplest agent with `ToolCallingAgent` and `WebSearchTool` |
| 02 | [custom_tools_decorator.py](examples/02_custom_tools_decorator.py) | Create custom tools using `@tool` decorator |
| 03 | [code_agent_multi_tools.py](examples/03_code_agent_multi_tools.py) | `CodeAgent` with multiple built-in and custom tools |
| 04 | [single_agent_pandas.py](examples/04_single_agent_pandas.py) | Geographic calculations, pandas integration, planning intervals |
| 05 | [multi_agent_orchestration.py](examples/05_multi_agent_orchestration.py) | Hierarchical multi-agent architecture with vision validation |
| 06 | [agentic_rag.py](examples/06_agentic_rag.py) | RAG integration with LangChain and BM25 retrieval |
| 07 | [mcp_integration.py](examples/07_mcp_integration.py) | Model Context Protocol (MCP) server integration |

## Features Covered

### Agent Types
- **ToolCallingAgent**: Simple tool selection and calling
- **CodeAgent**: Writes and executes Python code with tools

### Tools
- Built-in: `WebSearchTool`, `DuckDuckGoSearchTool`, `VisitWebpageTool`, `FinalAnswerTool`
- Custom tools via `@tool` decorator
- Custom tools via `Tool` class extension

### Advanced Features
- **Managed Agents**: Hierarchical multi-agent orchestration
- **Final Answer Checks**: Validation functions (including vision-based)
- **Planning Intervals**: Control agent replanning frequency
- **MCP Integration**: Connect to external tool ecosystems
- **RAG Pipelines**: LangChain + BM25 retrieval integration

### Model Providers
- `InferenceClientModel`: Hugging Face, Together AI, etc.
- `OpenAIServerModel`: GPT-4o for multimodal tasks

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# For MCP support (Example 07)
pip install "smolagents[mcp]"
brew install uv  # macOS only
```

## Configuration

Set up your Hugging Face token for model access:

```bash
huggingface-cli login
```

For examples using OpenAI (Example 05):
```bash
export OPENAI_API_KEY="your-api-key"
```

For examples using Together AI:
```bash
export TOGETHER_API_KEY="your-api-key"
```

## Usage

Run any example directly:

```bash
python examples/01_basic_tool_calling.py
```

## Learning Path

1. **Start here**: Example 01-02 for basic concepts
2. **Intermediate**: Example 03-04 for custom tools and data processing
3. **Advanced**: Example 05-07 for multi-agent, RAG, and MCP

## Resources

- [smolagents Documentation](https://huggingface.co/docs/smolagents)
- [smolagents GitHub](https://github.com/huggingface/smolagents)
- [Hugging Face Hub](https://huggingface.co/)

## License

MIT

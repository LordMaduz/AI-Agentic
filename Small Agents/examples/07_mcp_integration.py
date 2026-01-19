"""
Example 07: Model Context Protocol (MCP) Integration

Demonstrates how to connect smolagents with MCP servers to access
external tool ecosystems. This example uses PubMed research tools.

Features:
- ToolCollection.from_mcp(): Load tools from MCP servers
- StdioServerParameters: Configure MCP server communication
- External tool ecosystems: Access tools beyond built-in options
- Context manager pattern: Proper MCP server lifecycle management

Requirements:
    pip install "smolagents[mcp]"
    brew install uv  # macOS
    # uvx is used by MCP to run commands
    # If uvx is missing, MCP server cannot start and tools won't load
"""

import os

from mcp import StdioServerParameters
from smolagents import CodeAgent, InferenceClientModel, ToolCollection

# Configure model
model = InferenceClientModel("Qwen/Qwen2.5-Coder-32B-Instruct")

# Configure MCP server parameters
# This example connects to PubMed research tools
server_parameters = StdioServerParameters(
    command="uvx",
    args=["--quiet", "pubmedmcp@0.1.3"],
    env={"UV_PYTHON": "3.12", **os.environ},
)

# Use context manager to properly handle MCP server lifecycle
with ToolCollection.from_mcp(
    server_parameters, trust_remote_code=True
) as tool_collection:
    # Create agent with MCP tools plus base tools
    agent = CodeAgent(
        tools=[*tool_collection.tools],
        model=model,
        add_base_tools=True,
    )

    # Run query using PubMed research tools
    agent.run("Please find a remedy for hangover.")

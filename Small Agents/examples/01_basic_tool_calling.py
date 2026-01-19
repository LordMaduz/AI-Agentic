"""
Example 01: Basic Tool Calling Agent

Demonstrates the simplest smolagents implementation using ToolCallingAgent
with a single WebSearchTool. This is the entry point for understanding
how agents interact with tools.

Features:
- ToolCallingAgent: Simple agent that selects and calls tools
- WebSearchTool: Built-in tool for web searches
- InferenceClientModel: Default model interface
"""

from smolagents import ToolCallingAgent, WebSearchTool, InferenceClientModel

agent = ToolCallingAgent(tools=[WebSearchTool()], model=InferenceClientModel())

agent.run("Search for the best music playlist recommendations for a weekend house party.")

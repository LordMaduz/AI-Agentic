"""
Advanced Agent Workflows with State Management
==============================================

Demonstrates async context-aware tools that track and update
workflow state across multiple operations.

Prerequisites:
    pip install -r requirements.txt

Setup:
    Create a .env file with your HuggingFace token:
    HF_TOKEN=your_token_here
"""

import os
import asyncio
from dotenv import load_dotenv

from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.core.workflow import Context
from llama_index.core.agent.workflow import AgentWorkflow, ReActAgent

# Load environment variables
load_dotenv()

# Retrieve HF_TOKEN from environment
hf_token = os.getenv("HF_TOKEN")

if not hf_token:
    raise ValueError("HF_TOKEN not found. Please set it in your .env file.")


# Define async context-aware tools
async def add(ctx: Context, a: int, b: int) -> int:
    """Add two numbers."""
    # Update function call count in state
    cur_state = await ctx.store.get("state")
    cur_state["num_fn_calls"] += 1
    await ctx.store.set("state", cur_state)
    return a + b


async def multiply(ctx: Context, a: int, b: int) -> int:
    """Multiply two numbers."""
    # Update function call count in state
    cur_state = await ctx.store.get("state")
    cur_state["num_fn_calls"] += 1
    await ctx.store.set("state", cur_state)
    return a * b


# Initialize LLM
llm = HuggingFaceInferenceAPI(
    model_name="Qwen/Qwen3-Next-80B-A3B-Thinking",
    temperature=0.7,
    max_tokens=100,
    token=hf_token,
    provider="auto",
)

# Create specialized agents
multiply_agent = ReActAgent(
    name="multiply_agent",
    description="Is able to multiply two integers",
    system_prompt="A helpful assistant that can use a tool to multiply numbers.",
    tools=[multiply],
    llm=llm,
)

addition_agent = ReActAgent(
    name="add_agent",
    description="Is able to add two integers",
    system_prompt="A helpful assistant that can use a tool to add numbers.",
    tools=[add],
    llm=llm,
)

# Create the workflow with initial state
workflow = AgentWorkflow(
    agents=[multiply_agent, addition_agent],
    root_agent="multiply_agent",
    initial_state={"num_fn_calls": 0},
    state_prompt="Current state: {state}. User message: {msg}",
)

# Create workflow context
ctx = Context(workflow)


async def main():
    # First operation: addition
    query = "Can you add 5 and 3?"
    response = await workflow.run(user_msg=query, ctx=ctx)
    print("Operation Response:")
    print(response)

    # Second operation: multiplication
    query = "Can you multiply 5 by 3?"
    response = await workflow.run(user_msg=query, ctx=ctx)
    print("\nOperation Response:")
    print(response)

    # Inspect the state - shows total function calls across operations
    state = await ctx.store.get("state")
    print("\nTotal function calls:")
    print(state["num_fn_calls"])


if __name__ == "__main__":
    asyncio.run(main())

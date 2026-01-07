
"""

---------------------------------
Installation
---------------------------------

pip install llama-index-utils-workflow
"""

import asyncio

from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.core.workflow import Context

from llama_index.core.agent.workflow import (
    AgentWorkflow,
    ReActAgent
)

# Define some tools
async def add(ctx: Context, a: int, b: int) -> int:
    """Add two numbers."""
    # update our count
    cur_state = await ctx.store.get("state")
    cur_state["num_fn_calls"] += 1
    await ctx.store.set("state", cur_state)

    return a + b

async def multiply(ctx: Context, a: int, b: int) -> int:
    """Multiply two numbers."""
    # update our count
    cur_state = await ctx.store.get("state")
    cur_state["num_fn_calls"] += 1
    await ctx.store.set("state", cur_state)

    return a * b

llm = HuggingFaceInferenceAPI(
    model_name="Qwen/Qwen3-Next-80B-A3B-Thinking",
    temperature=0.7,
    max_tokens=100,
    token=TOKEN,
    provider="auto",
)

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

# Create the workflow
workflow = AgentWorkflow(
    agents=[multiply_agent, addition_agent],
    root_agent="multiply_agent",
    initial_state={"num_fn_calls": 0},
    state_prompt="Current state: {state}. User message: {msg}",
)

# run the workflow with context
ctx = Context(workflow)


async def main():

    query = "Can you add 5 and 3?"
    response = await workflow.run(user_msg=query, ctx=ctx)
    print("Opration Response : ")
    print(response)

    query = "Can you multiply 5 by 3?"
    response = await workflow.run(user_msg=query, ctx=ctx)
    print("Opration Response : ")
    print(response)

    # pull out and inspect the state
    state = await ctx.store.get("state")
    print("State : ")
    print(state["num_fn_calls"])

asyncio.run(main())





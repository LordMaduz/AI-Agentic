
"""

---------------------------------
Installation
---------------------------------

pip install llama-index-llms-huggingface-api llama-index-embeddings-huggingface
"""

import chromadb
import asyncio
import os

import asyncio

from llama_index.vector_stores.chroma import ChromaVectorStore

from llama_index.core import SimpleDirectoryReader
from llama_index.core import Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core import VectorStoreIndex
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.core.evaluation import FaithfulnessEvaluator

from llama_index.core.tools import QueryEngineTool


from llama_index.core.agent.workflow import (
    AgentWorkflow,
    FunctionAgent,
    ReActAgent
)

from dotenv import load_dotenv

# Define some tools
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


def subtract(a: int, b: int) -> int:
    """Subtract two numbers."""
    return a - b

# Load the .env file
load_dotenv()

# Retrieve HF_TOKEN from the environment variables
hf_token = os.getenv("HF_TOKEN")

reader = SimpleDirectoryReader(input_dir="/Users/ruchira/Development/AI Agentic/Llama Index/Docs")
documents = reader.load_data()

db = chromadb.PersistentClient(path="./ruchira_chroma_db")
chroma_collection = db.get_or_create_collection("ruchira")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

# create the pipeline with transformations
pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(chunk_overlap=0),
        HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5"),
    ],
    vector_store=vector_store,
)

pipeline.run(documents=documents)

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)

llm = HuggingFaceInferenceAPI(
    model_name="Qwen/Qwen3-Next-80B-A3B-Thinking", # Think Efficiently general purpose LLM optimized for reasoning and concise “thinking”
    temperature=0.7,
    max_tokens=100,
    token=hf_token,
    provider="auto",
)

query_engine = index.as_query_engine(
    llm=llm,
    response_mode="tree_summarize",
)

query_engine_tool = QueryEngineTool.from_defaults(
    query_engine=query_engine,
    name="info_lookup",
    description="Looks up information about Kate Edgar Foundation",
    return_direct=False,
)


# Create agent configs
# NOTE: we can use FunctionAgent or ReActAgent here.
# FunctionAgent works for LLMs with a function calling API.
# ReActAgent works for any LLM.
calculator_agent = ReActAgent(
    name="calculator",
    description="Performs basic arithmetic operations",
    system_prompt="You are a calculator assistant. Use your tools for any math operation.",
    tools=[add, subtract],
    llm=llm,
)

query_agent = ReActAgent(
    name="info_lookup",
    description="Looks up information about Kate Edgar Foundation",
    system_prompt="Use your tool to query a RAG system to answer information about Kate Edgar Foundation",
    tools=[query_engine_tool],
    llm=llm
)

# Create and run the workflow
agent = AgentWorkflow(
    agents=[calculator_agent, query_agent], root_agent="calculator"
)

async def main():
    #response = await agent.run(user_msg="Can you add 5 and 3?")
    #print("Answer Is:")
    #print(response)

    query = "What is the definition of Agents work?"
    response = await agent.run(user_msg=query)
    print("Answer Is:")
    print(response)

asyncio.run(main())





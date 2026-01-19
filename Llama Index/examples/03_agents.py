"""
Multi-Agent Workflow Example
============================

Demonstrates multi-agent orchestration combining calculator
operations with RAG-based information lookup.

Prerequisites:
    pip install -r requirements.txt

Setup:
    Create a .env file with your HuggingFace token:
    HF_TOKEN=your_token_here
"""

import os
import asyncio
from pathlib import Path
from dotenv import load_dotenv

import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.core.tools import QueryEngineTool
from llama_index.core.agent.workflow import AgentWorkflow, ReActAgent

# Load environment variables
load_dotenv()

# Retrieve HF_TOKEN from environment
hf_token = os.getenv("HF_TOKEN")

if not hf_token:
    raise ValueError("HF_TOKEN not found. Please set it in your .env file.")


# Define calculator tools
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


def subtract(a: int, b: int) -> int:
    """Subtract two numbers."""
    return a - b


# Get the docs directory path (relative to this file)
DOCS_DIR = Path(__file__).parent.parent / "docs"

# Load documents
reader = SimpleDirectoryReader(input_dir=str(DOCS_DIR))
documents = reader.load_data()

# Setup ChromaDB vector store
db = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = db.get_or_create_collection("llama_index_examples")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

# Create ingestion pipeline
pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(chunk_overlap=0),
        HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5"),
    ],
    vector_store=vector_store,
)

pipeline.run(documents=documents)

# Create index
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)

# Initialize LLM
llm = HuggingFaceInferenceAPI(
    model_name="Qwen/Qwen3-Next-80B-A3B-Thinking",
    temperature=0.7,
    max_tokens=100,
    token=hf_token,
    provider="auto",
)

# Create query engine and tool
query_engine = index.as_query_engine(
    llm=llm,
    response_mode="tree_summarize",
)

query_engine_tool = QueryEngineTool.from_defaults(
    query_engine=query_engine,
    name="info_lookup",
    description="Looks up information from the documentation",
    return_direct=False,
)

# Create agents
# NOTE: ReActAgent works for any LLM (FunctionAgent requires function calling API)
calculator_agent = ReActAgent(
    name="calculator",
    description="Performs basic arithmetic operations",
    system_prompt="You are a calculator assistant. Use your tools for any math operation.",
    tools=[add, subtract],
    llm=llm,
)

query_agent = ReActAgent(
    name="info_lookup",
    description="Looks up information from documentation",
    system_prompt="Use your tool to query a RAG system to answer information questions.",
    tools=[query_engine_tool],
    llm=llm,
)

# Create and run the workflow
agent = AgentWorkflow(
    agents=[calculator_agent, query_agent],
    root_agent="calculator",
)


async def main():
    # Example: Math operation
    # response = await agent.run(user_msg="Can you add 5 and 3?")
    # print("Answer:", response)

    # Example: Information lookup
    query = "What is the definition of Agents work?"
    response = await agent.run(user_msg=query)
    print("Answer:", response)


if __name__ == "__main__":
    asyncio.run(main())

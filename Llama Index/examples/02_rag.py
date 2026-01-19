"""
RAG (Retrieval-Augmented Generation) Example
=============================================

Demonstrates document ingestion, vector storage with ChromaDB,
and querying with faithfulness evaluation.

Prerequisites:
    pip install -r requirements.txt

Setup:
    Create a .env file with your HuggingFace token:
    HF_TOKEN=your_token_here
"""

import os
from pathlib import Path
from dotenv import load_dotenv

import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.core.evaluation import FaithfulnessEvaluator

# Load environment variables
load_dotenv()

# Retrieve HF_TOKEN from environment
hf_token = os.getenv("HF_TOKEN")

if not hf_token:
    raise ValueError("HF_TOKEN not found. Please set it in your .env file.")

# Get the docs directory path (relative to this file)
DOCS_DIR = Path(__file__).parent.parent / "docs"

# Load documents
reader = SimpleDirectoryReader(input_dir=str(DOCS_DIR))
documents = reader.load_data()

# Setup ChromaDB vector store
db = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = db.get_or_create_collection("llama_index_examples")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

# Create ingestion pipeline with transformations
pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(chunk_overlap=0),
        HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5"),
    ],
    vector_store=vector_store,
)

# Ingest documents
pipeline.run(documents=documents)
print("Documents ingested into vector store.")

# Create index from vector store
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

# Create query engine
query_engine = index.as_query_engine(
    llm=llm,
    response_mode="tree_summarize",
)

# Setup evaluator for faithfulness
evaluator = FaithfulnessEvaluator(llm=llm)

# Query the index
query = "What is the definition of Agents work?"
response = query_engine.query(query)

print("Query Result:")
print(response)

# Evaluate response faithfulness
eval_result = evaluator.evaluate_response(response=response)

print("\nEval Result:")
print(eval_result)

print("\nEval Result Passing:")
print(eval_result.passing)

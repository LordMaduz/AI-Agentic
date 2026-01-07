
"""

---------------------------------
Installation
---------------------------------

pip install llama-index-llms-huggingface-api llama-index-embeddings-huggingface
"""

import chromadb
import asyncio
import os
from llama_index.vector_stores.chroma import ChromaVectorStore

from llama_index.core import SimpleDirectoryReader
from llama_index.core import Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core import VectorStoreIndex
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.core.evaluation import FaithfulnessEvaluator

from dotenv import load_dotenv

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
print("Documents ingested into vector store.")

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)

llm = HuggingFaceInferenceAPI(
    model_name="Qwen/Qwen3-Next-80B-A3B-Thinking", # Think Efficiently general purpose LLM optimized for reasoning and concise “thinking”
    #model_name="Qwen/Qwen2.5-Coder-32B-Instruct", # Gives longer, more explanatory answers, often verbose
    temperature=0.7,
    max_tokens=100,
    token=hf_token,
    provider="auto",
)

query_engine = index.as_query_engine(
    llm=llm,
    response_mode="tree_summarize",
)

evaluator = FaithfulnessEvaluator(llm=llm)

query = "What is the definition of Agents work?"
response = query_engine.query(query)

print("Query Result:")
print(response)

eval_result = evaluator.evaluate_response(response=response)

print("Eval Result:")
print(eval_result)

print("Eval Result Passing:")
print(eval_result.passing)


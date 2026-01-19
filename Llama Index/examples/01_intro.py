"""
LlamaIndex Introduction
=======================

Basic example demonstrating LlamaIndex setup with HuggingFace Inference API.

Prerequisites:
    pip install -r requirements.txt

Setup:
    Create a .env file with your HuggingFace token:
    HF_TOKEN=your_token_here
"""

import os
from dotenv import load_dotenv
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI

# Load environment variables
load_dotenv()

# Retrieve HF_TOKEN from environment
hf_token = os.getenv("HF_TOKEN")

if not hf_token:
    raise ValueError("HF_TOKEN not found. Please set it in your .env file.")

# Initialize the LLM
llm = HuggingFaceInferenceAPI(
    model_name="Qwen/Qwen3-Next-80B-A3B-Thinking",
    temperature=0.7,
    max_tokens=100,
    token=hf_token,
    provider="auto"
)

# Simple completion example
response = llm.complete("Hello, how are you?")
print(response)

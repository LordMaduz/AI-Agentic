"""
Example 06: Agentic RAG (Retrieval-Augmented Generation)

Demonstrates how to integrate LangChain's retrieval components with smolagents
to create a RAG-powered agent. Uses BM25 for keyword-based retrieval.

Features:
- LangChain integration: Document handling and text splitting
- BM25Retriever: Keyword-based document retrieval
- Custom retriever Tool class: Wraps retriever as agent tool
- RecursiveCharacterTextSplitter: Document chunking with overlap
- Knowledge base simulation

Requirements:
    pip install langchain-community langchain-text-splitters rank-bm25
"""

from langchain_community.docstore.document import Document
from langchain_community.retrievers import BM25Retriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from smolagents import CodeAgent, InferenceClientModel, Tool


class TravelGuideRetrieverTool(Tool):
    """
    Custom retriever tool that uses BM25 for semantic search over a knowledge base.
    This pattern can be adapted for any document retrieval use case.
    """

    name = "travel_guide_retriever"
    description = "Uses semantic search to retrieve relevant travel tips and destination information from a curated knowledge base."
    inputs = {
        "query": {
            "type": "string",
            "description": "The query to search for. This should be related to travel destinations, tips, or recommendations.",
        }
    }
    output_type = "string"

    def __init__(self, docs, **kwargs):
        super().__init__(**kwargs)
        self.retriever = BM25Retriever.from_documents(
            docs, k=5  # Retrieve the top 5 documents
        )

    def forward(self, query: str) -> str:
        assert isinstance(query, str), "Your search query must be a string"

        docs = self.retriever.invoke(query)
        return "\nRetrieved information:\n" + "".join(
            [
                f"\n\n===== Result {str(i)} =====\n" + doc.page_content
                for i, doc in enumerate(docs)
            ]
        )


# Simulated knowledge base about travel
travel_knowledge = [
    {
        "text": "Tokyo is best visited during spring (March-May) for cherry blossoms or autumn (September-November) for colorful foliage. Must-see spots include Senso-ji Temple, Shibuya Crossing, and the Tsukiji Outer Market.",
        "source": "Destination Guide - Tokyo",
    },
    {
        "text": "For budget travel in Europe, consider traveling during shoulder season (April-May or September-October). Use trains instead of flights, stay in hostels or Airbnbs, and eat at local markets.",
        "source": "Budget Travel Tips",
    },
    {
        "text": "Essential packing list for international travel: passport, travel adapter, comfortable walking shoes, layers for varying weather, basic first-aid kit, and copies of important documents.",
        "source": "Packing Guide",
    },
    {
        "text": "Paris highlights include the Eiffel Tower, Louvre Museum, and Montmartre. Book popular attractions in advance. The best croissants are found in local neighborhood bakeries, not tourist areas.",
        "source": "Destination Guide - Paris",
    },
    {
        "text": "For solo travelers: stay in social hostels, join free walking tours, use apps like Meetup to find local events, and always share your itinerary with someone back home for safety.",
        "source": "Solo Travel Tips",
    },
    {
        "text": "Bali offers a mix of beaches, temples, and rice terraces. Visit Ubud for culture, Seminyak for nightlife, and Uluwatu for stunning cliff views. Best time to visit is April-October (dry season).",
        "source": "Destination Guide - Bali",
    },
]

# Convert to LangChain Documents
source_docs = [
    Document(page_content=doc["text"], metadata={"source": doc["source"]})
    for doc in travel_knowledge
]

# Split documents into smaller chunks for more efficient search
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    add_start_index=True,
    strip_whitespace=True,
    separators=["\n\n", "\n", ".", " ", ""],
)
docs_processed = text_splitter.split_documents(source_docs)

# Create the retriever tool
travel_retriever = TravelGuideRetrieverTool(docs_processed)

# Initialize the agent with the retriever tool
agent = CodeAgent(tools=[travel_retriever], model=InferenceClientModel())

# Run a RAG query
response = agent.run(
    "I'm planning a trip to Japan. What's the best time to visit and what should I pack?"
)

print(response)

# example.py (place this in your root directory)

# Import components directly
from search_system.types import SearchResult
from search_system.spacy_embedder import SpacyEmbedder
from search_system.semantic_retriever import SemanticRetriever
from search_system.search_system import SearchSystem

# Create components manually
embedder = SpacyEmbedder()
embedder.initialize()

# Sample documents
documents = [
    "Patient has type 2 diabetes",
    "Blood test shows elevated glucose levels",
    "Patient reports fatigue and increased thirst"
]

# Build index
embedder.build_index(documents)

# Create retriever
retriever = SemanticRetriever(embedder)

# Create system
system = SearchSystem()
system.add_retriever(retriever)

# Test search
results = system.search("What are the symptoms?")

# Print results
for result in results:
    print(f"Score: {result.score:.4f}, Content: {result.content}")
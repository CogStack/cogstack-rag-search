from .search_system import SearchSystem
from .spacy_embedder import SpacyEmbedder
from .semantic_retriever import SemanticRetriever
from .colbert_reranker import ColBERTReranker
from .date_filter import DateFilter
from .gpt_judge import GPTJudge

def create_default_search_system(openai_api_key: str) -> SearchSystem:
    """Creates a default search system with standard components"""
    embedder = SpacyEmbedder()
    retriever = SemanticRetriever(embedder)
    reranker = ColBERTReranker()
    date_filter = DateFilter()
    llm_judge = GPTJudge(openai_api_key)

    search_system = SearchSystem()
    search_system.add_retriever(retriever)
    search_system.add_reranker(reranker)
    search_system.add_filter(date_filter)
    search_system.add_llm_judge(llm_judge)

    return search_system
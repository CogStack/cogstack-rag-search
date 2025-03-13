from typing import List, Dict, Optional
from .base import BaseRetriever, BaseReranker, BaseFilter, BaseLLMJudge
from .types import SearchResult

class SearchSystem:
    """Main search system that orchestrates all components"""
    def __init__(self):
        self.retriever: Optional[BaseRetriever] = None
        self.reranker: Optional[BaseReranker] = None
        self.filters: List[BaseFilter] = []
        self.llm_judge: Optional[BaseLLMJudge] = None

    def add_retriever(self, retriever: BaseRetriever) -> None:
        self.retriever = retriever
        self.retriever.initialize()

    def add_reranker(self, reranker: BaseReranker) -> None:
        self.reranker = reranker
        self.reranker.initialize()

    def add_filter(self, filter_module: BaseFilter) -> None:
        filter_module.initialize()
        self.filters.append(filter_module)

    def add_llm_judge(self, llm_judge: BaseLLMJudge) -> None:
        self.llm_judge = llm_judge
        self.llm_judge.initialize()

    def search(self, query: str, filter_criteria: Dict = None) -> List[SearchResult]:
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
            
        if not self.retriever:
            raise ValueError("No retriever configured")

        try:
            # Initial retrieval
            results = self.retriever.retrieve(query)

            # Apply filters
            if filter_criteria and self.filters:
                for filter_module in self.filters:
                    results = filter_module.filter(results, filter_criteria)

            # Apply reranking if configured
            if self.reranker and results:
                results = self.reranker.rerank(query, results)

            # Apply LLM evaluation if configured
            if self.llm_judge and results:
                results = self.llm_judge.evaluate(query, results)

            return results
        except Exception as e:
            raise RuntimeError(f"Search failed: {str(e)}") from e

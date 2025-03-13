from typing import List
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from .base import BaseRetriever, BaseEmbedder
from .types import SearchResult

class SemanticRetriever(BaseRetriever):
    """Semantic search implementation using embeddings"""
    def __init__(self, embedder: BaseEmbedder):
        self.embedder = embedder

    def initialize(self) -> None:
        self.embedder.initialize()

    def validate(self) -> bool:
        return self.embedder.validate()

    def retrieve(self, query: str, top_k: int = 10) -> List[SearchResult]:
        query_embedding = self.embedder.embed(query)
        scores = cosine_similarity([query_embedding], self.embedder.index)[0]
        top_indices = np.argsort(-scores)[:top_k]
        
        results = []
        for idx in top_indices:
            results.append(SearchResult(
                id=str(idx),
                content=self.embedder.documents[idx],
                score=float(scores[idx]),
                metadata={}
            ))
        return results

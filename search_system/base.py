from abc import ABC, abstractmethod
from typing import List, Dict
import numpy as np
from .types import SearchResult

class BaseModule(ABC):
    """Abstract base class for all search system modules"""
    @abstractmethod
    def initialize(self) -> None:
        """Initialize any resources needed by the module"""
        pass

    @abstractmethod
    def validate(self) -> bool:
        """Validate if the module is properly configured"""
        pass

class BaseRetriever(BaseModule):
    """Base class for retrieval modules"""
    @abstractmethod
    def retrieve(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """Retrieve relevant documents for a query"""
        pass

class BaseReranker(BaseModule):
    """Base class for reranking modules"""
    @abstractmethod
    def rerank(self, query: str, results: List[SearchResult]) -> List[SearchResult]:
        """Rerank the retrieved results"""
        pass

class BaseFilter(BaseModule):
    """Base class for filtering modules"""
    @abstractmethod
    def filter(self, results: List[SearchResult], criteria: Dict) -> List[SearchResult]:
        """Filter results based on given criteria"""
        pass

class BaseEmbedder(BaseModule):
    """Base class for embedding modules"""
    @abstractmethod
    def embed(self, text: str) -> np.ndarray:
        """Generate embeddings for given text"""
        pass

    @abstractmethod
    def build_index(self, documents: List[str]) -> None:
        """Build search index from documents"""
        pass

class BaseLLMJudge(BaseModule):
    """Base class for LLM-based result evaluation"""
    @abstractmethod
    def evaluate(self, query: str, results: List[SearchResult]) -> List[SearchResult]:
        """Evaluate and possibly modify results using LLM"""
        pass

from typing import List
import torch
from transformers import AutoTokenizer, AutoModel
from .base import BaseReranker
from .types import SearchResult

class ColBERTReranker(BaseReranker):
    """ColBERT-based reranking implementation"""
    def __init__(self, model_name: str = "bert-base-uncased", max_length: int = 512):
        self.model_name = model_name
        self.max_length = max_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None

    def initialize(self) -> None:
        """Initialize the ColBERT model and tokenizer"""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

    def validate(self) -> bool:
        """Check if model and tokenizer are properly loaded"""
        return self.model is not None and self.tokenizer is not None

    def _get_colbert_embeddings(self, text: str) -> torch.Tensor:
        """Get ColBERT embeddings for a piece of text"""
        inputs = self.tokenizer(
            text,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use the last hidden state
            embeddings = outputs.last_hidden_state
            # Create attention mask for valid tokens
            mask = inputs['attention_mask'].unsqueeze(-1)
            # Apply mask and get final embeddings
            masked_embeddings = embeddings * mask
        
        return masked_embeddings.squeeze(0)

    def _compute_similarity(self, query_emb: torch.Tensor, doc_emb: torch.Tensor) -> float:
        """Compute MaxSim between query and document embeddings"""
        # Compute similarity matrix
        similarity = torch.matmul(query_emb, doc_emb.transpose(0, 1))
        # Get maximum similarity for each query token
        max_sim = similarity.max(dim=1)[0]
        # Average over query tokens
        score = max_sim.mean().item()
        return score

    def rerank(self, query: str, results: List[SearchResult]) -> List[SearchResult]:
        """Rerank results using ColBERT similarity"""
        if not results:
            return results

        # Get query embeddings
        query_embeddings = self._get_colbert_embeddings(query)

        # Compute scores for all documents
        reranked_results = []
        for result in results:
            doc_embeddings = self._get_colbert_embeddings(result.content)
            score = self._compute_similarity(query_embeddings, doc_embeddings)
            
            # Create new SearchResult with updated score
            reranked_results.append(SearchResult(
                id=result.id,
                content=result.content,
                score=score,
                metadata=result.metadata
            ))

        # Sort by new scores in descending order
        reranked_results.sort(key=lambda x: x.score, reverse=True)
        return reranked_results 
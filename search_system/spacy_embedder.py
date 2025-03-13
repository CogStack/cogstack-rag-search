import numpy as np
from typing import List
import spacy
from .base import BaseEmbedder

class SpacyEmbedder(BaseEmbedder):
    """Spacy-based embedding implementation"""
    def __init__(self, model_name: str = "en_core_web_md"):
        self.model_name = model_name
        self.nlp = None
        self.index = None
        self.documents = None

    def initialize(self) -> None:
        self.nlp = spacy.load(self.model_name)

    def validate(self) -> bool:
        return self.nlp is not None

    def embed(self, text: str) -> np.ndarray:
        return self.nlp(text).vector

    def build_index(self, documents: List[str]) -> None:
        self.documents = documents
        self.index = np.vstack([self.embed(doc) for doc in documents])

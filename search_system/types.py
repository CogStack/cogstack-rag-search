from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class SearchResult:
    """Data class to store search results in a standardized format"""
    id: str
    content: str
    score: float
    metadata: Dict[Any, Any]

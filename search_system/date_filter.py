from typing import List, Dict
from .base import BaseFilter
from .types import SearchResult

class DateFilter(BaseFilter):
    """Filter results by date range"""
    def initialize(self) -> None:
        pass

    def validate(self) -> bool:
        return True

    def filter(self, results: List[SearchResult], criteria: Dict) -> List[SearchResult]:
        if 'start_date' not in criteria or 'end_date' not in criteria:
            return results
            
        filtered_results = []
        for result in results:
            if 'date' in result.metadata:
                date = result.metadata['date']
                if criteria['start_date'] <= date <= criteria['end_date']:
                    filtered_results.append(result)
        return filtered_results

from dataclasses import dataclass
from typing import List, Dict, Any
from .search_result import SearchResult

@dataclass
class AgentMessage:
    query_id: str
    agent_type: str
    results: List[SearchResult]
    confidence: float
    metadata: Dict[str, Any]
    processing_time: float = 0.0
    explanation: str = ""
    success: bool = True
    error_message: str = ""
    
    def to_dict(self) -> Dict:
        return {
            'query_id': self.query_id,
            'agent_type': self.agent_type,
            'results': [r.to_dict() for r in self.results],
            'confidence': self.confidence,
            'metadata': self.metadata,
            'processing_time': self.processing_time,
            'explanation': self.explanation,
            'success': self.success,
            'error_message': self.error_message
        }
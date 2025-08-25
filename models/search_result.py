from dataclasses import dataclass
from typing import Dict, Optional, Any

@dataclass
class SearchResult:
    video_id: str
    keyframe_id: Optional[str] = None
    score: float = 0.0
    source_agent: str = ""
    result_type: str = ""  # video|keyframe|object
    metadata: Optional[Dict[str, Any]] = None
    explanation: str = ""
    
    def to_dict(self) -> Dict:
        return {
            'video_id': self.video_id,
            'keyframe_id': self.keyframe_id,
            'score': self.score,
            'source_agent': self.source_agent,
            'result_type': self.result_type,
            'metadata': self.metadata or {},
            'explanation': self.explanation
        }
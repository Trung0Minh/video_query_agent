from dataclasses import dataclass
from typing import List, Dict, Optional

@dataclass
class QueryIntent:
    intent_type: str  # text|visual|hybrid|temporal
    agents_needed: List[str]
    text_params: Optional[Dict] = None
    visual_params: Optional[Dict] = None
    temporal_params: Optional[Dict] = None
    fusion_strategy: str = "weighted"
    confidence: float = 0.0
    reasoning: str = ""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import time
import json
from dataclasses import dataclass
from tools.gemini_client import GeminiClient
from models.search_result import SearchResult
from config.prompts import get_agent_prompt

@dataclass
class AgentMessage:
    """Communication between agents"""
    query_id: str
    agent_type: str
    results: List[SearchResult]
    confidence: float
    metadata: Dict[str, Any]
    success: bool
    processing_time: float = 0.0
    error_message: str = ""
    explanation: str = ""

class BaseAgent(ABC):
    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.llm = GeminiClient()
        self.cache = {}  # Simple in-memory cache
        
    @abstractmethod
    async def process(self, query: str, context: Dict = None) -> AgentMessage:
        """Process query and return agent message"""
        pass
    
    @abstractmethod
    def get_available_functions(self) -> List[Dict]:
        """Get available functions for this agent"""
        pass
    
    async def llm_call(self, user_message: str, response_format: str = "json") -> str:
        """Make LLM call with agent-specific prompt"""
        return await self.llm.generate(
            system_prompt=get_agent_prompt(self.__class__.__name__),
            user_message=user_message,
            response_format=response_format
        )
    
    async def llm_function_call(self, user_message: str) -> Dict:
        """Make LLM call with function calling"""
        return await self.llm.generate_with_functions(
            system_prompt=get_agent_prompt(self.__class__.__name__),
            user_message=user_message,
            functions=self.get_available_functions()
        )
    
    def cache_result(self, query: str, result: AgentMessage):
        """Cache result for performance"""
        cache_key = f"{self.agent_name}:{hash(query)}"
        self.cache[cache_key] = result
    
    def get_cached_result(self, query: str) -> Optional[AgentMessage]:
        """Get cached result if exists"""
        cache_key = f"{self.agent_name}:{hash(query)}"
        return self.cache.get(cache_key)
    
    async def process_with_cache(self, query: str, context: Dict = None) -> AgentMessage:
        """Process with caching support"""
        # Check cache first
        cached = self.get_cached_result(query)
        if cached:
            print(f"[{self.agent_name}] Cache hit for query")
            return cached
        
        # Process and cache
        start_time = time.time()
        result = await self.process(query, context)
        result.processing_time = time.time() - start_time
        
        self.cache_result(query, result)
        return result
    
    def log(self, message: str):
        """Agent logging"""
        print(f"[{self.agent_name}] {message}")
        
    def validate_result(self, result: AgentMessage) -> bool:
        """Validate agent result"""
        return (
            result.query_id and
            result.agent_type == self.agent_name and
            isinstance(result.results, list) and
            0 <= result.confidence <= 1.0
        )

    async def _analyze_strategy(self, query: str, context: Dict, context_key: str, fallback_strategy: Dict) -> Dict:
        """Generic method to analyze strategy"""
        try:
            if context and context.get(context_key):
                return fallback_strategy
            
            response = await self.llm_call(query, "json")
            return json.loads(response)
        except Exception as e:
            self.log(f"Strategy analysis failed: {e}")
            return fallback_strategy

    def _create_error_message(self, query_id: str, e: Exception) -> AgentMessage:
        """Create a standardized error message."""
        self.log(f"Error: {e}")
        return AgentMessage(
            query_id=query_id,
            agent_type=self.agent_name,
            results=[],
            confidence=0.0,
            metadata={},
            explanation=f"Lỗi: {str(e)}",
            success=False,
            error_message=str(e)
        )

    def _create_search_results(self, results: List[Dict], score_field: str, explanation_field: str, result_type_field: str = 'result_type') -> List[SearchResult]:
        """Convert a list of dictionaries into a list of SearchResult objects."""
        search_results = []
        for result in results:
            search_results.append(SearchResult(
                video_id=result['video_id'],
                keyframe_id=result.get('keyframe_id'),
                score=result.get(score_field, 0.0),
                source_agent=self.agent_name,
                result_type=result.get(result_type_field, 'video'),
                metadata=result,
                explanation=result.get(explanation_field, '')
            ))
        return search_results
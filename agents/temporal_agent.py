import json
from typing import List, Dict, Any
from .base_agent import BaseAgent, AgentMessage
from models.search_result import SearchResult
from tools.sqlite_tool import SQLiteTool

class TemporalAgent(BaseAgent):
    def __init__(self):
        super().__init__("TemporalAgent")
        self.sqlite_tool = SQLiteTool()
    
    def get_available_functions(self) -> List[Dict]:
        return [
            {
                "name": "search_time_range",
                "description": "Search keyframes in time range",
                "parameters": {
                    "video_id": "Video ID",
                    "start_time": "Start time in seconds",
                    "end_time": "End time in seconds"
                }
            }
        ]
    
    async def process(self, query: str, context: Dict = None) -> AgentMessage:
        """Process temporal query"""
        query_id = context.get('query_id', 'unknown') if context else 'unknown'
        
        try:
            # Step 1: Analyze temporal strategy
            strategy = await self._analyze_temporal_strategy(query, context)
            self.log(f"Temporal strategy: {strategy['temporal_type']}")
            
            # Step 2: Execute temporal search
            results = await self._execute_temporal_search(strategy)
            
            # Step 3: Convert to SearchResult objects
            search_results = self._create_search_results(results, 'temporal_score', 'explanation')
            
            confidence = self._calculate_temporal_confidence(search_results, strategy)
            
            return AgentMessage(
                query_id=query_id,
                agent_type=self.agent_name,
                results=search_results,
                confidence=confidence,
                metadata={'strategy': strategy},
                explanation=f"Tìm thấy {len(search_results)} kết quả temporal",
                success=True
            )
            
        except Exception as e:
            return self._create_error_message(query_id, e)
    
    async def _analyze_temporal_strategy(self, query: str, context: Dict = None) -> Dict:
        """Analyze temporal query strategy"""
        fallback_strategy = {
            "temporal_type": "DURATION",
            "duration_filter": {"sort_by_duration": False},
            "explanation": "Fallback temporal strategy"
        }
        return await self._analyze_strategy(query, context, 'video_id', fallback_strategy)
    
    async def _execute_temporal_search(self, strategy: Dict) -> List[Dict]:
        """Execute temporal search based on strategy"""
        temporal_type = strategy['temporal_type']
        
        if temporal_type == 'TIME_RANGE':
            return await self._search_time_range(strategy.get('time_params', {}))
        elif temporal_type == 'SEQUENCE':
            return await self._search_sequence(strategy.get('sequence_params', {}))
        elif temporal_type == 'DURATION':
            return await self._search_by_duration(strategy.get('duration_filter', {}))
        elif temporal_type == 'PUBLISH_DATE':
            return await self._search_by_publish_date(strategy.get('date_filter', {}))
        else:
            return []
    
    async def _search_time_range(self, time_params: Dict) -> List[Dict]:
        """Search keyframes in specific time range"""
        video_id = time_params.get('video_id')
        start_time = time_params.get('start_time', 0)
        end_time = time_params.get('end_time', 999999)
        
        if not video_id:
            return []
        
        results = self.sqlite_tool.get_keyframes_in_timerange(video_id, start_time, end_time)
        
        for result in results:
            result['temporal_score'] = 1.0
            result['explanation'] = f"Keyframe tại {result['pts_time']:.1f}s"
        
        return results
    
    async def _search_sequence(self, sequence_params: Dict) -> List[Dict]:
        """Search keyframes in sequence relative to reference"""
        # Placeholder for sequence search - would need more complex logic
        return []
    
    async def _search_by_duration(self, duration_filter: Dict) -> List[Dict]:
        """Search videos by duration"""
        query = "SELECT * FROM videos WHERE 1=1"
        params = []
        
        if duration_filter.get('min_duration'):
            query += " AND length >= ?"
            params.append(duration_filter['min_duration'])
        
        if duration_filter.get('max_duration'):
            query += " AND length <= ?"
            params.append(duration_filter['max_duration'])
        
        if duration_filter.get('sort_by_duration'):
            query += " ORDER BY length DESC"
        
        results = self.sqlite_tool.execute_query(query, tuple(params))
        
        for result in results:
            result['temporal_score'] = 1.0
            result['explanation'] = f"Video dài {result['length']}s"
        
        return results
    
    async def _search_by_publish_date(self, date_filter: Dict) -> List[Dict]:
        """Search videos by publish date"""
        # Placeholder for date-based search
        return []
    
    def _calculate_temporal_confidence(self, results: List[SearchResult], strategy: Dict) -> float:
        """Calculate confidence for temporal search"""
        if not results:
            return 0.0
        
        # High confidence for specific temporal queries
        temporal_type = strategy['temporal_type']
        
        confidence_map = {
            'TIME_RANGE': 0.95,  # Very specific
            'SEQUENCE': 0.8,
            'DURATION': 0.7,
            'PUBLISH_DATE': 0.75
        }
        
        return confidence_map.get(temporal_type, 0.7)
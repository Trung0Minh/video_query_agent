from typing import List, Dict, Any
from .base_agent import BaseAgent, AgentMessage
from models.search_result import SearchResult
from collections import defaultdict

class ResultFusionAgent(BaseAgent):
    def __init__(self):
        super().__init__("ResultFusionAgent")
    
    def get_available_functions(self) -> List[Dict]:
        return []  # Fusion agent doesn't need external functions
    
    async def process(self, query: str, context: Dict = None) -> AgentMessage:
        """Process result fusion"""
        query_id = context.get('query_id', 'unknown') if context else 'unknown'
        
        try:
            agent_results = context.get('agent_results', [])
            fusion_strategy = context.get('fusion_strategy', 'weighted')
            
            if not agent_results:
                return AgentMessage(
                    query_id=query_id,
                    agent_type=self.agent_name,
                    results=[],
                    confidence=0.0,
                    metadata={},
                    explanation="No agent results to fuse",
                    success=True
                )
            
            # Step 1: Analyze fusion strategy
            strategy = await self._analyze_fusion_strategy(agent_results, fusion_strategy)
            
            # Step 2: Execute fusion
            fused_results = await self._execute_fusion(agent_results, strategy)
            
            # Step 3: Deduplicate results
            if strategy.get('deduplication', {}).get('enabled', True):
                fused_results = self._deduplicate_results(fused_results, strategy)
            
            # Step 4: Final ranking
            final_results = self._final_ranking(fused_results, strategy)
            
            confidence = self._calculate_fusion_confidence(agent_results, final_results)
            
            return AgentMessage(
                query_id=query_id,
                agent_type=self.agent_name,
                results=final_results[:50],  # Limit to top 50
                confidence=confidence,
                metadata={
                    'fusion_strategy': strategy,
                    'input_agents': [r.agent_type for r in agent_results],
                    'total_input_results': sum(len(r.results) for r in agent_results),
                    'fused_count': len(final_results)
                },
                explanation=f"Fused {len(final_results)} results from {len(agent_results)} agents",
                success=True
            )
            
        except Exception as e:
            return self._create_error_message(query_id, e)
    
    async def _analyze_fusion_strategy(self, agent_results: List[AgentMessage], 
                                     default_strategy: str) -> Dict:
        """Analyze and determine fusion strategy"""
        # Simple strategy selection based on agent types and confidences
        agent_types = [r.agent_type for r in agent_results]
        avg_confidence = sum(r.confidence for r in agent_results) / len(agent_results)
        
        if len(agent_results) == 1:
            strategy_type = "UNION"
        elif avg_confidence > 0.8:
            strategy_type = "WEIGHTED"
        else:
            strategy_type = "RANKED"
        
        return {
            "fusion_strategy": strategy_type,
            "ranking_criteria": {
                "visual_similarity": 0.4,
                "text_relevance": 0.3,
                "temporal_accuracy": 0.2,
                "object_confidence": 0.1
            },
            "deduplication": {
                "enabled": True,
                "merge_similar": True,
                "similarity_threshold": 0.9
            },
            "explanation": f"Selected {strategy_type} for {len(agent_results)} agents"
        }
    
    async def _execute_fusion(self, agent_results: List[AgentMessage], 
                            strategy: Dict) -> List[SearchResult]:
        """Execute fusion based on strategy"""
        fusion_type = strategy['fusion_strategy']
        
        if fusion_type == 'INTERSECTION':
            return self._intersection_fusion(agent_results)
        elif fusion_type == 'UNION':
            return self._union_fusion(agent_results)
        elif fusion_type == 'WEIGHTED':
            return self._weighted_fusion(agent_results, strategy)
        elif fusion_type == 'RANKED':
            return self._ranked_fusion(agent_results, strategy)
        else:
            return self._union_fusion(agent_results)  # Default
    
    def _intersection_fusion(self, agent_results: List[AgentMessage]) -> List[SearchResult]:
        """Find results that appear in multiple agents"""
        if len(agent_results) < 2:
            return self._union_fusion(agent_results)
        
        # Group results by video_id + keyframe_id
        result_groups = defaultdict(list)
        
        for agent_result in agent_results:
            for result in agent_result.results:
                key = f"{result.video_id}#{result.keyframe_id or 'video'}"
                result_groups[key].append((result, agent_result.agent_type, agent_result.confidence))
        
        # Only keep results that appear in at least 2 agents
        intersection_results = []
        for key, group in result_groups.items():
            if len(group) >= 2:
                # Take the best result from the group
                best_result = max(group, key=lambda x: x[0].score * x[2])
                result = best_result[0]
                # Boost score for intersection
                result.score = min(result.score * 1.2, 1.0)
                result.explanation += " (multi-agent match)"
                intersection_results.append(result)
        
        return intersection_results
    
    def _union_fusion(self, agent_results: List[AgentMessage]) -> List[SearchResult]:
        """Combine all results from all agents"""
        all_results = []
        for agent_result in agent_results:
            for result in agent_result.results:
                # Add agent info to explanation
                result.explanation += f" [{agent_result.agent_type}]"
                all_results.append(result)
        return all_results
    
    def _weighted_fusion(self, agent_results: List[AgentMessage], 
                        strategy: Dict) -> List[SearchResult]:
        """Weighted fusion based on agent confidence and type"""
        agent_weights = {
            'TextSearchAgent': 0.8,
            'VisualSearchAgent': 0.9,
            'TemporalAgent': 0.95
        }
        
        weighted_results = []
        for agent_result in agent_results:
            agent_weight = agent_weights.get(agent_result.agent_type, 0.7)
            confidence_weight = agent_result.confidence
            
            for result in agent_result.results:
                # Calculate weighted score
                weighted_score = result.score * agent_weight * confidence_weight
                result.score = min(weighted_score, 1.0)
                result.explanation += f" [weighted by {agent_result.agent_type}]"
                weighted_results.append(result)
        
        return weighted_results
    
    def _ranked_fusion(self, agent_results: List[AgentMessage], 
                      strategy: Dict) -> List[SearchResult]:
        """Advanced ranking fusion with multiple criteria"""
        ranking_criteria = strategy.get('ranking_criteria', {})
        
        all_results = self._union_fusion(agent_results)
        
        # Re-calculate scores based on multiple criteria
        for result in all_results:
            new_score = 0.0
            
            # Visual similarity component
            if 'VisualSearchAgent' in result.explanation:
                new_score += result.score * ranking_criteria.get('visual_similarity', 0.4)
            
            # Text relevance component  
            if 'TextSearchAgent' in result.explanation:
                new_score += result.score * ranking_criteria.get('text_relevance', 0.3)
            
            # Temporal accuracy component
            if 'TemporalAgent' in result.explanation:
                new_score += result.score * ranking_criteria.get('temporal_accuracy', 0.2)
            
            # Object confidence (from metadata)
            if result.metadata and result.metadata.get('object_confidence'):
                obj_conf = result.metadata['object_confidence']
                new_score += obj_conf * ranking_criteria.get('object_confidence', 0.1)
            
            result.score = min(new_score, 1.0)
        
        return all_results
    
    def _deduplicate_results(self, results: List[SearchResult], strategy: Dict) -> List[SearchResult]:
        """Remove duplicate results"""
        dedup_config = strategy.get('deduplication', {})
        similarity_threshold = dedup_config.get('similarity_threshold', 0.9)
        
        # Simple deduplication by video_id + keyframe_id
        seen_keys = set()
        deduplicated = []
        
        # Sort by score first to keep best results
        sorted_results = sorted(results, key=lambda x: x.score, reverse=True)
        
        for result in sorted_results:
            key = f"{result.video_id}#{result.keyframe_id or 'video'}"
            
            if key not in seen_keys:
                seen_keys.add(key)
                deduplicated.append(result)
        
        return deduplicated
    
    def _final_ranking(self, results: List[SearchResult], strategy: Dict) -> List[SearchResult]:
        """Final ranking of fused results"""
        return sorted(results, key=lambda x: x.score, reverse=True)
    
    def _calculate_fusion_confidence(self, agent_results: List[AgentMessage], 
                                   final_results: List[SearchResult]) -> float:
        """Calculate confidence for fusion result"""
        if not agent_results or not final_results:
            return 0.0
        
        # Base confidence from input agents
        agent_confidences = [r.confidence for r in agent_results if r.success]
        if not agent_confidences:
            return 0.0
        
        avg_agent_confidence = sum(agent_confidences) / len(agent_confidences)
        
        # Boost if multiple agents agree
        if len(agent_results) > 1:
            avg_agent_confidence += 0.1
        
        # Adjust based on result quality
        if final_results:
            avg_result_score = sum(r.score for r in final_results) / len(final_results)
            score_factor = min(avg_result_score, 0.2)
            avg_agent_confidence += score_factor
        
        return min(avg_agent_confidence, 1.0)
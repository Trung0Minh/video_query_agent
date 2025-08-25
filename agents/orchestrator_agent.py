import json
import asyncio
import uuid
from typing import List, Dict, Any
from .base_agent import BaseAgent, AgentMessage
from .result_fusion_agent import ResultFusionAgent
from .temporal_agent import TemporalAgent
from .text_search_agent import TextSearchAgent
from .visual_search_agent import VisualSearchAgent
from models import QueryIntent, SearchResult

class OrchestratorAgent(BaseAgent):
    def __init__(self):
        super().__init__("OrchestratorAgent")
        
        # Initialize specialized agents
        self.text_agent = TextSearchAgent()
        self.visual_agent = VisualSearchAgent()
        self.temporal_agent = TemporalAgent()
        self.fusion_agent = ResultFusionAgent()
        
        self.agent_map = {
            'TextSearchAgent': self.text_agent,
            'VisualSearchAgent': self.visual_agent,
            'TemporalAgent': self.temporal_agent,
            'ResultFusionAgent': self.fusion_agent
        }
    
    def get_available_functions(self) -> List[Dict]:
        return []  # Orchestrator =doesn't use function calling
    
    async def process(self, query: str, context: Dict = None) -> AgentMessage:
        """Process user query and orchestrate other agents"""
        query_id = str(uuid.uuid4())
        self.log(f"Processing query: {query}")
        
        try:
            # Step 1: Analyze query intent
            intent = await self._analyze_intent(query)
            self.log(f"Intent analysis: {intent.intent_type}, agents: {intent.agents_needed}")
            
            # Step 2: Execute agents in parallel
            agent_tasks = []
            for agent_name in intent.agents_needed:
                if agent_name == 'ResultFusionAgent':
                    continue  # Will be called after other agents
                
                agent = self.agent_map.get(agent_name)
                if agent:
                    task = self._execute_agent(agent, query, intent, query_id)
                    agent_tasks.append(task)
            
            # Execute agents concurrently
            if agent_tasks:
                agent_results = await asyncio.gather(*agent_tasks, return_exceptions=True)
            else:
                agent_results = []
            
            # Filter successful results
            successful_results = []
            for result in agent_results:
                if isinstance(result, AgentMessage) and result.success:
                    successful_results.append(result)
                elif isinstance(result, Exception):
                    self.log(f"Agent error: {result}")
            
            # Step 3: Fuse results if multiple agents were used
            if len(successful_results) > 1:
                fusion_result = await self._fuse_results(
                    successful_results, intent, query_id
                )
                final_results = fusion_result.results
                explanation = fusion_result.explanation
            elif len(successful_results) == 1:
                final_results = successful_results[0].results
                explanation = successful_results[0].explanation
            else:
                final_results = []
                explanation = "Không tìm thấy kết quả phù hợp"
            
            # Step 4: Return orchestrated result
            return AgentMessage(
                query_id=query_id,
                agent_type=self.agent_name,
                results=final_results,
                confidence=self._calculate_confidence(successful_results),
                metadata={
                    'intent': intent,
                    'agents_used': [r.agent_type for r in successful_results],
                    'total_results': len(final_results)
                },
                explanation=explanation,
                success=True
            )
            
        except Exception as e:
            return self._create_error_message(query_id, e)
    
    async def _analyze_intent(self, query: str) -> QueryIntent:
        """Analyze user query to determine intent"""
        try:
            response = await self.llm_call(query, "json")
            intent_data = json.loads(response)
            
            return QueryIntent(
                intent_type=intent_data.get('intent_type', 'text'),
                agents_needed=intent_data.get('agents_needed', ['TextSearchAgent']),
                text_params=intent_data.get('text_params'),
                visual_params=intent_data.get('visual_params'),
                temporal_params=intent_data.get('temporal_params'),
                fusion_strategy=intent_data.get('fusion_strategy', 'weighted'),
                reasoning=intent_data.get('reasoning', '')
            )
            
        except Exception as e:
            self.log(f"Intent analysis failed: {e}")
            # Fallback to text search
            return QueryIntent(
                intent_type='text',
                agents_needed=['TextSearchAgent'],
                text_params={'search_terms': [query], 'fields': ['title', 'description']},
                reasoning='Fallback to text search due to analysis failure'
            )
    
    async def _execute_agent(self, agent: BaseAgent, query: str, 
                           intent: QueryIntent, query_id: str) -> AgentMessage:
        """Execute specific agent with proper parameters"""
        try:
            # Prepare agent-specific context
            context = {
                'query_id': query_id,
                'intent': intent,
                'original_query': query
            }
            
            # Add agent-specific parameters
            if agent.agent_name == 'TextSearchAgent' and intent.text_params:
                context.update(intent.text_params)
            elif agent.agent_name == 'VisualSearchAgent' and intent.visual_params:
                context.update(intent.visual_params)
            elif agent.agent_name == 'TemporalAgent' and intent.temporal_params:
                context.update(intent.temporal_params)
            
            result = await agent.process_with_cache(query, context)
            return result
            
        except Exception as e:
            self.log(f"Agent execution failed for {agent.agent_name}: {e}")
            return AgentMessage(
                query_id=query_id,
                agent_type=agent.agent_name,
                results=[],
                confidence=0.0,
                metadata={},
                explanation=f"Agent failed: {str(e)}",
                success=False,
                error_message=str(e)
            )
    
    async def _fuse_results(self, agent_results: List[AgentMessage], 
                          intent: QueryIntent, query_id: str) -> AgentMessage:
        """Fuse results from multiple agents"""
        try:
            context = {
                'query_id': query_id,
                'agent_results': agent_results,
                'fusion_strategy': intent.fusion_strategy,
                'intent': intent
            }
            
            return await self.fusion_agent.process_with_cache(
                "Fuse agent results", context
            )
            
        except Exception as e:
            self.log(f"Result fusion failed: {e}")
            # Fallback: combine all results
            all_results = []
            for result in agent_results:
                all_results.extend(result.results)
            
            return AgentMessage(
                query_id=query_id,
                agent_type='ResultFusionAgent',
                results=all_results[:50],  # Limit results
                confidence=0.5,
                metadata={'fusion_strategy': 'fallback'},
                explanation="Fallback fusion due to error",
                success=True
            )
    
    def _calculate_confidence(self, agent_results: List[AgentMessage]) -> float:
        """Calculate overall confidence from agent results"""
        if not agent_results:
            return 0.0
        
        confidences = [r.confidence for r in agent_results if r.success]
        if not confidences:
            return 0.0
        
        # Weighted average based on result count
        total_weight = 0
        weighted_sum = 0
        
        for result in agent_results:
            if result.success:
                weight = len(result.results) + 1  # +1 to avoid zero weight
                weighted_sum += result.confidence * weight
                total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
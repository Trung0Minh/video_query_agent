import json
from typing import List, Dict, Any
from .base_agent import BaseAgent, AgentMessage
from models.search_result import SearchResult
from tools.qdrant_tool import QdrantTool
from tools.sqlite_tool import SQLiteTool
from config.settings import settings
from utils.result_ranker import ResultRanker

class VisualSearchAgent(BaseAgent):
    def __init__(self):
        super().__init__("VisualSearchAgent")
        self.qdrant_tool = QdrantTool(settings.QDRANT_VIDEO_COLLECTION_NAME)
        self.sqlite_tool = SQLiteTool()
    
    def get_available_functions(self) -> List[Dict]:
        return [
            {
                "name": "text_to_visual_search",
                "description": "Convert text to visual embedding and search",
                "parameters": {
                    "description": "Visual description text",
                    "threshold": "Similarity threshold"
                }
            },
            {
                "name": "similarity_search",
                "description": "Search visually similar keyframes",
                "parameters": {
                    "query_vector": "Query vector",
                    "limit": "Max results"
                }
            }
        ]
    
    async def process(self, query: str, context: Dict = None) -> AgentMessage:
        """Process visual search query"""
        query_id = context.get('query_id', 'unknown') if context else 'unknown'
        
        try:
            # Step 1: Analyze visual search strategy
            strategy = await self._analyze_visual_strategy(query, context)
            self.log(f"Visual strategy: {strategy['search_strategy']}")
            
            # Step 2: Generate visual embedding
            query_embedding = await self._generate_embedding(strategy['visual_query'])
            
            if not query_embedding:
                raise Exception("Failed to generate visual embedding")
            
            # Step 3: Execute visual search
            visual_results = await self._execute_visual_search(query_embedding, strategy)
            
            # Step 4: Enrich with metadata
            enriched_results = await self._enrich_with_metadata(visual_results)
            
            # Step 5: Apply post-processing
            final_results = self._post_process_results(enriched_results, strategy)
            
            # Step 6: Convert to SearchResult objects
            search_results = self._create_search_results(final_results[:50], 'similarity_score', 'explanation', 'keyframe')
            search_results = ResultRanker.diversity_ranking(search_results)
            
            confidence = self._calculate_visual_confidence(search_results, strategy)
            
            return AgentMessage(
                query_id=query_id,
                agent_type=self.agent_name,
                results=search_results,
                confidence=confidence,
                metadata={
                    'strategy': strategy,
                    'embedding_generated': len(query_embedding) > 0,
                    'total_found': len(visual_results),
                    'returned': len(search_results)
                },
                explanation=f"Tìm thấy {len(search_results)} keyframes tương tự visual",
                success=True
            )
            
        except Exception as e:
            return self._create_error_message(query_id, e)
    
    async def _analyze_visual_strategy(self, query: str, context: Dict = None) -> Dict:
        """Analyze query to determine visual search strategy"""
        fallback_strategy = {
            "search_strategy": "TEXT_TO_VISUAL",
            "visual_query": {
                "description": query,
                "keywords": [],
                "scene_type": "mixed"
            },
            "search_params": {
                "similarity_threshold": 0.6,
                "max_results": 100,
                "diversity_filter": True
            },
            "explanation": "Fallback to simple text-to-visual search"
        }
        return await self._analyze_strategy(query, context, 'search_description', fallback_strategy)
    
    async def _generate_embedding(self, visual_query: Dict) -> List[float]:
        """Generate CLIP embedding from visual description"""
        try:
            description = visual_query.get('description', '')
            keywords = visual_query.get('keywords', [])
            
            # Combine description with keywords
            full_text = description
            if keywords:
                full_text += " " + " ".join(keywords)
            
            print(full_text)
            from sentence_transformers import SentenceTransformer            
            text_model = SentenceTransformer('sentence-transformers/clip-ViT-B-32-multilingual-v1')
            embedding = text_model.encode(full_text)
            
            if embedding is None or len(embedding) == 0:
                # Fallback: create a dummy embedding (in real implementation, use proper CLIP)
                self.log("Warning: Using dummy embedding - implement proper CLIP embedding")
                return [0.1] * 512  # Dummy 512-dim vector
            
            return embedding.tolist()
            
        except Exception as e:
            self.log(f"Embedding generation failed: {e}")
            return []
    
    async def _execute_visual_search(self, query_embedding: List[float], 
                                   strategy: Dict) -> List[Dict]:
        """Execute visual similarity search"""
        try:
            search_params = strategy.get('search_params', {})
            metadata_filters = strategy.get('metadata_filters', {})
            
            similarity_threshold = search_params.get('similarity_threshold', 0.7)
            max_results = search_params.get('max_results', 100)
            video_filter = metadata_filters.get('video_ids')
            
            results = self.qdrant_tool.search_similar_keyframes(
                query_vector=query_embedding,
                limit=max_results,
                similarity_threshold=0.05,
                video_filter=video_filter
            )
            
            return results
            
        except Exception as e:
            self.log(f"Visual search execution failed: {e}")
            return []
    
    async def _enrich_with_metadata(self, visual_results: List[Dict]) -> List[Dict]:
        """Enrich visual results with metadata from SQLite"""
        enriched_results = []
        
        for result in visual_results:
            try:
                # Get video metadata
                video_metadata = self.sqlite_tool.get_video_metadata(result['video_id'])
                
                # Get keyframe objects
                keyframe_objects = self.sqlite_tool.get_keyframe_objects(
                    result['video_id'], 
                    result['keyframe_id']
                )
                
                enriched_result = result.copy()
                enriched_result['video_metadata'] = video_metadata
                enriched_result['detected_objects'] = keyframe_objects
                enriched_result['object_count'] = len(keyframe_objects)
                
                # Calculate object confidence score
                if keyframe_objects:
                    avg_object_confidence = sum(obj['confidence'] for obj in keyframe_objects) / len(keyframe_objects)
                    enriched_result['object_confidence'] = avg_object_confidence
                else:
                    enriched_result['object_confidence'] = 0.0
                
                enriched_results.append(enriched_result)
                
            except Exception as e:
                self.log(f"Failed to enrich result for {result['video_id']}/{result['keyframe_id']}: {e}")
                enriched_results.append(result)  # Add without enrichment
        
        return enriched_results
    
    def _post_process_results(self, results: List[Dict], strategy: Dict) -> List[Dict]:
        """Post-process results based on strategy"""
        search_params = strategy.get('search_params', {})
        
        # Apply diversity filter if requested
        if search_params.get('diversity_filter', False):
            results = self._apply_diversity_filter(results)
        
        # Re-rank based on combined scores
        for result in results:
            visual_score = result.get('similarity_score', 0.0)
            object_score = result.get('object_confidence', 0.0)
            
            # Combined score: 70% visual similarity, 30% object confidence
            combined_score = 0.7 * visual_score + 0.3 * object_score
            result['combined_score'] = combined_score
        
        # Sort by combined score
        return sorted(results, key=lambda x: x.get('combined_score', 0), reverse=True)
    
    def _apply_diversity_filter(self, results: List[Dict]) -> List[Dict]:
        """Apply diversity filter to avoid too many results from same video"""
        video_counts = {}
        filtered_results = []
        max_per_video = 5  # Maximum keyframes per video
        
        for result in results:
            video_id = result['video_id']
            current_count = video_counts.get(video_id, 0)
            
            if current_count < max_per_video:
                filtered_results.append(result)
                video_counts[video_id] = current_count + 1
        
        return filtered_results
    
    def _calculate_visual_confidence(self, results: List[SearchResult], strategy: Dict) -> float:
        """Calculate confidence score for visual search"""
        if not results:
            return 0.0
        
        # Base confidence from strategy
        strategy_confidence = {
            'TEXT_TO_VISUAL': 0.7,
            'SIMILARITY_SEARCH': 0.8,
            'FILTERED_VISUAL': 0.85,
            'OBJECT_GUIDED': 0.9
        }
        
        base_conf = strategy_confidence.get(strategy['search_strategy'], 0.7)
        
        # Adjust based on result quality
        if results:
            avg_similarity = sum(r.score for r in results) / len(results)
            
            # High similarity scores boost confidence
            if avg_similarity > 0.8:
                base_conf += 0.15
            elif avg_similarity > 0.6:
                base_conf += 0.05
            else:
                base_conf -= 0.1
        
        # Adjust based on result count
        if len(results) > 20:
            base_conf += 0.1
        elif len(results) < 5:
            base_conf -= 0.15
        
        return max(0.0, min(base_conf, 1.0))
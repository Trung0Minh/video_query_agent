import json
from typing import List, Dict, Any
from .base_agent import BaseAgent, AgentMessage
from models.search_result import SearchResult
from tools.sqlite_tool import SQLiteTool
from tools.qdrant_tool import QdrantTool
from config.settings import settings
from utils.result_ranker import ResultRanker

class TextSearchAgent(BaseAgent):
    def __init__(self):
        super().__init__("TextSearchAgent")
        self.qdrant_tool = QdrantTool(settings.QDRANT_KEYWORD_COLLECTION_NAME)
        self.sqlite_tool = SQLiteTool()
    
    def get_available_functions(self) -> List[Dict]:
        return [
            {
                "name": "search_metadata",
                "description": "Search videos by metadata text",
                "parameters": {
                    "text": "Search text",
                    "fields": "Fields to search in"
                }
            },
            {
                "name": "search_objects",
                "description": "Search keyframes by object detection",
                "parameters": {
                    "object_names": "List of object names",
                    "confidence_threshold": "Minimum confidence"
                }
            }
        ]
    
    async def process(self, query: str, context: Dict = None) -> AgentMessage:
        """Process text search query"""
        query_id = context.get('query_id', 'unknown') if context else 'unknown'
        
        search_term_update = set()
        for search_term in context['search_terms']:
            from sentence_transformers import SentenceTransformer
            embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
            search_term_vector = embedding_model.encode(search_term)
            search_term_update.update(self.qdrant_tool.search_similar_keyword(search_term_vector))
        
        if search_term_update != "":
            context['search_terms'] = search_term_update
        
        try:
            # Step 1: Analyze search strategy
            strategy = await self._analyze_search_strategy(query, context)
            self.log(f"Search strategy: {strategy['search_strategy']}")
            
            # Step 2: Execute search based on strategy
            results = await self._execute_search(strategy)
            
            # Step 3: Rank and filter results
            ranked_results = self._rank_results(results, strategy)
            
            # Step 4: Convert to SearchResult objects
            search_results = self._create_search_results(ranked_results[:50], 'score', 'explanation')
            search_results = ResultRanker.diversity_ranking(search_results)
            
            confidence = self._calculate_confidence(search_results, strategy)
            
            return AgentMessage(
                query_id=query_id,
                agent_type=self.agent_name,
                results=search_results,
                confidence=confidence,
                metadata={
                    'strategy': strategy,
                    'total_found': len(results),
                    'returned': len(search_results)
                },
                explanation=f"Tìm thấy {len(search_results)} kết quả bằng {strategy['search_strategy']}",
                success=True
            )
            
        except Exception as e:
            return self._create_error_message(query_id, e)
    
    async def _analyze_search_strategy(self, query: str, context: Dict = None) -> Dict:
        """Analyze query to determine search strategy"""
        fallback_strategy = {
            "search_strategy": "METADATA_SEARCH",
            "metadata_search": {
                "terms": [query.lower()],
                "fields": ["title", "description", "keywords"],
                "exact_match": False
            },
            "explanation": "Fallback to simple metadata search"
        }
        return await self._analyze_strategy(query, context, 'search_terms', fallback_strategy)
    
    async def _execute_search(self, strategy: Dict) -> List[Dict]:
        """Execute search based on strategy"""
        all_results = []
        
        strategy_type = strategy['search_strategy']
        
        if strategy_type in ['METADATA_SEARCH', 'COMBINED_SEARCH']:
            metadata_results = await self._search_metadata(strategy.get('metadata_search', {}))
            all_results.extend(metadata_results)
        
        if strategy_type in ['OBJECT_SEARCH', 'COMBINED_SEARCH']:
            object_results = await self._search_objects(strategy.get('object_search', {}))
            all_results.extend(object_results)
        
        if strategy_type == 'AUTHOR_SEARCH':
            author_results = await self._search_by_author(strategy.get('filters', {}))
            all_results.extend(author_results)
        
        return all_results
    
    async def _search_metadata(self, metadata_config: Dict) -> List[Dict]:
        """Search in video metadata"""
        results = []
        terms = metadata_config.get('terms', [])
        fields = metadata_config.get('fields', ['title', 'description', 'keywords'])
        
        for term in terms:
            db_results = self.sqlite_tool.search_videos_by_text(term, fields)
            for result in db_results:
                result['result_type'] = 'video'
                result['search_term'] = term
                result['explanation'] = f"Tìm thấy '{term}' trong {', '.join(fields)}"
                results.append(result)
        
        return results
    
    async def _search_objects(self, object_config: Dict) -> List[Dict]:
        """Search by object detection"""
        object_names = object_config.get('object_names', [])
        confidence_threshold = object_config.get('confidence_threshold', 0.5)
        
        if not object_names:
            return []
        
        db_results = self.sqlite_tool.search_objects(object_names, confidence_threshold)
        
        results = []
        for result in db_results:
            result['result_type'] = 'keyframe'
            result['explanation'] = f"Chứa {result['object_name']} (confidence: {result['avg_confidence']:.2f})"
            results.append(result)
        
        return results
    
    async def _search_by_author(self, filters: Dict) -> List[Dict]:
        """Search by author"""
        author = filters.get('author')
        if not author:
            return []
        
        results = self.sqlite_tool.search_videos_by_text(author, ['author'])
        for result in results:
            result['result_type'] = 'video'
            result['explanation'] = f"Video của tác giả {result['author']}"
        
        return results
    
    def _rank_results(self, results: List[Dict], strategy: Dict) -> List[Dict]:
        """Rank results based on strategy weights"""
        weights = strategy.get('ranking_weights', {
            'title_match': 1.0,
            'keyword_match': 0.8,
            'description_match': 0.6,
            'object_confidence': 0.9
        })
        
        for result in results:
            score = 0.0
            
            # Title match scoring
            if result.get('search_term') and result.get('title'):
                if result['search_term'].lower() in result['title'].lower():
                    score += weights.get('title_match', 1.0)
            
            # Object confidence scoring
            if result.get('avg_confidence'):
                score += result['avg_confidence'] * weights.get('object_confidence', 0.9)
            
            # Keyword match scoring
            if result.get('search_term') and result.get('keywords'):
                if result['search_term'].lower() in result['keywords'].lower():
                    score += weights.get('keyword_match', 0.8)
            
            result['score'] = score
        
        # Sort by score descending
        return sorted(results, key=lambda x: x.get('score', 0), reverse=True)
    
    def _calculate_confidence(self, results: List[SearchResult], strategy: Dict) -> float:
        """Calculate confidence score"""
        if not results:
            return 0.0
        
        # Base confidence from strategy
        strategy_confidence = {
            'METADATA_SEARCH': 0.8,
            'OBJECT_SEARCH': 0.7,
            'COMBINED_SEARCH': 0.9,
            'AUTHOR_SEARCH': 0.95
        }
        
        base_conf = strategy_confidence.get(strategy['search_strategy'], 0.7)
        
        # Adjust based on result quality
        if len(results) > 10:
            base_conf += 0.1
        elif len(results) < 3:
            base_conf -= 0.2
        
        # Adjust based on scores
        avg_score = sum(r.score for r in results) / len(results) if results else 0
        score_factor = min(avg_score / 2.0, 0.2)  # Cap at 0.2 boost
        
        return min(base_conf + score_factor, 1.0)
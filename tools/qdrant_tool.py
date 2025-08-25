from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, Range
from typing import List, Dict, Any, Optional
from config.settings import settings

class QdrantTool:
    def __init__(self, collection_name):
        self.client = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)
        self.collection_name = collection_name
    
    def search_similar_keyframes(self, query_vector: List[float], 
                                limit: int = 100,
                                similarity_threshold: float = 0.7,
                                video_filter: Optional[List[str]] = None) -> List[Dict]:
        """Search for visually similar keyframes"""
        try:
            # Build filter if needed
            search_filter = None
            if video_filter:
                search_filter = Filter(
                    must=[
                        FieldCondition(
                            key="video_id",
                            match={"any": video_filter}
                        )
                    ]
                )
            
            print(f"similarity threshold: {similarity_threshold}")
            # Perform search
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=limit,
                score_threshold=similarity_threshold,
                query_filter=search_filter
            )
            
            # Format results
            results = []
            for result in search_results:
                results.append({
                    'video_id': result.payload['video_id'],
                    'keyframe_id': result.payload['keyframe_id'],
                    'similarity_score': float(result.score),
                    'qdrant_id': result.id
                })
            
            return results
            
        except Exception as e:
            print(f"Qdrant search error: {e}")
            return []
    
    def search_similar_keyword(self, query_vector: List[float],
                               limit: int = 10,
                               similarity_threshold: float = 0.7):
        try:
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=limit,
                score_threshold=similarity_threshold,
                with_vectors=True
            )
            
            results = []
            for result in search_results:
                results.append(result.payload.get("keyword", ""))
            return results
        
        except Exception as e:
            print(f"Qdrant search error: {e}")
            return []
    
    def search_by_video_ids(self, video_ids: List[str], limit: int = 50) -> List[Dict]:
        """Get all keyframes from specific videos"""
        try:
            search_filter = Filter(
                must=[
                    FieldCondition(
                        key="video_id",
                        match={"any": video_ids}
                    )
                ]
            )
            
            # Use scroll for getting all results
            scroll_results = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=search_filter,
                limit=limit
            )
            
            results = []
            for point in scroll_results[0]:  # scroll_results is (points, next_page_offset)
                results.append({
                    'video_id': point.payload['video_id'],
                    'keyframe_id': point.payload['keyframe_id'],
                    'qdrant_id': point.id
                })
            
            return results
            
        except Exception as e:
            print(f"Qdrant video filter error: {e}")
            return []
    
    def get_collection_info(self) -> Dict:
        """Get collection statistics"""
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                'points_count': info.points_count,
                'vectors_count': info.vectors_count,
                'status': info.status
            }
        except Exception as e:
            print(f"Collection info error: {e}")
            return {}
    
    def health_check(self) -> bool:
        """Check if Qdrant is healthy"""
        try:
            collections = self.client.get_collections()
            return self.collection_name in [c.name for c in collections.collections]
        except Exception:
            return False
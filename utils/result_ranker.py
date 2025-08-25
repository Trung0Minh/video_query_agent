from typing import List, Dict
from models.search_result import SearchResult

class ResultRanker:
    """Advanced result ranking utilities"""
    @staticmethod
    def normalize_scores(results: List[SearchResult]) -> List[SearchResult]:
        """Normalize scores to 0-1 range"""
        if not results:
            return results
        
        scores = [r.score for r in results]
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score:
            return results
        
        for result in results:
            result.score = (result.score - min_score) / (max_score - min_score)
        
        return results
    
    @staticmethod
    def diversity_ranking(results: List[SearchResult], 
                         diversity_weight: float = 0.2) -> List[SearchResult]:
        """Re-rank to promote diversity"""
        if len(results) <= 1:
            return results
        
        # Group by video_id
        video_groups = {}
        for result in results:
            video_id = result.video_id
            if video_id not in video_groups:
                video_groups[video_id] = []
            video_groups[video_id].append(result)
        
        # Apply diversity penalty
        diverse_results = []
        for video_id, group in video_groups.items():
            # Sort group by score
            group.sort(key=lambda x: x.score, reverse=True)
            
            # Apply diminishing returns for same video
            for i, result in enumerate(group):
                diversity_penalty = (1 - diversity_weight) ** i
                result.score *= diversity_penalty
                diverse_results.append(result)
        
        return sorted(diverse_results, key=lambda x: x.score, reverse=True)
    
    @staticmethod
    def temporal_clustering(results: List[SearchResult],
                          time_window: float = 30.0) -> List[SearchResult]:
        """Group temporally close results"""
        # Implementation for temporal clustering
        # Group keyframes that are close in time
        pass
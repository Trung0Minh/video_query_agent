import sqlite3
import json
from typing import List, Dict, Any, Optional
from config.settings import settings

class SQLiteTool:
    def __init__(self):
        self.db_path = settings.METADATA_KEYFRAME_OBJECT_DB_PATH
    
    def execute_query(self, query: str, params: tuple = ()) -> List[Dict]:
        """Execute SQL query and return results as dict list"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row  # Enable dict-like access
                cursor = conn.cursor()
                cursor.execute(query, params)
                
                results = []
                for row in cursor.fetchall():
                    results.append(dict(row))
                
                return results
                
        except sqlite3.Error as e:
            print(f"SQLite Error: {e}")
            return []
    
    def search_videos_by_text(self, text: str, fields: List[str] = None) -> List[Dict]:
        """Search videos by text in specified fields"""
        if fields is None:
            fields = ['title', 'description', 'keywords', 'author']
        
        conditions = []
        params = []
        
        for field in fields:
            conditions.append(f"{field} LIKE ?")
            params.append(f"%{text.lower()}%")
        
        query = f"""
        SELECT DISTINCT video_id, title, description, author, length, publish_date, keywords
        FROM videos 
        WHERE {' OR '.join(conditions)}
        ORDER BY 
            CASE 
                WHEN title LIKE ? THEN 1
                WHEN keywords LIKE ? THEN 2
                WHEN description LIKE ? THEN 3
                ELSE 4
            END
        """
        
        # Add ranking parameters
        ranking_params = [f"%{text.lower()}%"] * 3
        all_params = params + ranking_params
        
        return self.execute_query(query, tuple(all_params))
    
    def search_objects(self, object_names: List[str], 
                      confidence_threshold: float = 0.5) -> List[Dict]:
        """Search keyframes containing specific objects"""
        placeholders = ','.join(['?' for _ in object_names])
        
        query = f"""
        SELECT DISTINCT o.video_id, o.keyframe_id, o.object_name, 
               AVG(o.confidence) as avg_confidence,
               COUNT(*) as object_count,
               k.pts_time, k.frame_idx
        FROM objects o
        JOIN keyframes k ON o.video_id = k.video_id AND o.keyframe_id = k.keyframe_id
        WHERE o.object_name IN ({placeholders}) 
        AND o.confidence >= ?
        GROUP BY o.video_id, o.keyframe_id
        ORDER BY avg_confidence DESC, object_count DESC
        """
        
        params = tuple(object_names + [confidence_threshold])
        return self.execute_query(query, params)
    
    def get_keyframes_in_timerange(self, video_id: str, 
                                  start_time: float, end_time: float) -> List[Dict]:
        """Get keyframes within time range"""
        query = """
        SELECT k.*, v.title, v.author
        FROM keyframes k
        JOIN videos v ON k.video_id = v.video_id
        WHERE k.video_id = ? AND k.pts_time BETWEEN ? AND ?
        ORDER BY k.pts_time
        """
        
        return self.execute_query(query, (video_id, start_time, end_time))
    
    def get_video_metadata(self, video_id: str) -> Optional[Dict]:
        """Get full metadata for a video"""
        query = "SELECT * FROM videos WHERE video_id = ?"
        results = self.execute_query(query, (video_id,))
        return results[0] if results else None
    
    def get_keyframe_objects(self, video_id: str, keyframe_id: str) -> List[Dict]:
        """Get all objects in a specific keyframe"""
        query = """
        SELECT object_name, confidence, ymin, xmin, ymax, xmax
        FROM objects 
        WHERE video_id = ? AND keyframe_id = ?
        ORDER BY confidence DESC
        """
        
        return self.execute_query(query, (video_id, keyframe_id))
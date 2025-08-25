import re
from typing import Dict, List, Tuple

class QueryParser:
    """Utility class for parsing complex queries"""
    
    @staticmethod
    def extract_time_references(query: str) -> List[Dict]:
        """Extract time references from query"""
        time_patterns = [
            (r'phút\s+(\d+)', lambda m: int(m.group(1)) * 60),
            (r'(\d+):(\d+)', lambda m: int(m.group(1)) * 60 + int(m.group(2))),
            (r'giây\s+(\d+)', lambda m: int(m.group(1))),
            (r'từ\s+(\d+)\s+đến\s+(\d+)', lambda m: (int(m.group(1)), int(m.group(2))))
        ]
        
        time_refs = []
        for pattern, converter in time_patterns:
            matches = re.finditer(pattern, query.lower())
            for match in matches:
                try:
                    time_refs.append({
                        'text': match.group(0),
                        'value': converter(match),
                        'start': match.start(),
                        'end': match.end()
                    })
                except:
                    continue
        
        return time_refs
    
    @staticmethod
    def extract_video_ids(query: str) -> List[str]:
        """Extract video IDs from query"""
        pattern = r'[L]\d+_V\d+'
        return re.findall(pattern, query.upper())
    
    @staticmethod
    def extract_colors(query: str) -> List[str]:
        """Extract color references"""
        colors = ['đỏ', 'xanh', 'vàng', 'đen', 'trắng', 'xám', 'nâu', 'hồng', 'tím', 'cam']
        found_colors = []
        query_lower = query.lower()
        
        for color in colors:
            if color in query_lower:
                found_colors.append(color)
        
        return found_colors
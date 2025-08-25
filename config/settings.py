import os
from dotenv import load_dotenv
from pathlib import Path
from pydantic_settings import BaseSettings
from typing import Optional, List, Dict

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

class Settings(BaseSettings):
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY")
    
    LLM_MODEL_NAME: str = "gemini-2.5-flash"
    
    EMBEDDING_MODEL_NAME: str = "models/embedding-001"
    
    # AGENT_MODEL: str = "mistralai/Mistral-7B-Instruct-v0.3"
    # VISION_MODEL: str = "llava-hf/llava-1.5-7b-hf"
    # EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    VECTOR_STORE_PATH: Path = BASE_DIR / "data" / "processed_data" / "vector_store"
    METADATA_KEYFRAME_OBJECT_DB_PATH: Path = BASE_DIR / "data" / "processed_data" / "metadata_keyframe_object.db"
    
    RAW_DATA: Path = BASE_DIR / "data" / "raw_data"
    RAW_VIDEO_DIR: Path = RAW_DATA / "videos"
    RAW_KEYFRAME_DIR: Path = RAW_DATA / "keyframes"
    RAW_MAP_KEYFRAME_DIR: Path = RAW_DATA / "map-keyframes"
    RAW_OBJECT_DIR: Path = RAW_DATA / "objects"
    RAW_CLIPFEATURE_DIR: Path = RAW_DATA / "clip-features-32"
    RAW_METADATA_DIR: Path = RAW_DATA / "media-info"
    
    QDRANT_HOST: str = "127.0.0.1"
    QDRANT_PORT: int = 6333
    QDRANT_VIDEO_COLLECTION_NAME: str = "video_collection"
    QDRANT_KEYWORD_COLLECTION_NAME: str = "keyword_collection"
    
    # # ===== AGENT SETTINGS =====
    # # LLM Settings
    # LLM_TEMPERATURE: float = 0.1
    # LLM_MAX_TOKENS: int = 2048
    # LLM_TIMEOUT: int = 30 # seconds
    
    # # Search Settings
    # DEFAULT_SIMILARITY_THRESHOLD: float = 0.7
    # DEFAULT_MAX_RESULTS: int = 50
    # DEFAULT_CONFIDENCE_THRESHOLD: float = 0.6
    
    # # Caching Settings
    # ENABLE_CACHE: bool = True
    # CACHE_TTL: int = 3600  # 1 hour
    # MAX_CACHE_SIZE: int = 1000
    
    # # ===== PERFORMANCE SETTINGS =====
    # # Parallel processing
    # MAX_CONCURRENT_AGENTS: int = 3
    # AGENT_TIMEOUT: int = 60  # seconds
    
    # # Result limits
    # MAX_RESULTS_PER_AGENT: int = 100
    # MAX_FINAL_RESULTS: int = 50
    
    # # ===== AGENT WEIGHTS =====
    # AGENT_CONFIDENCE_WEIGHTS: Dict = {
    #     'TextSearchAgent': 0.8,
    #     'VisualSearchAgent': 0.9,
    #     'TemporalAgent': 0.95,
    #     'ResultFusionAgent': 1.0
    # }
    
    # # ===== SEARCH PARAMETERS =====
    # # Text search
    # TEXT_SEARCH_FIELDS: List = ['title', 'description', 'keywords', 'author']
    # TEXT_RANKING_WEIGHTS: Dict = {
    #     'title_match': 1.0,
    #     'keyword_match': 0.8,
    #     'description_match': 0.6,
    #     'author_match': 0.9
    # }
    
    # # Object search
    # OBJECT_CONFIDENCE_THRESHOLD: float = 0.6
    # OBJECT_RANKING_WEIGHT: float = 0.9
    
    # # Visual search
    # VISUAL_SIMILARITY_THRESHOLD: float = 0.7
    # VISUAL_DIVERSITY_FILTER: bool = True
    # MAX_KEYFRAMES_PER_VIDEO: int = 5
    
    # # ===== LOGGING =====
    # LOG_LEVEL: str = os.getenv('LOG_LEVEL', 'INFO')
    # LOG_FORMAT: str = '[%(asctime)s] %(name)s - %(levelname)s - %(message)s'
    
    # # ===== DEVELOPMENT =====
    # DEBUG_MODE: bool = os.getenv('DEBUG_MODE', 'False').lower() == 'true'
    # MOCK_LLM_RESPONSES: bool = False  # For testing without API calls
    
    def validate_settings(self) -> List[str]:
        """Validate required settings"""
        errors = []
        
        # Check API key
        if not self.GOOGLE_API_KEY or self.GOOGLE_API_KEY == 'YOUR_API_KEY':
            errors.append("GOOGLE_API_KEY not configured")
        
        # Check database file
        if not self.METADATA_KEYFRAME_OBJECT_DB_PATH:
            errors.append(f"Database not found: {self.METADATA_KEYFRAME_OBJECT_DB_PATH}")
        
        # Check raw data directories
        required_dirs = [
            self.RAW_VIDEO_DIR,
            self.RAW_KEYFRAME_DIR,
            self.RAW_MAP_KEYFRAME_DIR,
            self.RAW_OBJECT_DIR,
            self.RAW_CLIPFEATURE_DIR, 
            self.RAW_METADATA_DIR,
        ]
        
        for directory in required_dirs:
            if not directory:
                errors.append(f"Required directory not found: {directory}")
        
        return errors

settings = Settings()
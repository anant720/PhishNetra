"""
Configuration settings for PhishNetra
Production-ready configuration with environment-based settings
"""

import os
from typing import Optional, List
from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    """Application settings with Pydantic validation"""

    # API Configuration
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    api_workers: int = Field(default=1, env="API_WORKERS")

    # Model Configuration
    model_cache_dir: str = Field(default="./models", env="MODEL_CACHE_DIR")
    fasttext_model_path: str = Field(default="./models/fasttext_scam.bin", env="FASTTEXT_MODEL_PATH")
    sentence_transformer_model: str = Field(default="all-MiniLM-L6-v2", env="SENTENCE_TRANSFORMER_MODEL")
    distilbert_model_path: str = Field(default="./models/distilbert_scam", env="DISTILBERT_MODEL_PATH")
    faiss_index_path: str = Field(default="./models/scam_index.faiss", env="FAISS_INDEX_PATH")

    # Inference Configuration
    max_sequence_length: int = Field(default=512, env="MAX_SEQUENCE_LENGTH")
    batch_size: int = Field(default=16, env="BATCH_SIZE")
    inference_timeout: float = Field(default=30.0, env="INFERENCE_TIMEOUT")

    # Decision Fusion Configuration
    fasttext_weight: float = Field(default=0.25, env="FASTTEXT_WEIGHT")
    sentence_transformer_weight: float = Field(default=0.30, env="SENTENCE_TRANSFORMER_WEIGHT")
    distilbert_weight: float = Field(default=0.35, env="DISTILBERT_WEIGHT")
    similarity_weight: float = Field(default=0.10, env="SIMILARITY_WEIGHT")

    # Similarity Engine Configuration
    similarity_threshold: float = Field(default=0.85, env="SIMILARITY_THRESHOLD")
    max_similar_results: int = Field(default=5, env="MAX_SIMILAR_RESULTS")
    faiss_nprobe: int = Field(default=10, env="FAISS_NPROBE")

    # Explainability Configuration
    highlight_top_k: int = Field(default=3, env="HIGHLIGHT_TOP_K")
    min_confidence_threshold: float = Field(default=0.1, env="MIN_CONFIDENCE_THRESHOLD")

    # Performance Configuration
    enable_gpu: bool = Field(default=False, env="ENABLE_GPU")
    gpu_memory_fraction: float = Field(default=0.7, env="GPU_MEMORY_FRACTION")
    enable_model_caching: bool = Field(default=True, env="ENABLE_MODEL_CACHING")

    # URL Analysis Configuration
    url_analysis_timeout_seconds: float = Field(default=10.0, env="URL_ANALYSIS_TIMEOUT_SECONDS")

    # Security Configuration
    rate_limit_requests: int = Field(default=100, env="RATE_LIMIT_REQUESTS")
    rate_limit_window: int = Field(default=60, env="RATE_LIMIT_WINDOW")  # seconds

    # Logging Configuration
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_format: str = Field(default="json", env="LOG_FORMAT")
    enable_access_logs: bool = Field(default=True, env="ENABLE_ACCESS_LOGS")

    # Environment
    environment: str = Field(default="development", env="ENVIRONMENT")
    debug_mode: bool = Field(default=False, env="DEBUG_MODE")

    # CORS Configuration - stored as string, parsed to list
    _cors_origins_str: str = Field(default="http://localhost:3000,http://localhost:3001", env="CORS_ORIGINS")
    
    @property
    def cors_origins(self) -> List[str]:
        """Parse CORS origins from string (comma-separated or JSON array)"""
        raw = self._cors_origins_str if hasattr(self, '_cors_origins_str') else ''
        if not raw or (isinstance(raw, str) and raw.strip() == ''):
            return ["http://localhost:3000", "http://localhost:3001"]
        
        # Try JSON first (for backward compatibility)
        if isinstance(raw, str):
            try:
                import json
                parsed = json.loads(raw)
                if isinstance(parsed, list):
                    return parsed
            except (json.JSONDecodeError, ValueError, TypeError):
                pass
            
            # Fall back to comma-separated string
            origins = [origin.strip() for origin in raw.split(',') if origin.strip()]
            # Handle wildcard
            if origins and origins[0] == '*':
                return ['*']
            return origins if origins else ["http://localhost:3000", "http://localhost:3001"]
        
        # If somehow it's already a list
        if isinstance(raw, list):
            return raw
        
        return ["http://localhost:3000", "http://localhost:3001"]

    # Database Configuration (for future use)
    database_url: Optional[str] = Field(default=None, env="DATABASE_URL")

    class Config:
        """Pydantic configuration"""
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get application settings"""
    return settings


def create_directories():
    """Create necessary directories"""
    directories = [
        settings.model_cache_dir,
        os.path.dirname(settings.fasttext_model_path),
        os.path.dirname(settings.distilbert_model_path),
        os.path.dirname(settings.faiss_index_path),
        "./logs",
        "./temp"
    ]

    for directory in directories:
        if directory:
            os.makedirs(directory, exist_ok=True)


# Create directories on import
create_directories()
# src/config.py
import os
from pathlib import Path
from typing import Dict, List
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Database
    DATABASE_URL: str = "postgresql://user:password@localhost:5432/taskrouter"
    REDIS_URL: str = "redis://localhost:6379"
    
    # API Keys
    GITHUB_TOKEN: str = ""
    OPENAI_API_KEY: str = ""
    
    # Security
    SECRET_KEY: str = "dev-secret-key"
    DEBUG: bool = True
    
    # Model Configuration
    CODEBERT_MODEL: str = "microsoft/codebert-base"
    GRAPHCODEBERT_MODEL: str = "microsoft/graphcodebert-base"
    SENTENCE_TRANSFORMER_MODEL: str = "all-MiniLM-L6-v2"
    
    # Expertise Modeling
    SKILL_VECTOR_DIM: int = 768
    COLLABORATION_WINDOW_DAYS: int = 90
    LEARNING_VELOCITY_WINDOW_DAYS: int = 180
    MIN_COMMITS_FOR_ANALYSIS: int = 10
    
    # File Paths
    BASE_DIR: Path = Path(__file__).parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    MODELS_DIR: Path = DATA_DIR / "models"
    EMBEDDINGS_DIR: Path = DATA_DIR / "embeddings"
    
    # Supported Programming Languages
    SUPPORTED_LANGUAGES: List[str] = [
        "python", "javascript", "typescript", "java", "cpp", 
        "c", "go", "rust", "ruby", "php", "kotlin", "swift"
    ]
    
    # Domain Knowledge Categories
    DOMAIN_CATEGORIES: List[str] = [
        "frontend", "backend", "database", "devops", "ml", "security",
        "testing", "mobile", "desktop", "web", "cloud", "infrastructure"
    ]
    
    class Config:
        env_file = ".env"

settings = Settings()

# Ensure data directories exist
settings.DATA_DIR.mkdir(exist_ok=True)
settings.MODELS_DIR.mkdir(exist_ok=True)
settings.EMBEDDINGS_DIR.mkdir(exist_ok=True)
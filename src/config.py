# src/config.py
import os
from pathlib import Path
from typing import Dict, List
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Get absolute paths BEFORE class definition
_BASE_DIR = Path(__file__).parent.parent.absolute()
_DATA_DIR = _BASE_DIR / "data"
_DB_PATH = _DATA_DIR / "taskrouter.db"

# Load environment variables
load_dotenv(_BASE_DIR / ".env")

class Settings(BaseSettings):
    # Database (using pre-computed absolute string paths)
    DATABASE_URL: str = f"sqlite:///{str(_DB_PATH)}"
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
    
    # File Paths (using pre-computed absolute paths)
    BASE_DIR: Path = _BASE_DIR
    DATA_DIR: Path = _DATA_DIR
    MODELS_DIR: Path = _DATA_DIR / "models"
    EMBEDDINGS_DIR: Path = _DATA_DIR / "embeddings"
    
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
    
    # Learning System Configuration
    LEARNING_RATE: float = 0.1
    MIN_SAMPLE_SIZE_FOR_LEARNING: int = 5
    CONFIDENCE_THRESHOLD: float = 0.7
    MODEL_UPDATE_THRESHOLD: float = 0.05

    # Performance Monitoring
    ALERT_ASSIGNMENT_SUCCESS_RATE: float = 0.7
    ALERT_DEVELOPER_SATISFACTION: float = 0.6
    ALERT_PREDICTION_ACCURACY: float = 0.7
    ALERT_SYSTEM_PERFORMANCE: float = 0.75

    # A/B Testing
    DEFAULT_EXPERIMENT_DURATION_DAYS: int = 14
    MIN_EXPERIMENT_SAMPLE_SIZE: int = 20
    STATISTICAL_SIGNIFICANCE_THRESHOLD: float = 0.05

    # ROI Calculation
    DEVELOPER_HOURLY_RATE: float = 75.0
    ESTIMATED_TIME_SAVED_PER_ASSIGNMENT: float = 2.0
    BASELINE_MANUAL_SUCCESS_RATE: float = 0.6

    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'

settings = Settings()

# Ensure data directories exist (using absolute paths)
_DATA_DIR.mkdir(exist_ok=True)
(_DATA_DIR / "models").mkdir(exist_ok=True)
(_DATA_DIR / "embeddings").mkdir(exist_ok=True)

# Debug verification
print(f"🔧 DEBUG: DATABASE_URL: {settings.DATABASE_URL}")
print(f"🔧 DEBUG: Database file exists: {_DB_PATH.exists()}")
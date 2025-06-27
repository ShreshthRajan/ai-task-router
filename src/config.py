# src/config.py
import os
from pathlib import Path
from typing import Dict, List
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

from pydantic import field_validator

# Get absolute paths BEFORE class definition
_BASE_DIR = Path(__file__).parent.parent.absolute()
_DATA_DIR = _BASE_DIR / "data"
_DB_PATH = _DATA_DIR / "taskrouter.db"

# Load environment variables
env_file_path = _BASE_DIR / ".env"
print(f"🔍 DEBUG: Looking for .env at: {env_file_path}")
print(f"🔍 DEBUG: .env file exists: {env_file_path.exists()}")
load_dotenv(env_file_path)

# Debug what we got from environment
print(f"🔍 DEBUG: Raw GITHUB_TOKENS from os.getenv: '{os.getenv('GITHUB_TOKENS', 'NOT_FOUND')}'")
print(f"🔍 DEBUG: Raw GITHUB_TOKEN from os.getenv: '{os.getenv('GITHUB_TOKEN', 'NOT_FOUND')}'")
print(f"🔍 DEBUG: All env vars with GITHUB: {[k for k in os.environ.keys() if 'GITHUB' in k]}")

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

    
    # ───────────────────────── GitHub ingestion ──────────────────────────
    # Raw comma-separated value from .env (e.g. PAT1,PAT2,PAT3)
    GITHUB_TOKENS_RAW: str = ""
    # Upper-bound for concurrent GitHub calls
    MAX_API_CONCURRENCY: int = 8

    _token_idx: int = -1  # round-robin cursor

    @property
    def GITHUB_TOKENS(self) -> List[str]:
        """Return the PAT pool as a clean list."""
        return [tok.strip() for tok in self.GITHUB_TOKENS_RAW.split(",") if tok.strip()]

    def pick_github_token(self) -> str:
        """
        Round-robin selection from the PAT pool.
        Falls back to the single `GITHUB_TOKEN` when the pool is empty.
        """
        tokens = self.GITHUB_TOKENS
        if not tokens:
            # Fallback to single token if available
            return self.GITHUB_TOKEN if self.GITHUB_TOKEN else ""
        
        # Use instance variable instead of class variable for thread safety
        if not hasattr(self, '_current_token_idx'):
            self._current_token_idx = -1
            
        self._current_token_idx = (self._current_token_idx + 1) % len(tokens)
        selected_token = tokens[self._current_token_idx]
        
        print(f"🔍 DEBUG: Selected token #{self._current_token_idx + 1} of {len(tokens)}: {selected_token[:10]}...")
        return selected_token


    @field_validator('GITHUB_TOKENS_RAW', mode='before')
    @classmethod
    def parse_github_tokens_raw(cls, v):
        """Parse GITHUB_TOKENS from environment."""
        # Always try to get from environment first
        env_value = os.getenv("GITHUB_TOKENS", "")
        if env_value:
            return env_value
        # Fallback to passed value
        return v if isinstance(v, str) else ""
    
    @field_validator('MAX_API_CONCURRENCY', mode='before')
    @classmethod
    def parse_max_api_concurrency(cls, v):
        """Parse MAX_API_CONCURRENCY from environment."""
        # Always try to get from environment first
        env_value = os.getenv("MAX_API_CONCURRENCY")
        if env_value:
            return int(env_value)
        # Fallback to passed value
        return v if isinstance(v, int) else 8

    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'
        extra = "ignore"

settings = Settings()

# Ensure data directories exist (using absolute paths)
_DATA_DIR.mkdir(exist_ok=True)
(_DATA_DIR / "models").mkdir(exist_ok=True)
(_DATA_DIR / "embeddings").mkdir(exist_ok=True)

# Debug verification
print(f"🔧 DEBUG: DATABASE_URL: {settings.DATABASE_URL}")
print(f"🔧 DEBUG: Database file exists: {_DB_PATH.exists()}")
print(f"🔍 DEBUG: GITHUB_TOKENS_RAW: '{settings.GITHUB_TOKENS_RAW}'")
print(f"🔍 DEBUG: GITHUB_TOKENS list: {settings.GITHUB_TOKENS}")
print(f"🔍 DEBUG: GITHUB_TOKEN (fallback): '{settings.GITHUB_TOKEN}'")

# Test token selection
if settings.GITHUB_TOKENS:
    test_token = settings.pick_github_token()
    print(f"🔍 DEBUG: Token selection test successful: {test_token[:10]}...")
else:
    print("⚠️  WARNING: No GitHub tokens available")
#!/usr/bin/env python3
"""
Setup script for AI Task Router Phase 1
Downloads models, initializes database, and verifies installation.
"""

import os
import sys
import asyncio
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.config import settings
from src.models.database import create_tables, get_db, SessionLocal
from src.core.developer_modeling.code_analyzer import CodeAnalyzer
from src.core.developer_modeling.skill_extractor import SkillExtractor
from src.integrations.github_client import GitHubClient

def download_models():
    """Download and cache required ML models."""
    print("📥 Downloading ML models...")
    
    try:
        # Initialize CodeAnalyzer (downloads CodeBERT)
        print("  • Loading CodeBERT model...")
        analyzer = CodeAnalyzer()
        print("  ✅ CodeBERT loaded successfully")
        
        # Initialize SkillExtractor (downloads SentenceTransformer)
        print("  • Loading SentenceTransformer model...")
        extractor = SkillExtractor()
        print("  ✅ SentenceTransformer loaded successfully")
        
        print("✅ All models downloaded and cached")
        return True
        
    except Exception as e:
        print(f"❌ Error downloading models: {e}")
        return False

def setup_database():
    """Initialize database tables."""
    print("🗄️  Setting up database...")
    
    try:
        # Create all tables
        create_tables()
        print("  ✅ Database tables created")
        
        # Test connection
        db = SessionLocal()
        db.execute("SELECT 1")
        db.close()
        print("  ✅ Database connection verified")
        
        return True
        
    except Exception as e:
        print(f"❌ Error setting up database: {e}")
        return False

def create_directories():
    """Create necessary directories."""
    print("📁 Creating directories...")
    
    directories = [
        settings.DATA_DIR,
        settings.MODELS_DIR,
        settings.EMBEDDINGS_DIR,
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"  ✅ Created {directory}")
    
    return True

def verify_installation():
    """Verify that all components are working."""
    print("🔍 Verifying installation...")
    
    try:
        # Test CodeAnalyzer
        analyzer = CodeAnalyzer()
        test_files = [{
            "filename": "test.py",
            "patch": "+def hello(): return 'world'",
            "additions": 1
        }]
        metrics = analyzer.analyze_commit(test_files, {"hash": "test"})
        print("  ✅ CodeAnalyzer working")
        
        # Test SkillExtractor
        extractor = SkillExtractor()
        test_data = {
            "developer_id": "test",
            "github_username": "test",
            "commits": [{
                "timestamp": "2024-01-01T00:00:00Z",
                "files": test_files
            }],
            "pr_reviews": [],
            "issue_comments": [],
            "discussions": [],
            "pr_descriptions": [],
            "commit_messages": ["test commit"]
        }
        profile = extractor.extract_comprehensive_profile(test_data)
        print("  ✅ SkillExtractor working")
        
        # Test GitHub client (without making actual API calls)
        github_client = GitHubClient()
        print("  ✅ GitHubClient initialized")
        
        print("✅ All components verified")
        return True
        
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        return False

def display_configuration():
    """Display current configuration."""
    print("\n⚙️  Configuration:")
    print(f"  • Database URL: {settings.DATABASE_URL}")
    print(f"  • Redis URL: {settings.REDIS_URL}")
    print(f"  • GitHub Token: {'✅ Set' if settings.GITHUB_TOKEN else '❌ Not set'}")
    print(f"  • Debug Mode: {settings.DEBUG}")
    print(f"  • Data Directory: {settings.DATA_DIR}")
    print(f"  • Models Directory: {settings.MODELS_DIR}")
    print(f"  • Skill Vector Dimension: {settings.SKILL_VECTOR_DIM}")

def main():
    """Main setup routine."""
    print("🚀 AI Task Router Phase 1 Setup")
    print("=" * 40)
    
    # Display configuration
    display_configuration()
    print()
    
    # Setup steps
    steps = [
        ("Creating directories", create_directories),
        ("Setting up database", setup_database),
        ("Downloading ML models", download_models),
        ("Verifying installation", verify_installation),
    ]
    
    for step_name, step_func in steps:
        print(f"\n{step_name}...")
        if not step_func():
            print(f"❌ Setup failed at: {step_name}")
            sys.exit(1)
    
    print("\n" + "=" * 40)
    print("✅ Setup completed successfully!")
    print("\nNext steps:")
    print("1. Set your GitHub token in .env file (optional but recommended)")
    print("2. Start the development server: docker-compose up")
    print("3. Visit http://localhost:8000 for the API")
    print("4. Visit http://localhost:8501 for the dashboard")
    print("\nPhase 1 (Developer Expertise Modeling) is ready! 🎉")

if __name__ == "__main__":
    main()
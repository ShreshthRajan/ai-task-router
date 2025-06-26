# scripts/setup.py
#!/usr/bin/env python3
"""
Setup script for AI Task Router Phase 1
Downloads models, initializes database, and verifies installation.
"""

import os
import sys
import asyncio
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import settings
import sys
sys.path.append('src')
from models.database import create_tables, get_db, SessionLocal

from src.core.developer_modeling.code_analyzer import CodeAnalyzer
from src.core.developer_modeling.skill_extractor import SkillExtractor
from src.integrations.github_client import GitHubClient
from sqlalchemy import text

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
        db.execute(text("SELECT 1"))
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
                "timestamp": "2024-01-01T00:00:00",
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
        ("Seeding database", seed_database),
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

def seed_database():
    """Seed database with initial sample data for testing."""
    print("🌱 Seeding database with sample data...")
    
    try:
        from src.models.database import (
            Developer, Task, TaskAssignment, AssignmentOutcome, 
            ModelPerformance, DeveloperPreference, SkillImportanceFactor
        )
        
        db = SessionLocal()
        
        # Check if data already exists
        existing_dev = db.query(Developer).filter(Developer.email == "dev@example.com").first()
        if existing_dev:
            print("  ✅ Sample data already exists, skipping seeding")
            db.close()
            return True
        
        # Create sample developer
        developer = Developer(
            github_username="sample_dev",
            name="Sample Developer",
            email="dev@example.com",
            skill_vector={"python": 0.8, "javascript": 0.6},
            primary_languages={"python": 0.8, "javascript": 0.6},
            domain_expertise={"backend": 0.7, "frontend": 0.5},
            collaboration_score=0.75,
            learning_velocity=0.6
        )
        db.add(developer)
        db.commit()
        
        # Create sample task
        task = Task(
            title="Sample Task",
            description="A sample task for testing",
            repository="test/repo",
            status="completed",
            technical_complexity=0.6,
            domain_difficulty=0.5,
            collaboration_requirements=0.4,
            learning_opportunities=0.7,
            business_impact=0.8,
            estimated_hours=8.0,
            complexity_confidence=0.9
        )
        db.add(task)
        db.commit()
        
        # Create sample assignment
        assignment = TaskAssignment(
            task_id=task.id,
            developer_id=developer.id,
            status="completed",
            confidence_score=0.85,
            reasoning="Good skill match",
            actual_hours=7.5,
            feedback_score=0.9,
            productivity_score=0.88,
            skill_development_score=0.75,
            collaboration_effectiveness=0.8
        )
        db.add(assignment)
        db.commit()
        
        # Create sample outcome
        outcome = AssignmentOutcome(
            assignment_id=assignment.id,
            task_completion_quality=0.9,
            developer_satisfaction=0.85,
            learning_achieved=0.75,
            collaboration_effectiveness=0.8,
            time_estimation_accuracy=0.95,
            performance_metrics={"code_quality": 0.9, "requirements_met": 0.95},
            skill_improvements=["python", "testing"],
            challenges_faced=["complex algorithms"],
            success_factors=["good documentation", "clear requirements"]
        )
        db.add(outcome)
        
        # Create sample model performance
        model_perf = ModelPerformance(
            model_name="complexity_predictor",
            version="1.0",
            accuracy_score=0.87,
            prediction_count=100,
            correct_predictions=87,
            average_confidence=0.82
        )
        db.add(model_perf)
        
        # Create sample developer preference
        dev_pref = DeveloperPreference(
            developer_id=developer.id,
            preferred_complexity_min=0.4,
            preferred_complexity_max=0.8,
            complexity_comfort_zone=0.6,
            learning_appetite=0.7,
            preference_confidence=0.8,
            sample_size=10
        )
        db.add(dev_pref)
        
        # Create sample skill importance factor
        skill_factor = SkillImportanceFactor(
            task_type="feature_development",
            complexity_range="medium",
            domain="backend",
            skill_name="python",
            importance_factor=0.9,
            confidence=0.8,
            successful_assignments=8,
            total_assignments=10
        )
        db.add(skill_factor)
        
        db.commit()
        db.close()
        
        print("  ✅ Sample data created successfully")
        return True
        
    except Exception as e:
        print(f"❌ Error seeding database: {e}")
        return False

if __name__ == "__main__":
    main()
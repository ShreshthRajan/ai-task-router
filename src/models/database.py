# src/models/database.py
from sqlalchemy import Column, Integer, String, DateTime, Text, JSON, Float, ForeignKey, Boolean
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from sqlalchemy import create_engine
from datetime import datetime
from typing import Dict, List, Optional
import json

from config import settings

Base = declarative_base()

class Developer(Base):
    __tablename__ = "developers"
    
    id = Column(Integer, primary_key=True, index=True)
    github_username = Column(String, unique=True, index=True, nullable=False)
    email = Column(String, unique=True, index=True)
    name = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Current expertise snapshot
    skill_vector = Column(JSON)  # Multi-dimensional skill representation
    primary_languages = Column(JSON)  # Top programming languages
    domain_expertise = Column(JSON)  # Domain knowledge areas
    collaboration_score = Column(Float, default=0.0)
    learning_velocity = Column(Float, default=0.0)
    
    # Relationships
    code_analyses = relationship("CodeAnalysis", back_populates="developer")
    collaborations = relationship("CollaborationEvent", back_populates="developer")
    expertise_snapshots = relationship("ExpertiseSnapshot", back_populates="developer")
    assignments = relationship("TaskAssignment", back_populates="developer")

class CodeAnalysis(Base):
    __tablename__ = "code_analyses"
    
    id = Column(Integer, primary_key=True, index=True)
    developer_id = Column(Integer, ForeignKey("developers.id"))
    
    # Commit information
    commit_hash = Column(String, index=True)
    repository = Column(String)
    timestamp = Column(DateTime)
    
    # Code metrics
    files_changed = Column(Integer)
    lines_added = Column(Integer)
    lines_deleted = Column(Integer)
    complexity_score = Column(Float)
    
    # Language analysis
    primary_language = Column(String)
    language_distribution = Column(JSON)
    
    # Semantic analysis
    code_embedding = Column(JSON)  # CodeBERT embedding
    semantic_topics = Column(JSON)  # Extracted topics/patterns
    technical_concepts = Column(JSON)  # Domain-specific concepts
    
    developer = relationship("Developer", back_populates="code_analyses")

class CollaborationEvent(Base):
    __tablename__ = "collaboration_events"
    
    id = Column(Integer, primary_key=True, index=True)
    developer_id = Column(Integer, ForeignKey("developers.id"))
    
    # Event details
    event_type = Column(String)  # pr_review, issue_comment, discussion
    repository = Column(String)
    timestamp = Column(DateTime)
    
    # Collaboration metrics
    participants = Column(JSON)  # Other developers involved
    interaction_quality = Column(Float)  # Helpfulness/expertise shown
    knowledge_shared = Column(Boolean, default=False)
    
    # Content analysis
    content_embedding = Column(JSON)  # Semantic embedding of text
    technical_keywords = Column(JSON)  # Extracted technical terms
    
    developer = relationship("Developer", back_populates="collaborations")

class ExpertiseSnapshot(Base):
    __tablename__ = "expertise_snapshots"
    
    id = Column(Integer, primary_key=True, index=True)
    developer_id = Column(Integer, ForeignKey("developers.id"))
    
    # Temporal tracking
    snapshot_date = Column(DateTime, default=datetime.utcnow)
    
    # Expertise metrics at this point in time
    skill_vector = Column(JSON)
    confidence_scores = Column(JSON)  # Confidence in each skill area
    learning_trends = Column(JSON)  # Skills being acquired/lost
    
    # Performance indicators
    productivity_score = Column(Float)
    code_quality_score = Column(Float)
    collaboration_effectiveness = Column(Float)
    
    developer = relationship("Developer", back_populates="expertise_snapshots")

# NEW PHASE 2 MODELS

class Task(Base):
    __tablename__ = "tasks"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Task identification
    github_issue_id = Column(Integer, index=True)
    repository = Column(String, index=True)
    title = Column(String, nullable=False)
    description = Column(Text)
    
    # Task metadata
    status = Column(String, default="open")  # open, in_progress, completed, closed
    priority = Column(String, default="medium")  # low, medium, high, critical
    labels = Column(JSON)  # GitHub labels
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Complexity analysis results
    technical_complexity = Column(Float)
    domain_difficulty = Column(Float)
    collaboration_requirements = Column(Float)
    learning_opportunities = Column(Float)
    business_impact = Column(Float)
    estimated_hours = Column(Float)
    complexity_confidence = Column(Float)
    
    # Structured data
    required_skills = Column(JSON)  # Skills needed for this task
    complexity_factors = Column(JSON)  # Detailed complexity breakdown
    risk_factors = Column(JSON)  # Identified risks
    
    # Relationships
    complexity_analysis = relationship("TaskComplexityAnalysis", back_populates="task", uselist=False)
    requirements = relationship("TaskRequirement", back_populates="task")
    assignments = relationship("TaskAssignment", back_populates="task")

class TaskComplexityAnalysis(Base):
    __tablename__ = "task_complexity_analyses"
    
    id = Column(Integer, primary_key=True, index=True)
    task_id = Column(Integer, ForeignKey("tasks.id"), unique=True)
    
    # Analysis metadata
    analysis_date = Column(DateTime, default=datetime.utcnow)
    analysis_version = Column(String, default="1.0")
    
    # Raw analysis data
    extracted_features = Column(JSON)  # Raw features extracted from task
    mentioned_technologies = Column(JSON)  # Technologies identified
    affected_components = Column(JSON)  # System components affected
    architectural_impact = Column(Float)  # Architecture change impact
    
    # Text analysis results
    text_complexity = Column(JSON)  # Linguistic complexity metrics
    semantic_features = Column(JSON)  # NLP semantic analysis
    urgency_indicators = Column(JSON)  # Urgency/priority indicators
    
    # Repository context
    repo_complexity = Column(Float)  # Repository complexity score
    recent_activity = Column(Float)  # Recent development activity
    team_size_estimate = Column(Integer)  # Estimated team size
    
    task = relationship("Task", back_populates="complexity_analysis")

class TaskRequirement(Base):
    __tablename__ = "task_requirements"
    
    id = Column(Integer, primary_key=True, index=True)
    task_id = Column(Integer, ForeignKey("tasks.id"))
    
    # Requirement details
    requirement_id = Column(String)  # Internal requirement ID
    category = Column(String)  # functional, non_functional, technical
    priority = Column(String)  # must, should, could, won't
    description = Column(Text)
    
    # Requirement analysis
    acceptance_criteria = Column(JSON)  # List of acceptance criteria
    dependencies = Column(JSON)  # Dependencies on other requirements
    technical_constraints = Column(JSON)  # Technical constraints
    estimated_complexity = Column(Float)  # Individual requirement complexity
    confidence_score = Column(Float)  # Parsing confidence
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    task = relationship("Task", back_populates="requirements")

class TaskAssignment(Base):
    __tablename__ = "task_assignments"
    
    id = Column(Integer, primary_key=True, index=True)
    task_id = Column(Integer, ForeignKey("tasks.id"))
    developer_id = Column(Integer, ForeignKey("developers.id"))
    
    # Assignment details
    assigned_at = Column(DateTime, default=datetime.utcnow)
    status = Column(String, default="suggested")  # suggested, accepted, rejected, completed
    confidence_score = Column(Float)  # Assignment confidence
    reasoning = Column(Text)  # Why this assignment was made
    
    # Outcome tracking
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    actual_hours = Column(Float)
    feedback_score = Column(Float)  # Post-completion feedback
    feedback_comments = Column(Text)
    
    # Performance metrics
    productivity_score = Column(Float)  # How well the assignment went
    skill_development_score = Column(Float)  # Learning achieved
    collaboration_effectiveness = Column(Float)  # Team collaboration quality
    
    task = relationship("Task", back_populates="assignments")
    developer = relationship("Developer", back_populates="assignments")

# Database setup
engine = create_engine(settings.DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def create_tables():
    """Create all database tables."""
    Base.metadata.create_all(bind=engine)

def get_db():
    """Get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

class AssignmentOutcome(Base):
    """Detailed outcome tracking for completed assignments."""
    __tablename__ = "assignment_outcomes"
    
    id = Column(Integer, primary_key=True, index=True)
    assignment_id = Column(Integer, ForeignKey("task_assignments.id"), unique=True)
    
    # Performance metrics
    task_completion_quality = Column(Float)  # Code quality, requirements met
    developer_satisfaction = Column(Float)  # Developer feedback on assignment fit
    learning_achieved = Column(Float)  # Actual skill development
    collaboration_effectiveness = Column(Float)  # Team interaction quality
    time_estimation_accuracy = Column(Float)  # Predicted vs actual time
    
    # Detailed outcome data
    performance_metrics = Column(JSON)  # Detailed performance breakdown
    skill_improvements = Column(JSON)  # Specific skills developed
    challenges_faced = Column(JSON)  # Difficulties encountered
    success_factors = Column(JSON)  # What made the assignment successful
    
    # Learning insights
    prediction_accuracy = Column(JSON)  # How accurate were our predictions
    unexpected_learnings = Column(JSON)  # Unexpected skill developments
    improvement_suggestions = Column(JSON)  # Suggestions for future assignments
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationship
    assignment = relationship("TaskAssignment", backref="outcome")

class ModelPerformance(Base):
    """Track performance of ML models over time."""
    __tablename__ = "model_performance"
    
    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String, index=True)  # e.g., "complexity_predictor", "skill_extractor"
    version = Column(String)
    
    # Performance metrics
    accuracy_score = Column(Float)
    precision_score = Column(Float)
    recall_score = Column(Float)
    f1_score = Column(Float)
    mse_score = Column(Float)  # For regression tasks
    
    # Model details
    training_data_size = Column(Integer)
    validation_data_size = Column(Integer)
    feature_count = Column(Integer)
    hyperparameters = Column(JSON)
    
    # Performance tracking
    prediction_count = Column(Integer, default=0)
    correct_predictions = Column(Integer, default=0)
    average_confidence = Column(Float)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class SystemMetrics(Base):
    """System-wide performance and health metrics."""
    __tablename__ = "system_metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Performance metrics
    avg_assignment_quality = Column(Float)
    avg_developer_satisfaction = Column(Float)
    avg_skill_development_rate = Column(Float)
    avg_collaboration_effectiveness = Column(Float)
    
    # System health
    total_assignments = Column(Integer)
    completed_assignments = Column(Integer)
    successful_assignments = Column(Integer)
    assignment_success_rate = Column(Float)
    
    # Learning system metrics
    active_learning_models = Column(Integer)
    learning_accuracy_trend = Column(Float)
    prediction_confidence_avg = Column(Float)
    
    # Productivity metrics
    team_productivity_score = Column(Float)
    task_completion_velocity = Column(Float)
    workload_balance_score = Column(Float)
    
    # Time period
    metric_period = Column(String)  # daily, weekly, monthly
    period_start = Column(DateTime)
    period_end = Column(DateTime)
    
    created_at = Column(DateTime, default=datetime.utcnow)

class LearningExperiment(Base):
    """Track A/B testing and experimental configurations."""
    __tablename__ = "learning_experiments"
    
    id = Column(Integer, primary_key=True, index=True)
    experiment_name = Column(String, unique=True, index=True)
    
    # Experiment configuration
    experiment_type = Column(String)  # ab_test, parameter_optimization, model_comparison
    status = Column(String, default="active")  # active, completed, paused, cancelled
    
    # Configuration details
    control_config = Column(JSON)  # Baseline configuration
    experimental_config = Column(JSON)  # New configuration being tested
    success_metrics = Column(JSON)  # Metrics to measure success
    
    # Results
    control_results = Column(JSON)
    experimental_results = Column(JSON)
    statistical_significance = Column(Float)
    confidence_level = Column(Float)
    
    # Experiment timeline
    start_date = Column(DateTime, default=datetime.utcnow)
    end_date = Column(DateTime)
    sample_size = Column(Integer)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class DeveloperPreference(Base):
    """Learned preferences for individual developers."""
    __tablename__ = "developer_preferences"
    
    id = Column(Integer, primary_key=True, index=True)
    developer_id = Column(Integer, ForeignKey("developers.id"), unique=True)
    
    # Complexity preferences (learned from assignment outcomes)
    preferred_complexity_min = Column(Float, default=0.3)
    preferred_complexity_max = Column(Float, default=0.8)
    complexity_comfort_zone = Column(Float, default=0.6)
    
    # Collaboration preferences
    prefers_solo_work = Column(Float, default=0.5)  # 0 = team work, 1 = solo work
    optimal_team_size = Column(Float, default=3.0)
    mentoring_preference = Column(Float, default=0.5)  # 0 = learn, 1 = teach
    
    # Learning preferences
    learning_appetite = Column(Float, default=0.7)  # Desire for challenging tasks
    preferred_learning_style = Column(String, default="gradual")  # gradual, immersive, mixed
    
    # Workload preferences
    optimal_workload_hours = Column(Float, default=40.0)
    workload_tolerance = Column(Float, default=0.2)  # How much variance they can handle
    
    # Performance patterns (learned)
    peak_performance_hours = Column(JSON)  # Hours when developer performs best
    task_switching_penalty = Column(Float, default=0.1)  # Performance impact of context switching
    
    # Confidence and accuracy
    preference_confidence = Column(Float, default=0.5)  # How confident we are in these preferences
    last_updated = Column(DateTime, default=datetime.utcnow)
    sample_size = Column(Integer, default=0)  # Number of assignments used to learn preferences
    
    developer = relationship("Developer", backref="preferences")

class SkillImportanceFactor(Base):
    """Learned importance factors for different skills in various contexts."""
    __tablename__ = "skill_importance_factors"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Context
    task_type = Column(String, index=True)  # e.g., "bug_fix", "feature_development", "refactoring"
    complexity_range = Column(String)  # low, medium, high
    domain = Column(String)  # frontend, backend, ml, etc.
    
    # Skill importance (learned from outcomes)
    skill_name = Column(String, index=True)
    importance_factor = Column(Float, default=1.0)  # How important this skill is in this context
    confidence = Column(Float, default=0.5)  # Confidence in this importance factor
    
    # Learning data
    successful_assignments = Column(Integer, default=0)
    total_assignments = Column(Integer, default=0)
    avg_performance_with_skill = Column(Float, default=0.0)  
    avg_performance_without_skill = Column(Float, default=0.0)  
    
    # Temporal tracking
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_learning_update = Column(DateTime, default=datetime.utcnow)

    
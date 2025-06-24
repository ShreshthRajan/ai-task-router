# src/models/database.py
from sqlalchemy import Column, Integer, String, DateTime, Text, JSON, Float, ForeignKey, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from sqlalchemy import create_engine
from datetime import datetime
from typing import Dict, List, Optional
import json

from ..config import settings

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
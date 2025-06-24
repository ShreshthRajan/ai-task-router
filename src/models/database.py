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
from pydantic import BaseModel, EmailStr
from typing import Dict, List, Optional
from datetime import datetime

# Developer Schemas
class DeveloperBase(BaseModel):
    github_username: str
    name: Optional[str] = None
    email: Optional[EmailStr] = None

class DeveloperCreate(DeveloperBase):
    pass

class DeveloperUpdate(BaseModel):
    name: Optional[str] = None
    email: Optional[EmailStr] = None

class DeveloperProfile(DeveloperBase):
    id: int
    programming_languages: Dict[str, float] = {}
    domain_expertise: Dict[str, float] = {}
    collaboration_score: float = 0.0
    learning_velocity: float = 0.0
    confidence_scores: Dict[str, float] = {}
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True

# Task Schemas
class TaskBase(BaseModel):
    title: str
    description: str
    repository: Optional[str] = None
    labels: List[str] = []
    priority: str = "medium"  # low, medium, high, critical

class TaskCreate(TaskBase):
    pass

class TaskUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    labels: Optional[List[str]] = None
    priority: Optional[str] = None
    status: Optional[str] = None

class Task(TaskBase):
    id: int
    status: str = "open"  # open, in_progress, completed, closed
    complexity_score: Optional[float] = None
    estimated_hours: Optional[float] = None
    required_skills: Dict[str, float] = {}
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True

# Assignment Schemas
class AssignmentBase(BaseModel):
    task_id: int
    developer_id: int
    confidence_score: float
    reasoning: str

class AssignmentCreate(AssignmentBase):
    pass

class Assignment(AssignmentBase):
    id: int
    status: str = "suggested"  # suggested, accepted, rejected, completed
    assigned_at: datetime
    completed_at: Optional[datetime] = None
    feedback_score: Optional[float] = None
    
    class Config:
        from_attributes = True

# Optimization Schemas
class OptimizationRequest(BaseModel):
    task_ids: List[int]
    developer_ids: List[int]
    objectives: List[str] = ["productivity", "skill_development", "workload_balance"]
    constraints: Dict[str, float] = {}

class OptimizationResult(BaseModel):
    assignments: List[Assignment]
    objective_scores: Dict[str, float]
    alternative_solutions: List[Dict] = []
    optimization_time_ms: float
    
# Skill Analysis Schemas
class SkillTrend(BaseModel):
    skill_name: str
    trend_direction: str  # increasing, decreasing, stable
    change_rate: float
    confidence: float

class LearningRecommendation(BaseModel):
    skill_area: str
    recommendation_type: str  # strengthen, explore, maintain
    priority: float
    reasoning: str

class SkillPrediction(BaseModel):
    skill_name: str
    prediction_type: str  # growth, decline, stable
    confidence: float
    projected_value: float
    growth_rate: float
    months_projected: int

# Team Analysis Schemas
class TeamMember(BaseModel):
    id: int
    name: str
    programming_languages: Dict[str, float]
    domain_expertise: Dict[str, float]
    collaboration_score: float
    learning_velocity: float

class TeamSkillMatrix(BaseModel):
    team_members: List[TeamMember]
    programming_languages: List[str]
    domain_expertise: List[str]
    team_strengths: List[str]
    team_gaps: List[str]
    collaboration_metrics: Dict[str, float]

# GitHub Integration Schemas
class GitHubCommit(BaseModel):
    hash: str
    message: str
    timestamp: datetime
    files_changed: List[Dict]
    additions: int
    deletions: int

class GitHubPullRequest(BaseModel):
    number: int
    title: str
    description: str
    state: str
    created_at: datetime
    merged_at: Optional[datetime] = None
    
class GitHubIssue(BaseModel):
    number: int
    title: str
    body: str
    state: str
    labels: List[str]
    created_at: datetime
    closed_at: Optional[datetime] = None

# Analytics Schemas
class DeveloperAnalytics(BaseModel):
    developer_id: int
    productivity_score: float
    code_quality_score: float
    collaboration_effectiveness: float
    skill_growth_rate: float
    recent_contributions: int
    
class TeamAnalytics(BaseModel):
    team_id: str
    team_productivity: float
    skill_coverage: float
    collaboration_health: float
    knowledge_distribution: Dict[str, float]
    bottlenecks: List[str]

# API Response Schemas
class SuccessResponse(BaseModel):
    message: str
    data: Optional[Dict] = None
    timestamp: datetime = datetime.utcnow()

class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    timestamp: datetime = datetime.utcnow()

class HealthCheck(BaseModel):
    status: str
    database: str
    components: Dict[str, str]
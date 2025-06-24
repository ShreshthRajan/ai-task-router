# src/models/schemas.py
from pydantic import BaseModel, EmailStr, Field
from typing import Dict, List, Optional
from datetime import datetime
from enum import Enum

# Developer Schemas (existing)
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

# NEW PHASE 2 SCHEMAS

# Task Schemas
class TaskBase(BaseModel):
    title: str
    description: Optional[str] = ""
    repository: Optional[str] = None
    labels: List[str] = []
    priority: str = Field(default="medium", regex="^(low|medium|high|critical)$")

class TaskCreate(TaskBase):
    github_issue_id: Optional[int] = None

class TaskUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    labels: Optional[List[str]] = None
    priority: Optional[str] = Field(None, regex="^(low|medium|high|critical)$")
    status: Optional[str] = Field(None, regex="^(open|in_progress|completed|closed)$")

class TaskComplexityResult(BaseModel):
    """Task complexity prediction result."""
    technical_complexity: float = Field(..., ge=0.0, le=1.0)
    domain_difficulty: float = Field(..., ge=0.0, le=1.0)
    collaboration_requirements: float = Field(..., ge=0.0, le=1.0)
    learning_opportunities: float = Field(..., ge=0.0, le=1.0)
    business_impact: float = Field(..., ge=0.0, le=1.0)
    estimated_hours: float = Field(..., gt=0)
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    complexity_factors: Dict[str, float] = {}
    required_skills: Dict[str, float] = {}
    risk_factors: List[str] = []

class Task(TaskBase):
    id: int
    github_issue_id: Optional[int] = None
    status: str = "open"
    created_at: datetime
    updated_at: datetime
    
    # Complexity analysis (optional)
    complexity_analysis: Optional[TaskComplexityResult] = None
    
    class Config:
        from_attributes = True

# Requirement Schemas
class RequirementBase(BaseModel):
    category: str = Field(..., regex="^(functional|non_functional|technical)$")
    priority: str = Field(..., regex="^(must|should|could|wont)$")
    description: str

class RequirementCreate(RequirementBase):
    acceptance_criteria: List[str] = []
    dependencies: List[str] = []
    technical_constraints: List[str] = []

class Requirement(RequirementBase):
    id: int
    requirement_id: str
    task_id: int
    acceptance_criteria: List[str] = []
    dependencies: List[str] = []
    technical_constraints: List[str] = []
    estimated_complexity: float
    confidence_score: float
    created_at: datetime
    
    class Config:
        from_attributes = True

class TaskRequirementsAnalysis(BaseModel):
    """Complete requirements analysis result."""
    task_id: str
    requirements: List[Requirement]
    overall_scope: str = Field(..., regex="^(small|medium|large|epic)$")
    technical_stack: List[str] = []
    quality_requirements: Dict[str, str] = {}
    constraints: List[str] = []
    assumptions: List[str] = []
    risks: List[str] = []

# Assignment Schemas
class AssignmentBase(BaseModel):
    task_id: int
    developer_id: int
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    reasoning: str

class AssignmentCreate(AssignmentBase):
    pass

class AssignmentUpdate(BaseModel):
    status: Optional[str] = Field(None, regex="^(suggested|accepted|rejected|completed)$")
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    actual_hours: Optional[float] = None
    feedback_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    feedback_comments: Optional[str] = None

class Assignment(AssignmentBase):
    id: int
    status: str = "suggested"
    assigned_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    actual_hours: Optional[float] = None
    feedback_score: Optional[float] = None
    feedback_comments: Optional[str] = None
    productivity_score: Optional[float] = None
    skill_development_score: Optional[float] = None
    collaboration_effectiveness: Optional[float] = None
    
    class Config:
        from_attributes = True

# Analysis Request/Response Schemas
class ComplexityAnalysisRequest(BaseModel):
    """Request for task complexity analysis."""
    title: str
    description: str = ""
    repository: Optional[str] = None
    labels: List[str] = []
    github_issue_data: Optional[Dict] = None

class ComplexityAnalysisResponse(BaseModel):
    """Response from complexity analysis."""
    task_id: Optional[str] = None
    complexity_result: TaskComplexityResult
    requirements_analysis: Optional[TaskRequirementsAnalysis] = None
    analysis_metadata: Dict[str, any] = {}
    processing_time_ms: float

class BatchComplexityRequest(BaseModel):
    """Request for batch complexity analysis."""
    tasks: List[ComplexityAnalysisRequest]
    include_requirements: bool = True

class BatchComplexityResponse(BaseModel):
    """Response from batch complexity analysis."""
    results: List[ComplexityAnalysisResponse]
    summary: Dict[str, any] = {}
    total_processing_time_ms: float

# Task Matching Schemas
class TaskDeveloperMatch(BaseModel):
    """Represents a task-developer match with scoring."""
    task_id: int
    developer_id: int
    overall_score: float = Field(..., ge=0.0, le=1.0)
    skill_match_score: float = Field(..., ge=0.0, le=1.0)
    complexity_fit_score: float = Field(..., ge=0.0, le=1.0)
    learning_potential_score: float = Field(..., ge=0.0, le=1.0)
    workload_impact_score: float = Field(..., ge=0.0, le=1.0)
    reasoning: str
    risk_factors: List[str] = []

class TaskMatchingRequest(BaseModel):
   """Request for task-developer matching."""
   task_ids: List[int]
   developer_ids: Optional[List[int]] = None  # If None, consider all developers
   matching_criteria: Dict[str, float] = {
       "skill_match": 0.4,
       "complexity_fit": 0.3,
       "learning_potential": 0.2,
       "workload_balance": 0.1
   }
   max_matches_per_task: int = Field(default=3, ge=1, le=10)

class TaskMatchingResponse(BaseModel):
   """Response from task-developer matching."""
   matches: List[TaskDeveloperMatch]
   matching_metadata: Dict[str, any] = {}
   processing_time_ms: float

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

# Skill Analysis Schemas (existing - keeping for compatibility)
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

# Team Analysis Schemas (existing)
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

# GitHub Integration Schemas (existing)
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

# Analytics Schemas (existing)
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

# API Response Schemas (existing)
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

# NEW Phase 2 Validation Schemas
class RequirementValidation(BaseModel):
   """Validation results for requirements analysis."""
   errors: List[str] = []
   warnings: List[str] = []
   suggestions: List[str] = []
   completeness_score: float = Field(..., ge=0.0, le=1.0)
   clarity_score: float = Field(..., ge=0.0, le=1.0)

class TaskComplexityValidation(BaseModel):
   """Validation results for complexity analysis."""
   is_valid: bool
   confidence_level: str = Field(..., regex="^(low|medium|high)$")
   validation_issues: List[str] = []
   recommendations: List[str] = []

# Performance Metrics Schemas
class AnalysisPerformanceMetrics(BaseModel):
   """Performance metrics for analysis operations."""
   operation_type: str
   processing_time_ms: float
   memory_usage_mb: Optional[float] = None
   items_processed: int
   success_rate: float = Field(..., ge=0.0, le=1.0)
   error_count: int = 0
   timestamp: datetime = datetime.utcnow()

class SystemHealthMetrics(BaseModel):
   """Overall system health metrics."""
   cpu_usage_percent: float
   memory_usage_percent: float
   database_connections: int
   active_analyses: int
   avg_response_time_ms: float
   error_rate_last_hour: float
   timestamp: datetime = datetime.utcnow()

# Assignment Optimization Schemas
class OptimizationObjective(str, Enum):
    PRODUCTIVITY = "productivity"
    SKILL_DEVELOPMENT = "skill_development"
    WORKLOAD_BALANCE = "workload_balance"
    COLLABORATION = "collaboration"
    BUSINESS_IMPACT = "business_impact"

class OptimizationConstraint(BaseModel):
    """Optimization constraint specification."""
    constraint_type: str = Field(..., regex="^(max_workload|skill_requirement|availability|deadline)$")
    value: float
    applies_to: Optional[List[int]] = None  # Developer or task IDs

class OptimizationRequest(BaseModel):
    """Request for assignment optimization."""
    task_ids: List[int] = Field(..., min_items=1)
    developer_ids: List[int] = Field(..., min_items=1)
    objectives: List[OptimizationObjective] = [
        OptimizationObjective.PRODUCTIVITY,
        OptimizationObjective.SKILL_DEVELOPMENT,
        OptimizationObjective.WORKLOAD_BALANCE
    ]
    constraints: List[OptimizationConstraint] = []
    max_assignments_per_developer: int = Field(default=3, ge=1, le=10)

class AssignmentSuggestion(BaseModel):
    """Individual assignment suggestion with detailed scoring."""
    task_id: int
    developer_id: int
    overall_score: float = Field(..., ge=0.0, le=1.0)
    skill_match_score: float = Field(..., ge=0.0, le=1.0)
    complexity_fit_score: float = Field(..., ge=0.0, le=1.0)
    learning_potential_score: float = Field(..., ge=0.0, le=1.0)
    workload_impact_score: float = Field(..., ge=0.0, le=1.0)
    collaboration_score: float = Field(..., ge=0.0, le=1.0)
    business_impact_score: float = Field(..., ge=0.0, le=1.0)
    confidence: float = Field(..., ge=0.0, le=1.0)
    reasoning: str
    risk_factors: List[str] = []
    predicted_performance: Optional[float] = None
    prediction_confidence: Optional[float] = None

class OptimizationResult(BaseModel):
    """Result from assignment optimization."""
    assignments: List[Assignment]
    objective_scores: Dict[str, float]
    alternative_solutions: List[Dict[str, Any]] = []
    optimization_metadata: Dict[str, Any] = {}
    optimization_time_ms: float
    total_score: float = 0.0
    confidence_level: str = Field(default="medium", regex="^(low|medium|high)$")

# Learning System Schemas
class LearningFeedback(BaseModel):
    """Feedback for learning system."""
    assignment_id: int
    outcome_metrics: Dict[str, float]
    qualitative_feedback: Optional[str] = None
    improvement_suggestions: List[str] = []

class DeveloperPreferenceModel(BaseModel):
    """Developer preference model for learning system."""
    developer_id: int
    preferred_complexity_range: Tuple[float, float]
    collaboration_preference: float = Field(..., ge=0.0, le=1.0)
    learning_preference: float = Field(..., ge=0.0, le=1.0)
    workload_tolerance: float = Field(..., ge=0.0, le=1.0)
    performance_history: List[Dict[str, Any]] = []

class LearningAnalytics(BaseModel):
    """Analytics from the learning system."""
    total_assignments_learned: int
    current_learning_rate: float
    prediction_accuracy: float = Field(..., ge=0.0, le=1.0)
    developers_modeled: int
    skills_tracked: int
    top_important_skills: List[Tuple[str, float]] = []
    recent_performance_trend: str = Field(..., regex="^(improving|stable|declining|unknown)$")

# Team Analytics Schemas
class TeamPerformanceMetrics(BaseModel):
    """Team-level performance metrics."""
    team_size: int
    avg_assignment_score: float = Field(..., ge=0.0, le=1.0)
    skill_development_rate: float = Field(..., ge=0.0, le=1.0)
    collaboration_effectiveness: float = Field(..., ge=0.0, le=1.0)
    workload_balance_score: float = Field(..., ge=0.0, le=1.0)
    completion_rate: float = Field(..., ge=0.0, le=1.0)
    average_delivery_time_hours: Optional[float] = None

class AssignmentDistribution(BaseModel):
    """Assignment distribution across team members."""
    developer_assignments: Dict[str, int]
    balance_score: float = Field(..., ge=0.0, le=1.0)
    overloaded_developers: List[int] = []
    underutilized_developers: List[int] = []

class TeamSkillGapAnalysis(BaseModel):
    """Team skill gap analysis."""
    required_skills: Dict[str, float]
    available_skills: Dict[str, float]
    skill_gaps: Dict[str, float]
    skill_redundancies: Dict[str, int]
    recommendations: List[str] = []

# Advanced Analytics Schemas
class AssignmentOutcomeAnalysis(BaseModel):
    """Detailed analysis of assignment outcomes."""
    assignment_id: int
    predicted_vs_actual_performance: Dict[str, float]
    time_estimation_accuracy: float
    skill_development_achieved: float = Field(..., ge=0.0, le=1.0)
    learning_impact_areas: List[str] = []
    success_factors: List[str] = []
    improvement_areas: List[str] = []

class PredictiveInsights(BaseModel):
    """Predictive insights for future assignments."""
    developer_id: int
    predicted_performance_trends: Dict[str, float]
    recommended_growth_areas: List[str] = []
    optimal_task_characteristics: Dict[str, Any]
    risk_indicators: List[str] = []
    confidence_in_predictions: float = Field(..., ge=0.0, le=1.0)

# Batch Operation Schemas
class BatchAssignmentRequest(BaseModel):
    """Request for batch assignment operations."""
    assignments: List[AssignmentCreate]
    optimization_mode: bool = True
    conflict_resolution: str = Field(default="optimize", regex="^(optimize|reject|override)$")

class BatchAssignmentResponse(BaseModel):
    """Response from batch assignment operations."""
    successful_assignments: List[Assignment]
    failed_assignments: List[Dict[str, Any]]
    conflicts_resolved: int
    optimization_applied: bool
    processing_time_ms: float

# Real-time Update Schemas  
class AssignmentUpdate(BaseModel):
    """Real-time assignment update."""
    assignment_id: int
    update_type: str = Field(..., regex="^(status_change|progress_update|completion|feedback)$")
    update_data: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.now)

class AssignmentNotification(BaseModel):
    """Assignment notification for real-time updates."""
    notification_type: str = Field(..., regex="^(new_assignment|assignment_update|deadline_alert|performance_alert)$")
    assignment_id: int
    developer_id: int
    message: str
    priority: str = Field(default="medium", regex="^(low|medium|high|urgent)$")
    action_required: bool = False
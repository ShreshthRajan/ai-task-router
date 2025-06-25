# src/models/schemas.py
from pydantic import BaseModel, EmailStr, Field, ConfigDict
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from enum import Enum

# Fix Pydantic warnings about model namespace
import warnings
warnings.filterwarnings("ignore", message="Field .* has conflict with protected namespace")

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
    
    model_config = ConfigDict(from_attributes=True)

# NEW PHASE 2 SCHEMAS

# Task Schemas
class TaskBase(BaseModel):
    title: str
    description: Optional[str] = ""
    repository: Optional[str] = None
    labels: List[str] = []
    priority: str = Field(default="medium", pattern="^(low|medium|high|critical)$")

class TaskCreate(TaskBase):
    github_issue_id: Optional[int] = None

class TaskUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    labels: Optional[List[str]] = None
    priority: Optional[str] = Field(None, pattern="^(low|medium|high|critical)$")
    status: Optional[str] = Field(None, pattern="^(open|in_progress|completed|closed)$")

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
    
    model_config = ConfigDict(from_attributes=True)

# Requirement Schemas
class RequirementBase(BaseModel):
    category: str = Field(..., pattern="^(functional|non_functional|technical)$")
    priority: str = Field(..., pattern="^(must|should|could|wont)$")
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
    
    model_config = ConfigDict(from_attributes=True)


class TaskRequirement(BaseModel):
    """Task requirement response schema."""
    id: int
    task_id: int
    requirement_id: str
    category: str = Field(..., pattern="^(functional|non_functional|technical)$")
    priority: str = Field(..., pattern="^(must|should|could|wont)$")
    description: str
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
    overall_scope: str = Field(..., pattern="^(small|medium|large|epic)$")
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
    status: Optional[str] = Field(None, pattern="^(suggested|accepted|rejected|completed)$")
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
    
    model_config = ConfigDict(from_attributes=True)

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
    analysis_metadata: Dict[str, Any] = {}
    processing_time_ms: float

class BatchComplexityRequest(BaseModel):
    """Request for batch complexity analysis."""
    tasks: List[ComplexityAnalysisRequest]
    include_requirements: bool = True

class BatchComplexityResponse(BaseModel):
    """Response from batch complexity analysis."""
    results: List[ComplexityAnalysisResponse]
    summary: Dict[str, Any] = {}
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
   matching_metadata: Dict[str, Any] = {}
   processing_time_ms: float

# Optimization Schemas
class OptimizationRequest(BaseModel):
   task_ids: List[int]
   developer_ids: List[int]
   objectives: List[str] = ["productivity", "skill_development", "workload_balance"]
   constraints: Dict[str, float] = {}

class AssignmentResult(BaseModel):
    """Assignment result from optimization (without database fields)."""
    task_id: int
    developer_id: int
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    reasoning: str
    status: str = "suggested"

class OptimizationResult(BaseModel):
    """Result from assignment optimization."""
    assignments: List[AssignmentResult]
    objective_scores: Dict[str, float]
    alternative_solutions: List[Dict[str, Any]] = []
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
   confidence_level: str = Field(..., pattern="^(low|medium|high)$")
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
    constraint_type: str = Field(..., pattern="^(max_workload|skill_requirement|availability|deadline)$")
    value: float
    applies_to: Optional[List[int]] = None  # Developer or task IDs

class OptimizationRequest(BaseModel):
    """Request for assignment optimization."""
    task_ids: List[int] = Field(..., min_length=1)
    developer_ids: List[int] = Field(..., min_length=1)
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
    recent_performance_trend: str = Field(..., pattern="^(improving|stable|declining|unknown)$")

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
    conflict_resolution: str = Field(default="optimize", pattern="^(optimize|reject|override)$")

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
    update_type: str = Field(..., pattern="^(status_change|progress_update|completion|feedback)$")
    update_data: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.now)

class AssignmentNotification(BaseModel):
    """Assignment notification for real-time updates."""
    notification_type: str = Field(..., pattern="^(new_assignment|assignment_update|deadline_alert|performance_alert)$")
    assignment_id: int
    developer_id: int
    message: str
    priority: str = Field(default="medium", pattern="^(low|medium|high|urgent)$")
    action_required: bool = False

class AssignmentOutcomeCreate(BaseModel):
    """Create assignment outcome record."""
    assignment_id: int
    task_completion_quality: float = Field(..., ge=0.0, le=1.0)
    developer_satisfaction: float = Field(..., ge=0.0, le=1.0)
    learning_achieved: float = Field(..., ge=0.0, le=1.0)
    collaboration_effectiveness: float = Field(..., ge=0.0, le=1.0)
    time_estimation_accuracy: float = Field(..., ge=0.0, le=1.0)
    performance_metrics: Dict[str, float] = {}
    skill_improvements: List[str] = []
    challenges_faced: List[str] = []
    success_factors: List[str] = []

class AssignmentOutcome(AssignmentOutcomeCreate):
    """Assignment outcome response."""
    id: int
    prediction_accuracy: Dict[str, float] = {}
    unexpected_learnings: List[str] = []
    improvement_suggestions: List[str] = []
    created_at: datetime
    
    model_config = ConfigDict(from_attributes=True)

class ModelPerformanceMetrics(BaseModel):
    """Model performance tracking."""
    model_name: str
    version: str
    accuracy_score: float = Field(..., ge=0.0, le=1.0)
    precision_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    recall_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    f1_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    mse_score: Optional[float] = None
    training_data_size: int
    prediction_count: int = 0
    correct_predictions: int = 0
    average_confidence: float = Field(..., ge=0.0, le=1.0)

class SystemHealthMetrics(BaseModel):
    """System-wide health and performance metrics."""
    avg_assignment_quality: float = Field(..., ge=0.0, le=1.0)
    avg_developer_satisfaction: float = Field(..., ge=0.0, le=1.0)
    avg_skill_development_rate: float = Field(..., ge=0.0, le=1.0)
    assignment_success_rate: float = Field(..., ge=0.0, le=1.0)
    team_productivity_score: float = Field(..., ge=0.0, le=1.0)
    learning_accuracy_trend: float
    prediction_confidence_avg: float = Field(..., ge=0.0, le=1.0)
    total_assignments: int
    completed_assignments: int

class LearningExperimentCreate(BaseModel):
    """Create learning experiment."""
    experiment_name: str
    experiment_type: str = Field(..., pattern="^(ab_test|parameter_optimization|model_comparison)$")
    control_config: Dict[str, Any]
    experimental_config: Dict[str, Any]
    success_metrics: List[str]
    sample_size: int = Field(..., gt=0)

class LearningExperiment(LearningExperimentCreate):
    """Learning experiment response."""
    id: int
    status: str = "active"
    control_results: Dict[str, float] = {}
    experimental_results: Dict[str, float] = {}
    statistical_significance: Optional[float] = None
    confidence_level: Optional[float] = None
    start_date: datetime
    end_date: Optional[datetime] = None
    created_at: datetime
    
    model_config = ConfigDict(from_attributes=True)

class DeveloperPreferenceProfile(BaseModel):
    """Developer preference profile."""
    developer_id: int
    preferred_complexity_range: Tuple[float, float]
    collaboration_preference: float = Field(..., ge=0.0, le=1.0)
    learning_appetite: float = Field(..., ge=0.0, le=1.0)
    optimal_workload_hours: float = Field(..., gt=0)
    workload_tolerance: float = Field(..., ge=0.0, le=1.0)
    preferred_learning_style: str = Field(..., pattern="^(gradual|immersive|mixed)$")
    preference_confidence: float = Field(..., ge=0.0, le=1.0)
    sample_size: int

class SkillImportanceAnalysis(BaseModel):
    """Skill importance analysis results."""
    skill_name: str
    task_type: str
    complexity_range: str
    domain: str
    importance_factor: float = Field(..., ge=0.0, le=2.0)
    confidence: float = Field(..., ge=0.0, le=1.0)
    success_rate_with_skill: float = Field(..., ge=0.0, le=1.0)
    success_rate_without_skill: float = Field(..., ge=0.0, le=1.0)
    sample_size: int

class LearningSystemAnalytics(BaseModel):
    """Comprehensive learning system analytics."""
    total_outcomes_processed: int
    active_experiments: int
    model_performance_trends: Dict[str, List[float]]
    developer_preferences_learned: int
    skill_importance_factors: int
    system_improvement_rate: float
    prediction_accuracy_improvement: float
    recent_learnings: List[str]

class FeedbackProcessingResult(BaseModel):
    """Result from processing assignment feedback."""
    outcomes_processed: int
    models_updated: int
    preferences_learned: int
    skill_factors_updated: int
    system_improvements: List[str]
    processing_time_ms: float

class ModelUpdateResult(BaseModel):
    """Result from model update operation."""
    model_name: str
    previous_version: str
    new_version: str
    performance_improvement: float
    update_type: str = Field(..., pattern="^(retrain|parameter_tune|architecture_change)$")
    validation_results: Dict[str, float]
    rollback_available: bool

class PredictiveInsights(BaseModel):
    """Predictive insights for team and individual performance."""
    developer_id: Optional[int] = None
    predicted_performance_trend: str = Field(..., pattern="^(improving|stable|declining)$")
    skill_development_forecast: Dict[str, float]
    optimal_assignment_characteristics: Dict[str, Any]
    risk_factors: List[str]
    recommendations: List[str]
    confidence: float = Field(..., ge=0.0, le=1.0)

class SystemOptimizationSuggestion(BaseModel):
    """System optimization suggestions."""
    optimization_type: str = Field(..., pattern="^(algorithm_tuning|workflow_improvement|resource_allocation)$")
    current_performance: float
    expected_improvement: float
    implementation_effort: str = Field(..., pattern="^(low|medium|high)$")
    impact_areas: List[str]
    description: str
    confidence: float = Field(..., ge=0.0, le=1.0)

class LearningProgress(BaseModel):
    """Learning progress tracking."""
    learning_component: str
    current_accuracy: float = Field(..., ge=0.0, le=1.0)
    accuracy_trend: List[float]
    data_points_processed: int
    last_significant_improvement: Optional[datetime] = None
    learning_rate: float = Field(..., ge=0.0, le=1.0)
    convergence_status: str = Field(..., pattern="^(converging|converged|diverging|unstable)$")
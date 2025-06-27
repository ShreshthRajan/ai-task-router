# src/api/learning.py
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta

from models.database import get_db
from models.schemas import (
    AssignmentOutcomeCreate, AssignmentOutcome, SystemHealthMetrics,
    LearningSystemAnalytics, PredictiveInsights, SystemOptimizationSuggestion,
    LearningProgress, FeedbackProcessingResult, ModelUpdateResult,
    LearningExperimentCreate, LearningExperiment, TeamPerformanceMetrics,
    DeveloperPreferenceProfile, SkillImportanceAnalysis
)
from core.learning_system.feedback_processor import FeedbackProcessor
from core.learning_system.model_updater import ModelUpdater
from core.learning_system.system_analytics import SystemAnalytics

router = APIRouter()

# Initialize learning system components
feedback_processor = FeedbackProcessor()
model_updater = ModelUpdater()
system_analytics = SystemAnalytics()

@router.post("/learning/feedback", response_model=FeedbackProcessingResult)
async def submit_assignment_feedback(
    outcomes: List[AssignmentOutcomeCreate],
    update_models: bool = Query(True, description="Whether to update ML models"),
    db: Session = Depends(get_db)
):
    """
    Submit assignment outcome feedback for learning system processing.
    
    This endpoint processes assignment outcomes to:
    - Learn developer preferences
    - Update skill importance factors
    - Improve model performance
    - Generate system insights
    """
    try:
        result = await feedback_processor.process_assignment_outcomes(
            db, outcomes, update_models
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing feedback: {str(e)}")

@router.get("/learning/system-health")
async def get_system_health(db: Session = Depends(get_db)):
    """Get comprehensive system health metrics."""
    try:
        metrics = await system_analytics.get_system_health_metrics(db)
        analytics = await system_analytics.get_learning_analytics_for_frontend(db)
        return {
          "system_metrics": {
            "active_analyses": metrics.active_analyses,
            "avg_response_time_ms": metrics.avg_response_time_ms,
            "uptime_hours": metrics.uptime_hours
          },
          "model_performance": {
            "assignment_accuracy": analytics.model_performance.assignment_accuracy,
            "prediction_confidence": analytics.model_performance.prediction_confidence,
            "learning_rate": analytics.model_performance.learning_rate,
            "improvement_trend": analytics.model_performance.improvement_trend
          },
          "productivity_metrics": {
            "cost_savings_monthly": analytics.productivity_metrics.cost_savings_monthly,
            "developer_satisfaction_score": analytics.productivity_metrics.developer_satisfaction_score,
            "time_saved_hours": analytics.productivity_metrics.time_saved_hours,
            "avg_task_completion_improvement": analytics.productivity_metrics.avg_task_completion_improvement
          },
          "recent_optimizations": analytics.recent_optimizations
        }
    except Exception as e:
        raise HTTPException(500, detail=str(e))

# REPLACE the entire function with:
@router.get("/learning/analytics")
async def get_learning_analytics(db: Session = Depends(get_db)):
    """Get comprehensive learning system analytics."""
    try:
        return await system_analytics.get_learning_analytics_for_frontend(db)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting learning analytics: {str(e)}")
    
@router.get("/learning/insights", response_model=List[PredictiveInsights])
async def get_predictive_insights(
    developer_id: Optional[int] = Query(None, description="Specific developer ID"),
    db: Session = Depends(get_db)
):
    """Generate predictive insights for team or individual performance."""
    try:
        return await system_analytics.generate_predictive_insights(db, developer_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating insights: {str(e)}")

@router.get("/learning/optimization-suggestions", response_model=List[SystemOptimizationSuggestion])
async def get_optimization_suggestions(db: Session = Depends(get_db)):
    """Get system optimization suggestions based on performance analysis."""
    try:
        return await system_analytics.generate_optimization_suggestions(db)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating suggestions: {str(e)}")

@router.get("/learning/progress", response_model=List[LearningProgress])
async def get_learning_progress(db: Session = Depends(get_db)):
    """Get learning progress for all system components."""
    try:
        return await system_analytics.get_learning_progress(db)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting learning progress: {str(e)}")

@router.post("/learning/update-models", response_model=List[ModelUpdateResult])
async def update_models(
    force_update: bool = Query(False, description="Force update even if performance hasn't improved"),
    db: Session = Depends(get_db)
):
    """Update ML models based on recent assignment outcomes."""
    try:
        return await model_updater.update_models_from_outcomes(db, force_update)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating models: {str(e)}")

@router.post("/learning/experiments", response_model=LearningExperiment)
async def create_experiment(
    experiment: LearningExperimentCreate,
    db: Session = Depends(get_db)
):
    """Create a new learning experiment for A/B testing."""
    try:
        return await model_updater.create_learning_experiment(db, experiment)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating experiment: {str(e)}")

@router.post("/learning/experiments/{experiment_id}/start")
async def start_experiment(
    experiment_id: int,
    duration_days: int = Query(14, description="Experiment duration in days"),
    db: Session = Depends(get_db)
):
    """Start an A/B test experiment."""
    try:
        success = await model_updater.run_ab_test(db, experiment_id, duration_days)
        if success:
            return {"message": "Experiment started successfully"}
        else:
            raise HTTPException(status_code=404, detail="Experiment not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error starting experiment: {str(e)}")

@router.get("/learning/experiments/{experiment_id}/results", response_model=LearningExperiment)
async def get_experiment_results(
    experiment_id: int,
    db: Session = Depends(get_db)
):
    """Get results from a completed experiment."""
    try:
        result = await model_updater.evaluate_experiment(db, experiment_id)
        if result:
            return result
        else:
            raise HTTPException(status_code=404, detail="Experiment not found or not completed")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting experiment results: {str(e)}")

@router.get("/learning/developer-preferences/{developer_id}", response_model=DeveloperPreferenceProfile)
async def get_developer_preferences(
    developer_id: int,
    db: Session = Depends(get_db)
):
    """Get learned preferences for a specific developer."""
    try:
        return await feedback_processor.analyze_developer_performance_patterns(db, developer_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting developer preferences: {str(e)}")

@router.get("/learning/skill-importance", response_model=List[SkillImportanceAnalysis])
async def get_skill_importance(
    task_type: Optional[str] = Query(None, description="Filter by task type"),
    domain: Optional[str] = Query(None, description="Filter by domain"),
    db: Session = Depends(get_db)
):
    """Get skill importance analysis for given context."""
    try:
        return await feedback_processor.get_skill_importance_analysis(db, task_type, domain)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting skill importance: {str(e)}")

@router.get("/learning/team-performance", response_model=TeamPerformanceMetrics)
async def get_team_performance(db: Session = Depends(get_db)):
    """Get comprehensive team performance metrics."""
    try:
        return await system_analytics.get_team_performance_metrics(db)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting team performance: {str(e)}")

@router.get("/learning/alerts")
async def get_performance_alerts(db: Session = Depends(get_db)):
    """Get current performance alerts and issues."""
    try:
        return await system_analytics.detect_performance_alerts(db)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting alerts: {str(e)}")

@router.get("/learning/roi-report")
async def get_roi_report(
    period_days: int = Query(30, description="Analysis period in days"),
    db: Session = Depends(get_db)
):
    """Generate ROI report for the task routing system."""
    try:
        return await system_analytics.generate_roi_report(db, period_days)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating ROI report: {str(e)}")

@router.post("/learning/models/{model_name}/rollback")
async def rollback_model(
    model_name: str,
    target_version: Optional[str] = Query(None, description="Target version to rollback to"),
    db: Session = Depends(get_db)
):
    """Rollback a model to a previous version."""
    try:
        success = await model_updater.rollback_model(db, model_name, target_version)
        if success:
            return {"message": f"Model {model_name} rolled back successfully"}
        else:
            raise HTTPException(status_code=400, detail="Rollback failed")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error rolling back model: {str(e)}")

@router.get("/learning/models/{model_name}/performance-history")
async def get_model_performance_history(
    model_name: str,
    days: int = Query(30, description="Number of days to analyze"),
    db: Session = Depends(get_db)
):
    """Get performance history for a specific model."""
    try:
        return await model_updater.get_model_performance_history(db, model_name, days)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting model history: {str(e)}")

@router.post("/learning/optimize-hyperparameters")
async def optimize_hyperparameters(
    model_name: str,
    parameter_space: Dict[str, List[Any]],
    db: Session = Depends(get_db)
):
    """Optimize hyperparameters for a specific model."""
    try:
        result = await model_updater.optimize_hyperparameters(db, model_name, parameter_space)
        if result:
            return result
        else:
            return {"message": "No improvement found with current parameter space"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error optimizing hyperparameters: {str(e)}")
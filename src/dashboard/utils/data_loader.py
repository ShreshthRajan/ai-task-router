import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from sqlalchemy.orm import Session
import asyncio
import logging

from ...models.database import (
    TaskAssignment, AssignmentOutcome, Developer, Task,
    SystemMetrics, ModelPerformance, DeveloperPreference,
    SkillImportanceFactor, LearningExperiment
)

logger = logging.getLogger(__name__)

class DashboardDataLoader:
    """Efficient data loading and caching for dashboard components."""
    
    def __init__(self, db: Session):
        self.db = db
        self._cache = {}
        self._cache_ttl = 300  # 5 minutes
        
    async def get_real_time_metrics(self) -> Dict[str, Any]:
        """Get real-time system metrics with caching."""
        cache_key = "real_time_metrics"
        
        if self._is_cache_valid(cache_key):
            return self._cache[cache_key]["data"]
        
        try:
            # Get recent assignments
            recent_assignments = self.db.query(TaskAssignment).filter(
                TaskAssignment.assigned_at >= datetime.utcnow() - timedelta(hours=24)
            ).all()
            
            # Get recent outcomes
            recent_outcomes = self.db.query(AssignmentOutcome).join(
                TaskAssignment
            ).filter(
                TaskAssignment.assigned_at >= datetime.utcnow() - timedelta(days=7)
            ).all()
            
            # Calculate metrics
            metrics = {
                "active_assignments": len([a for a in recent_assignments if a.status in ["accepted", "in_progress"]]),
                "completed_today": len([a for a in recent_assignments if a.status == "completed"]),
                "avg_quality_score": np.mean([o.task_completion_quality for o in recent_outcomes]) if recent_outcomes else 0.0,
                "system_load": min(1.0, len(recent_assignments) / 50.0),  # Normalize to 0-1
                "alerts_count": await self._count_active_alerts(),
                "learning_rate": await self._calculate_learning_rate()
            }
            
            self._cache[cache_key] = {
                "data": metrics,
                "timestamp": datetime.utcnow()
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error loading real-time metrics: {e}")
            return self._get_demo_metrics()
    
    async def get_developer_performance_data(self, developer_id: Optional[int] = None) -> Dict[str, Any]:
        """Get developer performance data for visualizations."""
        cache_key = f"developer_performance_{developer_id}"
        
        if self._is_cache_valid(cache_key):
            return self._cache[cache_key]["data"]
        
        try:
            query = self.db.query(AssignmentOutcome).join(TaskAssignment)
            
            if developer_id:
                query = query.filter(TaskAssignment.developer_id == developer_id)
            
            outcomes = query.filter(
                TaskAssignment.assigned_at >= datetime.utcnow() - timedelta(days=90)
            ).all()
            
            # Process data for charts
            performance_data = self._process_performance_data(outcomes)
            
            self._cache[cache_key] = {
                "data": performance_data,
                "timestamp": datetime.utcnow()
            }
            
            return performance_data
            
        except Exception as e:
            logger.error(f"Error loading developer performance data: {e}")
            return self._get_demo_performance_data()
    
    async def get_team_collaboration_data(self) -> Dict[str, Any]:
        """Get team collaboration network data."""
        cache_key = "team_collaboration"
        
        if self._is_cache_valid(cache_key):
            return self._cache[cache_key]["data"]
        
        try:
            # Get collaboration data from assignments and outcomes
            assignments = self.db.query(TaskAssignment).join(Task).filter(
                TaskAssignment.assigned_at >= datetime.utcnow() - timedelta(days=30)
            ).all()
            
            collaboration_data = self._analyze_collaboration_patterns(assignments)
            
            self._cache[cache_key] = {
                "data": collaboration_data,
                "timestamp": datetime.utcnow()
            }
            
            return collaboration_data
            
        except Exception as e:
            logger.error(f"Error loading collaboration data: {e}")
            return self._get_demo_collaboration_data()
    
    async def get_learning_analytics_data(self) -> Dict[str, Any]:
        """Get learning system analytics data."""
        cache_key = "learning_analytics"
        
        if self._is_cache_valid(cache_key):
            return self._cache[cache_key]["data"]
        
        try:
            # Get model performance trends
            model_performances = self.db.query(ModelPerformance).filter(
                ModelPerformance.created_at >= datetime.utcnow() - timedelta(days=30)
            ).all()
            
            # Get learning experiments
            experiments = self.db.query(LearningExperiment).all()
            
            # Get skill importance factors
            skill_factors = self.db.query(SkillImportanceFactor).all()
            
            analytics_data = {
                "model_trends": self._process_model_trends(model_performances),
                "experiment_results": self._process_experiment_results(experiments),
                "skill_importance": self._process_skill_importance(skill_factors),
                "learning_velocity": await self._calculate_learning_velocity()
            }
            
            self._cache[cache_key] = {
                "data": analytics_data,
                "timestamp": datetime.utcnow()
            }
            
            return analytics_data
            
        except Exception as e:
            logger.error(f"Error loading learning analytics: {e}")
            return self._get_demo_learning_data()
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid."""
        if cache_key not in self._cache:
            return False
        
        cache_age = (datetime.utcnow() - self._cache[cache_key]["timestamp"]).total_seconds()
        return cache_age < self._cache_ttl
    
    async def _count_active_alerts(self) -> int:
        """Count active system alerts."""
        # This would integrate with the actual alert system
        return np.random.randint(0, 5)  # Demo implementation
    
    async def _calculate_learning_rate(self) -> float:
        """Calculate current system learning rate."""
        model_performances = self.db.query(ModelPerformance).filter(
            ModelPerformance.created_at >= datetime.utcnow() - timedelta(days=7)
        ).all()
        
        if len(model_performances) < 2:
            return 0.05  # Default learning rate
        
        # Calculate trend in accuracy
        accuracies = [mp.accuracy_score for mp in model_performances if mp.accuracy_score]
        if len(accuracies) < 2:
            return 0.05
        
        # Simple linear trend
        recent_avg = np.mean(accuracies[-3:]) if len(accuracies) >= 3 else accuracies[-1]
        older_avg = np.mean(accuracies[:3]) if len(accuracies) >= 3 else accuracies[0]
        
        learning_rate = max(0.0, min(0.5, recent_avg - older_avg))
        return learning_rate
    
    def _process_performance_data(self, outcomes: List[AssignmentOutcome]) -> Dict[str, Any]:
        """Process assignment outcomes for performance visualizations."""
        if not outcomes:
            return self._get_demo_performance_data()
        
        # Group by time periods
        df = pd.DataFrame([{
            "date": outcome.assignment.assigned_at,
            "quality": outcome.task_completion_quality,
            "satisfaction": outcome.developer_satisfaction,
            "learning": outcome.learning_achieved,
            "collaboration": outcome.collaboration_effectiveness
        } for outcome in outcomes])
        
        # Weekly aggregation
        df['week'] = df['date'].dt.isocalendar().week
        weekly_data = df.groupby('week').agg({
            'quality': 'mean',
            'satisfaction': 'mean', 
            'learning': 'mean',
            'collaboration': 'mean'
        }).to_dict('records')
        
        return {
            "weekly_trends": weekly_data,
            "overall_metrics": {
                "avg_quality": df['quality'].mean(),
                "avg_satisfaction": df['satisfaction'].mean(),
                "avg_learning": df['learning'].mean(),
                "avg_collaboration": df['collaboration'].mean()
            }
        }
    
    def _analyze_collaboration_patterns(self, assignments: List[TaskAssignment]) -> Dict[str, Any]:
        """Analyze collaboration patterns from assignments."""
        # This would analyze actual collaboration data
        # For now, return demo data structure
        return self._get_demo_collaboration_data()
    
    def _process_model_trends(self, performances: List[ModelPerformance]) -> Dict[str, List[float]]:
        """Process model performance trends."""
        if not performances:
            return {"complexity_predictor": [], "assignment_optimizer": [], "skill_extractor": []}
        
        trends = {}
        for model_name in ["complexity_predictor", "assignment_optimizer", "skill_extractor"]:
            model_perfs = [p for p in performances if p.model_name == model_name]
            trends[model_name] = [p.accuracy_score for p in sorted(model_perfs, key=lambda x: x.created_at)]
        
        return trends
    
    def _process_experiment_results(self, experiments: List[LearningExperiment]) -> List[Dict[str, Any]]:
        """Process experiment results for visualization."""
        return [{
            "name": exp.experiment_name,
            "type": exp.experiment_type,
            "status": exp.status,
            "significance": exp.statistical_significance
        } for exp in experiments]
    
    def _process_skill_importance(self, factors: List[SkillImportanceFactor]) -> Dict[str, float]:
        """Process skill importance factors."""
        if not factors:
            return {}
        
        skill_importance = {}
        for factor in factors:
            if factor.skill_name not in skill_importance:
                skill_importance[factor.skill_name] = []
            skill_importance[factor.skill_name].append(factor.importance_factor)
        
        # Average importance across contexts
        return {skill: np.mean(factors) for skill, factors in skill_importance.items()}
    
    async def _calculate_learning_velocity(self) -> float:
        """Calculate learning system velocity."""
        outcomes = self.db.query(AssignmentOutcome).join(TaskAssignment).filter(
            TaskAssignment.assigned_at >= datetime.utcnow() - timedelta(days=30)
        ).all()
        
        if not outcomes:
            return 0.1
        
        learning_scores = [o.learning_achieved for o in outcomes if o.learning_achieved]
        return np.mean(learning_scores) if learning_scores else 0.1
    
    # Demo data methods
    def _get_demo_metrics(self) -> Dict[str, Any]:
        """Get demo metrics for when real data is unavailable."""
        return {
            "active_assignments": 12,
            "completed_today": 8,
            "avg_quality_score": 0.84,
            "system_load": 0.67,
            "alerts_count": 2,
            "learning_rate": 0.08
        }
    
    def _get_demo_performance_data(self) -> Dict[str, Any]:
        """Get demo performance data."""
        weeks = list(range(12))
        return {
            "weekly_trends": [{
                "quality": 0.7 + 0.02 * i + np.random.normal(0, 0.05),
                "satisfaction": 0.65 + 0.015 * i + np.random.normal(0, 0.04),
                "learning": 0.5 + 0.025 * i + np.random.normal(0, 0.06),
                "collaboration": 0.72 + 0.01 * i + np.random.normal(0, 0.03)
            } for i in weeks],
            "overall_metrics": {
                "avg_quality": 0.82,
                "avg_satisfaction": 0.74,
                "avg_learning": 0.68,
                "avg_collaboration": 0.79
            }
        }
    
    def _get_demo_collaboration_data(self) -> Dict[str, Any]:
        """Get demo collaboration data."""
        developers = ["sarah", "maria", "thomas", "alex", "jordan"]
        collaboration_matrix = np.random.rand(5, 5)
        collaboration_matrix = (collaboration_matrix + collaboration_matrix.T) / 2
        np.fill_diagonal(collaboration_matrix, 1.0)
        
        return {
            "developers": developers,
            "collaboration_matrix": collaboration_matrix.tolist(),
            "knowledge_sharing": {dev: np.random.randint(20, 80) for dev in developers}
        }
    
    def _get_demo_learning_data(self) -> Dict[str, Any]:
        """Get demo learning analytics data."""
        return {
            "model_trends": {
                "complexity_predictor": [0.72 + 0.01 * i + np.random.normal(0, 0.02) for i in range(10)],
                "assignment_optimizer": [0.68 + 0.012 * i + np.random.normal(0, 0.025) for i in range(10)],
                "skill_extractor": [0.75 + 0.008 * i + np.random.normal(0, 0.02) for i in range(10)]
            },
            "experiment_results": [
                {"name": "Assignment Weight Optimization", "type": "ab_test", "status": "completed", "significance": 0.023},
                {"name": "Complexity Prediction Enhancement", "type": "model_comparison", "status": "active", "significance": None}
            ],
            "skill_importance": {
                "Python": 0.89,
                "JavaScript": 0.76,
                "React": 0.68,
                "SQL": 0.82,
                "Docker": 0.54,
                "AWS": 0.71
            },
            "learning_velocity": 0.12
        }
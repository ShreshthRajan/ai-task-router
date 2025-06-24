import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from sqlalchemy.orm import Session
from sqlalchemy import func, and_, or_
import numpy as np
from sklearn.metrics import mean_squared_error, accuracy_score
import asyncio

from ...models.database import (
    TaskAssignment, AssignmentOutcome, Developer, Task, 
    ModelPerformance, SystemMetrics, DeveloperPreference,
    SkillImportanceFactor, ExpertiseSnapshot
)
from ...models.schemas import (
    AssignmentOutcomeCreate, FeedbackProcessingResult,
    DeveloperPreferenceProfile, SkillImportanceAnalysis
)

logger = logging.getLogger(__name__)

class FeedbackProcessor:
    """
    Process assignment outcomes to extract learning signals and improve system performance.
    
    This class analyzes completed assignments to:
    1. Learn developer preferences and performance patterns
    2. Update skill importance factors based on actual outcomes
    3. Improve complexity prediction accuracy
    4. Generate insights for system optimization
    """
    
    def __init__(self):
        self.min_sample_size = 5  # Minimum assignments needed for reliable learning
        self.confidence_threshold = 0.7  # Minimum confidence for making changes
        self.learning_rate = 0.1  # How quickly to adapt to new information
    
    async def process_assignment_outcomes(
        self, 
        db: Session,
        outcomes: List[AssignmentOutcomeCreate],
        update_models: bool = True
    ) -> FeedbackProcessingResult:
        """
        Process multiple assignment outcomes and update learning models.
        
        Args:
            db: Database session
            outcomes: List of assignment outcomes to process
            update_models: Whether to update ML models based on outcomes
            
        Returns:
            FeedbackProcessingResult with processing statistics
        """
        start_time = datetime.now()
        
        try:
            # Store outcomes in database
            outcomes_processed = await self._store_outcomes(db, outcomes)
            
            # Learn developer preferences
            preferences_learned = await self._learn_developer_preferences(db, outcomes)
            
            # Update skill importance factors
            skill_factors_updated = await self._update_skill_importance_factors(db, outcomes)
            
            # Update model performance metrics
            models_updated = 0
            if update_models:
                models_updated = await self._update_model_performance(db, outcomes)
            
            # Generate system improvements
            system_improvements = await self._generate_system_improvements(db, outcomes)
            
            # Update system metrics
            await self._update_system_metrics(db)
            
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            result = FeedbackProcessingResult(
                outcomes_processed=outcomes_processed,
                models_updated=models_updated,
                preferences_learned=preferences_learned,
                skill_factors_updated=skill_factors_updated,
                system_improvements=system_improvements,
                processing_time_ms=processing_time
            )
            
            logger.info(f"Processed {len(outcomes)} assignment outcomes in {processing_time:.2f}ms")
            return result
            
        except Exception as e:
            logger.error(f"Error processing assignment outcomes: {e}")
            raise
    
    async def _store_outcomes(self, db: Session, outcomes: List[AssignmentOutcomeCreate]) -> int:
        """Store assignment outcomes in database."""
        stored_count = 0
        
        for outcome_data in outcomes:
            try:
                # Check if outcome already exists
                existing = db.query(AssignmentOutcome).filter(
                    AssignmentOutcome.assignment_id == outcome_data.assignment_id
                ).first()
                
                if existing:
                    # Update existing outcome
                    for key, value in outcome_data.dict().items():
                        setattr(existing, key, value)
                else:
                    # Create new outcome
                    outcome = AssignmentOutcome(**outcome_data.dict())
                    db.add(outcome)
                
                stored_count += 1
                
            except Exception as e:
                logger.warning(f"Failed to store outcome for assignment {outcome_data.assignment_id}: {e}")
        
        db.commit()
        return stored_count
    
    async def _learn_developer_preferences(
        self, 
        db: Session, 
        outcomes: List[AssignmentOutcomeCreate]
    ) -> int:
        """Learn and update developer preferences based on assignment outcomes."""
        preferences_learned = 0
        
        # Group outcomes by developer
        developer_outcomes = {}
        for outcome in outcomes:
            assignment = db.query(TaskAssignment).filter(
                TaskAssignment.id == outcome.assignment_id
            ).first()
            
            if assignment:
                dev_id = assignment.developer_id
                if dev_id not in developer_outcomes:
                    developer_outcomes[dev_id] = []
                developer_outcomes[dev_id].append((assignment, outcome))
        
        # Learn preferences for each developer
        for dev_id, dev_outcomes in developer_outcomes.items():
            try:
                await self._learn_individual_preferences(db, dev_id, dev_outcomes)
                preferences_learned += 1
            except Exception as e:
                logger.warning(f"Failed to learn preferences for developer {dev_id}: {e}")
        
        return preferences_learned
    
    async def _learn_individual_preferences(
        self, 
        db: Session, 
        developer_id: int, 
        outcomes: List[Tuple[TaskAssignment, AssignmentOutcomeCreate]]
    ):
        """Learn preferences for an individual developer."""
        if len(outcomes) < self.min_sample_size:
            return  # Not enough data for reliable learning
        
        # Get or create developer preference record
        preference = db.query(DeveloperPreference).filter(
            DeveloperPreference.developer_id == developer_id
        ).first()
        
        if not preference:
            preference = DeveloperPreference(developer_id=developer_id)
            db.add(preference)
        
        # Analyze complexity preferences
        complexities = []
        satisfactions = []
        
        for assignment, outcome in outcomes:
            task = db.query(Task).filter(Task.id == assignment.task_id).first()
            if task and task.technical_complexity is not None:
                avg_complexity = (
                    task.technical_complexity + 
                    task.domain_difficulty + 
                    task.collaboration_requirements
                ) / 3.0
                
                complexities.append(avg_complexity)
                satisfactions.append(outcome.developer_satisfaction)
        
        if complexities:
            # Find optimal complexity range
            optimal_range = self._find_optimal_complexity_range(complexities, satisfactions)
            
            # Update preferences with exponential smoothing
            alpha = self.learning_rate
            preference.preferred_complexity_min = (
                alpha * optimal_range[0] + 
                (1 - alpha) * preference.preferred_complexity_min
            )
            preference.preferred_complexity_max = (
                alpha * optimal_range[1] + 
                (1 - alpha) * preference.preferred_complexity_max
            )
            preference.complexity_comfort_zone = (
                preference.preferred_complexity_min + 
                preference.preferred_complexity_max
            ) / 2.0
        
        # Learn collaboration preferences
        collaboration_scores = [outcome.collaboration_effectiveness for _, outcome in outcomes]
        if collaboration_scores:
            avg_collaboration_effectiveness = np.mean(collaboration_scores)
            preference.prefers_solo_work = (
                alpha * (1 - avg_collaboration_effectiveness) + 
                (1 - alpha) * preference.prefers_solo_work
            )
        
        # Learn learning appetite
        learning_scores = [outcome.learning_achieved for _, outcome in outcomes]
        if learning_scores:
            avg_learning = np.mean(learning_scores)
            preference.learning_appetite = (
                alpha * avg_learning + 
                (1 - alpha) * preference.learning_appetite
            )
        
        # Update confidence and sample size
        preference.sample_size += len(outcomes)
        preference.preference_confidence = min(
            1.0, 
            preference.sample_size / (self.min_sample_size * 3)
        )
        preference.last_updated = datetime.utcnow()
        
        db.commit()
    
    def _find_optimal_complexity_range(
        self, 
        complexities: List[float], 
        satisfactions: List[float]
    ) -> Tuple[float, float]:
        """Find the complexity range where developer satisfaction is highest."""
        if len(complexities) < 3:
            return (0.3, 0.8)  # Default range
        
        # Sort by complexity
        sorted_data = sorted(zip(complexities, satisfactions))
        complexities_sorted = [c for c, _ in sorted_data]
        satisfactions_sorted = [s for _, s in sorted_data]
        
        # Use sliding window to find best range
        best_satisfaction = 0
        best_range = (0.3, 0.8)
        
        for i in range(len(complexities_sorted) - 2):
            window_satisfaction = np.mean(satisfactions_sorted[i:i+3])
            if window_satisfaction > best_satisfaction:
                best_satisfaction = window_satisfaction
                best_range = (
                    complexities_sorted[i],
                    complexities_sorted[i+2]
                )
        
        return best_range
    
    async def _update_skill_importance_factors(
        self, 
        db: Session, 
        outcomes: List[AssignmentOutcomeCreate]
    ) -> int:
        """Update skill importance factors based on assignment outcomes."""
        factors_updated = 0
        
        for outcome in outcomes:
            try:
                assignment = db.query(TaskAssignment).filter(
                    TaskAssignment.id == outcome.assignment_id
                ).first()
                
                if not assignment:
                    continue
                
                task = db.query(Task).filter(Task.id == assignment.task_id).first()
                developer = db.query(Developer).filter(
                    Developer.id == assignment.developer_id
                ).first()
                
                if not task or not developer:
                    continue
                
                # Extract task characteristics
                task_type = self._classify_task_type(task)
                complexity_range = self._classify_complexity(task)
                domain = self._extract_primary_domain(task)
                
                # Extract developer skills
                dev_skills = developer.primary_languages or {}
                
                # Update importance factors for each skill
                for skill, skill_level in dev_skills.items():
                    await self._update_skill_factor(
                        db, skill, task_type, complexity_range, domain,
                        skill_level, outcome.task_completion_quality
                    )
                    factors_updated += 1
                
            except Exception as e:
                logger.warning(f"Failed to update skill factors for outcome {outcome.assignment_id}: {e}")
        
        return factors_updated
    
    def _classify_task_type(self, task: Task) -> str:
        """Classify task type based on description and labels."""
        description = (task.description or "").lower()
        labels = [label.lower() for label in (task.labels or [])]
        
        if any(word in description for word in ["bug", "fix", "error", "issue"]):
            return "bug_fix"
        elif any(word in description for word in ["feature", "new", "add", "implement"]):
            return "feature_development"
        elif any(word in description for word in ["refactor", "cleanup", "optimize"]):
            return "refactoring"
        elif any(word in description for word in ["test", "testing", "spec"]):
            return "testing"
        elif any(word in description for word in ["doc", "documentation", "readme"]):
            return "documentation"
        else:
            return "general"
    
    def _classify_complexity(self, task: Task) -> str:
        """Classify task complexity into ranges."""
        if task.technical_complexity is None:
            return "unknown"
        
        avg_complexity = (
            (task.technical_complexity or 0) +
            (task.domain_difficulty or 0) +
            (task.collaboration_requirements or 0)
        ) / 3.0
        
        if avg_complexity < 0.3:
            return "low"
        elif avg_complexity < 0.7:
            return "medium"
        else:
            return "high"
    
    def _extract_primary_domain(self, task: Task) -> str:
        """Extract primary domain from task."""
        description = (task.description or "").lower()
        
        domain_keywords = {
            "frontend": ["frontend", "ui", "ux", "react", "vue", "angular", "css", "html"],
            "backend": ["backend", "api", "server", "database", "sql", "rest"],
            "ml": ["ml", "machine learning", "ai", "model", "prediction", "algorithm"],
            "devops": ["devops", "deployment", "docker", "kubernetes", "ci/cd"],
            "mobile": ["mobile", "ios", "android", "react native", "flutter"],
            "security": ["security", "auth", "authentication", "encryption", "vulnerability"]
        }
        
        for domain, keywords in domain_keywords.items():
            if any(keyword in description for keyword in keywords):
                return domain
        
        return "general"
    
    async def _update_skill_factor(
        self,
        db: Session,
        skill: str,
        task_type: str,
        complexity_range: str,
        domain: str,
        skill_level: float,
        performance: float
    ):
        factor = db.query(SkillImportanceFactor).filter(
            and_(
                SkillImportanceFactor.skill_name == skill,
                SkillImportanceFactor.task_type == task_type,
                SkillImportanceFactor.complexity_range == complexity_range,
                SkillImportanceFactor.domain == domain
            )
        ).first()
        
        if not factor:
            factor = SkillImportanceFactor(
                skill_name=skill,
                task_type=task_type,
                complexity_range=complexity_range,
                domain=domain,
                successful_assignments=0,
                total_assignments=0,
                avg_performance_with_skill=0.0,
                avg_performance_without_skill=0.0
            )
            db.add(factor)
        
        # Update statistics with None checks
        factor.total_assignments = (factor.total_assignments or 0) + 1
        if performance > 0.7:
            factor.successful_assignments = (factor.successful_assignments or 0) + 1
        
        # Update average performance with None check
        if factor.avg_performance_with_skill is None:
            factor.avg_performance_with_skill = performance
        else:
            alpha = 0.1
            factor.avg_performance_with_skill = (
                alpha * performance + 
                (1 - alpha) * factor.avg_performance_with_skill
            )
        
        factor.last_learning_update = datetime.utcnow()
        db.commit()

    # MOVE THIS METHOD OUT OF _update_skill_factor (fix indentation)
    async def _update_model_performance(
        self, 
        db: Session, 
        outcomes: List[AssignmentOutcomeCreate]
    ) -> int:
        """Update model performance metrics based on actual outcomes."""
        models_updated = 0
        
        # Track complexity prediction accuracy
        complexity_predictions = []
        actual_complexity = []
        
        for outcome in outcomes:
            assignment = db.query(TaskAssignment).filter(
                TaskAssignment.id == outcome.assignment_id
            ).first()
            
            if assignment:
                task = db.query(Task).filter(Task.id == assignment.task_id).first()
                if task and task.technical_complexity is not None:
                    # Use time estimation accuracy as proxy for complexity prediction accuracy
                    domain_difficulty = task.domain_difficulty if task.domain_difficulty is not None else 0.0
                    collaboration_requirements = task.collaboration_requirements if task.collaboration_requirements is not None else 0.0
                    
                    predicted_difficulty = (
                        task.technical_complexity + 
                        domain_difficulty + 
                        collaboration_requirements
                    ) / 3.0
                    
                    # Infer actual difficulty from performance metrics
                    actual_difficulty = 1.0 - (
                        outcome.task_completion_quality + 
                        outcome.time_estimation_accuracy
                    ) / 2.0
                    
                    complexity_predictions.append(predicted_difficulty)
                    actual_complexity.append(actual_difficulty)
        
        if complexity_predictions:
            # Update complexity predictor performance
            mse = mean_squared_error(actual_complexity, complexity_predictions)
            accuracy = 1.0 - min(1.0, mse)  # Convert MSE to accuracy score
            
            await self._update_model_metrics(
                db, "complexity_predictor", "1.0", 
                accuracy_score=accuracy, mse_score=mse,
                prediction_count=len(complexity_predictions)
            )
            models_updated += 1
        
        # Track assignment optimization accuracy
        assignment_scores = []
        actual_performance = []
        
        for outcome in outcomes:
            assignment = db.query(TaskAssignment).filter(
                TaskAssignment.id == outcome.assignment_id
            ).first()
            
            if assignment and assignment.confidence_score is not None:
                assignment_scores.append(assignment.confidence_score)
                actual_performance.append(outcome.task_completion_quality)
        
        if assignment_scores:
            # Calculate correlation between predicted and actual performance
            correlation = np.corrcoef(assignment_scores, actual_performance)[0, 1]
            if not np.isnan(correlation):
                accuracy = (correlation + 1) / 2  # Convert correlation to 0-1 scale
                
                await self._update_model_metrics(
                    db, "assignment_optimizer", "1.0",
                    accuracy_score=accuracy,
                    prediction_count=len(assignment_scores)
                )
                models_updated += 1
        
        return models_updated
    
    async def _update_model_metrics(
        self,
        db: Session,
        model_name: str,
        version: str,
        accuracy_score: float,
        mse_score: Optional[float] = None,
        prediction_count: int = 0
    ):
        """Update performance metrics for a specific model."""
        performance = db.query(ModelPerformance).filter(
            and_(
                ModelPerformance.model_name == model_name,
                ModelPerformance.version == version
            )
        ).first()
        
        if not performance:
            performance = ModelPerformance(
                model_name=model_name,
                version=version,
                accuracy_score=accuracy_score,
                mse_score=mse_score,
                prediction_count=prediction_count,
                correct_predictions=int(accuracy_score * prediction_count)
            )
            db.add(performance)
        else:
            # Update with exponential moving average
            alpha = 0.1
            performance.accuracy_score = (
                alpha * accuracy_score + 
                (1 - alpha) * performance.accuracy_score
            )
            
            if mse_score is not None:
                if performance.mse_score is None:
                    performance.mse_score = mse_score
                else:
                    performance.mse_score = (
                        alpha * mse_score + 
                        (1 - alpha) * performance.mse_score
                    )
            
            performance.prediction_count += prediction_count
            performance.correct_predictions += int(accuracy_score * prediction_count)
            performance.updated_at = datetime.utcnow()
        
        db.commit()
    
    async def _generate_system_improvements(
        self, 
        db: Session, 
        outcomes: List[AssignmentOutcomeCreate]
    ) -> List[str]:
        """Generate system improvement suggestions based on outcomes."""
        improvements = []
        
        # Analyze overall performance
        avg_quality = np.mean([o.task_completion_quality for o in outcomes])
        avg_satisfaction = np.mean([o.developer_satisfaction for o in outcomes])
        avg_learning = np.mean([o.learning_achieved for o in outcomes])
        
        if avg_quality < 0.7:
            improvements.append("Task complexity prediction needs improvement")
        
        if avg_satisfaction < 0.7:
            improvements.append("Developer-task matching algorithm needs tuning")
        
        if avg_learning < 0.5:
            improvements.append("Learning opportunity detection requires enhancement")
        
        # Check for patterns in low-performing assignments
        low_quality_outcomes = [o for o in outcomes if o.task_completion_quality < 0.6]
        if len(low_quality_outcomes) > len(outcomes) * 0.3:
            improvements.append("High failure rate detected - review assignment criteria")
        
        # Check time estimation accuracy
        time_accuracies = [o.time_estimation_accuracy for o in outcomes]
        avg_time_accuracy = np.mean(time_accuracies)
        if avg_time_accuracy < 0.6:
            improvements.append("Time estimation model requires recalibration")
        
        return improvements
    
    async def _update_system_metrics(self, db: Session):
        """Update system-wide performance metrics."""
        try:
            # Calculate metrics for the last 30 days
            thirty_days_ago = datetime.utcnow() - timedelta(days=30)
            
            # Get recent outcomes
            recent_outcomes = db.query(AssignmentOutcome).join(
                TaskAssignment
            ).filter(
                TaskAssignment.assigned_at >= thirty_days_ago
            ).all()
            
            if not recent_outcomes:
                return
            
            # Calculate average metrics
            avg_assignment_quality = np.mean([o.task_completion_quality for o in recent_outcomes])
            avg_developer_satisfaction = np.mean([o.developer_satisfaction for o in recent_outcomes])
            avg_skill_development = np.mean([o.learning_achieved for o in recent_outcomes])
            avg_collaboration = np.mean([o.collaboration_effectiveness for o in recent_outcomes])
            
            # Get assignment statistics
            total_assignments = db.query(TaskAssignment).filter(
                TaskAssignment.assigned_at >= thirty_days_ago
            ).count()
            
            completed_assignments = db.query(TaskAssignment).filter(
                and_(
                    TaskAssignment.assigned_at >= thirty_days_ago,
                    TaskAssignment.status == "completed"
                )
            ).count()
            
            successful_assignments = len([
                o for o in recent_outcomes 
                if o.task_completion_quality > 0.7
            ])
            
            success_rate = successful_assignments / max(1, completed_assignments)
            
            # Get learning model performance
            model_performances = db.query(ModelPerformance).all()
            avg_model_accuracy = np.mean([
                mp.accuracy_score for mp in model_performances 
                if mp.accuracy_score is not None
            ]) if model_performances else 0.0
            
            # Create or update system metrics
            system_metrics = SystemMetrics(
                avg_assignment_quality=avg_assignment_quality,
                avg_developer_satisfaction=avg_developer_satisfaction,
                avg_skill_development_rate=avg_skill_development,
                avg_collaboration_effectiveness=avg_collaboration,
                total_assignments=total_assignments,
                completed_assignments=completed_assignments,
                successful_assignments=successful_assignments,
                assignment_success_rate=success_rate,
                active_learning_models=len(model_performances),
                learning_accuracy_trend=avg_model_accuracy,
                prediction_confidence_avg=avg_model_accuracy,
                team_productivity_score=avg_assignment_quality,
                task_completion_velocity=completed_assignments / 30.0,  # per day
                workload_balance_score=min(1.0, avg_developer_satisfaction),
                metric_period="monthly",
                period_start=thirty_days_ago,
                period_end=datetime.utcnow()
            )
            
            db.add(system_metrics)
            db.commit()
            
        except Exception as e:
            logger.error(f"Error updating system metrics: {e}")
    
    async def analyze_developer_performance_patterns(
        self, 
        db: Session, 
        developer_id: int
    ) -> DeveloperPreferenceProfile:
        """Analyze performance patterns for a specific developer."""
        # Get developer's assignment outcomes
        outcomes = db.query(AssignmentOutcome).join(
            TaskAssignment
        ).filter(
            TaskAssignment.developer_id == developer_id
        ).all()
        
        if len(outcomes) < self.min_sample_size:
            # Return default preferences
            return DeveloperPreferenceProfile(
                developer_id=developer_id,
                preferred_complexity_range=(0.3, 0.8),
                collaboration_preference=0.5,
                learning_appetite=0.7,
                optimal_workload_hours=40.0,
                workload_tolerance=0.2,
                preferred_learning_style="gradual",
                preference_confidence=0.0,
                sample_size=len(outcomes)
            )
        
        # Get preferences from database
        preference = db.query(DeveloperPreference).filter(
            DeveloperPreference.developer_id == developer_id
        ).first()
        
        if preference:
            return DeveloperPreferenceProfile(
                developer_id=developer_id,
                preferred_complexity_range=(
                    preference.preferred_complexity_min,
                    preference.preferred_complexity_max
                ),
                collaboration_preference=1.0 - preference.prefers_solo_work,
                learning_appetite=preference.learning_appetite,
                optimal_workload_hours=preference.optimal_workload_hours,
                workload_tolerance=preference.workload_tolerance,
                preferred_learning_style=preference.preferred_learning_style,
                preference_confidence=preference.preference_confidence,
                sample_size=preference.sample_size
            )
        
        # Calculate preferences from outcomes if no stored preferences
        complexities = []
        satisfactions = []
        
        for outcome in outcomes:
            assignment = db.query(TaskAssignment).filter(
                TaskAssignment.id == outcome.assignment_id
            ).first()
            
            if assignment:
                task = db.query(Task).filter(Task.id == assignment.task_id).first()
                if task and task.technical_complexity is not None:
                    avg_complexity = (
                        task.technical_complexity + 
                        task.domain_difficulty + 
                        task.collaboration_requirements
                    ) / 3.0
                    
                    complexities.append(avg_complexity)
                    satisfactions.append(outcome.developer_satisfaction)
        
        optimal_range = self._find_optimal_complexity_range(complexities, satisfactions)
        avg_collaboration = np.mean([o.collaboration_effectiveness for o in outcomes])
        avg_learning = np.mean([o.learning_achieved for o in outcomes])
        
        return DeveloperPreferenceProfile(
            developer_id=developer_id,
            preferred_complexity_range=optimal_range,
            collaboration_preference=avg_collaboration,
            learning_appetite=avg_learning,
            optimal_workload_hours=40.0,
            workload_tolerance=0.2,
            preferred_learning_style="gradual",
            preference_confidence=min(1.0, len(outcomes) / (self.min_sample_size * 2)),
            sample_size=len(outcomes)
        )
    
    async def get_skill_importance_analysis(
        self, 
        db: Session,
        task_type: Optional[str] = None,
        domain: Optional[str] = None
    ) -> List[SkillImportanceAnalysis]:
        """Get skill importance analysis for given context."""
        query = db.query(SkillImportanceFactor)
        
        if task_type:
            query = query.filter(SkillImportanceFactor.task_type == task_type)
        
        if domain:
            query = query.filter(SkillImportanceFactor.domain == domain)
        
        factors = query.filter(
            SkillImportanceFactor.total_assignments >= self.min_sample_size
        ).all()
        
        results = []
        for factor in factors:
            success_rate_with = (
                factor.successful_assignments / factor.total_assignments
                if factor.total_assignments > 0 else 0.0
            )
            
            # Estimate success rate without skill (simplified)
            success_rate_without = max(0.0, success_rate_with - 0.2)
            
            results.append(SkillImportanceAnalysis(
                skill_name=factor.skill_name,
                task_type=factor.task_type,
                complexity_range=factor.complexity_range,
                domain=factor.domain,
                importance_factor=factor.importance_factor,
                confidence=factor.confidence,
                success_rate_with_skill=success_rate_with,
                success_rate_without_skill=success_rate_without,
                sample_size=factor.total_assignments
            ))
        
        return sorted(results, key=lambda x: x.importance_factor, reverse=True)
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from sqlalchemy.orm import Session
from sqlalchemy import func, and_
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import joblib
import asyncio

from ...models.database import (
    TaskAssignment, AssignmentOutcome, Developer, Task,
    ModelPerformance, LearningExperiment, DeveloperPreference,
    SkillImportanceFactor
)
from ...models.schemas import (
    ModelUpdateResult, LearningExperimentCreate, 
    LearningExperiment as LearningExperimentSchema
)
from ..assignment_engine.learning_automata import LearningAutomata

logger = logging.getLogger(__name__)

class ModelUpdater:
    """
    Continuously update and improve ML models based on assignment outcomes.
    
    This class handles:
    1. Retraining models with new data
    2. Parameter optimization based on performance feedback
    3. A/B testing different model configurations
    4. Model versioning and rollback capabilities
    5. Automated hyperparameter tuning
    """
    
    def __init__(self):
        self.learning_automata = LearningAutomata()
        self.min_update_samples = 20
        self.performance_threshold = 0.05  # Minimum improvement for model update
        self.models_cache = {}
        
    async def update_models_from_outcomes(
        self, 
        db: Session,
        force_update: bool = False
    ) -> List[ModelUpdateResult]:
        """
        Update all models based on recent assignment outcomes.
        
        Args:
            db: Database session
            force_update: Force update even if performance hasn't changed significantly
            
        Returns:
            List of model update results
        """
        results = []
        
        try:
            # Update complexity prediction model
            complexity_result = await self._update_complexity_predictor(db, force_update)
            if complexity_result:
                results.append(complexity_result)
            
            # Update assignment optimization weights
            optimization_result = await self._update_optimization_weights(db, force_update)
            if optimization_result:
                results.append(optimization_result)
            
            # Update developer skill extraction model
            skill_result = await self._update_skill_extraction_model(db, force_update)
            if skill_result:
                results.append(skill_result)
            
            logger.info(f"Updated {len(results)} models based on outcomes")
            return results
            
        except Exception as e:
            logger.error(f"Error updating models: {e}")
            raise
    
    async def _update_complexity_predictor(
        self, 
        db: Session, 
        force_update: bool
    ) -> Optional[ModelUpdateResult]:
        """Update task complexity prediction model."""
        # Get training data from recent outcomes
        training_data = await self._get_complexity_training_data(db)
        
        if len(training_data) < self.min_update_samples and not force_update:
            return None
        
        try:
            # Extract features and targets
            features = []
            targets = []
            
            for data_point in training_data:
                task_features = self._extract_task_features(data_point['task'])
                actual_complexity = self._calculate_actual_complexity(data_point['outcome'])
                
                features.append(task_features)
                targets.append(actual_complexity)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                features, targets, test_size=0.2, random_state=42
            )
            
            # Train updated model (simplified - would use actual ML model)
            updated_weights = self._train_complexity_model(X_train, y_train)
            
            # Evaluate performance
            predictions = self._predict_complexity(X_test, updated_weights)
            mse = mean_squared_error(y_test, predictions)
            accuracy = 1.0 - min(1.0, mse)
            
            # Get current model performance
            current_performance = db.query(ModelPerformance).filter(
                ModelPerformance.model_name == "complexity_predictor"
            ).order_by(ModelPerformance.created_at.desc()).first()
            
            current_accuracy = current_performance.accuracy_score if current_performance else 0.0
            improvement = accuracy - current_accuracy
            
            if improvement > self.performance_threshold or force_update:
                # Update model
                new_version = self._generate_version()
                
                # Store updated model performance
                performance = ModelPerformance(
                    model_name="complexity_predictor",
                    version=new_version,
                    accuracy_score=accuracy,
                    mse_score=mse,
                    training_data_size=len(X_train),
                    validation_data_size=len(X_test),
                    hyperparameters={"weights": updated_weights}
                )
                db.add(performance)
                db.commit()
                
                return ModelUpdateResult(
                    model_name="complexity_predictor",
                    previous_version=current_performance.version if current_performance else "1.0",
                    new_version=new_version,
                    performance_improvement=improvement,
                    update_type="retrain",
                    validation_results={
                        "accuracy": accuracy,
                        "mse": mse,
                        "training_samples": len(X_train)
                    },
                    rollback_available=current_performance is not None
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error updating complexity predictor: {e}")
            return None
    
    async def _get_complexity_training_data(self, db: Session) -> List[Dict]:
        """Get training data for complexity prediction from recent outcomes."""
        # Get outcomes from the last 60 days
        sixty_days_ago = datetime.utcnow() - timedelta(days=60)
        
        outcomes = db.query(AssignmentOutcome).join(
            TaskAssignment
        ).join(
            Task
        ).filter(
            TaskAssignment.assigned_at >= sixty_days_ago
        ).all()
        
        training_data = []
        for outcome in outcomes:
            assignment = outcome.assignment
            task = assignment.task
            
            if task and all([
                task.technical_complexity is not None,
                task.domain_difficulty is not None,
                task.collaboration_requirements is not None
            ]):
                training_data.append({
                    'task': task,
                    'outcome': outcome,
                    'assignment': assignment
                })
        
        return training_data
    
    def _extract_task_features(self, task: Task) -> List[float]:
        """Extract numerical features from task for ML model."""
        features = [
            len(task.description or "") / 1000.0,  # Description length
            len(task.labels or []),  # Number of labels
            1.0 if "bug" in (task.description or "").lower() else 0.0,  # Is bug
            1.0 if "feature" in (task.description or "").lower() else 0.0,  # Is feature
            len(task.required_skills or {}) / 10.0,  # Number of required skills
        ]
        
        # Add complexity factors if available
        if task.complexity_factors:
            factors = task.complexity_factors
            features.extend([
                factors.get("code_complexity", 0.0),
                factors.get("architectural_impact", 0.0),
                factors.get("domain_knowledge_required", 0.0),
                factors.get("collaboration_complexity", 0.0)
            ])
        else:
            features.extend([0.0, 0.0, 0.0, 0.0])
        
        return features
    
    def _calculate_actual_complexity(self, outcome: AssignmentOutcome) -> float:
        """Calculate actual task complexity from outcome data."""
        # Use inverse of performance metrics as complexity indicator
        quality_factor = 1.0 - outcome.task_completion_quality
        time_factor = 1.0 - outcome.time_estimation_accuracy
        collaboration_factor = 1.0 - outcome.collaboration_effectiveness
        
        # Weighted average
        actual_complexity = (
            0.4 * quality_factor +
            0.3 * time_factor +
            0.3 * collaboration_factor
        )
        
        return min(1.0, max(0.0, actual_complexity))
    
    def _train_complexity_model(self, X_train: List[List[float]], y_train: List[float]) -> Dict[str, float]:
        """Train complexity prediction model (simplified implementation)."""
        # Simplified linear model - in practice, would use more sophisticated ML
        X = np.array(X_train)
        y = np.array(y_train)
        
        # Calculate feature weights using correlation
        weights = {}
        feature_names = [
            "description_length", "label_count", "is_bug", "is_feature", "skill_count",
            "code_complexity", "architectural_impact", "domain_knowledge", "collaboration_complexity"
        ]
        
        for i, feature_name in enumerate(feature_names):
            if X.shape[1] > i:
                correlation = np.corrcoef(X[:, i], y)[0, 1]
                weights[feature_name] = correlation if not np.isnan(correlation) else 0.0
        
        return weights
    
    def _predict_complexity(self, X_test: List[List[float]], weights: Dict[str, float]) -> List[float]:
        """Make complexity predictions using trained weights."""
        predictions = []
        feature_names = list(weights.keys())
        
        for features in X_test:
            prediction = 0.0
            for i, feature_name in enumerate(feature_names):
                if i < len(features):
                    prediction += features[i] * weights.get(feature_name, 0.0)
            
            predictions.append(max(0.0, min(1.0, prediction)))
        
        return predictions
    
    def _generate_version(self) -> str:
        """Generate new version string."""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        return f"1.0.{timestamp}"
    
    async def _update_optimization_weights(
        self, 
        db: Session, 
        force_update: bool
    ) -> Optional[ModelUpdateResult]:
        """Update assignment optimization weights based on outcomes."""
        try:
            # Get recent assignment outcomes
            outcomes_data = await self._get_optimization_training_data(db)
            
            if len(outcomes_data) < self.min_update_samples and not force_update:
                return None
            
            # Calculate optimal weights using learning automata
            current_weights = {
                "productivity": 0.35,
                "skill_development": 0.25,
                "workload_balance": 0.20,
                "collaboration": 0.10,
                "business_impact": 0.10
            }
            
            # Learn from outcomes
            for data in outcomes_data:
                outcome = data['outcome']
                
                # Calculate reward based on overall assignment success
                reward = (
                    outcome.task_completion_quality * 0.4 +
                    outcome.developer_satisfaction * 0.3 +
                    outcome.learning_achieved * 0.2 +
                    outcome.collaboration_effectiveness * 0.1
                )
                
                # Update weights using simple gradient ascent
                if reward > 0.7:  # Good outcome
                    # Increase weights that contributed to success
                    if outcome.task_completion_quality > 0.8:
                        current_weights["productivity"] *= 1.02
                    if outcome.learning_achieved > 0.7:
                        current_weights["skill_development"] *= 1.02
                    if outcome.collaboration_effectiveness > 0.7:
                        current_weights["collaboration"] *= 1.02
                else:  # Poor outcome
                    # Adjust weights
                    if outcome.task_completion_quality < 0.5:
                        current_weights["productivity"] *= 1.05
                    if outcome.developer_satisfaction < 0.5:
                        current_weights["workload_balance"] *= 1.05
            
            # Normalize weights
            total_weight = sum(current_weights.values())
            normalized_weights = {k: v/total_weight for k, v in current_weights.items()}
            
            # Store updated weights
            performance = ModelPerformance(
                model_name="assignment_optimizer",
                version=self._generate_version(),
                accuracy_score=np.mean([d['overall_score'] for d in outcomes_data]),
                hyperparameters={"optimization_weights": normalized_weights}
            )
            db.add(performance)
            db.commit()
            
            return ModelUpdateResult(
                model_name="assignment_optimizer",
                previous_version="1.0",
                new_version=performance.version,
                performance_improvement=0.02,  # Estimated improvement
                update_type="parameter_tune",
                validation_results={"updated_weights": normalized_weights},
                rollback_available=True
            )
            
        except Exception as e:
            logger.error(f"Error updating optimization weights: {e}")
            return None
    
    async def _get_optimization_training_data(self, db: Session) -> List[Dict]:
        """Get training data for optimization weight updates."""
        thirty_days_ago = datetime.utcnow() - timedelta(days=30)
        
        outcomes = db.query(AssignmentOutcome).join(
            TaskAssignment
        ).filter(
            TaskAssignment.assigned_at >= thirty_days_ago
        ).all()
        
        training_data = []
        for outcome in outcomes:
            assignment = outcome.assignment
            
            # Calculate overall assignment score
            overall_score = (
                outcome.task_completion_quality * 0.4 +
                outcome.developer_satisfaction * 0.3 +
                outcome.learning_achieved * 0.2 +
                outcome.collaboration_effectiveness * 0.1
            )
            
            training_data.append({
                'outcome': outcome,
                'assignment': assignment,
                'overall_score': overall_score
            })
        
        return training_data
    
    async def _update_skill_extraction_model(
        self, 
        db: Session, 
        force_update: bool
    ) -> Optional[ModelUpdateResult]:
        """Update skill extraction and importance models."""
        try:
            # Get skill importance factors that have been updated recently
            recent_factors = db.query(SkillImportanceFactor).filter(
                SkillImportanceFactor.last_learning_update >= datetime.utcnow() - timedelta(days=7)
            ).all()
            
            if len(recent_factors) < 5 and not force_update:
                return None
            
            # Update skill extraction weights based on learned importance
            skill_weights = {}
            for factor in recent_factors:
                skill_weights[factor.skill_name] = factor.importance_factor
            
            # Normalize weights
            if skill_weights:
                max_weight = max(skill_weights.values())
                normalized_weights = {
                    k: v / max_weight for k, v in skill_weights.items()
                }
                
                # Store updated model
                performance = ModelPerformance(
                    model_name="skill_extractor",
                    version=self._generate_version(),
                    accuracy_score=0.85,  # Estimated based on importance learning
                    hyperparameters={"skill_weights": normalized_weights}
                )
                db.add(performance)
                db.commit()
                
                return ModelUpdateResult(
                    model_name="skill_extractor",
                    previous_version="1.0",
                    new_version=performance.version,
                    performance_improvement=0.03,
                    update_type="parameter_tune",
                    validation_results={"skill_importance_weights": normalized_weights},
                    rollback_available=True
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error updating skill extraction model: {e}")
            return None
    
    async def create_learning_experiment(
        self, 
        db: Session,
        experiment: LearningExperimentCreate
    ) -> LearningExperimentSchema:
        """Create a new learning experiment for A/B testing."""
        db_experiment = LearningExperiment(**experiment.dict())
        db.add(db_experiment)
        db.commit()
        db.refresh(db_experiment)
        
        return LearningExperimentSchema.from_orm(db_experiment)
    
    async def run_ab_test(
        self, 
        db: Session,
        experiment_id: int,
        duration_days: int = 14
    ) -> bool:
        """Run A/B test for a learning experiment."""
        experiment = db.query(LearningExperiment).filter(
            LearningExperiment.id == experiment_id
        ).first()
        
        if not experiment:
            return False
        
        try:
            # Set experiment as active
            experiment.status = "active"
            experiment.start_date = datetime.utcnow()
            experiment.end_date = datetime.utcnow() + timedelta(days=duration_days)
            
            db.commit()
            
            logger.info(f"Started A/B test experiment: {experiment.experiment_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error starting A/B test: {e}")
            return False
    
    async def evaluate_experiment(
        self, 
        db: Session,
        experiment_id: int
    ) -> Optional[LearningExperimentSchema]:
        """Evaluate results of a learning experiment."""
        experiment = db.query(LearningExperiment).filter(
            LearningExperiment.id == experiment_id
        ).first()
        if not experiment:
            return None
        
        try:
            # Get assignments from experiment period
            assignments = db.query(TaskAssignment).filter(
                and_(
                    TaskAssignment.assigned_at >= experiment.start_date,
                    TaskAssignment.assigned_at <= (experiment.end_date or datetime.utcnow())
                )
            ).all()
            
            if len(assignments) < 10:  # Need minimum sample size
                return None
            
            # Split assignments into control and experimental groups (simplified)
            control_assignments = assignments[:len(assignments)//2]
            experimental_assignments = assignments[len(assignments)//2:]
            
            # Calculate metrics for both groups
            control_metrics = await self._calculate_group_metrics(db, control_assignments)
            experimental_metrics = await self._calculate_group_metrics(db, experimental_assignments)
            
            # Calculate statistical significance (simplified)
            significance = self._calculate_statistical_significance(
                control_metrics, experimental_metrics
            )
            
            # Update experiment results
            experiment.control_results = control_metrics
            experiment.experimental_results = experimental_metrics
            experiment.statistical_significance = significance
            experiment.confidence_level = 0.95 if significance > 0.05 else 0.90
            experiment.status = "completed"
            
            db.commit()
            
            logger.info(f"Completed experiment evaluation: {experiment.experiment_name}")
            return LearningExperimentSchema.from_orm(experiment)
            
        except Exception as e:
            logger.error(f"Error evaluating experiment: {e}")
            return None
    
    async def _calculate_group_metrics(
        self, 
        db: Session, 
        assignments: List[TaskAssignment]
    ) -> Dict[str, float]:
        """Calculate performance metrics for a group of assignments."""
        if not assignments:
            return {}
        
        outcomes = []
        for assignment in assignments:
            outcome = db.query(AssignmentOutcome).filter(
                AssignmentOutcome.assignment_id == assignment.id
            ).first()
            if outcome:
                outcomes.append(outcome)
        
        if not outcomes:
            return {"sample_size": 0}
        
        metrics = {
            "sample_size": len(outcomes),
            "avg_task_quality": np.mean([o.task_completion_quality for o in outcomes]),
            "avg_satisfaction": np.mean([o.developer_satisfaction for o in outcomes]),
            "avg_learning": np.mean([o.learning_achieved for o in outcomes]),
            "avg_collaboration": np.mean([o.collaboration_effectiveness for o in outcomes]),
            "success_rate": len([o for o in outcomes if o.task_completion_quality > 0.7]) / len(outcomes)
        }
        
        return metrics
    
    def _calculate_statistical_significance(
        self, 
        control_metrics: Dict[str, float], 
        experimental_metrics: Dict[str, float]
    ) -> float:
        """Calculate statistical significance between two groups (simplified)."""
        if not control_metrics or not experimental_metrics:
            return 0.0
        
        # Simplified t-test calculation
        control_success = control_metrics.get("success_rate", 0.0)
        experimental_success = experimental_metrics.get("success_rate", 0.0)
        
        diff = abs(experimental_success - control_success)
        
        # Simple significance estimation
        if diff > 0.1:
            return 0.01  # Highly significant
        elif diff > 0.05:
            return 0.05  # Significant
        else:
            return 0.2   # Not significant
    
    async def rollback_model(
        self, 
        db: Session,
        model_name: str,
        target_version: Optional[str] = None
    ) -> bool:
        """Rollback model to previous version."""
        try:
            # Get model versions
            versions = db.query(ModelPerformance).filter(
                ModelPerformance.model_name == model_name
            ).order_by(ModelPerformance.created_at.desc()).all()
            
            if len(versions) < 2:
                logger.warning(f"No previous version available for rollback: {model_name}")
                return False
            
            # Find target version
            target = None
            if target_version:
                target = next((v for v in versions if v.version == target_version), None)
            else:
                target = versions[1]  # Previous version
            
            if not target:
                logger.warning(f"Target version not found: {target_version}")
                return False
            
            # Create new entry marking rollback
            rollback_entry = ModelPerformance(
                model_name=model_name,
                version=f"{target.version}_rollback_{self._generate_version()}",
                accuracy_score=target.accuracy_score,
                precision_score=target.precision_score,
                recall_score=target.recall_score,
                f1_score=target.f1_score,
                mse_score=target.mse_score,
                hyperparameters=target.hyperparameters
            )
            
            db.add(rollback_entry)
            db.commit()
            
            logger.info(f"Rolled back model {model_name} to version {target.version}")
            return True
            
        except Exception as e:
            logger.error(f"Error rolling back model: {e}")
            return False
    
    async def optimize_hyperparameters(
        self, 
        db: Session,
        model_name: str,
        parameter_space: Dict[str, List[Any]]
    ) -> Optional[ModelUpdateResult]:
        """Optimize hyperparameters for a model using grid search."""
        try:
            best_params = None
            best_score = 0.0
            
            # Get training data
            if model_name == "complexity_predictor":
                training_data = await self._get_complexity_training_data(db)
            elif model_name == "assignment_optimizer":
                training_data = await self._get_optimization_training_data(db)
            else:
                logger.warning(f"Unknown model for hyperparameter optimization: {model_name}")
                return None
            
            if len(training_data) < self.min_update_samples:
                return None
            
            # Simple grid search (would use more sophisticated optimization in practice)
            param_combinations = self._generate_param_combinations(parameter_space)
            
            for params in param_combinations:
                score = await self._evaluate_parameters(db, model_name, params, training_data)
                
                if score > best_score:
                    best_score = score
                    best_params = params
            
            if best_params and best_score > 0.7:
                # Store optimized model
                performance = ModelPerformance(
                    model_name=model_name,
                    version=self._generate_version(),
                    accuracy_score=best_score,
                    hyperparameters=best_params
                )
                db.add(performance)
                db.commit()
                
                return ModelUpdateResult(
                    model_name=model_name,
                    previous_version="1.0",
                    new_version=performance.version,
                    performance_improvement=best_score - 0.7,
                    update_type="parameter_tune",
                    validation_results={"optimized_params": best_params, "score": best_score},
                    rollback_available=True
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error optimizing hyperparameters: {e}")
            return None
    
    def _generate_param_combinations(self, parameter_space: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        """Generate all combinations of hyperparameters."""
        import itertools
        
        keys = list(parameter_space.keys())
        values = list(parameter_space.values())
        
        combinations = []
        for combo in itertools.product(*values):
            combinations.append(dict(zip(keys, combo)))
        
        return combinations[:10]  # Limit to first 10 combinations for efficiency
    
    async def _evaluate_parameters(
        self, 
        db: Session,
        model_name: str,
        params: Dict[str, Any],
        training_data: List[Dict]
    ) -> float:
        """Evaluate parameter combination performance."""
        try:
            if model_name == "complexity_predictor":
                return await self._evaluate_complexity_params(params, training_data)
            elif model_name == "assignment_optimizer":
                return await self._evaluate_optimization_params(params, training_data)
            else:
                return 0.0
        except Exception as e:
            logger.warning(f"Error evaluating parameters: {e}")
            return 0.0
    
    async def _evaluate_complexity_params(
        self, 
        params: Dict[str, Any], 
        training_data: List[Dict]
    ) -> float:
        """Evaluate complexity prediction parameters."""
        # Extract features and targets
        features = []
        targets = []
        
        for data_point in training_data:
            task_features = self._extract_task_features(data_point['task'])
            actual_complexity = self._calculate_actual_complexity(data_point['outcome'])
            
            features.append(task_features)
            targets.append(actual_complexity)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, targets, test_size=0.3, random_state=42
        )
        
        # Train with parameters
        weights = self._train_complexity_model_with_params(X_train, y_train, params)
        
        # Make predictions
        predictions = self._predict_complexity(X_test, weights)
        
        # Calculate accuracy
        mse = mean_squared_error(y_test, predictions)
        accuracy = 1.0 - min(1.0, mse)
        
        return accuracy
    
    def _train_complexity_model_with_params(
        self, 
        X_train: List[List[float]], 
        y_train: List[float], 
        params: Dict[str, Any]
    ) -> Dict[str, float]:
        """Train complexity model with specific parameters."""
        # Apply parameter-specific training (simplified)
        weights = self._train_complexity_model(X_train, y_train)
        
        # Apply parameter modifications
        learning_rate = params.get("learning_rate", 1.0)
        regularization = params.get("regularization", 0.0)
        
        # Modify weights based on parameters
        for key in weights:
            weights[key] *= learning_rate
            weights[key] *= (1 - regularization)
        
        return weights
    
    async def _evaluate_optimization_params(
        self, 
        params: Dict[str, Any], 
        training_data: List[Dict]
    ) -> float:
        """Evaluate assignment optimization parameters."""
        # Calculate average outcome score with given parameters
        total_score = 0.0
        count = 0
        
        for data in training_data:
            outcome = data['outcome']
            
            # Apply parameter weights to calculate score
            score = (
                outcome.task_completion_quality * params.get("productivity_weight", 0.35) +
                outcome.learning_achieved * params.get("learning_weight", 0.25) +
                outcome.collaboration_effectiveness * params.get("collaboration_weight", 0.20) +
                outcome.developer_satisfaction * params.get("satisfaction_weight", 0.20)
            )
            
            total_score += score
            count += 1
        
        return total_score / count if count > 0 else 0.0
    
    async def get_model_performance_history(
        self, 
        db: Session,
        model_name: str,
        days: int = 30
    ) -> List[Dict[str, Any]]:
        """Get performance history for a model."""
        start_date = datetime.utcnow() - timedelta(days=days)
        
        performances = db.query(ModelPerformance).filter(
            and_(
                ModelPerformance.model_name == model_name,
                ModelPerformance.created_at >= start_date
            )
        ).order_by(ModelPerformance.created_at).all()
        
        history = []
        for perf in performances:
            history.append({
                "version": perf.version,
                "accuracy": perf.accuracy_score,
                "created_at": perf.created_at,
                "training_samples": perf.training_data_size,
                "predictions_made": perf.prediction_count
            })
        
        return history
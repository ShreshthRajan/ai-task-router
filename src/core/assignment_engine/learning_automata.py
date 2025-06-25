"""
Learning Automata for Adaptive Assignment Optimization

This module implements adaptive algorithms that learn from assignment outcomes
to improve future assignment decisions.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from collections import defaultdict, deque
import json
import logging
from sqlalchemy.orm import Session

from models.database import TaskAssignment, Developer, Task

logger = logging.getLogger(__name__)

class LearningAutomata:
    """
    Adaptive learning system that improves assignment decisions based on outcomes.
    Implements reinforcement learning principles for task-developer matching.
    """
    
    def __init__(self, learning_rate: float = 0.1, memory_size: int = 1000):
        self.learning_rate = learning_rate
        self.memory_size = memory_size
        
        # Learning components
        self.skill_accuracy_history = defaultdict(lambda: deque(maxlen=100))
        self.complexity_prediction_errors = deque(maxlen=memory_size)
        self.assignment_success_patterns = defaultdict(list)
        self.developer_preference_models = {}
        self.task_difficulty_adjustments = {}
        
        # Adaptive parameters
        self.complexity_weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])  # Equal initial weights
        self.skill_importance_factors = defaultdict(lambda: 1.0)
        self.collaboration_effectiveness_matrix = defaultdict(lambda: defaultdict(float))
        
        # Performance tracking
        self.prediction_accuracy_history = []
        self.learning_convergence_metrics = {}
        
    async def learn_from_assignment_outcome(
        self,
        db: Session,
        assignment_id: int,
        outcome_metrics: Dict[str, float]
    ) -> None:
        """
        Learn from assignment outcome to improve future predictions.
        
        Args:
            db: Database session
            assignment_id: Completed assignment ID
            outcome_metrics: Performance metrics from the assignment
        """
        try:
            assignment = db.query(TaskAssignment).filter(TaskAssignment.id == assignment_id).first()
            if not assignment or not assignment.task:
                return
            
            # Extract learning signals
            await self._update_skill_accuracy_models(assignment, outcome_metrics)
            await self._update_complexity_prediction_models(assignment, outcome_metrics)
            await self._update_developer_preference_models(assignment, outcome_metrics)
            await self._update_collaboration_effectiveness_models(assignment, outcome_metrics)
            
            # Store learning pattern
            self.assignment_success_patterns[assignment.developer_id].append({
                'task_complexity': [
                    assignment.task.technical_complexity,
                    assignment.task.domain_difficulty,
                    assignment.task.collaboration_requirements,
                    assignment.task.learning_opportunities,
                    assignment.task.business_impact
                ],
                'outcome_score': outcome_metrics.get('productivity_score', 0.5),
                'skill_development': outcome_metrics.get('skill_development_score', 0.5),
                'satisfaction': outcome_metrics.get('feedback_score', 0.5),
                'timestamp': datetime.now()
            })
            
            # Adapt global parameters
            await self._adapt_global_parameters(assignment, outcome_metrics)
            
            logger.info(f"Learned from assignment {assignment_id} outcome")
            
        except Exception as e:
            logger.error(f"Learning from assignment outcome failed: {e}")
    
    async def _update_skill_accuracy_models(
        self,
        assignment: TaskAssignment,
        outcome_metrics: Dict[str, float]
    ) -> None:
        """Update skill matching accuracy models based on assignment outcomes."""
        
        if not assignment.task.required_skills:
            return
        
        required_skills = json.loads(assignment.task.required_skills)
        productivity_score = outcome_metrics.get('productivity_score', 0.5)
        
        # Update skill importance factors based on outcome
        for skill, importance in required_skills.items():
            if productivity_score > 0.7:
                # Good outcome - reinforce skill importance
                self.skill_importance_factors[skill] = min(2.0, 
                    self.skill_importance_factors[skill] * (1 + self.learning_rate))
            elif productivity_score < 0.4:
                # Poor outcome - reduce skill importance if it was overweighted
                self.skill_importance_factors[skill] = max(0.5,
                    self.skill_importance_factors[skill] * (1 - self.learning_rate))
            
            # Track accuracy for this skill prediction
            self.skill_accuracy_history[skill].append({
                'predicted_importance': importance,
                'actual_outcome': productivity_score,
                'developer_id': assignment.developer_id
            })
    
    async def _update_complexity_prediction_models(
        self,
        assignment: TaskAssignment,
        outcome_metrics: Dict[str, float]
    ) -> None:
        """Update complexity prediction accuracy based on actual outcomes."""
        
        # Calculate prediction error
        estimated_hours = assignment.task.estimated_hours or 8.0
        actual_hours = assignment.actual_hours or estimated_hours
        
        time_error = abs(actual_hours - estimated_hours) / estimated_hours
        productivity_score = outcome_metrics.get('productivity_score', 0.5)
        
        # Record prediction error
        complexity_error = {
            'time_error': time_error,
            'productivity_error': abs(productivity_score - 0.8),  # Expected good performance
            'task_complexity': [
                assignment.task.technical_complexity or 0.5,
                assignment.task.domain_difficulty or 0.5,
                assignment.task.collaboration_requirements or 0.3,
                assignment.task.learning_opportunities or 0.4,
                assignment.task.business_impact or 0.5
            ],
            'developer_id': assignment.developer_id
        }
        
        self.complexity_prediction_errors.append(complexity_error)
        
        # Adapt complexity weights based on prediction accuracy
        if len(self.complexity_prediction_errors) >= 10:
            await self._adapt_complexity_weights()
    
    async def _update_developer_preference_models(
        self,
        assignment: TaskAssignment,
        outcome_metrics: Dict[str, float]
    ) -> None:
        """Update developer preference and performance models."""
        
        developer_id = assignment.developer_id
        
        if developer_id not in self.developer_preference_models:
            self.developer_preference_models[developer_id] = {
                'preferred_complexity_factors': np.array([0.5, 0.5, 0.5, 0.5, 0.5]),
                'performance_by_complexity': [],
                'collaboration_preference': 0.5,
                'learning_preference': 0.5,
                'workload_tolerance': 0.7
            }
        
        model = self.developer_preference_models[developer_id]
        
        # Update complexity preferences based on satisfaction and performance
        task_complexity = np.array([
            assignment.task.technical_complexity or 0.5,
            assignment.task.domain_difficulty or 0.5,
            assignment.task.collaboration_requirements or 0.3,
            assignment.task.learning_opportunities or 0.4,
            assignment.task.business_impact or 0.5
        ])
        
        satisfaction = outcome_metrics.get('feedback_score', 0.5)
        productivity = outcome_metrics.get('productivity_score', 0.5)
        
        # Record performance at this complexity level
        model['performance_by_complexity'].append({
            'complexity': task_complexity,
            'satisfaction': satisfaction,
            'productivity': productivity,
            'timestamp': datetime.now()
        })
        
        # Update preferences using exponential moving average
        if satisfaction > 0.7 and productivity > 0.6:
            # Good outcome - move preferences toward this complexity
            model['preferred_complexity_factors'] = (
                (1 - self.learning_rate) * model['preferred_complexity_factors'] +
                self.learning_rate * task_complexity
            )
        
        # Update collaboration and learning preferences
        collab_score = outcome_metrics.get('collaboration_effectiveness', 0.5)
        learning_score = outcome_metrics.get('skill_development_score', 0.5)
        
        if collab_score > 0.7:
            model['collaboration_preference'] = min(1.0,
                model['collaboration_preference'] + self.learning_rate * 0.1)
        
        if learning_score > 0.7:
            model['learning_preference'] = min(1.0,
                model['learning_preference'] + self.learning_rate * 0.1)
    
    async def _update_collaboration_effectiveness_models(
        self,
        assignment: TaskAssignment,
        outcome_metrics: Dict[str, float]
    ) -> None:
        """Update collaboration effectiveness between developers."""
        
        if assignment.task.collaboration_requirements < 0.5:
            return  # Not a collaborative task
        
        developer_id = assignment.developer_id
        collaboration_score = outcome_metrics.get('collaboration_effectiveness', 0.5)
        
        # This would be enhanced to track specific developer-developer collaboration
        # For now, track general collaboration effectiveness
        self.collaboration_effectiveness_matrix[developer_id]['general'] = (
            0.8 * self.collaboration_effectiveness_matrix[developer_id]['general'] +
            0.2 * collaboration_score
        )
    
    async def _adapt_complexity_weights(self) -> None:
        """Adapt complexity dimension weights based on prediction accuracy."""
        
        if len(self.complexity_prediction_errors) < 20:
            return
        
        # Analyze which complexity dimensions correlate with prediction errors
        recent_errors = list(self.complexity_prediction_errors)[-20:]
        
        complexity_matrix = np.array([error['task_complexity'] for error in recent_errors])
        time_errors = np.array([error['time_error'] for error in recent_errors])
        
        # Calculate correlation between each complexity dimension and prediction error
        correlations = []
        for i in range(5):
            if np.std(complexity_matrix[:, i]) > 0:
                correlation = np.corrcoef(complexity_matrix[:, i], time_errors)[0, 1]
                correlations.append(abs(correlation))
            else:
                correlations.append(0.0)
        
        # Adapt weights - reduce weight for dimensions highly correlated with errors
        correlations = np.array(correlations)
        if np.sum(correlations) > 0:
            # Inverse correlation weighting
            new_weights = 1.0 / (1.0 + correlations)
            new_weights = new_weights / np.sum(new_weights)  # Normalize
            
            # Smooth update
            self.complexity_weights = (
                0.9 * self.complexity_weights + 0.1 * new_weights
            )
    
    async def _adapt_global_parameters(
        self,
        assignment: TaskAssignment,
        outcome_metrics: Dict[str, float]
    ) -> None:
        """Adapt global optimization parameters based on outcomes."""
        
        productivity = outcome_metrics.get('productivity_score', 0.5)
        satisfaction = outcome_metrics.get('feedback_score', 0.5)
        
        # Track overall prediction accuracy
        prediction_accuracy = (productivity + satisfaction) / 2.0
        self.prediction_accuracy_history.append({
            'accuracy': prediction_accuracy,
            'timestamp': datetime.now()
        })
        
        # Adapt learning rate based on recent performance
        if len(self.prediction_accuracy_history) >= 50:
            recent_accuracy = np.mean([p['accuracy'] for p in self.prediction_accuracy_history[-20:]])
            
            if recent_accuracy > 0.8:
                # Good performance - reduce learning rate for stability
                self.learning_rate = max(0.01, self.learning_rate * 0.95)
            elif recent_accuracy < 0.6:
                # Poor performance - increase learning rate for faster adaptation
                self.learning_rate = min(0.3, self.learning_rate * 1.05)
    
    def get_enhanced_assignment_weights(self, task_id: int, developer_id: int) -> Dict[str, float]:
        """
        Get enhanced assignment weights based on learned patterns.
        
        Args:
            task_id: Task ID for assignment
            developer_id: Developer ID for assignment
            
        Returns:
            Dictionary of adjusted weights for assignment scoring
        """
        
        base_weights = {
            'productivity': 0.35,
            'skill_development': 0.25,
            'workload_balance': 0.20,
            'collaboration': 0.10,
            'business_impact': 0.10
        }
        
        # Adjust weights based on developer preferences
        if developer_id in self.developer_preference_models:
            model = self.developer_preference_models[developer_id]
            
            # Increase learning weight if developer has high learning preference
            if model['learning_preference'] > 0.7:
                base_weights['skill_development'] *= 1.2
                base_weights['productivity'] *= 0.9
            
            # Adjust collaboration weight based on collaboration preference
            if model['collaboration_preference'] > 0.7:
                base_weights['collaboration'] *= 1.5
            elif model['collaboration_preference'] < 0.3:
                base_weights['collaboration'] *= 0.5
        
        # Normalize weights
        total_weight = sum(base_weights.values())
        return {k: v / total_weight for k, v in base_weights.items()}
    
    def get_adjusted_complexity_scores(self, task_complexity: List[float]) -> List[float]:
        """
        Get complexity scores adjusted by learned weights.
        
        Args:
            task_complexity: Original 5-dimensional complexity scores
            
        Returns:
            Adjusted complexity scores
        """
        
        complexity_array = np.array(task_complexity)
        adjusted_scores = complexity_array * self.complexity_weights
        
        # Normalize to maintain same scale
        if np.sum(complexity_array) > 0:
            scale_factor = np.sum(complexity_array) / np.sum(adjusted_scores)
            adjusted_scores *= scale_factor
        
        return adjusted_scores.tolist()
    
    def get_skill_importance_factor(self, skill: str) -> float:
        """Get learned importance factor for a specific skill."""
        return self.skill_importance_factors.get(skill, 1.0)
    
    def get_developer_performance_prediction(
        self,
        developer_id: int,
        task_complexity: List[float]
    ) -> Dict[str, float]:
        """
        Predict developer performance on a task based on learned patterns.
        
        Args:
            developer_id: Developer ID
            task_complexity: 5-dimensional task complexity
            
        Returns:
            Performance prediction with confidence
        """
        
        if developer_id not in self.developer_preference_models:
            return {'predicted_performance': 0.7, 'confidence': 0.3}
        
        model = self.developer_preference_models[developer_id]
        performance_history = model['performance_by_complexity']
        
        if not performance_history:
            return {'predicted_performance': 0.7, 'confidence': 0.3}
        
        # Find similar complexity tasks in history
        task_complexity_array = np.array(task_complexity)
        similarities = []
        outcomes = []
        
        for record in performance_history[-20:]:  # Use recent history
            record_complexity = np.array(record['complexity'])
            similarity = 1.0 - np.linalg.norm(task_complexity_array - record_complexity)
            similarities.append(max(0.0, similarity))
            outcomes.append(record['productivity'])
        
        if not similarities:
            return {'predicted_performance': 0.7, 'confidence': 0.3}
        
        # Weighted average of outcomes by similarity
        similarities = np.array(similarities)
        outcomes = np.array(outcomes)
        
        if np.sum(similarities) > 0:
            predicted_performance = np.average(outcomes, weights=similarities)
            confidence = min(1.0, np.sum(similarities) / len(similarities))
        else:
            predicted_performance = np.mean(outcomes)
            confidence = 0.3
        
        return {
            'predicted_performance': predicted_performance,
            'confidence': confidence
        }
    
    def get_learning_analytics(self) -> Dict[str, Any]:
        """Get analytics about the learning system performance."""
        
        analytics = {
            'total_assignments_learned': len(self.prediction_accuracy_history),
            'current_learning_rate': self.learning_rate,
            'complexity_weights': self.complexity_weights.tolist(),
            'developers_modeled': len(self.developer_preference_models),
            'skills_tracked': len(self.skill_importance_factors)
        }
        
        # Recent accuracy metrics
        if self.prediction_accuracy_history:
            recent_accuracy = [p['accuracy'] for p in self.prediction_accuracy_history[-20:]]
            analytics['recent_prediction_accuracy'] = np.mean(recent_accuracy)
            analytics['prediction_variance'] = np.var(recent_accuracy)
        
        # Skill importance insights
        if self.skill_importance_factors:
            analytics['top_important_skills'] = sorted(
                self.skill_importance_factors.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
        
        return analytics
    
    async def reset_learning_for_developer(self, developer_id: int) -> None:
        """Reset learning models for a specific developer (e.g., after major role change)."""
        
        if developer_id in self.developer_preference_models:
            del self.developer_preference_models[developer_id]
        
        # Remove from assignment success patterns
        if developer_id in self.assignment_success_patterns:
            del self.assignment_success_patterns[developer_id]
        
        logger.info(f"Reset learning models for developer {developer_id}")
    
    async def export_learning_models(self) -> Dict[str, Any]:
        """Export learning models for backup or analysis."""
        
        return {
            'complexity_weights': self.complexity_weights.tolist(),
            'skill_importance_factors': dict(self.skill_importance_factors),
            'developer_preference_models': {
                str(dev_id): {
                    'preferred_complexity_factors': model['preferred_complexity_factors'].tolist(),
                    'collaboration_preference': model['collaboration_preference'],
                    'learning_preference': model['learning_preference'],
                    'workload_tolerance': model['workload_tolerance']
                }
                for dev_id, model in self.developer_preference_models.items()
            },
            'learning_rate': self.learning_rate,
            'export_timestamp': datetime.now().isoformat()
        }
    
    async def import_learning_models(self, models_data: Dict[str, Any]) -> bool:
        """Import previously exported learning models."""
        
        try:
            self.complexity_weights = np.array(models_data['complexity_weights'])
            self.skill_importance_factors = defaultdict(lambda: 1.0, models_data['skill_importance_factors'])
            self.learning_rate = models_data['learning_rate']
            
            # Import developer models
            for dev_id_str, model_data in models_data['developer_preference_models'].items():
                dev_id = int(dev_id_str)
                self.developer_preference_models[dev_id] = {
                    'preferred_complexity_factors': np.array(model_data['preferred_complexity_factors']),
                    'performance_by_complexity': [],
                    'collaboration_preference': model_data['collaboration_preference'],
                    'learning_preference': model_data['learning_preference'],
                    'workload_tolerance': model_data['workload_tolerance']
                }
            
            logger.info("Successfully imported learning models")
            return True
            
        except Exception as e:
            logger.error(f"Failed to import learning models: {e}")
            return False
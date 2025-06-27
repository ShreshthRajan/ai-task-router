"""
Multi-Objective Assignment Optimizer

This module implements sophisticated assignment optimization algorithms that balance
multiple objectives: productivity, skill development, workload balance, collaboration,
and business impact.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import asyncio
from sqlalchemy.orm import Session
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cosine
import json
import logging

from models.database import Developer, Task, TaskAssignment, ExpertiseSnapshot
from models.schemas import TaskDeveloperMatch, OptimizationRequest, OptimizationResult, Assignment, AssignmentResult
logger = logging.getLogger(__name__)

@dataclass
class AssignmentScore:
     """Comprehensive scoring for task-developer assignments."""
     overall_score: float
     skill_match_score: float
     complexity_fit_score: float
     learning_potential_score: float
     workload_impact_score: float
     collaboration_score: float
     business_impact_score: float
     confidence: float
     reasoning: str
     risk_factors: List[str]

@dataclass
class DeveloperCapacity:
     """Developer workload and capacity information."""
     developer_id: int
     current_workload: float  # 0.0 to 1.0 (percentage of capacity)
     max_concurrent_tasks: int
     skill_vector: np.ndarray
     availability_score: float
     recent_performance: float
     preferred_complexity_range: Tuple[float, float]

@dataclass
class TaskProfile:
     """Enhanced task profile for assignment optimization."""
     task_id: int
     complexity_vector: np.ndarray  # 5-dimensional complexity
     required_skills: Dict[str, float]
     estimated_hours: float
     priority_weight: float
     collaboration_requirements: float
     learning_opportunities: float
     business_impact: float
     deadline_pressure: float

class AssignmentOptimizer:
     """Advanced assignment optimization engine using multi-objective algorithms."""
     
     def __init__(self):
         self.skill_vector_dim = 768
         self.complexity_dimensions = 5
         self.assignment_history = []
         
         # Optimization weights - can be customized per team
         self.default_weights = {
             'productivity': 0.35,
             'skill_development': 0.25, 
             'workload_balance': 0.20,
             'collaboration': 0.10,
             'business_impact': 0.10
         }
         
         # Performance tracking
         self.assignment_outcomes = {}
         
     async def optimize_assignments(
        self,
        db: Session,
        task_ids: List[int],
        developer_ids: Optional[List[int]] = None,
        optimization_objectives: Optional[Dict[str, float]] = None,
        constraints: Optional[Dict[str, Any]] = None
    ) -> OptimizationResult:
        """
        Main optimization method that finds optimal task-developer assignments.
        
        Args:
            db: Database session
            task_ids: List of task IDs to assign
            developer_ids: Optional list of developer IDs (if None, considers all)
            optimization_objectives: Custom objective weights
            constraints: Assignment constraints
            
        Returns:
            OptimizationResult with optimal assignments and metadata
        """
        start_time = datetime.now()
        
        try:
            # Load and prepare data
            tasks = self._load_tasks(db, task_ids)
            developers = self._load_developers(db, developer_ids)
            
            if not tasks or not developers:
                raise ValueError("No valid tasks or developers found")
            
            # Build optimization matrices
            task_profiles = await self._build_task_profiles(db, tasks)
            developer_capacities = await self._build_developer_capacities(db, developers)
            
            # Calculate assignment scores matrix
            score_matrix = await self._calculate_assignment_matrix(
                task_profiles, developer_capacities, optimization_objectives or self.default_weights
            )
            
            # Apply constraints
            if constraints:
                score_matrix = self._apply_constraints(score_matrix, constraints, task_profiles, developer_capacities)
            
            # Perform optimization
            optimal_assignments = self._optimize_assignments_hungarian(score_matrix, task_profiles, developer_capacities)
            
            # Generate alternative solutions
            alternatives = await self._generate_alternatives(score_matrix, task_profiles, developer_capacities, optimal_assignments)
            
            # Calculate objective scores
            objective_scores = self._calculate_objective_scores(optimal_assignments, task_profiles, developer_capacities)
            
            # Create assignment records using the correct schema
            assignment_records = []
            for assignment in optimal_assignments:
                assignment_record = AssignmentResult(
                    task_id=assignment['task_id'],
                    developer_id=assignment['developer_id'],
                    confidence_score=assignment.get('confidence', assignment.get('score', 0.5)),
                    reasoning=assignment['reasoning'],
                    status='suggested'
                )
                assignment_records.append(assignment_record)
            
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return OptimizationResult(
                assignments=assignment_records,
                objective_scores=objective_scores,
                alternative_solutions=alternatives,
                optimization_time_ms=processing_time
            )
            
        except Exception as e:
            logger.error(f"Assignment optimization failed: {e}")
            raise
     
     async def calculate_task_developer_matches(
         self,
         db: Session,
         task_id: int,
         developer_ids: Optional[List[int]] = None,
         max_matches: int = 5
     ) -> List[TaskDeveloperMatch]:
         """
         Calculate compatibility scores for a single task with multiple developers.
         
         Args:
             db: Database session
             task_id: Task ID to match
             developer_ids: Optional developer ID filter
             max_matches: Maximum number of matches to return
             
         Returns:
             List of TaskDeveloperMatch objects sorted by score
         """
         try:
             # Load task and developers
             task = db.query(Task).filter(Task.id == task_id).first()
             if not task:
                 raise ValueError(f"Task {task_id} not found")
             
             developers = self._load_developers(db, developer_ids)
             if not developers:
                 raise ValueError("No valid developers found")
             
             # Build profiles
             task_profile = await self._build_single_task_profile(db, task)
             developer_capacities = await self._build_developer_capacities(db, developers)
             
             # Calculate matches
             matches = []
             for dev_capacity in developer_capacities:
                 score = await self._calculate_single_assignment_score(
                     task_profile, dev_capacity, self.default_weights
                 )
                 
                 match = TaskDeveloperMatch(
                     task_id=task_id,
                     developer_id=dev_capacity.developer_id,
                     overall_score=score.overall_score,
                     skill_match_score=score.skill_match_score,
                     complexity_fit_score=score.complexity_fit_score,
                     learning_potential_score=score.learning_potential_score,
                     workload_impact_score=score.workload_impact_score,
                     reasoning=score.reasoning,
                     risk_factors=score.risk_factors
                 )
                 matches.append(match)
             
             # Sort by overall score and return top matches
             matches.sort(key=lambda x: x.overall_score, reverse=True)
             return matches[:max_matches]
             
         except Exception as e:
             logger.error(f"Task-developer matching failed: {e}")
             raise
     
     def _load_tasks(self, db: Session, task_ids: List[int]) -> List[Task]:
         """Load tasks from database with complexity analysis."""
         return db.query(Task).filter(Task.id.in_(task_ids)).all()
     
     def _load_developers(self, db: Session, developer_ids: Optional[List[int]]) -> List[Developer]:
         """Load developers from database with skill vectors."""
         query = db.query(Developer)
         if developer_ids:
             query = query.filter(Developer.id.in_(developer_ids))
         return query.all()
     
     async def _build_task_profiles(self, db: Session, tasks: List[Task]) -> List[TaskProfile]:
         """Build enhanced task profiles for optimization."""
         profiles = []
         
         for task in tasks:
             # Build complexity vector [technical, domain, collaboration, learning, business]
             complexity_vector = np.array([
                 task.technical_complexity or 0.5,
                 task.domain_difficulty or 0.5,
                 task.collaboration_requirements or 0.3,
                 task.learning_opportunities or 0.4,
                 task.business_impact or 0.5
             ])
             
             # Parse required skills
             required_skills = json.loads(task.required_skills) if task.required_skills else {}
             
             # Calculate priority weight based on labels and business impact
             priority_weight = self._calculate_priority_weight(task)
             
             # Calculate deadline pressure
             deadline_pressure = self._calculate_deadline_pressure(task)
             
             profile = TaskProfile(
                 task_id=task.id,
                 complexity_vector=complexity_vector,
                 required_skills=required_skills,
                 estimated_hours=task.estimated_hours or 8.0,
                 priority_weight=priority_weight,
                 collaboration_requirements=task.collaboration_requirements or 0.3,
                 learning_opportunities=task.learning_opportunities or 0.4,
                 business_impact=task.business_impact or 0.5,
                 deadline_pressure=deadline_pressure
             )
             profiles.append(profile)
         
         return profiles
     
     async def _build_single_task_profile(self, db: Session, task: Task) -> TaskProfile:
         """Build profile for a single task."""
         profiles = await self._build_task_profiles(db, [task])
         return profiles[0]
     
     async def _build_developer_capacities(self, db: Session, developers: List[Developer]) -> List[DeveloperCapacity]:
         """Build developer capacity profiles for optimization."""
         capacities = []
         
         for developer in developers:
             # Get current workload from active assignments
             current_workload = self._calculate_current_workload(db, developer.id)
             
             # Parse skill vector
             skill_vector = np.array(json.loads(developer.skill_vector)) if developer.skill_vector else np.zeros(self.skill_vector_dim)
             
             # Calculate availability score
             availability_score = max(0.0, 1.0 - current_workload)
             
             # Get recent performance metrics
             recent_performance = self._get_recent_performance(db, developer.id)
             
             # Determine preferred complexity range based on history
             complexity_range = self._get_preferred_complexity_range(db, developer.id)
             
             capacity = DeveloperCapacity(
                 developer_id=developer.id,
                 current_workload=current_workload,
                 max_concurrent_tasks=5,  # Can be made configurable
                 skill_vector=skill_vector,
                 availability_score=availability_score,
                 recent_performance=recent_performance,
                 preferred_complexity_range=complexity_range
             )
             capacities.append(capacity)
         
         return capacities
     
     async def _calculate_assignment_matrix(
         self,
         task_profiles: List[TaskProfile],
         developer_capacities: List[DeveloperCapacity],
         objective_weights: Dict[str, float]
     ) -> np.ndarray:
         """Calculate assignment score matrix for all task-developer combinations."""
         num_tasks = len(task_profiles)
         num_developers = len(developer_capacities)
         
         score_matrix = np.zeros((num_tasks, num_developers))
         
         for i, task_profile in enumerate(task_profiles):
             for j, dev_capacity in enumerate(developer_capacities):
                 score = await self._calculate_single_assignment_score(
                     task_profile, dev_capacity, objective_weights
                 )
                 score_matrix[i, j] = score.overall_score
         
         return score_matrix
     
     async def _calculate_single_assignment_score(
         self,
         task_profile: TaskProfile,
         dev_capacity: DeveloperCapacity,
         objective_weights: Dict[str, float]
     ) -> AssignmentScore:
         """Calculate comprehensive assignment score for task-developer pair."""
         
         # 1. Skill Match Score
         skill_match = self._calculate_skill_match_score(task_profile, dev_capacity)
         
         # 2. Complexity Fit Score
         complexity_fit = self._calculate_complexity_fit_score(task_profile, dev_capacity)
         
         # 3. Learning Potential Score
         learning_potential = self._calculate_learning_potential_score(task_profile, dev_capacity)
         
         # 4. Workload Impact Score
         workload_impact = self._calculate_workload_impact_score(task_profile, dev_capacity)
         
         # 5. Collaboration Score
         collaboration_score = self._calculate_collaboration_score(task_profile, dev_capacity)
         
         # 6. Business Impact Score
         business_impact = self._calculate_business_impact_score(task_profile, dev_capacity)
         
         # Calculate weighted overall score
         overall_score = (
             objective_weights.get('productivity', 0.35) * (skill_match * 0.6 + complexity_fit * 0.4) +
             objective_weights.get('skill_development', 0.25) * learning_potential +
             objective_weights.get('workload_balance', 0.20) * workload_impact +
             objective_weights.get('collaboration', 0.10) * collaboration_score +
             objective_weights.get('business_impact', 0.10) * business_impact
         )
         
         # Add semantic code quality bonus
         overall_score += 0.15 * dev_capacity.skill_vector[353]    # reward clean, idiomatic code
         
         # Generate reasoning and risk factors
         reasoning, risk_factors = self._generate_assignment_reasoning(
             task_profile, dev_capacity, skill_match, complexity_fit, learning_potential
         )
         
         # Calculate confidence based on score components
         confidence = self._calculate_assignment_confidence(
             skill_match, complexity_fit, learning_potential, workload_impact
         )
         
         return AssignmentScore(
             overall_score=overall_score,
             skill_match_score=skill_match,
             complexity_fit_score=complexity_fit,
             learning_potential_score=learning_potential,
             workload_impact_score=workload_impact,
             collaboration_score=collaboration_score,
             business_impact_score=business_impact,
             confidence=confidence,
             reasoning=reasoning,
             risk_factors=risk_factors
         )
     
     def _calculate_skill_match_score(self, task_profile: TaskProfile, dev_capacity: DeveloperCapacity) -> float:
         """Calculate how well developer skills match task requirements."""
         if not task_profile.required_skills:
             return 0.7  # Default score when no specific skills required
         
         total_weight = sum(task_profile.required_skills.values())
         if total_weight == 0:
             return 0.7
         
         # Calculate weighted skill match using cosine similarity for semantic skills
         # and direct matching for specific technologies
         skill_scores = []
         
         for skill, importance in task_profile.required_skills.items():
             # This is simplified - in practice, would use semantic matching
             # against the 768-dimensional skill vector
             skill_level = self._get_developer_skill_level(dev_capacity, skill)
             skill_scores.append(skill_level * importance)
         
         return min(1.0, sum(skill_scores) / total_weight)
     
     def _calculate_complexity_fit_score(self, task_profile: TaskProfile, dev_capacity: DeveloperCapacity) -> float:
         """Calculate how well task complexity fits developer's capabilities."""
         avg_complexity = np.mean(task_profile.complexity_vector)
         
         # Check if complexity is within developer's preferred range
         min_complexity, max_complexity = dev_capacity.preferred_complexity_range
         
         if min_complexity <= avg_complexity <= max_complexity:
             # Perfect fit
             return 1.0
         elif avg_complexity < min_complexity:
             # Task might be too simple
             gap = min_complexity - avg_complexity
             return max(0.3, 1.0 - gap * 2)  # Gradual penalty
         else:
             # Task might be too complex
             gap = avg_complexity - max_complexity
             return max(0.1, 1.0 - gap * 3)  # Steeper penalty for overcomplex tasks
     
     def _calculate_learning_potential_score(self, task_profile: TaskProfile, dev_capacity: DeveloperCapacity) -> float:
         """Calculate learning and skill development potential."""
         base_learning_score = task_profile.learning_opportunities
         
         # Boost score if task complexity is slightly above comfort zone
         avg_complexity = np.mean(task_profile.complexity_vector)
         _, max_comfort = dev_capacity.preferred_complexity_range
         
         if max_comfort < avg_complexity <= max_comfort + 0.2:
             # Sweet spot for learning
             learning_boost = 1.5
         elif avg_complexity > max_comfort + 0.2:
             # Too challenging, might be overwhelming
             learning_boost = 0.5
         else:
             # Within comfort zone, moderate learning
             learning_boost = 1.0
         
         return min(1.0, base_learning_score * learning_boost)
     
     def _calculate_workload_impact_score(self, task_profile: TaskProfile, dev_capacity: DeveloperCapacity) -> float:
         """Calculate impact on developer's workload balance."""
         # Higher score means better workload balance
         if dev_capacity.current_workload >= 0.9:
             return 0.1  # Developer is overloaded
         elif dev_capacity.current_workload <= 0.3:
             return 1.0  # Developer has good capacity
         else:
             # Linear interpolation between 0.3 and 0.9
             return 1.0 - (dev_capacity.current_workload - 0.3) / 0.6
     
     def _calculate_collaboration_score(self, task_profile: TaskProfile, dev_capacity: DeveloperCapacity) -> float:
         """Calculate collaboration effectiveness score."""
         # This would use collaboration history and team dynamics
         # Simplified implementation
         if task_profile.collaboration_requirements > 0.7:
             # High collaboration task
             return min(1.0, dev_capacity.recent_performance * 1.2)
         else:
             # Independent task
             return 0.8
     
     def _calculate_business_impact_score(self, task_profile: TaskProfile, dev_capacity: DeveloperCapacity) -> float:
         """Calculate business impact alignment score."""
         # Prefer higher-performing developers for high business impact tasks
         if task_profile.business_impact > 0.7:
             return dev_capacity.recent_performance
         else:
             return 0.8  # Standard score for lower impact tasks
     
     def _apply_constraints(
         self,
         score_matrix: np.ndarray,
         constraints: Dict[str, Any],
         task_profiles: List[TaskProfile],
         developer_capacities: List[DeveloperCapacity]
     ) -> np.ndarray:
         """Apply hard constraints to the score matrix."""
         constrained_matrix = score_matrix.copy()
         
         # Workload constraints
         max_workload = constraints.get('max_workload_per_developer', 0.9)
         for j, dev_capacity in enumerate(developer_capacities):
             if dev_capacity.current_workload >= max_workload:
                 constrained_matrix[:, j] = 0  # Block all assignments
         
         # Skill requirements constraints
         if constraints.get('enforce_skill_requirements', True):
             for i, task_profile in enumerate(task_profiles):
                 for j, dev_capacity in enumerate(developer_capacities):
                     if not self._meets_minimum_skill_requirements(task_profile, dev_capacity):
                         constrained_matrix[i, j] = 0
         
         return constrained_matrix
     
     def _optimize_assignments_hungarian(
         self,
         score_matrix: np.ndarray,
         task_profiles: List[TaskProfile],
         developer_capacities: List[DeveloperCapacity]
     ) -> List[Dict[str, Any]]:
         """Use Hungarian algorithm for optimal assignment."""
         # Convert maximization to minimization problem
         cost_matrix = 1.0 - score_matrix
         
         # Handle case where we have more tasks than developers or vice versa
         num_tasks, num_developers = cost_matrix.shape
         
         if num_tasks != num_developers:
             # Pad with high cost dummy assignments
             max_dim = max(num_tasks, num_developers)
             padded_matrix = np.ones((max_dim, max_dim)) * 10  # High cost
             padded_matrix[:num_tasks, :num_developers] = cost_matrix
             cost_matrix = padded_matrix
         
         # Solve assignment problem
         row_indices, col_indices = linear_sum_assignment(cost_matrix)
         
         assignments = []
         for task_idx, dev_idx in zip(row_indices, col_indices):
             # Skip dummy assignments
             if task_idx >= len(task_profiles) or dev_idx >= len(developer_capacities):
                 continue
             
             # Skip zero-score assignments (blocked by constraints)
             if score_matrix[task_idx, dev_idx] == 0:
                 continue
             
             task_profile = task_profiles[task_idx]
             dev_capacity = developer_capacities[dev_idx]
             assignment_score = score_matrix[task_idx, dev_idx]
             
             assignments.append({
                 'task_id': task_profile.task_id,
                 'developer_id': dev_capacity.developer_id,
                 'score': assignment_score,
                 'confidence': min(1.0, assignment_score * 1.2),
                 'reasoning': f"Optimal assignment with score {assignment_score:.3f}"
             })
         
         return assignments
     
     async def _generate_alternatives(
         self,
         score_matrix: np.ndarray,
         task_profiles: List[TaskProfile],
         developer_capacities: List[DeveloperCapacity],
         optimal_assignments: List[Dict[str, Any]]
     ) -> List[Dict[str, Any]]:
         """Generate alternative assignment solutions."""
         alternatives = []
         
         # Generate greedy alternative (always pick highest score)
         greedy_assignment = self._generate_greedy_assignment(score_matrix, task_profiles, developer_capacities)
         if greedy_assignment != optimal_assignments:
             alternatives.append({
                 'strategy': 'greedy',
                 'assignments': greedy_assignment,
                 'description': 'Greedy assignment (highest scores first)'
             })
         
         # Generate balanced alternative (prioritize workload balance)
         balanced_assignment = self._generate_balanced_assignment(score_matrix, task_profiles, developer_capacities)
         alternatives.append({
             'strategy': 'balanced',
             'assignments': balanced_assignment,
             'description': 'Workload-balanced assignment'
         })
         
         return alternatives[:3]  # Limit to 3 alternatives
     
     def _generate_greedy_assignment(
         self,
         score_matrix: np.ndarray,
         task_profiles: List[TaskProfile],
         developer_capacities: List[DeveloperCapacity]
     ) -> List[Dict[str, Any]]:
         """Generate greedy assignment solution."""
         assignments = []
         available_tasks = set(range(len(task_profiles)))
         available_developers = set(range(len(developer_capacities)))
         
         while available_tasks and available_developers:
             # Find highest score among available combinations
             best_score = -1
             best_task_idx = -1
             best_dev_idx = -1
             
             for task_idx in available_tasks:
                 for dev_idx in available_developers:
                     if score_matrix[task_idx, dev_idx] > best_score:
                         best_score = score_matrix[task_idx, dev_idx]
                         best_task_idx = task_idx
                         best_dev_idx = dev_idx
             
             if best_score <= 0:
                 break
             
             # Make assignment
             assignments.append({
                 'task_id': task_profiles[best_task_idx].task_id,
                 'developer_id': developer_capacities[best_dev_idx].developer_id,
                 'score': best_score,
                 'confidence': min(1.0, best_score * 1.1),
                 'reasoning': f"Greedy selection with score {best_score:.3f}"
             })
             
             available_tasks.remove(best_task_idx)
             available_developers.remove(best_dev_idx)
         
         return assignments
     
     def _generate_balanced_assignment(
         self,
         score_matrix: np.ndarray,
         task_profiles: List[TaskProfile],
         developer_capacities: List[DeveloperCapacity]
     ) -> List[Dict[str, Any]]:
         """Generate workload-balanced assignment solution."""
         # Sort developers by current workload (ascending)
         dev_workloads = [(i, dev.current_workload) for i, dev in enumerate(developer_capacities)]
         dev_workloads.sort(key=lambda x: x[1])
         
         assignments = []
         available_tasks = list(range(len(task_profiles)))
         
         for dev_idx, _ in dev_workloads:
             if not available_tasks:
                 break
             
             # Find best task for this developer
             best_task_idx = max(available_tasks, key=lambda t: score_matrix[t, dev_idx])
             
             if score_matrix[best_task_idx, dev_idx] > 0:
                 assignments.append({
                     'task_id': task_profiles[best_task_idx].task_id,
                     'developer_id': developer_capacities[dev_idx].developer_id,
                     'score': score_matrix[best_task_idx, dev_idx],
                     'confidence': min(1.0, score_matrix[best_task_idx, dev_idx] * 1.1),
                     'reasoning': f"Balanced assignment with score {score_matrix[best_task_idx, dev_idx]:.3f}"
                 })
                 available_tasks.remove(best_task_idx)
         
         return assignments
     
     def _calculate_objective_scores(
         self,
         assignments: List[Dict[str, Any]],
         task_profiles: List[TaskProfile],
         developer_capacities: List[DeveloperCapacity]
     ) -> Dict[str, float]:
         """Calculate objective function scores for the assignment solution."""
         if not assignments:
             return {'productivity': 0.0, 'skill_development': 0.0, 'workload_balance': 0.0}
         
         # Productivity score (average assignment scores)
         productivity = np.mean([a['score'] for a in assignments])
         
         # Skill development score (based on learning opportunities)
         skill_development = 0.0
         for assignment in assignments:
             task_profile = next(t for t in task_profiles if t.task_id == assignment['task_id'])
             skill_development += task_profile.learning_opportunities
         skill_development /= len(assignments)
         
         # Workload balance score (standard deviation of workloads)
         assigned_devs = [a['developer_id'] for a in assignments]
         dev_workloads = []
         for dev_capacity in developer_capacities:
             workload = dev_capacity.current_workload
             if dev_capacity.developer_id in assigned_devs:
                 # Add estimated workload from new assignment
                 task_hours = next(t.estimated_hours for t in task_profiles 
                                 if t.task_id == next(a['task_id'] for a in assignments 
                                                    if a['developer_id'] == dev_capacity.developer_id))
                 workload += task_hours / 40.0  # Assuming 40-hour work week
             dev_workloads.append(workload)
         
         workload_balance = 1.0 - (np.std(dev_workloads) / np.mean(dev_workloads)) if np.mean(dev_workloads) > 0 else 1.0
         workload_balance = max(0.0, min(1.0, workload_balance))
         
         return {
             'productivity': productivity,
             'skill_development': skill_development,
             'workload_balance': workload_balance
         }
     
     # Helper methods
     def _calculate_priority_weight(self, task: Task) -> float:
         """Calculate priority weight based on task metadata."""
         priority_mapping = {'low': 0.3, 'medium': 0.6, 'high': 0.8, 'critical': 1.0}
         return priority_mapping.get(task.priority, 0.6)
     
     def _calculate_deadline_pressure(self, task: Task) -> float:
         """Calculate deadline pressure score."""
         # This would calculate based on deadline vs current date
         # Simplified implementation
         return 0.5
     
     def _calculate_current_workload(self, db: Session, developer_id: int) -> float:
         """Calculate developer's current workload from active assignments."""
         active_assignments = db.query(TaskAssignment).filter(
             TaskAssignment.developer_id == developer_id,
             TaskAssignment.status.in_(['suggested', 'accepted', 'in_progress'])
         ).all()
         
         total_hours = sum(a.task.estimated_hours or 8.0 for a in active_assignments if a.task)
         return min(1.0, total_hours / 40.0)  # Normalize to weekly capacity
     
     def _get_recent_performance(self, db: Session, developer_id: int) -> float:
         """Get developer's recent performance score."""
         recent_assignments = db.query(TaskAssignment).filter(
             TaskAssignment.developer_id == developer_id,
             TaskAssignment.status == 'completed',
             TaskAssignment.completed_at >= datetime.now() - timedelta(days=90)
         ).all()
         
         if not recent_assignments:
             return 0.7  # Default score
         
         scores = [a.productivity_score for a in recent_assignments if a.productivity_score]
         return np.mean(scores) if scores else 0.7
     
     def _get_preferred_complexity_range(self, db: Session, developer_id: int) -> Tuple[float, float]:
        """Determine developer's preferred complexity range from assignment history."""
        completed_assignments = db.query(TaskAssignment).filter(
            TaskAssignment.developer_id == developer_id,
            TaskAssignment.status == 'completed',
            TaskAssignment.productivity_score.isnot(None)
        ).all()
        
        if not completed_assignments:
            return (0.3, 0.7)  # Default range for new developers
        
        # Get complexity scores for assignments with good outcomes
        good_assignments = [a for a in completed_assignments if a.productivity_score >= 0.7]
        
        if not good_assignments:
            return (0.2, 0.6)  # Conservative range if no good outcomes
        
        complexities = []
        for assignment in good_assignments:
            if assignment.task:
                avg_complexity = np.mean([
                    assignment.task.technical_complexity or 0.5,
                    assignment.task.domain_difficulty or 0.5,
                    assignment.task.collaboration_requirements or 0.3,
                    assignment.task.learning_opportunities or 0.4,
                    assignment.task.business_impact or 0.5
                ])
                complexities.append(avg_complexity)
        
        if complexities:
            min_complexity = max(0.0, np.percentile(complexities, 25) - 0.1)
            max_complexity = min(1.0, np.percentile(complexities, 75) + 0.1)
            return (min_complexity, max_complexity)
        
        return (0.3, 0.7)
    
     def _get_developer_skill_level(self, dev_capacity: DeveloperCapacity, skill: str) -> float:
        """Get developer's skill level for a specific skill."""
        # This is a simplified implementation
        # In practice, would use semantic matching against the 768-dimensional vector
        
        # Basic keyword matching for common skills
        skill_keywords = {
            'python': [0, 50, 100],  # Indices in skill vector for Python-related skills
            'javascript': [1, 51, 101],
            'react': [2, 52, 102],
            'database': [10, 60, 110],
            'api': [15, 65, 115],
            'machine_learning': [20, 70, 120]
        }
        
        skill_lower = skill.lower()
        if skill_lower in skill_keywords:
            indices = skill_keywords[skill_lower]
            # Take average of relevant vector components
            relevant_scores = [dev_capacity.skill_vector[i] for i in indices 
                             if i < len(dev_capacity.skill_vector)]
            return np.mean(relevant_scores) if relevant_scores else 0.5
        
        # Default to moderate skill level for unknown skills
        return 0.5
    
     def _meets_minimum_skill_requirements(self, task_profile: TaskProfile, dev_capacity: DeveloperCapacity) -> bool:
        """Check if developer meets minimum skill requirements for task."""
        if not task_profile.required_skills:
            return True
        
        # Check critical skills (importance > 0.8)
        critical_skills = {skill: importance for skill, importance in task_profile.required_skills.items() 
                          if importance > 0.8}
        
        for skill, importance in critical_skills.items():
            skill_level = self._get_developer_skill_level(dev_capacity, skill)
            if skill_level < 0.4:  # Minimum threshold for critical skills
                return False
        
        return True
    
     def _generate_assignment_reasoning(
        self,
        task_profile: TaskProfile,
        dev_capacity: DeveloperCapacity,
        skill_match: float,
        complexity_fit: float,
        learning_potential: float
    ) -> Tuple[str, List[str]]:
        """Generate human-readable reasoning for assignment decision."""
        
        reasons = []
        risk_factors = []
        
        # Skill match reasoning
        if skill_match > 0.8:
            reasons.append("Excellent skill match for task requirements")
        elif skill_match > 0.6:
            reasons.append("Good skill alignment with opportunities for growth")
        else:
            reasons.append("Some skill gaps present")
            risk_factors.append("Skill gap may impact delivery time")
        
        # Complexity fit reasoning
        if complexity_fit > 0.8:
            reasons.append("Task complexity well-suited to developer experience")
        elif complexity_fit < 0.4:
            avg_complexity = np.mean(task_profile.complexity_vector)
            if avg_complexity > dev_capacity.preferred_complexity_range[1]:
                reasons.append("Task complexity above comfort zone - growth opportunity")
                risk_factors.append("High complexity may cause delays")
            else:
                reasons.append("Task may be below optimal challenge level")
        
        # Learning potential reasoning
        if learning_potential > 0.7:
            reasons.append("Strong learning and skill development opportunities")
        
        # Workload reasoning
        if dev_capacity.current_workload > 0.8:
            risk_factors.append("Developer has high current workload")
        elif dev_capacity.current_workload < 0.3:
            reasons.append("Developer has good availability")
        
        # Performance reasoning
        if dev_capacity.recent_performance > 0.8:
            reasons.append("Strong recent performance track record")
        elif dev_capacity.recent_performance < 0.5:
            risk_factors.append("Recent performance below average")
        
        reasoning = "; ".join(reasons)
        return reasoning, risk_factors
    
     def _calculate_assignment_confidence(
        self,
        skill_match: float,
        complexity_fit: float,
        learning_potential: float,
        workload_impact: float
    ) -> float:
        """Calculate confidence score for assignment recommendation."""
        
        # Base confidence from core metrics
        base_confidence = (skill_match * 0.4 + complexity_fit * 0.3 + 
                          workload_impact * 0.2 + learning_potential * 0.1)
        
        # Penalty for extreme scores (too high or too low can indicate uncertainty)
        penalty = 0.0
        
        if skill_match < 0.3 or skill_match > 0.95:
            penalty += 0.1
        
        if complexity_fit < 0.2:
            penalty += 0.15
        
        if workload_impact < 0.3:
            penalty += 0.1
        
        confidence = max(0.1, min(1.0, base_confidence - penalty))
        return confidence
     
     async def update_assignment_outcome(
         self,
         db: Session,
         assignment_id: int,
         outcome_data: Dict[str, Any]
     ) -> bool:
         """Update assignment outcome for learning feedback."""
         try:
             assignment = db.query(TaskAssignment).filter(TaskAssignment.id == assignment_id).first()
             if not assignment:
                 return False
             
             # Update assignment record
             assignment.actual_hours = outcome_data.get('actual_hours')
             assignment.feedback_score = outcome_data.get('feedback_score')
             assignment.feedback_comments = outcome_data.get('feedback_comments')
             assignment.productivity_score = outcome_data.get('productivity_score')
             assignment.skill_development_score = outcome_data.get('skill_development_score')
             assignment.collaboration_effectiveness = outcome_data.get('collaboration_effectiveness')
             assignment.completed_at = datetime.now()
             assignment.status = 'completed'
             
             db.commit()
             
             # Store outcome for learning
             self.assignment_outcomes[assignment_id] = outcome_data
             
             logger.info(f"Updated assignment {assignment_id} outcome")
             return True
             
         except Exception as e:
             logger.error(f"Failed to update assignment outcome: {e}")
             db.rollback()
             return False

     def get_assignment_analytics(self, db: Session, developer_id: Optional[int] = None) -> Dict[str, Any]:
         """Get assignment analytics and performance metrics."""
         try:
             query = db.query(TaskAssignment).filter(TaskAssignment.status == 'completed')
             
             if developer_id:
                 query = query.filter(TaskAssignment.developer_id == developer_id)
             
             assignments = query.all()
             
             if not assignments:
                 return {'total_assignments': 0}
             
             # Calculate metrics
             productivity_scores = [a.productivity_score for a in assignments if a.productivity_score]
             skill_development_scores = [a.skill_development_score for a in assignments if a.skill_development_score]
             feedback_scores = [a.feedback_score for a in assignments if a.feedback_score]
             
             actual_vs_estimated = []
             for a in assignments:
                 if a.actual_hours and a.task and a.task.estimated_hours:
                     ratio = a.actual_hours / a.task.estimated_hours
                     actual_vs_estimated.append(ratio)
             
             analytics = {
                 'total_assignments': len(assignments),
                 'avg_productivity_score': np.mean(productivity_scores) if productivity_scores else 0,
                 'avg_skill_development_score': np.mean(skill_development_scores) if skill_development_scores else 0,
                 'avg_feedback_score': np.mean(feedback_scores) if feedback_scores else 0,
                 'time_estimation_accuracy': 1.0 / np.mean(actual_vs_estimated) if actual_vs_estimated else 1.0,
                 'assignment_success_rate': len([a for a in assignments if a.productivity_score and a.productivity_score >= 0.7]) / len(assignments)
             }
             
             return analytics
             
         except Exception as e:
             logger.error(f"Failed to get assignment analytics: {e}")
             return {'error': str(e)}
     
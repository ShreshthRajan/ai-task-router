# src/api/assignments.py
import logging
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any
import asyncio

from ..models.database import get_db, TaskAssignment, Task, Developer
from ..models.schemas import (
    Assignment, AssignmentCreate, AssignmentUpdate,
    TaskMatchingRequest, TaskMatchingResponse, TaskDeveloperMatch,
    OptimizationRequest, OptimizationResult, AssignmentResult,
    SuccessResponse, ErrorResponse
)
from ..core.assignment_engine.optimizer import AssignmentOptimizer
from ..core.assignment_engine.learning_automata import LearningAutomata

router = APIRouter()
logger = logging.getLogger(__name__)  # Add this line

# Initialize assignment components
assignment_optimizer = AssignmentOptimizer()
learning_automata = LearningAutomata()

@router.post("/assignments/", response_model=Assignment)
async def create_assignment(
    assignment: AssignmentCreate,
    db: Session = Depends(get_db)
):
    """Create a new task assignment."""
    try:
        # Verify task and developer exist
        task = db.query(Task).filter(Task.id == assignment.task_id).first()
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        
        developer = db.query(Developer).filter(Developer.id == assignment.developer_id).first()
        if not developer:
            raise HTTPException(status_code=404, detail="Developer not found")
        
        # Create assignment record
        db_assignment = TaskAssignment(
            task_id=assignment.task_id,
            developer_id=assignment.developer_id,
            confidence_score=assignment.confidence_score,
            reasoning=assignment.reasoning,
            status="suggested"
        )
        
        db.add(db_assignment)
        db.commit()
        db.refresh(db_assignment)
        
        return db_assignment
        
    except Exception as e:
       db.rollback()
       raise HTTPException(status_code=500, detail=f"Failed to create assignment: {str(e)}")

@router.get("/assignments/", response_model=List[Assignment])
async def list_assignments(
   developer_id: Optional[int] = Query(None),
   task_id: Optional[int] = Query(None),
   status: Optional[str] = Query(None),
   limit: int = Query(50, le=100),
   offset: int = Query(0),
   db: Session = Depends(get_db)
):
   """List assignments with optional filtering."""
   try:
       query = db.query(TaskAssignment)
       
       if developer_id:
           query = query.filter(TaskAssignment.developer_id == developer_id)
       
       if task_id:
           query = query.filter(TaskAssignment.task_id == task_id)
       
       if status:
           query = query.filter(TaskAssignment.status == status)
       
       assignments = query.offset(offset).limit(limit).all()
       return assignments
       
   except Exception as e:
       raise HTTPException(status_code=500, detail=f"Failed to list assignments: {str(e)}")

@router.get("/assignments/{assignment_id}", response_model=Assignment)
async def get_assignment(assignment_id: int, db: Session = Depends(get_db)):
   """Get a specific assignment by ID."""
   assignment = db.query(TaskAssignment).filter(TaskAssignment.id == assignment_id).first()
   if not assignment:
       raise HTTPException(status_code=404, detail="Assignment not found")
   return assignment

@router.put("/assignments/{assignment_id}", response_model=Assignment)
async def update_assignment(
   assignment_id: int,
   assignment_update: AssignmentUpdate,
   db: Session = Depends(get_db)
):
   """Update an assignment."""
   try:
       assignment = db.query(TaskAssignment).filter(TaskAssignment.id == assignment_id).first()
       if not assignment:
           raise HTTPException(status_code=404, detail="Assignment not found")
       
       # Update fields if provided
       update_data = assignment_update.dict(exclude_unset=True)
       for field, value in update_data.items():
           setattr(assignment, field, value)
       
       db.commit()
       db.refresh(assignment)
       
       # If assignment is completed, trigger learning
       if assignment.status == "completed" and assignment.productivity_score is not None:
           outcome_metrics = {
               'productivity_score': assignment.productivity_score,
               'skill_development_score': assignment.skill_development_score,
               'collaboration_effectiveness': assignment.collaboration_effectiveness,
               'feedback_score': assignment.feedback_score
           }
           
           # Run learning in background
           asyncio.create_task(
               learning_automata.learn_from_assignment_outcome(db, assignment_id, outcome_metrics)
           )
       
       return assignment
       
   except Exception as e:
       db.rollback()
       raise HTTPException(status_code=500, detail=f"Failed to update assignment: {str(e)}")

@router.delete("/assignments/{assignment_id}")
async def delete_assignment(assignment_id: int, db: Session = Depends(get_db)):
   """Delete an assignment."""
   try:
       assignment = db.query(TaskAssignment).filter(TaskAssignment.id == assignment_id).first()
       if not assignment:
           raise HTTPException(status_code=404, detail="Assignment not found")
       
       db.delete(assignment)
       db.commit()
       
       return SuccessResponse(message="Assignment deleted successfully")
       
   except Exception as e:
       db.rollback()
       raise HTTPException(status_code=500, detail=f"Failed to delete assignment: {str(e)}")

@router.post("/assignments/optimize", response_model=OptimizationResult)
async def optimize_assignments(
    request: OptimizationRequest,
    db: Session = Depends(get_db)
):
    """
    Optimize task assignments using multi-objective optimization.
    
    This endpoint finds the optimal assignment of tasks to developers
    considering multiple objectives like productivity, skill development,
    and workload balance.
    """
    try:
        if not request.task_ids:
            raise HTTPException(status_code=400, detail="No task IDs provided")
        
        if not request.developer_ids:
            raise HTTPException(status_code=400, detail="No developer IDs provided")
        
        # Create custom objective weights if provided
        objective_weights = None
        if hasattr(request, 'objectives') and request.objectives:
            objective_mapping = {
                'productivity': 0.35,
                'skill_development': 0.25,
                'workload_balance': 0.20,
                'collaboration': 0.10,
                'business_impact': 0.10
            }
            
            total_weight = 0.0
            objective_weights = {}
            for objective in request.objectives:
                if objective in objective_mapping:
                    weight = objective_mapping[objective]
                    objective_weights[objective] = weight
                    total_weight += weight
            
            # Normalize weights
            if total_weight > 0:
                objective_weights = {k: v/total_weight for k, v in objective_weights.items()}
        
        # Perform optimization
        result = await assignment_optimizer.optimize_assignments(
            db=db,
            task_ids=request.task_ids,
            developer_ids=request.developer_ids,
            optimization_objectives=objective_weights,
            constraints=getattr(request, 'constraints', None)
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Assignment optimization endpoint failed: {e}")
        raise HTTPException(status_code=500, detail=f"Assignment optimization failed: {str(e)}")
    

@router.post("/assignments/match-task", response_model=TaskMatchingResponse)
async def match_task_to_developers(
   request: TaskMatchingRequest,
   db: Session = Depends(get_db)
):
   """
   Find the best developer matches for specific tasks.
   
   This endpoint analyzes task requirements and finds developers
   with the best skill match, complexity fit, and learning potential.
   """
   try:
       start_time = datetime.now()
       
       all_matches = []
       
       for task_id in request.task_ids:
           matches = await assignment_optimizer.calculate_task_developer_matches(
               db=db,
               task_id=task_id,
               developer_ids=request.developer_ids,
               max_matches=request.max_matches_per_task
           )
           all_matches.extend(matches)
       
       processing_time = (datetime.now() - start_time).total_seconds() * 1000
       
       return TaskMatchingResponse(
           matches=all_matches,
           matching_metadata={
               'total_tasks': len(request.task_ids),
               'total_developers_considered': len(request.developer_ids) if request.developer_ids else "all",
               'matching_criteria': request.matching_criteria
           },
           processing_time_ms=processing_time
       )
       
   except Exception as e:
       raise HTTPException(status_code=500, detail=f"Task matching failed: {str(e)}")

@router.post("/assignments/suggest")
async def suggest_assignment(
   task_id: int,
   max_suggestions: int = Query(3, ge=1, le=10),
   db: Session = Depends(get_db)
):
   """Get assignment suggestions for a specific task."""
   try:
       matches = await assignment_optimizer.calculate_task_developer_matches(
           db=db,
           task_id=task_id,
           max_matches=max_suggestions
       )
       
       suggestions = []
       for match in matches:
           # Get enhanced weights from learning automata
           enhanced_weights = learning_automata.get_enhanced_assignment_weights(
               task_id, match.developer_id
           )
           
           # Get performance prediction
           task = db.query(Task).filter(Task.id == task_id).first()
           if task:
               task_complexity = [
                   task.technical_complexity or 0.5,
                   task.domain_difficulty or 0.5,
                   task.collaboration_requirements or 0.3,
                   task.learning_opportunities or 0.4,
                   task.business_impact or 0.5
               ]
               
               performance_pred = learning_automata.get_developer_performance_prediction(
                   match.developer_id, task_complexity
               )
               
               suggestions.append({
                   'developer_id': match.developer_id,
                   'overall_score': match.overall_score,
                   'skill_match_score': match.skill_match_score,
                   'complexity_fit_score': match.complexity_fit_score,
                   'learning_potential_score': match.learning_potential_score,
                   'workload_impact_score': match.workload_impact_score,
                   'reasoning': match.reasoning,
                   'risk_factors': match.risk_factors,
                   'enhanced_weights': enhanced_weights,
                   'predicted_performance': performance_pred['predicted_performance'],
                   'prediction_confidence': performance_pred['confidence']
               })
       
       return SuccessResponse(
           message="Assignment suggestions generated",
           data={'suggestions': suggestions}
       )
       
   except Exception as e:
       raise HTTPException(status_code=500, detail=f"Failed to generate suggestions: {str(e)}")

@router.get("/assignments/{assignment_id}/outcome")
async def get_assignment_outcome(assignment_id: int, db: Session = Depends(get_db)):
   """Get detailed outcome analysis for a completed assignment."""
   try:
       assignment = db.query(TaskAssignment).filter(TaskAssignment.id == assignment_id).first()
       if not assignment:
           raise HTTPException(status_code=404, detail="Assignment not found")
       
       if assignment.status != "completed":
           raise HTTPException(status_code=400, detail="Assignment not completed yet")
       
       # Calculate outcome metrics
       outcome_analysis = {
           'assignment_id': assignment_id,
           'task_id': assignment.task_id,
           'developer_id': assignment.developer_id,
           'duration_days': (assignment.completed_at - assignment.started_at).days if assignment.started_at and assignment.completed_at else None,
           'estimated_vs_actual_hours': {
               'estimated': assignment.task.estimated_hours,
               'actual': assignment.actual_hours,
               'variance_percent': ((assignment.actual_hours - assignment.task.estimated_hours) / assignment.task.estimated_hours * 100) if assignment.task.estimated_hours and assignment.actual_hours else None
           },
           'performance_scores': {
               'productivity': assignment.productivity_score,
               'skill_development': assignment.skill_development_score,
               'collaboration_effectiveness': assignment.collaboration_effectiveness,
               'feedback_score': assignment.feedback_score
           },
           'learning_impact': await _calculate_learning_impact(db, assignment),
           'recommendations': await _generate_outcome_recommendations(assignment)
       }
       
       return SuccessResponse(
           message="Assignment outcome retrieved",
           data=outcome_analysis
       )
       
   except Exception as e:
       raise HTTPException(status_code=500, detail=f"Failed to get assignment outcome: {str(e)}")

@router.get("/assignments/analytics/team-performance")
async def get_team_assignment_analytics(
   team_developer_ids: List[int] = Query(...),
   days_back: int = Query(30, ge=1, le=365),
   db: Session = Depends(get_db)
):
   """Get team assignment performance analytics."""
   try:
       from datetime import timedelta
       
       cutoff_date = datetime.now() - timedelta(days=days_back)
       
       # Get assignments for team members
       assignments = db.query(TaskAssignment).filter(
           TaskAssignment.developer_id.in_(team_developer_ids),
           TaskAssignment.updated_at >= cutoff_date,
           TaskAssignment.status == "completed"
       ).all()
       
       if not assignments:
           return SuccessResponse(
               message="No completed assignments found for the specified period",
               data={'analytics': {}}
           )
       
       # Calculate team analytics
       analytics = assignment_optimizer.get_assignment_analytics(db)
       
       # Add team-specific metrics
       team_analytics = {
           'team_size': len(team_developer_ids),
           'period_days': days_back,
           'total_assignments': len(assignments),
           'avg_assignment_score': sum(a.productivity_score for a in assignments if a.productivity_score) / len([a for a in assignments if a.productivity_score]),
           'skill_development_rate': sum(a.skill_development_score for a in assignments if a.skill_development_score) / len([a for a in assignments if a.skill_development_score]),
           'collaboration_effectiveness': sum(a.collaboration_effectiveness for a in assignments if a.collaboration_effectiveness) / len([a for a in assignments if a.collaboration_effectiveness]),
           'assignment_distribution': _calculate_assignment_distribution(assignments, team_developer_ids),
           'completion_rate': len([a for a in assignments if a.status == "completed"]) / len(assignments) if assignments else 0,
           'average_delivery_time': _calculate_average_delivery_time(assignments)
       }
       
       return SuccessResponse(
           message="Team assignment analytics retrieved",
           data={'analytics': team_analytics}
       )
       
   except Exception as e:
       raise HTTPException(status_code=500, detail=f"Failed to get team analytics: {str(e)}")

@router.get("/assignments/learning/analytics")
async def get_learning_analytics(db: Session = Depends(get_db)):
   """Get learning system analytics and insights."""
   try:
       analytics = learning_automata.get_learning_analytics()
       
       return SuccessResponse(
           message="Learning analytics retrieved",
           data={'analytics': analytics}
       )
       
   except Exception as e:
       raise HTTPException(status_code=500, detail=f"Failed to get learning analytics: {str(e)}")

@router.post("/assignments/learning/reset-developer/{developer_id}")
async def reset_developer_learning(developer_id: int, db: Session = Depends(get_db)):
   """Reset learning models for a specific developer."""
   try:
       # Verify developer exists
       developer = db.query(Developer).filter(Developer.id == developer_id).first()
       if not developer:
           raise HTTPException(status_code=404, detail="Developer not found")
       
       await learning_automata.reset_learning_for_developer(developer_id)
       
       return SuccessResponse(
           message=f"Learning models reset for developer {developer_id}"
       )
       
   except Exception as e:
       raise HTTPException(status_code=500, detail=f"Failed to reset learning models: {str(e)}")

@router.post("/assignments/learning/export")
async def export_learning_models():
   """Export learning models for backup or analysis."""
   try:
       models_data = await learning_automata.export_learning_models()
       
       return SuccessResponse(
           message="Learning models exported successfully",
           data={'models': models_data}
       )
       
   except Exception as e:
       raise HTTPException(status_code=500, detail=f"Failed to export learning models: {str(e)}")

@router.post("/assignments/learning/import")
async def import_learning_models(models_data: Dict[str, Any]):
   """Import previously exported learning models."""
   try:
       success = await learning_automata.import_learning_models(models_data)
       
       if success:
           return SuccessResponse(message="Learning models imported successfully")
       else:
           raise HTTPException(status_code=400, detail="Failed to import learning models")
       
   except Exception as e:
       raise HTTPException(status_code=500, detail=f"Failed to import learning models: {str(e)}")

# Helper functions
async def _calculate_learning_impact(db: Session, assignment: TaskAssignment) -> Dict[str, Any]:
   """Calculate the learning impact of an assignment."""
   
   if not assignment.skill_development_score:
       return {'impact': 'unknown', 'details': 'No skill development score available'}
   
   # Get developer's skill progression
   developer = assignment.developer
   if not developer:
       return {'impact': 'unknown', 'details': 'Developer not found'}
   
   # This would analyze skill vector changes before/after assignment
   # Simplified implementation
   learning_impact = {
       'skill_development_score': assignment.skill_development_score,
       'impact_level': 'high' if assignment.skill_development_score > 0.7 else 'medium' if assignment.skill_development_score > 0.4 else 'low',
       'learning_areas': [],  # Would be extracted from task requirements and developer skill gaps
       'recommended_follow_up': []  # Suggestions for next learning opportunities
   }
   
   return learning_impact

async def _generate_outcome_recommendations(assignment: TaskAssignment) -> List[str]:
   """Generate recommendations based on assignment outcome."""
   
   recommendations = []
   
   if assignment.productivity_score and assignment.productivity_score < 0.5:
       recommendations.append("Consider providing additional training or support for similar tasks")
   
   if assignment.skill_development_score and assignment.skill_development_score > 0.8:
       recommendations.append("Excellent learning outcome - consider more challenging assignments")
   
   if assignment.actual_hours and assignment.task.estimated_hours:
       time_variance = abs(assignment.actual_hours - assignment.task.estimated_hours) / assignment.task.estimated_hours
       if time_variance > 0.3:
           recommendations.append("Review time estimation accuracy for similar tasks")
   
   if assignment.collaboration_effectiveness and assignment.collaboration_effectiveness < 0.4:
       recommendations.append("Consider team collaboration training or pairing opportunities")
   
   return recommendations

def _calculate_assignment_distribution(assignments: List[TaskAssignment], team_developer_ids: List[int]) -> Dict[str, int]:
   """Calculate how assignments are distributed across team members."""
   
   distribution = {str(dev_id): 0 for dev_id in team_developer_ids}
   
   for assignment in assignments:
       if assignment.developer_id in team_developer_ids:
           distribution[str(assignment.developer_id)] += 1
   
   return distribution

def _calculate_average_delivery_time(assignments: List[TaskAssignment]) -> Optional[float]:
   """Calculate average delivery time for completed assignments."""
   
   delivery_times = []
   
   for assignment in assignments:
       if assignment.started_at and assignment.completed_at:
           delivery_time = (assignment.completed_at - assignment.started_at).total_seconds() / 3600  # Convert to hours
           delivery_times.append(delivery_time)
   
   return sum(delivery_times) / len(delivery_times) if delivery_times else None
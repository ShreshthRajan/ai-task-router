# src/api/tasks.py
from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List, Optional, Dict
import asyncio
from datetime import datetime

from ..models.database import get_db, Task, TaskComplexityAnalysis, TaskRequirement
from ..models.schemas import (
    Task as TaskSchema, TaskCreate, TaskUpdate, TaskComplexityResult,
    ComplexityAnalysisRequest, ComplexityAnalysisResponse, BatchComplexityRequest, BatchComplexityResponse,
    TaskRequirementsAnalysis, RequirementValidation, TaskDeveloperMatch, TaskMatchingRequest, TaskMatchingResponse,
    TaskRequirement as TaskRequirementSchema
)
from ..core.task_analysis.complexity_predictor import ComplexityPredictor
from ..core.task_analysis.requirement_parser import RequirementParser
from ..integrations.github_client import GitHubClient

router = APIRouter(prefix="/api/v1/tasks", tags=["tasks"])

# Initialize components
complexity_predictor = ComplexityPredictor()
requirement_parser = RequirementParser()
github_client = GitHubClient()

@router.post("/", response_model=TaskSchema)
async def create_task(
    task: TaskCreate,
    db: Session = Depends(get_db)
):
    """Create a new task."""
    db_task = Task(
        title=task.title,
        description=task.description,
        repository=task.repository,
        labels=task.labels,
        priority=task.priority,
        github_issue_id=task.github_issue_id
    )
    
    db.add(db_task)
    db.commit()
    db.refresh(db_task)
    
    return db_task

@router.get("/", response_model=List[TaskSchema])
async def list_tasks(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    status: Optional[str] = Query(None, pattern="^(open|in_progress|completed|closed)$"),
    priority: Optional[str] = Query(None, pattern="^(low|medium|high|critical)$"),
    repository: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """List tasks with filtering options."""
    query = db.query(Task)
    
    if status:
        query = query.filter(Task.status == status)
    if priority:
        query = query.filter(Task.priority == priority)
    if repository:
        query = query.filter(Task.repository == repository)
    
    tasks = query.offset(skip).limit(limit).all()
    return tasks

@router.get("/{task_id}", response_model=TaskSchema)
async def get_task(
    task_id: int,
    include_complexity: bool = Query(True),
    db: Session = Depends(get_db)
):
    """Get a specific task by ID."""
    task = db.query(Task).filter(Task.id == task_id).first()
    
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    if include_complexity and task.technical_complexity is not None:
        # Include complexity analysis in response
        task.complexity_analysis = TaskComplexityResult(
            technical_complexity=task.technical_complexity,
            domain_difficulty=task.domain_difficulty,
            collaboration_requirements=task.collaboration_requirements,
            learning_opportunities=task.learning_opportunities,
            business_impact=task.business_impact,
            estimated_hours=task.estimated_hours,
            confidence_score=task.complexity_confidence,
            complexity_factors=task.complexity_factors or {},
            required_skills=task.required_skills or {},
            risk_factors=task.risk_factors or []
        )
    
    return task

@router.put("/{task_id}", response_model=TaskSchema)
async def update_task(
    task_id: int,
    task_update: TaskUpdate,
    db: Session = Depends(get_db)
):
    """Update a task."""
    task = db.query(Task).filter(Task.id == task_id).first()
    
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    update_data = task_update.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(task, field, value)
    
    task.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(task)
    
    return task

@router.delete("/{task_id}")
async def delete_task(
    task_id: int,
    db: Session = Depends(get_db)
):
    """Delete a task."""
    task = db.query(Task).filter(Task.id == task_id).first()
    
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    db.delete(task)
    db.commit()
    
    return {"message": "Task deleted successfully"}

@router.post("/{task_id}/analyze-complexity", response_model=ComplexityAnalysisResponse)
async def analyze_task_complexity(
    task_id: int,
    include_requirements: bool = Query(True),
    db: Session = Depends(get_db)
):
    """Analyze complexity for a specific task."""
    task = db.query(Task).filter(Task.id == task_id).first()
    
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    start_time = datetime.utcnow()
    
    # Prepare task data for analysis
    task_data = {
        'id': str(task.id),
        'title': task.title,
        'body': task.description or '',
        'labels': [{'name': label} for label in (task.labels or [])],
        'repository': task.repository
    }
    
    try:
        # Perform complexity analysis
        complexity_result = await complexity_predictor.predict_task_complexity(task_data)
        
        # Update task with complexity results
        task.technical_complexity = complexity_result.technical_complexity
        task.domain_difficulty = complexity_result.domain_difficulty
        task.collaboration_requirements = complexity_result.collaboration_requirements
        task.learning_opportunities = complexity_result.learning_opportunities
        task.business_impact = complexity_result.business_impact
        task.estimated_hours = complexity_result.estimated_hours
        task.complexity_confidence = complexity_result.confidence_score
        task.required_skills = complexity_result.required_skills
        task.complexity_factors = complexity_result.complexity_factors
        task.risk_factors = complexity_result.risk_factors
        
        # Save complexity analysis details
        complexity_analysis = TaskComplexityAnalysis(
            task_id=task.id,
            extracted_features={'analysis_completed': True},
            mentioned_technologies=list(complexity_result.complexity_factors.get('mentioned_technologies', [])),
            affected_components=list(complexity_result.complexity_factors.get('affected_components', [])),
            architectural_impact=complexity_result.complexity_factors.get('architectural_impact', 0.5),
            urgency_indicators=complexity_result.complexity_factors.get('urgency_indicators', [])
        )
        
        # Check if analysis already exists
        existing_analysis = db.query(TaskComplexityAnalysis).filter(
            TaskComplexityAnalysis.task_id == task.id
        ).first()
        
        if existing_analysis:
            # Update existing analysis
            for field, value in complexity_analysis.__dict__.items():
                if not field.startswith('_') and field != 'id':
                    setattr(existing_analysis, field, value)
            existing_analysis.analysis_date = datetime.utcnow()
        else:
            db.add(complexity_analysis)
        
        db.commit()
        db.refresh(task)
        
        # Optionally analyze requirements
        requirements_analysis = None
        if include_requirements:
            try:
                requirements_analysis = await requirement_parser.parse_task_requirements(task_data)
                
                # Save requirements to database
                for req in requirements_analysis.requirements:
                    db_req = TaskRequirement(
                        task_id=task.id,
                        requirement_id=req.requirement_id,
                        category=req.category,
                        priority=req.priority,
                        description=req.description,
                        acceptance_criteria=req.acceptance_criteria,
                        dependencies=req.dependencies,
                        technical_constraints=req.technical_constraints,
                        estimated_complexity=req.estimated_complexity,
                        confidence_score=req.confidence_score
                    )
                    db.add(db_req)
                
                db.commit()
                
            except Exception as e:
                print(f"Error in requirements analysis: {e}")
                # Continue without requirements analysis
        
        processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        return ComplexityAnalysisResponse(
            task_id=str(task.id),
            complexity_result=TaskComplexityResult(
                technical_complexity=complexity_result.technical_complexity,
                domain_difficulty=complexity_result.domain_difficulty,
                collaboration_requirements=complexity_result.collaboration_requirements,
                learning_opportunities=complexity_result.learning_opportunities,
                business_impact=complexity_result.business_impact,
                estimated_hours=complexity_result.estimated_hours,
                confidence_score=complexity_result.confidence_score,
                complexity_factors=complexity_result.complexity_factors,
                required_skills=complexity_result.required_skills,
                risk_factors=complexity_result.risk_factors
            ),
            requirements_analysis=requirements_analysis,
            analysis_metadata={
                'analysis_version': '2.0',
                'features_extracted': len(complexity_result.complexity_factors),
                'technologies_identified': len(complexity_result.complexity_factors.get('mentioned_technologies', [])),
                'risk_factors_identified': len(complexity_result.risk_factors)
            },
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error analyzing task complexity: {str(e)}"
        )

@router.post("/analyze-complexity", response_model=ComplexityAnalysisResponse)
async def analyze_complexity_from_data(
    request: ComplexityAnalysisRequest
):
    """Analyze complexity from provided task data (without saving to database)."""
    start_time = datetime.utcnow()
    
    # Prepare task data
    task_data = {
        'id': 'temp',
        'title': request.title,
        'body': request.description,
        'labels': [{'name': label} for label in request.labels],
        'repository': request.repository
    }
    
    # Add any additional GitHub issue data
    if request.github_issue_data:
        task_data.update(request.github_issue_data)
    
    try:
        # Perform complexity analysis
        complexity_result = await complexity_predictor.predict_task_complexity(task_data)
        
        # Optionally analyze requirements
        requirements_analysis = None
        try:
            requirements_analysis = await requirement_parser.parse_task_requirements(task_data)
        except Exception as e:
            print(f"Error in requirements analysis: {e}")
        
        processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        return ComplexityAnalysisResponse(
            task_id=None,
            complexity_result=TaskComplexityResult(
                technical_complexity=complexity_result.technical_complexity,
                domain_difficulty=complexity_result.domain_difficulty,
                collaboration_requirements=complexity_result.collaboration_requirements,
                learning_opportunities=complexity_result.learning_opportunities,
                business_impact=complexity_result.business_impact,
                estimated_hours=complexity_result.estimated_hours,
                confidence_score=complexity_result.confidence_score,
                complexity_factors=complexity_result.complexity_factors,
                required_skills=complexity_result.required_skills,
                risk_factors=complexity_result.risk_factors
            ),
            requirements_analysis=requirements_analysis,
            analysis_metadata={
                'analysis_version': '2.0',
                'source': 'api_request'
            },
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error analyzing complexity: {str(e)}"
        )

@router.post("/batch-analyze-complexity", response_model=BatchComplexityResponse)
async def batch_analyze_complexity(
    request: BatchComplexityRequest
):
    """Analyze complexity for multiple tasks in batch."""
    start_time = datetime.utcnow()
    
    if len(request.tasks) > 50:
        raise HTTPException(
            status_code=400,
            detail="Batch size too large. Maximum 50 tasks per batch."
        )
    
    results = []
    errors = 0
    
    # Process tasks
    for i, task_request in enumerate(request.tasks):
        try:
            # Convert to analysis request format
            analysis_request = ComplexityAnalysisRequest(
                title=task_request.title,
                description=task_request.description,
                repository=task_request.repository,
                labels=task_request.labels,
                github_issue_data=task_request.github_issue_data
            )
            
            # Analyze each task
            result = await analyze_complexity_from_data(analysis_request)
            results.append(result)
            
        except Exception as e:
            errors += 1
            print(f"Error processing task {i}: {e}")
            # Add error result
            results.append(ComplexityAnalysisResponse(
                task_id=None,
                complexity_result=TaskComplexityResult(
                    technical_complexity=0.5,
                    domain_difficulty=0.5,
                    collaboration_requirements=0.5,
                    learning_opportunities=0.5,
                    business_impact=0.5,
                    estimated_hours=8.0,
                    confidence_score=0.1,
                    complexity_factors={},
                    required_skills={},
                    risk_factors=[f"Analysis error: {str(e)}"]
                ),
                requirements_analysis=None,
                analysis_metadata={'error': str(e)},
                processing_time_ms=0
            ))
    
    total_processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
    
    # Calculate summary statistics
    valid_results = [r for r in results if r.complexity_result.confidence_score > 0.3]
    avg_complexity = sum(r.complexity_result.technical_complexity for r in valid_results) / max(len(valid_results), 1)
    avg_hours = sum(r.complexity_result.estimated_hours for r in valid_results) / max(len(valid_results), 1)
    
    return BatchComplexityResponse(
        results=results,
        summary={
            'total_tasks': len(request.tasks),
            'successful_analyses': len(valid_results),
            'error_count': errors,
            'average_complexity': avg_complexity,
            'average_estimated_hours': avg_hours,
            'total_estimated_hours': sum(r.complexity_result.estimated_hours for r in valid_results)
        },
        total_processing_time_ms=total_processing_time
    )

@router.get("/{task_id}/requirements", response_model=List[TaskRequirementSchema])
async def get_task_requirements(
    task_id: int,
    db: Session = Depends(get_db)
):
    """Get requirements for a specific task."""
    task = db.query(Task).filter(Task.id == task_id).first()
    
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    requirements = db.query(TaskRequirement).filter(
        TaskRequirement.task_id == task_id
    ).all()
    
    return requirements

@router.get("/{task_id}/complexity-analysis")
async def get_task_complexity_analysis(
    task_id: int,
    db: Session = Depends(get_db)
):
    """Get detailed complexity analysis for a task."""
    analysis = db.query(TaskComplexityAnalysis).filter(
        TaskComplexityAnalysis.task_id == task_id
    ).first()
    
    if not analysis:
        raise HTTPException(status_code=404, detail="Complexity analysis not found")
    
    return analysis

@router.post("/import-from-github")
async def import_tasks_from_github(
    repository: str,
    state: str = Query("open", pattern="^(open|closed|all)$"),
    labels: Optional[List[str]] = Query(None),
    max_issues: int = Query(50, ge=1, le=200),
    analyze_complexity: bool = Query(True),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    db: Session = Depends(get_db)
):
    """Import tasks from GitHub issues."""
    
    try:
        # This would implement GitHub API integration
        # For now, we'll return a placeholder response
        
        return {
            "message": f"GitHub import initiated for repository: {repository}",
            "repository": repository,
            "state": state,
            "labels": labels,
            "max_issues": max_issues,
            "analyze_complexity": analyze_complexity,
            "status": "processing"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error importing from GitHub: {str(e)}"
        )

@router.get("/analytics/complexity-distribution")
async def get_complexity_distribution(
    repository: Optional[str] = None,
    days_back: int = Query(30, ge=1, le=365),
    db: Session = Depends(get_db)
):
    """Get complexity distribution analytics."""
    
    query = db.query(Task).filter(Task.technical_complexity.isnot(None))
    
    if repository:
        query = query.filter(Task.repository == repository)
    
    # Filter by date
    from datetime import timedelta
    cutoff_date = datetime.utcnow() - timedelta(days=days_back)
    query = query.filter(Task.created_at >= cutoff_date)
    
    tasks = query.all()
    
    if not tasks:
        return {
            "repository": repository,
            "period_days": days_back,
            "total_tasks": 0,
            "distribution": {}
        }
    
    # Calculate distributions
    complexity_buckets = {"low": 0, "medium": 0, "high": 0}
    hours_buckets = {"1-4": 0, "5-16": 0, "17-40": 0, "40+": 0}
    
    total_complexity = 0
    total_hours = 0
    
    for task in tasks:
        # Complexity distribution
        if task.technical_complexity < 0.4:
            complexity_buckets["low"] += 1
        elif task.technical_complexity < 0.7:
            complexity_buckets["medium"] += 1
        else:
            complexity_buckets["high"] += 1
        
        # Hours distribution
        if task.estimated_hours <= 4:
            hours_buckets["1-4"] += 1
        elif task.estimated_hours <= 16:
            hours_buckets["5-16"] += 1
        elif task.estimated_hours <= 40:
            hours_buckets["17-40"] += 1
        else:
            hours_buckets["40+"] += 1
        
        total_complexity += task.technical_complexity
        total_hours += task.estimated_hours
    
    return {
        "repository": repository,
        "period_days": days_back,
        "total_tasks": len(tasks),
        "average_complexity": total_complexity / len(tasks),
        "average_estimated_hours": total_hours / len(tasks),
        "complexity_distribution": complexity_buckets,
        "hours_distribution": hours_buckets,
        "domain_breakdown": {},  # Could be enhanced
        "risk_factors": {}  # Could be enhanced
    }
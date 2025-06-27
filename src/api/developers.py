from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Optional, Dict
from datetime import datetime
import json

from models.database import get_db, Developer, ExpertiseSnapshot
from models.schemas import DeveloperProfile, DeveloperCreate, DeveloperUpdate
from core.developer_modeling.expertise_tracker import ExpertiseTracker
from integrations.github_client import GitHubClient

router = APIRouter()
expertise_tracker = ExpertiseTracker()
github_client = GitHubClient()

@router.post("/developers/", response_model=DeveloperProfile)
async def create_developer(developer: DeveloperCreate, db: Session = Depends(get_db)):
    """Create a new developer profile."""
    
    # Check if developer already exists
    existing = db.query(Developer).filter(
        Developer.github_username == developer.github_username
    ).first()
    
    if existing:
        raise HTTPException(status_code=400, detail="Developer already exists")
    
    try:
        # Fetch GitHub data for initial profile
        github_data = await github_client.get_developer_data(developer.github_username)
        
        # Extract initial skills
        profile = expertise_tracker.track_developer_progress(
            developer_id=None,  # Will be set after creation
            new_data=github_data,
            db=db
        )
        
        return DeveloperProfile(
            id=profile.developer_id,
            github_username=developer.github_username,
            name=developer.name,
            email=developer.email,
            programming_languages=profile.programming_languages,
            domain_expertise=profile.domain_expertise,
            collaboration_score=profile.collaboration_score,
            learning_velocity=profile.learning_velocity,
            confidence_scores=profile.confidence_scores,
            semantic_code_quality=profile.skill_vector[353],  # we reserved dim-353
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating developer: {str(e)}")

@router.get("/developers/", response_model=List[DeveloperProfile])
async def list_developers(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    db: Session = Depends(get_db)
):
    """List all developers with pagination."""
    
    developers = db.query(Developer).offset(skip).limit(limit).all()
    
    return [
        DeveloperProfile(
            id=dev.id,
            github_username=dev.github_username,
            name=dev.name,
            email=dev.email,
            programming_languages=dev.primary_languages or {},
            domain_expertise=dev.domain_expertise or {},
            collaboration_score=dev.collaboration_score,
            learning_velocity=dev.learning_velocity,
            confidence_scores={},  # Could be enhanced to fetch from latest snapshot
            semantic_code_quality=0.0,  # historic profile isn't available
            created_at=dev.created_at,
            updated_at=dev.updated_at
        )
        for dev in developers
    ]

@router.get("/developers/{developer_id}", response_model=DeveloperProfile)
async def get_developer(developer_id: int, db: Session = Depends(get_db)):
    """Get detailed developer profile by ID."""
    
    developer = db.query(Developer).filter(Developer.id == developer_id).first()
    
    if not developer:
        raise HTTPException(status_code=404, detail="Developer not found")
    
    # Get latest expertise snapshot for confidence scores
    latest_snapshot = db.query(ExpertiseSnapshot).filter(
        ExpertiseSnapshot.developer_id == developer_id
    ).order_by(ExpertiseSnapshot.snapshot_date.desc()).first()
    
    confidence_scores = {}
    if latest_snapshot and latest_snapshot.confidence_scores:
        confidence_scores = latest_snapshot.confidence_scores
    
    return DeveloperProfile(
        id=developer.id,
        github_username=developer.github_username,
        name=developer.name,
        email=developer.email,
        programming_languages=developer.primary_languages or {},
        domain_expertise=developer.domain_expertise or {},
        collaboration_score=developer.collaboration_score,
        learning_velocity=developer.learning_velocity,
        confidence_scores=confidence_scores,
        semantic_code_quality=0.0,  # historic profile isn't available
        created_at=developer.created_at,
        updated_at=developer.updated_at
    )

@router.put("/developers/{developer_id}", response_model=DeveloperProfile)
async def update_developer(
    developer_id: int, 
    developer_update: DeveloperUpdate,
    db: Session = Depends(get_db)
):
    """Update developer profile."""
    
    developer = db.query(Developer).filter(Developer.id == developer_id).first()
    
    if not developer:
        raise HTTPException(status_code=404, detail="Developer not found")
    
    # Update basic fields
    if developer_update.name is not None:
        developer.name = developer_update.name
    if developer_update.email is not None:
        developer.email = developer_update.email
    
    developer.updated_at = datetime.utcnow()
    db.commit()
    
    return await get_developer(developer_id, db)

@router.post("/developers/{developer_id}/refresh")
async def refresh_developer_skills(developer_id: int, db: Session = Depends(get_db)):
    """Refresh developer skills by fetching latest GitHub data."""
    
    developer = db.query(Developer).filter(Developer.id == developer_id).first()
    
    if not developer:
        raise HTTPException(status_code=404, detail="Developer not found")
    
    try:
        # Fetch fresh GitHub data
        github_data = await github_client.get_developer_data(developer.github_username)
        
        # Update skills
        updated_profile = expertise_tracker.track_developer_progress(
            developer_id=developer_id,
            new_data=github_data,
            db=db
        )
        
        return {
            "message": "Developer skills refreshed successfully",
            "updated_at": datetime.utcnow().isoformat(),
            "changes": {
                "languages_updated": len(updated_profile.programming_languages),
                "domains_updated": len(updated_profile.domain_expertise),
                "collaboration_score": updated_profile.collaboration_score,
                "learning_velocity": updated_profile.learning_velocity
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error refreshing skills: {str(e)}")

@router.get("/developers/{developer_id}/skills/timeline")
async def get_skill_timeline(
    developer_id: int,
    skill_name: str = Query(..., description="Name of the skill to track"),
    days_back: int = Query(180, ge=30, le=365),
    db: Session = Depends(get_db)
):
    """Get timeline of how a specific skill has evolved."""
    
    developer = db.query(Developer).filter(Developer.id == developer_id).first()
    
    if not developer:
        raise HTTPException(status_code=404, detail="Developer not found")
    
    timeline = expertise_tracker.get_skill_evolution_timeline(
        developer_id, skill_name, days_back, db
    )
    
    return timeline

@router.get("/developers/{developer_id}/recommendations")
async def get_learning_recommendations(developer_id: int, db: Session = Depends(get_db)):
    """Get personalized learning recommendations for developer."""
    
    developer = db.query(Developer).filter(Developer.id == developer_id).first()
    
    if not developer:
        raise HTTPException(status_code=404, detail="Developer not found")
    
    recommendations = expertise_tracker.generate_learning_recommendations(developer_id, db)
    
    return {
        "developer_id": developer_id,
        "recommendations": [
            {
                "skill_area": rec.skill_area,
                "type": rec.recommendation_type,
                "priority": rec.priority,
                "reasoning": rec.reasoning
            }
            for rec in recommendations
        ],
        "generated_at": datetime.utcnow().isoformat()
    }

@router.get("/developers/{developer_id}/predictions")
async def get_skill_predictions(
    developer_id: int,
    skill_name: str = Query(..., description="Skill to predict"),
    months_ahead: int = Query(6, ge=1, le=24),
    db: Session = Depends(get_db)
):
    """Predict future skill development."""
    
    developer = db.query(Developer).filter(Developer.id == developer_id).first()
    
    if not developer:
        raise HTTPException(status_code=404, detail="Developer not found")
    
    prediction = expertise_tracker.predict_skill_development(
        developer_id, skill_name, months_ahead, db
    )
    
    return {
        "developer_id": developer_id,
        "skill_name": skill_name,
        "prediction": prediction,
        "generated_at": datetime.utcnow().isoformat()
    }

@router.post("/developers/compare")
async def compare_developers(
    developer_ids: List[int],
    skill_areas: List[str],
    db: Session = Depends(get_db)
):
    """Compare multiple developers across specific skill areas."""
    
    # Validate all developers exist
    existing_devs = db.query(Developer).filter(Developer.id.in_(developer_ids)).all()
    
    if len(existing_devs) != len(developer_ids):
        raise HTTPException(status_code=404, detail="One or more developers not found")
    
    comparison = expertise_tracker.compare_developers(developer_ids, skill_areas, db)
    
    return {
        "comparison": comparison,
        "generated_at": datetime.utcnow().isoformat()
    }

@router.get("/developers/team/{team_id}/matrix")
async def get_team_skill_matrix(team_id: str, db: Session = Depends(get_db)):
    """Get comprehensive skill matrix for a team."""
    
    # For now, assume team_id contains comma-separated developer IDs
    # In a real implementation, you'd have a Team model
    try:
        developer_ids = [int(id_str) for id_str in team_id.split(',')]
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid team_id format")
    
    # Validate developers exist
    existing_devs = db.query(Developer).filter(Developer.id.in_(developer_ids)).all()
    
    if len(existing_devs) != len(developer_ids):
        raise HTTPException(status_code=404, detail="One or more team members not found")
    
    skill_matrix = expertise_tracker.get_team_skill_matrix(developer_ids, db)
    
    return {
        "team_id": team_id,
        "skill_matrix": skill_matrix,
        "generated_at": datetime.utcnow().isoformat()
    }
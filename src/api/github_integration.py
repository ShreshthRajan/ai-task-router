"""
Enhanced GitHub Integration API for live repository analysis
"""

# src/api/github_integration.py

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, HttpUrl
import asyncio
import re
from datetime import datetime

from models.database import get_db, Developer, Task, TaskComplexityAnalysis
from models.schemas import (
    DeveloperCreate, TaskCreate, ComplexityAnalysisRequest,
    ComplexityAnalysisResponse
)
from integrations.github_client import GitHubClient
from core.developer_modeling.skill_extractor import SkillExtractor
from core.developer_modeling.code_analyzer import CodeAnalyzer
from core.task_analysis.complexity_predictor import ComplexityPredictor
from core.assignment_engine.optimizer import AssignmentOptimizer

router = APIRouter(prefix="/api/v1/github", tags=["github"])

# ===== PYDANTIC SCHEMAS =====
class GitHubRepoRequest(BaseModel):
    repo_url: str
    analyze_team: bool = True
    days_back: int = 90

class GitHubAnalysisResponse(BaseModel):
    repository: Dict[str, Any]
    developers: List[Dict[str, Any]]
    tasks: List[Dict[str, Any]]
    team_metrics: Dict[str, Any]
    analysis_time_ms: int

class LiveAnalysisStatus(BaseModel):
    status: str
    progress: float
    current_step: str
    message: str

class TeamMemberResponse(BaseModel):
    id: int
    github_username: str
    name: str
    email: str
    skill_vector: Dict[str, float]
    primary_languages: Dict[str, float]
    domain_expertise: Dict[str, float]
    collaboration_score: float
    learning_velocity: float
    commits_analyzed: int
    lines_of_code: int

class TaskResponse(BaseModel):
    id: int
    title: str
    description: str
    github_issue_number: int
    labels: List[str]
    technical_complexity: float
    domain_difficulty: float
    collaboration_requirements: float
    learning_opportunities: float
    business_impact: float
    estimated_hours: float
    confidence_score: float
    required_skills: List[str]
    risk_factors: List[str]
    complexity_analysis: Dict[str, Any]

# ===== COMPONENT INITIALIZATION =====
github_client = GitHubClient()
skill_extractor = SkillExtractor()
code_analyzer = CodeAnalyzer()
complexity_predictor = ComplexityPredictor()
assignment_optimizer = AssignmentOptimizer()

# ===== MAIN ENDPOINTS =====
@router.post("/analyze-repository", response_model=GitHubAnalysisResponse)
async def analyze_github_repository(
    request: GitHubRepoRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Analyze a GitHub repository and extract team intelligence, task complexity, and optimal assignments.
    """
    start_time = datetime.now()
    
    try:
        # Parse GitHub URL
        repo_info = parse_github_url(str(request.repo_url))
        owner, repo = repo_info["owner"], repo_info["repo"]
        
        # Step 1: Get repository information
        repo_data = await get_repository_info(owner, repo)
        
        # Step 2: Extract team from repository
        if request.analyze_team:
            developers_data = await extract_team_from_repo_internal(owner, repo, request.days_back, db)
        else:
            developers_data = []
        
        # Step 3: Analyze tasks from issues
        tasks_data = await extract_tasks_from_repo_internal(owner, repo, db)
        
        # Step 4: Calculate team metrics
        team_metrics = calculate_team_metrics(developers_data, tasks_data)
        
        # Calculate analysis time
        analysis_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return GitHubAnalysisResponse(
            repository=repo_data,
            developers=developers_data,
            tasks=tasks_data,
            team_metrics=team_metrics,
            analysis_time_ms=int(analysis_time)
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Analysis failed: {str(e)}")

@router.get("/repository/{owner}/{repo}")
async def get_repository_info(owner: str, repo: str) -> Dict[str, Any]:
    """Get basic repository information from GitHub."""
    try:
        # Use GitHub client to get repo info
        # For demo, return mock data
        return {
            "name": repo,
            "owner": owner,
            "description": "AI-powered development intelligence system",
            "language": "Python",
            "stars": 1247,
            "forks": 89,
            "watchers": 156,
            "created_at": "2024-01-15T10:30:00Z",
            "updated_at": "2024-06-24T15:45:00Z"
        }
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Repository not found: {str(e)}")

@router.post("/extract-team", response_model=List[TeamMemberResponse])
async def extract_team_from_repo(
    owner: str, 
    repo: str, 
    days_back: int = 90
) -> List[TeamMemberResponse]:
    """Extract team members and their skills from repository commits and contributions."""
    try:
        # Get repository contributors
        # For production, this would use the GitHub API
        # For demo, return realistic team data
        
        team_members = [
            TeamMemberResponse(
                id=1,
                github_username="sarah_backend",
                name="Sarah Chen",
                email="sarah@company.com",
                skill_vector={
                    "python": 0.89,
                    "api_design": 0.82,
                    "docker": 0.76,
                    "postgresql": 0.71,
                    "fastapi": 0.85
                },
                primary_languages={
                    "python": 0.89,
                    "javascript": 0.45,
                    "sql": 0.67
                },
                domain_expertise={
                    "backend": 0.87,
                    "security": 0.72,
                    "api_development": 0.85,
                    "database_design": 0.69
                },
                collaboration_score=0.84,
                learning_velocity=0.67,
                commits_analyzed=234,
                lines_of_code=15420
            ),
            TeamMemberResponse(
                id=2,
                github_username="maria_frontend",
                name="Maria Rodriguez",
                email="maria@company.com",
                skill_vector={
                    "react": 0.91,
                    "typescript": 0.85,
                    "css": 0.78,
                    "nextjs": 0.82,
                    "tailwind": 0.74
                },
                primary_languages={
                    "javascript": 0.91,
                    "typescript": 0.85,
                    "css": 0.78
                },
                domain_expertise={
                    "frontend": 0.93,
                    "ui_ux": 0.71,
                    "responsive_design": 0.83,
                    "accessibility": 0.65
                },
                collaboration_score=0.79,
                learning_velocity=0.73,
                commits_analyzed=189,
                lines_of_code=12890
            ),
            TeamMemberResponse(
                id=3,
                github_username="thomas_db",
                name="Thomas Kim",
                email="thomas@company.com",
                skill_vector={
                    "sql": 0.94,
                    "python": 0.76,
                    "optimization": 0.82,
                    "redis": 0.71,
                    "mongodb": 0.68
                },
                primary_languages={
                    "sql": 0.94,
                    "python": 0.76,
                    "bash": 0.62
                },
                domain_expertise={
                    "database": 0.96,
                    "performance": 0.84,
                    "data_modeling": 0.88,
                    "caching": 0.73
                },
                collaboration_score=0.71,
                learning_velocity=0.58,
                commits_analyzed=156,
                lines_of_code=8930
            )
        ]
        
        return team_members
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Team extraction failed: {str(e)}")

@router.post("/analyze-tasks", response_model=List[TaskResponse])
async def extract_tasks_from_repo(
    owner: str, 
    repo: str
) -> List[TaskResponse]:
    """Analyze GitHub issues and extract task complexity information."""
    try:
        # Get issues from repository
        # For production, this would use GitHub Issues API
        # For demo, return realistic task data
        
        tasks = [
            TaskResponse(
                id=1,
                title="Implement OAuth 2.0 authentication flow",
                description="Add OAuth 2.0 support with PKCE for mobile apps, integrate with existing JWT system",
                github_issue_number=42,
                labels=["enhancement", "security", "api"],
                technical_complexity=0.78,
                domain_difficulty=0.72,
                collaboration_requirements=0.45,
                learning_opportunities=0.67,
                business_impact=0.89,
                estimated_hours=16.5,
                confidence_score=0.84,
                required_skills=["oauth", "security", "api_design", "jwt", "mobile"],
                risk_factors=["security_complexity", "mobile_integration", "token_management"],
                complexity_analysis={
                    "features_detected": ["authentication", "authorization", "mobile_support"],
                    "technologies": ["oauth2", "pkce", "jwt"],
                    "architectural_impact": "medium",
                    "testing_complexity": "high"
                }
            ),
            TaskResponse(
                id=2,
                title="Optimize database query performance for analytics dashboard",
                description="Improve slow queries in user analytics dashboard, add proper indexing, consider query caching",
                github_issue_number=38,
                labels=["performance", "database", "optimization"],
                technical_complexity=0.65,
                domain_difficulty=0.82,
                collaboration_requirements=0.23,
                learning_opportunities=0.45,
                business_impact=0.91,
                estimated_hours=12.0,
                confidence_score=0.89,
                required_skills=["sql", "optimization", "indexing", "caching", "performance"],
                risk_factors=["data_migration", "downtime", "query_breaking_changes"],
                complexity_analysis={
                    "features_detected": ["database_optimization", "indexing", "caching"],
                    "technologies": ["postgresql", "redis", "sql"],
                    "architectural_impact": "low",
                    "testing_complexity": "medium"
                }
            ),
            TaskResponse(
                id=3,
                title="Redesign mobile responsive layout for dashboard",
                description="Current dashboard doesn't work well on mobile devices. Need responsive design overhaul.",
                github_issue_number=35,
                labels=["ui", "mobile", "responsive"],
                technical_complexity=0.52,
                domain_difficulty=0.41,
                collaboration_requirements=0.67,
                learning_opportunities=0.58,
                business_impact=0.74,
                estimated_hours=18.0,
                confidence_score=0.76,
                required_skills=["css", "responsive_design", "mobile_ui", "testing"],
                risk_factors=["cross_browser_compatibility", "device_testing"],
                complexity_analysis={
                    "features_detected": ["responsive_design", "mobile_optimization"],
                    "technologies": ["css", "tailwind", "flexbox", "grid"],
                    "architectural_impact": "low",
                    "testing_complexity": "high"
                }
            )
        ]
        
        return tasks
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Task analysis failed: {str(e)}")

# ===== INTERNAL HELPER FUNCTIONS =====
async def extract_team_from_repo_internal(
    owner: str, 
    repo: str, 
    days_back: int = 90,
    db: Optional[Session] = None
) -> List[Dict[str, Any]]:
    """Internal function to extract team members (used by main analysis endpoint)."""
    team_members_data = [
        {
            "id": 1,
            "github_username": "sarah_backend",
            "name": "Sarah Chen",
            "email": "sarah@company.com",
            "skill_vector": {
                "python": 0.89,
                "api_design": 0.82,
                "docker": 0.76,
                "postgresql": 0.71,
                "fastapi": 0.85
            },
            "primary_languages": {
                "python": 0.89,
                "javascript": 0.45,
                "sql": 0.67
            },
            "domain_expertise": {
                "backend": 0.87,
                "security": 0.72,
                "api_development": 0.85,
                "database_design": 0.69
            },
            "collaboration_score": 0.84,
            "learning_velocity": 0.67,
            "commits_analyzed": 234,
            "lines_of_code": 15420
        },
        {
            "id": 2,
            "github_username": "maria_frontend",
            "name": "Maria Rodriguez",
            "email": "maria@company.com",
            "skill_vector": {
                "react": 0.91,
                "typescript": 0.85,
                "css": 0.78,
                "nextjs": 0.82,
                "tailwind": 0.74
            },
            "primary_languages": {
                "javascript": 0.91,
                "typescript": 0.85,
                "css": 0.78
            },
            "domain_expertise": {
                "frontend": 0.93,
                "ui_ux": 0.71,
                "responsive_design": 0.83,
                "accessibility": 0.65
            },
            "collaboration_score": 0.79,
            "learning_velocity": 0.73,
            "commits_analyzed": 189,
            "lines_of_code": 12890
        },
        {
            "id": 3,
            "github_username": "thomas_db",
            "name": "Thomas Kim",
            "email": "thomas@company.com",
            "skill_vector": {
                "sql": 0.94,
                "python": 0.76,
                "optimization": 0.82,
                "redis": 0.71,
                "mongodb": 0.68
            },
            "primary_languages": {
                "sql": 0.94,
                "python": 0.76,
                "bash": 0.62
            },
            "domain_expertise": {
                "database": 0.96,
                "performance": 0.84,
                "data_modeling": 0.88,
                "caching": 0.73
            },
            "collaboration_score": 0.71,
            "learning_velocity": 0.58,
            "commits_analyzed": 156,
            "lines_of_code": 8930
        }
    ]
    
    # If database session provided, store developers
    if db:
        for member_data in team_members_data:
            existing_dev = db.query(Developer).filter(
                Developer.github_username == member_data["github_username"]
            ).first()
            
            if not existing_dev:
                developer = Developer(
                    github_username=member_data["github_username"],
                    name=member_data["name"],
                    email=member_data["email"],
                    skill_vector=member_data["skill_vector"],
                    primary_languages=member_data["primary_languages"],
                    domain_expertise=member_data["domain_expertise"],
                    collaboration_score=member_data["collaboration_score"],
                    learning_velocity=member_data["learning_velocity"]
                )
                db.add(developer)
        
        db.commit()
    
    return team_members_data

async def extract_tasks_from_repo_internal(
    owner: str, 
    repo: str,
    db: Optional[Session] = None
) -> List[Dict[str, Any]]:
    """Internal function to extract tasks (used by main analysis endpoint)."""
    tasks_data = [
        {
            "id": 1,
            "title": "Implement OAuth 2.0 authentication flow",
            "description": "Add OAuth 2.0 support with PKCE for mobile apps, integrate with existing JWT system",
            "github_issue_number": 42,
            "labels": ["enhancement", "security", "api"],
            "technical_complexity": 0.78,
            "domain_difficulty": 0.72,
            "collaboration_requirements": 0.45,
            "learning_opportunities": 0.67,
            "business_impact": 0.89,
            "estimated_hours": 16.5,
            "confidence_score": 0.84,
            "required_skills": ["oauth", "security", "api_design", "jwt", "mobile"],
            "risk_factors": ["security_complexity", "mobile_integration", "token_management"],
            "complexity_analysis": {
                "features_detected": ["authentication", "authorization", "mobile_support"],
                "technologies": ["oauth2", "pkce", "jwt"],
                "architectural_impact": "medium",
                "testing_complexity": "high"
            }
        },
        {
            "id": 2,
            "title": "Optimize database query performance for analytics dashboard",
            "description": "Improve slow queries in user analytics dashboard, add proper indexing, consider query caching",
            "github_issue_number": 38,
            "labels": ["performance", "database", "optimization"],
            "technical_complexity": 0.65,
            "domain_difficulty": 0.82,
            "collaboration_requirements": 0.23,
            "learning_opportunities": 0.45,
            "business_impact": 0.91,
            "estimated_hours": 12.0,
            "confidence_score": 0.89,
            "required_skills": ["sql", "optimization", "indexing", "caching", "performance"],
            "risk_factors": ["data_migration", "downtime", "query_breaking_changes"],
            "complexity_analysis": {
                "features_detected": ["database_optimization", "indexing", "caching"],
                "technologies": ["postgresql", "redis", "sql"],
                "architectural_impact": "low",
                "testing_complexity": "medium"
            }
        },
        {
            "id": 3,
            "title": "Redesign mobile responsive layout for dashboard",
            "description": "Current dashboard doesn't work well on mobile devices. Need responsive design overhaul.",
            "github_issue_number": 35,
            "labels": ["ui", "mobile", "responsive"],
            "technical_complexity": 0.52,
            "domain_difficulty": 0.41,
            "collaboration_requirements": 0.67,
            "learning_opportunities": 0.58,
            "business_impact": 0.74,
            "estimated_hours": 18.0,
            "confidence_score": 0.76,
            "required_skills": ["css", "responsive_design", "mobile_ui", "testing"],
            "risk_factors": ["cross_browser_compatibility", "device_testing"],
            "complexity_analysis": {
                "features_detected": ["responsive_design", "mobile_optimization"],
                "technologies": ["css", "tailwind", "flexbox", "grid"],
                "architectural_impact": "low",
                "testing_complexity": "high"
            }
        }
    ]
    
    # If database session provided, store tasks
    if db:
        for task_data in tasks_data:
            existing_task = db.query(Task).filter(
                Task.title == task_data["title"]
            ).first()
            
            if not existing_task:
                task = Task(
                    title=task_data["title"],
                    description=task_data["description"],
                    technical_complexity=task_data["technical_complexity"],
                    domain_difficulty=task_data["domain_difficulty"],
                    collaboration_requirements=task_data["collaboration_requirements"],
                    learning_opportunities=task_data["learning_opportunities"],
                    business_impact=task_data["business_impact"],
                    estimated_hours=task_data["estimated_hours"],
                    required_skills=task_data["required_skills"],
                    risk_factors=task_data["risk_factors"]
                )
                db.add(task)
        
        db.commit()
    
    return tasks_data

# ===== UTILITY FUNCTIONS =====
def parse_github_url(url: str) -> Dict[str, str]:
    """Parse GitHub URL to extract owner and repository name."""
    pattern = r"github\.com/([^/]+)/([^/]+)"
    match = re.search(pattern, url)
    
    if not match:
        raise ValueError("Invalid GitHub URL format")
    
    owner, repo = match.groups()
    repo = repo.replace('.git', '')  # Remove .git suffix if present
    
    return {"owner": owner, "repo": repo}

def calculate_team_metrics(developers: List[Dict], tasks: List[Dict]) -> Dict[str, Any]:
    """Calculate team-level metrics from developers and tasks analysis."""
    if not developers:
        return {
            "total_developers": 0,
            "avg_skill_level": 0.0,
            "collaboration_score": 0.0,
            "skill_diversity": 0.0
        }
    
    # Calculate average skill level
    avg_skill_level = sum(
        sum(dev["skill_vector"].values()) / len(dev["skill_vector"]) 
        for dev in developers
    ) / len(developers)
    
    # Calculate average collaboration score
    avg_collaboration = sum(dev["collaboration_score"] for dev in developers) / len(developers)
    
    # Calculate skill diversity (number of unique skills / total possible)
    all_skills = set()
    for dev in developers:
        all_skills.update(dev["skill_vector"].keys())
    
    skill_diversity = len(all_skills) / max(len(all_skills), 10)  # Normalize to reasonable max
    
    return {
        "total_developers": len(developers),
        "avg_skill_level": round(avg_skill_level, 2),
        "collaboration_score": round(avg_collaboration, 2),
        "skill_diversity": round(skill_diversity, 2),
        "total_skills_identified": len(all_skills),
        "avg_learning_velocity": round(
            sum(dev["learning_velocity"] for dev in developers) / len(developers), 2
        ),
        "total_commits_analyzed": sum(dev.get("commits_analyzed", 0) for dev in developers),
        "total_lines_of_code": sum(dev.get("lines_of_code", 0) for dev in developers)
    }

@router.get("/analysis-status/{analysis_id}", response_model=LiveAnalysisStatus)
async def get_analysis_status(analysis_id: str):
    """Get the status of a running analysis (for real-time updates)."""
    # This would track real analysis progress in production
    # For demo, return mock progress
    return LiveAnalysisStatus(
        status="completed",
        progress=1.0,
        current_step="analysis_complete",
        message="Analysis completed successfully"
    )
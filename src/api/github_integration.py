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
import aiohttp  
from config import settings

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


def get_github_client():
    global github_client
    if github_client is None:
        github_client = GitHubClient()
    return github_client

def get_skill_extractor():
    global skill_extractor
    if skill_extractor is None:
        from core.developer_modeling.skill_extractor import SkillExtractor
        skill_extractor = SkillExtractor()
    return skill_extractor

def get_code_analyzer():
    global code_analyzer
    if code_analyzer is None:
        from core.developer_modeling.code_analyzer import CodeAnalyzer
        code_analyzer = CodeAnalyzer()
    return code_analyzer

def get_complexity_predictor():
    global complexity_predictor
    if complexity_predictor is None:
        from core.task_analysis.complexity_predictor import ComplexityPredictor
        complexity_predictor = ComplexityPredictor()
    return complexity_predictor

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
        
        # Step 1: Get repository information (using real API now)
        repo_data = await get_repository_info(owner, repo)
        
        # Step 2: Extract team from repository
        developers_data = []
        if request.analyze_team:
            try:
                developers_data = await extract_team_from_repo_internal(owner, repo, request.days_back, db)
                if not developers_data:
                    print(f"Warning: No developers found for {owner}/{repo}")
            except Exception as e:
                print(f"Error extracting team: {e}")
                # Continue with empty developers list rather than failing
        
        # Step 3: Analyze tasks from issues
        tasks_data = []
        try:
            tasks_data = await extract_tasks_from_repo_internal(owner, repo, db)
            if not tasks_data:
                print(f"Warning: No tasks found for {owner}/{repo}")
        except Exception as e:
            print(f"Error extracting tasks: {e}")
            # Continue with empty tasks list rather than failing
        
        # Step 4: Calculate team metrics
        team_metrics = calculate_team_metrics(developers_data, tasks_data)
        
        # Calculate analysis timex
        analysis_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return GitHubAnalysisResponse(
            repository=repo_data,
            developers=developers_data,
            tasks=tasks_data,
            team_metrics=team_metrics,
            analysis_time_ms=int(analysis_time)
        )
        
    except Exception as e:
        print(f"Repository analysis failed: {e}")
        raise HTTPException(status_code=400, detail=f"Analysis failed: {str(e)}")

@router.get("/repository/{owner}/{repo}")
async def get_repository_info(owner: str, repo: str) -> Dict[str, Any]:
    """Get basic repository information from GitHub."""
    try:
        # Use real GitHub API to get repo info
        async with aiohttp.ClientSession(headers=get_github_client().headers) as session:
            url = f"{github_client.base_url}/repos/{owner}/{repo}"
            async with session.get(url) as response:
                if response.status == 200:
                    repo_data = await response.json()
                    return {
                        "name": repo_data.get("name", repo),
                        "owner": repo_data.get("owner", {}).get("login", owner),
                        "description": repo_data.get("description", ""),
                        "language": repo_data.get("language", "Unknown"),
                        "stars": repo_data.get("stargazers_count", 0),
                        "forks": repo_data.get("forks_count", 0),
                        "watchers": repo_data.get("watchers_count", 0),
                        "created_at": repo_data.get("created_at", ""),
                        "updated_at": repo_data.get("updated_at", ""),
                        "size": repo_data.get("size", 0),
                        "open_issues": repo_data.get("open_issues_count", 0)
                    }
                else:
                    raise HTTPException(status_code=404, detail=f"Repository not found: HTTP {response.status}")
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

async def extract_team_from_repo_internal(
    owner: str, 
    repo: str, 
    days_back: int = 90,
    db: Optional[Session] = None
) -> List[Dict[str, Any]]:
    """Internal function to extract team members using real Phase 1 components."""
    try:
        # Get real contributor data from GitHub
        contributors = await get_github_client().get_repository_contributors(owner, repo)
        team_members_data = []
        
        for i, contributor in enumerate(contributors[:10]):  # Limit to top 10 contributors
            username = contributor.get('login')
            if not username:
                continue
                
            try:
                # Use real Phase 1 components
                developer_data = await get_github_client().get_developer_data(username, days_back)
                profile = get_skill_extractor().extract_comprehensive_profile(developer_data, days_back)
                
                # Convert to expected format
                member_data = {
                    "id": i + 1,
                    "github_username": username,
                    "name": developer_data.get('name') or username,
                    "email": developer_data.get('email', f"{username}@github.local"),
                    "skill_vector": {
                        "programming_languages": profile.programming_languages,
                        "domain_expertise": profile.domain_expertise,
                        "collaboration_patterns": {
                            "code_reviews": profile.collaboration_score * 0.8,
                            "issue_discussions": profile.collaboration_score * 0.6,
                            "knowledge_sharing": profile.collaboration_score * 0.9
                        },
                        "technical_skills": {
                            skill: score.get('proficiency', 0.0) if isinstance(score, dict) else score
                            for skill, score in profile.programming_languages.items()
                        }
                    },
                    "primary_languages": profile.programming_languages,
                    "collaboration_score": profile.collaboration_score,
                    "learning_velocity": profile.learning_velocity,
                    "commits_analyzed": len(developer_data.get('commits', [])),
                    "lines_of_code": sum(commit.get('additions', 0) for commit in developer_data.get('commits', [])),
                    "expertise_confidence": profile.confidence_scores.get('overall', 0.7)
                }
                team_members_data.append(member_data)
                
            except Exception as e:
                print(f"Error analyzing developer {username}: {e}")
                continue
        
        return team_members_data
        
    except Exception as e:
        print(f"Error extracting team: {e}")
        # Return empty list on error instead of crashing
        return []

async def extract_tasks_from_repo_internal(
    owner: str, 
    repo: str,
    db: Optional[Session] = None
) -> List[Dict[str, Any]]:
    """Internal function to extract tasks using real Phase 2 components."""
    try:
        # Get real issues from GitHub
        issues = await get_github_client().get_repository_issues(owner, repo, max_issues=20)
        tasks_data = []
        
        for i, issue in enumerate(issues):
            try:
                # Use real Phase 2 complexity prediction
                task_input = {
                    'id': str(issue.get('number', i)),
                    'title': issue.get('title', ''),
                    'body': issue.get('body', ''),
                    'labels': [label.get('name', '') for label in issue.get('labels', [])],
                    'repository': f"{owner}/{repo}"
                }
                
                complexity_result = await get_complexity_predictor().predict_task_complexity(task_input)
                
                # Convert to expected format
                task_data = {
                    "id": i + 1,
                    "title": issue.get('title', ''),
                    "description": issue.get('body', '')[:500] + "..." if len(issue.get('body', '')) > 500 else issue.get('body', ''),
                    "github_issue_number": issue.get('number'),
                    "labels": [label.get('name', '') for label in issue.get('labels', [])],
                    "complexity_analysis": {
                        "technical_complexity": complexity_result.technical_complexity,
                        "domain_difficulty": complexity_result.domain_difficulty,
                        "collaboration_requirements": complexity_result.collaboration_requirements,
                        "learning_opportunities": complexity_result.learning_opportunities,
                        "business_impact": complexity_result.business_impact,
                        "estimated_hours": complexity_result.estimated_hours,
                        "confidence_score": complexity_result.confidence_score,
                        "complexity_factors": complexity_result.complexity_factors,
                        "required_skills": complexity_result.required_skills,
                        "risk_factors": complexity_result.risk_factors
                    }
                }
                tasks_data.append(task_data)
                
            except Exception as e:
                print(f"Error analyzing task {issue.get('number', i)}: {e}")
                continue
        
        return tasks_data
        
    except Exception as e:
        print(f"Error extracting tasks: {e}")
        return []

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
    """Calculate comprehensive team-level metrics from real analysis."""
    if not developers:
        return {
            "total_developers": 0,
            "avg_skill_level": 0.0,
            "collaboration_score": 0.0,
            "skill_diversity": 0.0,
            "total_skills_identified": 0,
            "avg_learning_velocity": 0.0,
            "total_commits_analyzed": 0,
            "total_lines_of_code": 0,
            "expertise_coverage": {},
            "team_strength_score": 0.0
        }
    
    # Calculate average skill level across all languages
    total_skill_sum = 0
    skill_count = 0
    all_skills = set()
    
    for dev in developers:
        for lang, prof in dev.get("primary_languages", {}).items():
            if isinstance(prof, dict):
                total_skill_sum += prof.get('proficiency', 0.0)
            else:
                total_skill_sum += prof
            skill_count += 1
            all_skills.add(lang)
        
        # Add domain skills
        for domain in dev.get("skill_vector", {}).get("domain_expertise", {}):
            all_skills.add(domain)
    
    avg_skill_level = total_skill_sum / max(skill_count, 1)
    avg_collaboration = sum(dev.get("collaboration_score", 0.0) for dev in developers) / len(developers)
    avg_learning_velocity = sum(dev.get("learning_velocity", 0.0) for dev in developers) / len(developers)
    
    # Calculate expertise coverage
    expertise_areas = ["frontend", "backend", "database", "devops", "testing", "security"]
    expertise_coverage = {}
    
    for area in expertise_areas:
        area_experts = sum(1 for dev in developers 
                          if dev.get("skill_vector", {}).get("domain_expertise", {}).get(area, 0.0) > 0.5)
        expertise_coverage[area] = min(area_experts / max(len(developers), 1), 1.0)
    
    # Calculate team strength score
    skill_diversity = len(all_skills) / max(len(all_skills), 20)  # Normalize
    team_strength = (avg_skill_level * 0.3 + avg_collaboration * 0.3 + 
                    skill_diversity * 0.2 + avg_learning_velocity * 0.2)
    
    return {
        "total_developers": len(developers),
        "avg_skill_level": round(avg_skill_level, 3),
        "collaboration_score": round(avg_collaboration, 3),
        "skill_diversity": round(skill_diversity, 3),
        "total_skills_identified": len(all_skills),
        "avg_learning_velocity": round(avg_learning_velocity, 3),
        "total_commits_analyzed": sum(dev.get("commits_analyzed", 0) for dev in developers),
        "total_lines_of_code": sum(dev.get("lines_of_code", 0) for dev in developers),
        "expertise_coverage": expertise_coverage,
        "team_strength_score": round(team_strength, 3)
    }

@router.get("/debug/github/{owner}/{repo}")
async def debug_github_data(owner: str, repo: str):
    """Debug endpoint to test GitHub API connectivity and data flow."""
    debug_info = {
        "github_token_configured": bool(settings.GITHUB_TOKEN),
        "github_token_prefix": settings.GITHUB_TOKEN[:10] + "..." if settings.GITHUB_TOKEN else "None",
        "test_results": {}
    }
    
    try:
        # Test 1: Basic repo access
        print(f"🔍 DEBUG: Testing basic repo access for {owner}/{repo}")
        async with aiohttp.ClientSession(headers=get_github_client().headers) as session:
            url = f"{github_client.base_url}/repos/{owner}/{repo}"
            async with session.get(url) as response:
                debug_info["test_results"]["repo_access"] = {
                    "status": response.status,
                    "headers": dict(response.headers),
                    "rate_limit_remaining": response.headers.get('X-RateLimit-Remaining'),
                    "rate_limit_limit": response.headers.get('X-RateLimit-Limit')
                }
                if response.status == 200:
                    repo_data = await response.json()
                    debug_info["test_results"]["repo_data_sample"] = {
                        "name": repo_data.get("name"),
                        "stars": repo_data.get("stargazers_count"),
                        "language": repo_data.get("language")
                    }
                else:
                    error_text = await response.text()
                    debug_info["test_results"]["repo_error"] = error_text
        
        # Test 2: Contributors access
        print(f"🔍 DEBUG: Testing contributors access")
        contributors = await get_github_client().get_repository_contributors(owner, repo, max_contributors=5)
        debug_info["test_results"]["contributors"] = {
            "count": len(contributors) if contributors else 0,
            "sample_data": contributors[:2] if contributors else [],
            "first_contributor": contributors[0] if contributors else None
        }
        
        # Test 3: Individual developer data
        if contributors:
            first_contributor = contributors[0]
            username = first_contributor.get('login')
            print(f"🔍 DEBUG: Testing individual developer data for {username}")
            
            developer_data = await github_client.get_developer_data(username, days_back=30)
            debug_info["test_results"]["developer_data"] = {
                "username": username,
                "commits_count": len(developer_data.get('commits', [])),
                "pr_reviews_count": len(developer_data.get('pr_reviews', [])),
                "issue_comments_count": len(developer_data.get('issue_comments', [])),
                "sample_commit": developer_data.get('commits', [{}])[0] if developer_data.get('commits') else None
            }
            
            # Test 4: Skill extraction
            print(f"🔍 DEBUG: Testing skill extraction for {username}")
            try:
                profile = skill_extractor.extract_comprehensive_profile(developer_data, 30)
                debug_info["test_results"]["skill_extraction"] = {
                    "programming_languages": dict(profile.programming_languages),
                    "domain_expertise": dict(profile.domain_expertise),
                    "collaboration_score": profile.collaboration_score,
                    "learning_velocity": profile.learning_velocity,
                    "confidence_scores": dict(profile.confidence_scores)
                }
            except Exception as e:
                debug_info["test_results"]["skill_extraction_error"] = str(e)
        
        return debug_info
        
    except Exception as e:
        debug_info["error"] = str(e)
        return debug_info

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
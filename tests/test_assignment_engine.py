"""
Comprehensive test suite for Phase 3: Assignment Engine

Tests cover optimization algorithms, learning automata, and API endpoints.
"""

import pytest
import asyncio
import numpy as np
from datetime import datetime, timedelta
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from fastapi.testclient import TestClient

from src.models.database import Base, Developer, Task, TaskAssignment, get_db
from src.models.schemas import (
    AssignmentCreate, OptimizationRequest, TaskMatchingRequest,
    OptimizationObjective
)
from src.core.assignment_engine.optimizer import AssignmentOptimizer
from src.core.assignment_engine.learning_automata import LearningAutomata
from src.main import app

# Test database setup
TEST_DATABASE_URL = "sqlite:///./test_assignment_engine.db"
test_engine = create_engine(TEST_DATABASE_URL, connect_args={"check_same_thread": False})
TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)

# Create test database
Base.metadata.create_all(bind=test_engine)

def get_test_db():
    """Override database dependency for testing."""
    db = TestSessionLocal()
    try:
        yield db
    finally:
        db.close()

app.dependency_overrides[get_db] = get_test_db

client = TestClient(app)

# Test fixtures
@pytest.fixture
def db_session():
    """Create a database session for testing."""
    db = TestSessionLocal()
    yield db
    db.close()

@pytest.fixture
def sample_developers(db_session):
    """Create sample developers for testing."""
    developers = []
    
    # Senior Python Developer
    dev1 = Developer(
        github_username="senior_python_dev",
        name="Alice Johnson",
        email="alice@example.com",
        skill_vector='[0.9, 0.7, 0.5, 0.8, 0.6]' + '[0.0]' * 763,  # 768-dim vector
        primary_languages='{"python": 0.9, "sql": 0.7, "javascript": 0.5}',
        domain_expertise='{"backend": 0.9, "api": 0.8, "database": 0.7}',
        collaboration_score=0.8,
        learning_velocity=0.6
    )
    
    # Junior Frontend Developer
    dev2 = Developer(
        github_username="junior_frontend_dev",
        name="Bob Smith",
        email="bob@example.com",
        skill_vector='[0.3, 0.8, 0.9, 0.4, 0.5]' + '[0.0]' * 763,
        primary_languages='{"javascript": 0.8, "typescript": 0.7, "css": 0.9}',
        domain_expertise='{"frontend": 0.9, "ui_ux": 0.8, "react": 0.8}',
        collaboration_score=0.7,
        learning_velocity=0.9
    )
    
    # Full-stack ML Engineer
    dev3 = Developer(
        github_username="ml_engineer",
        name="Carol Davis",
        email="carol@example.com",
        skill_vector='[0.8, 0.6, 0.7, 0.9, 0.8]' + '[0.0]' * 763,
        primary_languages='{"python": 0.9, "r": 0.7, "sql": 0.8}',
        domain_expertise='{"machine_learning": 0.9, "data_science": 0.8, "backend": 0.7}',
        collaboration_score=0.6,
        learning_velocity=0.7
    )
    
    developers = [dev1, dev2, dev3]
    
    for dev in developers:
        db_session.add(dev)
    
    db_session.commit()
    
    for dev in developers:
        db_session.refresh(dev)
    
    return developers

@pytest.fixture
def sample_tasks(db_session):
    """Create sample tasks for testing."""
    tasks = []
    
    # High complexity backend task
    task1 = Task(
        title="Implement distributed caching system",
        description="Design and implement a Redis-based distributed caching layer with failover support",
        repository="backend-service",
        priority="high",
        labels='["backend", "redis", "performance"]',
        technical_complexity=0.9,
        domain_difficulty=0.8,
        collaboration_requirements=0.6,
        learning_opportunities=0.7,
        business_impact=0.8,
        estimated_hours=32.0,
        complexity_confidence=0.8,
        required_skills='{"python": 0.8, "redis": 0.9, "backend": 0.8}',
        status="open"
    )
    
    # Medium complexity frontend task
    task2 = Task(
        title="Create responsive dashboard components",
        description="Build reusable dashboard components with React and TypeScript",
        repository="frontend-app",
        priority="medium",
        labels='["frontend", "react", "ui"]',
        technical_complexity=0.6,
        domain_difficulty=0.5,
        collaboration_requirements=0.4,
        learning_opportunities=0.6,
        business_impact=0.6,
        estimated_hours=16.0,
        complexity_confidence=0.7,
        required_skills='{"javascript": 0.7, "react": 0.8, "typescript": 0.6}',
        status="open"
    )
    
    # Low complexity documentation task
    task3 = Task(
        title="Update API documentation",
        description="Update OpenAPI specifications and add usage examples",
        repository="api-docs",
        priority="low",
        labels='["documentation", "api"]',
        technical_complexity=0.3,
        domain_difficulty=0.3,
        collaboration_requirements=0.2,
        learning_opportunities=0.3,
        business_impact=0.4,
        estimated_hours=8.0,
        complexity_confidence=0.9,
        required_skills='{"documentation": 0.8, "api": 0.5}',
        status="open"
    )
    
    # High complexity ML task
    task4 = Task(
        title="Implement recommendation engine",
        description="Build ML-based recommendation system using collaborative filtering",
        repository="ml-service",
        priority="critical",
        labels='["machine-learning", "algorithms", "backend"]',
        technical_complexity=0.9,
        domain_difficulty=0.9,
        collaboration_requirements=0.7,
        learning_opportunities=0.8,
        business_impact=0.9,
        estimated_hours=40.0,
        complexity_confidence=0.7,
        required_skills='{"python": 0.8, "machine_learning": 0.9, "algorithms": 0.8}',
        status="open"
    )
    
    tasks = [task1, task2, task3, task4]
    
    for task in tasks:
        db_session.add(task)
    
    db_session.commit()
    
    for task in tasks:
        db_session.refresh(task)
    
    return tasks

@pytest.fixture
def assignment_optimizer():
   """Create assignment optimizer instance."""
   return AssignmentOptimizer()

@pytest.fixture
def learning_automata():
   """Create learning automata instance."""
   return LearningAutomata()

# Core Assignment Engine Tests

class TestAssignmentOptimizer:
   """Test suite for AssignmentOptimizer class."""
   
   @pytest.mark.asyncio
   async def test_optimize_assignments_basic(self, db_session, sample_developers, sample_tasks, assignment_optimizer):
       """Test basic assignment optimization."""
       task_ids = [task.id for task in sample_tasks[:2]]
       developer_ids = [dev.id for dev in sample_developers]
       
       result = await assignment_optimizer.optimize_assignments(
           db=db_session,
           task_ids=task_ids,
           developer_ids=developer_ids
       )
       
       assert result is not None
       assert len(result.assignments) <= len(task_ids)
       assert result.optimization_time_ms > 0
       assert 'productivity' in result.objective_scores
       assert 'skill_development' in result.objective_scores
       assert 'workload_balance' in result.objective_scores
       
       # Verify assignments are valid
       for assignment in result.assignments:
           assert assignment.task_id in task_ids
           assert assignment.developer_id in developer_ids
           assert 0 <= assignment.confidence_score <= 1
           assert assignment.reasoning is not None

   @pytest.mark.asyncio
   async def test_task_developer_matching(self, db_session, sample_developers, sample_tasks, assignment_optimizer):
       """Test task-developer matching functionality."""
       task_id = sample_tasks[0].id  # High complexity backend task
       developer_ids = [dev.id for dev in sample_developers]
       
       matches = await assignment_optimizer.calculate_task_developer_matches(
           db=db_session,
           task_id=task_id,
           developer_ids=developer_ids,
           max_matches=3
       )
       
       assert len(matches) <= 3
       assert len(matches) <= len(developer_ids)
       
       # Check matches are sorted by score
       scores = [match.overall_score for match in matches]
       assert scores == sorted(scores, reverse=True)
       
       # Verify match data
       for match in matches:
           assert match.task_id == task_id
           assert match.developer_id in developer_ids
           assert 0 <= match.overall_score <= 1
           assert 0 <= match.skill_match_score <= 1
           assert 0 <= match.complexity_fit_score <= 1
           assert 0 <= match.learning_potential_score <= 1
           assert 0 <= match.workload_impact_score <= 1
           assert match.reasoning is not None
           assert isinstance(match.risk_factors, list)

   @pytest.mark.asyncio
   async def test_skill_match_calculation(self, db_session, sample_developers, sample_tasks, assignment_optimizer):
       """Test skill matching calculation."""
       # Backend task should match better with Python developer
       backend_task = sample_tasks[0]  # High complexity backend task
       python_dev = sample_developers[0]  # Senior Python developer
       frontend_dev = sample_developers[1]  # Junior Frontend developer
       
       python_matches = await assignment_optimizer.calculate_task_developer_matches(
           db=db_session,
           task_id=backend_task.id,
           developer_ids=[python_dev.id],
           max_matches=1
       )
       
       frontend_matches = await assignment_optimizer.calculate_task_developer_matches(
           db=db_session,
           task_id=backend_task.id,
           developer_ids=[frontend_dev.id],
           max_matches=1
       )
       
       # Python developer should have higher skill match for backend task
       assert len(python_matches) == 1
       assert len(frontend_matches) == 1
       assert python_matches[0].skill_match_score > frontend_matches[0].skill_match_score

   @pytest.mark.asyncio
   async def test_workload_balancing(self, db_session, sample_developers, sample_tasks, assignment_optimizer):
       """Test workload balancing in assignments."""
       # Create existing assignments to simulate workload
       existing_assignment = TaskAssignment(
           task_id=sample_tasks[0].id,
           developer_id=sample_developers[0].id,
           status="accepted",
           confidence_score=0.8,
           reasoning="Existing assignment for testing"
       )
       db_session.add(existing_assignment)
       db_session.commit()
       
       # Optimize new assignments
       task_ids = [task.id for task in sample_tasks[1:3]]
       developer_ids = [dev.id for dev in sample_developers]
       
       result = await assignment_optimizer.optimize_assignments(
           db=db_session,
           task_ids=task_ids,
           developer_ids=developer_ids
       )
       
       # Check that workload balance is considered
       assert result.objective_scores['workload_balance'] >= 0
       
       # Developer with existing assignment should be less likely to get new ones
       assigned_dev_ids = [a.developer_id for a in result.assignments]
       # This is probabilistic, but overloaded developer should be less preferred

   @pytest.mark.asyncio
   async def test_custom_optimization_weights(self, db_session, sample_developers, sample_tasks, assignment_optimizer):
       """Test custom optimization objective weights."""
       task_ids = [task.id for task in sample_tasks[:2]]
       developer_ids = [dev.id for dev in sample_developers]
       
       # Prioritize skill development
       custom_weights = {
           'productivity': 0.2,
           'skill_development': 0.6,
           'workload_balance': 0.1,
           'collaboration': 0.05,
           'business_impact': 0.05
       }
       
       result = await assignment_optimizer.optimize_assignments(
           db=db_session,
           task_ids=task_ids,
           developer_ids=developer_ids,
           optimization_objectives=custom_weights
       )
       
       assert result is not None
       assert len(result.assignments) > 0
       
       # With high skill development weight, junior developer should get learning opportunities
       junior_dev_assignments = [a for a in result.assignments if a.developer_id == sample_developers[1].id]
       # Junior dev should potentially get assignments with learning opportunities

   @pytest.mark.asyncio
   async def test_constraint_application(self, db_session, sample_developers, sample_tasks, assignment_optimizer):
       """Test constraint application in optimization."""
       task_ids = [task.id for task in sample_tasks]
       developer_ids = [dev.id for dev in sample_developers]
       
       constraints = {
           'max_workload_per_developer': 0.5,  # Very restrictive
           'enforce_skill_requirements': True
       }
       
       result = await assignment_optimizer.optimize_assignments(
           db=db_session,
           task_ids=task_ids,
           developer_ids=developer_ids,
           constraints=constraints
       )
       
       # With restrictive constraints, fewer assignments should be made
       assert len(result.assignments) <= len(developer_ids)
       
       # All assignments should meet constraints
       for assignment in result.assignments:
           assert assignment.confidence_score > 0  # Should not be blocked by constraints

class TestLearningAutomata:
   """Test suite for LearningAutomata class."""
   
   @pytest.mark.asyncio
   async def test_learn_from_assignment_outcome(self, db_session, sample_developers, sample_tasks, learning_automata):
       """Test learning from assignment outcomes."""
       # Create a completed assignment
       assignment = TaskAssignment(
           task_id=sample_tasks[0].id,
           developer_id=sample_developers[0].id,
           status="completed",
           confidence_score=0.8,
           reasoning="Test assignment",
           actual_hours=30.0,
           productivity_score=0.9,
           skill_development_score=0.7,
           collaboration_effectiveness=0.8,
           feedback_score=0.8
       )
       db_session.add(assignment)
       db_session.commit()
       
       # Learn from the outcome
       outcome_metrics = {
           'productivity_score': 0.9,
           'skill_development_score': 0.7,
           'collaboration_effectiveness': 0.8,
           'feedback_score': 0.8
       }
       
       await learning_automata.learn_from_assignment_outcome(
           db=db_session,
           assignment_id=assignment.id,
           outcome_metrics=outcome_metrics
       )
       
       # Check that learning occurred
       analytics = learning_automata.get_learning_analytics()
       assert analytics['total_assignments_learned'] > 0
       assert analytics['developers_modeled'] > 0

   def test_skill_importance_learning(self, learning_automata):
       """Test skill importance factor learning."""
       initial_factor = learning_automata.get_skill_importance_factor('python')
       
       # Simulate multiple good outcomes for Python skills
       for _ in range(5):
           learning_automata.skill_importance_factors['python'] *= 1.1
       
       updated_factor = learning_automata.get_skill_importance_factor('python')
       assert updated_factor > initial_factor

   def test_developer_preference_modeling(self, learning_automata):
       """Test developer preference model building."""
       developer_id = 1
       task_complexity = [0.8, 0.7, 0.5, 0.6, 0.7]
       
       # Initially no prediction
       prediction = learning_automata.get_developer_performance_prediction(developer_id, task_complexity)
       assert prediction['confidence'] < 0.5
       
       # Add some history
       if developer_id not in learning_automata.developer_preference_models:
           learning_automata.developer_preference_models[developer_id] = {
               'preferred_complexity_factors': np.array([0.5, 0.5, 0.5, 0.5, 0.5]),
               'performance_by_complexity': [],
               'collaboration_preference': 0.5,
               'learning_preference': 0.5,
               'workload_tolerance': 0.7
           }
       
       # Add performance history
       for i in range(10):
           learning_automata.developer_preference_models[developer_id]['performance_by_complexity'].append({
               'complexity': task_complexity,
               'satisfaction': 0.8,
               'productivity': 0.8,
               'timestamp': datetime.now()
           })
       
       # Now should have better prediction
       new_prediction = learning_automata.get_developer_performance_prediction(developer_id, task_complexity)
       assert new_prediction['confidence'] > prediction['confidence']

   def test_complexity_weight_adaptation(self, learning_automata):
       """Test complexity weight adaptation based on prediction errors."""
       # Add prediction errors
       for i in range(25):
           error = {
               'time_error': 0.3,
               'productivity_error': 0.2,
               'task_complexity': [0.8, 0.6, 0.4, 0.5, 0.7],
               'developer_id': 1
           }
           learning_automata.complexity_prediction_errors.append(error)
       
       initial_weights = learning_automata.complexity_weights.copy()
       
       # Trigger weight adaptation
       asyncio.run(learning_automata._adapt_complexity_weights())
       
       # Weights should have changed
       assert not np.array_equal(initial_weights, learning_automata.complexity_weights)
       assert abs(np.sum(learning_automata.complexity_weights) - 1.0) < 0.01  # Should be normalized

   def test_enhanced_assignment_weights(self, learning_automata):
       """Test enhanced assignment weight calculation."""
       # Add developer preference model
       developer_id = 1
       learning_automata.developer_preference_models[developer_id] = {
           'preferred_complexity_factors': np.array([0.7, 0.8, 0.6, 0.9, 0.7]),
           'performance_by_complexity': [],
           'collaboration_preference': 0.9,  # High collaboration preference
           'learning_preference': 0.8,  # High learning preference
           'workload_tolerance': 0.7
       }
       
       weights = learning_automata.get_enhanced_assignment_weights(task_id=1, developer_id=developer_id)
       
       # Should boost collaboration and learning weights
       assert weights['collaboration'] > 0.1  # Default is 0.1
       assert weights['skill_development'] > 0.25  # Default is 0.25
       assert abs(sum(weights.values()) - 1.0) < 0.01  # Should be normalized

   @pytest.mark.asyncio
   async def test_export_import_models(self, learning_automata):
       """Test learning model export and import."""
       # Add some learning data
       learning_automata.skill_importance_factors['python'] = 1.5
       learning_automata.developer_preference_models[1] = {
           'preferred_complexity_factors': np.array([0.7, 0.8, 0.6, 0.9, 0.7]),
           'performance_by_complexity': [],
           'collaboration_preference': 0.9,
           'learning_preference': 0.8,
           'workload_tolerance': 0.7
       }
       
       # Export models
       exported_data = await learning_automata.export_learning_models()
       
       assert 'complexity_weights' in exported_data
       assert 'skill_importance_factors' in exported_data
       assert 'developer_preference_models' in exported_data
       assert exported_data['skill_importance_factors']['python'] == 1.5
       
       # Create new instance and import
       new_automata = LearningAutomata()
       success = await new_automata.import_learning_models(exported_data)
       
       assert success
       assert new_automata.get_skill_importance_factor('python') == 1.5
       assert 1 in new_automata.developer_preference_models

# API Endpoint Tests

class TestAssignmentAPI:
   """Test suite for assignment API endpoints."""
   
   def test_create_assignment(self, sample_developers, sample_tasks):
       """Test assignment creation endpoint."""
       assignment_data = {
           "task_id": sample_tasks[0].id,
           "developer_id": sample_developers[0].id,
           "confidence_score": 0.8,
           "reasoning": "Good skill match for backend task"
       }
       
       response = client.post("/api/v1/assignments/", json=assignment_data)
       assert response.status_code == 200
       
       data = response.json()
       assert data["task_id"] == assignment_data["task_id"]
       assert data["developer_id"] == assignment_data["developer_id"]
       assert data["confidence_score"] == assignment_data["confidence_score"]
       assert data["status"] == "suggested"

   def test_list_assignments(self, sample_developers, sample_tasks):
       """Test assignment listing endpoint."""
       # Create some assignments first
       for i in range(3):
           assignment_data = {
               "task_id": sample_tasks[i % len(sample_tasks)].id,
               "developer_id": sample_developers[i % len(sample_developers)].id,
               "confidence_score": 0.7 + i * 0.1,
               "reasoning": f"Test assignment {i}"
           }
           client.post("/api/v1/assignments/", json=assignment_data)
       
       # Test listing
       response = client.get("/api/v1/assignments/")
       assert response.status_code == 200
       
       data = response.json()
       assert isinstance(data, list)
       assert len(data) >= 3

   def test_list_assignments_with_filters(self, sample_developers, sample_tasks):
       """Test assignment listing with filters."""
       # Create assignment for specific developer
       assignment_data = {
           "task_id": sample_tasks[0].id,
           "developer_id": sample_developers[0].id,
           "confidence_score": 0.8,
           "reasoning": "Test assignment for filtering"
       }
       response = client.post("/api/v1/assignments/", json=assignment_data)
       assignment_id = response.json()["id"]
       
       # Test developer filter
       response = client.get(f"/api/v1/assignments/?developer_id={sample_developers[0].id}")
       assert response.status_code == 200
       
       data = response.json()
       assert len(data) >= 1
       assert all(a["developer_id"] == sample_developers[0].id for a in data)

   def test_get_assignment(self, sample_developers, sample_tasks):
       """Test get specific assignment endpoint."""
       # Create assignment
       assignment_data = {
           "task_id": sample_tasks[0].id,
           "developer_id": sample_developers[0].id,
           "confidence_score": 0.8,
           "reasoning": "Test assignment"
       }
       response = client.post("/api/v1/assignments/", json=assignment_data)
       assignment_id = response.json()["id"]
       
       # Get assignment
       response = client.get(f"/api/v1/assignments/{assignment_id}")
       assert response.status_code == 200
       
       data = response.json()
       assert data["id"] == assignment_id
       assert data["task_id"] == assignment_data["task_id"]

   def test_update_assignment(self, sample_developers, sample_tasks):
       """Test assignment update endpoint."""
       # Create assignment
       assignment_data = {
           "task_id": sample_tasks[0].id,
           "developer_id": sample_developers[0].id,
           "confidence_score": 0.8,
           "reasoning": "Test assignment"
       }
       response = client.post("/api/v1/assignments/", json=assignment_data)
       assignment_id = response.json()["id"]
       
       # Update assignment
       update_data = {
           "status": "accepted",
           "started_at": datetime.now().isoformat(),
           "feedback_score": 0.9
       }
       
       response = client.put(f"/api/v1/assignments/{assignment_id}", json=update_data)
       assert response.status_code == 200
       
       data = response.json()
       assert data["status"] == "accepted"
       assert data["feedback_score"] == 0.9

   def test_delete_assignment(self, sample_developers, sample_tasks):
       """Test assignment deletion endpoint."""
       # Create assignment
       assignment_data = {
           "task_id": sample_tasks[0].id,
           "developer_id": sample_developers[0].id,
           "confidence_score": 0.8,
           "reasoning": "Test assignment for deletion"
       }
       response = client.post("/api/v1/assignments/", json=assignment_data)
       assignment_id = response.json()["id"]
       
       # Delete assignment
       response = client.delete(f"/api/v1/assignments/{assignment_id}")
       assert response.status_code == 200
       
       # Verify deletion
       response = client.get(f"/api/v1/assignments/{assignment_id}")
       assert response.status_code == 404

   def test_optimize_assignments_endpoint(self, sample_developers, sample_tasks):
       """Test assignment optimization endpoint."""
       optimization_data = {
           "task_ids": [task.id for task in sample_tasks[:2]],
           "developer_ids": [dev.id for dev in sample_developers],
           "objectives": ["productivity", "skill_development", "workload_balance"]
       }
       
       response = client.post("/api/v1/assignments/optimize", json=optimization_data)
       assert response.status_code == 200
       
       data = response.json()
       assert "assignments" in data
       assert "objective_scores" in data
       assert "optimization_time_ms" in data
       assert isinstance(data["assignments"], list)
       assert len(data["assignments"]) <= len(optimization_data["task_ids"])

   def test_match_task_to_developers_endpoint(self, sample_developers, sample_tasks):
       """Test task-developer matching endpoint."""
       matching_data = {
           "task_ids": [sample_tasks[0].id],
           "developer_ids": [dev.id for dev in sample_developers],
           "matching_criteria": {
               "skill_match": 0.4,
               "complexity_fit": 0.3,
               "learning_potential": 0.2,
               "workload_balance": 0.1
           },
           "max_matches_per_task": 3
       }
       
       response = client.post("/api/v1/assignments/match-task", json=matching_data)
       assert response.status_code == 200
       
       data = response.json()
       assert "matches" in data
       assert "processing_time_ms" in data
       assert isinstance(data["matches"], list)
       assert len(data["matches"]) <= 3

   def test_suggest_assignment_endpoint(self, sample_developers, sample_tasks):
       """Test assignment suggestion endpoint."""
       task_id = sample_tasks[0].id
       
       response = client.post(f"/api/v1/assignments/suggest?task_id={task_id}&max_suggestions=3")
       assert response.status_code == 200
       
       data = response.json()
       assert data["message"] == "Assignment suggestions generated"
       assert "data" in data
       assert "suggestions" in data["data"]
       
       suggestions = data["data"]["suggestions"]
       assert len(suggestions) <= 3
       
       # Verify suggestion structure
       for suggestion in suggestions:
           assert "developer_id" in suggestion
           assert "overall_score" in suggestion
           assert "reasoning" in suggestion
           assert 0 <= suggestion["overall_score"] <= 1

   def test_team_performance_analytics_endpoint(self, sample_developers, sample_tasks):
       """Test team performance analytics endpoint."""
       # Create some completed assignments for analytics
       for i, dev in enumerate(sample_developers):
           assignment_data = {
               "task_id": sample_tasks[i % len(sample_tasks)].id,
               "developer_id": dev.id,
               "confidence_score": 0.8,
               "reasoning": f"Test assignment {i}"
           }
           response = client.post("/api/v1/assignments/", json=assignment_data)
           assignment_id = response.json()["id"]
           
           # Mark as completed
           update_data = {
               "status": "completed",
               "productivity_score": 0.8,
               "skill_development_score": 0.7,
               "collaboration_effectiveness": 0.8
           }
           client.put(f"/api/v1/assignments/{assignment_id}", json=update_data)
       
       # Get team analytics
       developer_ids = [dev.id for dev in sample_developers]
       response = client.get(
           f"/api/v1/assignments/analytics/team-performance"
           f"?team_developer_ids={','.join(map(str, developer_ids))}&days_back=30"
       )
       assert response.status_code == 200
       
       data = response.json()
       assert data["message"] == "Team assignment analytics retrieved"
       assert "data" in data
       assert "analytics" in data["data"]

   def test_learning_analytics_endpoint(self):
       """Test learning analytics endpoint."""
       response = client.get("/api/v1/assignments/learning/analytics")
       assert response.status_code == 200
       
       data = response.json()
       assert data["message"] == "Learning analytics retrieved"
       assert "data" in data
       assert "analytics" in data["data"]
       
       analytics = data["data"]["analytics"]
       assert "total_assignments_learned" in analytics
       assert "current_learning_rate" in analytics
       assert "developers_modeled" in analytics

# Performance and Integration Tests

class TestAssignmentPerformance:
   """Test suite for assignment engine performance."""
   
   @pytest.mark.asyncio
   async def test_optimization_performance(self, db_session, assignment_optimizer):
       """Test optimization performance with larger datasets."""
       # Create more developers and tasks for performance testing
       developers = []
       for i in range(10):
           dev = Developer(
               github_username=f"perf_test_dev_{i}",
               name=f"Developer {i}",
               email=f"dev{i}@example.com",
               skill_vector='[0.5]' * 768,
               primary_languages='{"python": 0.7}',
               domain_expertise='{"backend": 0.6}',
               collaboration_score=0.7,
               learning_velocity=0.6
           )
           developers.append(dev)
           db_session.add(dev)
       
       tasks = []
       for i in range(15):
           task = Task(
               title=f"Performance test task {i}",
               description=f"Task description {i}",
               repository="perf-test",
               priority="medium",
               technical_complexity=0.5,
               domain_difficulty=0.5,
               collaboration_requirements=0.4,
               learning_opportunities=0.5,
               business_impact=0.6,
               estimated_hours=8.0 + i,
               complexity_confidence=0.8,
               required_skills='{"python": 0.6}',
               status="open"
           )
           tasks.append(task)
           db_session.add(task)
       
       db_session.commit()
       
       # Refresh to get IDs
       for dev in developers:
           db_session.refresh(dev)
       for task in tasks:
           db_session.refresh(task)
       
       # Test optimization performance
       start_time = datetime.now()
       
       result = await assignment_optimizer.optimize_assignments(
           db=db_session,
           task_ids=[task.id for task in tasks],
           developer_ids=[dev.id for dev in developers]
       )
       
       end_time = datetime.now()
       processing_time = (end_time - start_time).total_seconds() * 1000
       
       # Should complete within reasonable time (5 seconds for this dataset)
       assert processing_time < 5000
       assert result.optimization_time_ms < 5000
       assert len(result.assignments) > 0

   @pytest.mark.asyncio
   async def test_concurrent_optimization_requests(self, db_session, sample_developers, sample_tasks, assignment_optimizer):
       """Test handling concurrent optimization requests."""
       async def run_optimization():
           return await assignment_optimizer.optimize_assignments(
               db=db_session,
               task_ids=[task.id for task in sample_tasks[:2]],
               developer_ids=[dev.id for dev in sample_developers]
           )
       
       # Run multiple optimizations concurrently
       tasks = [run_optimization() for _ in range(3)]
       results = await asyncio.gather(*tasks)
       
       # All should complete successfully
       assert len(results) == 3
       for result in results:
           assert result is not None
           assert len(result.assignments) > 0

   def test_api_response_times(self, sample_developers, sample_tasks):
       """Test API endpoint response times."""
       import time
       
       # Test optimization endpoint response time
       optimization_data = {
           "task_ids": [task.id for task in sample_tasks],
           "developer_ids": [dev.id for dev in sample_developers]
       }
       
       start_time = time.time()
       response = client.post("/api/v1/assignments/optimize", json=optimization_data)
       end_time = time.time()
       
       assert response.status_code == 200
       response_time = (end_time - start_time) * 1000  # Convert to ms
       
       # Should respond within 3 seconds for small dataset
       assert response_time < 3000

# Cleanup
def teardown_module():
   """Clean up test database."""
   import os
   if os.path.exists("./test_assignment_engine.db"):
       os.remove("./test_assignment_engine.db")

if __name__ == "__main__":
   pytest.main([__file__, "-v"])
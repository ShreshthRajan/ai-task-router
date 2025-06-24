import pytest
import asyncio
import sys
import os
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from unittest.mock import Mock, patch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.core.learning_system.feedback_processor import FeedbackProcessor
from src.core.learning_system.model_updater import ModelUpdater
from src.core.learning_system.system_analytics import SystemAnalytics
from src.models.database import (
    TaskAssignment, AssignmentOutcome, Developer, Task,
    ModelPerformance, DeveloperPreference, SkillImportanceFactor
)
from src.models.schemas import AssignmentOutcomeCreate, LearningExperimentCreate

@pytest.fixture
def db_session():
    import tempfile
    from sqlalchemy import create_engine
    from src.models.database import Base, SessionLocal
    
    # Use in-memory SQLite for tests
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    
    # Create session
    from sqlalchemy.orm import sessionmaker
    TestSession = sessionmaker(bind=engine)
    session = TestSession()
    
    yield session
    session.close()

@pytest.fixture
def feedback_processor():
    return FeedbackProcessor()

@pytest.fixture
def model_updater():
    return ModelUpdater()

@pytest.fixture
def system_analytics():
    return SystemAnalytics()

@pytest.fixture
def sample_outcomes():
    return [
        AssignmentOutcomeCreate(
            assignment_id=1,
            task_completion_quality=0.8,
            developer_satisfaction=0.9,
            learning_achieved=0.7,
            collaboration_effectiveness=0.8,
            time_estimation_accuracy=0.75,
            performance_metrics={"code_quality": 0.85},
            skill_improvements=["python", "api_design"],
            challenges_faced=["complex_algorithm"],
            success_factors=["good_requirements", "mentor_support"]
        ),
        AssignmentOutcomeCreate(
            assignment_id=2,
            task_completion_quality=0.6,
            developer_satisfaction=0.5,
            learning_achieved=0.4,
            collaboration_effectiveness=0.6,
            time_estimation_accuracy=0.5,
            performance_metrics={"code_quality": 0.55},
            skill_improvements=["debugging"],
            challenges_faced=["unclear_requirements", "time_pressure"],
            success_factors=["peer_support"]
        )
    ]

class TestFeedbackProcessor:
    
    @pytest.mark.asyncio
    async def test_process_assignment_outcomes(self, feedback_processor, db_session, sample_outcomes):
        """Test processing assignment outcomes."""
        # Setup test data
        developer = Developer(id=1, github_username="test_dev", name="Test Developer")
        task = Task(id=1, title="Test Task", technical_complexity=0.7)
        assignment1 = TaskAssignment(id=1, task_id=1, developer_id=1)
        assignment2 = TaskAssignment(id=2, task_id=1, developer_id=1)
        
        db_session.add_all([developer, task, assignment1, assignment2])
        db_session.commit()
        
        # Process outcomes
        result = await feedback_processor.process_assignment_outcomes(
            db_session, sample_outcomes, update_models=True
        )
        
        # Verify results
        assert result.outcomes_processed == 2
        assert result.outcomes_processed == 2
        assert result.processing_time_ms > 0
        assert len(result.system_improvements) >= 0
       
        # Verify outcomes were stored
        stored_outcomes = db_session.query(AssignmentOutcome).all()
        assert len(stored_outcomes) == 2
        
        # Verify developer preferences were learned
        preferences = db_session.query(DeveloperPreference).filter(
            DeveloperPreference.developer_id == 1
        ).first()
        assert preferences is not None
        assert preferences.sample_size > 0

    @pytest.mark.asyncio
    async def test_learn_developer_preferences(self, feedback_processor, db_session, sample_outcomes):
       """Test learning individual developer preferences."""
       # Setup test data
       developer = Developer(id=1, github_username="test_dev")
       task = Task(id=1, title="Test Task", technical_complexity=0.6, domain_difficulty=0.5, collaboration_requirements=0.4)
       
       # Create multiple assignments for learning
       assignments = []
       for i in range(6):  # More than min_sample_size
           assignment = TaskAssignment(id=i+1, task_id=1, developer_id=1)
           assignments.append(assignment)
       
       db_session.add_all([developer, task] + assignments)
       db_session.commit()
       
       # Create outcomes with varying satisfaction
       outcomes = []
       for i, assignment in enumerate(assignments):
           outcome = AssignmentOutcomeCreate(
               assignment_id=assignment.id,
               task_completion_quality=0.7 + i*0.05,
               developer_satisfaction=0.6 + i*0.06,
               learning_achieved=0.5 + i*0.08,
               collaboration_effectiveness=0.7,
               time_estimation_accuracy=0.8
           )
           outcomes.append(outcome)
       
       # Process outcomes
       result = await feedback_processor.process_assignment_outcomes(db_session, outcomes)
       
       # Verify preferences were learned
       preferences = db_session.query(DeveloperPreference).filter(
           DeveloperPreference.developer_id == 1
       ).first()
       
       assert preferences is not None
       assert preferences.preference_confidence > 0.0
       assert 0.0 <= preferences.preferred_complexity_min <= 1.0
       assert 0.0 <= preferences.preferred_complexity_max <= 1.0
       assert preferences.preferred_complexity_min <= preferences.preferred_complexity_max

    @pytest.mark.asyncio
    async def test_update_skill_importance_factors(self, feedback_processor, db_session):
       """Test updating skill importance factors."""
       # Setup test data
       developer = Developer(
           id=1, 
           github_username="test_dev",
           primary_languages={"python": 0.8, "javascript": 0.6}
       )
       task = Task(
           id=1, 
           title="Python API Development",
           description="Build REST API using Python",
           technical_complexity=0.7
       )
       assignment = TaskAssignment(id=1, task_id=1, developer_id=1)
       
       db_session.add_all([developer, task, assignment])
       db_session.commit()
       
       # Create outcome
       outcome = AssignmentOutcomeCreate(
           assignment_id=1,
           task_completion_quality=0.9,
           developer_satisfaction=0.8,
           learning_achieved=0.7,
           collaboration_effectiveness=0.8,
           time_estimation_accuracy=0.8
       )
       
       # Process outcome
       await feedback_processor.process_assignment_outcomes(db_session, [outcome])
       
       # Verify skill importance factors were created/updated
       factors = db_session.query(SkillImportanceFactor).all()
       assert len(factors) > 0
       
       python_factor = next((f for f in factors if f.skill_name == "python"), None)
       assert python_factor is not None
       assert python_factor.total_assignments > 0

    @pytest.mark.asyncio
    async def test_analyze_developer_performance_patterns(self, feedback_processor, db_session):
       """Test analyzing developer performance patterns."""
       # Setup test data with sufficient history
       developer = Developer(id=1, github_username="test_dev")
       task = Task(id=1, title="Test Task", technical_complexity=0.6)
       
       db_session.add_all([developer, task])
       
       # Create assignment outcomes
       for i in range(8):  # More than min_sample_size
           assignment = TaskAssignment(id=i+1, task_id=1, developer_id=1)
           outcome = AssignmentOutcome(
               assignment_id=i+1,
               task_completion_quality=0.7 + (i % 3) * 0.1,
               developer_satisfaction=0.8,
               learning_achieved=0.6,
               collaboration_effectiveness=0.7,
               time_estimation_accuracy=0.75
           )
           db_session.add(assignment)
           db_session.add(outcome)
       
       db_session.commit()
       
       # Analyze patterns
       profile = await feedback_processor.analyze_developer_performance_patterns(db_session, 1)
       
       assert profile.developer_id == 1
       assert profile.sample_size > 0
       assert 0.0 <= profile.preferred_complexity_range[0] <= 1.0
       assert 0.0 <= profile.preferred_complexity_range[1] <= 1.0
       assert 0.0 <= profile.collaboration_preference <= 1.0
       assert 0.0 <= profile.learning_appetite <= 1.0

class TestModelUpdater:
   @pytest.mark.asyncio
   async def test_update_models_from_outcomes(self, model_updater, db_session):
       """Test updating models from assignment outcomes."""
       # Setup test data
       developer = Developer(id=1, github_username="test_dev")
       task = Task(
           id=1, 
           title="Test Task",
           technical_complexity=0.7,
           domain_difficulty=0.6,
           collaboration_requirements=0.5
       )
       
       # Create assignments and outcomes
       assignments_data = []
       for i in range(25):  # More than min_update_samples
           assignment = TaskAssignment(id=i+1, task_id=1, developer_id=1)
           outcome = AssignmentOutcome(
               assignment_id=i+1,
               task_completion_quality=0.6 + (i % 4) * 0.1,
               developer_satisfaction=0.7,
               learning_achieved=0.5,
               collaboration_effectiveness=0.8,
               time_estimation_accuracy=0.7
           )
           assignments_data.extend([assignment, outcome])
       
       db_session.add_all([developer, task] + assignments_data)
       db_session.commit()
       
       # Update models
       results = await model_updater.update_models_from_outcomes(db_session, force_update=True)
       
       # Verify results
       assert isinstance(results, list)
       # At least one model should be updated with force_update=True
       if results:
           assert all(isinstance(r.model_name, str) for r in results)
           assert all(isinstance(r.new_version, str) for r in results)

   @pytest.mark.asyncio
   async def test_create_learning_experiment(self, model_updater, db_session):
       """Test creating learning experiments."""
       experiment_data = LearningExperimentCreate(
           experiment_name="test_assignment_weights",
           experiment_type="ab_test",
           control_config={"productivity_weight": 0.35, "learning_weight": 0.25},
           experimental_config={"productivity_weight": 0.40, "learning_weight": 0.30},
           success_metrics=["assignment_success_rate", "developer_satisfaction"],
           sample_size=50
       )
       
       # Create experiment
       experiment = await model_updater.create_learning_experiment(db_session, experiment_data)
       
       assert experiment.experiment_name == "test_assignment_weights"
       assert experiment.experiment_type == "ab_test"
       assert experiment.status == "active"
       assert experiment.sample_size == 50

   @pytest.mark.asyncio
   async def test_run_ab_test(self, model_updater, db_session):
       """Test running A/B tests."""
       # Create experiment first
       experiment_data = LearningExperimentCreate(
           experiment_name="test_ab_experiment",
           experiment_type="ab_test",
           control_config={"test": "control"},
           experimental_config={"test": "experimental"},
           success_metrics=["test_metric"],
           sample_size=20
       )
       
       experiment = await model_updater.create_learning_experiment(db_session, experiment_data)
       
       # Start A/B test
       success = await model_updater.run_ab_test(db_session, experiment.id, duration_days=7)
       
       assert success is True
       
       # Verify experiment was updated
       db_session.refresh(experiment)
       assert experiment.status == "active"
       assert experiment.start_date is not None
       assert experiment.end_date is not None

   @pytest.mark.asyncio
   async def test_rollback_model(self, model_updater, db_session):
       """Test model rollback functionality."""
       # Create model performance records
       perf1 = ModelPerformance(
           model_name="test_model",
           version="1.0",
           accuracy_score=0.75,
           created_at=datetime.utcnow() - timedelta(days=1)
       )
       perf2 = ModelPerformance(
           model_name="test_model", 
           version="1.1",
           accuracy_score=0.65,  # Worse performance
           created_at=datetime.utcnow()
       )
       
       db_session.add_all([perf1, perf2])
       db_session.commit()
       
       # Rollback to previous version
       success = await model_updater.rollback_model(db_session, "test_model", "1.0")
       
       assert success is True
       
       # Verify rollback record was created
       rollback_records = db_session.query(ModelPerformance).filter(
           ModelPerformance.model_name == "test_model"
       ).filter(
           ModelPerformance.version.contains("rollback")
       ).all()
       
       assert len(rollback_records) > 0

class TestSystemAnalytics:
   
   @pytest.mark.asyncio
   async def test_get_system_health_metrics(self, system_analytics, db_session):
       """Test getting system health metrics."""
       # Setup test data
       developer = Developer(id=1, github_username="test_dev")
       task = Task(id=1, title="Test Task")
       
       # Create recent assignments and outcomes
       for i in range(10):
           assignment = TaskAssignment(
               id=i+1, 
               task_id=1, 
               developer_id=1,
               assigned_at=datetime.utcnow() - timedelta(days=i),
               status="completed"
           )
           outcome = AssignmentOutcome(
               assignment_id=i+1,
               task_completion_quality=0.7 + (i % 3) * 0.1,
               developer_satisfaction=0.8,
               learning_achieved=0.6,
               collaboration_effectiveness=0.75,
               time_estimation_accuracy=0.7
           )
           db_session.add_all([assignment, outcome])
       
       db_session.add_all([developer, task])
       db_session.commit()
       
       # Get health metrics
       metrics = await system_analytics.get_system_health_metrics(db_session)
       
       assert 0.0 <= metrics.avg_assignment_quality <= 1.0
       assert 0.0 <= metrics.avg_developer_satisfaction <= 1.0
       assert 0.0 <= metrics.avg_skill_development_rate <= 1.0
       assert 0.0 <= metrics.assignment_success_rate <= 1.0
       assert metrics.total_assignments > 0
       assert metrics.completed_assignments > 0

   @pytest.mark.asyncio
   async def test_get_learning_system_analytics(self, system_analytics, db_session):
       """Test getting learning system analytics."""
       # Setup test data
       outcome = AssignmentOutcome(
           assignment_id=1,
           task_completion_quality=0.8,
           developer_satisfaction=0.7,
           learning_achieved=0.6,
           collaboration_effectiveness=0.8,
           time_estimation_accuracy=0.75
       )
       
       model_perf = ModelPerformance(
           model_name="test_model",
           version="1.0",
           accuracy_score=0.85
       )
       
       pref = DeveloperPreference(developer_id=1)
       factor = SkillImportanceFactor(
           skill_name="python",
           task_type="feature_development",
           complexity_range="medium",
           domain="backend"
       )
       
       db_session.add_all([outcome, model_perf, pref, factor])
       db_session.commit()
       
       # Get analytics
       analytics = await system_analytics.get_learning_system_analytics(db_session)
       
       assert analytics.total_outcomes_processed > 0
       assert analytics.developer_preferences_learned > 0
       assert analytics.skill_importance_factors > 0
       assert isinstance(analytics.model_performance_trends, dict)
       assert isinstance(analytics.recent_learnings, list)

   @pytest.mark.asyncio
   async def test_generate_predictive_insights(self, system_analytics, db_session):
       """Test generating predictive insights."""
       # Setup comprehensive test data
       developer = Developer(id=1, github_username="test_dev")
       task = Task(id=1, title="Test Task")
       
       # Create assignment history with outcomes
       for i in range(8):
           assignment = TaskAssignment(
               id=i+1,
               task_id=1,
               developer_id=1,
               assigned_at=datetime.utcnow() - timedelta(days=i*2)
           )
           outcome = AssignmentOutcome(
               assignment_id=i+1,
               task_completion_quality=0.6 + i*0.05,  # Improving trend
               developer_satisfaction=0.7 + i*0.03,
               learning_achieved=0.5 + i*0.04,
               collaboration_effectiveness=0.8,
               time_estimation_accuracy=0.7
           )
           db_session.add_all([assignment, outcome])
       
       # Add developer preferences
       preferences = DeveloperPreference(
           developer_id=1,
           preferred_complexity_min=0.4,
           preferred_complexity_max=0.8,
           learning_appetite=0.7,
           preference_confidence=0.8
       )
       
       db_session.add_all([developer, task, preferences])
       db_session.commit()
       
       # Generate insights
       insights = await system_analytics.generate_predictive_insights(db_session, developer_id=1)
       
       assert len(insights) > 0
       insight = insights[0]
       assert insight.developer_id == 1
       assert insight.predicted_performance_trend in ["improving", "stable", "declining"]
       assert 0.0 <= insight.confidence <= 1.0
       assert isinstance(insight.skill_development_forecast, dict)
       assert isinstance(insight.risk_factors, list)
       assert isinstance(insight.recommendations, list)

   @pytest.mark.asyncio
   async def test_generate_optimization_suggestions(self, system_analytics, db_session):
       """Test generating system optimization suggestions."""
       # Setup test data with poor performance to trigger suggestions
       developer = Developer(id=1, github_username="test_dev")
       task = Task(id=1, title="Test Task")
       
       # Create assignments with low success rate
       for i in range(15):
           assignment = TaskAssignment(
               id=i+1,
               task_id=1,
               developer_id=1,
               assigned_at=datetime.utcnow() - timedelta(days=i),
               status="completed"
           )
           outcome = AssignmentOutcome(
               assignment_id=i+1,
               task_completion_quality=0.4,  # Low quality to trigger alerts
               developer_satisfaction=0.5,   # Low satisfaction
               learning_achieved=0.3,
               collaboration_effectiveness=0.6,
               time_estimation_accuracy=0.5
           )
           db_session.add_all([assignment, outcome])
       
       db_session.add_all([developer, task])
       db_session.commit()
       
       # Generate suggestions
       suggestions = await system_analytics.generate_optimization_suggestions(db_session)
       
       assert len(suggestions) > 0
       for suggestion in suggestions:
           assert suggestion.optimization_type in [
               "algorithm_tuning", "workflow_improvement", "resource_allocation"
           ]
           assert 0.0 <= suggestion.current_performance <= 1.0
           assert suggestion.expected_improvement > 0.0
           assert suggestion.implementation_effort in ["low", "medium", "high"]
           assert len(suggestion.impact_areas) > 0
           assert len(suggestion.description) > 0
           assert 0.0 <= suggestion.confidence <= 1.0

   @pytest.mark.asyncio
   async def test_detect_performance_alerts(self, system_analytics, db_session):
       """Test detecting performance alerts."""
       # Setup test data with poor performance
       developer = Developer(id=1, github_username="test_dev")
       task = Task(id=1, title="Test Task")
       
       # Create assignments with very low performance
       for i in range(10):
           assignment = TaskAssignment(
               id=i+1,
               task_id=1,
               developer_id=1,
               assigned_at=datetime.utcnow() - timedelta(days=i),
               status="completed"
           )
           outcome = AssignmentOutcome(
               assignment_id=i+1,
               task_completion_quality=0.3,  # Very low to trigger alerts
               developer_satisfaction=0.4,
               learning_achieved=0.2,
               collaboration_effectiveness=0.5,
               time_estimation_accuracy=0.4
           )
           db_session.add_all([assignment, outcome])
       
       db_session.add_all([developer, task])
       db_session.commit()
       
       # Detect alerts
       alerts = await system_analytics.detect_performance_alerts(db_session)
       
       assert len(alerts) > 0
       for alert in alerts:
           assert "type" in alert
           assert "severity" in alert
           assert "message" in alert
           assert "recommendation" in alert
           assert "timestamp" in alert
           assert alert["severity"] in ["low", "medium", "high"]

   @pytest.mark.asyncio
   async def test_get_team_performance_metrics(self, system_analytics, db_session):
       """Test getting team performance metrics."""
       # Setup test data
       developers = [
           Developer(id=1, github_username="dev1"),
           Developer(id=2, github_username="dev2"),
           Developer(id=3, github_username="dev3")
       ]
       task = Task(id=1, title="Test Task")
       
       # Create assignments for multiple developers
       for dev_id in [1, 2, 3]:
           for i in range(5):
               assignment = TaskAssignment(
                   id=dev_id*10 + i,
                   task_id=1,
                   developer_id=dev_id,
                   assigned_at=datetime.utcnow() - timedelta(days=i),
                   status="completed",
                   actual_hours=8.0 + i
               )
               outcome = AssignmentOutcome(
                   assignment_id=dev_id*10 + i,
                   task_completion_quality=0.7 + dev_id*0.05,
                   developer_satisfaction=0.8,
                   learning_achieved=0.6,
                   collaboration_effectiveness=0.75,
                   time_estimation_accuracy=0.7
               )
               db_session.add_all([assignment, outcome])
       
       db_session.add_all(developers + [task])
       db_session.commit()
       
       # Get team metrics
       metrics = await system_analytics.get_team_performance_metrics(db_session)
       
       assert metrics.team_size == 3
       assert 0.0 <= metrics.avg_assignment_score <= 1.0
       assert 0.0 <= metrics.skill_development_rate <= 1.0
       assert 0.0 <= metrics.collaboration_effectiveness <= 1.0
       assert 0.0 <= metrics.workload_balance_score <= 1.0
       assert 0.0 <= metrics.completion_rate <= 1.0
       assert metrics.average_delivery_time_hours is not None
       assert metrics.average_delivery_time_hours > 0

   @pytest.mark.asyncio
   async def test_generate_roi_report(self, system_analytics, db_session):
       """Test generating ROI reports."""
       # Setup test data
       developer = Developer(id=1, github_username="test_dev")
       task = Task(id=1, title="Test Task")
       
       # Create successful assignments
       for i in range(20):
           assignment = TaskAssignment(
               id=i+1,
               task_id=1,
               developer_id=1,
               assigned_at=datetime.utcnow() - timedelta(days=i),
               status="completed"
           )
           outcome = AssignmentOutcome(
               assignment_id=i+1,
               task_completion_quality=0.85,  # High quality for good ROI
               developer_satisfaction=0.9,
               learning_achieved=0.7,
               collaboration_effectiveness=0.8,
               time_estimation_accuracy=0.8
           )
           db_session.add_all([assignment, outcome])
       
       db_session.add_all([developer, task])
       db_session.commit()
       
       # Generate ROI report
       report = await system_analytics.generate_roi_report(db_session, period_days=30)
       
       assert "period_days" in report
       assert "total_assignments" in report
       assert "success_rate_improvement" in report
       assert "satisfaction_improvement" in report
       assert "skill_development_improvement" in report
       assert "estimated_time_saved_hours" in report
       assert "estimated_cost_savings_usd" in report
       assert "velocity_improvement_percent" in report
       assert "roi_metrics" in report
       assert "generated_at" in report
       
       assert report["period_days"] == 30
       assert report["total_assignments"] > 0
       assert isinstance(report["roi_metrics"], dict)

# Performance Tests
class TestLearningSystemPerformance:
   
   @pytest.mark.asyncio
   async def test_feedback_processing_performance(self, feedback_processor, db_session):
       """Test feedback processing performance with large datasets."""
       # Setup large dataset
       developer = Developer(id=1, github_username="test_dev")
       task = Task(id=1, title="Test Task", technical_complexity=0.7)
       
       assignments = []
       outcomes = []
       
       # Create 100 assignments and outcomes
       for i in range(100):
           assignment = TaskAssignment(id=i+1, task_id=1, developer_id=1)
           assignments.append(assignment)
           
           outcome = AssignmentOutcomeCreate(
               assignment_id=i+1,
               task_completion_quality=0.6 + (i % 4) * 0.1,
               developer_satisfaction=0.7 + (i % 3) * 0.1,
               learning_achieved=0.5 + (i % 5) * 0.1,
               collaboration_effectiveness=0.8,
               time_estimation_accuracy=0.7
           )
           outcomes.append(outcome)
       
       db_session.add_all([developer, task] + assignments)
       db_session.commit()
       
       # Time the processing
       start_time = datetime.now()
       result = await feedback_processor.process_assignment_outcomes(
           db_session, outcomes, update_models=False  # Skip model updates for speed
       )
       processing_time = (datetime.now() - start_time).total_seconds()
       
       # Verify performance
       assert processing_time < 10.0  # Should complete in under 10 seconds
       assert result.outcomes_processed == 100
       assert result.processing_time_ms > 0

   @pytest.mark.asyncio
   async def test_analytics_performance(self, system_analytics, db_session):
       """Test analytics performance with large datasets."""
       # Setup large dataset
       developers = [Developer(id=i, github_username=f"dev{i}") for i in range(1, 21)]  # 20 developers
       task = Task(id=1, title="Test Task")
       
       # Create 500 assignments
       assignments_data = []
       for i in range(500):
           dev_id = (i % 20) + 1
           assignment = TaskAssignment(
               id=i+1,
               task_id=1,
               developer_id=dev_id,
               assigned_at=datetime.utcnow() - timedelta(days=i//10),
               status="completed"
           )
           outcome = AssignmentOutcome(
               assignment_id=i+1,
               task_completion_quality=0.6 + (i % 4) * 0.1,
               developer_satisfaction=0.7,
               learning_achieved=0.5,
               collaboration_effectiveness=0.8,
               time_estimation_accuracy=0.7
           )
           assignments_data.extend([assignment, outcome])
       
       db_session.add_all(developers + [task] + assignments_data)
       db_session.commit()
       
       # Time analytics operations
       start_time = datetime.now()
       
       health_metrics = await system_analytics.get_system_health_metrics(db_session)
       learning_analytics = await system_analytics.get_learning_system_analytics(db_session)
       team_metrics = await system_analytics.get_team_performance_metrics(db_session)
       
       processing_time = (datetime.now() - start_time).total_seconds()
       
       # Verify performance
       assert processing_time < 5.0  # Should complete in under 5 seconds
       assert health_metrics.total_assignments > 0
       assert team_metrics.team_size == 20

if __name__ == "__main__":
   pytest.main([__file__])
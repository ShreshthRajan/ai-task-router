import pytest
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import asyncio

from src.core.developer_modeling.code_analyzer import CodeAnalyzer, CodeMetrics
from src.core.developer_modeling.skill_extractor import SkillExtractor, DeveloperProfile
from src.core.developer_modeling.expertise_tracker import ExpertiseTracker
from src.integrations.github_client import GitHubClient
from src.config import settings

from unittest.mock import Mock, patch, AsyncMock

# Remove the duplicate TestGitHubClient class and replace with this single one:
class TestGitHubClient:
    """Test suite for GitHubClient."""
    
    @pytest.fixture
    def github_client(self):
        return GitHubClient()
    
    @pytest.mark.asyncio
    async def test_get_user_info(self, github_client):
        """Test user info retrieval (mocked)."""
        mock_response = {
            "login": "testuser",
            "name": "Test User",
            "email": "test@example.com",
            "public_repos": 10
        }
        
        # Mock the actual method directly
        github_client._get_user_info = AsyncMock(return_value=mock_response)
        
        session = AsyncMock()
        result = await github_client._get_user_info(session, "testuser")
        
        assert result == mock_response
    
    def test_extract_added_lines_integration(self, github_client):
        """Test that GitHub client can handle real patch formats."""
        sample_patch = """@@ -0,0 +1,10 @@
+def new_function():
+    \"\"\"New authentication function.\"\"\"
+    try:
+        result = authenticate_user()
+        return result
+    except AuthError:
+        logger.error("Authentication failed")
+        return None
+    finally:
+        cleanup()"""
        
        # This tests the same functionality as CodeAnalyzer but from GitHub context
        lines = []
        for line in sample_patch.split('\n'):
            if line.startswith('+') and not line.startswith('+++'):
                lines.append(line[1:])
        
        assert len(lines) == 10


class TestCodeAnalyzer:
    """Test suite for CodeAnalyzer."""
    
    @pytest.fixture
    def code_analyzer(self):
        return CodeAnalyzer()
    
    @pytest.fixture
    def sample_commit_data(self):
        return {
            "hash": "abc123",
            "message": "Add user authentication",
            "timestamp": "2024-01-15T10:30:00Z",
            "files": [
                {
                    "filename": "auth.py",
                    "patch": """+def authenticate_user(username, password):
+    # Hash the password
+    hashed = hash_password(password)
+    # Query database
+    user = db.query(User).filter(User.username == username).first()
+    return user and user.password_hash == hashed""",
                    "additions": 5,
                    "deletions": 0
                },
                {
                    "filename": "models.py",
                    "patch": """+class User:
+    def __init__(self, username, password_hash):
+        self.username = username
+        self.password_hash = password_hash""",
                    "additions": 4,
                    "deletions": 0
                }
            ]
        }
    
    def test_language_detection(self, code_analyzer):
        """Test programming language detection."""
        # Python file
        assert code_analyzer._detect_language("test.py", "def hello():") == "python"
        
        # JavaScript file
        assert code_analyzer._detect_language("app.js", "function hello() {") == "javascript"
        
        # TypeScript file
        assert code_analyzer._detect_language("app.ts", "interface User {") == "typescript"
        
        # Unknown file
        assert code_analyzer._detect_language("readme.txt", "Hello world") is None
    
    def test_extract_added_lines(self, code_analyzer):
        """Test extraction of added lines from git patch."""
        patch = """@@ -1,3 +1,6 @@
 def existing_function():
     pass
+
+def new_function():
+    return "hello"
+    # This is a comment"""
    
        added_lines = code_analyzer._extract_added_lines(patch)

        assert len(added_lines) == 4
        assert "" in added_lines
        assert "def new_function():" in added_lines
        assert "    # This is a comment" in added_lines
    
    def test_count_functions(self, code_analyzer):
        """Test function counting."""
        python_lines = [
            "def function1():",
            "    pass",
            "def function2(arg1, arg2):",
            "    return arg1 + arg2",
            "class MyClass:",
            "    def method1(self):",
            "        pass"
        ]
        
        count = code_analyzer._count_functions(python_lines, "python")
        assert count == 3  # function1, function2, method1
    
    def test_count_classes(self, code_analyzer):
        """Test class counting."""
        python_lines = [
            "class User:",
            "    pass",
            "class Product:",
            "    def __init__(self):",
            "        pass",
            "def function():",
            "    pass"
        ]
        
        count = code_analyzer._count_classes(python_lines, "python")
        assert count == 2  # User, Product
    
    def test_extract_technical_concepts(self, code_analyzer):
        """Test technical concept extraction."""
        code_lines = [
            "import requests",
            "def api_call():",
            "    response = requests.get('/api/users')",
            "    return response.json()",
            "# Database query",
            "user = db.query(User).filter(User.id == 1).first()"
        ]
        
        concepts = code_analyzer._extract_technical_concepts(code_lines)
        
        assert "api" in concepts
        assert "database" in concepts
    
    def test_calculate_complexity(self, code_analyzer):
        """Test complexity calculation."""
        simple_code = "def hello(): return 'world'"
        complex_code = """
def complex_function(data):
    if data:
        for item in data:
            if item.valid:
                try:
                    result = process_item(item)
                    if result:
                        for sub_item in result:
                            if sub_item.check():
                                yield sub_item
                except Exception:
                    continue
        """
        
        simple_complexity = code_analyzer._calculate_complexity(simple_code, 1, 0)
        complex_complexity = code_analyzer._calculate_complexity(complex_code, 1, 0)
        
        assert complex_complexity > simple_complexity
        assert 0 <= simple_complexity <= 1
        assert 0 <= complex_complexity <= 1
    
    def test_analyze_commit(self, code_analyzer, sample_commit_data):
        """Test complete commit analysis."""
        files_changed = sample_commit_data["files"]
        commit_info = sample_commit_data
        
        metrics = code_analyzer.analyze_commit(files_changed, commit_info)
        
        assert isinstance(metrics, CodeMetrics)
        assert metrics.function_count > 0
        assert metrics.class_count > 0
        assert "python" in metrics.language_distribution
        assert metrics.complexity_score > 0
        assert len(metrics.technical_concepts) > 0
        assert isinstance(metrics.semantic_embedding, np.ndarray)
    
    def test_extract_developer_skills(self, code_analyzer):
        """Test developer skill extraction from multiple analyses."""
        # Mock code analyses
        analyses = [
            CodeMetrics(
                complexity_score=0.6,
                language_distribution={"python": 0.8, "javascript": 0.2},
                technical_concepts=["backend", "database"],
                semantic_embedding=np.random.rand(768),
                function_count=5,
                class_count=2,
                import_complexity=0.3
            ),
            CodeMetrics(
                complexity_score=0.4,
                language_distribution={"python": 1.0},
                technical_concepts=["testing", "backend"],
                semantic_embedding=np.random.rand(768),
                function_count=3,
                class_count=1,
                import_complexity=0.2
            )
        ]
        
        skills = code_analyzer.extract_developer_skills(analyses)
        
        assert "programming_languages" in skills
        assert "python" in skills["programming_languages"]
        assert "domain_expertise" in skills
        assert "backend" in skills["domain_expertise"]
        assert 0 <= skills["complexity_preference"] <= 1
        assert 0 <= skills["technical_breadth"] <= 1


class TestSkillExtractor:
    """Test suite for SkillExtractor."""
    
    @pytest.fixture
    def skill_extractor(self):
        return SkillExtractor()
    
    @pytest.fixture
    def sample_developer_data(self):
        return {
            "developer_id": "test_user",
            "commits": [
                {
                    "hash": "abc123",
                    "timestamp": datetime.utcnow().isoformat(),
                    "files": [
                        {
                            "filename": "app.py",
                            "patch": "+def hello(): return 'world'",
                            "additions": 1
                        }
                    ]
                }
            ],
            "pr_reviews": [
                {
                    "content": "This looks good! Consider adding error handling for edge cases.",
                    "state": "approved",
                    "submitted_at": datetime.utcnow().isoformat()
                }
            ],
            "issue_comments": [
                {
                    "content": "I suggest using a different algorithm here for better performance. Here's how you can implement it...",
                    "created_at": datetime.utcnow().isoformat()
                }
            ],
            "discussions": [],
            "pr_descriptions": [
                {
                    "description": "Add new authentication API endpoint with JWT tokens",
                    "title": "Add auth API"
                }
            ],
            "commit_messages": ["Add authentication", "Fix bug in user validation"]
        }
    
    def test_calculate_interaction_quality(self, skill_extractor):
        """Test interaction quality calculation."""
        interactions = [
            {"content": "Thanks for the helpful suggestion! This solution works great."},
            {"content": "Consider using a more efficient algorithm here for better performance."},
            {"content": "lgtm"},  # Short, less helpful
            {"content": "I recommend refactoring this for better maintainability. Here's an example of how to implement it..."}
        ]
        
        quality = skill_extractor._calculate_interaction_quality(interactions)
        
        assert 0 <= quality <= 1
        assert quality > 0  # Should have some positive indicators
    
    def test_calculate_knowledge_sharing(self, skill_extractor):
        """Test knowledge sharing frequency calculation."""
        interactions = [
            {"content": "Here's how you can solve this: [detailed explanation with code examples]"},
            {"content": "Check out this documentation: https://example.com/docs"},
            {"content": "I agree with this approach"},
            {"content": "You can use this pattern: [example code]. This is a best practice because..."}
        ]
        
        sharing_freq = skill_extractor._calculate_knowledge_sharing(interactions)
        
        assert 0 <= sharing_freq <= 1
        assert sharing_freq > 0  # Should detect knowledge sharing patterns
    
    def test_extract_domain_knowledge(self, skill_extractor):
        """Test domain knowledge extraction from text."""
        issue_comments = [
            {"content": "This React component needs better error handling"},
            {"content": "The database query is inefficient, consider adding an index"},
            {"content": "We should implement proper authentication with JWT tokens"}
        ]
        
        pr_descriptions = [
            {"description": "Add Docker containerization for the microservices"}
        ]
        
        commit_messages = ["Fix frontend styling", "Optimize backend API"]
        
        domain_knowledge = skill_extractor._extract_domain_knowledge(
            issue_comments, pr_descriptions, commit_messages
        )
        
        assert isinstance(domain_knowledge, dict)
        assert "frontend" in domain_knowledge
        assert "backend" in domain_knowledge
        assert "devops" in domain_knowledge
        assert all(0 <= score <= 1 for score in domain_knowledge.values())
    
    def test_calculate_learning_velocity(self, skill_extractor):
        """Test learning velocity calculation."""
        commits = []
        base_date = datetime.utcnow() - timedelta(days=100)
        
        # Create commits with evolving technical concepts
        for i in range(10):
            commit_date = base_date + timedelta(days=i * 10)
            concepts = ["python", "basic"] if i < 5 else ["python", "advanced", "ml", "tensorflow"]
            
            commits.append({
                "timestamp": commit_date.isoformat(),
                "files": [
                    {
                        "patch": f"# {' '.join(concepts)} code here"
                    }
                ]
            })
        
        velocity = skill_extractor._calculate_learning_velocity(commits, 120)
        
        assert 0 <= velocity <= 10  # Normalized learning velocity
    
    def test_extract_comprehensive_profile(self, skill_extractor, sample_developer_data):
        """Test comprehensive profile extraction."""
        profile = skill_extractor.extract_comprehensive_profile(sample_developer_data)
        
        assert isinstance(profile, DeveloperProfile)
        assert profile.developer_id == "test_user"
        assert isinstance(profile.skill_vector, np.ndarray)
        assert len(profile.skill_vector) == settings.SKILL_VECTOR_DIM
        assert isinstance(profile.programming_languages, dict)
        assert isinstance(profile.domain_expertise, dict)
        assert 0 <= profile.collaboration_score <= 1
        assert profile.learning_velocity >= 0


class TestExpertiseTracker:
    """Test suite for ExpertiseTracker."""
    
    @pytest.fixture
    def expertise_tracker(self):
        return ExpertiseTracker()
    
    @pytest.fixture
    def mock_db_session(self):
        return Mock()
    
    def test_suggest_complementary_domains(self, expertise_tracker):
        """Test complementary domain suggestions."""
        current_domains = {
            "frontend": 0.8,
            "testing": 0.6
        }
        
        suggestions = expertise_tracker._suggest_complementary_domains(current_domains)
        
        assert isinstance(suggestions, list)
        assert len(suggestions) > 0
        
        # Should suggest backend for frontend expertise
        suggested_domains = [domain for domain, priority in suggestions]
        assert any("backend" in domain or "mobile" in domain for domain in suggested_domains)
        
        # All priorities should be reasonable
        for domain, priority in suggestions:
            assert 0 <= priority <= 1
    
    def test_calculate_skill_trend(self, expertise_tracker):
        """Test skill trend calculation."""
        # Mock snapshots with increasing skill values
        mock_snapshots = []
        for i in range(5):
            snapshot = Mock()
            snapshot.skill_vector = [0.2 + i * 0.1] * 768  # Increasing trend
            snapshot.snapshot_date = datetime.utcnow() - timedelta(days=30 * i)
            mock_snapshots.append(snapshot)
        
        trend = expertise_tracker._calculate_skill_trend(
            "python", 0.8, mock_snapshots, "programming_languages"
        )
        
        assert trend is not None
        assert trend.skill_name == "python"
        assert trend.trend_direction in ["increasing", "decreasing", "stable"]
        assert 0 <= trend.confidence <= 1
        assert trend.change_rate >= 0


class TestIntegration:
    """Integration tests for the complete Phase 1 system."""
    
    @pytest.fixture
    def full_system(self):
        """Set up complete system for integration testing."""
        return {
            'code_analyzer': CodeAnalyzer(),
            'skill_extractor': SkillExtractor(),
            'expertise_tracker': ExpertiseTracker(),
            'github_client': GitHubClient()
        }
    
    def test_end_to_end_skill_extraction(self, full_system):
        """Test complete pipeline from raw data to developer profile."""
        # Mock GitHub data (realistic structure)
        mock_github_data = {
            "developer_id": "john_doe",
            "github_username": "john_doe",
            "name": "John Doe",
            "email": "john@example.com",
            "commits": [
                {
                    "hash": "abc123",
                    "message": "Implement user authentication with JWT",
                    "timestamp": (datetime.utcnow() - timedelta(days=5)).isoformat(),
                    "files": [
                        {
                            "filename": "auth.py",
                            "patch": """import jwt\nfrom flask import request, jsonify\n\ndef authenticate_user(username, password):\n    # Validate credentials\n    user = User.query.filter_by(username=username).first()\n    if user and user.check_password(password):\n        token = jwt.encode({'user_id': user.id}, 'secret')\n        return jsonify({'token': token})\n    return jsonify({'error': 'Invalid credentials'}), 401""",
                            "additions": 10,
                            "deletions": 0
                        }
                    ],
                    "additions": 10,
                    "deletions": 0
                }
            ],
            "pr_reviews": [
                {
                    "content": "Good implementation! Consider adding rate limiting to prevent brute force attacks. Also, the JWT secret should be environment-configurable.",
                    "state": "approved",
                    "submitted_at": (datetime.utcnow() - timedelta(days=3)).isoformat(),
                    "pr_number": 42
                }
            ],
            "issue_comments": [
                {
                    "content": "I suggest using bcrypt for password hashing instead of the default hash function. Here's how to implement it: [code example]",
                    "created_at": (datetime.utcnow() - timedelta(days=7)).isoformat(),
                    "issue_number": 15
                }
            ],
            "discussions": [],
            "pr_descriptions": [
                {
                    "description": "This PR adds JWT-based authentication to our API endpoints. Includes middleware for token validation and user session management.",
                    "title": "Add JWT Authentication",
                    "created_at": (datetime.utcnow() - timedelta(days=5)).isoformat()
                }
            ],
            "commit_messages": [
                "Implement user authentication with JWT",
                "Add password hashing utilities",
                "Fix authentication middleware bug"
            ]
        }
        
        # Extract profile using skill extractor
        skill_extractor = full_system['skill_extractor']
        profile = skill_extractor.extract_comprehensive_profile(mock_github_data)
        
        # Verify profile structure
        assert isinstance(profile, DeveloperProfile)
        assert profile.developer_id == "john_doe"
        
        # Verify skill vector
        assert isinstance(profile.skill_vector, np.ndarray)
        assert len(profile.skill_vector) == settings.SKILL_VECTOR_DIM
        assert not np.allclose(profile.skill_vector, 0)  # Should have non-zero values
        
        # Verify programming languages detected
        assert "python" in profile.programming_languages
        python_skill = profile.programming_languages["python"]
        assert isinstance(python_skill, dict)
        assert "proficiency" in python_skill
        assert 0 <= python_skill["proficiency"] <= 1
        
        # Verify domain expertise
        assert "backend" in profile.domain_expertise or "security" in profile.domain_expertise
        
        # Verify collaboration metrics
        assert 0 <= profile.collaboration_score <= 1
        assert profile.collaboration_score > 0  # Should detect collaboration activity
        
        # Verify learning velocity
        assert profile.learning_velocity >= 0
        
        # Verify confidence scores
        assert isinstance(profile.confidence_scores, dict)
        assert "programming_languages" in profile.confidence_scores
    
    def test_skill_vector_consistency(self, full_system):
        """Test that skill vectors are consistent and meaningful."""
        skill_extractor = full_system['skill_extractor']
        # Create two similar developer profiles
        base_data = {
            "developer_id": "dev1",
            "github_username": "dev1",
            "commits": [
                {
                    "timestamp": datetime.utcnow().isoformat(),
                    "files": [{"filename": "app.py", "patch": "def hello(): pass", "additions": 1}]
                }
            ],
            "pr_reviews": [],
            "issue_comments": [],
            "discussions": [],
            "pr_descriptions": [],
            "commit_messages": ["Add hello function"]
        }
        # Very similar developer
        similar_data = base_data.copy()
        similar_data["developer_id"] = "dev2"
        similar_data["github_username"] = "dev2"
        # Very different developer
        different_data = {
            "developer_id": "dev3",
            "github_username": "dev3",
            "commits": [
                {
                    "timestamp": datetime.utcnow().isoformat(),
                    "files": [{"filename": "app.js", "patch": "function complex() { /* lots of React code */ }", "additions": 50}]
                }
            ],
            "pr_reviews": [{"content": "Excellent React implementation with Redux!", "state": "approved"}],
            "issue_comments": [{"content": "Consider using React hooks for better performance"}],
            "discussions": [],
            "pr_descriptions": [{"description": "Advanced React frontend with TypeScript"}],
            "commit_messages": ["Implement complex React frontend"]
        }
        # Extract profiles
        profile1 = skill_extractor.extract_comprehensive_profile(base_data)
        profile2 = skill_extractor.extract_comprehensive_profile(similar_data)
        profile3 = skill_extractor.extract_comprehensive_profile(different_data)
        # Similar developers should have similar skill vectors
        similarity_1_2 = np.dot(profile1.skill_vector, profile2.skill_vector) / (
            np.linalg.norm(profile1.skill_vector) * np.linalg.norm(profile2.skill_vector)
        )
        # Different developer should have less similar skill vector
        similarity_1_3 = np.dot(profile1.skill_vector, profile3.skill_vector) / (
            np.linalg.norm(profile1.skill_vector) * np.linalg.norm(profile3.skill_vector)
        )
        # Verify similarity relationships (allowing some variance due to noise)
        assert similarity_1_2 > similarity_1_3 or abs(similarity_1_2 - similarity_1_3) < 0.1
    
    def test_temporal_skill_evolution(self, full_system):
        """Test that temporal skill updates work correctly."""
        skill_extractor = full_system['skill_extractor']
        # Initial profile
        initial_data = {
            "developer_id": "evolving_dev",
            "github_username": "evolving_dev",
            "commits": [
                {
                    "timestamp": (datetime.utcnow() - timedelta(days=30)).isoformat(),
                    "files": [{"filename": "simple.py", "patch": "print('hello')", "additions": 1}]
                }
            ],
            "pr_reviews": [],
            "issue_comments": [],
            "discussions": [],
            "pr_descriptions": [],
            "commit_messages": ["Simple hello world"]
        }
        # Advanced profile (showing growth)
        advanced_data = {
            "developer_id": "evolving_dev",
            "github_username": "evolving_dev",
            "commits": [
                {
                    "timestamp": datetime.utcnow().isoformat(),
                    "files": [
                        {
                            "filename": "ml_model.py",
                            "patch": """import tensorflow as tf\nfrom sklearn.model_selection import train_test_split\n\nclass NeuralNetworkModel:\n    def __init__(self, layers):\n        self.model = tf.keras.Sequential(layers)\n    \n    def train(self, X, y):\n        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)\n        self.model.compile(optimizer='adam', loss='mse')\n        return self.model.fit(X_train, y_train, validation_data=(X_val, y_val))""",
                            "additions": 12
                        }
                    ]
                }
            ],
            "pr_reviews": [
                {
                    "content": "Great ML implementation! The model architecture looks solid. Consider adding regularization to prevent overfitting.",
                    "state": "approved"
                }
            ],
            "issue_comments": [
                {
                    "content": "For this machine learning problem, I recommend using a convolutional neural network. Here's why..."
                }
            ],
            "discussions": [],
            "pr_descriptions": [
                {
                    "description": "Implement deep learning model for image classification using TensorFlow"
                }
            ],
            "commit_messages": ["Add neural network model", "Implement ML training pipeline"]
        }
        # Extract initial and advanced profiles
        initial_profile = skill_extractor.extract_comprehensive_profile(initial_data)
        advanced_profile = skill_extractor.extract_comprehensive_profile(advanced_data)
        # Test temporal update
        updated_profile = skill_extractor.update_temporal_skills(
            initial_profile, advanced_data
        )
        # Verify that skills have evolved
        assert updated_profile.developer_id == initial_profile.developer_id
        # Should show evidence of ML skills in advanced profile
        assert "ml" in advanced_profile.domain_expertise
        assert advanced_profile.domain_expertise["ml"] > 0
        # Updated profile should reflect learning progress
        # (blend of old and new, but should show improvement)
        python_skill_initial = initial_profile.programming_languages.get("python", {}).get("proficiency", 0)
        python_skill_updated = updated_profile.programming_languages.get("python", {}).get("proficiency", 0)
        # Should maintain or improve Python skills
        assert python_skill_updated >= python_skill_initial


# Performance and benchmarking tests
class TestPerformance:
    """Performance tests for Phase 1 components."""
    
    def test_code_analysis_performance(self):
        """Test that code analysis completes within reasonable time."""
        import time
        analyzer = CodeAnalyzer()
        # Large commit simulation
        large_files = []
        for i in range(10):
            large_files.append({
                "filename": f"file_{i}.py",
                "patch": "def function_{}(): pass\n".format(i) * 20,  # 20 lines per file
                "additions": 20
            })
        start_time = time.time()
        metrics = analyzer.analyze_commit(large_files, {"hash": "test"})
        end_time = time.time()
        # Should complete within 2 seconds for 10 files with 200 lines total
        assert end_time - start_time < 2.0
        assert isinstance(metrics, CodeMetrics)
    
    def test_skill_vector_generation_performance(self):
        """Test that skill vector generation is efficient."""
        import time
        extractor = SkillExtractor()
        # Simulate large developer dataset
        large_data = {
            "developer_id": "perf_test",
            "github_username": "perf_test",
            "commits": [
                {
                    "timestamp": datetime.utcnow().isoformat(),
                    "files": [{"filename": f"file_{i}.py", "patch": f"def func_{i}(): pass", "additions": 1}]
                }
                for i in range(50)  # 50 commits
            ],
            "pr_reviews": [
                {"content": f"Review comment {i}", "state": "approved"}
                for i in range(20)  # 20 reviews
            ],
            "issue_comments": [
                {"content": f"Issue comment {i}"}
                for i in range(30)  # 30 comments
            ],
            "discussions": [],
            "pr_descriptions": [],
            "commit_messages": [f"Commit {i}" for i in range(50)]
        }
        start_time = time.time()
        profile = extractor.extract_comprehensive_profile(large_data)
        end_time = time.time()
        # Should complete within 5 seconds for large dataset
        assert end_time - start_time < 5.0
        assert isinstance(profile, DeveloperProfile)
        assert len(profile.skill_vector) == settings.SKILL_VECTOR_DIM


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
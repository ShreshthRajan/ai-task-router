# src/core/task_analysis/complexity_predictor.py
import numpy as np
import spacy
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import re
import asyncio
from dataclasses import dataclass
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json

import sys
sys.path.append('../../..')
from config import settings
from core.developer_modeling.code_analyzer import CodeAnalyzer
from integrations.github_client import GitHubClient

@dataclass
class TaskComplexity:
    """Comprehensive task complexity assessment."""
    task_id: str
    technical_complexity: float
    domain_difficulty: float
    collaboration_requirements: float
    learning_opportunities: float
    business_impact: float
    estimated_hours: float
    confidence_score: float
    complexity_factors: Dict[str, float]
    required_skills: Dict[str, float]
    risk_factors: List[str]

class ComplexityPredictor:
    """Multi-dimensional task complexity prediction engine."""
    
    def __init__(self):
        self.nlp = None
        self.code_analyzer = CodeAnalyzer()
        self.github_client = GitHubClient()
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self._load_models()
        
        # Complexity weights for different factors
        self.complexity_weights = {
            'technical': 0.25,
            'domain': 0.20,
            'collaboration': 0.15,
            'learning': 0.15,
            'business': 0.25
        }
        
        # Technical complexity indicators
        self.technical_indicators = {
            'architecture_keywords': ['refactor', 'redesign', 'architecture', 'framework', 'migration'],
            'performance_keywords': ['optimize', 'performance', 'scalability', 'bottleneck', 'slow'],
            'integration_keywords': ['api', 'integration', 'webhook', 'third-party', 'external'],
            'security_keywords': ['security', 'auth', 'authentication', 'encryption', 'vulnerability'],
            'data_keywords': ['database', 'migration', 'schema', 'query', 'data model']
        }
        
        # Domain difficulty patterns
        self.domain_complexity = {
            'ml': 0.9,
            'ai': 0.9,
            'security': 0.8,
            'performance': 0.7,
            'architecture': 0.8,
            'frontend': 0.5,
            'backend': 0.6,
            'devops': 0.7,
            'testing': 0.4,
            'documentation': 0.2
        }
    
    def _load_models(self):
        """Load required NLP models."""
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Warning: spaCy model 'en_core_web_sm' not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
    
    async def predict_task_complexity(self, task_data: Dict) -> TaskComplexity:
        """Main entry point for task complexity prediction."""
        
        # Extract task features
        task_features = await self._extract_task_features(task_data)
        
        # Calculate individual complexity dimensions
        technical_complexity = self._calculate_technical_complexity(task_features)
        domain_difficulty = self._calculate_domain_difficulty(task_features)
        collaboration_reqs = self._calculate_collaboration_requirements(task_features)
        learning_opportunities = self._calculate_learning_opportunities(task_features)
        business_impact = self._calculate_business_impact(task_features)
        
        # Estimate time requirements
        estimated_hours = self._estimate_time_requirements(task_features, {
            'technical': technical_complexity,
            'domain': domain_difficulty,
            'collaboration': collaboration_reqs
        })
        
        # Calculate overall confidence
        confidence_score = self._calculate_prediction_confidence(task_features)
        
        # Extract required skills
        required_skills = self._extract_required_skills(task_features)
        
        # Identify risk factors
        risk_factors = self._identify_risk_factors(task_features)
        
        return TaskComplexity(
            task_id=task_data.get('id', 'unknown'),
            technical_complexity=technical_complexity,
            domain_difficulty=domain_difficulty,
            collaboration_requirements=collaboration_reqs,
            learning_opportunities=learning_opportunities,
            business_impact=business_impact,
            estimated_hours=estimated_hours,
            confidence_score=confidence_score,
            complexity_factors=task_features.get('complexity_factors', {}),
            required_skills=required_skills,
            risk_factors=risk_factors
        )
    
    async def _extract_task_features(self, task_data: Dict) -> Dict:
        """Extract comprehensive features from task data."""
        features = {
            'title': task_data.get('title', ''),
            'description': task_data.get('body', ''),
            'labels': task_data.get('labels', []),
            'repository': task_data.get('repository', ''),
            'issue_type': self._determine_issue_type(task_data),
            'urgency_indicators': self._extract_urgency_indicators(task_data),
            'complexity_factors': {},
            'mentioned_technologies': [],
            'affected_components': [],
            'dependencies': []
        }
        
        # Combine text for analysis
        full_text = f"{features['title']} {features['description']}"
        
        # Extract technical features
        features['mentioned_technologies'] = self._extract_technologies(full_text)
        features['affected_components'] = self._extract_components(full_text)
        features['code_patterns'] = self._extract_code_patterns(full_text)
        features['architectural_impact'] = self._assess_architectural_impact(full_text)
        
        # Analyze text complexity
        if self.nlp:
            features['text_complexity'] = self._analyze_text_complexity(full_text)
            features['semantic_features'] = self._extract_semantic_features(full_text)
        
        # Repository context analysis
        if features['repository']:
            repo_context = await self._analyze_repository_context(
                features['repository'], task_data
            )
            features.update(repo_context)
        
        return features
    
    def _determine_issue_type(self, task_data: Dict) -> str:
        """Determine the type of issue/task."""
        title = task_data.get('title', '').lower()
        labels = [label.get('name', '').lower() for label in task_data.get('labels', [])]
        
        # Check labels first
        if any(label in ['bug', 'defect', 'error'] for label in labels):
            return 'bug'
        elif any(label in ['feature', 'enhancement', 'new'] for label in labels):
            return 'feature'
        elif any(label in ['refactor', 'cleanup', 'maintenance'] for label in labels):
            return 'refactor'
        elif any(label in ['documentation', 'docs'] for label in labels):
            return 'documentation'
        
        # Check title patterns
        if any(word in title for word in ['fix', 'bug', 'error', 'broken']):
            return 'bug'
        elif any(word in title for word in ['add', 'implement', 'create', 'new']):
            return 'feature'
        elif any(word in title for word in ['refactor', 'cleanup', 'improve']):
            return 'refactor'
        elif any(word in title for word in ['document', 'readme', 'guide']):
            return 'documentation'
        
        return 'unknown'
    
    def _extract_urgency_indicators(self, task_data: Dict) -> List[str]:
        """Extract urgency indicators from task data."""
        indicators = []
        text = f"{task_data.get('title', '')} {task_data.get('body', '')}".lower()
        labels = [label.get('name', '').lower() for label in task_data.get('labels', [])]
        
        urgency_patterns = {
            'critical': ['critical', 'urgent', 'emergency', 'asap', 'immediately'],
            'high': ['high priority', 'important', 'blocker', 'blocking'],
            'production': ['production', 'prod', 'live', 'customer impact'],
            'security': ['security', 'vulnerability', 'exploit', 'breach']
        }
        
        for category, patterns in urgency_patterns.items():
            if any(pattern in text for pattern in patterns) or any(pattern in labels for pattern in patterns):
                indicators.append(category)
        
        return indicators
    
    def _extract_technologies(self, text: str) -> List[str]:
        """Extract mentioned technologies and frameworks."""
        tech_patterns = {
            'languages': r'\b(python|javascript|typescript|java|cpp|go|rust|ruby|php|kotlin|swift|scala|clojure)\b',
            'frameworks': r'\b(react|angular|vue|django|flask|spring|express|rails|laravel|tensorflow|pytorch)\b',
            'databases': r'\b(mysql|postgresql|mongodb|redis|elasticsearch|cassandra|dynamodb)\b',
            'tools': r'\b(docker|kubernetes|jenkins|gitlab|github|aws|azure|gcp|terraform)\b'
        }
        
        technologies = []
        text_lower = text.lower()
        
        for category, pattern in tech_patterns.items():
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            technologies.extend(matches)
        
        return list(set(technologies))
    
    def _extract_components(self, text: str) -> List[str]:
        """Extract affected system components."""
        component_patterns = {
            'frontend': r'\b(ui|frontend|client|browser|react|angular|vue)\b',
            'backend': r'\b(api|backend|server|service|microservice)\b',
            'database': r'\b(database|db|schema|table|query|migration)\b',
            'infrastructure': r'\b(server|deployment|docker|kubernetes|cloud)\b',
            'auth': r'\b(auth|authentication|authorization|login|user)\b',
            'payment': r'\b(payment|billing|checkout|stripe|paypal)\b'
        }
        
        components = []
        text_lower = text.lower()
        
        for component, pattern in component_patterns.items():
            if re.search(pattern, text_lower, re.IGNORECASE):
                components.append(component)
        
        return components
    
    def _extract_code_patterns(self, text: str) -> List[str]:
        """Extract code-related patterns and complexity indicators."""
        patterns = []
        text_lower = text.lower()
        
        code_indicators = {
            'algorithm': ['algorithm', 'optimization', 'performance', 'complexity'],
            'architecture': ['architecture', 'design pattern', 'refactor', 'structure'],
            'integration': ['integration', 'api', 'webhook', 'third-party'],
            'testing': ['test', 'testing', 'unit test', 'integration test'],
            'deployment': ['deploy', 'deployment', 'ci/cd', 'pipeline']
        }
        
        for pattern_type, keywords in code_indicators.items():
            if any(keyword in text_lower for keyword in keywords):
                patterns.append(pattern_type)
        
        return patterns
    
    def _assess_architectural_impact(self, text: str) -> float:
        """Assess the architectural impact of the task."""
        impact_keywords = {
            'high': ['architecture', 'redesign', 'migration', 'breaking change', 'major refactor'],
            'medium': ['refactor', 'restructure', 'api change', 'schema change'],
            'low': ['minor change', 'bug fix', 'style update', 'documentation']
        }
        
        text_lower = text.lower()
        
        for level, keywords in impact_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                if level == 'high':
                    return 0.9
                elif level == 'medium':
                    return 0.6
                else:
                    return 0.3
        
        return 0.5  # Default medium impact
    
    def _analyze_text_complexity(self, text: str) -> Dict[str, float]:
        """Analyze linguistic complexity of task description."""
        if not self.nlp:
            return {'readability': 0.5, 'technical_density': 0.5}
        
        doc = self.nlp(text)
        
        # Calculate readability metrics
        sentences = list(doc.sents)
        words = [token for token in doc if not token.is_punct and not token.is_space]
        
        avg_sentence_length = len(words) / max(len(sentences), 1)
        technical_terms = len([token for token in words if token.pos_ in ['NOUN', 'ADJ'] and len(token.text) > 6])
        
        readability = min(avg_sentence_length / 20, 1.0)  # Normalize
        technical_density = technical_terms / max(len(words), 1)
        
        return {
            'readability': readability,
            'technical_density': technical_density,
            'sentence_count': len(sentences),
            'word_count': len(words)
        }
    
    def _extract_semantic_features(self, text: str) -> Dict[str, float]:
        """Extract semantic features using NLP."""
        if not self.nlp:
            return {}
        
        doc = self.nlp(text)
        
        # Extract entities and their types
        entities = {}
        for ent in doc.ents:
            entities[ent.label_] = entities.get(ent.label_, 0) + 1
        
        # Calculate semantic complexity
        unique_concepts = len(set([token.lemma_ for token in doc if token.pos_ in ['NOUN', 'VERB']]))
        
        return {
            'entities': entities,
            'unique_concepts': unique_concepts,
            'semantic_complexity': unique_concepts / max(len(doc), 1)
        }
    
    async def _analyze_repository_context(self, repository: str, task_data: Dict) -> Dict:
        """Analyze repository context for additional complexity factors."""
        context = {
            'repo_complexity': 0.5,
            'recent_activity': 0.5,
            'team_size': 1,
            'codebase_size': 'medium'
        }
        
        try:
            # This would integrate with GitHub API to get repository metrics
            # For now, we'll use simplified heuristics
            
            # Analyze repository from task data
            if 'repository' in task_data:
                repo_name = task_data['repository']
                
                # Estimate complexity based on repository characteristics
                if any(tech in repo_name.lower() for tech in ['ml', 'ai', 'data']):
                    context['repo_complexity'] = 0.8
                elif any(tech in repo_name.lower() for tech in ['api', 'service']):
                    context['repo_complexity'] = 0.6
                elif any(tech in repo_name.lower() for tech in ['frontend', 'ui']):
                    context['repo_complexity'] = 0.4
        
        except Exception as e:
            print(f"Error analyzing repository context: {e}")
        
        return context
    
    def _calculate_technical_complexity(self, features: Dict) -> float:
        """Calculate technical complexity score."""
        complexity_score = 0.0
        
        # Base complexity from issue type
        type_complexity = {
            'bug': 0.4,
            'feature': 0.7,
            'refactor': 0.8,
            'documentation': 0.2,
            'unknown': 0.5
        }
        complexity_score += type_complexity.get(features['issue_type'], 0.5) * 0.3
        
        # Technology complexity
        tech_complexity = 0.0
        for tech in features['mentioned_technologies']:
            if tech in ['tensorflow', 'pytorch', 'kubernetes']:
                tech_complexity += 0.3
            elif tech in ['react', 'angular', 'django']:
                tech_complexity += 0.2
            else:
                tech_complexity += 0.1
        complexity_score += min(tech_complexity, 0.4) * 0.2
        
        # Architectural impact
        complexity_score += features['architectural_impact'] * 0.3
        
        # Code pattern complexity
        pattern_complexity = len(features['code_patterns']) * 0.1
        complexity_score += min(pattern_complexity, 0.3) * 0.2
        
        return min(complexity_score, 1.0)
    
    def _calculate_domain_difficulty(self, features: Dict) -> float:
        """Calculate domain-specific difficulty."""
        difficulty = 0.0
        
        # Check for high-difficulty domains
        text = f"{features['title']} {features['description']}".lower()
        
        for domain, complexity in self.domain_complexity.items():
            if domain in text or domain in features['mentioned_technologies']:
                difficulty = max(difficulty, complexity)
        
        # Adjust based on component complexity
        component_difficulty = {
            'auth': 0.7,
            'payment': 0.8,
            'database': 0.6,
            'infrastructure': 0.7,
            'frontend': 0.4,
            'backend': 0.5
        }
        
        for component in features['affected_components']:
            if component in component_difficulty:
                difficulty = max(difficulty, component_difficulty[component])
        
        return min(difficulty, 1.0)
    
    def _calculate_collaboration_requirements(self, features: Dict) -> float:
        """Calculate collaboration requirements score."""
        collaboration_score = 0.2  # Base collaboration level
        
        # Multiple components increase collaboration needs
        component_count = len(features['affected_components'])
        collaboration_score += min(component_count * 0.15, 0.4)
        
        # Cross-functional requirements
        if any(comp in features['affected_components'] for comp in ['frontend', 'backend']):
            collaboration_score += 0.2
        
        # Infrastructure changes require DevOps collaboration
        if 'infrastructure' in features['affected_components']:
            collaboration_score += 0.3
        
        # External integrations
        if 'integration' in features['code_patterns']:
            collaboration_score += 0.2
        
        return min(collaboration_score, 1.0)
    
    def _calculate_learning_opportunities(self, features: Dict) -> float:
        """Calculate learning opportunity score."""
        learning_score = 0.0
        
        # New technologies provide learning opportunities
        tech_count = len(features['mentioned_technologies'])
        learning_score += min(tech_count * 0.15, 0.5)
        
        # Complex domains offer more learning
        if features.get('domain_difficulty', 0) > 0.7:
            learning_score += 0.3
        
        # Architectural work is educational
        if features['architectural_impact'] > 0.6:
            learning_score += 0.3
        
        # Research-heavy tasks
        text = f"{features['title']} {features['description']}".lower()
        if any(word in text for word in ['research', 'investigate', 'explore', 'prototype']):
            learning_score += 0.4
        
        return min(learning_score, 1.0)
    
    def _calculate_business_impact(self, features: Dict) -> float:
        """Calculate business impact score."""
        impact_score = 0.3  # Base impact
        
        # Urgency indicators
        urgency_impact = {
            'critical': 0.5,
            'high': 0.3,
            'production': 0.4,
            'security': 0.4
        }
        
        for indicator in features['urgency_indicators']:
            if indicator in urgency_impact:
                impact_score += urgency_impact[indicator]
        
        # Customer-facing features
        text = f"{features['title']} {features['description']}".lower()
        if any(word in text for word in ['user', 'customer', 'ui', 'frontend', 'experience']):
            impact_score += 0.2
        
        # Revenue-related features
        if any(word in text for word in ['payment', 'billing', 'subscription', 'revenue']):
            impact_score += 0.3
        
        return min(impact_score, 1.0)
    
    def _estimate_time_requirements(self, features: Dict, complexity_scores: Dict) -> float:
        """Estimate time requirements in hours."""
        
        # Base time by issue type
        base_hours = {
            'bug': 4,
            'feature': 16,
            'refactor': 24,
            'documentation': 2,
            'unknown': 8
        }
        
        estimated_hours = base_hours.get(features['issue_type'], 8)
        
        # Adjust based on complexity
        complexity_multiplier = 1.0
        complexity_multiplier += complexity_scores['technical'] * 2
        complexity_multiplier += complexity_scores['domain'] * 1.5
        complexity_multiplier += complexity_scores['collaboration'] * 0.5
        
        # Component count impact
        component_multiplier = 1 + (len(features['affected_components']) - 1) * 0.3
        
        # Technology complexity
        tech_multiplier = 1 + min(len(features['mentioned_technologies']) * 0.2, 1.0)
        
        final_estimate = estimated_hours * complexity_multiplier * component_multiplier * tech_multiplier
        
        return min(max(final_estimate, 1), 200)  # Cap between 1 and 200 hours
    
    def _calculate_prediction_confidence(self, features: Dict) -> float:
        """Calculate confidence in the complexity prediction."""
        confidence = 0.7  # Base confidence
        
        # More information increases confidence
        if features['description'] and len(features['description']) > 100:
            confidence += 0.1
        
        if features['mentioned_technologies']:
            confidence += 0.1
        
        if features['labels']:
            confidence += 0.05
        
        # Clear issue type increases confidence
        if features['issue_type'] != 'unknown':
            confidence += 0.1
        
        # Repository context
        if features.get('repo_complexity') is not None:
            confidence += 0.05
        
        return min(confidence, 1.0)
    
    def _extract_required_skills(self, features: Dict) -> Dict[str, float]:
        """Extract required skills and their importance levels."""
        required_skills = {}
        
        # Programming languages
        for tech in features['mentioned_technologies']:
            if tech in settings.SUPPORTED_LANGUAGES:
                required_skills[f"lang_{tech}"] = 0.8
        
        # Domain skills
        for component in features['affected_components']:
            required_skills[f"domain_{component}"] = 0.7
        
        # Technical skills from patterns
        skill_mapping = {
            'algorithm': 'algorithmic_thinking',
            'architecture': 'system_design',
            'integration': 'api_development',
            'testing': 'testing_frameworks',
            'deployment': 'devops'
        }
        
        for pattern in features['code_patterns']:
            if pattern in skill_mapping:
                required_skills[skill_mapping[pattern]] = 0.6
        
        # Security skills detection
        text = f"{features['title']} {features['description']}".lower()
        if any(word in text for word in ['security', 'auth', 'authentication', 'oauth', 'jwt', 'encryption']):
            required_skills['security'] = 0.8
        
        # Add specific security technologies
        security_techs = ['jwt', 'oauth', 'encryption', 'bcrypt', 'ssl', 'tls']
        for tech in features['mentioned_technologies']:
            if tech.lower() in security_techs:
                required_skills['security'] = 0.9
                break
        
        return required_skills
    
    def _identify_risk_factors(self, features: Dict) -> List[str]:
        """Identify potential risk factors for the task."""
        risks = []
        
        # Technical risks
        if features['architectural_impact'] > 0.7:
            risks.append("High architectural impact - potential for breaking changes")
        
        if len(features['affected_components']) > 3:
            risks.append("Multiple components affected - coordination complexity")
        
        # Domain risks
        text = f"{features['title']} {features['description']}".lower()
        if any(word in text for word in ['security', 'auth', 'payment']):
            risks.append("Security-sensitive area - requires careful review")
        
        if any(word in text for word in ['performance', 'optimization', 'scale']):
            risks.append("Performance critical - requires thorough testing")
        
        # Collaboration risks
        if features.get('collaboration_requirements', 0) > 0.7:
            risks.append("High collaboration requirements - potential coordination delays")
        
        # Urgency risks
        if 'critical' in features['urgency_indicators']:
            risks.append("Critical priority - limited time for thorough development")
        
        return risks

    async def batch_predict_complexity(self, tasks: List[Dict]) -> List[TaskComplexity]:
        """Predict complexity for multiple tasks efficiently."""
        results = []
        
        for task in tasks:
            try:
                complexity = await self.predict_task_complexity(task)
                results.append(complexity)
            except Exception as e:
                print(f"Error predicting complexity for task {task.get('id', 'unknown')}: {e}")
                # Return default complexity on error
                results.append(TaskComplexity(
                    task_id=task.get('id', 'unknown'),
                    technical_complexity=0.5,
                    domain_difficulty=0.5,
                    collaboration_requirements=0.5,
                    learning_opportunities=0.5,
                    business_impact=0.5,
                    estimated_hours=8.0,
                    confidence_score=0.3,
                    complexity_factors={},
                    required_skills={},
                    risk_factors=["Error in complexity analysis"]
                ))
        
        return results
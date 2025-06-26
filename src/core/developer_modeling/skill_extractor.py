# src/core/developer_modeling/skill_extractor.py
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import defaultdict
import re

from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from config import settings
from .code_analyzer import CodeAnalyzer, CodeMetrics

@dataclass
class CollaborationAnalysis:
    interaction_quality: float
    knowledge_sharing_frequency: float
    technical_leadership: float
    communication_effectiveness: float
    team_influence: float

@dataclass
class DeveloperProfile:
    developer_id: str
    skill_vector: np.ndarray
    programming_languages: Dict[str, float]
    domain_expertise: Dict[str, float]
    collaboration_score: float
    learning_velocity: float
    confidence_scores: Dict[str, float]
    last_updated: datetime


print(f"🔍 DEBUG: GITHUB_TOKEN from settings: '{settings.GITHUB_TOKEN}'")
print(f"🔍 DEBUG: GITHUB_TOKEN length: {len(settings.GITHUB_TOKEN) if settings.GITHUB_TOKEN else 0}")

class SkillExtractor:
    """Main orchestrator for developer skill extraction and modeling."""
    
    def __init__(self):
        self.code_analyzer = CodeAnalyzer()
        self.sentence_model = SentenceTransformer(settings.SENTENCE_TRANSFORMER_MODEL)
        self.tfidf_vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
        
        # Technical keyword patterns for domain expertise
        self.domain_keywords = {
            'frontend': [
                'react', 'vue', 'angular', 'javascript', 'typescript', 'css', 'html',
                'component', 'jsx', 'tsx', 'ui', 'ux', 'responsive', 'bootstrap',
                'material-ui', 'styled-components', 'webpack', 'babel'
            ],
            'backend': [
                'api', 'server', 'database', 'endpoint', 'middleware', 'authentication',
                'authorization', 'rest', 'graphql', 'microservices', 'sql', 'nosql',
                'redis', 'mongodb', 'postgresql', 'mysql', 'express', 'django', 'flask'
            ],
            'devops': [
                'docker', 'kubernetes', 'ci/cd', 'jenkins', 'github actions', 'aws',
                'azure', 'gcp', 'terraform', 'ansible', 'monitoring', 'logging',
                'infrastructure', 'deployment', 'container', 'orchestration'
            ],
            'ml': [
                'machine learning', 'deep learning', 'neural network', 'tensorflow',
                'pytorch', 'scikit-learn', 'pandas', 'numpy', 'data science',
                'model training', 'feature engineering', 'nlp', 'computer vision'
            ],
            'mobile': [
                'ios', 'android', 'react native', 'flutter', 'swift', 'kotlin',
                'mobile app', 'app store', 'play store', 'xamarin', 'cordova'
            ],
            'testing': [
                'unit test', 'integration test', 'e2e test', 'jest', 'pytest',
                'selenium', 'cypress', 'testing framework', 'mock', 'stub',
                'test coverage', 'tdd', 'bdd'
            ],
            'security': [
                'security', 'authentication', 'authorization', 'encryption', 'ssl',
                'oauth', 'jwt', 'vulnerability', 'penetration testing', 'owasp',
                'sql injection', 'xss', 'csrf'
            ]
        }
    
    def extract_comprehensive_profile(self, 
                                developer_data: Dict,
                                time_window_days: int = 180) -> DeveloperProfile:
        """Extract complete developer profile from multiple data sources."""
        
        print(f"🔍 DEBUG: Starting profile extraction for {developer_data.get('developer_id', 'unknown')}")
        print(f"🔍 DEBUG: Input data summary:")
        print(f"  - Commits: {len(developer_data.get('commits', []))}")
        print(f"  - PR Reviews: {len(developer_data.get('pr_reviews', []))}")
        print(f"  - Issue Comments: {len(developer_data.get('issue_comments', []))}")
        print(f"  - Commit Messages: {len(developer_data.get('commit_messages', []))}")
        print(f"  - PR Descriptions: {len(developer_data.get('pr_descriptions', []))}")
        
        # Extract code-based skills
        print(f"🔍 DEBUG: Extracting code skills...")
        code_skills = self._extract_code_skills(
            developer_data.get('commits', []),
            time_window_days
        )
        print(f"🔍 DEBUG: Code skills extracted: {code_skills}")
        
        # Extract collaboration-based skills
        print(f"🔍 DEBUG: Analyzing collaboration patterns...")
        collaboration_analysis = self._analyze_collaboration_patterns(
            developer_data.get('pr_reviews', []),
            developer_data.get('issue_comments', []),
            developer_data.get('discussions', [])
        )
        print(f"🔍 DEBUG: Collaboration analysis:")
        print(f"  - interaction_quality: {collaboration_analysis.interaction_quality}")
        print(f"  - knowledge_sharing_frequency: {collaboration_analysis.knowledge_sharing_frequency}")
        print(f"  - technical_leadership: {collaboration_analysis.technical_leadership}")
        print(f"  - communication_effectiveness: {collaboration_analysis.communication_effectiveness}")
        print(f"  - team_influence: {collaboration_analysis.team_influence}")
        
        # Extract domain knowledge from text
        print(f"🔍 DEBUG: Extracting domain knowledge...")
        domain_knowledge = self._extract_domain_knowledge(
            developer_data.get('issue_comments', []),
            developer_data.get('pr_descriptions', []),
            developer_data.get('commit_messages', [])
        )
        print(f"🔍 DEBUG: Domain knowledge: {domain_knowledge}")
        
        # Calculate learning velocity
        print(f"🔍 DEBUG: Calculating learning velocity...")
        learning_velocity = self._calculate_learning_velocity(
            developer_data.get('commits', []),
            time_window_days
        )
        print(f"🔍 DEBUG: Learning velocity: {learning_velocity}")
        
        # Generate unified skill vector
        print(f"🔍 DEBUG: Generating skill vector...")
        skill_vector = self._generate_skill_vector(
            code_skills, domain_knowledge, collaboration_analysis
        )
        print(f"🔍 DEBUG: Skill vector shape: {skill_vector.shape}")
        print(f"🔍 DEBUG: Skill vector non-zero elements: {np.count_nonzero(skill_vector)}")
        
        # Calculate confidence scores
        print(f"🔍 DEBUG: Calculating confidence scores...")
        confidence_scores = self._calculate_confidence_scores(
            code_skills, collaboration_analysis, developer_data
        )
        print(f"🔍 DEBUG: Confidence scores: {confidence_scores}")
        
        # Create final profile
        print(f"🔍 DEBUG: Creating final developer profile...")
        profile = DeveloperProfile(
            developer_id=developer_data['developer_id'],
            skill_vector=skill_vector,
            programming_languages=code_skills.get('programming_languages', {}),
            domain_expertise=domain_knowledge,
            collaboration_score=collaboration_analysis.interaction_quality,
            learning_velocity=learning_velocity,
            confidence_scores=confidence_scores,
            last_updated=datetime.utcnow()
        )
        
        print(f"🔍 DEBUG: Profile creation complete for {developer_data.get('developer_id', 'unknown')}")
        print(f"🔍 DEBUG: Final profile summary:")
        print(f"  - Programming languages: {len(profile.programming_languages)} found")
        print(f"  - Domain expertise areas: {len(profile.domain_expertise)} found")
        print(f"  - Collaboration score: {profile.collaboration_score}")
        print(f"  - Learning velocity: {profile.learning_velocity}")
        print(f"  - Overall confidence: {profile.confidence_scores.get('overall', 0.0)}")
        
        return profile
    
    def _extract_code_skills(self, commits: List[Dict], time_window_days: int) -> Dict:
        """Extract skills from code commit analysis."""
        
        print(f"🔍 DEBUG: _extract_code_skills called with {len(commits)} commits")
        
        # Filter recent commits - FIX THE DATETIME ISSUE
        cutoff_date = datetime.utcnow().replace(tzinfo=None) - timedelta(days=time_window_days)
        print(f"🔍 DEBUG: Cutoff date: {cutoff_date}")
        
        recent_commits = []
        for commit in commits:
            timestamp_str = commit.get('timestamp', '2020-01-01')
            try:
                # Parse timestamp and make it timezone-naive for comparison
                if 'T' in timestamp_str:
                    # Remove timezone info to make it naive
                    if timestamp_str.endswith('Z'):
                        timestamp_str = timestamp_str[:-1]
                    elif '+' in timestamp_str:
                        timestamp_str = timestamp_str.split('+')[0]
                    elif timestamp_str.endswith('+00:00'):
                        timestamp_str = timestamp_str[:-6]
                    
                    commit_date = datetime.fromisoformat(timestamp_str)
                    
                    print(f"🔍 DEBUG: Commit date {commit_date} vs cutoff {cutoff_date}")
                    if commit_date > cutoff_date:
                        recent_commits.append(commit)
                        print(f"🔍 DEBUG: ✅ Commit included: {commit.get('hash', 'unknown')}")
                    else:
                        print(f"🔍 DEBUG: ❌ Commit too old: {commit.get('hash', 'unknown')}")
                else:
                    print(f"🔍 DEBUG: Skipping commit with invalid timestamp format: {timestamp_str}")
                    
            except Exception as e:
                print(f"🔍 DEBUG: Error parsing timestamp '{timestamp_str}': {e}")
                continue
        
        print(f"🔍 DEBUG: Recent commits after filtering: {len(recent_commits)}")
        
        if not recent_commits:
            print(f"🔍 DEBUG: No recent commits found, returning empty skills")
            return {}
        
        # Analyze each commit
        code_analyses = []
        for i, commit in enumerate(recent_commits):
            try:
                print(f"🔍 DEBUG: Analyzing commit {i+1}/{len(recent_commits)}: {commit.get('hash', 'unknown')}")
                print(f"🔍 DEBUG: Commit files: {len(commit.get('files', []))}")
                
                analysis = self.code_analyzer.analyze_commit(
                    commit.get('files', []),
                    commit
                )
                code_analyses.append(analysis)
                print(f"🔍 DEBUG: Commit analysis complete - complexity: {analysis.complexity_score}")
                
            except Exception as e:
                print(f"🔍 DEBUG: Error analyzing commit {commit.get('hash', 'unknown')}: {e}")
                continue
        
        print(f"🔍 DEBUG: Total successful commit analyses: {len(code_analyses)}")
        
        # Extract skills from analyses
        skills = self.code_analyzer.extract_developer_skills(code_analyses, time_window_days)
        print(f"🔍 DEBUG: Final extracted skills: {skills}")
        
        return skills
        
    def _analyze_collaboration_patterns(self, 
                                      pr_reviews: List[Dict],
                                      issue_comments: List[Dict],
                                      discussions: List[Dict]) -> CollaborationAnalysis:
        """Analyze collaboration patterns from PR reviews, issues, and discussions."""
        
        all_interactions = pr_reviews + issue_comments + discussions
        
        if not all_interactions:
            return CollaborationAnalysis(
                interaction_quality=0.0,
                knowledge_sharing_frequency=0.0,
                technical_leadership=0.0,
                communication_effectiveness=0.0,
                team_influence=0.0
            )
        
        # Analyze interaction quality
        interaction_quality = self._calculate_interaction_quality(all_interactions)
        
        # Calculate knowledge sharing frequency
        knowledge_sharing_frequency = self._calculate_knowledge_sharing(all_interactions)
        
        # Assess technical leadership
        technical_leadership = self._assess_technical_leadership(pr_reviews, issue_comments)
        
        # Evaluate communication effectiveness
        communication_effectiveness = self._evaluate_communication(all_interactions)
        
        # Calculate team influence
        team_influence = self._calculate_team_influence(all_interactions)
        
        return CollaborationAnalysis(
            interaction_quality=interaction_quality,
            knowledge_sharing_frequency=knowledge_sharing_frequency,
            technical_leadership=technical_leadership,
            communication_effectiveness=communication_effectiveness,
            team_influence=team_influence
        )
    
    def _calculate_interaction_quality(self, interactions: List[Dict]) -> float:
        """Calculate quality of interactions based on content analysis."""
        
        if not interactions:
            return 0.0
        
        quality_indicators = []
        
        for interaction in interactions:
            content = interaction.get('content', '').lower()
            
            # Positive indicators
            helpful_keywords = [
                'thanks', 'helpful', 'great', 'excellent', 'good point',
                'agree', 'solution', 'fix', 'resolve', 'clarify'
            ]
            
            # Technical depth indicators
            technical_keywords = [
                'implementation', 'algorithm', 'optimization', 'performance',
                'security', 'testing', 'documentation', 'architecture'
            ]
            
            # Question/answer patterns
            qa_patterns = [
                r'\?', r'how to', r'why', r'what if', r'consider',
                r'suggest', r'recommend', r'alternative'
            ]
            
            helpful_score = sum(1 for keyword in helpful_keywords if keyword in content)
            technical_score = sum(1 for keyword in technical_keywords if keyword in content)
            qa_score = sum(1 for pattern in qa_patterns if re.search(pattern, content))
            
            # Length indicates thoughtfulness (up to a point)
            length_score = min(len(content.split()) / 50, 1.0)
            
            interaction_quality = (
                min(helpful_score / 3, 1.0) * 0.3 +
                min(technical_score / 5, 1.0) * 0.4 +
                min(qa_score / 3, 1.0) * 0.2 +
                length_score * 0.1
            )
            
            quality_indicators.append(interaction_quality)
        
        return np.mean(quality_indicators)
    
    def _calculate_knowledge_sharing(self, interactions: List[Dict]) -> float:
        """Calculate frequency and quality of knowledge sharing."""
        
        if not interactions:
            return 0.0
        
        knowledge_sharing_score = 0.0
        total_interactions = len(interactions)
        
        sharing_patterns = [
            r'here\'s how', r'you can', r'try this', r'example',
            r'documentation', r'link to', r'resource', r'tutorial',
            r'best practice', r'pattern', r'approach', r'recommend',
            r'suggest', r'consider', r'solution'
        ]
        
        for interaction in interactions:
            content = interaction.get('content', '').lower()
            word_count = len(content.split())
            
            # Check for knowledge sharing patterns
            sharing_indicators = sum(1 for pattern in sharing_patterns 
                                if re.search(pattern, content))
            
            if sharing_indicators > 0:
                # Score based on both indicators and content depth
                content_weight = min(word_count / 30, 1.0)  # More content = higher weight
                indicator_weight = min(sharing_indicators / 3, 1.0)  # More indicators = higher weight
                
                interaction_score = (content_weight * 0.6 + indicator_weight * 0.4)
                knowledge_sharing_score += interaction_score
        
        # Normalize by total interactions
        return min(knowledge_sharing_score / total_interactions, 1.0)
    
    def _assess_technical_leadership(self, pr_reviews: List[Dict], 
                                   issue_comments: List[Dict]) -> float:
        """Assess technical leadership based on PR reviews and issue guidance."""
        
        leadership_indicators = 0
        total_opportunities = len(pr_reviews) + len(issue_comments)
        
        if total_opportunities == 0:
            return 0.0
        
        leadership_patterns = [
            r'architecture', r'design pattern', r'refactor', r'optimization',
            r'security concern', r'performance', r'scalability', r'maintainability',
            r'best practice', r'code quality', r'testing strategy'
        ]
        
        for review in pr_reviews:
            content = review.get('content', '').lower()
            
            # Look for architectural guidance
            leadership_score = sum(1 for pattern in leadership_patterns 
                                 if re.search(pattern, content))
            
            # Constructive feedback patterns
            if any(word in content for word in ['suggest', 'consider', 'recommend']) and leadership_score > 0:
                leadership_indicators += 1
        
        for comment in issue_comments:
            content = comment.get('content', '').lower()
            
            # Technical problem-solving leadership
            if any(word in content for word in ['solution', 'approach', 'strategy']) and len(content.split()) > 20:
                leadership_indicators += 1
        
        return leadership_indicators / total_opportunities
    
    def _evaluate_communication(self, interactions: List[Dict]) -> float:
        """Evaluate communication effectiveness."""
        
        if not interactions:
            return 0.0
        
        communication_scores = []
        
        for interaction in interactions:
            content = interaction.get('content', '')
            
            # Clear communication indicators
            clarity_score = 0
            
            # Proper formatting (basic heuristics)
            if '```' in content or '`' in content:  # Code blocks
                clarity_score += 0.2
            
            if any(marker in content for marker in ['1.', '2.', '-', '*']):  # Lists
                clarity_score += 0.2
            
            # Appropriate length (not too short or too long)
            word_count = len(content.split())
            if 10 <= word_count <= 200:
                clarity_score += 0.3
            
            # Question asking (shows engagement)
            if '?' in content:
                clarity_score += 0.1
            
            # Polite language
            polite_words = ['please', 'thanks', 'thank you', 'appreciate']
            if any(word in content.lower() for word in polite_words):
                clarity_score += 0.2
            
            communication_scores.append(min(clarity_score, 1.0))
        
        return np.mean(communication_scores)
    
    def _calculate_team_influence(self, interactions: List[Dict]) -> float:
        """Calculate team influence based on interaction patterns."""
        
        if not interactions:
            return 0.0
        
        # Simple heuristic: interactions that generate responses indicate influence
        responses_generated = 0
        total_interactions = len(interactions)
        
        for interaction in interactions:
            # Look for indicators that this generated discussion
            if interaction.get('replies', 0) > 0:
                responses_generated += 1
            
            # Long, detailed posts often generate discussion
            if len(interaction.get('content', '').split()) > 50:
                responses_generated += 0.5
        
        return min(responses_generated / total_interactions, 1.0)
    
    def _extract_domain_knowledge(self, 
                                issue_comments: List[Dict],
                                pr_descriptions: List[Dict],
                                commit_messages: List[str]) -> Dict[str, float]:
        """Extract domain knowledge from textual content."""
        
        all_text = []
        
        # Collect all text content
        for comment in issue_comments:
            all_text.append(comment.get('content', ''))
        
        for pr in pr_descriptions:
            all_text.append(pr.get('description', ''))
        
        all_text.extend(commit_messages)
        
        combined_text = ' '.join(all_text).lower()
        
        # Calculate domain expertise scores
        domain_scores = {}
        
        for domain, keywords in self.domain_keywords.items():
            score = 0
            for keyword in keywords:
                # Count keyword occurrences
                score += len(re.findall(rf'\b{keyword}\b', combined_text))
            
            # Normalize by text length and keyword count
            text_length = len(combined_text.split())
            if text_length > 0:
                normalized_score = score / (text_length / 100)  # Per 100 words
                domain_scores[domain] = min(normalized_score, 1.0)
            else:
                domain_scores[domain] = 0.0
        
        return domain_scores
    
    def _calculate_learning_velocity(self, commits: List[Dict], 
                               time_window_days: int) -> float:
        """Calculate how quickly developer is learning new skills."""
        
        if len(commits) < 2:
            return 0.0
        
        # Helper function to parse timestamps safely (same as in _extract_code_skills)
        def parse_timestamp(timestamp_str: str) -> datetime:
            try:
                if 'T' in timestamp_str:
                    # Remove timezone info to make it naive
                    if timestamp_str.endswith('Z'):
                        timestamp_str = timestamp_str[:-1]
                    elif '+' in timestamp_str:
                        timestamp_str = timestamp_str.split('+')[0]
                    elif timestamp_str.endswith('+00:00'):
                        timestamp_str = timestamp_str[:-6]
                    
                    return datetime.fromisoformat(timestamp_str)
                else:
                    return datetime.fromisoformat(timestamp_str)
            except Exception:
                return datetime(2020, 1, 1)  # fallback date
        
        # Sort commits by date
        sorted_commits = sorted(commits, 
                            key=lambda x: parse_timestamp(x.get('timestamp', '2020-01-01')))
        
        # Divide time window into periods
        periods = 4
        period_length = time_window_days // periods
        cutoff_date = datetime.utcnow().replace(tzinfo=None) - timedelta(days=time_window_days)
        
        period_skills = []
        
        for i in range(periods):
            period_start = cutoff_date + timedelta(days=i * period_length)
            period_end = cutoff_date + timedelta(days=(i + 1) * period_length)
            
            period_commits = [
                commit for commit in sorted_commits
                if period_start <= parse_timestamp(commit.get('timestamp', '2020-01-01')) < period_end
            ]
            
            if period_commits:
                # Extract unique technical concepts for this period
                concepts = set()
                for commit in period_commits:
                    for file_info in commit.get('files', []):
                        patch = file_info.get('patch', '')
                        concepts.update(self._extract_technical_concepts_from_text(patch))
                
                period_skills.append(concepts)
            else:
                period_skills.append(set())
        
        # Calculate skill acquisition rate
        if len(period_skills) < 2:
            return 0.0
        
        skill_growth = []
        for i in range(1, len(period_skills)):
            new_skills = period_skills[i] - period_skills[i-1]
            skill_growth.append(len(new_skills))
        
        # Return average skill acquisition rate
        return np.mean(skill_growth) / 10  # Normalize
    
    def _extract_technical_concepts_from_text(self, text: str) -> set:
        """Extract technical concepts from text content."""
        concepts = set()
        text_lower = text.lower()
        
        for domain, keywords in self.domain_keywords.items():
            for keyword in keywords:
                if re.search(rf'\b{keyword}\b', text_lower):
                    concepts.add(domain)
        
        return concepts
    
    def _generate_skill_vector(self, 
                             code_skills: Dict,
                             domain_knowledge: Dict,
                             collaboration_analysis: CollaborationAnalysis) -> np.ndarray:
        """Generate unified multi-dimensional skill vector."""
        
        # Initialize skill vector
        skill_vector = np.zeros(settings.SKILL_VECTOR_DIM)
        
        # Programming language skills (first 100 dimensions)
        lang_skills = code_skills.get('programming_languages', {})
        for i, (lang, skill_data) in enumerate(lang_skills.items()):
            if i < 50:  # Limit to first 50 languages
                skill_vector[i * 2] = skill_data.get('proficiency', 0.0)
                skill_vector[i * 2 + 1] = skill_data.get('consistency', 0.0)
        
        # Domain expertise (next 200 dimensions)
        domain_offset = 100
        for i, (domain, score) in enumerate(domain_knowledge.items()):
            if i < 100:  # Limit to first 100 domains
                skill_vector[domain_offset + i * 2] = score
                skill_vector[domain_offset + i * 2 + 1] = code_skills.get('domain_expertise', {}).get(domain, {}).get('expertise_level', 0.0)
        
        # Collaboration skills (next 50 dimensions)
        collab_offset = 300
        skill_vector[collab_offset] = collaboration_analysis.interaction_quality
        skill_vector[collab_offset + 1] = collaboration_analysis.knowledge_sharing_frequency
        skill_vector[collab_offset + 2] = collaboration_analysis.technical_leadership
        skill_vector[collab_offset + 3] = collaboration_analysis.communication_effectiveness
        skill_vector[collab_offset + 4] = collaboration_analysis.team_influence
        
        # Code quality metrics (next 50 dimensions)
        quality_offset = 350
        skill_vector[quality_offset] = code_skills.get('complexity_preference', 0.0)
        skill_vector[quality_offset + 1] = code_skills.get('technical_breadth', 0.0)
        skill_vector[quality_offset + 2] = code_skills.get('code_quality_indicator', 0.0)
        
        # Fill remaining dimensions with noise to prevent overfitting
        remaining_dims = settings.SKILL_VECTOR_DIM - 400
        if remaining_dims > 0:
            skill_vector[400:] = np.random.normal(0, 0.01, remaining_dims)
        
        return skill_vector
    
    def _calculate_confidence_scores(self, 
                                   code_skills: Dict,
                                   collaboration_analysis: CollaborationAnalysis,
                                   developer_data: Dict) -> Dict[str, float]:
        """Calculate confidence scores for different skill areas."""
        
        # Data quantity indicators
        commit_count = len(developer_data.get('commits', []))
        interaction_count = len(developer_data.get('pr_reviews', [])) + len(developer_data.get('issue_comments', []))
        
        # Base confidence on data availability
        code_confidence = min(commit_count / 50, 1.0)  # 50 commits for full confidence
        collaboration_confidence = min(interaction_count / 20, 1.0)  # 20 interactions for full confidence
        
        # Adjust based on consistency
        lang_skills = code_skills.get('programming_languages', {})
        if lang_skills:
            avg_consistency = np.mean([skill.get('consistency', 0.0) for skill in lang_skills.values()])
            code_confidence *= avg_consistency
        
        return {
            'programming_languages': code_confidence,
            'domain_expertise': (code_confidence + collaboration_confidence) / 2,
            'collaboration_skills': collaboration_confidence,
            'overall': (code_confidence + collaboration_confidence) / 2
        }
    
    def update_temporal_skills(self, 
                             current_profile: DeveloperProfile,
                             new_data: Dict,
                             decay_factor: float = 0.95) -> DeveloperProfile:
        """Update developer profile with temporal decay for skill evolution."""
        
        # Extract new skills from recent data
        new_profile = self.extract_comprehensive_profile(new_data)
        
        # Apply temporal decay to existing skills
        decayed_vector = current_profile.skill_vector * decay_factor
        
        # Blend with new skills (weighted average)
        blend_weight = 0.3  # 30% new data, 70% historical
        updated_vector = (decayed_vector * (1 - blend_weight) + 
                         new_profile.skill_vector * blend_weight)
        
        # Update programming languages with recency weighting
        updated_languages = {}
        for lang in set(list(current_profile.programming_languages.keys()) + 
                       list(new_profile.programming_languages.keys())):
            
            old_skill = current_profile.programming_languages.get(lang, {'proficiency': 0.0, 'consistency': 0.0})
            new_skill = new_profile.programming_languages.get(lang, {'proficiency': 0.0, 'consistency': 0.0})
            
            updated_languages[lang] = {
                'proficiency': old_skill['proficiency'] * decay_factor + new_skill['proficiency'] * blend_weight,
                'consistency': (old_skill['consistency'] + new_skill['consistency']) / 2,
                'recent_usage': new_skill.get('recent_usage', 0.0)
            }
        
        # Update domain expertise similarly
        updated_domains = {}
        for domain in set(list(current_profile.domain_expertise.keys()) + 
                         list(new_profile.domain_expertise.keys())):
            
            old_expertise = current_profile.domain_expertise.get(domain, 0.0)
            new_expertise = new_profile.domain_expertise.get(domain, 0.0)
            
            updated_domains[domain] = old_expertise * decay_factor + new_expertise * blend_weight
        
        return DeveloperProfile(
            developer_id=current_profile.developer_id,
            skill_vector=updated_vector,
            programming_languages=updated_languages,
            domain_expertise=updated_domains,
            collaboration_score=(current_profile.collaboration_score * decay_factor + 
                               new_profile.collaboration_score * blend_weight),
            learning_velocity=new_profile.learning_velocity,  # Use latest learning velocity
            confidence_scores=new_profile.confidence_scores,
            last_updated=datetime.utcnow()
        )
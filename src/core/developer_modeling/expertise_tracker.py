import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from collections import defaultdict
import json

from sqlalchemy.orm import Session
from ...models.database import Developer, ExpertiseSnapshot, get_db
from .skill_extractor import SkillExtractor, DeveloperProfile
from ...config import settings

@dataclass
class SkillTrend:
    skill_name: str
    trend_direction: str  # "increasing", "decreasing", "stable"
    change_rate: float
    confidence: float

@dataclass
class LearningRecommendation:
    skill_area: str
    recommendation_type: str  # "strengthen", "explore", "maintain"
    priority: float
    reasoning: str

class ExpertiseTracker:
    """Tracks and manages temporal evolution of developer expertise."""
    
    def __init__(self):
        self.skill_extractor = SkillExtractor()
    
    def track_developer_progress(self, developer_id: int, 
                               new_data: Dict,
                               db: Session) -> DeveloperProfile:
        """Track and update developer's skill progression over time."""
        
        # Get current developer profile
        developer = db.query(Developer).filter(Developer.id == developer_id).first()
        
        if not developer:
            # Create new developer profile
            new_profile = self.skill_extractor.extract_comprehensive_profile(new_data)
            
            developer = Developer(
                github_username=new_data.get('github_username', ''),
                email=new_data.get('email', ''),
                name=new_data.get('name', ''),
                skill_vector=new_profile.skill_vector.tolist(),
                primary_languages=dict(new_profile.programming_languages),
                domain_expertise=dict(new_profile.domain_expertise),
                collaboration_score=new_profile.collaboration_score,
                learning_velocity=new_profile.learning_velocity
            )
            
            db.add(developer)
            db.commit()
            
            # Create initial expertise snapshot
            self._create_expertise_snapshot(developer.id, new_profile, db)
            
            return new_profile
        
        else:
            # Update existing developer profile
            current_profile = DeveloperProfile(
                developer_id=str(developer.id),
                skill_vector=np.array(developer.skill_vector),
                programming_languages=developer.primary_languages or {},
                domain_expertise=developer.domain_expertise or {},
                collaboration_score=developer.collaboration_score,
                learning_velocity=developer.learning_velocity,
                confidence_scores={},  # Will be recalculated
                last_updated=developer.updated_at
            )
            
            # Update with new data
            updated_profile = self.skill_extractor.update_temporal_skills(
                current_profile, new_data
            )
            
            # Update database
            developer.skill_vector = updated_profile.skill_vector.tolist()
            developer.primary_languages = dict(updated_profile.programming_languages)
            developer.domain_expertise = dict(updated_profile.domain_expertise)
            developer.collaboration_score = updated_profile.collaboration_score
            developer.learning_velocity = updated_profile.learning_velocity
            developer.updated_at = datetime.utcnow()
            
            db.commit()
            
            # Create new expertise snapshot
            self._create_expertise_snapshot(developer.id, updated_profile, db)
            
            return updated_profile
    
    def _create_expertise_snapshot(self, developer_id: int, 
                                 profile: DeveloperProfile,
                                 db: Session):
        """Create a snapshot of current expertise for temporal tracking."""
        
        # Calculate performance scores
        productivity_score = self._calculate_productivity_score(profile)
        code_quality_score = self._calculate_code_quality_score(profile)
        collaboration_effectiveness = profile.collaboration_score
        
        # Create learning trends data
        learning_trends = self._analyze_learning_trends(developer_id, profile, db)
        
        snapshot = ExpertiseSnapshot(
            developer_id=developer_id,
            skill_vector=profile.skill_vector.tolist(),
            confidence_scores=profile.confidence_scores,
            learning_trends=learning_trends,
            productivity_score=productivity_score,
            code_quality_score=code_quality_score,
            collaboration_effectiveness=collaboration_effectiveness
        )
        
        db.add(snapshot)
        db.commit()
    
    def _calculate_productivity_score(self, profile: DeveloperProfile) -> float:
        """Calculate productivity score based on skill profile."""
        
        # Base score on technical breadth and depth
        lang_diversity = len(profile.programming_languages)
        domain_diversity = len(profile.domain_expertise)
        
        # Normalize diversity scores
        diversity_score = min((lang_diversity / 5 + domain_diversity / 8) / 2, 1.0)
        
        # Factor in collaboration and learning velocity
        productivity = (
            diversity_score * 0.4 +
            profile.collaboration_score * 0.3 +
            min(profile.learning_velocity, 1.0) * 0.3
        )
        
        return min(productivity, 1.0)
    
    def _calculate_code_quality_score(self, profile: DeveloperProfile) -> float:
        """Calculate code quality score from skill indicators."""
        
        # Extract quality indicators from skill vector
        if len(profile.skill_vector) > 350:
            complexity_preference = profile.skill_vector[350]
            technical_breadth = profile.skill_vector[351]
            code_quality_indicator = profile.skill_vector[352] if len(profile.skill_vector) > 352 else 0.0
        else:
            # Fallback calculation
            complexity_preference = 0.5
            technical_breadth = len(profile.domain_expertise) / 10
            code_quality_indicator = 0.5
        
        # Optimal complexity is around 0.5 (not too simple, not too complex)
        complexity_quality = 1.0 - abs(complexity_preference - 0.5)
        
        quality_score = (
            complexity_quality * 0.4 +
            min(technical_breadth, 1.0) * 0.3 +
            code_quality_indicator * 0.3
        )
        
        return min(quality_score, 1.0)
    
    def _analyze_learning_trends(self, developer_id: int, 
                               current_profile: DeveloperProfile,
                               db: Session) -> Dict:
        """Analyze learning trends by comparing with historical snapshots."""
        
        # Get recent snapshots for comparison
        recent_snapshots = db.query(ExpertiseSnapshot).filter(
            ExpertiseSnapshot.developer_id == developer_id
        ).order_by(ExpertiseSnapshot.snapshot_date.desc()).limit(5).all()
        
        if len(recent_snapshots) < 2:
            return {"trends": [], "overall_direction": "insufficient_data"}
        
        trends = []
        
        # Analyze programming language trends
        for lang, current_skill in current_profile.programming_languages.items():
            lang_trend = self._calculate_skill_trend(
                lang, current_skill['proficiency'], recent_snapshots, 'programming_languages'
            )
            if lang_trend:
                trends.append(lang_trend)
        
        # Analyze domain expertise trends
        for domain, current_expertise in current_profile.domain_expertise.items():
            domain_trend = self._calculate_skill_trend(
                domain, current_expertise, recent_snapshots, 'domain_expertise'
            )
            if domain_trend:
                trends.append(domain_trend)
        
        # Overall learning direction
        positive_trends = sum(1 for trend in trends if trend.trend_direction == "increasing")
        total_trends = len(trends)
        
        if total_trends == 0:
            overall_direction = "stable"
        elif positive_trends / total_trends > 0.6:
            overall_direction = "rapid_growth"
        elif positive_trends / total_trends > 0.4:
            overall_direction = "steady_growth"
        else:
            overall_direction = "needs_attention"
        
        return {
            "trends": [{"skill": t.skill_name, "direction": t.trend_direction, 
                       "rate": t.change_rate, "confidence": t.confidence} for t in trends],
            "overall_direction": overall_direction
        }
    
    def _calculate_skill_trend(self, skill_name: str, current_value: float,
                             snapshots: List[ExpertiseSnapshot],
                             skill_category: str) -> Optional[SkillTrend]:
        """Calculate trend for a specific skill over time."""
        
        values = [current_value]
        dates = [datetime.utcnow()]
        
        # Extract historical values
        for snapshot in snapshots:
            if skill_category == 'programming_languages':
                skill_vector = json.loads(snapshot.skill_vector) if isinstance(snapshot.skill_vector, str) else snapshot.skill_vector
                # Simplified extraction - in real implementation, would need proper indexing
                historical_value = 0.5  # Placeholder
            else:
                # Domain expertise stored directly
                historical_value = 0.5  # Placeholder
            
            values.append(historical_value)
            dates.append(snapshot.snapshot_date)
        
        if len(values) < 3:
            return None
        
        # Calculate trend using simple linear regression
        x = np.arange(len(values))
        y = np.array(values)
        
        # Fit line
        slope, intercept = np.polyfit(x, y, 1)
        
        # Determine trend direction
        if abs(slope) < 0.01:
            direction = "stable"
        elif slope > 0:
            direction = "increasing"
        else:
            direction = "decreasing"
        
        # Calculate confidence based on R-squared
        y_pred = slope * x + intercept
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        return SkillTrend(
            skill_name=skill_name,
            trend_direction=direction,
            change_rate=abs(slope),
            confidence=max(r_squared, 0.0)
        )
    
    def generate_learning_recommendations(self, developer_id: int,
                                        db: Session) -> List[LearningRecommendation]:
        """Generate personalized learning recommendations for developer."""
        
        developer = db.query(Developer).filter(Developer.id == developer_id).first()
        if not developer:
            return []
        
        # Get recent learning trends
        latest_snapshot = db.query(ExpertiseSnapshot).filter(
            ExpertiseSnapshot.developer_id == developer_id
        ).order_by(ExpertiseSnapshot.snapshot_date.desc()).first()
        
        recommendations = []
        
        # Analyze current skill gaps
        current_languages = developer.primary_languages or {}
        current_domains = developer.domain_expertise or {}
        
        # Recommend strengthening weak but used skills
        for lang, skill_data in current_languages.items():
            if isinstance(skill_data, dict) and skill_data.get('proficiency', 0) < 0.6:
                if skill_data.get('recent_usage', 0) > 0.1:  # Recently used
                    recommendations.append(LearningRecommendation(
                        skill_area=f"{lang} programming",
                        recommendation_type="strengthen",
                        priority=0.8,
                        reasoning=f"You've been using {lang} recently but could improve proficiency"
                    ))
        
        # Recommend exploring complementary domains
        domain_suggestions = self._suggest_complementary_domains(current_domains)
        for domain, priority in domain_suggestions:
            recommendations.append(LearningRecommendation(
                skill_area=domain,
                recommendation_type="explore",
                priority=priority,
                reasoning=f"This domain complements your current expertise in {', '.join(current_domains.keys())}"
            ))
        
        # Recommend maintaining strong skills
        for domain, expertise in current_domains.items():
            if expertise > 0.7:
                recommendations.append(LearningRecommendation(
                    skill_area=domain,
                    recommendation_type="maintain",
                    priority=0.6,
                    reasoning=f"You have strong expertise in {domain} - continue practicing to maintain edge"
                ))
        
        # Sort by priority
        recommendations.sort(key=lambda x: x.priority, reverse=True)
        
        return recommendations[:5]  # Return top 5 recommendations
    
    def _suggest_complementary_domains(self, current_domains: Dict[str, float]) -> List[Tuple[str, float]]:
        """Suggest complementary domains based on current expertise."""
        
        suggestions = []
        
        # Domain synergy mapping
        synergies = {
            'frontend': ['backend', 'mobile', 'testing'],
            'backend': ['frontend', 'devops', 'database', 'security'],
            'devops': ['backend', 'security', 'monitoring'],
            'ml': ['backend', 'data_engineering', 'cloud'],
            'mobile': ['frontend', 'backend'],
            'testing': ['devops', 'security'],
            'security': ['backend', 'devops', 'testing']
        }
        
        for current_domain, expertise_level in current_domains.items():
            if expertise_level > 0.5:  # Only suggest based on strong domains
                for suggested_domain in synergies.get(current_domain, []):
                    if suggested_domain not in current_domains:
                        # Priority based on how well it complements
                        priority = min(expertise_level * 0.8, 0.9)
                        suggestions.append((suggested_domain, priority))
        
        # Remove duplicates and sort
        unique_suggestions = list(set(suggestions))
        unique_suggestions.sort(key=lambda x: x[1], reverse=True)
        
        return unique_suggestions[:3]  # Top 3 suggestions
    
    def get_skill_evolution_timeline(self, developer_id: int, 
                                   db: Session,
                                   skill_name: str,
                                   days_back: int = 180) -> Dict:
        """Get timeline of how a specific skill has evolved."""
        
        cutoff_date = datetime.utcnow() - timedelta(days=days_back)
        
        snapshots = db.query(ExpertiseSnapshot).filter(
            ExpertiseSnapshot.developer_id == developer_id,
            ExpertiseSnapshot.snapshot_date >= cutoff_date
        ).order_by(ExpertiseSnapshot.snapshot_date.asc()).all()
        
        timeline = {
            'skill_name': skill_name,
            'data_points': [],
            'trend_analysis': {}
        }
        
        for snapshot in snapshots:
            # Extract skill value from snapshot
            # This would need proper implementation based on skill vector structure
            skill_value = self._extract_skill_from_snapshot(snapshot, skill_name)
            
            timeline['data_points'].append({
                'date': snapshot.snapshot_date.isoformat(),
                'value': skill_value,
                'confidence': snapshot.confidence_scores.get(skill_name, 0.5) if snapshot.confidence_scores else 0.5
            })
        
        # Calculate trend analysis
        if len(timeline['data_points']) >= 3:
            values = [point['value'] for point in timeline['data_points']]
            timeline['trend_analysis'] = {
                'direction': 'increasing' if values[-1] > values[0] else 'decreasing',
                'average_growth_rate': (values[-1] - values[0]) / len(values),
                'volatility': np.std(values),
                'current_trajectory': 'positive' if values[-1] > values[-2] else 'negative'
            }
        
        return timeline
    
    def _extract_skill_from_snapshot(self, snapshot: ExpertiseSnapshot, skill_name: str) -> float:
        """Extract specific skill value from expertise snapshot."""
        
        # This is a simplified implementation
        # In practice, would need proper mapping from skill names to vector positions
        
        if hasattr(snapshot, 'learning_trends') and snapshot.learning_trends:
            trends = snapshot.learning_trends
            if isinstance(trends, str):
                trends = json.loads(trends)
            
            for trend in trends.get('trends', []):
                if trend.get('skill') == skill_name:
                    return trend.get('rate', 0.5)
        
        # Fallback to average skill level
        return 0.5
    
    def compare_developers(self, developer_ids: List[int], 
                         skill_areas: List[str],
                         db: Session) -> Dict:
        """Compare developers across specific skill areas."""
        
        comparison = {
            'developers': {},
            'skill_areas': skill_areas,
            'rankings': {}
        }
        
        for dev_id in developer_ids:
            developer = db.query(Developer).filter(Developer.id == dev_id).first()
            if not developer:
                continue
            
            dev_skills = {}
            
            # Extract programming language skills
            for lang, skill_data in (developer.primary_languages or {}).items():
                if lang in skill_areas:
                    if isinstance(skill_data, dict):
                        dev_skills[lang] = skill_data.get('proficiency', 0.0)
                    else:
                        dev_skills[lang] = skill_data
            
            # Extract domain expertise
            for domain, expertise in (developer.domain_expertise or {}).items():
                if domain in skill_areas:
                    dev_skills[domain] = expertise
            
            comparison['developers'][dev_id] = {
                'name': developer.name or developer.github_username,
                'skills': dev_skills,
                'collaboration_score': developer.collaboration_score,
                'learning_velocity': developer.learning_velocity
            }
        
        # Calculate rankings for each skill area
        for skill_area in skill_areas:
            skill_scores = []
            for dev_id, dev_data in comparison['developers'].items():
                score = dev_data['skills'].get(skill_area, 0.0)
                skill_scores.append((dev_id, score))
            
            # Sort by score (descending)
            skill_scores.sort(key=lambda x: x[1], reverse=True)
            comparison['rankings'][skill_area] = skill_scores
        
        return comparison
    
    def predict_skill_development(self, developer_id: int, 
                                db: Session,
                                skill_name: str,
                                months_ahead: int = 6) -> Dict:
        """Predict future skill development based on current trends."""
        
        # Get historical data
        timeline = self.get_skill_evolution_timeline(developer_id, db, skill_name, days_back=180)
        
        if len(timeline['data_points']) < 3:
            return {
                'prediction': 'insufficient_data',
                'confidence': 0.0,
                'projected_value': 0.0
            }
        
        # Extract values and timestamps
        values = [point['value'] for point in timeline['data_points']]
        dates = [datetime.fromisoformat(point['date']) for point in timeline['data_points']]
        
        # Simple linear extrapolation
        x = np.arange(len(values))
        slope, intercept = np.polyfit(x, values, 1)
        
        # Project forward
        future_x = len(values) + (months_ahead * 30 / (len(values) > 1 and (dates[-1] - dates[0]).days / len(values) or 30))
        projected_value = slope * future_x + intercept
        
        # Clamp between 0 and 1
        projected_value = max(0.0, min(1.0, projected_value))
        
        # Calculate confidence based on trend consistency
        trend_consistency = 1.0 - np.std(np.diff(values)) if len(values) > 2 else 0.5
        
        prediction_type = "growth" if slope > 0.01 else ("decline" if slope < -0.01 else "stable")
        
        return {
            'prediction': prediction_type,
            'confidence': trend_consistency,
            'projected_value': projected_value,
            'growth_rate': slope,
            'months_projected': months_ahead
        }
    
    def get_team_skill_matrix(self, team_developer_ids: List[int], 
                            db: Session) -> Dict:
        """Generate comprehensive skill matrix for a team."""
        
        # Get all developers
        developers = db.query(Developer).filter(Developer.id.in_(team_developer_ids)).all()
        
        # Collect all skills across team
        all_languages = set()
        all_domains = set()
        
        for dev in developers:
            if dev.primary_languages:
                all_languages.update(dev.primary_languages.keys())
            if dev.domain_expertise:
                all_domains.update(dev.domain_expertise.keys())
        
        # Build skill matrix
        skill_matrix = {
            'team_members': [],
            'programming_languages': list(all_languages),
            'domain_expertise': list(all_domains),
            'collaboration_metrics': {},
            'team_strengths': [],
            'team_gaps': []
        }
        
        for dev in developers:
            member_skills = {
                'id': dev.id,
                'name': dev.name or dev.github_username,
                'programming_languages': {},
                'domain_expertise': {},
                'collaboration_score': dev.collaboration_score,
                'learning_velocity': dev.learning_velocity
            }
            
            # Fill in programming language skills
            for lang in all_languages:
                if dev.primary_languages and lang in dev.primary_languages:
                    skill_data = dev.primary_languages[lang]
                    if isinstance(skill_data, dict):
                        member_skills['programming_languages'][lang] = skill_data.get('proficiency', 0.0)
                    else:
                        member_skills['programming_languages'][lang] = skill_data
                else:
                    member_skills['programming_languages'][lang] = 0.0
            
            # Fill in domain expertise
            for domain in all_domains:
                if dev.domain_expertise and domain in dev.domain_expertise:
                    member_skills['domain_expertise'][domain] = dev.domain_expertise[domain]
                else:
                    member_skills['domain_expertise'][domain] = 0.0
            
            skill_matrix['team_members'].append(member_skills)
        
        # Analyze team strengths and gaps
        skill_matrix['team_strengths'] = self._identify_team_strengths(skill_matrix)
        skill_matrix['team_gaps'] = self._identify_team_gaps(skill_matrix)
        
        # Calculate team collaboration metrics
        skill_matrix['collaboration_metrics'] = {
            'average_collaboration_score': np.mean([dev.collaboration_score for dev in developers]),
            'team_learning_velocity': np.mean([dev.learning_velocity for dev in developers]),
            'skill_diversity': len(all_languages) + len(all_domains)
        }
        
        return skill_matrix
    
    def _identify_team_strengths(self, skill_matrix: Dict) -> List[str]:
        """Identify areas where team has strong collective expertise."""
        
        strengths = []
        
        # Check programming languages
        for lang in skill_matrix['programming_languages']:
            avg_proficiency = np.mean([
                member['programming_languages'][lang] 
                for member in skill_matrix['team_members']
            ])
            if avg_proficiency > 0.6:
                strengths.append(f"{lang} programming")
        
        # Check domain expertise
        for domain in skill_matrix['domain_expertise']:
            avg_expertise = np.mean([
                member['domain_expertise'][domain] 
                for member in skill_matrix['team_members']
            ])
            if avg_expertise > 0.6:
                strengths.append(f"{domain} domain")
        
        return strengths
    
    def _identify_team_gaps(self, skill_matrix: Dict) -> List[str]:
        """Identify areas where team lacks expertise."""
        
        gaps = []
        
        # Check programming languages
        for lang in skill_matrix['programming_languages']:
            max_proficiency = max([
                member['programming_languages'][lang] 
                for member in skill_matrix['team_members']
            ])
            if max_proficiency < 0.4:
                gaps.append(f"{lang} programming")
        
        # Check domain expertise
        for domain in skill_matrix['domain_expertise']:
            max_expertise = max([
                member['domain_expertise'][domain] 
                for member in skill_matrix['team_members']
            ])
            if max_expertise < 0.4:
                gaps.append(f"{domain} domain")
        
        return gaps
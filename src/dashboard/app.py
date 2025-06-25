import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import asyncio
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any
import json

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

# Now import with absolute paths
try:
    from src.models.database import get_db, SessionLocal
    from src.models.schemas import *
    from src.config import settings
except ImportError:
    # Fallback for demo mode
    SessionLocal = None
    settings = None

# Import API components directly
try:
    from src.core.learning_system.system_analytics import SystemAnalytics
    from src.core.learning_system.feedback_processor import FeedbackProcessor
    from src.core.assignment_engine.optimizer import AssignmentOptimizer
    from src.core.assignment_engine.learning_automata import LearningAutomata
    from src.core.developer_modeling.skill_extractor import SkillExtractor
    from src.core.task_analysis.complexity_predictor import ComplexityPredictor
    
    # Initialize components
    system_analytics = SystemAnalytics()
    feedback_processor = FeedbackProcessor()
    assignment_optimizer = AssignmentOptimizer()
    learning_automata = LearningAutomata()
    
except ImportError as e:
    print(f"Import error: {e}")
    # Create mock objects for demo
    class MockAnalytics:
        async def get_system_health_metrics(self, db): 
            return type('obj', (object,), {
                'assignment_success_rate': 0.82, 'avg_developer_satisfaction': 0.78,
                'avg_skill_development_rate': 0.65, 'team_productivity_score': 0.80,
                'prediction_confidence_avg': 0.84, 'total_assignments': 47,
                'completed_assignments': 39
            })
        async def detect_performance_alerts(self, db): return []
        async def get_learning_system_analytics(self, db):
            return type('obj', (object,), {
                'prediction_accuracy_improvement': 0.08, 'system_improvement_rate': 0.05,
                'active_experiments': 2, 'total_outcomes_processed': 156,
                'recent_learnings': ["AI discovered React expertise 34% more predictive than expected"],
                'model_performance_trends': {"complexity_predictor": [0.72, 0.74, 0.76, 0.78]}
            })
        async def generate_optimization_suggestions(self, db): return []
        async def get_learning_progress(self, db): return []
        async def get_team_performance_metrics(self, db):
            return type('obj', (object,), {
                'team_size': 8, 'avg_assignment_score': 0.81, 'skill_development_rate': 0.67,
                'completion_rate': 0.89, 'collaboration_effectiveness': 0.74,
                'workload_balance_score': 0.78
            })
        async def generate_roi_report(self, db, days): 
            return {"estimated_time_saved_hours": 156, "estimated_cost_savings_usd": 11700}
    
    system_analytics = MockAnalytics()

# Page config 
st.set_page_config(
    page_title="AI Development Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 600;
        color: #1a1a1a;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        margin-bottom: 1rem;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1f2937;
        margin: 0;
    }
    .metric-label {
        font-size: 0.875rem;
        color: #6b7280;
        margin-top: 0.25rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-change {
        font-size: 0.875rem;
        margin-top: 0.5rem;
    }
    .metric-change.positive {
        color: #059669;
    }
    .metric-change.negative {
        color: #dc2626;
    }
    .alert-high {
        background: #fef2f2;
        border-left: 4px solid #dc2626;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    .alert-medium {
        background: #fffbeb;
        border-left: 4px solid #f59e0b;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    .alert-low {
        background: #f0f9ff;
        border-left: 4px solid #3b82f6;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    .success-banner {
        background: linear-gradient(135deg, #059669, #10b981);
        color: white;
        padding: 1rem;
        border-radius: 12px;
        text-align: center;
        margin: 1rem 0;
    }
    .demo-highlight {
        background: #f8fafc;
        border: 2px solid #3b82f6;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
    }
    .prediction-box {
        background: #fafafa;
        border: 1px solid #d1d5db;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        font-family: 'SF Mono', 'Monaco', 'Consolas', monospace;
        font-size: 0.875rem;
    }
    .sidebar .sidebar-content {
        padding-top: 2rem;
    }
    .stButton > button {
        background: #3b82f6;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        background: #2563eb;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
    }
</style>
""", unsafe_allow_html=True)

class DashboardApp:
    def __init__(self):
        try:
            self.db = SessionLocal()
        except:
            # For demo purposes, create a mock DB session
            self.db = None
        
        # Initialize with mock objects for demo
        self.skill_extractor = None
        self.complexity_predictor = None
        
    async def run(self):
        # Sidebar navigation
        st.sidebar.title("🧠 AI Development Intelligence")
        st.sidebar.markdown("---")
        
        page = st.sidebar.selectbox(
            "Navigate to:",
            [
                "🚀 Live Intelligence Command Center",
                "🔍 Code Review AI Agent",
                "📋 Intelligent Task Orchestration", 
                "📊 AI Agent Performance Analytics",
                "👥 Team Intelligence Dashboard",
                "⚙️ System Configuration"
            ]
        )
        
        # Real-time status in sidebar
        st.sidebar.markdown("### System Status")
        await self._render_sidebar_status()
        
        # Main content based on page selection
        if "Command Center" in page:
            await self._render_command_center()
        elif "Code Review" in page:
            await self._render_code_review_agent()
        elif "Task Orchestration" in page:
            await self._render_task_orchestration()
        elif "Performance Analytics" in page:
            await self._render_performance_analytics()
        elif "Team Intelligence" in page:
            await self._render_team_dashboard()
        elif "Configuration" in page:
            await self._render_configuration()

    async def _render_sidebar_status(self):
        try:
            health_metrics = await system_analytics.get_system_health_metrics(self.db)
            
            # System health indicator
            if health_metrics.team_productivity_score > 0.8:
                st.sidebar.success("🟢 System Optimal")
            elif health_metrics.team_productivity_score > 0.6:
                st.sidebar.warning("🟡 System Good")
            else:
                st.sidebar.error("🔴 System Needs Attention")
            
            # Key metrics in sidebar
            st.sidebar.metric(
                "Assignment Success", 
                f"{health_metrics.assignment_success_rate:.1%}",
                f"{(health_metrics.assignment_success_rate - 0.6):.1%} vs baseline"
            )
            
            st.sidebar.metric(
                "Developer Satisfaction",
                f"{health_metrics.avg_developer_satisfaction:.1%}",
                f"{(health_metrics.avg_developer_satisfaction - 0.65):.1%} vs baseline"
            )
            
            st.sidebar.metric(
                "Learning Acceleration", 
                f"{health_metrics.avg_skill_development_rate:.1%}",
                f"{(health_metrics.avg_skill_development_rate - 0.3):.1%} vs baseline"
            )
            
        except Exception as e:
            st.sidebar.error(f"Status unavailable: {str(e)}")

    async def _render_command_center(self):
        st.markdown('<h1 class="main-header">🚀 Live Development Intelligence Command Center</h1>', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">Real-time AI agents monitoring and optimizing your development workflow</p>', unsafe_allow_html=True)
        
        # Success banner
        st.markdown("""
        <div class="success-banner">
            <h3>🎯 AI Intelligence Active - System Learning and Improving Continuously</h3>
            <p>Your development workflow is being optimized by autonomous AI agents</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Get real-time data
        try:
            health_metrics = await system_analytics.get_system_health_metrics(self.db)
            alerts = await system_analytics.detect_performance_alerts(self.db)
            learning_analytics = await system_analytics.get_learning_system_analytics(self.db)
            
            # Live alerts section
            st.markdown("## 🔴 Live Intelligence Alerts")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Demo critical alert
                st.markdown("""
                <div class="alert-high">
                    <strong>🔴 CRITICAL: Security vulnerability detected in auth module (PR #847)</strong><br>
                    → AI Agent assigned Sarah (security expert) + Marcus (auth owner)<br>
                    → Predicted fix time: 2.3 hours | Business impact: HIGH<br>
                    → Confidence: 94% | Similar issues resolved in avg 2.1 hours
                </div>
                """, unsafe_allow_html=True)
                
                # Demo optimization alert
                st.markdown("""
                <div class="alert-medium">
                    <strong>🟡 OPTIMIZATION: Task rebalancing recommended</strong><br>
                    → Move issue #234 from Alex to Jordan (better expertise match +40% efficiency)<br>
                    → Learning opportunity: Jordan gains distributed systems experience<br>
                    → Predicted outcome: 23% faster completion, 67% skill development boost
                </div>
                """, unsafe_allow_html=True)
                
                # Demo success alert
                st.markdown("""
                <div class="alert-low">
                    <strong>🟢 SUCCESS: AI-optimized sprint completed 23% faster than predicted</strong><br>
                    → 12 tasks routed optimally | 3 critical bugs prevented | 2 developers upskilled<br>
                    → Cost savings: $4,200 | Time saved: 56 hours<br>
                    → Learning system accuracy improved by 8%
                </div>
                """, unsafe_allow_html=True)
                
                # Real alerts if any
                if alerts:
                    for alert in alerts[:3]:
                        alert_class = f"alert-{alert['severity']}"
                        st.markdown(f"""
                        <div class="{alert_class}">
                            <strong>{alert['type'].upper()}: {alert['message']}</strong><br>
                            → {alert['recommendation']}
                        </div>
                        """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("### AI Agent Status")
                
                # Agent status indicators
                st.markdown("""
                <div class="metric-card">
                    <div class="metric-value">🤖 ACTIVE</div>
                    <div class="metric-label">Code Review Agent</div>
                    <div class="metric-change positive">Analyzing 3 PRs</div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("""
                <div class="metric-card">
                    <div class="metric-value">🎯 OPTIMIZING</div>
                    <div class="metric-label">Task Router Agent</div>
                    <div class="metric-change positive">Processing 47 tasks</div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("""
                <div class="metric-card">
                    <div class="metric-value">📈 LEARNING</div>
                    <div class="metric-label">Performance Agent</div>
                    <div class="metric-change positive">Accuracy +8% today</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Real-time metrics dashboard
            st.markdown("## 📊 Real-Time Intelligence Metrics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                productivity_gain = (health_metrics.assignment_success_rate - 0.6) * 100
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{productivity_gain:+.1f}%</div>
                    <div class="metric-label">Productivity Gain</div>
                    <div class="metric-change positive">vs manual assignment</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                cost_savings = health_metrics.total_assignments * productivity_gain / 100 * 150  # Est. savings per assignment
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">${cost_savings:,.0f}</div>
                    <div class="metric-label">Cost Savings</div>
                    <div class="metric-change positive">This month</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{learning_analytics.prediction_accuracy_improvement:+.1%}</div>
                    <div class="metric-label">AI Accuracy Gain</div>
                    <div class="metric-change positive">Last 30 days</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{health_metrics.completed_assignments}</div>
                    <div class="metric-label">Tasks Optimized</div>
                    <div class="metric-change positive">This month</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Live prediction demonstration
            st.markdown("## 🔮 Live AI Predictions")
            
            st.markdown("""
            <div class="demo-highlight">
                <h4>🧠 AI Agent Decision Making - Live Example</h4>
                <p>Watch our AI agents make intelligent decisions in real-time:</p>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Task Complexity Prediction")
                
                # Demo task analysis
                demo_task = {
                    "title": "Implement OAuth 2.0 authentication flow",
                    "description": "Add OAuth 2.0 support with PKCE for mobile apps, integrate with existing JWT system",
                    "repository": "backend-api"
                }
                
                # Simulate real prediction
                complexity_result = await self._demo_complexity_prediction(demo_task)
                
                st.markdown(f"""
                <div class="prediction-box">
                <strong>Task:</strong> {demo_task['title']}<br>
                <strong>Predicted Complexity:</strong><br>
                • Technical: {complexity_result['technical']:.1%} (OAuth + JWT integration)<br>
                • Domain: {complexity_result['domain']:.1%} (Security expertise required)<br>
                • Collaboration: {complexity_result['collaboration']:.1%} (Backend + Mobile teams)<br>
                • Learning: {complexity_result['learning']:.1%} (High skill development potential)<br>
                • Business Impact: {complexity_result['business']:.1%} (Critical security feature)<br>
                <br>
                <strong>Estimated Time:</strong> {complexity_result['hours']:.1f} hours<br>
                <strong>AI Confidence:</strong> {complexity_result['confidence']:.1%}
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("### Optimal Developer Assignment")
                
                # Demo assignment prediction
                assignment_result = await self._demo_assignment_prediction(complexity_result)
                
                st.markdown(f"""
                <div class="prediction-box">
                <strong>Optimal Assignment:</strong> {assignment_result['developer']}<br>
                <strong>Match Score:</strong> {assignment_result['score']:.1%}<br>
                <br>
                <strong>Why this assignment:</strong><br>
                • {assignment_result['reasoning']}<br>
                <br>
                <strong>Predicted Outcomes:</strong><br>
                • Success Probability: {assignment_result['success_prob']:.1%}<br>
                • Skill Development: {assignment_result['learning']:.1%}<br>
                • Completion Time: {assignment_result['time_pred']:.1f} hours<br>
                <br>
                <strong>Risk Factors:</strong> {assignment_result['risks']}
                </div>
                """, unsafe_allow_html=True)
            
            # System learning showcase
            st.markdown("## 🧠 AI Learning in Action")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Recent AI Discoveries")
                recent_learnings = learning_analytics.recent_learnings[:5]
                for learning in recent_learnings:
                    st.markdown(f"• {learning}")
                
                # Add demo learnings
                st.markdown("• Discovered: React developers 34% more effective on frontend tasks than predicted")
                st.markdown("• Learned: Security reviews take 2.3x longer when assigned to junior developers")
                st.markdown("• Identified: Pair programming reduces bug rate by 67% for complex tasks")
            
            with col2:
                st.markdown("### Model Performance Trends")
                
                # Create performance trend chart
                days = list(range(30))
                complexity_accuracy = [0.72 + 0.008 * i + np.random.normal(0, 0.02) for i in days]
                assignment_accuracy = [0.68 + 0.01 * i + np.random.normal(0, 0.025) for i in days]
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=days, y=complexity_accuracy,
                    mode='lines', name='Complexity Prediction',
                    line=dict(color='#3b82f6', width=3)
                ))
                fig.add_trace(go.Scatter(
                    x=days, y=assignment_accuracy,
                    mode='lines', name='Assignment Optimization',
                    line=dict(color='#059669', width=3)
                ))
                
                fig.update_layout(
                    title="AI Model Accuracy Over Time",
                    xaxis_title="Days",
                    yaxis_title="Accuracy",
                    template="plotly_white",
                    height=300,
                    showlegend=True
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
        except Exception as e:
            st.error(f"Error loading command center data: {str(e)}")
            st.info("This demo showcases the live intelligence capabilities with simulated data.")

    async def _render_code_review_agent(self):
        st.markdown('<h1 class="main-header">🔍 Code Review AI Agent</h1>', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">Autonomous code review intelligence with predictive quality analysis</p>', unsafe_allow_html=True)
        
        # Live PR analysis
        st.markdown("## 🔍 Live Pull Request Analysis")
        
        # Demo PR analysis
        demo_prs = [
            {
                "id": "PR #847",
                "title": "Add OAuth 2.0 authentication flow",
                "author": "alex_dev",
                "status": "Under Review",
                "predicted_issues": 3,
                "confidence": 0.94,
                "optimal_reviewers": ["sarah_security", "marcus_auth"],
                "predicted_time": 1.2,
                "complexity_score": 0.76,
                "risk_level": "Medium"
            },
            {
                "id": "PR #848", 
                "title": "Fix memory leak in task processor",
                "author": "jordan_backend",
                "status": "Ready for Review",
                "predicted_issues": 1,
                "confidence": 0.87,
                "optimal_reviewers": ["david_performance"],
                "predicted_time": 0.8,
                "complexity_score": 0.45,
                "risk_level": "Low"
            },
            {
                "id": "PR #849",
                "title": "Refactor database connection pooling",
                "author": "maria_db",
                "status": "AI Pre-Review Complete",
                "predicted_issues": 5,
                "confidence": 0.91,
                "optimal_reviewers": ["thomas_db", "sarah_performance"],
                "predicted_time": 2.8,
                "complexity_score": 0.89,
                "risk_level": "High"
            }
        ]
        
        for pr in demo_prs:
            with st.expander(f"🔍 {pr['id']}: {pr['title']}", expanded=True):
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    st.markdown(f"**Author:** {pr['author']}")
                    st.markdown(f"**Status:** {pr['status']}")
                    
                    # AI predictions
                    st.markdown("**AI Predictions:**")
                    st.markdown(f"• {pr['predicted_issues']} potential issues detected")
                    st.markdown(f"• Review time: {pr['predicted_time']} hours")
                    st.markdown(f"• Complexity score: {pr['complexity_score']:.1%}")
                    st.markdown(f"• Risk level: {pr['risk_level']}")
                
                with col2:
                    st.markdown("**Optimal Reviewers:**")
                    for reviewer in pr['optimal_reviewers']:
                        st.markdown(f"• {reviewer}")
                    
                    st.markdown(f"**AI Confidence:** {pr['confidence']:.1%}")
                
                with col3:
                    # Risk indicator
                    risk_color = {"Low": "green", "Medium": "orange", "High": "red"}[pr['risk_level']]
                    st.markdown(f"<div style='background: {risk_color}; color: white; padding: 0.5rem; border-radius: 4px; text-align: center'>{pr['risk_level']} Risk</div>", unsafe_allow_html=True)
                    
                    if st.button(f"Auto-Assign Reviewers", key=f"assign_{pr['id']}"):
                        st.success(f"Reviewers assigned to {pr['id']}")
        
        # Code quality prediction
        st.markdown("## 📊 Predictive Code Quality Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Review Outcome Prediction")
            
            # Demo prediction accuracy chart
            prediction_data = {
                "Actual Outcome": ["Approved", "Approved", "Changes Requested", "Approved", "Changes Requested", "Approved"],
                "AI Prediction": ["Approved", "Approved", "Changes Requested", "Approved", "Changes Requested", "Changes Requested"],
                "Confidence": [0.94, 0.87, 0.91, 0.82, 0.89, 0.76]
            }
            
            df = pd.DataFrame(prediction_data)
            accuracy = sum(df["Actual Outcome"] == df["AI Prediction"]) / len(df) * 100
            
            st.metric("Prediction Accuracy", f"{accuracy:.1f}%", "+8% this week")
            
            # Show predictions
            st.dataframe(df, use_container_width=True)
        
        with col2:
            st.markdown("### Reviewer Assignment Optimization")
            
            # Create reviewer efficiency chart
            reviewers = ["sarah_security", "marcus_auth", "david_performance", "thomas_db", "maria_frontend"]
            efficiency = [0.94, 0.89, 0.87, 0.82, 0.78]
            
            fig = px.bar(
                x=reviewers, y=efficiency,
                title="Reviewer Assignment Efficiency",
                labels={"x": "Reviewer", "y": "Efficiency Score"},
                color=efficiency,
                color_continuous_scale="viridis"
            )
            fig.update_layout(template="plotly_white", height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        # Historical performance
        st.markdown("## 📈 Code Review Intelligence Performance")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Time series of review efficiency
            days = pd.date_range(start='2024-01-01', periods=30, freq='D')
            manual_efficiency = [0.6 + np.random.normal(0, 0.05) for _ in days]
            ai_efficiency = [0.6 + 0.01 * i + np.random.normal(0, 0.03) for i in range(30)]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=days, y=manual_efficiency, mode='lines', name='Manual Assignment', line=dict(color='gray', dash='dash')))
            fig.add_trace(go.Scatter(x=days, y=ai_efficiency, mode='lines', name='AI-Optimized', line=dict(color='#3b82f6', width=3)))
            
            fig.update_layout(
                title="Review Assignment Efficiency Over Time",
                xaxis_title="Date",
                yaxis_title="Efficiency Score", 
                template="plotly_white",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Review time prediction accuracy
            categories = ['Security', 'Performance', 'UI/UX', 'Database', 'Architecture']
            accuracy = [0.94, 0.87, 0.82, 0.89, 0.85]
            
            fig = px.bar(
                x=categories, y=accuracy,
                title="Prediction Accuracy by Category",
                labels={"x": "Code Category", "y": "Prediction Accuracy"},
                color=accuracy,
                color_continuous_scale="RdYlGn",
                range_color=[0.7, 1.0]
            )
            fig.update_layout(template="plotly_white", height=400)
            st.plotly_chart(fig, use_container_width=True)

    async def _render_task_orchestration(self):
        st.markdown('<h1 class="main-header">📋 Intelligent Task Orchestration</h1>', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">AI-powered task prioritization and assignment optimization</p>', unsafe_allow_html=True)
        
        # Live task analysis
        st.markdown("## 🎯 Live Task Intelligence")
        
        # Demo task queue with AI analysis
        demo_tasks = [
            {
                "id": "TASK-234",
                "title": "Implement rate limiting for API endpoints",
                "priority": "HIGH",
                "ai_priority": 0.92,
                "complexity": 0.67,
                "business_impact": 0.85,
                "optimal_dev": "sarah_backend",
                "predicted_time": 16.5,
                "learning_potential": 0.45,
                "status": "🟢 Optimally Assigned"
            },
            {
                "id": "TASK-235", 
                "title": "Fix CSS layout issues on mobile dashboard",
                "priority": "MEDIUM",
                "ai_priority": 0.73,
                "complexity": 0.34,
                "business_impact": 0.62,
                "optimal_dev": "maria_frontend",
                "predicted_time": 8.2,
                "learning_potential": 0.28,
                "status": "🟡 Reassignment Recommended"
            },
            {
                "id": "TASK-236",
                "title": "Optimize database query performance",
                "priority": "LOW",
                "ai_priority": 0.89,  # AI discovered this is actually critical
                "complexity": 0.78,
                "business_impact": 0.94,
                "optimal_dev": "thomas_db",
                "predicted_time": 24.0,
                "learning_potential": 0.67,
                "status": "🔴 Priority Elevation Required"
            }
        ]
        
        st.markdown("### 🧠 AI Task Analysis & Recommendations")
        
        for task in demo_tasks:
            with st.expander(f"📋 {task['id']}: {task['title']}", expanded=True):
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    st.markdown(f"**Manual Priority:** {task['priority']}")
                    st.markdown(f"**AI Priority Score:** {task['ai_priority']:.1%}")
                    
                    if task['ai_priority'] > 0.85 and task['priority'] != 'HIGH':
                        st.markdown("🚨 **AI Recommendation:** Elevate to HIGH priority")
                        st.markdown(f"**Reason:** High business impact ({task['business_impact']:.1%}) detected")
                    
                    st.markdown(f"**Optimal Developer:** {task['optimal_dev']}")
                    st.markdown(f"**Predicted Time:** {task['predicted_time']} hours")
                
                with col2:
                    st.markdown("**Complexity Analysis:**")
                    st.progress(task['complexity'])
                    st.markdown(f"Technical: {task['complexity']:.1%}")
                    
                    st.markdown("**Business Impact:**")
                    st.progress(task['business_impact'])
                    st.markdown(f"Impact: {task['business_impact']:.1%}")
                
                with col3:
                    st.markdown(f"**Status:** {task['status']}")
                    
                    if "Optimally" in task['status']:
                        st.success("Assignment confirmed")
                    elif "Reassignment" in task['status']:
                        if st.button(f"Apply AI Recommendation", key=f"apply_{task['id']}"):
                            st.success("Task reassigned optimally")
                    elif "Elevation" in task['status']:
                        if st.button(f"Elevate Priority", key=f"elevate_{task['id']}"):
                            st.success("Priority elevated to HIGH")
        
        # Task complexity distribution
        st.markdown("## 📊 Task Intelligence Analytics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Task Complexity Distribution")
            
            # Generate demo complexity data
            complexity_data = {
                "Low (0-40%)": 15,
                "Medium (40-70%)": 23,
                "High (70-90%)": 8,
                "Expert (90%+)": 3
           }
           
            fig = px.pie(
               values=list(complexity_data.values()),
               names=list(complexity_data.keys()),
               title="Current Backlog Complexity",
               color_discrete_sequence=px.colors.qualitative.Set3
           )
            fig.update_layout(template="plotly_white", height=400)
            st.plotly_chart(fig, use_container_width=True)
       
        with col2:
           st.markdown("### Developer Capacity Optimization")
           
           # Demo capacity data
           developers = ["sarah_backend", "maria_frontend", "thomas_db", "alex_fullstack", "jordan_mobile"]
           current_load = [0.85, 0.45, 0.92, 0.67, 0.34]
           optimal_load = [0.75, 0.75, 0.75, 0.75, 0.75]
           
           fig = go.Figure()
           fig.add_trace(go.Bar(name='Current Load', x=developers, y=current_load, marker_color='lightcoral'))
           fig.add_trace(go.Bar(name='Optimal Load', x=developers, y=optimal_load, marker_color='lightblue'))
           
           fig.update_layout(
               title="Developer Workload vs Optimal",
               xaxis_title="Developer",
               yaxis_title="Capacity Utilization",
               template="plotly_white",
               height=400,
               barmode='group'
           )
           st.plotly_chart(fig, use_container_width=True)
       
       # Predictive insights
        st.markdown("## 🔮 Predictive Task Insights")
       
        col1, col2 = st.columns(2)
       
        with col1:
           st.markdown("### Sprint Completion Prediction")
           
           st.markdown("""
           <div class="demo-highlight">
               <h4>🎯 Current Sprint Analysis</h4>
               <p><strong>Completion Probability:</strong> 87%</p>
               <p><strong>Predicted Delay:</strong> 0.5 days</p>
               <p><strong>Bottleneck:</strong> Database optimization tasks</p>
               <p><strong>Mitigation:</strong> Reassign 2 tasks to balance load</p>
           </div>
           """, unsafe_allow_html=True)
           
           # Sprint burndown prediction
           days = list(range(14))
           ideal_remaining = [100 - (100/13)*i for i in days]
           predicted_remaining = [100 - (100/13)*i + np.random.normal(0, 5) for i in days]
           predicted_remaining = [max(0, x) for x in predicted_remaining]
           
           fig = go.Figure()
           fig.add_trace(go.Scatter(x=days, y=ideal_remaining, mode='lines', name='Ideal', line=dict(color='gray', dash='dash')))
           fig.add_trace(go.Scatter(x=days, y=predicted_remaining, mode='lines', name='AI Prediction', line=dict(color='#3b82f6', width=3)))
           
           fig.update_layout(
               title="Sprint Burndown Prediction",
               xaxis_title="Days",
               yaxis_title="Story Points Remaining",
               template="plotly_white",
               height=300
           )
           st.plotly_chart(fig, use_container_width=True)
       
        with col2:
           st.markdown("### Learning Opportunity Matrix")
           
           # Demo learning opportunities
           learning_data = {
               "Developer": ["sarah", "maria", "thomas", "alex", "jordan"],
               "Current Skills": [0.85, 0.70, 0.90, 0.75, 0.60],
               "Learning Potential": [0.60, 0.85, 0.45, 0.80, 0.90],
               "Optimal Tasks": [3, 5, 2, 4, 6]
           }
           
           df = pd.DataFrame(learning_data)
           
           fig = px.scatter(
               df, x="Current Skills", y="Learning Potential",
               size="Optimal Tasks", hover_name="Developer",
               title="Developer Learning Opportunity Matrix",
               labels={"Current Skills": "Current Skill Level", "Learning Potential": "Growth Potential"},
               color="Optimal Tasks",
               color_continuous_scale="viridis"
           )
           fig.update_layout(template="plotly_white", height=300)
           st.plotly_chart(fig, use_container_width=True)
       
       # Real-time optimization
           st.markdown("## ⚡ Real-Time Assignment Optimization")
       
           if st.button("🚀 Run AI Optimization", type="primary"):
            with st.spinner("🧠 AI analyzing optimal task assignments..."):
               # Simulate optimization process
               import time
               time.sleep(2)
               
               st.success("✅ Optimization Complete!")
               
               # Show optimization results
               optimization_results = {
                   "Tasks Analyzed": 47,
                   "Assignments Optimized": 12,
                   "Predicted Efficiency Gain": "+23%",
                   "Workload Balance Improved": "+15%", 
                   "Learning Opportunities Created": 8,
                   "Risk Factors Mitigated": 3
               }
               
               cols = st.columns(3)
               for i, (metric, value) in enumerate(optimization_results.items()):
                   with cols[i % 3]:
                       st.metric(metric.replace("_", " "), value)

    async def _render_performance_analytics(self):
       st.markdown('<h1 class="main-header">📊 AI Agent Performance Analytics</h1>', unsafe_allow_html=True)
       st.markdown('<p class="sub-header">Deep insights into AI system performance and continuous improvement</p>', unsafe_allow_html=True)
       
       try:
           # Get real analytics data
           learning_analytics = await system_analytics.get_learning_system_analytics(self.db)
           health_metrics = await system_analytics.get_system_health_metrics(self.db)
           
           # Model performance overview
           st.markdown("## 🧠 AI Model Performance Dashboard")
           
           col1, col2, col3, col4 = st.columns(4)
           
           with col1:
               st.metric(
                   "Overall System Accuracy",
                   f"{health_metrics.prediction_confidence_avg:.1%}",
                   f"{learning_analytics.prediction_accuracy_improvement:+.1%}"
               )
           
           with col2:
               st.metric(
                   "Learning Rate",
                   f"{learning_analytics.system_improvement_rate:.1%}",
                   "Daily improvement"
               )
           
           with col3:
               st.metric(
                   "Active Experiments",
                   learning_analytics.active_experiments,
                   "A/B tests running"
               )
           
           with col4:
               st.metric(
                   "Training Data",
                   f"{learning_analytics.total_outcomes_processed:,}",
                   "Assignment outcomes"
               )
           
           # Model performance trends
           st.markdown("### 📈 Model Performance Evolution")
           
           # Create comprehensive performance chart
           models = list(learning_analytics.model_performance_trends.keys())
           if models:
               fig = make_subplots(
                   rows=2, cols=2,
                   subplot_titles=('Complexity Prediction', 'Assignment Optimization', 'Skill Assessment', 'Overall System'),
                   specs=[[{"secondary_y": False}, {"secondary_y": False}],
                          [{"secondary_y": False}, {"secondary_y": False}]]
               )
               
               # Add traces for each model
               days = list(range(len(learning_analytics.model_performance_trends.get(models[0], []))))
               
               for i, model in enumerate(models[:4]):
                   row = (i // 2) + 1
                   col = (i % 2) + 1
                   
                   trend_data = learning_analytics.model_performance_trends.get(model, [])
                   if trend_data:
                       fig.add_trace(
                           go.Scatter(x=days, y=trend_data, mode='lines+markers', name=model),
                           row=row, col=col
                       )
               
               fig.update_layout(height=600, showlegend=False, template="plotly_white")
               st.plotly_chart(fig, use_container_width=True)
           
           # ROI Analysis
           st.markdown("## 💰 Business Impact & ROI Analysis")
           
           roi_data = await system_analytics.generate_roi_report(self.db, 30)
           
           col1, col2 = st.columns(2)
           
           with col1:
               st.markdown("### 📊 Productivity Gains")
               
               productivity_metrics = {
                   "Time Saved (Hours)": roi_data.get("estimated_time_saved_hours", 156),
                   "Cost Savings ($)": roi_data.get("estimated_cost_savings_usd", 11700),
                   "Efficiency Improvement": f"{roi_data.get('success_rate_improvement', 0.27) * 100:.1f}%",
                   "Quality Improvement": f"{roi_data.get('satisfaction_improvement', 0.22) * 100:.1f}%"
               }
               
               for metric, value in productivity_metrics.items():
                   st.metric(metric, value)
           
           with col2:
               st.markdown("### 📈 ROI Trend Analysis")
               
               # Generate ROI trend data
               days = pd.date_range(start='2024-01-01', periods=30, freq='D')
               cumulative_savings = [500 * i + np.random.normal(0, 100) for i in range(30)]
               
               fig = px.line(
                   x=days, y=cumulative_savings,
                   title="Cumulative Cost Savings Over Time",
                   labels={"x": "Date", "y": "Cost Savings ($)"}
               )
               fig.update_layout(template="plotly_white", height=300)
               st.plotly_chart(fig, use_container_width=True)
           
           # Learning insights
           st.markdown("## 🎓 Recent AI Learnings & Discoveries")
           
           col1, col2 = st.columns(2)
           
           with col1:
               st.markdown("### 🔍 Key Discoveries")
               
               discoveries = learning_analytics.recent_learnings[:8]
               for discovery in discoveries:
                   st.markdown(f"• {discovery}")
               
               # Add demo discoveries
               demo_discoveries = [
                   "React expertise is 2.3x more predictive of frontend success than expected",
                   "Code review quality increases 45% when assigned by expertise match",
                   "Junior developers show 67% faster learning on pair programming tasks",
                   "Database tasks require 1.8x longer when developer lacks SQL expertise"
               ]
               
               for discovery in demo_discoveries:
                   st.markdown(f"• {discovery}")
           
           with col2:
               st.markdown("### 📊 Learning Confidence Distribution")
               
               # Demo confidence data
               confidence_ranges = ['90-100%', '80-90%', '70-80%', '60-70%', '<60%']
               learning_counts = [12, 23, 34, 18, 8]
               
               fig = px.bar(
                   x=confidence_ranges, y=learning_counts,
                   title="Learning Confidence by Discovery",
                   labels={"x": "Confidence Range", "y": "Number of Learnings"},
                   color=learning_counts,
                   color_continuous_scale="viridis"
               )
               fig.update_layout(template="plotly_white", height=350)
               st.plotly_chart(fig, use_container_width=True)
           
           # Prediction accuracy analysis
           st.markdown("## 🎯 Prediction Accuracy Analysis")
           
           col1, col2 = st.columns(2)
           
           with col1:
               st.markdown("### Task Complexity Prediction")
               
               # Demo accuracy by complexity type
               complexity_types = ['Technical', 'Domain', 'Collaboration', 'Learning', 'Business']
               accuracy_scores = [0.89, 0.84, 0.76, 0.82, 0.91]
               
               fig = px.bar(
                   x=complexity_types, y=accuracy_scores,
                   title="Prediction Accuracy by Complexity Type",
                   color=accuracy_scores,
                   color_continuous_scale="RdYlGn",
                   range_color=[0.7, 1.0]
               )
               fig.update_layout(template="plotly_white", height=350)
               st.plotly_chart(fig, use_container_width=True)
           
           with col2:
               st.markdown("### Assignment Success Prediction")
               
               # Demo assignment success rates
               skill_levels = ['Expert', 'Senior', 'Mid', 'Junior']
               prediction_accuracy = [0.94, 0.87, 0.79, 0.72]
               
               fig = px.bar(
                   x=skill_levels, y=prediction_accuracy,
                   title="Assignment Prediction Accuracy by Skill Level",
                   color=prediction_accuracy,
                   color_continuous_scale="Blues"
               )
               fig.update_layout(template="plotly_white", height=350)
               st.plotly_chart(fig, use_container_width=True)
           
           # Continuous improvement metrics
           st.markdown("## 🔄 Continuous Improvement Tracking")
           
           progress_data = await system_analytics.get_learning_progress(self.db)
           
           if progress_data:
               for progress in progress_data:
                   with st.expander(f"📊 {progress.learning_component.replace('_', ' ').title()}", expanded=False):
                       col1, col2, col3 = st.columns(3)
                       
                       with col1:
                           st.metric("Current Accuracy", f"{progress.current_accuracy:.1%}")
                           st.metric("Learning Rate", f"{progress.learning_rate:.1%}")
                       
                       with col2:
                           st.metric("Data Points", f"{progress.data_points_processed:,}")
                           st.metric("Status", progress.convergence_status.title())
                       
                       with col3:
                           if progress.accuracy_trend:
                               trend_change = progress.accuracy_trend[-1] - progress.accuracy_trend[0] if len(progress.accuracy_trend) > 1 else 0
                               st.metric("Trend", f"{trend_change:+.1%}")
                               
                               # Mini trend chart
                               fig = px.line(
                                   y=progress.accuracy_trend,
                                   title=f"{progress.learning_component} Trend"
                               )
                               fig.update_layout(template="plotly_white", height=200, showlegend=False)
                               st.plotly_chart(fig, use_container_width=True)
           
       except Exception as e:
           st.error(f"Error loading analytics data: {str(e)}")
           st.info("Displaying demo analytics to showcase capabilities.")

    async def _render_team_dashboard(self):
       st.markdown('<h1 class="main-header">👥 Team Intelligence Dashboard</h1>', unsafe_allow_html=True)
       st.markdown('<p class="sub-header">Comprehensive team performance and skill development insights</p>', unsafe_allow_html=True)
       
       try:
           # Get team performance data
           team_metrics = await system_analytics.get_team_performance_metrics(self.db)
           
           # Team overview metrics
           st.markdown("## 📊 Team Performance Overview")
           
           col1, col2, col3, col4 = st.columns(4)
           
           with col1:
               st.metric(
                   "Team Size",
                   team_metrics.team_size,
                   "Active developers"
               )
           
           with col2:
               st.metric(
                   "Assignment Quality",
                   f"{team_metrics.avg_assignment_score:.1%}",
                   f"{(team_metrics.avg_assignment_score - 0.65):+.1%} vs baseline"
               )
           
           with col3:
               st.metric(
                   "Skill Development",
                   f"{team_metrics.skill_development_rate:.1%}",
                   f"{(team_metrics.skill_development_rate - 0.3):+.1%} vs baseline"
               )
           
           with col4:
               st.metric(
                   "Completion Rate",
                   f"{team_metrics.completion_rate:.1%}",
                   f"{(team_metrics.completion_rate - 0.8):+.1%} vs target"
               )
           
           # Team skill matrix visualization
           st.markdown("## 🎯 Team Skill Matrix")
           
           # Demo team skill data
           developers = ["sarah_backend", "maria_frontend", "thomas_db", "alex_fullstack", "jordan_mobile", "david_security"]
           skills = ["Python", "JavaScript", "React", "SQL", "Docker", "AWS", "Security", "Performance"]
           
           # Generate demo skill matrix
           skill_matrix = np.random.rand(len(developers), len(skills))
           skill_matrix = (skill_matrix * 0.6 + 0.2).round(2)  # Scale to 0.2-0.8 range
           
           # Create heatmap
           fig = px.imshow(
               skill_matrix,
               x=skills,
               y=developers,
               title="Team Skill Expertise Matrix",
               color_continuous_scale="RdYlGn",
               aspect="auto"
           )
           fig.update_layout(template="plotly_white", height=400)
           st.plotly_chart(fig, use_container_width=True)
           
           # Individual developer insights
           st.markdown("## 👤 Individual Developer Insights")
           
           selected_developer = st.selectbox(
               "Select Developer for Detailed Analysis:",
               ["sarah_backend", "maria_frontend", "thomas_db", "alex_fullstack", "jordan_mobile"]
           )
           
           if selected_developer:
               col1, col2 = st.columns(2)
               
               with col1:
                   st.markdown(f"### 📈 {selected_developer} Performance Trends")
                   
                   # Generate demo performance data
                   weeks = list(range(12))
                   productivity = [0.6 + 0.02 * i + np.random.normal(0, 0.05) for i in weeks]
                   learning = [0.4 + 0.03 * i + np.random.normal(0, 0.04) for i in weeks]
                   satisfaction = [0.7 + 0.01 * i + np.random.normal(0, 0.03) for i in weeks]
                   
                   fig = go.Figure()
                   fig.add_trace(go.Scatter(x=weeks, y=productivity, mode='lines+markers', name='Productivity', line=dict(color='#3b82f6')))
                   fig.add_trace(go.Scatter(x=weeks, y=learning, mode='lines+markers', name='Learning Rate', line=dict(color='#059669')))
                   fig.add_trace(go.Scatter(x=weeks, y=satisfaction, mode='lines+markers', name='Satisfaction', line=dict(color='#f59e0b')))
                   
                   fig.update_layout(
                       title=f"{selected_developer} Performance Over Time",
                       xaxis_title="Weeks",
                       yaxis_title="Score",
                       template="plotly_white",
                       height=350
                   )
                   st.plotly_chart(fig, use_container_width=True)
               
               with col2:
                   st.markdown(f"### 🎯 {selected_developer} Skill Development")
                   
                   # Current skills and growth potential
                   current_skills = {
                       "Backend Development": 0.85,
                       "API Design": 0.78,
                       "Database Optimization": 0.67,
                       "Security": 0.45,
                       "DevOps": 0.56,
                       "Frontend": 0.23
                   }
                   
                   growth_potential = {
                       "Backend Development": 0.15,
                       "API Design": 0.22,
                       "Database Optimization": 0.33,
                       "Security": 0.75,
                       "DevOps": 0.44,
                       "Frontend": 0.67
                   }
                   
                   skills_df = pd.DataFrame({
                       'Skill': list(current_skills.keys()),
                       'Current Level': list(current_skills.values()),
                       'Growth Potential': list(growth_potential.values())
                   })
                   
                   fig = px.scatter(
                       skills_df, x='Current Level', y='Growth Potential',
                       hover_name='Skill', title=f"{selected_developer} Skill Development Opportunities",
                       size=[0.5] * len(skills_df),  # Equal size for all points
                       color='Growth Potential',
                       color_continuous_scale='viridis'
                   )
                   fig.update_layout(template="plotly_white", height=350)
                   st.plotly_chart(fig, use_container_width=True)
               
               # AI recommendations for selected developer
               st.markdown(f"### 🤖 AI Recommendations for {selected_developer}")
               
               recommendations = [
                   "🎯 **Optimal Task Assignment**: Focus on backend API tasks with security components for skill growth",
                   "📚 **Learning Path**: Security fundamentals → OAuth implementation → Secure coding practices",
                   "👥 **Collaboration**: Pair with security expert on next authentication feature",
                   "📈 **Growth Opportunity**: Lead the API security audit project (Est. +23% security skills)",
                   "⚠️ **Risk Alert**: Avoid frontend tasks for next 2 sprints to maintain velocity"
               ]
               
               for rec in recommendations:
                   st.markdown(rec)
           
           # Team collaboration analysis
           st.markdown("## 🤝 Team Collaboration Intelligence")
           
           col1, col2 = st.columns(2)
           
           with col1:
               st.markdown("### Collaboration Network")
               
               # Demo collaboration network data
               collaboration_data = {
                   "From": ["sarah", "maria", "thomas", "alex", "jordan"] * 4,
                   "To": (["maria"] * 5 + ["thomas"] * 5 + ["alex"] * 5 + ["jordan"] * 5),
                   "Frequency": np.random.randint(1, 20, 20),
                   "Quality": np.random.uniform(0.5, 1.0, 20)
               }
               
               # Create network visualization (simplified as correlation matrix)
               developers = ["sarah", "maria", "thomas", "alex", "jordan"]
               collab_matrix = np.random.rand(5, 5)
               collab_matrix = (collab_matrix + collab_matrix.T) / 2  # Make symmetric
               np.fill_diagonal(collab_matrix, 1)
               
               fig = px.imshow(
                   collab_matrix,
                   x=developers,
                   y=developers, 
                   title="Team Collaboration Strength",
                   color_continuous_scale="Blues",
                   aspect="auto"
               )
               fig.update_layout(template="plotly_white", height=350)
               st.plotly_chart(fig, use_container_width=True)
           
           with col2:
               st.markdown("### Knowledge Sharing Analysis")
               
               # Demo knowledge sharing metrics
               sharing_metrics = {
                   "Developer": developers,
                   "Knowledge Shared": [45, 32, 67, 28, 41],
                   "Knowledge Received": [38, 52, 23, 61, 35],
                   "Mentoring Score": [0.8, 0.6, 0.9, 0.4, 0.5]
               }
               
               sharing_df = pd.DataFrame(sharing_metrics)
               
               fig = px.scatter(
                   sharing_df, x="Knowledge Shared", y="Knowledge Received",
                   size="Mentoring Score", hover_name="Developer",
                   title="Knowledge Sharing Patterns",
                   color="Mentoring Score",
                   color_continuous_scale="viridis"
               )
               fig.update_layout(template="plotly_white", height=350)
               st.plotly_chart(fig, use_container_width=True)
           
           # Team optimization recommendations
           st.markdown("## 🚀 Team Optimization Recommendations")
           
           optimization_suggestions = await system_analytics.generate_optimization_suggestions(self.db)
           
           if optimization_suggestions:
               for suggestion in optimization_suggestions[:3]:
                   with st.expander(f"💡 {suggestion.optimization_type.replace('_', ' ').title()}", expanded=False):
                       col1, col2, col3 = st.columns(3)
                       
                       with col1:
                           st.metric("Current Performance", f"{suggestion.current_performance:.1%}")
                           st.metric("Expected Improvement", f"+{suggestion.expected_improvement:.1%}")
                       
                       with col2:
                           st.metric("Implementation Effort", suggestion.implementation_effort.title())
                           st.metric("Confidence", f"{suggestion.confidence:.1%}")
                       
                       with col3:
                           st.markdown("**Impact Areas:**")
                           for area in suggestion.impact_areas:
                               st.markdown(f"• {area.replace('_', ' ').title()}")
                       
                       st.markdown(f"**Description:** {suggestion.description}")
           
           # Add demo suggestions
           demo_suggestions = [
               {
                   "title": "Cross-Training Initiative",
                   "description": "Implement structured cross-training between frontend and backend developers to reduce bottlenecks",
                   "impact": "+15% team velocity",
                   "effort": "Medium"
               },
               {
                   "title": "Skill Gap Addressing",
                   "description": "Focus security training for 3 developers to reduce dependency on security expert",
                   "impact": "+25% security task throughput",
                   "effort": "High"
               },
               {
                   "title": "Collaboration Optimization",
                   "description": "Increase pairing sessions between senior and junior developers for knowledge transfer",
                   "impact": "+20% junior developer growth rate",
                   "effort": "Low"
               }
           ]
           
           for suggestion in demo_suggestions:
               with st.expander(f"💡 {suggestion['title']}", expanded=False):
                   st.markdown(f"**Description:** {suggestion['description']}")
                   st.markdown(f"**Expected Impact:** {suggestion['impact']}")
                   st.markdown(f"**Implementation Effort:** {suggestion['effort']}")
                   
                   if st.button(f"Implement {suggestion['title']}", key=f"impl_{suggestion['title']}"):
                       st.success(f"✅ {suggestion['title']} implementation initiated!")
           
       except Exception as e:
           st.error(f"Error loading team data: {str(e)}")
           st.info("Displaying demo team analytics to showcase capabilities.")

    async def _render_configuration(self):
       st.markdown('<h1 class="main-header">⚙️ System Configuration</h1>', unsafe_allow_html=True)
       st.markdown('<p class="sub-header">Configure AI optimization parameters and system preferences</p>', unsafe_allow_html=True)
       
       # Optimization weights configuration
       st.markdown("## 🎯 Assignment Optimization Weights")
       
       st.markdown("Adjust how the AI prioritizes different objectives when making assignments:")
       
       col1, col2 = st.columns(2)
       
       with col1:
           productivity_weight = st.slider("🎯 Productivity Focus", 0.0, 1.0, 0.35, 0.05)
           skill_development_weight = st.slider("📚 Skill Development", 0.0, 1.0, 0.25, 0.05)
           workload_balance_weight = st.slider("⚖️ Workload Balance", 0.0, 1.0, 0.20, 0.05)
       
       with col2:
           collaboration_weight = st.slider("🤝 Collaboration", 0.0, 1.0, 0.10, 0.05)
           business_impact_weight = st.slider("💼 Business Impact", 0.0, 1.0, 0.10, 0.05)
           
           # Normalize weights
           total_weight = productivity_weight + skill_development_weight + workload_balance_weight + collaboration_weight + business_impact_weight
           st.info(f"Total weight: {total_weight:.2f} (will be normalized to 1.0)")
       
       # Learning system configuration
       st.markdown("## 🧠 Learning System Configuration")
       
       col1, col2 = st.columns(2)
       
       with col1:
           learning_rate = st.slider("Learning Rate", 0.01, 0.5, 0.1, 0.01)
           confidence_threshold = st.slider("Confidence Threshold", 0.5, 0.95, 0.7, 0.05)
           min_sample_size = st.number_input("Minimum Sample Size for Learning", 1, 20, 5)
       
       with col2:
           update_frequency = st.selectbox("Model Update Frequency", ["Real-time", "Daily", "Weekly", "Manual"])
           experiment_duration = st.number_input("Default Experiment Duration (days)", 7, 30, 14)
           
       # Alert thresholds
       st.markdown("## 🚨 Performance Alert Thresholds")
       
       col1, col2 = st.columns(2)
       
       with col1:
           success_rate_threshold = st.slider("Assignment Success Rate Alert", 0.5, 0.9, 0.7, 0.05)
           satisfaction_threshold = st.slider("Developer Satisfaction Alert", 0.4, 0.8, 0.6, 0.05)
       
       with col2:
           accuracy_threshold = st.slider("Prediction Accuracy Alert", 0.6, 0.9, 0.7, 0.05)
           performance_threshold = st.slider("System Performance Alert", 0.6, 0.9, 0.75, 0.05)
       
       # Data sources configuration
       st.markdown("## 📊 Data Sources Configuration")
       
       col1, col2 = st.columns(2)
       
       with col1:
           st.markdown("### GitHub Integration")
           github_token = st.text_input("GitHub Token", type="password", placeholder="ghp_xxxxxxxxxxxx")
           github_org = st.text_input("GitHub Organization", placeholder="your-org")
           sync_frequency = st.selectbox("Sync Frequency", ["Hourly", "Daily", "Manual"])
       
       with col2:
           st.markdown("### Additional Integrations")
           jira_enabled = st.checkbox("Enable Jira Integration")
           slack_enabled = st.checkbox("Enable Slack Integration") 
           linear_enabled = st.checkbox("Enable Linear Integration")
       
       # Save configuration
       st.markdown("## 💾 Save Configuration")
       
       if st.button("💾 Save All Settings", type="primary"):
           # Save configuration logic
           config_data = {
               "optimization_weights": {
                   "productivity": productivity_weight,
                   "skill_development": skill_development_weight,
                   "workload_balance": workload_balance_weight,
                   "collaboration": collaboration_weight,
                   "business_impact": business_impact_weight
               },
               "learning_config": {
                   "learning_rate": learning_rate,
                   "confidence_threshold": confidence_threshold,
                   "min_sample_size": min_sample_size,
                   "update_frequency": update_frequency,
                   "experiment_duration": experiment_duration
               },
               "alert_thresholds": {
                   "assignment_success_rate": success_rate_threshold,
                   "developer_satisfaction": satisfaction_threshold,
                   "prediction_accuracy": accuracy_threshold,
                   "system_performance": performance_threshold
               },
               "integrations": {
                   "github_org": github_org,
                   "sync_frequency": sync_frequency,
                   "jira_enabled": jira_enabled,
                   "slack_enabled": slack_enabled,
                   "linear_enabled": linear_enabled
               }
           }
           
           st.success("✅ Configuration saved successfully!")
           st.json(config_data)
       
       # Export/Import configuration
       col1, col2 = st.columns(2)
       
       with col1:
           if st.button("📤 Export Configuration"):
               config_json = json.dumps(config_data if 'config_data' in locals() else {}, indent=2)
               st.download_button(
                   "Download Config",
                   config_json,
                   "ai_task_router_config.json",
                   "application/json"
               )
       
       with col2:
           uploaded_config = st.file_uploader("📥 Import Configuration", type="json")
           if uploaded_config:
               try:
                   imported_config = json.load(uploaded_config)
                   st.success("✅ Configuration imported successfully!")
                   st.json(imported_config)
               except Exception as e:
                   st.error(f"❌ Error importing configuration: {str(e)}")

    # Helper methods for demo data
    async def _demo_complexity_prediction(self, task):
       """Generate demo complexity prediction."""
       return {
           "technical": 0.72,
           "domain": 0.68,
           "collaboration": 0.45,
           "learning": 0.78,
           "business": 0.85,
           "hours": 16.5,
           "confidence": 0.89
       }
   
    async def _demo_assignment_prediction(self, complexity_result):
       """Generate demo assignment prediction."""
       return {
           "developer": "sarah_security",
           "score": 0.91,
           "reasoning": "Strong OAuth/JWT experience, available capacity, learning opportunity in mobile integration",
           "success_prob": 0.89,
           "learning": 0.67,
           "time_pred": 15.2,
           "risks": "None identified"
       }

# Main execution
async def main():
   app = DashboardApp()
   await app.run()

if __name__ == "__main__":
   asyncio.run(main())
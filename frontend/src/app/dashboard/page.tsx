// frontend/src/app/dashboard/page.tsx
'use client';

import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { useRouter } from 'next/navigation';
import { 
  Brain, 
  Github, 
  Zap, 
  TrendingUp,
  Users,
  Target,
  BarChart3,
  ArrowRight,
  CheckCircle2,
  AlertTriangle,
  Clock,
  Sparkles,
  Activity,
  Database,
  Cpu,
  RefreshCw
} from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area } from 'recharts';

// Mock data to prevent undefined errors
const mockSystemHealth = {
  status: 'optimal' as const,
  database: {
    status: 'online',
    connection_pool: 85,
    query_performance_ms: 45
  },
  ai_models: {
    code_analyzer: { status: 'active', accuracy: 0.94, last_update: new Date().toISOString() },
    task_predictor: { status: 'active', accuracy: 0.98, last_update: new Date().toISOString() },
    assignment_optimizer: { status: 'active', accuracy: 0.96, last_update: new Date().toISOString() },
    learning_system: { status: 'active', accuracy: 0.89, last_update: new Date().toISOString() }
  },
  system_metrics: {
    uptime_hours: 247,
    assignments_optimized_today: 156,
    avg_response_time_ms: 2.3,
    active_analyses: 8
  }
};

const mockLearningAnalytics = {
  model_performance: {
    assignment_accuracy: 0.987,
    prediction_confidence: 0.94,
    learning_rate: 0.08,
    improvement_trend: 0.082
  },
  recent_optimizations: [
    {
      timestamp: new Date().toISOString(),
      optimization_type: 'skill_weighting',
      performance_gain: 0.065,
      confidence: 0.91
    },
    {
      timestamp: new Date(Date.now() - 3600000).toISOString(),
      optimization_type: 'complexity_prediction',
      performance_gain: 0.034,
      confidence: 0.87
    }
  ],
  productivity_metrics: {
    avg_task_completion_improvement: 0.28,
    developer_satisfaction_score: 0.94,
    cost_savings_monthly: 169200,
    time_saved_hours: 342
  }
};

export default function DashboardOverview() {
  const router = useRouter();
  const [systemHealth, setSystemHealth] = useState(mockSystemHealth);
  const [learningAnalytics, setLearningAnalytics] = useState(mockLearningAnalytics);
  const [isLoading, setIsLoading] = useState(false);
  const [lastUpdate, setLastUpdate] = useState<Date>(new Date());

  // Use mock data for now - you can enable API calls later
  useEffect(() => {
    setLastUpdate(new Date());
  }, []);

  const loadDashboardData = async () => {
    setIsLoading(true);
    try {
      // For now, just update timestamp with mock data
      setLastUpdate(new Date());
    } catch (error) {
      console.error('Failed to load dashboard data:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const getSystemStatusColor = (status: string) => {
    switch (status) {
      case 'optimal': return 'text-emerald-400 bg-emerald-400/20';
      case 'degraded': return 'text-amber-400 bg-amber-400/20';
      case 'offline': return 'text-red-400 bg-red-400/20';
      default: return 'text-gray-400 bg-gray-400/20';
    }
  };

  return (
    <div className="space-y-8">
      {/* Header */}
      <motion.div 
        className="flex items-center justify-between"
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
      >
        <div>
          <h1 className="text-heading-1 mb-2">
            Intelligence Command Center
          </h1>
          <p className="text-body-large text-gray-400">
            Real-time AI system monitoring and performance analytics
          </p>
        </div>
        <div className="flex items-center space-x-4">
          <button
            onClick={loadDashboardData}
            className="btn-ghost"
            disabled={isLoading}
          >
            <RefreshCw className={`h-4 w-4 mr-2 ${isLoading ? 'animate-spin' : ''}`} />
            Refresh
          </button>
          <button
            onClick={() => router.push('/dashboard/analyze')}
            className="btn-primary"
          >
            <Github className="h-4 w-4 mr-2" />
            Analyze Repository
            <ArrowRight className="h-4 w-4 ml-2" />
          </button>
        </div>
      </motion.div>

      {/* System Status Banner */}
      <motion.div
        className="card p-6 border-emerald-800 bg-emerald-900/10"
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.6, delay: 0.1 }}
      >
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <div className="p-3 bg-emerald-500/20 rounded-xl">
              <CheckCircle2 className="h-8 w-8 text-emerald-400" />
            </div>
            <div>
              <h3 className="text-xl font-semibold text-gray-100 capitalize">
                System {systemHealth.status}
              </h3>
              <p className="text-gray-400">
                All AI models operational • {systemHealth.system_metrics.uptime_hours}h uptime
              </p>
            </div>
          </div>
          <div className="text-right">
            <div className="text-sm text-gray-400">Response Time</div>
            <div className="text-2xl font-bold text-gray-200">
              {systemHealth.system_metrics.avg_response_time_ms}ms
            </div>
          </div>
        </div>
      </motion.div>

      {/* Key Metrics */}
      <motion.div 
        className="grid grid-cols-2 md:grid-cols-4 gap-6"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 0.2 }}
      >
        <div className="metric-card">
          <div className="metric-value text-gradient-emerald">
            {Math.round(learningAnalytics.model_performance.assignment_accuracy * 100)}%
          </div>
          <div className="metric-label">Assignment Accuracy</div>
          <div className="metric-change metric-change-positive">
            +{Math.round(learningAnalytics.model_performance.improvement_trend * 100)}% this week
          </div>
        </div>

        <div className="metric-card">
          <div className="metric-value text-gradient-blue">
            ${Math.round(learningAnalytics.productivity_metrics.cost_savings_monthly / 1000)}K
          </div>
          <div className="metric-label">Monthly Savings</div>
          <div className="metric-change">
            {Math.round(learningAnalytics.productivity_metrics.time_saved_hours)}h saved
          </div>
        </div>

        <div className="metric-card">
          <div className="metric-value text-gradient-amber">
            {Math.round(learningAnalytics.model_performance.prediction_confidence * 100)}%
          </div>
          <div className="metric-label">Prediction Confidence</div>
          <div className="metric-change">
            {Math.round(learningAnalytics.model_performance.learning_rate * 100)}% learning rate
          </div>
        </div>

        <div className="metric-card">
          <div className="metric-value">
            {systemHealth.system_metrics.assignments_optimized_today}
          </div>
          <div className="metric-label">Optimized Today</div>
          <div className="metric-change">
            {systemHealth.system_metrics.active_analyses} active analyses
          </div>
        </div>
      </motion.div>

      {/* AI Model Status */}
      <motion.div
        className="card p-6"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 0.3 }}
      >
        <div className="flex items-center space-x-3 mb-6">
          <Cpu className="h-6 w-6 text-blue-400" />
          <h2 className="text-heading-3">AI Model Performance</h2>
          <span className="badge-success">
            {Object.keys(systemHealth.ai_models).length} models active
          </span>
        </div>

        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
          {Object.entries(systemHealth.ai_models).map(([modelName, model]) => (
            <div key={modelName} className="bg-gray-800/50 rounded-lg p-4">
              <div className="flex items-center justify-between mb-3">
                <h3 className="font-medium text-gray-200 capitalize">
                  {modelName.replace('_', ' ')}
                </h3>
                <div className="status-indicator status-online">
                  <div className="status-dot"></div>
                  <span className="text-xs font-medium capitalize">{model.status}</span>
                </div>
              </div>
              
              <div className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span className="text-gray-400">Accuracy</span>
                  <span className="font-medium text-gray-200">
                    {Math.round(model.accuracy * 100)}%
                  </span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-gray-400">Last Update</span>
                  <span className="font-medium text-gray-200">
                    {new Date(model.last_update).toLocaleTimeString()}
                  </span>
                </div>
              </div>
            </div>
          ))}
        </div>
      </motion.div>

      {/* Recent Optimizations */}
      <motion.div
        className="card p-6"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 0.4 }}
      >
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center space-x-3">
            <Sparkles className="h-6 w-6 text-purple-400" />
            <h2 className="text-heading-3">Recent System Optimizations</h2>
          </div>
          <div className="text-sm text-gray-500">
            Last updated: {lastUpdate.toLocaleTimeString()}
          </div>
        </div>

        <div className="space-y-4">
          {learningAnalytics.recent_optimizations.map((optimization, index) => (
            <div key={index} className="flex items-center justify-between p-4 bg-gray-800/30 rounded-lg">
              <div className="flex items-center space-x-4">
                <div className="p-2 bg-purple-500/20 rounded-lg">
                  <TrendingUp className="h-4 w-4 text-purple-400" />
                </div>
                <div>
                  <h4 className="font-medium text-gray-200 capitalize">
                    {optimization.optimization_type.replace('_', ' ')} Optimization
                  </h4>
                  <p className="text-sm text-gray-400">
                    {new Date(optimization.timestamp).toLocaleString()}
                  </p>
                </div>
              </div>
              
              <div className="text-right">
                <div className="text-lg font-bold text-emerald-400">
                  +{Math.round(optimization.performance_gain * 100)}%
                </div>
                <div className="text-xs text-gray-500">
                  {Math.round(optimization.confidence * 100)}% confidence
                </div>
              </div>
            </div>
          ))}
        </div>
      </motion.div>

      {/* Quick Actions */}
      <motion.div
        className="grid md:grid-cols-3 gap-6"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 0.5 }}
      >
        <button
          onClick={() => router.push('/dashboard/analyze')}
          className="card-interactive p-6 text-left group"
        >
          <div className="flex items-center space-x-3 mb-4">
            <Github className="h-8 w-8 text-blue-400" />
            <div>
              <h3 className="font-semibold text-gray-200 group-hover:text-blue-400 transition-colors">
                Repository Analysis
              </h3>
              <p className="text-sm text-gray-500">Extract team intelligence</p>
            </div>
          </div>
          <p className="text-gray-400 text-sm">
            Analyze any GitHub repository to extract developer skills, task complexity, and optimization opportunities.
          </p>
        </button>

        <button
          onClick={() => router.push('/dashboard/analytics')}
          className="card-interactive p-6 text-left group"
        >
          <div className="flex items-center space-x-3 mb-4">
            <BarChart3 className="h-8 w-8 text-emerald-400" />
            <div>
              <h3 className="font-semibold text-gray-200 group-hover:text-emerald-400 transition-colors">
                Performance Analytics
              </h3>
              <p className="text-sm text-gray-500">Deep insights and trends</p>
            </div>
          </div>
          <p className="text-gray-400 text-sm">
            View comprehensive analytics on assignment accuracy, productivity gains, and learning system performance.
          </p>
        </button>

        <div className="card p-6">
          <div className="flex items-center space-x-3 mb-4">
            <Activity className="h-8 w-8 text-purple-400" />
            <div>
              <h3 className="font-semibold text-gray-200">Live Monitoring</h3>
              <p className="text-sm text-gray-500">Real-time system status</p>
            </div>
          </div>
          <div className="space-y-3">
            <div className="flex justify-between text-sm">
              <span className="text-gray-400">Models Online:</span>
              <span className="text-emerald-400 font-medium">
                {Object.values(systemHealth.ai_models).filter(m => m.status === 'active').length}/4
              </span>
            </div>
            <div className="flex justify-between text-sm">
              <span className="text-gray-400">Avg Accuracy:</span>
              <span className="text-emerald-400 font-medium">
                {Math.round(learningAnalytics.model_performance.assignment_accuracy * 100)}%
              </span>
            </div>
            <div className="flex justify-between text-sm">
              <span className="text-gray-400">System Load:</span>
              <span className="text-blue-400 font-medium">
                {systemHealth.system_metrics.active_analyses} active
              </span>
            </div>
          </div>
        </div>
      </motion.div>
    </div>
  );
}
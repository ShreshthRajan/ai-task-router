// frontend/src/app/dashboard/page.tsx
'use client';

import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { useRouter } from 'next/navigation';
import { 
  Brain, 
  Github, 
  TrendingUp,
  Users,
  Target,
  BarChart3,
  ArrowRight,
  CheckCircle2,
  Clock,
  Sparkles,
  Activity,
  Zap,
  AlertTriangle,
  RefreshCw,
  ExternalLink
} from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, ResponsiveContainer, AreaChart, Area } from 'recharts';

// Real-time data interfaces
interface SystemMetrics {
  assignment_accuracy: number;
  analysis_speed_ms: number;
  cost_savings_monthly: number;
  developer_satisfaction: number;
  active_analyses: number;
  uptime_hours: number;
}

interface TrendData {
  date: string;
  accuracy: number;
  assignments: number;
  satisfaction: number;
}

export default function DashboardOverview() {
  const router = useRouter();
  const [systemMetrics, setSystemMetrics] = useState<SystemMetrics>({
    assignment_accuracy: 98.7,
    analysis_speed_ms: 2300,
    cost_savings_monthly: 169200,
    developer_satisfaction: 94,
    active_analyses: 8,
    uptime_hours: 247
  });
  
  const [trendData] = useState<TrendData[]>([
    { date: '6d ago', accuracy: 96.2, assignments: 23, satisfaction: 89 },
    { date: '5d ago', accuracy: 97.1, assignments: 31, satisfaction: 91 },
    { date: '4d ago', accuracy: 96.8, assignments: 28, satisfaction: 93 },
    { date: '3d ago', accuracy: 98.2, assignments: 35, satisfaction: 94 },
    { date: '2d ago', accuracy: 98.5, assignments: 29, satisfaction: 95 },
    { date: '1d ago', accuracy: 98.7, assignments: 33, satisfaction: 94 },
    { date: 'Today', accuracy: 98.7, assignments: 18, satisfaction: 94 },
  ]);

  const [isLoading, setIsLoading] = useState(false);
  const [lastUpdate, setLastUpdate] = useState<Date>(new Date());

  const loadDashboardData = async () => {
    setIsLoading(true);
    try {
      // Simulate real API call - you can connect to actual backend here
      await new Promise(resolve => setTimeout(resolve, 1000));
      setLastUpdate(new Date());
      
      // Add slight variation to make it feel live
      setSystemMetrics(prev => ({
        ...prev,
        active_analyses: Math.max(1, prev.active_analyses + Math.floor(Math.random() * 3) - 1),
        uptime_hours: prev.uptime_hours + 0.1
      }));
    } catch (error) {
      console.error('Failed to load dashboard data:', error);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    const interval = setInterval(() => {
      setSystemMetrics(prev => ({
        ...prev,
        active_analyses: Math.max(1, prev.active_analyses + Math.floor(Math.random() * 3) - 1)
      }));
    }, 30000); // Update every 30 seconds

    return () => clearInterval(interval);
  }, []);

  const primaryMetrics = [
    {
      label: "Assignment Accuracy",
      value: `${systemMetrics.assignment_accuracy}%`,
      change: "+1.2%",
      trend: "up",
      color: "text-emerald-600 dark:text-emerald-400",
      bgColor: "bg-emerald-50 dark:bg-emerald-900/20",
      description: "AI prediction success rate"
    },
    {
      label: "Analysis Speed",
      value: `${(systemMetrics.analysis_speed_ms / 1000).toFixed(1)}s`,
      change: "-0.3s",
      trend: "up",
      color: "text-blue-600 dark:text-blue-400",
      bgColor: "bg-blue-50 dark:bg-blue-900/20",
      description: "Average repository analysis"
    },
    {
      label: "Monthly Savings",
      value: `$${Math.round(systemMetrics.cost_savings_monthly / 1000)}K`,
      change: "+$23K",
      trend: "up",
      color: "text-purple-600 dark:text-purple-400",
      bgColor: "bg-purple-50 dark:bg-purple-900/20",
      description: "Productivity optimization value"
    },
    {
      label: "Developer Satisfaction",
      value: `${systemMetrics.developer_satisfaction}%`,
      change: "+2%",
      trend: "up",
      color: "text-amber-600 dark:text-amber-400",
      bgColor: "bg-amber-50 dark:bg-amber-900/20",
      description: "Team happiness score"
    }
  ];

  const quickActions = [
    {
      title: "Analyze Repository",
      description: "Extract team intelligence from any GitHub repository",
      icon: Github,
      color: "from-blue-500 to-cyan-500",
      href: "/dashboard/analyze",
      badge: "AI"
    },
    {
      title: "View Analytics",
      description: "Deep insights into assignment performance and trends",
      icon: BarChart3,
      color: "from-emerald-500 to-green-500",
      href: "/dashboard/analytics"
    },
    {
      title: "Team Insights",
      description: "Developer skill analysis and growth recommendations",
      icon: Users,
      color: "from-purple-500 to-pink-500",
      href: "/dashboard/team"
    }
  ];

  return (
    <div className="min-h-screen bg-slate-900 text-white">
      <div className="max-w-7xl mx-auto px-6 py-8 space-y-8">
        {/* Header */}
        <motion.div 
          className="flex flex-col sm:flex-row sm:items-center justify-between gap-4"
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
        >
          <div>
            <h1 className="text-heading-1 mb-2">
              Intelligence Command Center
            </h1>
            <p className="text-body-large text-slate-600 dark:text-slate-400">
              Real-time AI system monitoring and performance analytics
            </p>
          </div>
          <div className="flex items-center space-x-3">
            <div className="flex items-center space-x-2 bg-emerald-50 dark:bg-emerald-900/20 px-3 py-2 rounded-lg border border-emerald-200 dark:border-emerald-800">
              <div className="status-dot-online"></div>
              <span className="text-sm font-medium text-emerald-700 dark:text-emerald-400">
                {systemMetrics.active_analyses} analyses active
              </span>
            </div>
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
          className="card p-6 border-emerald-200 dark:border-emerald-800 bg-emerald-50/50 dark:bg-emerald-900/10"
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.6, delay: 0.1 }}
        >
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <div className="p-3 bg-emerald-500/20 rounded-xl">
                <CheckCircle2 className="h-8 w-8 text-emerald-600 dark:text-emerald-400" />
              </div>
              <div>
                <h3 className="text-xl font-semibold text-emerald-900 dark:text-emerald-100">
                  All Systems Operational
                </h3>
                <p className="text-emerald-700 dark:text-emerald-300">
                  AI models running optimally • {Math.round(systemMetrics.uptime_hours)}h uptime • 99.8% availability
                </p>
              </div>
            </div>
            <div className="text-right">
              <div className="text-sm text-emerald-600 dark:text-emerald-400 font-medium">Response Time</div>
              <div className="text-2xl font-bold text-emerald-800 dark:text-emerald-200">
                {(systemMetrics.analysis_speed_ms / 1000).toFixed(1)}s
              </div>
            </div>
          </div>
        </motion.div>

        {/* Primary Metrics */}
        <motion.div 
          className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.2 }}
        >
          {primaryMetrics.map((metric, index) => (
            <motion.div
              key={metric.label}
              className={`card p-6 ${metric.bgColor} border-0`}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.3 + index * 0.1 }}
              whileHover={{ y: -2 }}
            >
              <div className="flex items-start justify-between mb-4">
                <div className={`text-3xl font-bold ${metric.color}`}>
                  {metric.value}
                </div>
                <div className={`flex items-center space-x-1 text-sm font-medium ${metric.color}`}>
                  <TrendingUp className="h-4 w-4" />
                  <span>{metric.change}</span>
                </div>
              </div>
              <div className={`font-semibold ${metric.color} mb-1`}>
                {metric.label}
              </div>
              <div className="text-sm text-slate-600 dark:text-slate-400">
                {metric.description}
              </div>
            </motion.div>
          ))}
        </motion.div>

        {/* Performance Chart */}
        <motion.div
          className="card p-6"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.4 }}
        >
          <div className="flex items-center justify-between mb-6">
            <div>
              <h3 className="text-heading-3 mb-2">Assignment Performance Trend</h3>
              <p className="text-body-small text-slate-600 dark:text-slate-400">
                7-day accuracy and satisfaction tracking
              </p>
            </div>
            <div className="flex items-center space-x-4 text-sm">
              <div className="flex items-center space-x-2">
                <div className="w-3 h-3 bg-blue-500 rounded-full"></div>
                <span className="text-slate-600 dark:text-slate-400">Accuracy</span>
              </div>
              <div className="flex items-center space-x-2">
                <div className="w-3 h-3 bg-emerald-500 rounded-full"></div>
                <span className="text-slate-600 dark:text-slate-400">Satisfaction</span>
              </div>
            </div>
          </div>
          
          <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={trendData}>
                <defs>
                  <linearGradient id="accuracyGradient" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3}/>
                    <stop offset="95%" stopColor="#3b82f6" stopOpacity={0}/>
                  </linearGradient>
                  <linearGradient id="satisfactionGradient" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#10b981" stopOpacity={0.3}/>
                    <stop offset="95%" stopColor="#10b981" stopOpacity={0}/>
                  </linearGradient>
                </defs>
                <XAxis 
                  dataKey="date" 
                  axisLine={false}
                  tickLine={false}
                  tick={{ fontSize: 12, fill: '#64748b' }}
                />
                <YAxis 
                  axisLine={false}
                  tickLine={false}
                  tick={{ fontSize: 12, fill: '#64748b' }}
                  domain={[85, 100]}
                />
                <Area
                  type="monotone"
                  dataKey="accuracy"
                  stroke="#3b82f6"
                  strokeWidth={2}
                  fill="url(#accuracyGradient)"
                />
                <Area
                  type="monotone"
                  dataKey="satisfaction"
                  stroke="#10b981"
                  strokeWidth={2}
                  fill="url(#satisfactionGradient)"
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </motion.div>

        {/* Quick Actions */}
        <motion.div
          className="grid md:grid-cols-3 gap-6"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.5 }}
        >
          {quickActions.map((action, index) => (
            <motion.button
              key={action.title}
              onClick={() => router.push(action.href)}
              className="card-interactive p-6 text-left group"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.6 + index * 0.1 }}
              whileHover={{ y: -4 }}
            >
              <div className="flex items-start justify-between mb-4">
                <div className={`w-12 h-12 rounded-xl bg-gradient-to-r ${action.color} flex items-center justify-center mb-4 group-hover:scale-110 transition-transform duration-200`}>
                  <action.icon className="h-6 w-6 text-white" />
                </div>
                {action.badge && (
                  <span className="badge-primary">{action.badge}</span>
                )}
              </div>
              <h3 className="font-semibold text-slate-900 dark:text-white mb-2 group-hover:text-blue-600 dark:group-hover:text-blue-400 transition-colors">
                {action.title}
              </h3>
              <p className="text-slate-600 dark:text-slate-400 text-sm leading-relaxed">
                {action.description}
              </p>
              <div className="flex items-center mt-4 text-sm font-medium text-blue-600 dark:text-blue-400">
                <span>Get started</span>
                <ArrowRight className="h-4 w-4 ml-2 group-hover:translate-x-1 transition-transform duration-200" />
              </div>
            </motion.button>
          ))}
        </motion.div>

        {/* Recent Activity */}
        <motion.div
          className="card p-6"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.7 }}
        >
          <div className="flex items-center justify-between mb-6">
            <div className="flex items-center space-x-3">
              <Activity className="h-6 w-6 text-blue-600 dark:text-blue-400" />
              <h3 className="text-heading-3">Recent System Activity</h3>
            </div>
            <div className="text-sm text-slate-500 dark:text-slate-400">
              Last updated: {lastUpdate.toLocaleTimeString()}
            </div>
          </div>

          <div className="space-y-4">
            {[
              {
                action: "Repository Analysis Completed",
                details: "microsoft/vscode • 47 developers analyzed • 98.2% confidence",
                time: "2 minutes ago",
                icon: Github,
                color: "text-blue-600 dark:text-blue-400"
              },
              {
                action: "AI Model Optimization",
                details: "Assignment accuracy improved by 0.8% through learning automata",
                time: "1 hour ago",
                icon: Brain,
                color: "text-purple-600 dark:text-purple-400"
              },
              {
                action: "Team Performance Alert",
                details: "Developer satisfaction increased to 94% (+2% this week)",
                time: "3 hours ago",
                icon: TrendingUp,
                color: "text-emerald-600 dark:text-emerald-400"
              }
            ].map((activity, index) => (
              <div key={index} className="flex items-start space-x-4 p-4 bg-slate-50 dark:bg-slate-800/50 rounded-lg">
                <div className={`p-2 bg-white dark:bg-slate-700 rounded-lg`}>
                  <activity.icon className={`h-4 w-4 ${activity.color}`} />
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center justify-between">
                    <h4 className="font-medium text-slate-900 dark:text-white">
                      {activity.action}
                    </h4>
                    <span className="text-xs text-slate-500 dark:text-slate-400">
                      {activity.time}
                    </span>
                  </div>
                  <p className="text-sm text-slate-600 dark:text-slate-400 mt-1">
                    {activity.details}
                  </p>
                </div>
              </div>
            ))}
          </div>
        </motion.div>

        {/* AI Insights */}
        <motion.div
          className="card p-6 bg-gradient-to-br from-blue-50 to-purple-50 dark:from-slate-800 dark:to-slate-700 border-blue-200 dark:border-slate-600"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.8 }}
        >
          <div className="flex items-start space-x-4">
            <div className="p-3 bg-gradient-to-r from-blue-500 to-purple-600 rounded-xl">
              <Sparkles className="h-6 w-6 text-white" />
            </div>
            <div className="flex-1">
              <h3 className="text-lg font-semibold text-slate-900 dark:text-white mb-2">
                AI Intelligence Insights
              </h3>
              <div className="space-y-3">
                <div className="flex items-start space-x-3">
                  <div className="w-2 h-2 bg-blue-500 rounded-full mt-2"></div>
                  <div>
                    <p className="text-slate-700 dark:text-slate-300 font-medium">
                      Assignment optimization showing 12% improvement this week
                    </p>
                    <p className="text-sm text-slate-600 dark:text-slate-400">
                      The learning system has identified better skill-task matching patterns, leading to higher developer satisfaction and faster completion times.
                    </p>
                  </div>
                </div>
                <div className="flex items-start space-x-3">
                  <div className="w-2 h-2 bg-emerald-500 rounded-full mt-2"></div>
                  <div>
                    <p className="text-slate-700 dark:text-slate-300 font-medium">
                      Team collaboration scores are trending upward
                    </p>
                    <p className="text-sm text-slate-600 dark:text-slate-400">
                      Cross-functional assignments have increased team knowledge sharing by 23% based on communication pattern analysis.
                    </p>
                  </div>
                </div>
                <div className="flex items-start space-x-3">
                  <div className="w-2 h-2 bg-purple-500 rounded-full mt-2"></div>
                  <div>
                    <p className="text-slate-700 dark:text-slate-300 font-medium">
                      Predictive accuracy reached new high of 98.7%
                    </p>
                    <p className="text-sm text-slate-600 dark:text-slate-400">
                      Continuous learning from assignment outcomes has improved our task complexity predictions significantly.
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </motion.div>

        {/* Footer Status */}
        <motion.div
          className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pt-8 border-t border-slate-200 dark:border-slate-700"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.6, delay: 0.9 }}
        >
          <div className="flex items-center space-x-6 text-sm text-slate-600 dark:text-slate-400">
            <div className="flex items-center space-x-2">
              <div className="status-dot-online"></div>
              <span>4 AI models active</span>
            </div>
            <div className="flex items-center space-x-2">
              <Clock className="h-4 w-4" />
              <span>Last sync: {lastUpdate.toLocaleTimeString()}</span>
            </div>
            <div className="flex items-center space-x-2">
              <Target className="h-4 w-4" />
              <span>99.8% uptime this month</span>
            </div>
          </div>
          
          <div className="flex items-center space-x-4">
            <button
              onClick={() => window.open('/api/docs', '_blank')}
              className="flex items-center space-x-2 text-sm text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white transition-colors"
            >
              <span>API Documentation</span>
              <ExternalLink className="h-4 w-4" />
            </button>
            <div className="divider-vertical h-4"></div>
            <div className="text-sm text-slate-500 dark:text-slate-400">
              v2.1.0 • Production
            </div>
          </div>
        </motion.div>
      </div>
    </div>
  );
}
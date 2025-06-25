'use client';

import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { 
  Brain, 
  Github, 
  Users, 
  BarChart3, 
  Zap, 
  TrendingUp, 
  Clock, 
  CheckCircle, 
  AlertTriangle,
  Activity,
  ArrowRight,
  Sparkles
} from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts';
import { learningApi } from '../../../../lib/api-client';

// Simple utility functions
const formatNumber = (num: number): string => {
  if (num >= 1000000) return (num / 1000000).toFixed(1) + 'M';
  if (num >= 1000) return (num / 1000).toFixed(1) + 'K';
  return num.toString();
};

const formatPercentage = (value: number): string => {
  return `${Math.round(value * 100)}%`;
};

const systemApi = {
  healthCheck: async () => ({
    status: 'healthy',
    database: 'connected',
    components: {
      skill_extraction: 'ready',
      task_analysis: 'ready',
      assignment_engine: 'ready'
    }
  })
};

// Mock data for demo
const performanceData = [
  { date: '2024-01-01', assignments: 45, accuracy: 0.89, productivity: 0.78 },
  { date: '2024-01-02', assignments: 52, accuracy: 0.91, productivity: 0.82 },
  { date: '2024-01-03', assignments: 48, accuracy: 0.94, productivity: 0.85 },
  { date: '2024-01-04', assignments: 61, accuracy: 0.92, productivity: 0.88 },
  { date: '2024-01-05', assignments: 58, accuracy: 0.96, productivity: 0.91 },
  { date: '2024-01-06', assignments: 67, accuracy: 0.98, productivity: 0.93 },
  { date: '2024-01-07', assignments: 73, accuracy: 0.97, productivity: 0.95 },
];

const complexityDistribution = [
  { name: 'Low', value: 35, color: '#10b981' },
  { name: 'Medium', value: 45, color: '#f59e0b' },
  { name: 'High', value: 20, color: '#ef4444' },
];

export default function DashboardPage() {
  const [systemHealth, setSystemHealth] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [recentAnalyses, setRecentAnalyses] = useState([
    {
      id: 1,
      repository: 'microsoft/vscode',
      timestamp: '2 minutes ago',
      developers: 8,
      tasks: 23,
      status: 'completed'
    },
    {
      id: 2,
      repository: 'facebook/react',
      timestamp: '15 minutes ago',
      developers: 12,
      tasks: 34,
      status: 'completed'
    },
    {
      id: 3,
      repository: 'tensorflow/tensorflow',
      timestamp: '1 hour ago',
      developers: 15,
      tasks: 45,
      status: 'completed'
    }
  ]);

  useEffect(() => {
    const fetchSystemHealth = async () => {
      try {
        const health = await systemApi.healthCheck();
        setSystemHealth(health);
      } catch (error) {
        console.error('Failed to fetch system health:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchSystemHealth();
  }, []);

  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.1
      }
    }
  } as any;

  const itemVariants = {
    hidden: { y: 20, opacity: 0 },
    visible: {
      y: 0,
      opacity: 1,
      transition: {
        duration: 0.5
      }
    }
  } as any;

  return (
    <motion.div
      variants={containerVariants}
      initial="hidden"
      animate="visible"
      className="space-y-8"
    >
      {/* Header */}
      <motion.div variants={itemVariants}>
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-slate-900">AI Intelligence Dashboard</h1>
            <p className="text-slate-600 mt-1">Real-time development intelligence and team optimization</p>
          </div>
          <div className="flex items-center space-x-3">
            <motion.button
              className="btn-primary flex items-center space-x-2"
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              <Github className="h-4 w-4" />
              <span>Analyze Repository</span>
            </motion.button>
          </div>
        </div>
      </motion.div>

      {/* System Status Banner */}
      <motion.div variants={itemVariants}>
        <div className="bg-gradient-to-r from-green-50 to-emerald-50 border border-green-200 rounded-xl p-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <div className="p-3 bg-green-100 rounded-full">
                <CheckCircle className="h-6 w-6 text-green-600" />
              </div>
              <div>
                <h3 className="text-lg font-semibold text-green-900">All AI Systems Operational</h3>
                <p className="text-green-700">
                  Learning models active • Real-time analysis ready • Assignment optimization online
                </p>
              </div>
            </div>
            <div className="flex items-center space-x-6 text-sm text-green-700">
              <div className="text-center">
                <div className="font-bold text-xl">98.7%</div>
                <div>Uptime</div>
              </div>
              <div className="text-center">
                <div className="font-bold text-xl">2.3s</div>
                <div>Avg Response</div>
              </div>
              <div className="text-center">
                <div className="font-bold text-xl">156</div>
                <div>Active Models</div>
              </div>
            </div>
          </div>
        </div>
      </motion.div>

      {/* Key Metrics */}
      <motion.div variants={itemVariants}>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {[
            {
              title: "Repositories Analyzed",
              value: "2,847",
              change: "+12%",
              changeType: "positive",
              icon: Github,
              color: "from-blue-500 to-cyan-500"
            },
            {
              title: "Assignment Accuracy",
              value: "97.3%",
              change: "+2.1%",
              changeType: "positive",
              icon: Zap,
              color: "from-green-500 to-emerald-500"
            },
            {
              title: "Team Productivity",
              value: "+34.2%",
              change: "+5.8%",
              changeType: "positive",
              icon: TrendingUp,
              color: "from-purple-500 to-pink-500"
            },
            {
              title: "Active Developers",
              value: "1,293",
              change: "+18%",
              changeType: "positive",
              icon: Users,
              color: "from-yellow-500 to-orange-500"
            }
          ].map((metric, index) => (
            <motion.div
              key={index}
              className="metric-card"
              whileHover={{ scale: 1.02 }}
            >
              <div className="flex items-center justify-between mb-4">
                <div className={`p-3 rounded-lg bg-gradient-to-r ${metric.color}`}>
                  <metric.icon className="h-6 w-6 text-white" />
                </div>
                <div className={`text-sm font-medium ${
                  metric.changeType === 'positive' ? 'text-green-600' : 'text-red-600'
                }`}>
                  {metric.change}
                </div>
              </div>
              <div className="text-2xl font-bold text-slate-900 mb-1">{metric.value}</div>
              <div className="text-sm text-slate-600">{metric.title}</div>
            </motion.div>
          ))}
        </div>
      </motion.div>

      {/* Charts Section */}
      <div className="grid lg:grid-cols-2 gap-8">
        {/* Performance Trends */}
        <motion.div variants={itemVariants} className="card p-6">
          <div className="flex items-center justify-between mb-6">
            <h3 className="text-xl font-semibold text-slate-900">Performance Trends</h3>
            <div className="flex items-center space-x-2 text-sm text-slate-500">
              <Activity className="h-4 w-4" />
              <span>Last 7 days</span>
            </div>
          </div>
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={performanceData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis 
                  dataKey="date" 
                  stroke="#64748b"
                  fontSize={12}
                  tickFormatter={(value) => new Date(value).toLocaleDateString('en-US', { month: 'short', day: 'numeric' })}
                />
                <YAxis stroke="#64748b" fontSize={12} />
                <Tooltip 
                  contentStyle={{ 
                    backgroundColor: 'white', 
                    border: '1px solid #e2e8f0', 
                    borderRadius: '8px',
                    boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)'
                  }}
                />
                <Line 
                  type="monotone" 
                  dataKey="accuracy" 
                  stroke="#3b82f6" 
                  strokeWidth={3}
                  dot={{ fill: '#3b82f6', strokeWidth: 2, r: 4 }}
                  name="Assignment Accuracy"
                />
                <Line 
                  type="monotone" 
                  dataKey="productivity" 
                  stroke="#10b981" 
                  strokeWidth={3}
                  dot={{ fill: '#10b981', strokeWidth: 2, r: 4 }}
                  name="Team Productivity"
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </motion.div>

        {/* Task Complexity Distribution */}
        <motion.div variants={itemVariants} className="card p-6">
          <div className="flex items-center justify-between mb-6">
            <h3 className="text-xl font-semibold text-slate-900">Task Complexity Distribution</h3>
            <div className="flex items-center space-x-2 text-sm text-slate-500">
              <BarChart3 className="h-4 w-4" />
              <span>Current week</span>
            </div>
          </div>
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={complexityDistribution}
                  cx="50%"
                  cy="50%"
                  innerRadius={60}
                  outerRadius={120}
                  paddingAngle={5}
                  dataKey="value"
                >
                  {complexityDistribution.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip 
                  contentStyle={{ 
                    backgroundColor: 'white', 
                    border: '1px solid #e2e8f0', 
                    borderRadius: '8px',
                    boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)'
                  }}
                />
              </PieChart>
            </ResponsiveContainer>
          </div>
          <div className="flex justify-center space-x-6 mt-4">
            {complexityDistribution.map((item, index) => (
              <div key={index} className="flex items-center space-x-2">
                <div 
                  className="w-3 h-3 rounded-full" 
                  style={{ backgroundColor: item.color }}
                />
                <span className="text-sm text-slate-600">{item.name} ({item.value}%)</span>
              </div>
            ))}
          </div>
        </motion.div>
      </div>

      {/* Recent Activity & AI Insights */}
      <div className="grid lg:grid-cols-3 gap-8">
        {/* Recent Analyses */}
        <motion.div variants={itemVariants} className="lg:col-span-2 card p-6">
          <div className="flex items-center justify-between mb-6">
            <h3 className="text-xl font-semibold text-slate-900">Recent Repository Analyses</h3>
            <button className="text-blue-600 hover:text-blue-700 text-sm font-medium flex items-center space-x-1">
              <span>View all</span>
              <ArrowRight className="h-4 w-4" />
            </button>
          </div>
          <div className="space-y-4">
            {recentAnalyses.map((analysis) => (
              <div key={analysis.id} className="flex items-center justify-between p-4 bg-slate-50 rounded-lg hover:bg-slate-100 transition-colors duration-200">
                <div className="flex items-center space-x-4">
                  <div className="p-2 bg-blue-100 rounded-lg">
                    <Github className="h-5 w-5 text-blue-600" />
                  </div>
                  <div>
                    <div className="font-medium text-slate-900">{analysis.repository}</div>
                    <div className="text-sm text-slate-500">{analysis.timestamp}</div>
                  </div>
                </div>
                <div className="flex items-center space-x-6 text-sm text-slate-600">
                  <div className="flex items-center space-x-1">
                    <Users className="h-4 w-4" />
                    <span>{analysis.developers} devs</span>
                  </div>
                  <div className="flex items-center space-x-1">
                    <BarChart3 className="h-4 w-4" />
                    <span>{analysis.tasks} tasks</span>
                  </div>
                  <div className="flex items-center space-x-1">
                    <CheckCircle className="h-4 w-4 text-green-500" />
                    <span className="text-green-600 font-medium">Complete</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </motion.div>

        {/* AI Insights */}
        <motion.div variants={itemVariants} className="card p-6">
          <div className="flex items-center space-x-2 mb-6">
            <div className="p-2 bg-gradient-to-r from-purple-500 to-pink-500 rounded-lg">
              <Sparkles className="h-5 w-5 text-white" />
            </div>
            <h3 className="text-xl font-semibold text-slate-900">AI Insights</h3>
          </div>
          <div className="space-y-4">
            <div className="p-4 bg-blue-50 border border-blue-200 rounded-lg">
              <div className="flex items-start space-x-2">
                <Brain className="h-5 w-5 text-blue-600 mt-0.5" />
                <div>
                  <div className="font-medium text-blue-900 mb-1">Learning Optimization</div>
                  <div className="text-sm text-blue-700">
                    Assignment accuracy improved by 3.2% this week through continuous learning.
                  </div>
                </div>
              </div>
            </div>
            
            <div className="p-4 bg-green-50 border border-green-200 rounded-lg">
              <div className="flex items-start space-x-2">
                <TrendingUp className="h-5 w-5 text-green-600 mt-0.5" />
                <div>
                  <div className="font-medium text-green-900 mb-1">Productivity Trend</div>
                  <div className="text-sm text-green-700">
                    Teams using AI assignments show 34% higher velocity than manual assignment.
                  </div>
                </div>
              </div>
            </div>
            
            <div className="p-4 bg-yellow-50 border border-yellow-200 rounded-lg">
              <div className="flex items-start space-x-2">
                <AlertTriangle className="h-5 w-5 text-yellow-600 mt-0.5" />
                <div>
                  <div className="font-medium text-yellow-900 mb-1">Skill Gap Alert</div>
                  <div className="text-sm text-yellow-700">
                    Security expertise shortage detected in 23% of analyzed teams.
                  </div>
                </div>
              </div>
            </div>
            
            <div className="p-4 bg-purple-50 border border-purple-200 rounded-lg">
              <div className="flex items-start space-x-2">
                <Zap className="h-5 w-5 text-purple-600 mt-0.5" />
                <div>
                  <div className="font-medium text-purple-900 mb-1">Model Performance</div>
                  <div className="text-sm text-purple-700">
                    Task complexity prediction accuracy reached 97.3% - new record!
                  </div>
                </div>
              </div>
            </div>
          </div>
        </motion.div>
      </div>

      {/* Quick Actions */}
      <motion.div variants={itemVariants}>
        <div className="card p-6">
          <h3 className="text-xl font-semibold text-slate-900 mb-6">Quick Actions</h3>
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-4">
            {[
              {
                title: "Analyze Repository",
                description: "Get instant team intelligence",
                icon: Github,
                href: "/dashboard/analyze",
                color: "from-blue-500 to-cyan-500"
              },
              {
                title: "View Team Matrix",
                description: "Comprehensive skill analysis",
                icon: Users,
                href: "/dashboard/team",
                color: "from-green-500 to-emerald-500"
              },
              {
                title: "Optimize Tasks",
                description: "AI-powered assignments",
                icon: Zap,
                href: "/dashboard/tasks",
                color: "from-purple-500 to-pink-500"
              },
              {
                title: "Learning Analytics",
                description: "System performance insights",
                icon: BarChart3,
                href: "/dashboard/analytics",
                color: "from-yellow-500 to-orange-500"
              }
            ].map((action, index) => (
              <motion.button
                key={index}
                className="p-6 bg-gradient-to-br from-white to-slate-50 border border-slate-200 rounded-xl text-left hover:shadow-lg transition-all duration-300 group"
                whileHover={{ scale: 1.02, y: -2 }}
                whileTap={{ scale: 0.98 }}
              >
                <div className={`w-12 h-12 rounded-lg bg-gradient-to-r ${action.color} flex items-center justify-center mb-4 group-hover:scale-110 transition-transform duration-300`}>
                  <action.icon className="h-6 w-6 text-white" />
                </div>
                <h4 className="font-semibold text-slate-900 mb-2">{action.title}</h4>
                <p className="text-sm text-slate-600">{action.description}</p>
              </motion.button>
            ))}
          </div>
        </div>
      </motion.div>
    </motion.div>
  );
}
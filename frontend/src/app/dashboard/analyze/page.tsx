'use client';

import { useState, useEffect } from 'react';
import { useSearchParams } from 'next/navigation';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Github, 
  Users, 
  Brain, 
  Zap, 
  BarChart3, 
  Clock, 
  CheckCircle, 
  ArrowRight,
  AlertTriangle,
  TrendingUp,
  Star,
  GitFork,
  Eye,
  Calendar,
  Code,
  Target
} from 'lucide-react';
// Simple circular progress component
const CircularProgressbar = ({ value, text }: { value: number; text: string }) => (
  <div className="relative w-16 h-16">
    <svg className="w-full h-full transform -rotate-90" viewBox="0 0 36 36">
      <path
        className="text-gray-200"
        d="M18 2.0845
          a 15.9155 15.9155 0 0 1 0 31.831
          a 15.9155 15.9155 0 0 1 0 -31.831"
        stroke="currentColor"
        strokeWidth="2"
        fill="none"
      />
      <path
        className="text-blue-600"
        d="M18 2.0845
          a 15.9155 15.9155 0 0 1 0 31.831
          a 15.9155 15.9155 0 0 1 0 -31.831"
        stroke="currentColor"
        strokeWidth="2"
        fill="none"
        strokeDasharray={`${value}, 100`}
      />
    </svg>
    <div className="absolute inset-0 flex items-center justify-center text-xs font-medium">
      {text}
    </div>
  </div>
);
const buildStyles = () => ({});
import { RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar, ResponsiveContainer, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip } from 'recharts';
import { githubApi, GitHubAnalysisResponse, TeamMember, TaskAnalysis } from '@/lib/api-client';
import { validateGitHubUrl, parseGitHubUrl, formatNumber, getComplexityLevel, getComplexityColor, generateAnalysisSteps, formatTimeAgo } from '@/lib/utils';

interface AnalysisStep {
  id: number;
  name: string;
  description: string;
  status: 'pending' | 'active' | 'completed' | 'error';
  duration?: number;
}

export default function AnalyzePage() {
  const searchParams = useSearchParams();
  const [repoUrl, setRepoUrl] = useState(searchParams?.get('repo') || '');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisSteps, setAnalysisSteps] = useState<AnalysisStep[]>(
    generateAnalysisSteps().map(step => ({ ...step, status: 'pending' as const }))
  );
  const [analysisResult, setAnalysisResult] = useState<GitHubAnalysisResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [currentStep, setCurrentStep] = useState(0);

  const simulateAnalysisSteps = async () => {
    const stepDurations = [3000, 8000, 5000, 7000, 4000, 3000]; // milliseconds for each step
    
    for (let i = 0; i < analysisSteps.length; i++) {
      setCurrentStep(i);
      setAnalysisSteps(prev => prev.map((step, index) => ({
        ...step,
        status: index === i ? 'active' : index < i ? 'completed' : 'pending'
      })));

      await new Promise(resolve => setTimeout(resolve, stepDurations[i]));
      
      setAnalysisSteps(prev => prev.map((step, index) => ({
        ...step,
        status: index <= i ? 'completed' : 'pending',
        duration: index === i ? stepDurations[i] : step.duration
      })));
    }
  };

  const handleAnalyze = async () => {
    if (!validateGitHubUrl(repoUrl)) return;

    setIsAnalyzing(true);
    setError(null);
    setAnalysisResult(null);

    try {
      // Start the visual progress simulation
      const stepsPromise = simulateAnalysisSteps();
      
      // Make the actual API call
      const analysisPromise = githubApi.analyzeRepository({
        repo_url: repoUrl,
        analyze_team: true,
        days_back: 90
      });

      // Wait for both to complete
      const [_, result] = await Promise.all([stepsPromise, analysisPromise]);
      
      setAnalysisResult(result);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Analysis failed. Please try again.');
      setAnalysisSteps(prev => prev.map(step => ({ 
        ...step, 
        status: step.status === 'active' ? 'error' : step.status 
      })));
    } finally {
      setIsAnalyzing(false);
    }
  };

  const getStepIcon = (status: string) => {
    switch (status) {
      case 'completed': return <CheckCircle className="h-5 w-5 text-green-500" />;
      case 'active': return <div className="w-5 h-5 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />;
      case 'error': return <AlertTriangle className="h-5 w-5 text-red-500" />;
      default: return <div className="w-5 h-5 border-2 border-slate-300 rounded-full" />;
    }
  };

  const renderTeamMember = (member: TeamMember, index: number) => {
    const topSkills = Object.entries(member.skill_vector)
      .sort(([,a], [,b]) => b - a)
      .slice(0, 3);

    return (
      <motion.div
        key={member.id}
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: index * 0.1 }}
        className="card p-6 hover:shadow-xl transition-all duration-300"
      >
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center space-x-3">
            <div className="w-12 h-12 bg-gradient-to-r from-blue-500 to-purple-600 rounded-full flex items-center justify-center text-white font-bold">
              {member.name.split(' ').map(n => n[0]).join('').substring(0, 2)}
            </div>
            <div>
              <h4 className="font-semibold text-slate-900">{member.name}</h4>
              <p className="text-sm text-slate-500">@{member.github_username}</p>
            </div>
          </div>
          <div className="text-right">
            <div className="text-sm text-slate-500">Collaboration Score</div>
            <div className="text-lg font-bold text-blue-600">{Math.round(member.collaboration_score * 100)}%</div>
          </div>
        </div>

        <div className="grid grid-cols-2 gap-4 mb-4">
          <div className="text-center">
            <div className="text-2xl font-bold text-slate-900">{formatNumber(member.commits_analyzed)}</div>
            <div className="text-sm text-slate-500">Commits</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-slate-900">{formatNumber(member.lines_of_code)}</div>
            <div className="text-sm text-slate-500">Lines of Code</div>
          </div>
        </div>

        <div className="space-y-2">
          <div className="text-sm font-medium text-slate-700">Top Skills</div>
          {topSkills.map(([skill, level]) => (
            <div key={skill} className="flex items-center justify-between">
              <span className="text-sm text-slate-600 capitalize">{skill.replace('_', ' ')}</span>
              <div className="flex items-center space-x-2">
                <div className="w-24 h-2 bg-slate-200 rounded-full overflow-hidden">
                  <div 
                    className="h-full bg-gradient-to-r from-blue-500 to-purple-600 rounded-full transition-all duration-1000 ease-out"
                    style={{ width: `${level * 100}%` }}
                  />
                </div>
                <span className="text-sm font-medium text-slate-700">{Math.round(level * 100)}%</span>
              </div>
            </div>
          ))}
        </div>

        <div className="mt-4 flex items-center justify-between text-sm">
          <div className="flex items-center space-x-1">
            <TrendingUp className="h-4 w-4 text-green-500" />
            <span className="text-green-600">Learning Velocity: {Math.round(member.learning_velocity * 100)}%</span>
          </div>
        </div>
      </motion.div>
    );
  };

  const renderTaskAnalysis = (task: TaskAnalysis, index: number) => {
    const complexityLevel = getComplexityLevel(task.technical_complexity);
    const complexityColor = getComplexityColor(complexityLevel);

    return (
      <motion.div
        key={task.id}
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: index * 0.1 }}
        className="card p-6 hover:shadow-xl transition-all duration-300"
      >
        <div className="flex items-start justify-between mb-4">
          <div className="flex-1">
            <div className="flex items-center space-x-2 mb-2">
              <span className="text-sm text-slate-500">#{task.github_issue_number}</span>
              {task.labels.map(label => (
                <span key={label} className="px-2 py-1 bg-blue-100 text-blue-800 text-xs rounded-full">
                  {label}
                </span>
              ))}
            </div>
            <h4 className="font-semibold text-slate-900 mb-2">{task.title}</h4>
            <p className="text-sm text-slate-600 line-clamp-2">{task.description}</p>
          </div>
          <div className={`px-3 py-1 rounded-full text-sm font-medium ${complexityColor}`}>
            {complexityLevel.toUpperCase()}
          </div>
        </div>

        <div className="grid grid-cols-5 gap-4 mb-4">
          {[
            { label: 'Technical', value: task.technical_complexity, color: 'text-blue-600' },
            { label: 'Domain', value: task.domain_difficulty, color: 'text-purple-600' },
            { label: 'Collaboration', value: task.collaboration_requirements, color: 'text-green-600' },
            { label: 'Learning', value: task.learning_opportunities, color: 'text-yellow-600' },
            { label: 'Business', value: task.business_impact, color: 'text-red-600' },
          ].map((dimension) => (
            <div key={dimension.label} className="text-center">
              <div className="w-12 h-12 mx-auto mb-2">
                <CircularProgressbar 
                  value={dimension.value * 100} 
                  text={`${Math.round(dimension.value * 100)}%`}
                />
              </div>
              <div className="text-xs text-slate-600">{dimension.label}</div>
              <div className={`text-sm font-bold ${dimension.color}`}>
                {Math.round(dimension.value * 100)}%
              </div>
            </div>
          ))}
        </div>

        <div className="flex items-center justify-between text-sm">
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-1">
              <Clock className="h-4 w-4 text-slate-400" />
              <span className="text-slate-600">{Math.round(task.estimated_hours)}h estimated</span>
            </div>
            <div className="flex items-center space-x-1">
              <Target className="h-4 w-4 text-slate-400" />
              <span className="text-slate-600">{Math.round(task.confidence_score * 100)}% confidence</span>
            </div>
          </div>
        </div>

        {task.risk_factors.length > 0 && (
          <div className="mt-3 pt-3 border-t border-slate-100">
            <div className="text-xs text-slate-500 mb-1">Risk Factors</div>
            <div className="flex flex-wrap gap-1">
              {task.risk_factors.map((risk, idx) => (
                <span key={idx} className="px-2 py-1 bg-red-50 text-red-700 text-xs rounded">
                  {risk}
                </span>
              ))}
            </div>
          </div>
        )}
      </motion.div>
    );
  };

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-slate-900">GitHub Repository Analysis</h1>
          <p className="text-slate-600 mt-1">AI-powered team intelligence and task complexity prediction</p>
        </div>
      </div>

      {/* Analysis Input */}
      <div className="card p-8">
        <h2 className="text-xl font-semibold text-slate-900 mb-6">Repository Analysis</h2>
        <div className="flex space-x-4">
          <div className="flex-1 relative">
            <input
              type="url"
              value={repoUrl}
              onChange={(e) => setRepoUrl(e.target.value)}
              placeholder="https://github.com/microsoft/vscode"
              className={`input-field pr-12 ${
                repoUrl && !validateGitHubUrl(repoUrl) ? 'border-red-300 focus:ring-red-500' : 
                validateGitHubUrl(repoUrl) ? 'border-green-300 focus:ring-green-500' : ''
              }`}
              disabled={isAnalyzing}
            />
            {validateGitHubUrl(repoUrl) && (
              <CheckCircle className="absolute right-3 top-1/2 transform -translate-y-1/2 h-5 w-5 text-green-500" />
            )}
          </div>
          <motion.button
            onClick={handleAnalyze}
            disabled={!validateGitHubUrl(repoUrl) || isAnalyzing}
            className={`btn-primary flex items-center space-x-2 px-8 ${
              !validateGitHubUrl(repoUrl) || isAnalyzing ? 'opacity-50 cursor-not-allowed' : 'hover:scale-105'
            }`}
            whileHover={validateGitHubUrl(repoUrl) && !isAnalyzing ? { scale: 1.05 } : {}}
            whileTap={validateGitHubUrl(repoUrl) && !isAnalyzing ? { scale: 0.95 } : {}}
          >
            <Brain className="h-5 w-5" />
            <span>{isAnalyzing ? 'Analyzing...' : 'Analyze Repository'}</span>
            {!isAnalyzing && <ArrowRight className="h-4 w-4" />}
          </motion.button>
        </div>
        {repoUrl && !validateGitHubUrl(repoUrl) && (
          <p className="text-red-500 text-sm mt-2">Please enter a valid GitHub repository URL</p>
        )}
      </div>

      {/* Analysis Progress */}
      <AnimatePresence>
        {isAnalyzing && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className="card p-8"
          >
            <div className="flex items-center space-x-3 mb-8">
              <div className="w-8 h-8 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />
              <div>
                <h3 className="text-xl font-semibold text-slate-900">AI Analysis in Progress</h3>
                <p className="text-slate-600">Extracting team intelligence and predicting task complexity...</p>
              </div>
            </div>

            <div className="space-y-4">
              {analysisSteps.map((step, index) => (
                <motion.div
                  key={step.id}
                  className={`flex items-center space-x-4 p-4 rounded-lg transition-all duration-300 ${
                    step.status === 'active' ? 'bg-blue-50 border border-blue-200' :
                    step.status === 'completed' ? 'bg-green-50 border border-green-200' :
                    step.status === 'error' ? 'bg-red-50 border border-red-200' :
                    'bg-slate-50 border border-slate-200'
                  }`}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: index * 0.1 }}
                >
                  <div className="flex-shrink-0">
                    {getStepIcon(step.status)}
                  </div>
                  <div className="flex-1">
                    <div className="font-medium text-slate-900">{step.name}</div>
                    <div className="text-sm text-slate-600">{step.description}</div>
                  </div>
                  {step.status === 'completed' && step.duration && (
                    <div className="text-sm text-slate-500">
                      {(step.duration / 1000).toFixed(1)}s
                    </div>
                  )}
                </motion.div>
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Error Display */}
      <AnimatePresence>
        {error && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className="bg-red-50 border border-red-200 rounded-xl p-6"
          >
            <div className="flex items-center space-x-3">
              <AlertTriangle className="h-6 w-6 text-red-500" />
              <div>
                <h3 className="font-semibold text-red-900">Analysis Failed</h3>
                <p className="text-red-700">{error}</p>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Analysis Results */}
      <AnimatePresence>
        {analysisResult && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="space-y-8"
          >
            {/* Repository Summary */}
            <div className="card p-8">
              <div className="flex items-center justify-between mb-6">
                <div className="flex items-center space-x-4">
                  <div className="p-3 bg-blue-100 rounded-full">
                    <Github className="h-8 w-8 text-blue-600" />
                  </div>
                  <div>
                    <h2 className="text-2xl font-bold text-slate-900">{analysisResult.repository.name}</h2>
                    <p className="text-slate-600">by {analysisResult.repository.owner}</p>
                  </div>
                </div>
                <div className="text-right">
                  <div className="text-sm text-slate-500">Analysis completed in</div>
                  <div className="text-2xl font-bold text-blue-600">
                    {(analysisResult.analysis_time_ms / 1000).toFixed(1)}s
                  </div>
                </div>
              </div>

              <p className="text-slate-600 mb-6">{analysisResult.repository.description}</p>

              <div className="grid grid-cols-2 md:grid-cols-5 gap-6">
                <div className="text-center">
                  <div className="flex items-center justify-center space-x-1 text-2xl font-bold text-slate-900">
                    <Star className="h-6 w-6 text-yellow-500" />
                    <span>{formatNumber(analysisResult.repository.stars)}</span>
                  </div>
                  <div className="text-sm text-slate-500">Stars</div>
                </div>
                <div className="text-center">
                  <div className="flex items-center justify-center space-x-1 text-2xl font-bold text-slate-900">
                    <GitFork className="h-6 w-6 text-blue-500" />
                    <span>{formatNumber(analysisResult.repository.forks)}</span>
                  </div>
                  <div className="text-sm text-slate-500">Forks</div>
                </div>
                <div className="text-center">
                  <div className="flex items-center justify-center space-x-1 text-2xl font-bold text-slate-900">
                    <Eye className="h-6 w-6 text-green-500" />
                    <span>{formatNumber(analysisResult.repository.watchers)}</span>
                  </div>
                  <div className="text-sm text-slate-500">Watchers</div>
                </div>
                <div className="text-center">
                  <div className="flex items-center justify-center space-x-1 text-2xl font-bold text-slate-900">
                    <Code className="h-6 w-6 text-purple-500" />
                    <span>{analysisResult.repository.language}</span>
                  </div>
                  <div className="text-sm text-slate-500">Primary Language</div>
                </div>
                <div className="text-center">
                  <div className="flex items-center justify-center space-x-1 text-2xl font-bold text-slate-900">
                    <Calendar className="h-6 w-6 text-orange-500" />
                    <span>{formatTimeAgo(analysisResult.repository.updated_at)}</span>
                  </div>
                  <div className="text-sm text-slate-500">Last Updated</div>
                </div>
              </div>
            </div>

            {/* Team Metrics Overview */}
            <div className="grid md:grid-cols-4 gap-6">
              {[
                {
                  title: "Team Members",
                  value: analysisResult.team_metrics.total_developers.toString(),
                  subtitle: "Active contributors",
                  icon: Users,
                  color: "from-blue-500 to-cyan-500"
                },
                {
                  title: "Avg Skill Level",
                  value: `${Math.round(analysisResult.team_metrics.avg_skill_level * 100)}%`,
                  subtitle: "Team expertise",
                  icon: Brain,
                  color: "from-purple-500 to-pink-500"
                },
                {
                  title: "Collaboration",
                  value: `${Math.round(analysisResult.team_metrics.collaboration_score * 100)}%`,
                  subtitle: "Team synergy",
                  icon: Users,
                  color: "from-green-500 to-emerald-500"
                },
                {
                  title: "Learning Velocity",
                  value: `${Math.round(analysisResult.team_metrics.avg_learning_velocity * 100)}%`,
                  subtitle: "Growth rate",
                  icon: TrendingUp,
                  color: "from-yellow-500 to-orange-500"
                }
              ].map((metric, index) => (
                <motion.div
                  key={index}
                  className="metric-card"
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: index * 0.1 }}
                >
                  <div className="flex items-center justify-between mb-4">
                    <div className={`p-3 rounded-lg bg-gradient-to-r ${metric.color}`}>
                      <metric.icon className="h-6 w-6 text-white" />
                    </div>
                  </div>
                  <div className="text-3xl font-bold text-slate-900 mb-1">{metric.value}</div>
                  <div className="text-sm font-medium text-slate-700 mb-1">{metric.title}</div>
                  <div className="text-xs text-slate-500">{metric.subtitle}</div>
                </motion.div>
              ))}
            </div>

            {/* Team Members */}
            <div>
              <div className="flex items-center justify-between mb-6">
                <h3 className="text-2xl font-bold text-slate-900">Team Intelligence</h3>
                <span className="text-sm text-slate-500">
                  {analysisResult.developers.length} developers analyzed
                </span>
              </div>
              <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
                {analysisResult.developers.map((member, index) => renderTeamMember(member, index))}
              </div>
            </div>

            {/* Task Analysis */}
            <div>
              <div className="flex items-center justify-between mb-6">
                <h3 className="text-2xl font-bold text-slate-900">Task Complexity Analysis</h3>
                <span className="text-sm text-slate-500">
                  {analysisResult.tasks.length} tasks analyzed
                </span>
              </div>
              <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
                {analysisResult.tasks.map((task, index) => renderTaskAnalysis(task, index))}
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
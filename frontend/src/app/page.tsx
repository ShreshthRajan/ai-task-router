'use client';

import { useState } from 'react';
import { motion } from 'framer-motion';
import { useRouter } from 'next/navigation';
import { 
  Brain, 
  Github, 
  Zap, 
  Target, 
  Users, 
  BarChart3, 
  ArrowRight,
  CheckCircle,
  Sparkles,
  TrendingUp,
  Clock,
  Shield
} from 'lucide-react';
import { validateGitHubUrl } from '../../../lib/utils';

export default function LandingPage() {
  const [repoUrl, setRepoUrl] = useState('');
  const [isValidUrl, setIsValidUrl] = useState(false);
  const router = useRouter();

  const handleUrlChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const url = e.target.value;
    setRepoUrl(url);
    setIsValidUrl(validateGitHubUrl(url));
  };

  const handleAnalyze = () => {
    if (isValidUrl) {
      const encodedUrl = encodeURIComponent(repoUrl);
      router.push(`/dashboard/analyze?repo=${encodedUrl}`);
    }
  };

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
        duration: 0.6
      }
    }
  } as any;

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-100">
      {/* Navigation */}
      <nav className="fixed top-0 w-full z-50 bg-white/80 backdrop-blur-md border-b border-slate-200/50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <motion.div 
              className="flex items-center space-x-3"
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.5 }}
            >
              <div className="p-2 bg-gradient-to-r from-blue-500 to-purple-600 rounded-lg">
                <Brain className="h-6 w-6 text-white" />
              </div>
              <span className="text-xl font-bold gradient-text">AI Development Intelligence</span>
            </motion.div>
            
            <motion.div 
              className="flex items-center space-x-4"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.5, delay: 0.1 }}
            >
              <button 
                onClick={() => router.push('/dashboard')}
                className="btn-secondary"
              >
                Dashboard
              </button>
              <div className="flex items-center space-x-2 text-sm text-green-600 bg-green-50 px-3 py-1 rounded-full">
                <div className="w-2 h-2 bg-green-500 rounded-full pulse-indicator"></div>
                <span className="font-medium">System Online</span>
              </div>
            </motion.div>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="pt-32 pb-20 px-4 sm:px-6 lg:px-8">
        <div className="max-w-7xl mx-auto">
          <motion.div 
            className="text-center max-w-4xl mx-auto"
            variants={containerVariants}
            initial="hidden"
            animate="visible"
          >
            <motion.div variants={itemVariants} className="mb-6">
              <span className="inline-flex items-center space-x-2 bg-blue-50 text-blue-700 px-4 py-2 rounded-full text-sm font-medium border border-blue-200">
                <Sparkles className="h-4 w-4" />
                <span>World's First Intelligent Task Assignment System</span>
              </span>
            </motion.div>

            <motion.h1 
              variants={itemVariants}
              className="text-5xl md:text-6xl lg:text-7xl font-bold text-slate-900 mb-6 leading-tight"
            >
              Revolutionary{' '}
              <span className="gradient-text">AI Development</span>{' '}
              Intelligence
            </motion.h1>

            <motion.p 
              variants={itemVariants}
              className="text-xl md:text-2xl text-slate-600 mb-8 leading-relaxed"
            >
              Analyze any GitHub repository in 30 seconds. Get instant team intelligence, 
              task complexity predictions, and optimal assignment recommendations powered by 
              cutting-edge AI research.
            </motion.p>

            {/* GitHub URL Input */}
            <motion.div 
              variants={itemVariants}
              className="max-w-2xl mx-auto mb-12"
            >
              <div className="glass-card p-8 rounded-2xl">
                <h3 className="text-lg font-semibold text-slate-900 mb-4">
                  Paste any GitHub repository URL to start
                </h3>
                <div className="flex space-x-4">
                  <div className="flex-1 relative">
                    <input
                      type="url"
                      value={repoUrl}
                      onChange={handleUrlChange}
                      placeholder="https://github.com/microsoft/vscode"
                      className={`input-field pr-12 ${
                        repoUrl && !isValidUrl ? 'border-red-300 focus:ring-red-500' : 
                        isValidUrl ? 'border-green-300 focus:ring-green-500' : ''
                      }`}
                    />
                    {isValidUrl && (
                      <CheckCircle className="absolute right-3 top-1/2 transform -translate-y-1/2 h-5 w-5 text-green-500" />
                    )}
                  </div>
                  <motion.button
                    onClick={handleAnalyze}
                    disabled={!isValidUrl}
                    className={`btn-primary flex items-center space-x-2 px-8 ${
                      !isValidUrl ? 'opacity-50 cursor-not-allowed' : 'hover:scale-105'
                    }`}
                    whileHover={isValidUrl ? { scale: 1.05 } : {}}
                    whileTap={isValidUrl ? { scale: 0.95 } : {}}
                  >
                    <Brain className="h-5 w-5" />
                    <span>Analyze Repository</span>
                    <ArrowRight className="h-4 w-4" />
                  </motion.button>
                </div>
                {repoUrl && !isValidUrl && (
                  <p className="text-red-500 text-sm mt-2">Please enter a valid GitHub repository URL</p>
                )}
              </div>
            </motion.div>

            {/* Quick Stats */}
            <motion.div 
              variants={itemVariants}
              className="grid grid-cols-2 md:grid-cols-4 gap-6 max-w-4xl mx-auto"
            >
              <div className="metric-card text-center">
                <div className="text-3xl font-bold gradient-text">768</div>
                <div className="text-sm text-slate-600">Skill Dimensions</div>
              </div>
              <div className="metric-card text-center">
                <div className="text-3xl font-bold gradient-text">5D</div>
                <div className="text-sm text-slate-600">Complexity Analysis</div>
              </div>
              <div className="metric-card text-center">
                <div className="text-3xl font-bold gradient-text">35%</div>
                <div className="text-sm text-slate-600">Productivity Boost</div>
              </div>
              <div className="metric-card text-center">
                <div className="text-3xl font-bold gradient-text">&lt;30s</div>
                <div className="text-sm text-slate-600">Analysis Time</div>
              </div>
            </motion.div>
          </motion.div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-20 bg-white/50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div 
            className="text-center mb-16"
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            viewport={{ once: true }}
          >
            <h2 className="text-4xl md:text-5xl font-bold text-slate-900 mb-4">
              Breakthrough AI Capabilities
            </h2>
            <p className="text-xl text-slate-600 max-w-3xl mx-auto">
              No comparable system exists. We've built the world's first truly intelligent 
              task assignment platform combining cutting-edge research with production value.
            </p>
          </motion.div>

          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
            {[
              {
                icon: Github,
                title: "Live GitHub Intelligence",
                description: "Real-time analysis of any repository. Extract team expertise, collaboration patterns, and development velocity in seconds.",
                color: "from-purple-500 to-pink-500"
              },
              {
                icon: Brain,
                title: "768-Dimensional Skill Modeling",
                description: "Multi-modal developer expertise combining code semantics, collaboration patterns, and temporal evolution.",
                color: "from-blue-500 to-cyan-500"
              },
              {
                icon: Target,
                title: "5D Task Complexity Prediction",
                description: "Technical complexity, domain difficulty, collaboration requirements, learning opportunities, and business impact.",
                color: "from-green-500 to-emerald-500"
              },
              {
                icon: Zap,
                title: "Multi-Objective Optimization",
                description: "Hungarian algorithm with constraint satisfaction balancing productivity, learning, and team dynamics.",
                color: "from-yellow-500 to-orange-500"
              },
              {
                icon: TrendingUp,
                title: "Continuous Learning System",
                description: "Self-improving AI that learns from every assignment outcome with A/B testing and model optimization.",
                color: "from-red-500 to-pink-500"
              },
              {
                icon: BarChart3,
                title: "Predictive Analytics",
                description: "Team performance forecasting, ROI measurement, and proactive issue detection with business impact analysis.",
                color: "from-indigo-500 to-purple-500"
              }
            ].map((feature, index) => (
              <motion.div
                key={index}
                className="card p-8 group hover:scale-105"
                initial={{ opacity: 0, y: 30 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: index * 0.1 }}
                viewport={{ once: true }}
              >
                <div className={`w-12 h-12 rounded-lg bg-gradient-to-r ${feature.color} flex items-center justify-center mb-6 group-hover:scale-110 transition-transform duration-300`}>
                  <feature.icon className="h-6 w-6 text-white" />
                </div>
                <h3 className="text-xl font-semibold text-slate-900 mb-3">{feature.title}</h3>
                <p className="text-slate-600 leading-relaxed">{feature.description}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* How It Works */}
      <section className="py-20">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div 
            className="text-center mb-16"
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            viewport={{ once: true }}
          >
            <h2 className="text-4xl md:text-5xl font-bold text-slate-900 mb-4">
              Revolutionary 6-Step Analysis
            </h2>
            <p className="text-xl text-slate-600">
              Watch AI analyze your repository and generate optimal assignments in real-time
            </p>
          </motion.div>

          <div className="space-y-12">
            {[
              {
                step: 1,
                title: "Repository Scanning",
                description: "AI analyzes codebase structure, languages, frameworks, and development patterns to understand project complexity.",
                icon: Github,
                time: "~3s"
              },
              {
                step: 2,
                title: "Team Intelligence Extraction",
                description: "Builds 768-dimensional skill vectors for each developer using commits, PRs, and collaboration data.",
                icon: Users,
                time: "~8s"
              },
              {
                step: 3,
                title: "Task Complexity Prediction",
                description: "Analyzes GitHub issues across 5 dimensions using advanced NLP and domain knowledge.",
                icon: Target,
                time: "~5s"
              },
              {
                step: 4,
                title: "Assignment Optimization",
                description: "Multi-objective algorithm finds optimal developer-task matches balancing productivity and learning.",
                icon: Zap,
                time: "~7s"
              },
              {
                step: 5,
                title: "Learning Integration",
                description: "Continuous learning system improves predictions based on historical assignment outcomes.",
                icon: Brain,
                time: "~4s"
              },
              {
                step: 6,
                title: "Intelligence Report",
                description: "Generates actionable insights with team analytics, assignment recommendations, and growth opportunities.",
                icon: BarChart3,
                time: "~3s"
              }
            ].map((step, index) => (
              <motion.div
                key={index}
                className="flex items-center space-x-8"
                initial={{ opacity: 0, x: index % 2 === 0 ? -50 : 50 }}
                whileInView={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.6, delay: index * 0.1 }}
                viewport={{ once: true }}
              >
                <div className={`flex-shrink-0 w-24 h-24 rounded-full bg-gradient-to-r from-blue-500 to-purple-600 flex items-center justify-center text-white font-bold text-xl ${index % 2 === 1 ? 'order-2' : ''}`}>
                  {step.step}
                </div>
                <div className={`card p-8 flex-1 ${index % 2 === 1 ? 'order-1' : ''}`}>
                  <div className="flex items-center justify-between mb-4">
                    <div className="flex items-center space-x-3">
                      <step.icon className="h-6 w-6 text-blue-600" />
                      <h3 className="text-2xl font-semibold text-slate-900">{step.title}</h3>
                    </div>
                    <div className="flex items-center space-x-2 text-slate-500">
                      <Clock className="h-4 w-4" />
                      <span className="text-sm">{step.time}</span>
                    </div>
                  </div>
                  <p className="text-slate-600 text-lg leading-relaxed">{step.description}</p>
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-20 bg-gradient-to-r from-blue-600 to-purple-700">
        <div className="max-w-4xl mx-auto text-center px-4 sm:px-6 lg:px-8">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            viewport={{ once: true }}
          >
            <h2 className="text-4xl md:text-5xl font-bold text-white mb-6">
              Ready to revolutionize your development process?
            </h2>
            <p className="text-xl text-blue-100 mb-8">
              Join the future of intelligent software development. Analyze your first repository in under 30 seconds.
            </p>
            <div className="flex justify-center space-x-4">
              <button 
                onClick={() => router.push('/dashboard/analyze')}
                className="bg-white text-blue-600 font-semibold py-4 px-8 rounded-lg hover:bg-blue-50 transition-all duration-200 transform hover:scale-105 shadow-lg"
              >
                Start Analysis
              </button>
              <button 
                onClick={() => router.push('/dashboard')}
                className="border-2 border-white text-white font-semibold py-4 px-8 rounded-lg hover:bg-white hover:text-blue-600 transition-all duration-200"
              >
                View Dashboard
              </button>
            </div>
          </motion.div>
        </div>
      </section>

      {/* Footer */}
      <footer className="bg-slate-900 text-white py-12">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex flex-col md:flex-row justify-between items-center">
            <div className="flex items-center space-x-3 mb-4 md:mb-0">
              <div className="p-2 bg-gradient-to-r from-blue-500 to-purple-600 rounded-lg">
                <Brain className="h-6 w-6 text-white" />
              </div>
              <span className="text-xl font-bold">AI Development Intelligence</span>
            </div>
            <div className="flex items-center space-x-6 text-slate-400">
              <div className="flex items-center space-x-2">
                <Shield className="h-4 w-4" />
                <span className="text-sm">Production Ready</span>
              </div>
              <div className="flex items-center space-x-2">
                <Brain className="h-4 w-4" />
                <span className="text-sm">AI Powered</span>
              </div>
              <div className="flex items-center space-x-2">
                <TrendingUp className="h-4 w-4" />
                <span className="text-sm">Research Grade</span>
              </div>
            </div>
          </div>
          <div className="mt-8 pt-8 border-t border-slate-800 text-center text-slate-400">
            <p>&copy; 2024 AI Development Intelligence. The future of software engineering productivity.</p>
          </div>
        </div>
      </footer>
    </div>
  );
}
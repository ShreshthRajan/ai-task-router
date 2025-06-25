// frontend/src/app/dashboard/analyze/page.tsx
'use client';

import { useState } from 'react';
import { motion } from 'framer-motion';
import { useSearchParams } from 'next/navigation';
import { 
  Github, 
  Brain, 
  CheckCircle2, 
  Clock, 
  Users, 
  Target,
  BarChart3,
  AlertCircle,
  Loader2,
  ArrowRight,
  TrendingUp,
  Code,
  Star
} from 'lucide-react';

export default function GitHubAnalyzer() {
  const searchParams = useSearchParams();
  const [repoUrl, setRepoUrl] = useState(searchParams?.get('repo') || '');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const validateGitHubUrl = (url: string): boolean => {
    return /^https?:\/\/github\.com\/[^\/]+\/[^\/]+/.test(url);
  };

  const isValidUrl = validateGitHubUrl(repoUrl);

  const handleAnalyze = async () => {
    if (!isValidUrl || isAnalyzing) return;
    
    setIsAnalyzing(true);
    setError(null);
    
    try {
      // Simulate analysis for now
      await new Promise(resolve => setTimeout(resolve, 3000));
      // You can add real API call here later
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Analysis failed');
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <div className="space-y-8">
      {/* Header */}
      <div>
        <h1 className="text-heading-1 mb-2">Repository Intelligence Analysis</h1>
        <p className="text-body-large text-gray-400">
          Extract team intelligence and task complexity insights from any GitHub repository
        </p>
      </div>

      {/* URL Input Section */}
      <div className="card p-8">
        <div className="flex items-center space-x-3 mb-6">
          <Github className="h-6 w-6 text-gray-400" />
          <h2 className="text-heading-3">Repository Analysis</h2>
        </div>
        
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">
              GitHub Repository URL
            </label>
            <div className="flex space-x-4">
              <input
                type="url"
                value={repoUrl}
                onChange={(e) => setRepoUrl(e.target.value)}
                placeholder="https://github.com/microsoft/vscode"
                className={`input-primary flex-1 ${
                  repoUrl && !isValidUrl ? 'input-error' : ''
                }`}
                disabled={isAnalyzing}
              />
              <button
                onClick={handleAnalyze}
                disabled={!isValidUrl || isAnalyzing}
                className="btn-primary px-8 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {isAnalyzing ? (
                  <>
                    <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                    Analyzing...
                  </>
                ) : (
                  <>
                    <Brain className="h-4 w-4 mr-2" />
                    Analyze Repository
                  </>
                )}
              </button>
            </div>
            {repoUrl && !isValidUrl && (
              <p className="text-sm text-red-400 mt-2">
                Please enter a valid GitHub repository URL
              </p>
            )}
          </div>
          
          <div className="text-sm text-gray-500">
            Example repositories: microsoft/vscode, facebook/react, or tensorflow/tensorflow
          </div>
        </div>
      </div>

      {/* Error Display */}
      {error && (
        <motion.div 
          className="card p-6 border-red-800 bg-red-900/20"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
        >
          <div className="flex items-center space-x-3">
            <AlertCircle className="h-5 w-5 text-red-400" />
            <div>
              <h3 className="font-medium text-red-400">Analysis Failed</h3>
              <p className="text-sm text-red-300 mt-1">{error}</p>
            </div>
          </div>
        </motion.div>
      )}

      {/* Analysis Progress */}
      {isAnalyzing && (
        <motion.div 
          className="card p-6"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
        >
          <div className="flex items-center space-x-3 mb-6">
            <Brain className="h-6 w-6 text-blue-400" />
            <h2 className="text-heading-3">Analysis in Progress</h2>
          </div>
          
          <div className="flex items-center justify-center py-12">
            <div className="text-center">
              <Loader2 className="h-12 w-12 text-blue-400 animate-spin mx-auto mb-4" />
              <p className="text-gray-400">Extracting team intelligence and task complexity...</p>
            </div>
          </div>
        </motion.div>
      )}

      {/* Feature Showcase (when not analyzing) */}
      {!isAnalyzing && (
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
          <div className="card p-6">
            <div className="flex items-center space-x-3 mb-4">
              <Users className="h-8 w-8 text-purple-400" />
              <div>
                <h3 className="font-semibold text-gray-200">Team Intelligence</h3>
                <p className="text-sm text-gray-500">768-dimensional modeling</p>
              </div>
            </div>
            <p className="text-gray-400 text-sm">
              Extract developer skills, collaboration patterns, and expertise evolution from commit history.
            </p>
          </div>

          <div className="card p-6">
            <div className="flex items-center space-x-3 mb-4">
              <Target className="h-8 w-8 text-emerald-400" />
              <div>
                <h3 className="font-semibold text-gray-200">Task Complexity</h3>
                <p className="text-sm text-gray-500">5D analysis framework</p>
              </div>
            </div>
            <p className="text-gray-400 text-sm">
              Predict technical complexity, domain difficulty, collaboration needs, and learning opportunities.
            </p>
          </div>

          <div className="card p-6">
            <div className="flex items-center space-x-3 mb-4">
              <BarChart3 className="h-8 w-8 text-blue-400" />
              <div>
                <h3 className="font-semibold text-gray-200">Assignment Optimization</h3>
                <p className="text-sm text-gray-500">Multi-objective algorithms</p>
              </div>
            </div>
            <p className="text-gray-400 text-sm">
              Generate optimal task assignments balancing productivity, learning, and team dynamics.
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
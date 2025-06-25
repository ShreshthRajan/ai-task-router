'use client'

import { useState } from 'react'
import { motion } from 'framer-motion'
import { Brain, Github, Zap, Users, Target, TrendingUp } from 'lucide-react'
import Link from 'next/link'

export default function LandingPage() {
  const [githubUrl, setGithubUrl] = useState('')

  const handleAnalyze = () => {
    if (githubUrl) {
      window.location.href = `/dashboard/analyze?repo=${encodeURIComponent(githubUrl)}`
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-green-50">
      {/* Header */}
      <header className="px-6 py-4">
        <nav className="max-w-7xl mx-auto flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <Brain className="h-8 w-8 text-primary" />
            <span className="text-xl font-bold text-gray-900">AI Development Intelligence</span>
          </div>
          <Link href="/dashboard" className="btn-primary">
            View Dashboard
          </Link>
        </nav>
      </header>

      {/* Hero Section */}
      <main className="max-w-7xl mx-auto px-6 py-12">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          className="text-center mb-16"
        >
          <h1 className="text-5xl font-bold text-gray-900 mb-6">
            Revolutionary AI-Powered
            <span className="bg-gradient-to-r from-blue-600 to-green-600 bg-clip-text text-transparent">
              {" "}Task Assignment
            </span>
          </h1>
          <p className="text-xl text-gray-600 mb-8 max-w-3xl mx-auto">
            Transform your development workflow with intelligent task routing, predictive analytics, 
            and continuous learning. Analyze any GitHub repository in seconds.
          </p>

          {/* GitHub URL Input */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.2 }}
            className="max-w-2xl mx-auto mb-12"
          >
            <div className="flex gap-4">
              <div className="flex-1">
                <input
                  type="url"
                  placeholder="Paste GitHub repository URL (e.g., https://github.com/owner/repo)"
                  value={githubUrl}
                  onChange={(e) => setGithubUrl(e.target.value)}
                  className="w-full px-6 py-4 text-lg border border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                />
              </div>
              <button
                onClick={handleAnalyze}
                disabled={!githubUrl}
                className="btn-primary px-8 py-4 text-lg disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <Zap className="h-5 w-5 mr-2" />
                Analyze Now
              </button>
            </div>
            <p className="text-sm text-gray-500 mt-2">
              See your team's true potential in 30 seconds
            </p>
          </motion.div>
        </motion.div>

        {/* Features Grid */}
        <motion.div
          initial={{ opacity: 0, y: 40 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.4 }}
          className="grid md:grid-cols-3 gap-8 mb-16"
        >
          <div className="metric-card text-center">
            <Github className="h-12 w-12 text-primary mx-auto mb-4" />
            <h3 className="text-xl font-bold mb-2">Live GitHub Analysis</h3>
            <p className="text-gray-600">
              Real-time extraction of developer skills, task complexity, and team dynamics from actual repositories.
            </p>
          </div>

          <div className="metric-card text-center">
            <Target className="h-12 w-12 text-secondary mx-auto mb-4" />
            <h3 className="text-xl font-bold mb-2">Optimal Assignment</h3>
            <p className="text-gray-600">
              AI-powered multi-objective optimization balancing productivity, learning, and collaboration.
            </p>
          </div>

          <div className="metric-card text-center">
            <TrendingUp className="h-12 w-12 text-accent mx-auto mb-4" />
            <h3 className="text-xl font-bold mb-2">Continuous Learning</h3>
            <p className="text-gray-600">
              System learns from outcomes to improve predictions and recommendations over time.
            </p>
          </div>
        </motion.div>

        {/* Demo Results Preview */}
        <motion.div
          initial={{ opacity: 0, y: 40 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.6 }}
          className="bg-white rounded-2xl border border-gray-200 p-8 shadow-lg"
        >
          <h2 className="text-2xl font-bold text-center mb-8">What You'll See</h2>
          
          <div className="grid md:grid-cols-4 gap-6 text-center">
            <div>
              <div className="metric-value text-primary">+34%</div>
              <div className="metric-label">Productivity Gain</div>
            </div>
            <div>
              <div className="metric-value text-secondary">89%</div>
              <div className="metric-label">Assignment Accuracy</div>
            </div>
            <div>
              <div className="metric-value text-accent">67%</div>
              <div className="metric-label">Faster Skill Development</div>
            </div>
            <div>
              <div className="metric-value text-purple-600">$11.7K</div>
              <div className="metric-label">Monthly Savings</div>
            </div>
          </div>

          <div className="mt-8 p-6 bg-gradient-to-r from-blue-50 to-green-50 rounded-xl">
            <h3 className="font-bold text-lg mb-2">🚀 Real-Time Intelligence</h3>
            <p className="text-gray-700">
              Watch AI agents analyze your repository, extract team skills, predict task complexity, 
              and generate optimal assignments in real-time. No setup required.
            </p>
          </div>
        </motion.div>

        {/* CTA Section */}
        <motion.div
          initial={{ opacity: 0, y: 40 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.8 }}
          className="text-center mt-16"
        >
          <h2 className="text-3xl font-bold mb-4">Ready to Transform Your Team?</h2>
          <p className="text-xl text-gray-600 mb-8">
            Join the AI-powered development revolution. Analyze your first repository now.
          </p>
          <Link href="/dashboard" className="btn-primary text-lg px-8 py-4">
            <Brain className="h-5 w-5 mr-2" />
            Launch Dashboard
          </Link>
        </motion.div>
      </main>
    </div>
  )
}
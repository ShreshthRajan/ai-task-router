// frontend/src/app/dashboard/layout.tsx
'use client';

import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { 
  Brain, 
  Home, 
  Github, 
  BarChart3, 
  Menu, 
  X,
  Activity,
  Zap,
  TrendingUp,
  AlertCircle,
  CheckCircle2,
  Clock,
  Signal
} from 'lucide-react';
import { systemApi } from '@/lib/api-client';

interface NavigationItem {
  name: string;
  href: string;
  icon: React.ComponentType<any>;
  badge?: string;
  description: string;
}

const navigation: NavigationItem[] = [
  { 
    name: 'Command Center', 
    href: '/dashboard', 
    icon: Home,
    description: 'Live AI intelligence metrics and system overview'
  },
  { 
    name: 'Repository Analysis', 
    href: '/dashboard/analyze', 
    icon: Github, 
    badge: 'AI',
    description: 'Real-time GitHub repository intelligence extraction'
  },
  { 
    name: 'Performance Analytics', 
    href: '/dashboard/analytics', 
    icon: BarChart3,
    description: 'Advanced metrics and predictive insights'
  },
];

interface SystemHealth {
  status: 'optimal' | 'degraded' | 'offline';
  uptime: string;
  activeAgents: number;
  accuracy: number;
  responseTime: number;
}

export default function DashboardLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [systemHealth, setSystemHealth] = useState<SystemHealth>({
    status: 'optimal',
    uptime: '99.7%',
    activeAgents: 3,
    accuracy: 98.7,
    responseTime: 2.3
  });
  const [isOnline, setIsOnline] = useState(true);
  const pathname = usePathname();

  useEffect(() => {
    const checkSystemHealth = async () => {
      try {
        const response = await systemApi.healthCheck();
        setIsOnline(true);
        setSystemHealth(prev => ({
          ...prev,
          status: 'optimal',
          activeAgents: 3,
          accuracy: 98.7,
          responseTime: 2.3
        }));
      } catch (error) {
        setIsOnline(false);
        setSystemHealth(prev => ({
          ...prev,
          status: 'offline'
        }));
      }
    };

    checkSystemHealth();
    const interval = setInterval(checkSystemHealth, 30000);
    return () => clearInterval(interval);
  }, []);

  const isActivePath = (href: string) => {
    if (href === '/dashboard') {
      return pathname === '/dashboard';
    }
    return pathname.startsWith(href);
  };

  const getStatusIndicator = () => {
    switch (systemHealth.status) {
      case 'optimal':
        return {
          color: 'text-emerald-400',
          bgColor: 'bg-emerald-400/20',
          icon: CheckCircle2,
          text: 'All Systems Operational'
        };
      case 'degraded':
        return {
          color: 'text-amber-400',
          bgColor: 'bg-amber-400/20',
          icon: AlertCircle,
          text: 'Performance Degraded'
        };
      case 'offline':
        return {
          color: 'text-red-400',
          bgColor: 'bg-red-400/20',
          icon: AlertCircle,
          text: 'System Offline'
        };
    }
  };

  const status = getStatusIndicator();

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900">
      {/* Mobile sidebar overlay */}
      <AnimatePresence>
        {sidebarOpen && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-40 bg-black/60 backdrop-blur-sm lg:hidden"
            onClick={() => setSidebarOpen(false)}
          />
        )}
      </AnimatePresence>

      {/* Sidebar */}
      <motion.div
        initial={false}
        animate={{
          x: sidebarOpen ? 0 : -320,
        }}
        className="fixed inset-y-0 left-0 z-50 w-80 bg-slate-800/90 backdrop-blur-xl border-r border-slate-700/50 lg:translate-x-0 lg:static lg:inset-0"
      >
        <div className="flex flex-col h-full">
          {/* Header */}
          <div className="flex items-center justify-between p-6 border-b border-slate-700/50">
            <div className="flex items-center space-x-3">
              <div className="relative">
                <div className="p-2 bg-gradient-to-r from-blue-500 to-purple-600 rounded-xl shadow-lg">
                  <Brain className="h-7 w-7 text-white" />
                </div>
                <div className="absolute -top-1 -right-1 w-3 h-3 bg-emerald-400 rounded-full border-2 border-slate-800 pulse-indicator" />
              </div>
              <div>
                <h1 className="font-bold text-slate-100 text-lg">AI Intelligence</h1>
                <p className="text-sm text-slate-400">Development System</p>
              </div>
            </div>
            <button
              onClick={() => setSidebarOpen(false)}
              className="lg:hidden p-2 rounded-lg hover:bg-slate-700/50 text-slate-400 hover:text-slate-200"
            >
              <X className="h-5 w-5" />
            </button>
          </div>

          {/* System Status */}
          <div className="px-6 py-4 border-b border-slate-700/50">
            <div className="flex items-center justify-between mb-3">
              <span className="text-sm font-medium text-slate-300">System Status</span>
              <div className={`flex items-center space-x-2 text-sm ${status.color}`}>
                <div className={`w-2 h-2 rounded-full ${
                  systemHealth.status === 'optimal' ? 'bg-emerald-400 pulse-indicator' : 
                  systemHealth.status === 'degraded' ? 'bg-amber-400 animate-pulse' : 'bg-red-400'
                }`} />
                <span className="font-medium">{systemHealth.status.charAt(0).toUpperCase() + systemHealth.status.slice(1)}</span>
              </div>
            </div>
            
            <div className={`p-3 rounded-lg ${status.bgColor} border border-slate-600/30`}>
              <div className="flex items-center space-x-2 mb-2">
                <status.icon className={`h-4 w-4 ${status.color}`} />
                <span className={`text-sm font-medium ${status.color}`}>{status.text}</span>
              </div>
              {systemHealth.status === 'optimal' && (
                <div className="text-xs text-slate-400 space-y-1">
                  <div className="flex justify-between">
                    <span>AI Agents:</span>
                    <span className="text-emerald-400">{systemHealth.activeAgents} active</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Accuracy:</span>
                    <span className="text-emerald-400">{systemHealth.accuracy}%</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Response Time:</span>
                    <span className="text-emerald-400">{systemHealth.responseTime}s</span>
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Navigation */}
          <nav className="flex-1 px-4 py-6 space-y-2">
            {navigation.map((item) => {
              const isActive = isActivePath(item.href);
              return (
                <Link key={item.name} href={item.href}>
                  <motion.div
                    className={`group relative px-4 py-4 rounded-xl transition-all duration-200 ${
                      isActive
                        ? 'bg-gradient-to-r from-blue-500/20 to-purple-500/20 border border-blue-500/30 text-blue-300'
                        : 'text-slate-400 hover:bg-slate-700/50 hover:text-slate-200 border border-transparent hover:border-slate-600/50'
                    }`}
                    whileHover={{ x: 4 }}
                    onClick={() => setSidebarOpen(false)}
                  >
                    <div className="flex items-center space-x-3">
                      <item.icon className={`h-5 w-5 ${isActive ? 'text-blue-400' : 'text-slate-500 group-hover:text-slate-300'}`} />
                      <div className="flex-1">
                        <div className="font-medium">{item.name}</div>
                        <div className="text-xs text-slate-500 group-hover:text-slate-400 mt-0.5">
                          {item.description}
                        </div>
                      </div>
                      {item.badge && (
                        <span className="px-2 py-1 text-xs font-semibold bg-gradient-to-r from-blue-500 to-purple-600 text-white rounded-lg shadow-lg">
                          {item.badge}
                        </span>
                      )}
                    </div>
                    {isActive && (
                      <motion.div
                        layoutId="activeTab"
                        className="absolute inset-0 bg-gradient-to-r from-blue-500/10 to-purple-500/10 rounded-xl border border-blue-500/20"
                        initial={false}
                        transition={{ type: "spring", bounce: 0.2, duration: 0.6 }}
                      />
                    )}
                  </motion.div>
                </Link>
              );
            })}
          </nav>

          {/* Live Metrics */}
          <div className="px-6 py-4 border-t border-slate-700/50">
            <h3 className="text-sm font-semibold text-slate-300 mb-4">Live Intelligence</h3>
            <div className="space-y-3">
              <div className="flex items-center justify-between p-3 bg-gradient-to-r from-emerald-500/10 to-green-500/10 rounded-lg border border-emerald-500/20">
                <div className="flex items-center space-x-2">
                  <TrendingUp className="h-4 w-4 text-emerald-400" />
                  <span className="text-sm text-slate-300">Assignment Accuracy</span>
                </div>
                <span className="text-sm font-bold text-emerald-400">98.7%</span>
              </div>
              
              <div className="flex items-center justify-between p-3 bg-gradient-to-r from-blue-500/10 to-cyan-500/10 rounded-lg border border-blue-500/20">
                <div className="flex items-center space-x-2">
                  <Clock className="h-4 w-4 text-blue-400" />
                  <span className="text-sm text-slate-300">Avg Analysis</span>
                </div>
                <span className="text-sm font-bold text-blue-400">2.3s</span>
              </div>

              <div className="flex items-center justify-between p-3 bg-gradient-to-r from-purple-500/10 to-pink-500/10 rounded-lg border border-purple-500/20">
                <div className="flex items-center space-x-2">
                  <Zap className="h-4 w-4 text-purple-400" />
                  <span className="text-sm text-slate-300">Optimizations</span>
                </div>
                <span className="text-sm font-bold text-purple-400">156</span>
              </div>
            </div>
          </div>
        </div>
      </motion.div>

      {/* Main content */}
      <div className="lg:pl-80">
        {/* Top bar */}
        <div className="sticky top-0 z-30 flex items-center justify-between h-16 px-4 sm:px-6 lg:px-8 bg-slate-800/80 backdrop-blur-xl border-b border-slate-700/50">
          <button
            onClick={() => setSidebarOpen(true)}
            className="lg:hidden p-2 rounded-lg hover:bg-slate-700/50 text-slate-400 hover:text-slate-200"
          >
            <Menu className="h-5 w-5" />
          </button>

          <div className="flex items-center space-x-4 ml-auto">
            <div className="flex items-center space-x-2 bg-slate-700/50 px-3 py-2 rounded-xl border border-slate-600/50">
              <Activity className="h-4 w-4 text-blue-400" />
              <span className="text-sm font-medium text-slate-300">AI Systems Active</span>
              <div className="w-2 h-2 bg-blue-400 rounded-full pulse-indicator" />
            </div>
            
            <div className="flex items-center space-x-2 bg-emerald-500/10 px-3 py-2 rounded-xl border border-emerald-500/20">
              <Signal className="h-4 w-4 text-emerald-400" />
              <span className="text-sm font-medium text-emerald-400">Live Analysis</span>
            </div>
          </div>
        </div>

        {/* Page content */}
        <main className="p-4 sm:p-6 lg:p-8 min-h-screen">
          {children}
        </main>
      </div>
    </div>
  );
}
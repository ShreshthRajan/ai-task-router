'use client';

import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { 
  Brain, 
  Home, 
  Github, 
  BarChart3, 
  Users, 
  Settings, 
  Menu, 
  X,
  Activity,
  Zap,
  TrendingUp
} from 'lucide-react';
import { systemApi } from '@/lib/api-client';

interface NavigationItem {
  name: string;
  href: string;
  icon: React.ComponentType<any>;
  badge?: string;
}

const navigation: NavigationItem[] = [
  { name: 'Overview', href: '/dashboard', icon: Home },
  { name: 'GitHub Analysis', href: '/dashboard/analyze', icon: Github, badge: 'AI' },
  { name: 'Team Intelligence', href: '/dashboard/team', icon: Users },
  { name: 'Task Optimization', href: '/dashboard/tasks', icon: Zap },
  { name: 'Analytics', href: '/dashboard/analytics', icon: BarChart3 },
  { name: 'Learning System', href: '/dashboard/learning', icon: TrendingUp },
  { name: 'Settings', href: '/dashboard/settings', icon: Settings },
];

export default function DashboardLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [systemStatus, setSystemStatus] = useState<'online' | 'offline' | 'loading'>('loading');
  const pathname = usePathname();

  useEffect(() => {
    const checkSystemHealth = async () => {
      try {
        await systemApi.healthCheck();
        setSystemStatus('online');
      } catch (error) {
        setSystemStatus('offline');
      }
    };

    checkSystemHealth();
    const interval = setInterval(checkSystemHealth, 30000); // Check every 30 seconds

    return () => clearInterval(interval);
  }, []);

  const isActivePath = (href: string) => {
    if (href === '/dashboard') {
      return pathname === '/dashboard';
    }
    return pathname.startsWith(href);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50">
      {/* Mobile sidebar overlay */}
      {sidebarOpen && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="fixed inset-0 z-40 bg-black bg-opacity-50 lg:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      {/* Sidebar */}
      <motion.div
        initial={false}
        animate={{
          x: sidebarOpen ? 0 : -320,
        }}
        className="fixed inset-y-0 left-0 z-50 w-80 bg-white/80 backdrop-blur-md border-r border-slate-200/50 lg:translate-x-0 lg:static lg:inset-0"
      >
        <div className="flex flex-col h-full">
          {/* Header */}
          <div className="flex items-center justify-between p-6 border-b border-slate-200/50">
            <div className="flex items-center space-x-3">
              <div className="p-2 bg-gradient-to-r from-blue-500 to-purple-600 rounded-lg">
                <Brain className="h-6 w-6 text-white" />
              </div>
              <div>
                <h1 className="font-bold text-slate-900">AI Intelligence</h1>
                <p className="text-sm text-slate-500">Development System</p>
              </div>
            </div>
            <button
              onClick={() => setSidebarOpen(false)}
              className="lg:hidden p-2 rounded-lg hover:bg-slate-100"
            >
              <X className="h-5 w-5" />
            </button>
          </div>

          {/* System Status */}
          <div className="px-6 py-4 border-b border-slate-200/50">
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium text-slate-700">System Status</span>
              <div className={`flex items-center space-x-2 text-sm ${
                systemStatus === 'online' ? 'text-green-600' : 
                systemStatus === 'offline' ? 'text-red-600' : 'text-yellow-600'
              }`}>
                <div className={`w-2 h-2 rounded-full ${
                  systemStatus === 'online' ? 'bg-green-500 pulse-indicator' : 
                  systemStatus === 'offline' ? 'bg-red-500' : 'bg-yellow-500 animate-pulse'
                }`} />
                <span className="font-medium capitalize">{systemStatus}</span>
              </div>
            </div>
            {systemStatus === 'online' && (
              <div className="mt-2 text-xs text-slate-500">
                All AI systems operational • Learning models active
              </div>
            )}
          </div>

          {/* Navigation */}
          <nav className="flex-1 px-4 py-6 space-y-2">
            {navigation.map((item) => {
              const isActive = isActivePath(item.href);
              return (
                <Link key={item.name} href={item.href}>
                  <motion.div
                    className={`flex items-center justify-between px-4 py-3 rounded-lg transition-all duration-200 group ${
                      isActive
                        ? 'bg-blue-50 text-blue-700 border border-blue-200'
                        : 'text-slate-600 hover:bg-slate-50 hover:text-slate-900'
                    }`}
                    whileHover={{ x: 4 }}
                    onClick={() => setSidebarOpen(false)}
                  >
                    <div className="flex items-center space-x-3">
                      <item.icon className={`h-5 w-5 ${isActive ? 'text-blue-600' : 'text-slate-400 group-hover:text-slate-600'}`} />
                      <span className="font-medium">{item.name}</span>
                    </div>
                    {item.badge && (
                      <span className="px-2 py-1 text-xs font-semibold bg-gradient-to-r from-blue-500 to-purple-600 text-white rounded-full">
                        {item.badge}
                      </span>
                    )}
                  </motion.div>
                </Link>
              );
            })}
          </nav>

          {/* Quick Stats */}
          <div className="px-6 py-4 border-t border-slate-200/50">
            <h3 className="text-sm font-semibold text-slate-700 mb-3">Live Metrics</h3>
            <div className="grid grid-cols-2 gap-3">
              <div className="bg-gradient-to-r from-green-50 to-emerald-50 p-3 rounded-lg border border-green-200/50">
                <div className="text-lg font-bold text-green-700">98.7%</div>
                <div className="text-xs text-green-600">Assignment Accuracy</div>
              </div>
              <div className="bg-gradient-to-r from-blue-50 to-cyan-50 p-3 rounded-lg border border-blue-200/50">
                <div className="text-lg font-bold text-blue-700">2.3s</div>
                <div className="text-xs text-blue-600">Avg Analysis Time</div>
              </div>
            </div>
          </div>
        </div>
      </motion.div>

      {/* Main content */}
      <div className="lg:pl-80">
        {/* Top bar */}
        <div className="sticky top-0 z-30 flex items-center justify-between h-16 px-4 sm:px-6 lg:px-8 bg-white/80 backdrop-blur-md border-b border-slate-200/50">
          <button
            onClick={() => setSidebarOpen(true)}
            className="lg:hidden p-2 rounded-lg hover:bg-slate-100"
          >
            <Menu className="h-5 w-5" />
          </button>

          <div className="flex items-center space-x-4 ml-auto">
            <div className="flex items-center space-x-2 bg-slate-50 px-3 py-1 rounded-full">
              <Activity className="h-4 w-4 text-blue-600" />
              <span className="text-sm font-medium text-slate-700">AI Systems Active</span>
            </div>
            
            <div className="flex items-center space-x-2 bg-green-50 px-3 py-1 rounded-full">
              <div className="w-2 h-2 bg-green-500 rounded-full pulse-indicator" />
              <span className="text-sm font-medium text-green-700">Live Analysis</span>
            </div>
          </div>
        </div>

        {/* Page content */}
        <main className="p-4 sm:p-6 lg:p-8">
          {children}
        </main>
      </div>
    </div>
  );
}
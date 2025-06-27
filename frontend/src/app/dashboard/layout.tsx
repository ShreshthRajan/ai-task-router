// frontend/src/app/dashboard/layout.tsx
'use client';

import { useState, useEffect } from 'react';
import clsx from 'clsx';      
import { motion, AnimatePresence } from 'framer-motion';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { 
  Brain, 
  Home, 
  Github, 
  Menu, 
  X,
  Activity,
  Zap,
  TrendingUp,
  CheckCircle2,
  Signal,
  Settings,
  HelpCircle
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
    name: 'Overview', 
    href: '/dashboard', 
    icon: Home,
    description: 'System metrics and AI insights'
  },
  { 
    name: 'Repository Analysis', 
    href: '/dashboard/analyze', 
    icon: Github, 
    badge: 'AI',
    description: 'Live GitHub intelligence extraction'
  },
];

interface SystemStatus {
  status: 'optimal' | 'degraded' | 'offline';
  activeModels: number;
  accuracy: number;
  responseTime: number;
  uptime: string;
}

export default function DashboardLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const [sidebarOpen, setSidebarOpen] = useState(false);

  const [isDesktop, setIsDesktop] = useState(false);
  useEffect(() => {
    /* media query listener keeps state in sync when user resizes window */
    const mql = window.matchMedia('(min-width:1024px)');
    const handler = () => setIsDesktop(mql.matches);
    handler();                       // initialise on first render
    mql.addEventListener('change', handler);
    return () => mql.removeEventListener('change', handler);
  }, []);
  const [systemStatus, setSystemStatus] = useState<SystemStatus>({
    status: 'optimal',
    activeModels: 4,
    accuracy: 98.7,
    responseTime: 2.3,
    uptime: '99.8%'
  });
  const [isOnline, setIsOnline] = useState(true);
  const pathname = usePathname();

  useEffect(() => {
    const checkSystemHealth = async () => {
      try {
        const response = await systemApi.healthCheck();
        setIsOnline(true);
        setSystemStatus(prev => ({
          ...prev,
          status: 'optimal'
        }));
      } catch (error) {
        setIsOnline(false);
        setSystemStatus(prev => ({
          ...prev,
          status: 'offline'
        }));
      }
    };

    checkSystemHealth();
    const interval = setInterval(checkSystemHealth, 60000);
    return () => clearInterval(interval);
  }, []);

  const isActivePath = (href: string) => {
    if (href === '/dashboard') {
      return pathname === '/dashboard';
    }
    return pathname.startsWith(href);
  };

  const getStatusConfig = () => {
    switch (systemStatus.status) {
      case 'optimal':
        return {
          color: 'text-emerald-600 dark:text-emerald-400',
          bgColor: 'bg-emerald-50 dark:bg-emerald-900/20',
          borderColor: 'border-emerald-200 dark:border-emerald-800',
          icon: CheckCircle2,
          text: 'All Systems Operational',
          dotClass: 'bg-emerald-500 animate-pulse'
        };
      case 'degraded':
        return {
          color: 'text-amber-600 dark:text-amber-400',
          bgColor: 'bg-amber-50 dark:bg-amber-900/20',
          borderColor: 'border-amber-200 dark:border-amber-800',
          icon: TrendingUp,
          text: 'Performance Degraded',
          dotClass: 'bg-amber-500 animate-pulse'
        };
      case 'offline':
        return {
          color: 'text-red-600 dark:text-red-400',
          bgColor: 'bg-red-50 dark:bg-red-900/20',
          borderColor: 'border-red-200 dark:border-red-800',
          icon: X,
          text: 'System Offline',
          dotClass: 'bg-red-500'
        };
    }
  };

  const statusConfig = getStatusConfig();

  return (
    <div className="min-h-screen bg-[#242422] dark">
      {/* Mobile sidebar overlay */}
      <AnimatePresence>
        {sidebarOpen && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-40 bg-[#242422]/60 backdrop-blur-sm lg:hidden"
            onClick={() => setSidebarOpen(false)}
          />
        )}
      </AnimatePresence>

      {/* Fixed Sidebar */}
      <motion.div
        /* 2️⃣  Animate only on mobile (< lg); lock at x = 0 on desktop */
        animate={isDesktop ? { x: 0 } : { x: sidebarOpen ? 0 : -320 }}
        initial={false}
        transition={{ type: 'tween', duration: 0.28 }}

        /* 3️⃣  Tailwind-powered fall-back transform so it still slides without JS during first paint */
        className={clsx(
          'fixed inset-y-0 left-0 z-50 w-80 bg-[#2a2a28] border-r border-[#404040]',
          'transform transition-transform duration-300',
          isDesktop              /* ≥ lg */
            ? 'translate-x-0'
            : sidebarOpen
              ? 'translate-x-0'   /* mobile + drawer open */
              : '-translate-x-full'/* mobile + drawer closed */
        )}
      >
        <div className="flex flex-col h-full">
          {/* Header */}
          <div className="flex items-center justify-between p-6 border-b border-[#404040]">
            <div className="flex items-center space-x-3">
              <div className="relative">
                <div className="p-2 bg-gradient-to-r from-blue-500 to-purple-600 rounded-xl shadow-lg">
                  <Brain className="h-6 w-6 text-white" />
                </div>
                <div className={`absolute -top-1 -right-1 w-3 h-3 ${statusConfig.dotClass} rounded-full border-2 border-[#2a2a28]`} />
              </div>
              <div>
                <h1 className="font-bold text-[#f4f4f4] text-lg">Development Intelligence</h1>
                <p className="text-sm text-[#a0a0a0]">AI Task Assignment</p>
              </div>
            </div>
            <button
              onClick={() => setSidebarOpen(false)}
              className="lg:hidden p-2 rounded-lg hover:bg-[#404040] text-[#a0a0a0]"
            >
              <X className="h-5 w-5" />
            </button>
          </div>

          {/* System Status */}
          <div className="p-6 border-b border-[#404040]">
            <div className="flex items-center justify-between mb-3">
              <span className="text-sm font-semibold text-[#f4f4f4]">System Status</span>
              <div className={`flex items-center space-x-2 text-sm ${statusConfig.color}`}>
                <div className={`w-2 h-2 rounded-full ${statusConfig.dotClass}`} />
                <span className="font-medium capitalize">{systemStatus.status}</span>
              </div>
            </div>
            
            <div className={`p-4 rounded-xl bg-[#404040]/30 border border-[#404040]`}>
              <div className="flex items-center space-x-2 mb-3">
                <statusConfig.icon className={`h-4 w-4 ${statusConfig.color}`} />
                <span className={`text-sm font-semibold ${statusConfig.color}`}>{statusConfig.text}</span>
              </div>
              {systemStatus.status === 'optimal' && (
                <div className="space-y-2 text-xs text-[#a0a0a0]">
                  <div className="flex justify-between">
                    <span>AI Models:</span>
                    <span className="font-medium text-[#f4f4f4]">{systemStatus.activeModels}/4 active</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Accuracy:</span>
                    <span className="font-medium text-[#f4f4f4]">{systemStatus.accuracy}%</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Response:</span>
                    <span className="font-medium text-[#f4f4f4]">{systemStatus.responseTime}s avg</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Uptime:</span>
                    <span className="font-medium text-[#f4f4f4]">{systemStatus.uptime}</span>
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
                        ? 'bg-blue-900/30 border border-blue-800 text-blue-300'
                        : 'text-[#a0a0a0] hover:bg-[#404040] hover:text-[#f4f4f4] border border-transparent hover:border-[#404040]'
                    }`}
                    whileHover={{ x: isActive ? 0 : 4 }}
                    onClick={() => setSidebarOpen(false)}
                  >
                    <div className="flex items-center space-x-3">
                      <item.icon className={`h-5 w-5 ${isActive ? 'text-blue-400' : 'text-[#a0a0a0] group-hover:text-[#f4f4f4]'}`} />
                      <div className="flex-1">
                        <div className="font-semibold">{item.name}</div>
                        <div className="text-xs text-[#a0a0a0] group-hover:text-[#a0a0a0] mt-0.5">
                          {item.description}
                        </div>
                      </div>
                      {item.badge && (
                        <span className="px-2 py-1 text-xs font-semibold bg-gradient-to-r from-blue-500 to-purple-600 text-white rounded-lg shadow-sm">
                          {item.badge}
                        </span>
                      )}
                    </div>
                  </motion.div>
                </Link>
              );
            })}
          </nav>

          {/* Live Metrics */}
          <div className="px-6 py-4 border-t border-[#404040]">
            <h3 className="text-sm font-semibold text-[#f4f4f4] mb-4">Live Intelligence</h3>
            <div className="space-y-3">
              <div className="flex items-center justify-between p-3 bg-emerald-900/20 rounded-lg border border-emerald-800">
                <div className="flex items-center space-x-2">
                  <TrendingUp className="h-4 w-4 text-emerald-400" />
                  <span className="text-sm font-medium text-emerald-300">Accuracy</span>
                </div>
                <span className="text-sm font-bold text-emerald-200">{systemStatus.accuracy}%</span>
              </div>
              
              <div className="flex items-center justify-between p-3 bg-blue-900/20 rounded-lg border border-blue-800">
                <div className="flex items-center space-x-2">
                  <Activity className="h-4 w-4 text-blue-400" />
                  <span className="text-sm font-medium text-blue-300">Response</span>
                </div>
                <span className="text-sm font-bold text-blue-200">{systemStatus.responseTime}s</span>
              </div>

              <div className="flex items-center justify-between p-3 bg-purple-900/20 rounded-lg border border-purple-800">
                <div className="flex items-center space-x-2">
                  <Zap className="h-4 w-4 text-purple-400" />
                  <span className="text-sm font-medium text-purple-300">Models</span>
                </div>
                <span className="text-sm font-bold text-purple-200">{systemStatus.activeModels}/4</span>
              </div>
            </div>
          </div>

          {/* Footer */}
          <div className="px-6 py-4 border-t border-[#404040]">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-3">
                <button className="p-2 rounded-lg hover:bg-[#404040] text-[#a0a0a0] hover:text-[#f4f4f4]">
                  <Settings className="h-4 w-4" />
                </button>
                <button className="p-2 rounded-lg hover:bg-[#404040] text-[#a0a0a0] hover:text-[#f4f4f4]">
                  <HelpCircle className="h-4 w-4" />
                </button>
              </div>
              <div className="text-xs text-[#a0a0a0]">
                v2.1.0
              </div>
            </div>
          </div>
        </div>
      </motion.div>

      {/* Main content area - FIXED STRUCTURE */}
      <div
        key={pathname}                       // ✨ forces remount per page
        className="lg:pl-80 flex flex-col flex-1"
      >
        {/* Fixed Top bar */}
        <div className="sticky top-0 z-30 flex items-center justify-between h-16 px-6 bg-[#242422]/80 backdrop-blur-xl border-b border-[#404040]">
          <button
            onClick={() => setSidebarOpen(true)}
            className="lg:hidden p-2 rounded-lg hover:bg-[#404040] text-[#a0a0a0]"
          >
            <Menu className="h-5 w-5" />
          </button>

          <div className="flex items-center space-x-4 ml-auto">
            <div className="flex items-center space-x-2 bg-[#404040] px-3 py-2 rounded-lg">
              <Signal className="h-4 w-4 text-blue-400" />
              <span className="text-sm font-medium text-[#f4f4f4]">AI Systems</span>
              <div className={`w-2 h-2 rounded-full ${statusConfig.dotClass}`} />
            </div>
          </div>
        </div>
        {/* Page content */}
        <main className="flex-1 bg-[#242422]">
          {children}
        </main>
      </div>
    </div>
  );
}
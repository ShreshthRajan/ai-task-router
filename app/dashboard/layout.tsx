// app/dashboard/layout.tsx
'use client';

import React from 'react';
import { useState } from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { 
  Brain, Home, Github, Users, Target, TrendingUp, 
  Settings, Menu, X, Activity 
} from 'lucide-react';

const navigation = [
  { name: 'Overview', href: '/dashboard', icon: Home },
  { name: 'Live Analysis', href: '/dashboard/analyze', icon: Github },
  { name: 'Team Intelligence', href: '/dashboard/team', icon: Users },
  { name: 'Task Optimization', href: '/dashboard/tasks', icon: Target },
  { name: 'Performance', href: '/dashboard/performance', icon: TrendingUp },
  { name: 'Settings', href: '/dashboard/settings', icon: Settings },
];

export default function DashboardLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const pathname = usePathname();

  return (
    <div className="min-h-screen bg-slate-900">
      {/* Mobile sidebar */}
      <div className={`fixed inset-0 flex z-40 md:hidden ${sidebarOpen ? '' : 'hidden'}`}>
        <div className="fixed inset-0 bg-gray-600 bg-opacity-75" onClick={() => setSidebarOpen(false)} />
        <div className="relative flex-1 flex flex-col max-w-xs w-full bg-slate-800 border-r border-slate-700">
          <div className="absolute top-0 right-0 -mr-12 pt-2">
            <button
              className="ml-1 flex items-center justify-center h-10 w-10 rounded-full focus:outline-none focus:ring-2 focus:ring-inset focus:ring-blue-500"
              onClick={() => setSidebarOpen(false)}
            >
              <X className="h-6 w-6 text-white" />
            </button>
          </div>
          <SidebarContent />
        </div>
      </div>

      {/* Static sidebar for desktop */}
      <div className="hidden md:flex md:w-64 md:flex-col md:fixed md:inset-y-0">
        <SidebarContent />
      </div>

      {/* Main content */}
      <div className="md:pl-64 flex flex-col min-h-screen">
        {/* Mobile menu button */}
        <div className="sticky top-0 z-10 md:hidden bg-slate-800 border-b border-slate-700">
          <div className="flex items-center justify-between h-16 px-4">
            <button
              className="inline-flex items-center justify-center p-2 rounded-md text-slate-400 hover:text-slate-200 hover:bg-slate-700 focus:outline-none focus:ring-2 focus:ring-inset focus:ring-blue-500"
              onClick={() => setSidebarOpen(true)}
            >
              <Menu className="h-6 w-6" />
            </button>
            <div className="flex items-center space-x-2">
              <Brain className="h-6 w-6 text-blue-400" />
              <span className="text-lg font-semibold text-slate-100">AI Intelligence</span>
            </div>
          </div>
        </div>

        {/* Page content */}
        <main className="flex-1 bg-slate-900">
          {children}
        </main>
      </div>
    </div>
  );

  function SidebarContent() {
    return (
      <div className="flex-1 flex flex-col bg-slate-800 border-r border-slate-700">
        <div className="flex-1 flex flex-col pt-5 pb-4 overflow-y-auto">
          {/* Logo */}
          <div className="flex items-center flex-shrink-0 px-4 mb-8">
            <div className="relative">
              <div className="p-2 bg-gradient-to-r from-blue-500 to-purple-600 rounded-xl shadow-lg">
                <Brain className="h-6 w-6 text-white" />
              </div>
              <div className="absolute -top-1 -right-1 w-3 h-3 bg-emerald-400 rounded-full border-2 border-slate-800" />
            </div>
            <div className="ml-3">
              <span className="text-lg font-bold text-slate-100">Development Intelligence</span>
              <div className="text-xs text-slate-400">AI Task Assignment System</div>
            </div>
          </div>

          {/* Navigation */}
          <nav className="flex-1 px-2 space-y-1">
            {navigation.map((item) => {
              const isActive = pathname === item.href;
              return (
                <Link
                  key={item.name}
                  href={item.href}
                  className={`group flex items-center px-3 py-2.5 text-sm font-medium rounded-xl transition-all duration-200 ${
                    isActive
                      ? 'bg-gradient-to-r from-blue-500 to-purple-600 text-white shadow-lg'
                      : 'text-slate-400 hover:text-slate-200 hover:bg-slate-700/50'
                  }`}
                >
                  <item.icon
                    className={`mr-3 flex-shrink-0 h-5 w-5 ${
                      isActive ? 'text-white' : 'text-slate-500 group-hover:text-slate-300'
                    }`}
                  />
                  {item.name}
                </Link>
              );
            })}
          </nav>
        </div>
        
        {/* System Status */}
        <div className="flex-shrink-0 border-t border-slate-700 p-4">
          <div className="flex items-center space-x-3">
            <div className="flex-shrink-0">
              <div className="relative">
                <Activity className="h-5 w-5 text-emerald-400" />
                <div className="absolute -top-1 -right-1 w-2 h-2 bg-emerald-400 rounded-full animate-pulse" />
              </div>
            </div>
            <div className="flex-1 min-w-0">
              <p className="text-sm font-medium text-slate-200">System Optimal</p>
              <p className="text-xs text-slate-500">All AI models operational</p>
            </div>
          </div>
        </div>
      </div>
    );
  }
}
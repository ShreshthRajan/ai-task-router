#!/usr/bin/env python3
"""
AI Task Router Dashboard Launcher

This script sets up and launches the revolutionary AI Development Intelligence Dashboard.
"""

import os
import sys
import subprocess
import asyncio
from pathlib import Path

def check_dependencies():
    """Check if all required dependencies are installed."""
    required_packages = [
        'streamlit',
        'plotly', 
        'pandas',
        'numpy',
        'sqlalchemy',
        'fastapi'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        print("Installing missing packages...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install"
        ] + missing_packages)
        print("✅ Dependencies installed!")
    else:
        print("✅ All dependencies satisfied!")

def setup_environment():
    """Set up environment variables and paths."""
    # Add project root and src to Python path
    project_root = Path(__file__).parent
    src_path = project_root / "src"
    
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    
    # Set environment variables if not already set
    env_vars = {
        "DATABASE_URL": f"sqlite:///{project_root}/data/taskrouter.db",
        "DEBUG": "True"
    }
    
    for var, default_value in env_vars.items():
        if var not in os.environ:
            os.environ[var] = default_value
    
    print(f"✅ Environment configured with paths: {project_root}, {src_path}")

def create_data_directory():
    """Create data directory if it doesn't exist."""
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    print(f"✅ Data directory ready: {data_dir.absolute()}")

def launch_dashboard():
    """Launch the Streamlit dashboard."""
    dashboard_path = Path("src/dashboard/app.py")
    
    if not dashboard_path.exists():
        print(f"❌ Dashboard app not found at {dashboard_path}")
        return False
    
    print("🚀 Launching AI Development Intelligence Dashboard...")
    print("📊 Opening in your default browser...")
    print("🧠 Prepare to be amazed by AI-powered development intelligence!")
    print()
    print("=" * 60)
    print("  AI TASK ROUTER - DEVELOPMENT INTELLIGENCE DASHBOARD")
    print("  Revolutionary AI system for optimal task assignment")
    print("=" * 60)
    print()
    
    # Launch Streamlit
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            str(dashboard_path),
            "--server.port", "8501",
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped. Thanks for using AI Task Router!")
    except Exception as e:
        print(f"❌ Error launching dashboard: {e}")
        return False
    
    return True

def main():
    """Main launcher function."""
    print("🧠 AI Task Router Dashboard Launcher")
    print("====================================")
    print()
    
    # Setup steps
    print("1. Checking dependencies...")
    check_dependencies()
    
    print("\n2. Setting up environment...")
    setup_environment()
    
    print("\n3. Creating data directory...")
    create_data_directory()
    
    print("\n4. Launching dashboard...")
    success = launch_dashboard()
    
    if not success:
        print("\n❌ Failed to launch dashboard. Please check the error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
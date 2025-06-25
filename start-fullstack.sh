# start-fullstack.sh
#!/bin/bash

# AI Task Router - Full Stack Startup Script
echo "🚀 Starting AI Development Intelligence System (Full Stack)"
echo "=========================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_highlight() {
    echo -e "${PURPLE}🎯 $1${NC}"
}

# Function to cleanup on exit
cleanup() {
    echo ""
    print_info "Shutting down all servers..."
    
    # Kill backend process
    if [ ! -z "$BACKEND_PID" ]; then
        kill $BACKEND_PID 2>/dev/null
    fi
    
    # Kill frontend process
    if [ ! -z "$FRONTEND_PID" ]; then
        kill $FRONTEND_PID 2>/dev/null
    fi
    
    # Kill any remaining processes
    pkill -f "uvicorn.*main:app" 2>/dev/null
    pkill -f "next dev" 2>/dev/null
    
    print_status "All servers stopped"
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Check prerequisites
print_info "Checking prerequisites..."

if ! command -v node &> /dev/null; then
    print_error "Node.js is not installed. Please install Node.js 16+ first."
    exit 1
fi

if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed. Please install Python 3.8+ first."
    exit 1
fi

# Check if we're in the right directory (look for requirements.txt instead)
if [ ! -f "requirements.txt" ]; then
    print_error "Please run this script from the project root directory (requirements.txt not found)."
    exit 1
fi

print_status "Prerequisites check passed"

# Backend Setup
print_info "Setting up backend..."

# Check if Python virtual environment exists
if [ ! -d "venv" ]; then
    print_info "Creating Python virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
print_info "Activating Python virtual environment..."
source venv/bin/activate

# Install Python dependencies
print_info "Installing backend dependencies..."
pip install -r requirements.txt > /dev/null 2>&1

if [ $? -ne 0 ]; then
    print_error "Backend dependency installation failed"
    exit 1
fi

print_status "Backend dependencies installed"

# Setup database
print_info "Setting up database..."
python3 scripts/setup.py > /dev/null 2>&1

print_status "Database setup completed"

# Frontend Setup
print_info "Setting up frontend..."

# Check if frontend directory exists
if [ ! -d "frontend" ]; then
    print_error "Frontend directory not found. Please ensure all frontend files are in place."
    exit 1
fi

cd frontend

# Install frontend dependencies
print_info "Installing frontend dependencies..."
npm install > /dev/null 2>&1

if [ $? -ne 0 ]; then
    print_error "Frontend dependency installation failed"
    cd ..
    exit 1
fi

print_status "Frontend dependencies installed"
cd ..

# Start Backend Server
print_info "Starting backend server on port 8000..."
cd src
python3 -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload > /dev/null 2>&1 &
BACKEND_PID=$!
cd ..

# Wait for backend to start
sleep 3

# Check if backend is running
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    print_status "Backend server started successfully"
else
    print_error "Backend server failed to start"
    kill $BACKEND_PID 2>/dev/null
    exit 1
fi

# Start Frontend Server
print_info "Starting frontend server on port 3000..."
cd frontend
npm run dev > /dev/null 2>&1 &
FRONTEND_PID=$!
cd ..

# Wait for frontend to start
sleep 5

# Check if frontend is running
if curl -s http://localhost:3000 > /dev/null 2>&1; then
    print_status "Frontend server started successfully"
else
    print_warning "Frontend server may still be starting..."
fi

# Success banner
echo ""
echo "🎉 AI Development Intelligence System is now LIVE!"
echo "===================================================="
echo ""
print_highlight "🌟 Revolutionary AI-Powered Task Assignment System"
print_highlight "🧠 768-Dimensional Developer Skill Modeling"
print_highlight "🎯 5-Dimensional Task Complexity Prediction"
print_highlight "⚡ Multi-Objective Assignment Optimization"
print_highlight "📊 Continuous Learning & Adaptation"
echo ""
echo "📱 Access Points:"
echo "┌─────────────────────────────────────────────────────┐"
echo "│  🎨 Frontend App: http://localhost:3000            │"
echo "│  🔧 Backend API:  http://localhost:8000            │"
echo "│  📖 API Docs:     http://localhost:8000/docs       │"
echo "│  ❤️  Health:      http://localhost:8000/health     │"
echo "└─────────────────────────────────────────────────────┘"
echo ""
print_info "🚀 Ready to analyze GitHub repositories!"
print_info "💡 Paste any GitHub URL to see AI-powered team intelligence"
print_info "🎯 Experience revolutionary task assignment optimization"
echo ""
print_warning "Press Ctrl+C to stop all servers"
echo ""

# Keep script running
while true; do
    sleep 1
done
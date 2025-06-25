#!/bin/bash

# AI Task Router - Production Startup Script
echo "🧠 Starting AI Development Intelligence System..."
echo "=================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
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

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    print_error "Node.js is not installed. Please install Node.js 18+ first."
    exit 1
fi

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed. Please install Python 3.8+ first."
    exit 1
fi

# Check if we're in the right directory
if [ ! -f "package.json" ]; then
    print_error "package.json not found. Please run this script from the project root."
    exit 1
fi

print_info "Installing frontend dependencies..."
npm install

if [ $? -ne 0 ]; then
    print_error "Frontend dependency installation failed"
    exit 1
fi

print_status "Frontend dependencies installed"

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
pip install -r requirements.txt

if [ $? -ne 0 ]; then
    print_error "Backend dependency installation failed"
    exit 1
fi

print_status "Backend dependencies installed"

# Setup database
print_info "Setting up database..."
python3 scripts/setup.py

if [ $? -ne 0 ]; then
    print_warning "Database setup had issues but continuing..."
fi

print_status "Database setup completed"

# Start backend server in background
print_info "Starting backend server on port 8000..."
cd src && python3 -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!

# Wait for backend to start
sleep 3

# Check if backend is running
if curl -s http://localhost:8000/docs > /dev/null; then
    print_status "Backend server started successfully"
else
    print_error "Backend server failed to start"
    kill $BACKEND_PID 2>/dev/null
    exit 1
fi

# Start frontend server
print_info "Starting frontend development server on port 3000..."
npm run dev &
FRONTEND_PID=$!

# Wait for frontend to start
sleep 5

# Check if frontend is running
if curl -s http://localhost:3000 > /dev/null; then
    print_status "Frontend server started successfully"
else
    print_error "Frontend server failed to start"
    kill $BACKEND_PID $FRONTEND_PID 2>/dev/null
    exit 1
fi

echo ""
echo "🚀 AI Development Intelligence System is now running!"
echo "=================================================="
echo ""
print_status "Frontend: http://localhost:3000"
print_status "Backend API: http://localhost:8000"
print_status "API Documentation: http://localhost:8000/api/docs"
echo ""
print_info "The system is ready for live GitHub repository analysis!"
print_info "Press Ctrl+C to stop all servers"
echo ""

# Function to cleanup on exit
cleanup() {
    echo ""
    print_info "Shutting down servers..."
    kill $BACKEND_PID $FRONTEND_PID 2>/dev/null
    print_status "All servers stopped"
    exit 0
}

# Trap Ctrl+C
trap cleanup SIGINT

# Wait for user to stop
wait
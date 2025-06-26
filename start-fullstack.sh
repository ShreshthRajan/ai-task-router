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

    pkill -f "uvicorn.*main:app" 2>/dev/null
    pkill -f "next dev" 2>/dev/null

    print_status "All servers stopped"
    exit 0
}

trap cleanup SIGINT SIGTERM

# Enhanced Node.js version management
setup_node() {
    print_info "Setting up Node.js environment..."
    
    # Try to use NVM if available
    export NVM_DIR="$HOME/.nvm"
    if [ -s "$NVM_DIR/nvm.sh" ]; then
        print_info "Loading NVM..."
        \. "$NVM_DIR/nvm.sh"
        \. "$NVM_DIR/bash_completion" 2>/dev/null
        
        # Check if Node 18 is installed via NVM
        if nvm ls 18 &>/dev/null; then
            print_info "Using Node 18 via NVM..."
            nvm use 18 &>/dev/null
        else
            print_warning "Node 18 not found in NVM, attempting to install..."
            nvm install 18 &>/dev/null
            nvm use 18 &>/dev/null
        fi
    fi
    
    # Verify Node version
    NODE_BIN="$(command -v node)"

    NPM_BIN="$(dirname "$NODE_BIN")/npm"

    if [ ! -x "$NPM_BIN" ]; then
        print_error "npm for Node $NODE_VERSION not found (looked in $(dirname "$NODE_BIN"))."
        exit 1
    fi
    print_status "Using npm $($NPM_BIN --version) at $NPM_BIN"
    
    if [ -z "$NODE_BIN" ]; then
        print_error "Node.js is not installed. Please install Node.js 18+ first."
        print_info "You can install it via:"
        print_info "- NVM: curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash"
        print_info "- Direct: https://nodejs.org/en/download/"
        exit 1
    fi
    
    NODE_VERSION=$($NODE_BIN --version | cut -d'v' -f2)
    NODE_MAJOR=$(echo $NODE_VERSION | cut -d'.' -f1)
    
    if [ "$NODE_MAJOR" -lt 18 ]; then
        print_error "Node.js version $NODE_VERSION detected. Node 18+ is required."
        print_info "Current Node path: $NODE_BIN"
        
        # Try alternative approaches
        print_info "Attempting to find Node 18+..."
        
        # Check common installation paths
        for NODE_PATH in \
            "$HOME/.nvm/versions/node/v18.*/bin/node" \
            "/usr/local/bin/node" \
            "/opt/homebrew/bin/node" \
            "/usr/bin/node"
        do
            if [ -x "$NODE_PATH" ]; then
                NODE_VERSION_CHECK=$($NODE_PATH --version 2>/dev/null | cut -d'v' -f2)
                NODE_MAJOR_CHECK=$(echo $NODE_VERSION_CHECK | cut -d'.' -f1)
                if [ "$NODE_MAJOR_CHECK" -ge 18 ]; then
                    print_status "Found Node $NODE_VERSION_CHECK at $NODE_PATH"
                    NODE_BIN="$NODE_PATH"
                    NPM_BIN="$(dirname $NODE_PATH)/npm"
                    export PATH="$(dirname $NODE_PATH):$PATH"
                    break
                fi
            fi
        done
        
        # Final version check
        NODE_VERSION=$($NODE_BIN --version | cut -d'v' -f2)
        NODE_MAJOR=$(echo $NODE_VERSION | cut -d'.' -f1)
        
        if [ "$NODE_MAJOR" -lt 18 ]; then
            print_error "Still using Node $NODE_VERSION. Please install Node 18+ manually."
            print_info "Installation options:"
            print_info "1. NVM: curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash && nvm install 18"
            print_info "2. Homebrew (macOS): brew install node@18"
            print_info "3. Direct download: https://nodejs.org/en/download/"
            exit 1
        fi
    fi
    
    print_status "Using Node.js $NODE_VERSION at $NODE_BIN"
    
    if [ -z "$NPM_BIN" ] || [ ! -x "$NPM_BIN" ]; then
        NPM_BIN="$(dirname $NODE_BIN)/npm"
        if [ ! -x "$NPM_BIN" ]; then
            print_error "npm not found. Please ensure npm is installed."
            exit 1
        fi
    fi
    
    NPM_VERSION=$($NPM_BIN --version)
    print_status "Using npm $NPM_VERSION at $NPM_BIN"
}

# Call the setup function
setup_node

# Check other prerequisites
print_info "Checking other prerequisites..."

if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed."
    exit 1
fi

if [ ! -f "requirements.txt" ]; then
    print_error "Please run this script from the project root directory (requirements.txt not found)."
    exit 1
fi

print_status "Prerequisites check passed"

# Backend Setup
print_info "Setting up backend..."

if [ ! -d "venv_clean" ]; then
    print_info "Creating Python virtual environment..."
    python3 -m venv venv_clean
fi

print_info "Activating Python virtual environment..."
source venv_clean/bin/activate

print_info "Installing backend dependencies..."
pip install -r requirements.txt > /dev/null 2>&1
if [ $? -ne 0 ]; then
    print_error "Backend dependency installation failed"
    exit 1
fi

print_status "Backend dependencies installed"

# Download spaCy model
print_info "Downloading spaCy language model..."
python -m spacy download en_core_web_sm > /dev/null 2>&1
if [ $? -eq 0 ]; then
    print_status "spaCy model downloaded successfully"
else
    print_warning "spaCy model download failed, but continuing..."
fi

print_info "Setting up database..."
python3 scripts/setup.py
if [ $? -ne 0 ]; then
    print_error "Database setup failed"
    exit 1
fi
print_status "Database setup completed"

# Frontend Setup
print_info "Setting up frontend..."
if [ ! -d "frontend" ]; then
    print_error "Frontend directory not found."
    exit 1
fi

cd frontend

print_info "Installing frontend dependencies..."
"$NPM_BIN" install > /dev/null 2>&1
if [ $? -ne 0 ]; then
    print_error "Frontend dependency installation failed"
    print_info "Trying with verbose output for debugging..."
    "$NPM_BIN" install
    if [ $? -ne 0 ]; then
        cd ..
        exit 1
    fi
fi
print_status "Frontend dependencies installed"
cd ..

# Start Backend
print_info "Starting backend server on port 8001..."
cd src
python3 -m uvicorn main:app --host 0.0.0.0 --port 8001 --reload &
BACKEND_PID=$!
cd ..

# Wait longer and check for backend startup more thoroughly
print_info "Waiting for backend to initialize..."
sleep 5

# Check multiple times with longer timeout
BACKEND_READY=false
for i in {1..12}; do
    if curl -s http://localhost:8001/health > /dev/null 2>&1; then
        BACKEND_READY=true
        break
    fi
    sleep 2
    echo -n "."
done
echo ""

if [ "$BACKEND_READY" = true ]; then
    print_status "Backend server started successfully"
else
    print_warning "Backend server may still be starting (continuing anyway...)"
fi

# Start Frontend using the verified Node/NPM paths
print_info "Starting frontend server on port 3000..."
cd frontend

# Set NODE_ENV for development
export NODE_ENV=development

# Use full path to ensure correct version
"$NPM_BIN" run dev &
FRONTEND_PID=$!
cd ..

# Wait for frontend
sleep 8

# Check frontend
FRONTEND_READY=false
for i in {1..6}; do
    if curl -s http://localhost:3000 > /dev/null 2>&1; then
        FRONTEND_READY=true
        break
    fi
    sleep 2
    echo -n "."
done
echo ""

if [ "$FRONTEND_READY" = true ]; then
    print_status "Frontend server started successfully"
else
    print_warning "Frontend server may still be starting..."
fi

# Success Output
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
echo "│  🔧 Backend API:  http://localhost:8001            │"
echo "│  📖 API Docs:     http://localhost:8001/docs       │"
echo "│  ❤️  Health:      http://localhost:8001/health     │"
echo "└─────────────────────────────────────────────────────┘"
echo ""
print_info "🚀 Ready to analyze GitHub repositories!"
print_info "💡 Paste any GitHub URL to see AI-powered team intelligence"
print_info "🎯 Experience revolutionary task assignment optimization"
echo ""
print_warning "Press Ctrl+C to stop all servers"
echo ""

while true; do
    sleep 1
done
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
import uvicorn

from .config import settings
from .models.database import create_tables, get_db
from .api import developers, assignments, tasks
from .core.developer_modeling.expertise_tracker import ExpertiseTracker

# Create FastAPI app
app = FastAPI(
    title="AI Engineering Task Router",
    description="Intelligent task assignment system for software engineering teams",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.DEBUG else ["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize expertise tracker
expertise_tracker = ExpertiseTracker()

@app.on_event("startup")
async def startup_event():
    """Initialize application on startup."""
    try:
        # Create database tables
        create_tables()
        print("Database tables created successfully")
        
        # Initialize model directories
        settings.MODELS_DIR.mkdir(exist_ok=True)
        settings.EMBEDDINGS_DIR.mkdir(exist_ok=True)
        
        print("AI Task Router started successfully")
        print(f"Debug mode: {settings.DEBUG}")
        print(f"Database: {settings.DATABASE_URL}")
        
    except Exception as e:
        print(f"Error during startup: {e}")
        raise

# Include API routers
app.include_router(developers.router, prefix="/api/v1", tags=["developers"])
app.include_router(assignments.router, prefix="/api/v1", tags=["assignments"])
app.include_router(tasks.router, prefix="/api/v1", tags=["tasks"])

@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "message": "AI Engineering Task Router",
        "status": "running",
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check(db: Session = Depends(get_db)):
    """Detailed health check with database connectivity."""
    try:
        # Test database connection
        db.execute("SELECT 1")
        
        return {
            "status": "healthy",
            "database": "connected",
            "components": {
                "skill_extraction": "ready",
                "task_analysis": "ready",
                "assignment_engine": "ready"
            }
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }

if __name__ == "__main__":
    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG
    )
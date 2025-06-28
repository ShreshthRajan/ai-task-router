# src/main.py
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
import uvicorn


from config import settings
from models.database import create_tables, get_db
from api import developers, assignments, tasks, learning
from api.github_integration import router as github_router
from core.developer_modeling.expertise_tracker import ExpertiseTracker
from core.learning_system.system_analytics import SystemAnalytics
from models.database import engine, Base

from sqlalchemy import text

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
system_analytics = SystemAnalytics()

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
app.include_router(learning.router, prefix="/api/v1", tags=["learning"])
app.include_router(github_router)

@app.get("/")
async def root():
   """Health check endpoint."""
   return {
       "message": "AI Engineering Task Router",
       "status": "running",
       "version": "1.0.0",
       "phase": "3 - Assignment Optimization Engine"
   }

@app.get("/health")
async def health_check(db: Session = Depends(get_db)):
    """Detailed health check with database connectivity."""
    try:
        db.execute(text("SELECT 1"))
        metrics = await system_analytics.get_system_health_metrics(db)
        return JSONResponse(metrics.model_dump())          # one-liner
    except Exception as e:
        raise HTTPException(500, detail=str(e))

if __name__ == "__main__":
   uvicorn.run(
       "src.main:app",
       host="0.0.0.0",
       port=8000,
       reload=settings.DEBUG
   )
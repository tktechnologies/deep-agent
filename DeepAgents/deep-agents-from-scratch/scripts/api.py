"""Deep Research Agent - FastAPI Server.

This module provides a REST API for the Deep Research Agent using FastAPI.
The API allows external interfaces to interact with the agent via HTTP requests.

Endpoints:
    GET  /              - API information
    GET  /health        - Health check
    POST /api/query     - Send question to agent
    GET  /api/sessions  - List active sessions (future feature)

Run with: uvicorn api:app --reload --host 0.0.0.0 --port 8000
"""

# ============================================================================
# IMPORTS
# ============================================================================

import os
import sys
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Add scripts folder to path
sys.path.insert(0, os.path.dirname(__file__))

# Import agent
from deep_agent import run_agent


# ============================================================================
# FASTAPI APP CONFIGURATION
# ============================================================================

app = FastAPI(
    title="Deep Research Agent API",
    description="REST API for Deep Research Agent powered by LangGraph",
    version="1.0.0",
    docs_url="/docs",  # Swagger UI at http://localhost:8000/docs
    redoc_url="/redoc"  # ReDoc at http://localhost:8000/redoc
)

# Configure CORS (allow requests from frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class QueryRequest(BaseModel):
    """Request model for agent queries."""
    question: str = Field(..., min_length=1, description="Question to ask the agent")
    verbose: bool = Field(default=False, description="Enable verbose logging")
    
    class Config:
        json_schema_extra = {
            "example": {
                "question": "Who was the first person to walk on the moon?",
                "verbose": False
            }
        }


class QueryResponse(BaseModel):
    """Response model for agent queries."""
    success: bool = Field(description="Whether the query succeeded")
    answer: str = Field(description="Agent's answer")
    question: str = Field(description="Original question")
    message_count: int = Field(description="Number of messages exchanged")
    timestamp: str = Field(description="Response timestamp")
    files_created: int = Field(description="Number of files created during research")
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "answer": "Neil Armstrong was the first person to walk on the moon...",
                "question": "Who was the first person to walk on the moon?",
                "message_count": 15,
                "timestamp": "2025-12-23T10:30:00",
                "files_created": 3
            }
        }


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    timestamp: str
    version: str


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/", tags=["Info"])
async def root():
    """API information and welcome message."""
    return {
        "message": "Deep Research Agent API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "health": "GET /health",
            "query": "POST /api/query"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint.
    
    Returns:
        API status and version information
    """
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }


@app.post("/api/query", response_model=QueryResponse, tags=["Agent"])
async def query_agent(request: QueryRequest):
    """Send a question to the Deep Research Agent.
    
    This endpoint processes user questions using the LangGraph agent,
    which performs web searches, analyzes results, and generates comprehensive answers.
    
    Args:
        request: Query request with question and options
        
    Returns:
        Agent's answer and metadata
        
    Raises:
        HTTPException: If agent execution fails
    """
    try:
        # Run agent
        result = run_agent(
            question=request.question,
            verbose=request.verbose
        )
        
        # Format response
        return QueryResponse(
            success=True,
            answer=result["answer"],
            question=result["question"],
            message_count=result["message_count"],
            timestamp=datetime.now().isoformat(),
            files_created=len(result.get("files", {}))
        )
        
    except Exception as e:
        # Log error (in production, use proper logging)
        print(f"❌ Error processing query: {str(e)}")
        
        raise HTTPException(
            status_code=500,
            detail=f"Agent execution failed: {str(e)}"
        )


# ============================================================================
# STARTUP/SHUTDOWN EVENTS
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Actions to perform on API startup."""
    print("\n" + "="*80)
    print("🚀 Deep Research Agent API Starting...")
    print("="*80)
    print(f"📍 API Documentation: http://localhost:8000/docs")
    print(f"🔍 ReDoc: http://localhost:8000/redoc")
    print(f"💡 Health Check: http://localhost:8000/health")
    print("="*80 + "\n")


@app.on_event("shutdown")
async def shutdown_event():
    """Actions to perform on API shutdown."""
    print("\n" + "="*80)
    print("👋 Deep Research Agent API Shutting Down...")
    print("="*80 + "\n")


# ============================================================================
# MAIN (for direct execution)
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Auto-reload on code changes
        log_level="info"
    )

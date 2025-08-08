
"""
SHIOL+ Dashboard API Endpoints
==============================

API endpoints specifically for the dashboard frontend (dashboard.html).
These endpoints provide administrative access and system management.
"""

from fastapi import APIRouter, HTTPException, Request, Depends
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from typing import Optional, Dict, Any, List
from datetime import datetime
from loguru import logger
import os
from pathlib import Path

from src.auth import get_current_user, User
from src.api_utils import convert_numpy_types
try:
    from src.database import get_all_draws, get_prediction_history, get_system_stats
except ImportError:
    # Fallback if database functions are not available
    logger.warning("Database functions not available, using fallback implementations")
    def get_all_draws():
        return []
    def get_prediction_history(limit=100):
        return []
    def get_system_stats():
        return {}
from src.predictor import Predictor
from src.intelligent_generator import IntelligentGenerator, DeterministicGenerator

# Create router for dashboard frontend endpoints
dashboard_frontend_router = APIRouter(tags=["dashboard_frontend"])

# Global components (will be injected from main API)
predictor = None
intelligent_generator = None
deterministic_generator = None

def set_dashboard_components(pred, intel_gen, det_gen):
    """Set the global components for dashboard endpoints."""
    global predictor, intelligent_generator, deterministic_generator
    predictor = pred
    intelligent_generator = intel_gen
    deterministic_generator = det_gen

# Build path to frontend templates
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "frontend"))
templates = Jinja2Templates(directory=FRONTEND_DIR)

@dashboard_frontend_router.get("/dashboard", response_class=HTMLResponse)
async def serve_dashboard(request: Request, current_user: User = Depends(get_current_user)):
    """Serve the dashboard page (requires authentication)"""
    return templates.TemplateResponse("dashboard.html", {"request": request, "user": current_user})

@dashboard_frontend_router.get("/login", response_class=HTMLResponse)
async def serve_login(request: Request):
    """Serve the login page"""
    return templates.TemplateResponse("login.html", {"request": request})

@dashboard_frontend_router.get("/api/v1/dashboard/system/status")
async def get_dashboard_system_status(current_user: User = Depends(get_current_user)):
    """Get system status for dashboard"""
    try:
        import psutil
        
        status = {
            "timestamp": datetime.now().isoformat(),
            "system": {
                "cpu_percent": psutil.cpu_percent(interval=1),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_percent": psutil.disk_usage('.').percent
            },
            "services": {
                "database": "connected",
                "model": "loaded" if predictor and hasattr(predictor, 'model') else "not_loaded",
                "api": "operational"
            }
        }
        
        return convert_numpy_types(status)
        
    except Exception as e:
        logger.error(f"Error getting dashboard system status: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting system status: {str(e)}")

@dashboard_frontend_router.get("/api/v1/dashboard/predictions/stats")
async def get_dashboard_prediction_stats(current_user: User = Depends(get_current_user)):
    """Get prediction statistics for dashboard"""
    try:
        # Get recent prediction history
        recent_predictions = get_prediction_history(limit=1000)
        
        stats = {
            "total_predictions": len(recent_predictions),
            "recent_activity": len([p for p in recent_predictions if 
                                  datetime.fromisoformat(p.get('timestamp', '1970-01-01')) > 
                                  datetime.now().replace(day=datetime.now().day-7)]),
            "model_version": "1.0.0",
            "last_update": datetime.now().isoformat()
        }
        
        return convert_numpy_types(stats)
        
    except Exception as e:
        logger.error(f"Error getting dashboard prediction stats: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting prediction stats: {str(e)}")

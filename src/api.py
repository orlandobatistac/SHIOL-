from fastapi import FastAPI, APIRouter, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from loguru import logger
import os
from datetime import datetime
import asyncio
import uuid
from typing import Optional, Dict, Any
from pathlib import Path

from src.predictor import Predictor
from src.intelligent_generator import IntelligentGenerator, DeterministicGenerator
from src.loader import update_database_from_source
import src.database as db
from src.adaptive_feedback import initialize_adaptive_system
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from contextlib import asynccontextmanager

# Import all modular API components
from src.api_utils import convert_numpy_types
from src.api_prediction_endpoints import prediction_router, set_prediction_components
from src.api_system_endpoints import system_router
from src.api_config_endpoints import config_router
from src.api_database_endpoints import database_router
from src.api_analytics_endpoints import analytics_router
from src.api_model_endpoints import model_router
from src.api_pipeline_endpoints import pipeline_router
from src.public_api import public_router, auth_router
from src.api_public_endpoints import public_frontend_router, set_public_components
from src.api_dashboard_endpoints import dashboard_frontend_router, set_dashboard_components

# --- Pipeline Monitoring Global Variables ---
# Global variables for pipeline monitoring
pipeline_orchestrator = None
pipeline_executions = {}  # Track running pipeline executions
pipeline_logs = []  # Store recent pipeline logs

# --- Scheduler and App Lifecycle ---
scheduler = AsyncIOScheduler()

async def update_data_automatically():
    """Task to update database from source."""
    logger.info("Running automatic data update task.")
    try:
        update_database_from_source()
        logger.info("Automatic data update completed successfully.")
    except Exception as e:
        logger.error(f"Error during automatic data update: {e}")

async def trigger_full_pipeline_automatically():
    """Task to trigger the full pipeline automatically with enhanced metadata."""
    logger.info("Running automatic full pipeline trigger.")
    try:
        # Check if pipeline is already running to prevent duplicates
        running_executions = [ex for ex in pipeline_executions.values() if ex.get("status") == "running"]
        if running_executions:
            logger.warning(f"Pipeline already running (ID: {running_executions[0].get('execution_id')}), skipping automatic execution.")
            return

        if pipeline_orchestrator:
            # Get current scheduler configuration
            current_time = datetime.now()
            current_day = current_time.strftime('%A').lower()
            current_time_str = current_time.strftime('%H:%M')

            # Expected scheduler configuration (from scheduler setup)
            expected_days = ['monday', 'wednesday', 'saturday']
            expected_time = '23:30'
            timezone = 'America/New_York'

            # Check if execution matches schedule
            matches_schedule = (
                current_day in expected_days and
                abs((current_time.hour * 60 + current_time.minute) - (23 * 60 + 30)) <= 5  # 5 minute tolerance
            )

            # Trigger the full pipeline execution with enhanced metadata
            execution_id = str(uuid.uuid4())[:8]
            pipeline_executions[execution_id] = {
                "execution_id": execution_id,
                "status": "starting",
                "start_time": current_time.isoformat(),
                "current_step": "automated_trigger",
                "steps_completed": 0,
                "total_steps": 7,  # Always 7 steps for full pipeline
                "num_predictions": 100,  # Standard 100 predictions
                "requested_steps": None,  # Full pipeline, all steps
                "error": None,
                "trigger_type": "automatic_scheduler",
                "execution_source": "automatic_scheduler",
                "trigger_details": {
                    "type": "scheduled",
                    "scheduled_config": {
                        "days": expected_days,
                        "time": expected_time,
                        "timezone": timezone
                    },
                    "actual_execution": {
                        "day": current_day,
                        "time": current_time_str,
                        "matches_schedule": matches_schedule
                    },
                    "triggered_by": "automatic_scheduler"
                }
            }

            # Run the full 7-step pipeline in background with 100 predictions
            from src.api import run_full_pipeline_background
            asyncio.create_task(run_full_pipeline_background(execution_id, 100))
            logger.info(f"Automatic pipeline execution started with ID: {execution_id} - Full 7-step pipeline (scheduled: {matches_schedule})")
        else:
            logger.warning("Pipeline orchestrator not available to trigger pipeline.")
    except Exception as e:
        logger.error(f"Error triggering automatic full pipeline: {e}")

async def run_full_pipeline_background(execution_id: str, num_predictions: int = 100):
    """Background task to run full pipeline execution"""
    try:
        if not pipeline_orchestrator:
            logger.error(f"Pipeline orchestrator not available for execution {execution_id}")
            return

        # Update execution status
        if execution_id in pipeline_executions:
            pipeline_executions[execution_id]["status"] = "running"
            pipeline_executions[execution_id]["current_step"] = "pipeline_execution"

        # Run the pipeline
        results = await pipeline_orchestrator.run_full_pipeline_async(num_predictions)

        # Update execution status on completion
        if execution_id in pipeline_executions:
            pipeline_executions[execution_id]["status"] = "completed"
            pipeline_executions[execution_id]["end_time"] = datetime.now().isoformat()
            pipeline_executions[execution_id]["steps_completed"] = 7
            pipeline_executions[execution_id]["results"] = results

        logger.info(f"Pipeline execution {execution_id} completed successfully")

    except Exception as e:
        logger.error(f"Pipeline execution {execution_id} failed: {e}")
        if execution_id in pipeline_executions:
            pipeline_executions[execution_id]["status"] = "failed"
            pipeline_executions[execution_id]["error"] = str(e)
            pipeline_executions[execution_id]["end_time"] = datetime.now().isoformat()

@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline_orchestrator
    # On startup
    logger.info("Application startup...")

    # Initialize pipeline orchestrator
    try:
        from main import PipelineOrchestrator # Import here to avoid circular dependencies if main imports this file
        pipeline_orchestrator = PipelineOrchestrator()
        app.state.orchestrator = pipeline_orchestrator # Attach to app state
        logger.info("Pipeline orchestrator initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize pipeline orchestrator: {e}")
        pipeline_orchestrator = None
        app.state.orchestrator = None

    # Schedule pipeline execution optimally:
    # 1. Full pipeline only on actual drawing days (Monday, Wednesday, Saturday)
    scheduler.add_job(
        func=trigger_full_pipeline_automatically,
        trigger="cron",
        day_of_week="mon,wed,sat", # Only on actual Powerball drawing days
        hour=23,                    # 11 PM ET
        minute=30,                  # 30 minutes after drawing
        id="post_drawing_pipeline",
        name="Full Pipeline After Drawing Results",
        max_instances=1,           # Prevent overlapping executions
        coalesce=True             # Merge multiple pending executions into one
    )

    # 2. Maintenance data update only (no full pipeline)
    scheduler.add_job(
        func=update_data_automatically,
        trigger="cron",
        day_of_week="tue,thu,fri,sun", # Non-drawing days only
        hour=6,                        # 6 AM instead of every 12 hours
        minute=0,
        id="maintenance_data_update",
        name="Maintenance Data Update on Non-Drawing Days",
        max_instances=1,
        coalesce=True
    )
    scheduler.start()
    logger.info("Scheduler started. Scheduled jobs are active.")
    yield
    # On shutdown
    logger.info("Application shutdown...")
    scheduler.shutdown()
    logger.info("Scheduler shut down.")

# --- Application Initialization ---
logger.info("Initializing FastAPI application...")
app = FastAPI(
    title="SHIOL+ Powerball Prediction API",
    description="Provides ML-based Powerball number predictions.",
    version="6.0.0", # Updated version to 6.0.0
    lifespan=lifespan
)

# --- CORS Configuration ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Global Components ---
try:
    logger.info("Loading model and generator instances...")
    predictor = Predictor()

    # Load historical data first
    from src.loader import DataLoader
    data_loader = DataLoader()
    historical_data = data_loader.load_historical_data()

    # Initialize generators with historical data
    intelligent_generator = IntelligentGenerator(historical_data)
    deterministic_generator = DeterministicGenerator(historical_data)

    logger.info("Model and generators loaded successfully.")

    # Set up prediction components for modular endpoints
    set_prediction_components(predictor, intelligent_generator, deterministic_generator)
    set_public_components(predictor, intelligent_generator, deterministic_generator)
    set_dashboard_components(predictor, intelligent_generator, deterministic_generator)

except Exception as e:
    logger.critical(f"Fatal error during startup: Failed to load model. Error: {e}")
    predictor = None
    intelligent_generator = None
    deterministic_generator = None

# --- API Router ---
api_router = APIRouter(prefix="/api/v1")

# Basic system info endpoint (kept in main API for core functionality)
@api_router.get("/system/info")
async def get_system_info():
    """Get system information"""
    return {
        "version": "6.0.0",
        "status": "operational",
        "database_status": "connected" if db.is_database_connected() else "disconnected",
        "model_status": "loaded" if predictor and hasattr(predictor, 'model') and predictor.model else "not_loaded"
    }

# --- Application Mounting ---
# Mount all API routers before static files
app.include_router(api_router)
app.include_router(prediction_router)
app.include_router(system_router)
app.include_router(config_router, prefix="/api/v1")
app.include_router(database_router, prefix="/api/v1")
app.include_router(analytics_router, prefix="/api/v1")
app.include_router(model_router, prefix="/api/v1")
app.include_router(pipeline_router, prefix="/api/v1")
app.include_router(public_router)
app.include_router(auth_router)
# Mount modular frontend routers
app.include_router(public_frontend_router)
app.include_router(dashboard_frontend_router)

@app.get("/api/v1/prediction-history-grouped")
async def get_prediction_history_grouped(limit_dates: int = Query(25, ge=1, le=100)):
    """Get grouped prediction history by date"""
    try:
        from src.public_api import get_predictions_performance_endpoint
        return await get_predictions_performance_endpoint(limit_dates)
    except Exception as e:
        logger.error(f"Error in grouped prediction history: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving grouped prediction history")


# Build an absolute path to the 'frontend' directory for robust file serving.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "frontend"))

# Ensure the frontend directory exists before mounting
if not os.path.exists(FRONTEND_DIR):
    logger.warning(f"Frontend directory not found at {FRONTEND_DIR}. Static file serving may fail.")

app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="static")
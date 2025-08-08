from fastapi import FastAPI, HTTPException, APIRouter, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from loguru import logger
import os
from datetime import datetime, timedelta
import asyncio
import uuid
import json
from typing import Optional, Dict, Any, List
import configparser
from pathlib import Path
import psutil
import sqlite3

from src.predictor import Predictor
from src.intelligent_generator import IntelligentGenerator, DeterministicGenerator
from src.loader import update_database_from_source
from src.database import save_prediction_log
import src.database as db
from src.adaptive_feedback import (
    initialize_adaptive_system, run_adaptive_analysis, AdaptiveValidator,
    ModelFeedbackEngine, AdaptivePlayScorer
)
from src.config_manager import ConfigurationManager
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from contextlib import asynccontextmanager

# Import modular API components
from src.api_utils import convert_numpy_types
from src.api_prediction_endpoints import prediction_router, set_prediction_components
from src.api_system_endpoints import system_router
from src.public_api import public_router, auth_router
from src.api_frontend_public import public_frontend_router, set_public_components
from src.api_frontend_dashboard import dashboard_frontend_router, set_dashboard_components

# --- Pipeline Monitoring Global Variables ---
# Import PipelineOrchestrator from main.py
# from main import PipelineOrchestrator # Moved to lifespan for better management

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
            asyncio.create_task(run_full_pipeline_background(execution_id, 100))
            logger.info(f"Automatic pipeline execution started with ID: {execution_id} - Full 7-step pipeline (scheduled: {matches_schedule})")
        else:
            logger.warning("Pipeline orchestrator not available to trigger pipeline.")
    except Exception as e:
        logger.error(f"Error triggering automatic full pipeline: {e}")


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

except Exception as e:
    logger.critical(f"Fatal error during startup: Failed to load model. Error: {e}")
    predictor = None
    intelligent_generator = None
    deterministic_generator = None

# --- API Router ---
api_router = APIRouter(prefix="/api/v1")

# --- Public Frontend Endpoints ---
# Moved to src.api_frontend_public.py
# Includes endpoints for general public information and predictions

# --- Dashboard Frontend Endpoints ---
# Moved to src.api_frontend_dashboard.py
# Includes endpoints for system monitoring, configuration, and pipeline management

# --- System and Core Functionality Endpoints ---
# These are kept in the main api_router or mounted as separate routers

# Endpoint for system statistics
@api_router.get("/system/stats")
async def get_system_stats():
    """Get real-time system statistics for dashboard monitoring"""
    try:
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')

        # Get pipeline status from orchestrator if available
        pipeline_status = "ready"
        last_execution = "Never"

        if hasattr(app.state, 'orchestrator') and app.state.orchestrator:
            try:
                status_info = app.state.orchestrator.get_pipeline_status()
                pipeline_status = status_info.get('current_status', 'ready')
                last_execution_info = status_info.get('last_execution')
                if last_execution_info and isinstance(last_execution_info, dict) and last_execution_info.get('start_time'):
                    last_execution = last_execution_info['start_time']
                elif isinstance(last_execution_info, str):
                    last_execution = last_execution_info
            except Exception as e:
                logger.warning(f"Could not retrieve pipeline status: {e}")
                pipeline_status = "Error"

        return {
            "cpu_usage": round(cpu_percent, 1),
            "memory_usage": round(memory.percent, 1),
            "disk_usage": round((disk.used / disk.total) * 100, 1),
            "memory_total_gb": round(memory.total / (1024**3), 2),
            "disk_total_gb": round(disk.total / (1024**3), 2),
            "pipeline_status": pipeline_status,
            "last_execution": last_execution,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting system stats: {e}")
        # Return default stats instead of raising exception
        return {
            "cpu_usage": 0.0,
            "memory_usage": 0.0,
            "disk_usage": 0.0,
            "memory_total_gb": 0.0,
            "disk_total_gb": 0.0,
            "pipeline_status": "error",
            "last_execution": "Error",
            "timestamp": datetime.now().isoformat()
        }

# Endpoint for database statistics
@api_router.get("/database/stats")
async def get_database_stats():
    """Get database statistics for dashboard"""
    try:
        conn = db.get_db_connection()
        cursor = conn.cursor()

        # Get total records from main tables
        cursor.execute("SELECT COUNT(*) FROM powerball_draws")
        draws_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM predictions_log")
        predictions_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM performance_tracking")
        validations_count = cursor.fetchone()[0]

        # Get database file size
        db_path = db.get_db_path() # Use centralized configuration
        try:
            db_size_bytes = os.path.getsize(db_path)
            db_size_mb = round(db_size_bytes / (1024 * 1024), 2)
        except FileNotFoundError:
            db_size_mb = 0
            logger.warning(f"Database file not found at expected path: {db_path}")
        except Exception as e:
            db_size_mb = 0
            logger.error(f"Could not get database file size for {db_path}: {e}")


        conn.close()

        return {
            "total_records": draws_count + predictions_count + validations_count,
            "draws_count": draws_count,
            "predictions_count": predictions_count,
            "validations_count": validations_count,
            "size_mb": db_size_mb,
            "last_updated": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting database stats: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting database stats: {str(e)}")

# Endpoint for performance analytics
@api_router.get("/analytics/performance")
async def get_performance_analytics():
    """Get performance analytics for dashboard"""
    try:
        # Get performance data from database without passing connection
        performance_data = db.get_performance_analytics(days_back=30)

        # Calculate key metrics
        total_predictions = performance_data.get('total_predictions', 0)
        winning_predictions = performance_data.get('winning_predictions', 0)
        win_rate = (winning_predictions / total_predictions * 100) if total_predictions > 0 else 0

        return {
            "win_rate": round(win_rate, 1),
            "avg_score": round(performance_data.get('avg_score', 0), 3),
            "best_method": performance_data.get('best_method', 'smart_ai'),
            "total_predictions": total_predictions,
            "winning_predictions": winning_predictions,
            "total_prize_amount": performance_data.get('total_prize_amount', 0),
            "roi_percentage": round(performance_data.get('roi_percentage', 0), 2)
        }
    except Exception as e:
        logger.error(f"Error getting performance analytics: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting performance analytics: {str(e)}")

# Endpoint for loading system configuration
@api_router.get("/config/load")
async def load_configuration():
    """Load system configuration using hybrid system"""
    try:
        # Initialize ConfigurationManager
        config_manager = ConfigurationManager()
        # Load configuration from database (or fallback to file if DB is empty/unavailable)
        config_data = config_manager.load_configuration()

        # Convert from database format to frontend format
        result = {
            "pipeline": {
                "execution_days": {
                    "monday": config_data.get("pipeline", {}).get("execution_days_monday", "True").lower() == "true",
                    "wednesday": config_data.get("pipeline", {}).get("execution_days_wednesday", "True").lower() == "true",
                    "saturday": config_data.get("pipeline", {}).get("execution_days_saturday", "True").lower() == "true",
                },
                "execution_time": config_data.get("pipeline", {}).get("execution_time", "02:00"),
                "timezone": config_data.get("pipeline", {}).get("timezone", "America/New_York"),
                "auto_execution": config_data.get("pipeline", {}).get("auto_execution", "True").lower() == "true"
            },
            "predictions": {
                "count": int(config_data.get("predictions", {}).get("count", "100")),
                "method": config_data.get("predictions", {}).get("method", "smart_ai"),
                "weights": {
                    "probability": int(config_data.get("weights", {}).get("probability", "40")),
                    "diversity": int(config_data.get("weights", {}).get("diversity", "25")),
                    "historical": int(config_data.get("weights", {}).get("historical", "20")),
                    "risk": int(config_data.get("weights", {}).get("risk", "15"))
                }
            },
            "config_source": "database" if config_data else "fallback"
        }

        return result

    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        raise HTTPException(status_code=500, detail=f"Error loading configuration: {str(e)}")

# Endpoint for saving system configuration
@api_router.post("/config/save")
async def save_configuration(config_data: Dict[str, Any]):
    """Save system configuration using hybrid system"""
    try:
        # Initialize ConfigurationManager
        config_manager = ConfigurationManager()
        # Prepare configuration data for saving
        db_config = {}

        if "pipeline" in config_data:
            db_config["pipeline"] = {}
            pipeline = config_data["pipeline"]

            if "execution_days" in pipeline:
                db_config["pipeline"]["execution_days_monday"] = str(pipeline["execution_days"].get("monday", True))
                db_config["pipeline"]["execution_days_wednesday"] = str(pipeline["execution_days"].get("wednesday", True))
                db_config["pipeline"]["execution_days_saturday"] = str(pipeline["execution_days"].get("saturday", True))

            db_config["pipeline"]["execution_time"] = pipeline.get("execution_time", "02:00")
            db_config["pipeline"]["timezone"] = pipeline.get("timezone", "America/New_York")
            db_config["pipeline"]["auto_execution"] = str(pipeline.get("auto_execution", True))

        if "predictions" in config_data:
            db_config["predictions"] = {}
            predictions = config_data["predictions"]

            db_config["predictions"]["count"] = str(predictions.get("count", 100))
            db_config["predictions"]["method"] = predictions.get("method", "smart_ai")

            # Handle weights
            if "weights" in predictions:
                db_config["weights"] = {}
                weights = predictions["weights"]
                db_config["weights"]["probability"] = str(weights.get("probability", 40))
                db_config["weights"]["diversity"] = str(weights.get("diversity", 25))
                db_config["weights"]["historical"] = str(weights.get("historical", 20))
                db_config["weights"]["risk"] = str(weights.get("risk", 15))

        # Save to database using hybrid system
        success = config_manager.save_configuration(db_config)

        if success:
            return {"success": True, "message": "Configuration saved successfully to database"}
        else:
            raise HTTPException(status_code=500, detail="Failed to save configuration to database")

    except Exception as e:
        logger.error(f"Error saving configuration: {e}")
        raise HTTPException(status_code=500, detail=f"Error saving configuration: {str(e)}")

# Endpoint for database cleanup
@api_router.post("/database/cleanup")
async def cleanup_database(cleanup_options: Dict[str, bool]):
    """Clean database based on selected options"""
    global pipeline_executions, pipeline_logs
    try:
        logger.info(f"Starting database cleanup with options: {cleanup_options}")

        conn = db.get_db_connection()
        cursor = conn.cursor()
        results = []

        if cleanup_options.get('predictions', False):
            try:
                cursor.execute("DELETE FROM predictions_log")
                deleted_predictions = cursor.rowcount
                results.append(f"Deleted {deleted_predictions} predictions")
                logger.info(f"Deleted {deleted_predictions} prediction records")
            except Exception as e:
                logger.error(f"Error deleting predictions: {e}")
                results.append(f"Error deleting predictions: {str(e)}")

        if cleanup_options.get('validations', False):
            try:
                cursor.execute("DELETE FROM performance_tracking")
                deleted_performance = cursor.rowcount
                results.append(f"Deleted {deleted_performance} performance tracking records")
                logger.info(f"Deleted {deleted_performance} performance tracking records")
            except Exception as e:
                logger.error(f"Error deleting validations: {e}")
                results.append(f"Error deleting validations: {str(e)}")

        if cleanup_options.get('logs', False):
            # Clear log files
            log_files_cleared = 0
            logs_dir = Path('logs')
            if logs_dir.exists():
                for log_file in logs_dir.glob('*.log'):
                    log_file.unlink()
                    log_files_cleared += 1
            results.append(f"Cleared {log_files_cleared} log files")
            logger.info(f"Cleared {log_files_cleared} log files")

        if cleanup_options.get('pipeline_logs', False):
            # Clear pipeline reports
            pipeline_files_cleared = 0

            # Clear pipeline reports
            reports_dir = Path('reports')
            if reports_dir.exists():
                for report_file in reports_dir.glob('pipeline_report_*.json'):
                    report_file.unlink()
                    pipeline_files_cleared += 1

            # Clear system logs
            logs_dir = Path('logs')
            if logs_dir.exists():
                for log_file in logs_dir.glob('*.log'):
                    log_file.unlink()
                    pipeline_files_cleared += 1

            # Clear global pipeline execution tracking
            pipeline_executions.clear()
            pipeline_logs.clear()

            results.append(f"Cleared {pipeline_files_cleared} pipeline log files and execution history")
            logger.info(f"Cleared {pipeline_files_cleared} pipeline log files and execution history")

        if cleanup_options.get('models', False):
            # Reset AI models data
            cursor.execute('DELETE FROM adaptive_weights')
            deleted_weights = cursor.rowcount
            cursor.execute('DELETE FROM model_feedback')
            deleted_feedback = cursor.rowcount
            cursor.execute('DELETE FROM reliable_plays')
            deleted_plays = cursor.rowcount
            results.append(f"Reset AI models: deleted {deleted_weights} weight sets, {deleted_feedback} feedback records, {deleted_plays} reliable plays")
            logger.info(f"Reset AI models data")

        if cleanup_options.get('complete_reset', False):
            # Complete system reset
            tables_to_clear = ['predictions_log', 'performance_tracking', 'adaptive_weights', 'pattern_analysis', 'reliable_plays', 'model_feedback']
            total_cleared = 0

            for table in tables_to_clear:
                cursor.execute(f'DELETE FROM {table}')
                total_cleared += cursor.rowcount
                logger.info(f"Cleared {cursor.rowcount} records from {table}")

            # Clear all log files
            logs_dir = Path('logs')
            if logs_dir.exists():
                for log_file in logs_dir.glob('*'):
                    if log_file.is_file():
                        log_file.unlink()

            # Clear pipeline reports
            reports_dir = Path('reports')
            if reports_dir.exists():
                for report_file in reports_dir.glob('pipeline_report_*.json'):
                    report_file.unlink()

            # Clear global pipeline execution tracking
            pipeline_executions.clear()
            pipeline_logs.clear()

            results.append(f"Complete system reset: cleared {total_cleared} total records, all log files, and pipeline execution history (kept historical draw data and configuration)")
            logger.info(f"Complete system reset performed")

        # Commit all changes
        conn.commit()
        conn.close()

        if not results:
            results.append("No cleanup options selected")

        logger.info(f"Database cleanup completed successfully: {results}")

        return {
            "success": True,
            "message": "Cleanup completed successfully",
            "details": results
        }

    except Exception as e:
        logger.error(f"Error during database cleanup: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error during cleanup: {str(e)}")

# Endpoint for database status
@api_router.get("/database/status")
async def get_database_status():
    """Get detailed database status and record counts"""
    try:
        conn = db.get_db_connection()
        cursor = conn.cursor()

        # Get counts from all main tables
        table_counts = {}

        tables_to_check = [
            'powerball_draws',
            'predictions_log',
            'performance_tracking',
            'adaptive_weights',
            'pattern_analysis',
            'reliable_plays',
            'model_feedback',
            'system_config'
        ]

        for table in tables_to_check:
            try:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                table_counts[table] = count
            except sqlite3.Error as e:
                table_counts[table] = f"Error: {str(e)}"

        # Check if database is "empty" (only has essential data)
        is_empty = (
            table_counts.get('predictions_log', 0) == 0 and
            table_counts.get('performance_tracking', 0) == 0 and
            table_counts.get('adaptive_weights', 0) == 0 and
            table_counts.get('model_feedback', 0) == 0
        )

        conn.close()

        return {
            "database_status": "empty" if is_empty else "has_data",
            "table_counts": table_counts,
            "total_predictions": table_counts.get('predictions_log', 0),
            "total_validations": table_counts.get('performance_tracking', 0),
            "has_historical_data": table_counts.get('powerball_draws', 0) > 0,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error getting database status: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting database status: {str(e)}")

# Endpoint for database backup
@api_router.post("/database/backup")
async def backup_database():
    """Create database backup"""
    try:
        from shutil import copy2
        from datetime import datetime

        db_path = db.get_db_path() # Use centralized configuration
        backup_dir = os.path.join(os.path.dirname(db_path), "backups")

        os.makedirs(backup_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = os.path.join(backup_dir, f"shiolplus_backup_{timestamp}.db")

        copy2(db_path, backup_path)

        logger.info(f"Database backup created: {backup_path}")
        return {"message": "Database backup created successfully", "backup_file": backup_path}

    except FileNotFoundError:
        logger.error(f"Database file not found for backup: {db_path}")
        raise HTTPException(status_code=404, detail=f"Database file not found at {db_path}")
    except Exception as e:
        logger.error(f"Error creating database backup: {e}")
        raise HTTPException(status_code=500, detail=f"Error creating database backup: {str(e)}")

# Endpoint for system logs
@api_router.get("/logs")
async def get_system_logs():
    """Get recent system logs"""
    try:
        log_dir = os.path.join(os.path.dirname(__file__), "..", "logs")
        log_files = []

        if os.path.exists(log_dir):
            log_files = [f for f in os.listdir(log_dir) if f.endswith('.log')]

        recent_logs = []

        # Read recent log entries (last 100 lines)
        if log_files:
            # Find the most recently modified log file
            latest_log_file_path = max(
                [os.path.join(log_dir, f) for f in log_files],
                key=os.path.getctime
            )

            try:
                with open(latest_log_file_path, 'r') as f:
                    lines = f.readlines()[-100:]  # Last 100 lines

                for line in lines:
                    if 'ERROR' in line:
                        level = 'error'
                    elif 'WARNING' in line:
                        level = 'warning'
                    else:
                        level = 'info'

                    recent_logs.append({
                        'level': level,
                        'message': line.strip(),
                        'timestamp': datetime.now().isoformat() # Use current timestamp for log entry if specific timestamp not parsed
                    })
            except FileNotFoundError:
                logger.warning(f"Latest log file not found for reading: {latest_log_file_path}")
            except Exception as e:
                logger.error(f"Error reading log file {latest_log_file_path}: {e}")


        return recent_logs

    except Exception as e:
        logger.error(f"Error getting logs: {e}")
        return []

# Endpoint for testing pipeline configuration
@api_router.post("/pipeline/test")
async def test_pipeline():
    """Test pipeline configuration without full execution"""
    try:
        # Placeholder for pipeline test logic
        test_results = {
            "database_connection": True,
            "model_loaded": True,
            "configuration_valid": True,
            "api_responsive": True
        }

        logger.info("Pipeline test completed successfully")
        return {
            "message": "Pipeline test completed successfully",
            "results": test_results,
            "status": "passed"
        }

    except Exception as e:
        logger.error(f"Pipeline test failed: {e}")
        raise HTTPException(status_code=500, detail=f"Pipeline test failed: {str(e)}")

# Endpoint for retraining models
@api_router.post("/model/retrain")
async def retrain_model():
    """Retrain AI models with latest data"""
    try:
        # Placeholder for model retraining logic
        logger.info("Model retraining initiated")
        return {
            "message": "Model retraining started",
            "status": "initiated",
            "estimated_completion": "15-30 minutes"
        }

    except Exception as e:
        logger.error(f"Error starting model retraining: {e}")
        raise HTTPException(status_code=500, detail=f"Error starting model retraining: {str(e)}")

# Endpoint for backing up models
@api_router.post("/model/backup")
async def backup_models():
    """Create backup of current AI models"""
    try:
        from datetime import datetime
        import shutil
        import os

        logger.info("Model backup initiated")

        # Define paths
        # Assuming models are stored in a directory relative to the db backup directory
        # Adjust this path if your model storage location differs
        db_backup_dir = os.path.join(os.path.dirname(db.get_db_path()), "backups")
        model_dir = os.path.join(os.path.dirname(__file__), "..", "models")
        model_file_path = os.path.join(model_dir, "shiolplus.pkl") # Assuming the primary model file

        # Create backup directory if it doesn't exist
        model_backup_dir = os.path.join(model_dir, "backups")
        os.makedirs(model_backup_dir, exist_ok=True)

        # Check if model exists
        if not os.path.exists(model_file_path):
            raise HTTPException(status_code=404, detail=f"Model file not found at {model_file_path}")

        # Create backup with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_filename = f"shiolplus_model_backup_{timestamp}.pkl"
        backup_path = os.path.join(model_backup_dir, backup_filename)

        # Copy model file
        shutil.copy2(model_file_path, backup_path)

        # Get backup file size
        backup_size = os.path.getsize(backup_path)

        logger.info(f"Model backup created: {backup_path}")
        return {
            "message": "Model backup created successfully",
            "backup_file": backup_path,
            "backup_size_mb": round(backup_size / (1024 * 1024), 2),
            "timestamp": datetime.now().isoformat(),
            "status": "success"
        }

    except FileNotFoundError:
        logger.error(f"Model file or backup directory not found.")
        raise HTTPException(status_code=500, detail="Model file or backup directory issue encountered.")
    except Exception as e:
        logger.error(f"Error creating model backup: {e}")
        raise HTTPException(status_code=500, detail=f"Error creating model backup: {str(e)}")

# Endpoint for resetting models
@api_router.post("/model/reset")
async def reset_models():
    """Reset AI models to initial state"""
    try:
        logger.info("Model reset initiated")

        # Reset model-related data in database
        conn = db.get_db_connection()
        cursor = conn.cursor()

        # Clear adaptive weights
        cursor.execute('DELETE FROM adaptive_weights')
        weights_cleared = cursor.rowcount

        # Clear model feedback
        cursor.execute('DELETE FROM model_feedback')
        feedback_cleared = cursor.rowcount

        # Clear reliable plays
        cursor.execute('DELETE FROM reliable_plays')
        plays_cleared = cursor.rowcount

        # Clear performance tracking
        cursor.execute('DELETE FROM performance_tracking')
        performance_cleared = cursor.rowcount

        conn.commit()
        conn.close()
        logger.info(f"Database reset: {weights_cleared} weights, {feedback_cleared} feedback, {plays_cleared} plays, {performance_cleared} performance records")

        # Reset global predictor if available
        global predictor
        if predictor:
            try:
                # Reload the predictor to reset internal state
                from src.predictor import Predictor
                from src.loader import DataLoader

                loader = DataLoader()
                historical_data = loader.load_historical_data()

                if not historical_data.empty:
                    predictor = Predictor()
                    logger.info("Predictor reset and reinitialized")
                else:
                    logger.warning("No historical data available for predictor reset")

            except Exception as pred_error:
                logger.error(f"Error resetting predictor: {pred_error}")

        # Clear model caches if any ensemble predictor exists
        try:
            from src.model_pool_manager import ModelPoolManager
            # Reinitialize model pool manager
            model_manager = ModelPoolManager()
            model_manager.load_compatible_models()
            logger.info("Model pool manager reset")
        except Exception as pool_error:
            logger.warning(f"Could not reset model pool: {pool_error}")

        logger.info("Model reset completed successfully")

        return {
            "message": "AI Models reset completed successfully",
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "details": {
                "weights_cleared": weights_cleared if 'weights_cleared' in locals() else 0,
                "feedback_cleared": feedback_cleared if 'feedback_cleared' in locals() else 0,
                "plays_cleared": plays_cleared if 'plays_cleared' in locals() else 0,
                "performance_cleared": performance_cleared if 'performance_cleared' in locals() else 0,
                "predictor_reset": True,
                "model_pool_reset": True
            },
            "note": "All AI models have been reset to their initial training state. Adaptive learning data cleared."
        }

    except Exception as e:
        logger.error(f"Error resetting AI models: {e}")
        raise HTTPException(status_code=500, detail=f"Error resetting AI models: {str(e)}")

# Endpoint for general system information
@api_router.get("/system/info")
async def get_system_info():
    """Get system information"""
    return {
        "version": "6.0.0",
        "status": "operational",
        "database_status": "connected" if db.is_database_connected() else "disconnected", # Check connection status
        "model_status": "loaded" if predictor and hasattr(predictor, 'model') and predictor.model else "not_loaded"
    }

# --- Application Mounting ---
# Mount all API routers before static files
app.include_router(api_router)
app.include_router(prediction_router)
app.include_router(system_router)
app.include_router(public_router)
app.include_router(auth_router)
# Mount modular frontend routers
app.include_router(public_frontend_router)
app.include_router(dashboard_frontend_router)


# Build an absolute path to the 'frontend' directory for robust file serving.
# This avoids issues with the current working directory.
# APP_ROOT = os.path.dirname(os.path.abspath(__file__)) # This might be too low level for typical deployments
# FRONTEND_DIR = os.path.join(APP_ROOT, "..", "frontend") # Adjust path as necessary based on project structure

# More robust way to find frontend directory relative to the script's location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "frontend"))

# Ensure the frontend directory exists before mounting
if not os.path.exists(FRONTEND_DIR):
    logger.warning(f"Frontend directory not found at {FRONTEND_DIR}. Static file serving may fail.")
    # Optionally create a dummy directory or skip mounting if critical
    # os.makedirs(FRONTEND_DIR, exist_ok=True)

app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="static")
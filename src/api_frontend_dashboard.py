
"""
SHIOL+ Dashboard Frontend API Endpoints
=======================================

API endpoints specifically for the dashboard frontend (dashboard.html).
These endpoints provide administrative access and system management.
"""

from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from loguru import logger
import os
import uuid
import psutil
from pathlib import Path

from src.api_utils import convert_numpy_types
from src.config_manager import ConfigurationManager
import src.database as db

# Create router for dashboard frontend endpoints
dashboard_frontend_router = APIRouter(prefix="/api/v1/dashboard", tags=["dashboard_frontend"])

# Global components (will be injected from main API)
pipeline_orchestrator = None
pipeline_executions = {}
predictor = None

def set_dashboard_components(orch, exec_dict, pred):
    """Set the components for dashboard endpoints."""
    global pipeline_orchestrator, pipeline_executions, predictor
    pipeline_orchestrator = orch
    pipeline_executions = exec_dict
    predictor = pred

@dashboard_frontend_router.get("/system/stats")
async def get_dashboard_system_stats():
    """Get real-time system statistics for dashboard monitoring."""
    try:
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('.')

        # Get pipeline status from orchestrator if available
        pipeline_status = "ready"
        last_execution = "Never"

        if pipeline_orchestrator:
            try:
                status_info = pipeline_orchestrator.get_pipeline_status()
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
        logger.error(f"Error getting dashboard system stats: {e}")
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

@dashboard_frontend_router.get("/database/stats")
async def get_dashboard_database_stats():
    """Get database statistics for dashboard."""
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
        db_path = db.get_db_path()
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
        logger.error(f"Error getting dashboard database stats: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting database stats: {str(e)}")

@dashboard_frontend_router.get("/analytics/performance")
async def get_dashboard_performance_analytics():
    """Get performance analytics for dashboard."""
    try:
        performance_data = db.get_performance_analytics(days_back=30)

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
        logger.error(f"Error getting dashboard performance analytics: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting performance analytics: {str(e)}")

@dashboard_frontend_router.get("/config/load")
async def load_dashboard_configuration():
    """Load system configuration for dashboard."""
    try:
        config_manager = ConfigurationManager()
        config_data = config_manager.load_configuration()

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
        logger.error(f"Error loading dashboard configuration: {e}")
        raise HTTPException(status_code=500, detail=f"Error loading configuration: {str(e)}")

@dashboard_frontend_router.post("/config/save")
async def save_dashboard_configuration(config_data: Dict[str, Any]):
    """Save system configuration from dashboard."""
    try:
        config_manager = ConfigurationManager()
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

            if "weights" in predictions:
                db_config["weights"] = {}
                weights = predictions["weights"]
                db_config["weights"]["probability"] = str(weights.get("probability", 40))
                db_config["weights"]["diversity"] = str(weights.get("diversity", 25))
                db_config["weights"]["historical"] = str(weights.get("historical", 20))
                db_config["weights"]["risk"] = str(weights.get("risk", 15))

        success = config_manager.save_configuration(db_config)

        if success:
            return {"success": True, "message": "Configuration saved successfully to database"}
        else:
            raise HTTPException(status_code=500, detail="Failed to save configuration to database")

    except Exception as e:
        logger.error(f"Error saving dashboard configuration: {e}")
        raise HTTPException(status_code=500, detail=f"Error saving configuration: {str(e)}")

@dashboard_frontend_router.post("/pipeline/execute")
async def execute_dashboard_pipeline(
    background_tasks: BackgroundTasks,
    execution_source: str = "manual_dashboard",
    triggered_by: str = "user_dashboard"
):
    """Execute pipeline from dashboard."""
    try:
        if not pipeline_orchestrator:
            raise HTTPException(status_code=503, detail="Pipeline orchestrator not available")

        # Check if pipeline is already running
        running_executions = [ex for ex in pipeline_executions.values() if ex.get("status") == "running"]
        if running_executions:
            raise HTTPException(
                status_code=409,
                detail=f"Pipeline execution already running (ID: {running_executions[0].get('execution_id')})"
            )

        execution_id = str(uuid.uuid4())[:8]
        current_time = datetime.now()

        pipeline_executions[execution_id] = {
            "execution_id": execution_id,
            "status": "starting",
            "start_time": current_time.isoformat(),
            "current_step": None,
            "steps_completed": 0,
            "total_steps": 7,
            "execution_source": execution_source,
            "triggered_by": triggered_by
        }

        # Execute pipeline in background
        background_tasks.add_task(run_dashboard_pipeline_background, execution_id)

        return {
            "execution_id": execution_id,
            "status": "started",
            "message": "Pipeline execution started in background",
            "timestamp": datetime.now().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error executing dashboard pipeline: {e}")
        raise HTTPException(status_code=500, detail="Error executing pipeline.")

@dashboard_frontend_router.get("/pipeline/status")
async def get_dashboard_pipeline_status():
    """Get pipeline status for dashboard."""
    try:
        if not pipeline_orchestrator:
            raise HTTPException(status_code=503, detail="Pipeline orchestrator not available")

        orchestrator_status = convert_numpy_types(pipeline_orchestrator.get_pipeline_status())

        # Get recent execution history
        recent_executions = []
        sorted_executions = sorted(pipeline_executions.items(), key=lambda item: item[1].get("start_time", ""), reverse=True)
        for exec_id, execution in sorted_executions[:5]:
            recent_executions.append({
                "execution_id": exec_id,
                "status": execution.get("status", "unknown"),
                "start_time": execution.get("start_time"),
                "end_time": execution.get("end_time"),
                "current_step": execution.get("current_step"),
                "steps_completed": execution.get("steps_completed", 0),
                "total_steps": execution.get("total_steps", 7),
                "execution_source": execution.get("execution_source", "unknown")
            })

        # Determine overall pipeline status
        current_status = "idle"
        if pipeline_executions:
            latest_execution = max(pipeline_executions.values(), key=lambda x: x.get("start_time", ""))
            if latest_execution.get("status") == "running":
                current_status = "running"
            elif latest_execution.get("status") == "failed":
                current_status = "failed"
            elif latest_execution.get("status") == "completed":
                current_status = "completed"

        return convert_numpy_types({
            "pipeline_status": {
                "current_status": current_status,
                "last_execution": sorted_executions[0] if sorted_executions else None,
                "recent_execution_history": recent_executions
            },
            "orchestrator_status": orchestrator_status,
            "timestamp": datetime.now().isoformat()
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting dashboard pipeline status: {e}")
        raise HTTPException(status_code=500, detail="Error getting pipeline status.")

async def run_dashboard_pipeline_background(execution_id: str):
    """Run pipeline in background for dashboard."""
    try:
        pipeline_executions[execution_id]["status"] = "running"
        
        logger.info(f"Starting dashboard pipeline execution {execution_id}")

        result = pipeline_orchestrator.run_full_pipeline(
            num_predictions=100,
            requested_steps=None,
            execution_source="manual_dashboard"
        )

        steps_completed = 0
        if result.get("results"):
            steps_completed = sum(1 for step_result in result["results"].values()
                               if step_result.get("status") == "success")

        final_status = result.get("status", "unknown")
        if final_status == "success" and steps_completed != 7:
            final_status = "partial_success"

        pipeline_executions[execution_id].update({
            "status": final_status,
            "end_time": datetime.now().isoformat(),
            "result": result,
            "steps_completed": steps_completed,
            "total_steps": 7,
            "error": result.get("error")
        })

        logger.info(f"Dashboard pipeline execution {execution_id} completed with status: {final_status}")

    except Exception as e:
        pipeline_executions[execution_id].update({
            "status": "failed",
            "end_time": datetime.now().isoformat(),
            "error": str(e),
            "current_step": "exception_handler"
        })
        logger.error(f"Critical error during dashboard pipeline execution {execution_id}: {e}")


"""
SHIOL+ System Management API Endpoints
=====================================

System monitoring, configuration, and management endpoints.
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional, Dict, Any, List
from datetime import datetime
from loguru import logger
import psutil
import os

from src.api_utils import convert_numpy_types
import src.database as db
from src.config_manager import ConfigurationManager

# Create router for system endpoints
system_router = APIRouter(prefix="/api/v1", tags=["system"])


@system_router.get("/system/stats")
async def get_system_stats():
    """Get real-time system statistics for dashboard monitoring."""
    try:
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('.')

        return {
            "cpu_usage": round(cpu_percent, 1),
            "memory_usage": round(memory.percent, 1),
            "disk_usage": round((disk.used / disk.total) * 100, 1),
            "memory_total_gb": round(memory.total / (1024**3), 2),
            "disk_total_gb": round(disk.total / (1024**3), 2),
            "pipeline_status": "ready",
            "last_execution": "Never",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting system stats: {e}")
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


@system_router.get("/database/stats")
async def get_database_stats():
    """Get database statistics for dashboard."""
    try:
        conn = db.get_db_connection()
        cursor = conn.cursor()

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


@system_router.get("/config/load")
async def load_configuration():
    """Load system configuration using hybrid system."""
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
        logger.error(f"Error loading configuration: {e}")
        raise HTTPException(status_code=500, detail=f"Error loading configuration: {str(e)}")


@system_router.post("/config/save")
async def save_configuration(config_data: Dict[str, Any]):
    """Save system configuration using hybrid system."""
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
        logger.error(f"Error saving configuration: {e}")
        raise HTTPException(status_code=500, detail=f"Error saving configuration: {str(e)}")


@system_router.get("/system/info")
async def get_system_info():
    """Get system information."""
    return {
        "version": "6.0.0",
        "status": "operational",
        "database_status": "connected" if db.is_database_connected() else "disconnected",
        "model_status": "loaded"
    }

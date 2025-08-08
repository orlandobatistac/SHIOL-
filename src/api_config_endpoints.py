
from fastapi import APIRouter, HTTPException
from typing import Dict, Any
from loguru import logger
from datetime import datetime

from src.config_manager import ConfigurationManager

config_router = APIRouter(prefix="/config", tags=["Configuration"])

@config_router.get("/load")
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

@config_router.post("/save")
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

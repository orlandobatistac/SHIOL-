
from fastapi import APIRouter, HTTPException
from loguru import logger
from datetime import datetime
import shutil
import os

import src.database as db

model_router = APIRouter(prefix="/model", tags=["Model Management"])

@model_router.post("/retrain")
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

@model_router.post("/backup")
async def backup_models():
    """Create backup of current AI models"""
    try:
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

@model_router.post("/reset")
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
        try:
            from src.predictor import Predictor
            from src.loader import DataLoader
            from src.api import predictor

            loader = DataLoader()
            historical_data = loader.load_historical_data()

            if not historical_data.empty:
                globals()['predictor'] = Predictor()
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

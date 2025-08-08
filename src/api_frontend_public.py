
"""
SHIOL+ Public Frontend API Endpoints
====================================

API endpoints specifically for the public frontend (index.html).
These endpoints provide public access to predictions and historical data.
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional, Dict, Any, List
from datetime import datetime
from loguru import logger

from src.api_utils import convert_numpy_types
from src.database import get_prediction_history, get_predictions_grouped_by_date, calculate_next_drawing_date
import src.database as db
from src.predictor import Predictor
from src.intelligent_generator import DeterministicGenerator

# Create router for public frontend endpoints
public_frontend_router = APIRouter(prefix="/api/v1/public", tags=["public_frontend"])

# Global components (will be injected from main API)
predictor = None
deterministic_generator = None

def set_public_components(pred, det_gen):
    """Set the prediction components for public endpoints."""
    global predictor, deterministic_generator
    predictor = pred
    deterministic_generator = det_gen

@public_frontend_router.get("/smart-predictions")
async def get_public_smart_predictions(
    limit: int = Query(100, ge=1, le=100, description="Number of Smart AI predictions to retrieve")
):
    """
    Get Smart AI predictions for public display on index.html.
    Returns latest Smart AI predictions with simplified format for public consumption.
    """
    try:
        logger.info(f"Received public request for {limit} Smart AI predictions")

        # Calculate next drawing date information
        next_drawing_date = calculate_next_drawing_date()
        current_date = datetime.now()
        next_date = datetime.strptime(next_drawing_date, '%Y-%m-%d')
        days_until_drawing = (next_date - current_date).days

        # Get latest Smart AI predictions from database
        predictions_df = db.get_prediction_history(limit=limit)

        smart_predictions = []
        if not predictions_df.empty:
            for i, pred in enumerate(predictions_df.to_dict('records')):
                try:
                    # Extract and convert individual numbers safely
                    numbers = []
                    for num_key in ["n1", "n2", "n3", "n4", "n5"]:
                        num_val = pred.get(num_key, 0)
                        numbers.append(int(num_val) if num_val is not None else 0)

                    powerball_val = pred.get("powerball", 0)
                    powerball = int(powerball_val) if powerball_val is not None else 0

                    score_val = pred.get("score_total", 0.0)
                    total_score = float(score_val) if score_val is not None else 0.0

                    smart_pred = {
                        "rank": i + 1,
                        "numbers": numbers,
                        "powerball": powerball,
                        "total_score": total_score,
                        "score_details": {
                            "probability": total_score * 0.4,
                            "diversity": total_score * 0.25,
                            "historical": total_score * 0.2,
                            "risk_adjusted": total_score * 0.15
                        },
                        "model_version": str(pred.get("model_version", "pipeline_v1.0")),
                        "dataset_hash": str(pred.get("dataset_hash", "pipeline_generated")),
                        "prediction_id": int(pred.get("id", 0)) if pred.get("id") else 0,
                        "generated_at": str(pred.get("timestamp", "")),
                        "method": "smart_ai_pipeline"
                    }
                    smart_predictions.append(smart_pred)
                except (ValueError, TypeError, KeyError) as e:
                    logger.warning(f"Error converting prediction record {i}: {e}")
                    continue

        # If no predictions in database, generate some as fallback
        if not smart_predictions and predictor:
            logger.warning("No Smart AI predictions found in database, generating new ones...")
            predictions = predictor.predict_diverse_plays(num_plays=min(limit, 10), save_to_log=True)

            for i, pred in enumerate(predictions):
                numbers = [int(x) for x in pred.get("numbers", [])]
                powerball = int(pred.get("powerball", 0))
                total_score = float(pred.get("score_total", 0.0))

                score_details = pred.get("score_details", {})
                safe_score_details = {}
                for key, value in score_details.items():
                    try:
                        if hasattr(value, 'item'):
                            safe_score_details[key] = float(value.item())
                        elif isinstance(value, (int, float)):
                            safe_score_details[key] = float(value)
                        else:
                            safe_score_details[key] = float(str(value)) if str(value).replace('.','').isdigit() else 0.0
                    except:
                        safe_score_details[key] = 0.0

                smart_pred = {
                    "rank": i + 1,
                    "numbers": numbers,
                    "powerball": powerball,
                    "total_score": total_score,
                    "score_details": safe_score_details,
                    "model_version": str(pred.get("model_version", "")),
                    "dataset_hash": str(pred.get("dataset_hash", "")),
                    "prediction_id": int(pred.get("log_id", 0)) if pred.get("log_id") else None,
                    "method": "smart_ai_realtime"
                }
                smart_predictions.append(smart_pred)

        # Calculate statistics
        if smart_predictions:
            avg_score = sum(float(p["total_score"]) for p in smart_predictions) / len(smart_predictions)
            best_score = max(float(p["total_score"]) for p in smart_predictions)
        else:
            avg_score = 0.0
            best_score = 0.0

        return {
            "method": "smart_ai_public",
            "smart_predictions": smart_predictions,
            "total_predictions": len(smart_predictions),
            "average_score": float(avg_score),
            "best_score": float(best_score),
            "data_source": "database" if smart_predictions and "pipeline" in smart_predictions[0].get("method", "") else "realtime_generation",
            "next_drawing": {
                "date": next_drawing_date,
                "formatted_date": next_date.strftime('%B %d, %Y'),
                "days_until": int(days_until_drawing),
                "is_today": days_until_drawing == 0,
                "current_date": datetime.now().strftime('%Y-%m-%d'),
                "current_formatted": datetime.now().strftime('%B %d, %Y')
            },
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error retrieving public Smart AI predictions: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving Smart AI predictions.")

@public_frontend_router.get("/prediction-history")
async def get_public_prediction_history(
    limit: int = Query(25, ge=1, le=50, description="Number of recent predictions to return")
):
    """
    Get public prediction history for display on index.html.
    Returns formatted prediction history for public interface.
    """
    try:
        logger.info(f"Received request for public prediction history (limit: {limit})")

        history = db.get_prediction_history(limit=limit)

        if history.empty:
            return {
                "history": [],
                "count": 0,
                "message": "No prediction history available"
            }

        # Format predictions for public display
        formatted_history = []
        for _, row in history.iterrows():
            formatted_pred = {
                "date": row['timestamp'].strftime('%Y-%m-%d') if hasattr(row['timestamp'], 'strftime') else str(row['timestamp']),
                "formatted_date": row['timestamp'].strftime('%d %b %Y') if hasattr(row['timestamp'], 'strftime') else str(row['timestamp']),
                "numbers": [int(row['n1']), int(row['n2']), int(row['n3']), int(row['n4']), int(row['n5'])],
                "powerball": int(row['powerball']),
                "score": float(row['score_total']),
                "prediction_id": int(row['id'])
            }
            formatted_history.append(formatted_pred)

        return {
            "history": formatted_history,
            "count": len(formatted_history),
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error retrieving public prediction history: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving public prediction history.")

@public_frontend_router.get("/prediction-history-grouped")
async def get_public_grouped_prediction_history(
    limit_dates: int = Query(25, ge=1, le=50, description="Number of recent dates to return")
):
    """
    Get prediction history grouped by date for enhanced public display.
    Returns predictions organized by generation date with statistics.
    """
    try:
        logger.info(f"Received request for public grouped prediction history ({limit_dates} dates)")

        # Get grouped predictions from database
        grouped_data = db.get_predictions_grouped_by_date(limit_dates=limit_dates)

        if not grouped_data:
            return {
                "grouped_dates": [],
                "total_dates": 0,
                "total_predictions": 0,
                "message": "No prediction history available"
            }

        # Calculate summary statistics
        total_predictions = sum(group['total_plays'] for group in grouped_data)
        total_winning_predictions = sum(group['winning_plays'] for group in grouped_data)
        overall_win_rate = (total_winning_predictions / total_predictions * 100) if total_predictions > 0 else 0.0

        return {
            "grouped_dates": grouped_data,
            "total_dates": len(grouped_data),
            "total_predictions": total_predictions,
            "total_winning_predictions": total_winning_predictions,
            "overall_win_rate": f"{overall_win_rate:.1f}%",
            "summary": {
                "best_single_prize": max((group['best_prize_amount'] for group in grouped_data), default=0),
                "total_prize_won": sum(group['total_prize_amount'] for group in grouped_data),
                "dates_with_winners": len([g for g in grouped_data if g['winning_plays'] > 0]),
                "average_plays_per_date": total_predictions / len(grouped_data) if grouped_data else 0
            },
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error retrieving public grouped prediction history: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving public grouped prediction history.")

@public_frontend_router.get("/system-status")
async def get_public_system_status():
    """
    Get basic system status for public display.
    Returns simplified system information without sensitive details.
    """
    try:
        # Basic system status for public consumption
        return {
            "status": "operational",
            "version": "6.0.0",
            "database_status": "connected" if db.is_database_connected() else "disconnected",
            "model_status": "loaded" if predictor else "not_loaded",
            "last_updated": datetime.now().isoformat(),
            "predictions_available": True
        }

    except Exception as e:
        logger.error(f"Error getting public system status: {e}")
        return {
            "status": "error",
            "version": "6.0.0",
            "database_status": "unknown",
            "model_status": "unknown",
            "last_updated": datetime.now().isoformat(),
            "predictions_available": False
        }

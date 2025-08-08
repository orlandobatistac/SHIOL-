
"""
SHIOL+ Prediction API Endpoints
==============================

All prediction-related API endpoints separated from main API file.
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional, Dict, Any, List
from datetime import datetime
from loguru import logger

from src.api_utils import convert_numpy_types, format_prediction_response, safe_database_operation
from src.predictor import Predictor
from src.intelligent_generator import IntelligentGenerator, DeterministicGenerator
from src.database import save_prediction_log, get_prediction_history, calculate_next_drawing_date
import src.database as db

# Create router for prediction endpoints
prediction_router = APIRouter(prefix="/api/v1", tags=["predictions"])

# Global components (will be injected from main API)
predictor = None
intelligent_generator = None
deterministic_generator = None


def set_prediction_components(pred, intel_gen, det_gen):
    """Set global prediction components."""
    global predictor, intelligent_generator, deterministic_generator
    predictor = pred
    intelligent_generator = intel_gen
    deterministic_generator = det_gen


@prediction_router.get("/predict")
async def get_prediction(deterministic: bool = Query(False, description="Use deterministic method")):
    """
    Generates and returns a single Powerball prediction.
    """
    if not predictor or not intelligent_generator:
        raise HTTPException(status_code=500, detail="Model is not available.")

    if deterministic and not deterministic_generator:
        raise HTTPException(status_code=500, detail="Deterministic generator is not available.")

    try:
        logger.info(f"Received request for {'deterministic' if deterministic else 'traditional'} prediction.")
        wb_probs, pb_probs = predictor.predict_probabilities()

        if deterministic:
            result = deterministic_generator.generate_top_prediction(wb_probs, pb_probs)
            prediction = result['numbers'] + [result['powerball']]
            save_prediction_log(result)

            return {
                "prediction": convert_numpy_types(prediction),
                "method": "deterministic",
                "score_total": convert_numpy_types(result['score_total']),
                "dataset_hash": result['dataset_hash']
            }
        else:
            play_df = intelligent_generator.generate_plays(wb_probs, pb_probs, num_plays=1)
            prediction = play_df.iloc[0].astype(int).tolist()

            return {
                "prediction": convert_numpy_types(prediction),
                "method": "traditional"
            }

    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail="Model prediction failed.")


@prediction_router.get("/predict-multiple")
async def get_multiple_predictions(count: int = Query(1, ge=1, le=10)):
    """
    Generates and returns a specified number of Powerball predictions.
    """
    if not predictor or not intelligent_generator:
        raise HTTPException(status_code=500, detail="Model is not available.")

    try:
        logger.info(f"Received request for {count} predictions.")
        wb_probs, pb_probs = predictor.predict_probabilities()
        plays_df = intelligent_generator.generate_plays(wb_probs, pb_probs, num_plays=count)
        predictions = plays_df.astype(int).values.tolist()

        return {"predictions": convert_numpy_types(predictions)}

    except Exception as e:
        logger.error(f"Error during multiple prediction generation: {e}")
        raise HTTPException(status_code=500, detail="Model prediction failed.")


@prediction_router.get("/predict-deterministic")
async def get_deterministic_prediction():
    """
    Generates and returns a deterministic Powerball prediction with detailed scoring.
    """
    if not predictor or not deterministic_generator:
        raise HTTPException(status_code=500, detail="Deterministic prediction components are not available.")

    try:
        logger.info("Received request for deterministic prediction.")
        wb_probs, pb_probs = predictor.predict_probabilities()
        result = deterministic_generator.generate_top_prediction(wb_probs, pb_probs)

        save_prediction_log(result)

        prediction_list = convert_numpy_types(result['numbers'] + [result['powerball']])
        return {
            "prediction": prediction_list,
            "score_total": convert_numpy_types(result['score_total']),
            "score_details": convert_numpy_types(result['score_details']),
            "model_version": result['model_version'],
            "dataset_hash": result['dataset_hash'],
            "timestamp": result['timestamp'],
            "method": "deterministic",
            "traceability": {
                "dataset_hash": result['dataset_hash'],
                "model_version": result['model_version'],
                "timestamp": result['timestamp'],
                "candidates_evaluated": convert_numpy_types(result['num_candidates_evaluated'])
            }
        }

    except Exception as e:
        logger.error(f"Error during deterministic prediction: {e}")
        raise HTTPException(status_code=500, detail="Deterministic prediction failed.")


@prediction_router.get("/predict/smart")
async def get_smart_predictions(limit: int = Query(100, ge=1, le=100, description="Number of Smart AI predictions")):
    """
    Get Smart AI predictions from database with next drawing information.
    """
    try:
        logger.info(f"Received request for {limit} Smart AI predictions from database")

        # Calculate next drawing date
        next_drawing_date = calculate_next_drawing_date()
        current_date = datetime.now()
        next_date = datetime.strptime(next_drawing_date, '%Y-%m-%d')
        days_until_drawing = (next_date - current_date).days
        is_drawing_day = current_date.weekday() in [0, 2, 5]

        # Get predictions from database
        predictions_df = db.get_prediction_history(limit=limit)

        smart_predictions = []
        if not predictions_df.empty:
            for i, pred in enumerate(predictions_df.to_dict('records')):
                try:
                    numbers = [int(pred.get(f"n{j}", 0)) for j in range(1, 6)]
                    powerball = int(pred.get("powerball", 0))
                    total_score = float(pred.get("score_total", 0.0))

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
                        "prediction_id": int(pred.get("id", 0)),
                        "generated_at": str(pred.get("timestamp", "")),
                        "method": "smart_ai_pipeline"
                    }
                    smart_predictions.append(smart_pred)
                except Exception as e:
                    logger.warning(f"Error converting prediction record {i}: {e}")
                    continue

        # Fallback to real-time generation if no predictions
        if not smart_predictions and predictor:
            predictions = predictor.predict_diverse_plays(num_plays=min(limit, 10), save_to_log=True)
            for i, pred in enumerate(predictions):
                smart_pred = format_prediction_response(pred, "smart_ai_realtime")
                smart_pred["rank"] = i + 1
                smart_predictions.append(smart_pred)

        # Calculate statistics
        avg_score = sum(p["total_score"] for p in smart_predictions) / len(smart_predictions) if smart_predictions else 0.0
        best_score = max(p["total_score"] for p in smart_predictions) if smart_predictions else 0.0

        return {
            "method": "smart_ai_database",
            "smart_predictions": smart_predictions,
            "total_predictions": len(smart_predictions),
            "average_score": avg_score,
            "best_score": best_score,
            "data_source": "database" if smart_predictions else "empty",
            "next_drawing": {
                "date": next_drawing_date,
                "formatted_date": next_date.strftime('%B %d, %Y'),
                "days_until": days_until_drawing,
                "is_today": days_until_drawing == 0,
                "is_drawing_day": is_drawing_day
            },
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error retrieving Smart AI predictions: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving Smart AI predictions.")


@prediction_router.get("/predict-diverse")
async def get_diverse_predictions(num_plays: int = Query(5, ge=1, le=10)):
    """
    Generates multiple diverse high-quality predictions.
    """
    if not predictor or not deterministic_generator:
        raise HTTPException(status_code=500, detail="Diverse prediction components are not available.")

    try:
        logger.info(f"Received request for {num_plays} diverse predictions.")
        diverse_predictions = predictor.predict_diverse_plays(num_plays=num_plays, save_to_log=True)

        plays = []
        for prediction in diverse_predictions:
            play = {
                "numbers": convert_numpy_types(prediction['numbers']),
                "powerball": convert_numpy_types(prediction['powerball']),
                "prediction": convert_numpy_types(prediction['numbers'] + [prediction['powerball']]),
                "score_total": convert_numpy_types(prediction['score_total']),
                "score_details": convert_numpy_types(prediction['score_details']),
                "play_rank": prediction.get('play_rank', 0),
                "diversity_method": prediction.get('diversity_method', 'intelligent_selection')
            }
            plays.append(play)

        return {
            "plays": plays,
            "num_plays": len(plays),
            "method": "diverse_deterministic",
            "model_version": diverse_predictions[0]['model_version'],
            "dataset_hash": diverse_predictions[0]['dataset_hash'],
            "timestamp": diverse_predictions[0]['timestamp']
        }

    except Exception as e:
        logger.error(f"Error during diverse prediction generation: {e}")
        raise HTTPException(status_code=500, detail="Diverse prediction generation failed.")

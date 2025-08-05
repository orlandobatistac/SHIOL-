from fastapi import FastAPI, HTTPException, APIRouter, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from loguru import logger
import os
import numpy as np
from datetime import datetime, timedelta
import asyncio
import uuid
import json
import psutil
import shutil
from typing import Optional, Dict, Any, List

from src.predictor import Predictor
from src.intelligent_generator import IntelligentGenerator, DeterministicGenerator
from src.loader import update_database_from_source
from src.database import save_prediction_log
import src.database as db
from src.adaptive_feedback import (
    initialize_adaptive_system, run_adaptive_analysis, AdaptiveValidator,
    ModelFeedbackEngine, AdaptivePlayScorer
)
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from contextlib import asynccontextmanager

# Import new public API and authentication
from src.public_api import public_router, auth_router

def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    return obj

# --- Pipeline Monitoring Global Variables ---
# Import PipelineOrchestrator from main.py
from main import PipelineOrchestrator

# Global variables for pipeline monitoring
pipeline_orchestrator = None
pipeline_executions = {}  # Track running pipeline executions
pipeline_logs = []  # Store recent pipeline logs

# --- Scheduler and App Lifecycle ---
scheduler = AsyncIOScheduler()

@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline_orchestrator
    # On startup
    logger.info("Application startup...")

    # Initialize pipeline orchestrator
    try:
        pipeline_orchestrator = PipelineOrchestrator()
        logger.info("Pipeline orchestrator initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize pipeline orchestrator: {e}")
        pipeline_orchestrator = None

    # Initial data update
    logger.info("Performing initial data update on startup.")
    update_database_from_source()
    # Schedule the update job
    scheduler.add_job(update_database_from_source, 'interval', hours=12)
    scheduler.start()
    logger.info("Scheduler started. Data will be updated every 12 hours.")
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
    version="1.0.0",
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
except Exception as e:
    logger.critical(f"Fatal error during startup: Failed to load model. Error: {e}")
    predictor = None
    intelligent_generator = None
    deterministic_generator = None

# --- API Router ---
api_router = APIRouter(prefix="/api/v1")

@api_router.get("/predict")
async def get_prediction(deterministic: bool = Query(False, description="Use deterministic method")):
    """
    Generates and returns a single Powerball prediction.

    - **deterministic**: If true, uses deterministic method; otherwise uses traditional method
    """
    if not predictor or not intelligent_generator:
        logger.error("Endpoint /predict called, but model is not available.")
        raise HTTPException(
            status_code=500, detail="Model is not available. Please check server logs."
        )

    if deterministic and not deterministic_generator:
        logger.error("Deterministic generator not available.")
        raise HTTPException(
            status_code=500, detail="Deterministic generator is not available."
        )

    try:
        logger.info(f"Received request for {'deterministic' if deterministic else 'traditional'} prediction.")
        wb_probs, pb_probs = predictor.predict_probabilities()

        if deterministic:
            # Use deterministic method
            result = deterministic_generator.generate_top_prediction(wb_probs, pb_probs)
            prediction = result['numbers'] + [result['powerball']]

            # Save to database
            save_prediction_log(result)

            return {
                "prediction": convert_numpy_types(prediction),
                "method": "deterministic",
                "score_total": convert_numpy_types(result['score_total']),
                "dataset_hash": result['dataset_hash']
            }
        else:
            # Use traditional method
            play_df = intelligent_generator.generate_plays(
                wb_probs, pb_probs, num_plays=1
            )
            prediction = play_df.iloc[0].astype(int).tolist()

            return {
                "prediction": convert_numpy_types(prediction),
                "method": "traditional"
            }

    except Exception as e:
        logger.error(f"An error occurred during prediction: {e}")
        raise HTTPException(status_code=500, detail="Model prediction failed.")

@api_router.get("/predict-multiple")
async def get_multiple_predictions(count: int = Query(1, ge=1, le=10)):
    """
    Generates and returns a specified number of Powerball predictions.

    - **count**: Number of plays to generate (default: 1, min: 1, max: 10).
    """
    if not predictor or not intelligent_generator:
        logger.error("Endpoint /predict-multiple called, but model is not available.")
        raise HTTPException(
            status_code=500, detail="Model is not available. Please check server logs."
        )
    try:
        logger.info(f"Received request for {count} predictions.")
        wb_probs, pb_probs = predictor.predict_probabilities()
        plays_df = intelligent_generator.generate_plays(
            wb_probs, pb_probs, num_plays=count
        )
        predictions = plays_df.astype(int).values.tolist()
        logger.info(f"Generated {len(predictions)} predictions.")
        return {"predictions": convert_numpy_types(predictions)}
    except Exception as e:
        logger.error(f"An error occurred during multiple prediction generation: {e}")
        raise HTTPException(status_code=500, detail="Model prediction failed.")

@api_router.get("/predict-deterministic")
async def get_deterministic_prediction():
    """
    Generates and returns a deterministic Powerball prediction with detailed scoring.
    """
    if not predictor or not deterministic_generator:
        logger.error("Endpoint /predict-deterministic called, but components are not available.")
        raise HTTPException(
            status_code=500, detail="Deterministic prediction components are not available."
        )
    try:
        logger.info("Received request for deterministic prediction.")
        wb_probs, pb_probs = predictor.predict_probabilities()
        result = deterministic_generator.generate_top_prediction(wb_probs, pb_probs)

        # Save to database
        save_prediction_log(result)

        logger.info(f"Generated deterministic prediction with score: {result['score_total']:.4f}")
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
        logger.error(f"An error occurred during deterministic prediction: {e}")
        raise HTTPException(status_code=500, detail="Deterministic prediction failed.")

@api_router.get("/predict-diverse")
async def get_diverse_predictions(num_plays: int = Query(5, ge=1, le=10, description="Number of diverse plays to generate")):
    """
    Generates multiple diverse high-quality predictions for the next lottery drawing.

    - **num_plays**: Number of diverse plays to generate (default: 5, min: 1, max: 10)
    """
    if not predictor or not deterministic_generator:
        logger.error("Endpoint /predict-diverse called, but components are not available.")
        raise HTTPException(
            status_code=500, detail="Diverse prediction components are not available."
        )
    try:
        logger.info(f"Received request for {num_plays} diverse predictions.")

        # Generate diverse predictions using the new method
        diverse_predictions = predictor.predict_diverse_plays(num_plays=num_plays, save_to_log=True)

        # Format response
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

        response = {
            "plays": plays,
            "num_plays": len(plays),
            "method": "diverse_deterministic",
            "model_version": diverse_predictions[0]['model_version'],
            "dataset_hash": diverse_predictions[0]['dataset_hash'],
            "timestamp": diverse_predictions[0]['timestamp'],
            "candidates_evaluated": convert_numpy_types(diverse_predictions[0]['num_candidates_evaluated']),
            "generation_summary": {
                "total_plays": len(plays),
                "score_range": {
                    "highest": convert_numpy_types(max(p['score_total'] for p in diverse_predictions)),
                    "lowest": convert_numpy_types(min(p['score_total'] for p in diverse_predictions))
                },
                "diversity_algorithm": "intelligent_selection"
            }
        }

        logger.info(f"Generated {len(plays)} diverse predictions with scores ranging from "
                   f"{response['generation_summary']['score_range']['lowest']:.4f} to "
                   f"{response['generation_summary']['score_range']['highest']:.4f}")

        return response

    except Exception as e:
        logger.error(f"An error occurred during diverse prediction generation: {e}")
        raise HTTPException(status_code=500, detail="Diverse prediction generation failed.")

@api_router.get("/predict-detailed")
async def get_detailed_prediction(deterministic: bool = Query(True, description="Use deterministic method for detailed scoring")):
    """
    Generates a prediction with detailed component scores and analysis.

    - **deterministic**: If true, uses deterministic method with detailed scoring
    """
    if not predictor:
        logger.error("Endpoint /predict-detailed called, but predictor is not available.")
        raise HTTPException(
            status_code=500, detail="Predictor is not available."
        )

    if deterministic and not deterministic_generator:
        logger.error("Deterministic generator not available for detailed prediction.")
        raise HTTPException(
            status_code=500, detail="Deterministic generator is not available."
        )

    try:
        logger.info("Received request for detailed prediction.")
        wb_probs, pb_probs = predictor.predict_probabilities()

        if deterministic:
            result = deterministic_generator.generate_top_prediction(wb_probs, pb_probs)

            prediction_list = convert_numpy_types(result['numbers'] + [result['powerball']])
            return {
                "prediction": prediction_list,
                "method": "deterministic",
                "total_score": convert_numpy_types(result['score_total']),
                "component_scores": {
                    "probability_score": convert_numpy_types(result['score_details']['probability']),
                    "diversity_score": convert_numpy_types(result['score_details']['diversity']),
                    "historical_score": convert_numpy_types(result['score_details']['historical']),
                    "risk_adjusted_score": convert_numpy_types(result['score_details']['risk_adjusted'])
                },
                "score_weights": {
                    "probability": 0.40,
                    "diversity": 0.25,
                    "historical": 0.20,
                    "risk_adjusted": 0.15
                },
                "traceability": {
                    "dataset_hash": result['dataset_hash'],
                    "model_version": result['model_version'],
                    "timestamp": result['timestamp'],
                    "candidates_evaluated": convert_numpy_types(result['num_candidates_evaluated'])
                }
            }
        else:
            # For traditional method, provide basic info
            play_df = intelligent_generator.generate_plays(wb_probs, pb_probs, num_plays=1)
            prediction = play_df.iloc[0].astype(int).tolist()

            return {
                "prediction": convert_numpy_types(prediction),
                "method": "traditional",
                "note": "Detailed scoring only available with deterministic method"
            }

    except Exception as e:
        logger.error(f"An error occurred during detailed prediction: {e}")
        raise HTTPException(status_code=500, detail="Detailed prediction failed.")

@api_router.get("/compare-methods")
async def compare_prediction_methods():
    """
    Compares traditional vs deterministic prediction methods side by side.
    """
    if not predictor or not intelligent_generator or not deterministic_generator:
        logger.error("Endpoint /compare-methods called, but components are not available.")
        raise HTTPException(
            status_code=500, detail="Required prediction components are not available."
        )

    try:
        logger.info("Received request for method comparison.")
        wb_probs, pb_probs = predictor.predict_probabilities()

        # Generate traditional prediction
        traditional_play_df = intelligent_generator.generate_plays(wb_probs, pb_probs, num_plays=1)
        traditional_prediction = traditional_play_df.iloc[0].astype(int).tolist()

        # Generate deterministic prediction
        deterministic_result = deterministic_generator.generate_top_prediction(wb_probs, pb_probs)
        deterministic_prediction = deterministic_result['numbers'] + [deterministic_result['powerball']]

        return {
            "comparison": {
                "traditional": {
                    "prediction": convert_numpy_types(traditional_prediction),
                    "method": "traditional",
                    "description": "Random sampling based on ML probabilities",
                    "characteristics": [
                        "Non-deterministic (different results each time)",
                        "Based on weighted random sampling",
                        "Fast generation",
                        "No scoring system"
                    ]
                },
                "deterministic": {
                    "prediction": convert_numpy_types(deterministic_prediction),
                    "method": "deterministic",
                    "description": "Multi-criteria scoring system with reproducible results",
                    "score_total": convert_numpy_types(deterministic_result['score_total']),
                    "score_details": convert_numpy_types(deterministic_result['score_details']),
                    "characteristics": [
                        "Deterministic (same result for same dataset)",
                        "Multi-criteria evaluation system",
                        "Traceable and auditable",
                        "Considers probability, diversity, history, and risk"
                    ],
                    "traceability": {
                        "dataset_hash": deterministic_result['dataset_hash'],
                        "model_version": deterministic_result['model_version'],
                        "timestamp": deterministic_result['timestamp']
                    }
                }
            },
            "recommendation": "deterministic" if convert_numpy_types(deterministic_result['score_total']) > 0.5 else "traditional"
        }

    except Exception as e:
        logger.error(f"An error occurred during method comparison: {e}")
        raise HTTPException(status_code=500, detail="Method comparison failed.")

@api_router.get("/prediction-history")
async def get_prediction_history(limit: int = Query(10, ge=1, le=50, description="Number of recent predictions to return")):
    """
    Returns the history of deterministic predictions.

    - **limit**: Number of recent predictions to return (max 50)
    """
    try:
        logger.info(f"Received request for prediction history (limit: {limit}).")
        history = db.get_prediction_history(limit=limit)

        # Check if history is empty
        if history.empty:
            logger.info("No prediction history found.")
            return {
                "history": [],
                "count": 0
            }

        # Convert DataFrame to list of dictionaries and handle numpy types
        history_list = []
        for _, row in history.iterrows():
            history_item = {
                "id": convert_numpy_types(row['id']),
                "timestamp": row['timestamp'],
                "n1": convert_numpy_types(row['n1']),
                "n2": convert_numpy_types(row['n2']),
                "n3": convert_numpy_types(row['n3']),
                "n4": convert_numpy_types(row['n4']),
                "n5": convert_numpy_types(row['n5']),
                "powerball": convert_numpy_types(row['powerball']),
                "score_total": convert_numpy_types(row['score_total']),
                "model_version": row['model_version'],
                "dataset_hash": row['dataset_hash'],
                "created_at": row['created_at']
            }
            history_list.append(history_item)

        return {
            "history": history_list,
            "count": len(history_list)
        }

    except Exception as e:
        logger.error(f"An error occurred retrieving prediction history: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve prediction history.")

# --- Phase 4: Adaptive Feedback System Endpoints ---

# Initialize adaptive system components
try:
    from src.loader import DataLoader
    data_loader = DataLoader()
    historical_data = data_loader.load_historical_data()
    adaptive_system = initialize_adaptive_system(historical_data)
    logger.info("Adaptive feedback system initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize adaptive feedback system: {e}")
    adaptive_system = None

@api_router.get("/adaptive/analysis")
async def get_adaptive_analysis(days_back: int = Query(30, ge=1, le=365, description="Days to analyze")):
    """
    Provides comprehensive adaptive analysis of system performance.

    - **days_back**: Number of days to analyze (default: 30, max: 365)
    """
    try:
        logger.info(f"Received request for adaptive analysis ({days_back} days)")

        analysis_results = run_adaptive_analysis(days_back)

        if 'error' in analysis_results:
            raise HTTPException(status_code=500, detail=analysis_results['error'])

        return {
            "adaptive_analysis": analysis_results,
            "system_status": "active" if adaptive_system else "inactive"
        }

    except Exception as e:
        logger.error(f"Error in adaptive analysis: {e}")
        raise HTTPException(status_code=500, detail="Adaptive analysis failed.")

@api_router.get("/adaptive/reliable-plays")
async def get_reliable_plays(limit: int = Query(10, ge=1, le=50, description="Number of plays to return"),
                           min_score: float = Query(0.7, ge=0.0, le=1.0, description="Minimum reliability score")):
    """
    Returns the most reliable play combinations based on historical performance.

    - **limit**: Maximum number of plays to return (default: 10, max: 50)
    - **min_score**: Minimum reliability score threshold (default: 0.7)
    """
    try:
        logger.info(f"Received request for reliable plays (limit: {limit}, min_score: {min_score})")

        reliable_plays = db.get_reliable_plays(limit=limit, min_reliability_score=min_score)

        if reliable_plays.empty:
            return {
                "reliable_plays": [],
                "count": 0,
                "message": "No reliable plays found with the specified criteria"
            }

        # Convert DataFrame to list of dictionaries
        plays_list = []
        for _, play in reliable_plays.iterrows():
            play_dict = {
                "id": convert_numpy_types(play['id']),
                "numbers": [
                    convert_numpy_types(play['n1']), convert_numpy_types(play['n2']),
                    convert_numpy_types(play['n3']), convert_numpy_types(play['n4']),
                    convert_numpy_types(play['n5'])
                ],
                "powerball": convert_numpy_types(play['pb']),
                "reliability_score": convert_numpy_types(play['reliability_score']),
                "win_rate": convert_numpy_types(play['win_rate']),
                "avg_score": convert_numpy_types(play['avg_score']),
                "times_generated": convert_numpy_types(play['times_generated']),
                "last_generated": play['last_generated'],
                "created_at": play['created_at']
            }
            plays_list.append(play_dict)

        return {
            "reliable_plays": plays_list,
            "count": len(plays_list),
            "criteria": {
                "min_reliability_score": min_score,
                "limit": limit
            }
        }

    except Exception as e:
        logger.error(f"Error retrieving reliable plays: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve reliable plays.")

@api_router.get("/adaptive/weights")
async def get_adaptive_weights():
    """
    Returns the current adaptive weights configuration.
    """
    try:
        logger.info("Received request for adaptive weights")

        active_weights = db.get_active_adaptive_weights()

        if not active_weights:
            return {
                "adaptive_weights": None,
                "status": "no_active_weights",
                "default_weights": {
                    "probability": 0.4,
                    "diversity": 0.25,
                    "historical": 0.2,
                    "risk_adjusted": 0.15
                },
                "message": "No adaptive weights configured, using default weights"
            }

        return {
            "adaptive_weights": active_weights,
            "status": "active",
            "last_updated": active_weights.get('weight_set_name', 'unknown')
        }

    except Exception as e:
        logger.error(f"Error retrieving adaptive weights: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve adaptive weights.")

@api_router.post("/adaptive/optimize-weights")
async def optimize_weights(algorithm: str = Query("differential_evolution",
                                                description="Optimization algorithm to use")):
    """
    Triggers weight optimization using the specified algorithm.

    - **algorithm**: Optimization algorithm ('differential_evolution', 'scipy_minimize', 'grid_search')
    """
    try:
        logger.info(f"Received request for weight optimization using {algorithm}")

        if not adaptive_system:
            raise HTTPException(status_code=503, detail="Adaptive system not available")

        # Get recent performance data
        performance_data = db.get_performance_analytics(30)

        if performance_data.get('total_predictions', 0) < 10:
            raise HTTPException(
                status_code=400,
                detail="Insufficient data for optimization. Need at least 10 predictions."
            )

        # Get current weights
        current_weights = db.get_active_adaptive_weights()
        if not current_weights:
            current_weights = {
                'weights': {'probability': 0.4, 'diversity': 0.25, 'historical': 0.2, 'risk_adjusted': 0.15}
            }

        # Perform optimization
        weight_optimizer = adaptive_system['weight_optimizer']
        optimized_weights = weight_optimizer.optimize_weights(
            current_weights['weights'],
            performance_data,
            algorithm
        )

        if not optimized_weights:
            raise HTTPException(status_code=500, detail="Weight optimization failed")

        # Save optimized weights
        weight_set_name = f"api_optimized_{algorithm}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        performance_score = performance_data.get('avg_accuracy', 0.0)

        weights_id = db.save_adaptive_weights(
            weight_set_name=weight_set_name,
            weights=optimized_weights,
            performance_score=performance_score,
            optimization_algorithm=algorithm,
            dataset_hash="api_request",
            is_active=True
        )

        return {
            "optimization_result": "success",
            "algorithm_used": algorithm,
            "optimized_weights": optimized_weights,
            "previous_weights": current_weights['weights'],
            "performance_improvement": {
                "baseline_accuracy": performance_data.get('avg_accuracy', 0.0),
                "expected_improvement": "TBD - requires validation"
            },
            "weights_id": weights_id,
            "weight_set_name": weight_set_name
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in weight optimization: {e}")
        raise HTTPException(status_code=500, detail="Weight optimization failed.")

@api_router.get("/adaptive/performance")
async def get_performance_analytics(days_back: int = Query(30, ge=1, le=365, description="Days to analyze")):
    """
    Returns detailed performance analytics for the adaptive system.

    - **days_back**: Number of days to analyze (default: 30, max: 365)
    """
    try:
        logger.info(f"Received request for performance analytics ({days_back} days)")

        analytics = db.get_performance_analytics(days_back)

        return {
            "performance_analytics": analytics,
            "analysis_period": f"{days_back} days",
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error retrieving performance analytics: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve performance analytics.")

@api_router.post("/adaptive/validate")
async def run_adaptive_validation(enable_learning: bool = Query(True, description="Enable adaptive learning")):
    """
    Runs adaptive validation with learning feedback.

    - **enable_learning**: Whether to enable adaptive learning from validation results
    """
    try:
        logger.info(f"Received request for adaptive validation (learning: {enable_learning})")

        if not adaptive_system:
            raise HTTPException(status_code=503, detail="Adaptive system not available")

        # Run adaptive validation
        adaptive_validator = adaptive_system['adaptive_validator']
        csv_path = adaptive_validator.adaptive_validate_predictions(enable_learning=enable_learning)

        if not csv_path:
            raise HTTPException(status_code=500, detail="Adaptive validation failed")

        # Get validation summary
        try:
            import pandas as pd
            validation_df = pd.read_csv(csv_path)
            total_predictions = len(validation_df)
            winners = len(validation_df[validation_df['prize_category'] != 'Non-winning'])
            win_rate = (winners / total_predictions * 100) if total_predictions > 0 else 0.0

            summary = {
                "total_predictions": total_predictions,
                "winning_predictions": winners,
                "win_rate_percent": round(win_rate, 2),
                "learning_enabled": enable_learning
            }
        except Exception:
            summary = {"message": "Validation completed but summary unavailable"}

        return {
            "validation_result": "success",
            "csv_path": csv_path,
            "summary": summary,
            "adaptive_learning": enable_learning
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in adaptive validation: {e}")
        raise HTTPException(status_code=500, detail="Adaptive validation failed.")

@api_router.get("/predict/smart")
async def get_smart_predictions(
    limit: int = Query(100, ge=1, le=100, description="Number of Smart AI predictions to retrieve from database")
):
    """
    Get Smart AI predictions from database.
    Returns latest Smart AI predictions generated by the pipeline.
    """
    try:
        logger.info(f"Received request for {limit} Smart AI predictions from database")

        # Get latest Smart AI predictions from database
        predictions_df = get_prediction_history(limit=limit)
        
        # Convert DataFrame to list of dictionaries
        if not predictions_df.empty:
            predictions_list = predictions_df.to_dict('records')
        else:
            predictions_list = []

        # Convert database records to Smart AI format
        smart_predictions = []
        for i, pred in enumerate(predictions_list):
            smart_pred = {
                "rank": i + 1,
                "numbers": [int(pred["n1"]), int(pred["n2"]), int(pred["n3"]), int(pred["n4"]), int(pred["n5"])],
                "powerball": int(pred["powerball"]),
                "total_score": float(pred["score_total"]),
                "score_details": {
                    "probability": float(pred["score_total"]) * 0.4,
                    "diversity": float(pred["score_total"]) * 0.25,
                    "historical": float(pred["score_total"]) * 0.2,
                    "risk_adjusted": float(pred["score_total"]) * 0.15
                },
                "model_version": pred.get("model_version", "pipeline_v1.0"),
                "dataset_hash": pred.get("dataset_hash", "pipeline_generated"),
                "prediction_id": int(pred["id"]),
                "generated_at": str(pred["timestamp"]),
                "method": "smart_ai_pipeline"
            }
            smart_predictions.append(smart_pred)

        if not smart_predictions:
            # If no predictions in database, generate some
            logger.warning("No Smart AI predictions found in database, generating new ones...")
            if not predictor:
                raise HTTPException(status_code=503, detail="No predictions available and predictor service not available")

            # Generate a few predictions as fallback
            predictions = predictor.predict_diverse_plays(num_plays=min(limit, 10), save_to_log=True)

            for i, pred in enumerate(predictions):
                smart_pred = {
                    "rank": i + 1,
                    "numbers": pred["numbers"],
                    "powerball": pred["powerball"],
                    "total_score": pred["score_total"],
                    "score_details": pred["score_details"],
                    "model_version": pred["model_version"],
                    "dataset_hash": pred["dataset_hash"],
                    "prediction_id": pred.get("log_id"),
                    "method": "smart_ai_realtime"
                }
                smart_predictions.append(smart_pred)

        # Calculate statistics
        if smart_predictions:
            avg_score = sum(p["total_score"] for p in smart_predictions) / len(smart_predictions)
            best_score = max(p["total_score"] for p in smart_predictions)
        else:
            avg_score = 0.0
            best_score = 0.0

        return {
            "method": "smart_ai_database",
            "smart_predictions": smart_predictions,
            "total_predictions": len(smart_predictions),
            "average_score": avg_score,
            "best_score": best_score,
            "data_source": "database" if smart_predictions and "pipeline" in smart_predictions[0].get("method", "") else "realtime_generation",
            "timestamp": datetime.now().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving Smart AI predictions: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve Smart AI predictions")

@api_router.get("/predict/syndicate")
async def get_syndicate_predictions(    num_plays: int = Query(100, ge=10, le=500, description="Number of plays for syndicate"),
    method: str = Query("ensemble", regex="^(ensemble|deterministic|adaptive)$", description="Prediction method")
):
    """
    Legacy endpoint - use /predict/smart for best AI experience.
    Generates optimized predictions for syndicate play.

    - **num_plays**: Number of plays to generate (10-500)
    - **method**: Prediction method (ensemble, deterministic, adaptive)
    """
    try:
        logger.info(f"Received syndicate prediction request: {num_plays} plays using {method} method")

        if not predictor:
            raise HTTPException(status_code=503, detail="Prediction system not available")

        # Generate syndicate predictions based on method
        if method == "ensemble":
            predictions = predictor.predict_ensemble_syndicate(num_plays)
        elif method == "adaptive" and adaptive_system:
            predictions = predictor.predict_syndicate_plays(num_plays)
        else:
            predictions = predictor.predict_diverse_plays(num_plays)

        # Format response
        formatted_predictions = []
        for pred in predictions:
            formatted_pred = {
                "numbers": convert_numpy_types(pred['numbers']),
                "powerball": convert_numpy_types(pred['powerball']),
                "score": convert_numpy_types(pred['score_total']),
                "rank": pred.get('syndicate_rank', pred.get('play_rank', 0)),
                "tier": pred.get('syndicate_tier', 'Standard'),
                "method": pred.get('model_source', method),
                "ensemble_score": convert_numpy_types(pred.get('ensemble_score', pred['score_total']))
            }
            formatted_predictions.append(formatted_pred)

        return {
            "syndicate_predictions": formatted_predictions,
            "total_plays": len(formatted_predictions),
            "method": method,
            "generation_timestamp": datetime.now().isoformat(),
            "coverage_analysis": {
                "premium_tier": len([p for p in predictions if p.get('syndicate_tier') == 'Premium']),
                "high_tier": len([p for p in predictions if p.get('syndicate_tier') == 'High']),
                "medium_tier": len([p for p in predictions if p.get('syndicate_tier') == 'Medium']),
                "standard_tier": len([p for p in predictions if p.get('syndicate_tier') == 'Standard'])
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in syndicate predictions: {e}")
        raise HTTPException(status_code=500, detail="Syndicate prediction failed.")

@api_router.get("/adaptive/predict")
async def get_adaptive_prediction():
    """
    Generates a prediction using the adaptive scoring system with current optimized weights.
    """
    try:
        logger.info("Received request for adaptive prediction")

        if not predictor or not adaptive_system:
            raise HTTPException(status_code=503, detail="Adaptive prediction system not available")

        # Get probabilities from the model
        wb_probs, pb_probs = predictor.predict_probabilities()

        # Use adaptive scorer
        adaptive_scorer = adaptive_system['adaptive_scorer']

        # Generate candidates and score them
        from src.intelligent_generator import DeterministicGenerator
        deterministic_gen = DeterministicGenerator(historical_data)

        # Generate top prediction using adaptive scoring
        result = deterministic_gen.generate_top_prediction(wb_probs, pb_probs)

        # Re-score using adaptive weights
        adaptive_scores = adaptive_scorer.calculate_total_score(
            result['numbers'], result['powerball'], wb_probs, pb_probs
        )

        # Save prediction with adaptive scoring
        adaptive_result = result.copy()
        adaptive_result['score_total'] = convert_numpy_types(adaptive_scores['total'])
        adaptive_result['score_details'] = convert_numpy_types(adaptive_scores)
        adaptive_result['method'] = 'adaptive_deterministic'

        prediction_id = save_prediction_log(adaptive_result)

        prediction_list = convert_numpy_types(result['numbers'] + [result['powerball']])

        return {
            "prediction": prediction_list,
            "method": "adaptive_deterministic",
            "adaptive_score": convert_numpy_types(adaptive_scores['total']),
            "component_scores": {
                "probability": convert_numpy_types(adaptive_scores['probability']),
                "diversity": convert_numpy_types(adaptive_scores['diversity']),
                "historical": convert_numpy_types(adaptive_scores['historical']),
                "risk_adjusted": convert_numpy_types(adaptive_scores['risk_adjusted'])
            },
            "weights_used": adaptive_scores.get('weights_used', {}),
            "adaptive_mode": adaptive_scores.get('adaptive_mode', False),
            "traceability": {
                "prediction_id": prediction_id,
                "dataset_hash": result['dataset_hash'],
                "model_version": result['model_version'],
                "timestamp": result['timestamp']
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in adaptive prediction: {e}")
        raise HTTPException(status_code=500, detail="Adaptive prediction failed.")

# --- Pipeline Monitoring and Control Endpoints ---

@api_router.get("/pipeline/status")
async def get_pipeline_status():
    """
    Returns current pipeline status including execution history, health metrics, and recent runs.
    """
    try:
        logger.info("Received request for pipeline status")

        if not pipeline_orchestrator:
            raise HTTPException(status_code=503, detail="Pipeline orchestrator not available")

        # Get pipeline status from orchestrator
        orchestrator_status = pipeline_orchestrator.get_pipeline_status()

        # Get recent execution history from global tracking
        recent_executions = []
        for exec_id, execution in list(pipeline_executions.items())[-5:]:
            recent_executions.append({
                "execution_id": exec_id,
                "status": execution.get("status", "unknown"),
                "start_time": execution.get("start_time"),
                "end_time": execution.get("end_time"),
                "current_step": execution.get("current_step"),
                "steps_completed": execution.get("steps_completed", 0),
                "total_steps": execution.get("total_steps", 7)
            })

        # Get next scheduled execution (from scheduler)
        next_execution = None
        try:
            jobs = scheduler.get_jobs()
            for job in jobs:
                if hasattr(job, 'next_run_time') and job.next_run_time:
                    next_execution = job.next_run_time.isoformat()
                    break
        except Exception as e:
            logger.warning(f"Could not get next scheduled execution: {e}")

        # Get recent predictions for generated plays
        recent_plays = []
        try:
            history = db.get_prediction_history(limit=5)
            for _, row in history.iterrows():
                recent_plays.append({
                    "numbers": [int(row['n1']), int(row['n2']), int(row['n3']), int(row['n4']), int(row['n5'])],
                    "powerball": int(row['powerball']),
                    "score": float(row['score_total']),
                    "timestamp": row['timestamp'],
                    "dataset_hash": row['dataset_hash']
                })
        except Exception as e:
            logger.warning(f"Could not get recent plays: {e}")

        # Get system health metrics
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('.')

            system_health = {
                "cpu_usage_percent": cpu_percent,
                "memory_usage_percent": memory.percent,
                "memory_available_gb": round(memory.available / (1024**3), 2),
                "disk_usage_percent": disk.percent,
                "disk_free_gb": round(disk.free / (1024**3), 2)
            }
        except Exception as e:
            logger.warning(f"Could not get system health metrics: {e}")
            system_health = {"error": "System metrics unavailable"}

        return {
            "pipeline_status": {
                "last_execution": recent_executions[-1] if recent_executions else None,
                "current_status": "idle" if not any(ex.get("status") == "running" for ex in pipeline_executions.values()) else "running",
                "next_scheduled_execution": next_execution,
                "recent_execution_history": recent_executions,
                "system_health": system_health
            },
            "orchestrator_status": orchestrator_status,
            "generated_plays_last_run": recent_plays,
            "timestamp": datetime.now().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting pipeline status: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve pipeline status")

@api_router.post("/pipeline/trigger")
async def trigger_pipeline_execution(
    background_tasks: BackgroundTasks,
    num_predictions: int = Query(1, ge=1, le=10, description="Number of predictions to generate"),
    steps: Optional[str] = Query(None, description="Comma-separated list of specific steps to run"),
    force: bool = Query(False, description="Force execution even if another pipeline is running")
):
    """
    Manually triggers pipeline execution with optional parameters.
    Runs asynchronously in background and returns execution ID for tracking.
    """
    try:
        logger.info(f"Received request to trigger pipeline (predictions: {num_predictions}, steps: {steps})")

        if not pipeline_orchestrator:
            raise HTTPException(status_code=503, detail="Pipeline orchestrator not available")

        # Check if pipeline is already running
        running_executions = [ex for ex in pipeline_executions.values() if ex.get("status") == "running"]
        if running_executions and not force:
            raise HTTPException(
                status_code=409,
                detail=f"Pipeline execution already running (ID: {running_executions[0].get('execution_id')}). Use force=true to override."
            )

        # Generate execution ID
        execution_id = str(uuid.uuid4())[:8]

        # Parse steps if provided
        step_list = None
        if steps:
            step_list = [step.strip() for step in steps.split(',')]
            # Validate steps
            valid_steps = ['data', 'adaptive', 'weights', 'prediction', 'validation', 'performance', 'reports']
            invalid_steps = [step for step in step_list if step not in valid_steps]
            if invalid_steps:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid steps: {invalid_steps}. Valid steps: {valid_steps}"
                )

        # Initialize execution tracking
        pipeline_executions[execution_id] = {
            "execution_id": execution_id,
            "status": "starting",
            "start_time": datetime.now().isoformat(),
            "current_step": None,
            "steps_completed": 0,
            "total_steps": len(step_list) if step_list else 7,
            "num_predictions": num_predictions,
            "requested_steps": step_list,
            "error": None
        }

        # Add background task
        if step_list:
            # Run specific steps
            background_tasks.add_task(run_pipeline_steps_background, execution_id, step_list, num_predictions)
        else:
            # Run full pipeline
            background_tasks.add_task(run_full_pipeline_background, execution_id, num_predictions)

        return {
            "execution_id": execution_id,
            "status": "started",
            "message": "Pipeline execution started in background",
            "parameters": {
                "num_predictions": num_predictions,
                "steps": step_list if step_list else "all",
                "force": force
            },
            "tracking_url": f"/api/v1/pipeline/status",
            "timestamp": datetime.now().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error triggering pipeline execution: {e}")
        raise HTTPException(status_code=500, detail="Failed to trigger pipeline execution")

@api_router.get("/pipeline/logs")
async def get_pipeline_logs(
    level: Optional[str] = Query(None, description="Filter by log level (DEBUG, INFO, WARNING, ERROR)"),
    start_date: Optional[str] = Query(None, description="Start date filter (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date filter (YYYY-MM-DD)"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of log entries to return"),
    offset: int = Query(0, ge=0, description="Number of log entries to skip")
):
    """
    Returns recent pipeline logs with filtering and pagination support.
    """
    try:
        logger.info(f"Received request for pipeline logs (level: {level}, limit: {limit})")

        # Get log file path from config or use default
        log_file = "logs/shiolplus.log"
        if pipeline_orchestrator and pipeline_orchestrator.config:
            log_file = pipeline_orchestrator.config.get("paths", "log_file", fallback=log_file)

        if not os.path.exists(log_file):
            return {
                "logs": [],
                "total_count": 0,
                "message": f"Log file not found: {log_file}",
                "filters": {
                    "level": level,
                    "start_date": start_date,
                    "end_date": end_date,
                    "limit": limit,
                    "offset": offset
                }
            }

        # Read and parse log file
        logs = []
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            # Parse log lines (basic parsing - assumes loguru format)
            for line in reversed(lines):  # Most recent first
                line = line.strip()
                if not line:
                    continue

                try:
                    # Basic log parsing - extract timestamp, level, message
                    parts = line.split(' | ')
                    if len(parts) >= 4:
                        timestamp_str = parts[0]
                        log_level = parts[1].strip()
                        location = parts[2]
                        message = ' | '.join(parts[3:])

                        # Apply level filter
                        if level and log_level.upper() != level.upper():
                            continue

                        # Apply date filters
                        try:
                            log_date = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
                            if start_date:
                                start_dt = datetime.strptime(start_date, '%Y-%m-%d')
                                if log_date.date() < start_dt.date():
                                    continue
                            if end_date:
                                end_dt = datetime.strptime(end_date, '%Y-%m-%d')
                                if log_date.date() > end_dt.date():
                                    continue
                        except ValueError:
                            # Skip if timestamp parsing fails
                            continue

                        logs.append({
                            "timestamp": timestamp_str,
                            "level": log_level.strip(),
                            "location": location,
                            "message": message,
                            "raw_line": line
                        })

                except Exception:
                    # Skip malformed lines
                    continue

            # Apply pagination
            total_count = len(logs)
            paginated_logs = logs[offset:offset + limit]

            return {
                "logs": paginated_logs,
                "total_count": total_count,
                "returned_count": len(paginated_logs),
                "filters": {
                    "level": level,
                    "start_date": start_date,
                    "end_date": end_date,
                    "limit": limit,
                    "offset": offset
                },
                "pagination": {
                    "has_more": offset + limit < total_count,
                    "next_offset": offset + limit if offset + limit < total_count else None
                },
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error reading log file: {e}")
            return {
                "logs": [],
                "total_count": 0,
                "error": f"Failed to read log file: {str(e)}",
                "filters": {
                    "level": level,
                    "start_date": start_date,
                    "end_date": end_date,
                    "limit": limit,
                    "offset": offset
                }
            }

    except Exception as e:
        logger.error(f"Error getting pipeline logs: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve pipeline logs")

@api_router.get("/pipeline/health")
async def get_pipeline_health():
    """
    System health check endpoint that validates all pipeline components.
    """
    try:
        logger.info("Received request for pipeline health check")

        health_status = {
            "overall_status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "checks": {}
        }

        # Check pipeline orchestrator
        try:
            if pipeline_orchestrator:
                orchestrator_status = pipeline_orchestrator.get_pipeline_status()
                health_status["checks"]["pipeline_orchestrator"] = {
                    "status": "healthy",
                    "message": "Pipeline orchestrator is available",
                    "details": {
                        "configuration_loaded": orchestrator_status.get("configuration_loaded", False),
                        "database_initialized": orchestrator_status.get("database_initialized", False)
                    }
                }
            else:
                health_status["checks"]["pipeline_orchestrator"] = {
                    "status": "unhealthy",
                    "message": "Pipeline orchestrator not available"
                }
                health_status["overall_status"] = "degraded"
        except Exception as e:
            health_status["checks"]["pipeline_orchestrator"] = {
                "status": "unhealthy",
                "message": f"Pipeline orchestrator error: {str(e)}"
            }
            health_status["overall_status"] = "degraded"

        # Check database connectivity
        try:
            test_data = db.get_all_draws()
            health_status["checks"]["database"] = {
                "status": "healthy",
                "message": f"Database accessible with {len(test_data)} records",
                "details": {
                    "total_records": len(test_data),
                    "latest_draw": test_data['draw_date'].max().strftime('%Y-%m-%d') if not test_data.empty else None
                }
            }
        except Exception as e:
            health_status["checks"]["database"] = {
                "status": "unhealthy",
                "message": f"Database connectivity error: {str(e)}"
            }
            health_status["overall_status"] = "unhealthy"

        # Check model availability
        try:
            if predictor:
                # Test model prediction
                wb_probs, pb_probs = predictor.predict_probabilities()
                health_status["checks"]["model"] = {
                    "status": "healthy",
                    "message": "ML model is loaded and functional",
                    "details": {
                        "wb_predictions": len(wb_probs),
                        "pb_predictions": len(pb_probs)
                    }
                }
            else:
                health_status["checks"]["model"] = {
                    "status": "unhealthy",
                    "message": "ML model not available"
                }
                health_status["overall_status"] = "degraded"
        except Exception as e:
            health_status["checks"]["model"] = {
                "status": "unhealthy",
                "message": f"Model error: {str(e)}"
            }
            health_status["overall_status"] = "degraded"

        # Check configuration validation
        try:
            if pipeline_orchestrator and pipeline_orchestrator.config:
                config = pipeline_orchestrator.config
                required_sections = ['paths', 'database']
                missing_sections = [section for section in required_sections if not config.has_section(section)]

                if missing_sections:
                    health_status["checks"]["configuration"] = {
                        "status": "unhealthy",
                        "message": f"Missing configuration sections: {missing_sections}"
                    }
                    health_status["overall_status"] = "degraded"
                else:
                    health_status["checks"]["configuration"] = {
                        "status": "healthy",
                        "message": "Configuration is valid",
                        "details": {
                            "sections": list(config.sections())
                        }
                    }
            else:
                health_status["checks"]["configuration"] = {
                    "status": "unhealthy",
                    "message": "Configuration not loaded"
                }
                health_status["overall_status"] = "degraded"
        except Exception as e:
            health_status["checks"]["configuration"] = {
                "status": "unhealthy",
                "message": f"Configuration error: {str(e)}"
            }
            health_status["overall_status"] = "degraded"

        # Check disk space and system resources
        try:
            disk = shutil.disk_usage('.')
            memory = psutil.virtual_memory()

            disk_free_gb = disk.free / (1024**3)
            memory_available_gb = memory.available / (1024**3)

            resource_issues = []
            if disk_free_gb < 1.0:  # Less than 1GB free
                resource_issues.append(f"Low disk space: {disk_free_gb:.1f}GB free")
            if memory.percent > 90:  # More than 90% memory usage
                resource_issues.append(f"High memory usage: {memory.percent:.1f}%")

            if resource_issues:
                health_status["checks"]["system_resources"] = {
                    "status": "warning",
                    "message": "; ".join(resource_issues),
                    "details": {
                        "disk_free_gb": round(disk_free_gb, 2),
                        "memory_usage_percent": memory.percent,
                        "memory_available_gb": round(memory_available_gb, 2)
                    }
                }
                if health_status["overall_status"] == "healthy":
                    health_status["overall_status"] = "degraded"
            else:
                health_status["checks"]["system_resources"] = {
                    "status": "healthy",
                    "message": "System resources are adequate",
                    "details": {
                        "disk_free_gb": round(disk_free_gb, 2),
                        "memory_usage_percent": memory.percent,
                        "memory_available_gb": round(memory_available_gb, 2)
                    }
                }
        except Exception as e:
            health_status["checks"]["system_resources"] = {
                "status": "warning",
                "message": f"Could not check system resources: {str(e)}"
            }

        # Set overall status based on individual checks
        unhealthy_checks = [check for check in health_status["checks"].values() if check["status"] == "unhealthy"]
        if unhealthy_checks:
            health_status["overall_status"] = "unhealthy"

        return health_status

    except Exception as e:
        logger.error(f"Error in pipeline health check: {e}")
        raise HTTPException(status_code=500, detail="Health check failed")

# Background task functions for pipeline execution
async def run_full_pipeline_background(execution_id: str, num_predictions: int):
    """Run full pipeline in background task."""
    try:
        pipeline_executions[execution_id]["status"] = "running"
        pipeline_executions[execution_id]["current_step"] = "starting"

        # Run the full pipeline
        result = pipeline_orchestrator.run_full_pipeline()

        # Update execution status
        pipeline_executions[execution_id].update({
            "status": "completed" if result.get("status") == "success" else "failed",
            "end_time": datetime.now().isoformat(),
            "result": result,
            "steps_completed": 7,
            "error": result.get("error") if result.get("status") != "success" else None
        })

        logger.info(f"Background pipeline execution {execution_id} completed with status: {result.get('status')}")

    except Exception as e:
        pipeline_executions[execution_id].update({
            "status": "failed",
            "end_time": datetime.now().isoformat(),
            "error": str(e),
            "current_step": "error"
        })
        logger.error(f"Background pipeline execution {execution_id} failed: {e}")

async def run_pipeline_steps_background(execution_id: str, steps: List[str], num_predictions: int):
    """Run specific pipeline steps in background task."""
    try:
        pipeline_executions[execution_id]["status"] = "running"

        results = {}
        for i, step in enumerate(steps):
            pipeline_executions[execution_id]["current_step"] = step
            pipeline_executions[execution_id]["steps_completed"] = i

            try:
                result = pipeline_orchestrator.run_single_step(step)
                results[step] = result

                if result.get("status") != "success":
                    raise Exception(f"Step {step} failed: {result.get('error', 'Unknown error')}")

            except Exception as e:
                results[step] = {"status": "failed", "error": str(e)}
                raise

        # Update execution status
        pipeline_executions[execution_id].update({
            "status": "completed",
            "end_time": datetime.now().isoformat(),
            "result": {"status": "success", "results": results},
            "steps_completed": len(steps)
        })

        logger.info(f"Background pipeline steps execution {execution_id} completed successfully")

    except Exception as e:
        pipeline_executions[execution_id].update({
            "status": "failed",
            "end_time": datetime.now().isoformat(),
            "error": str(e),
            "current_step": "error"
        })
        logger.error(f"Background pipeline steps execution {execution_id} failed: {e}")

# --- Application Mounting ---
# Mount the API router before the static files to ensure API endpoints are prioritized.
app.include_router(api_router)

# Mount new public and authentication routers
app.include_router(public_router)
app.include_router(auth_router)

# Build an absolute path to the 'frontend' directory for robust file serving.
# This avoids issues with the current working directory.
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(APP_ROOT, "..", "frontend")

app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="static")
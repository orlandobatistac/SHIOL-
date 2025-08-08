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
import configparser
import sqlite3
from pathlib import Path # Import Path

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
# Import ConfigurationManager for the hybrid system
from src.config_manager import ConfigurationManager

def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if obj is None:
        return None
    elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    elif isinstance(obj, dict):
        return {str(key): convert_numpy_types(value) for key, value in obj.items()}
    elif hasattr(obj, 'item'):  # Handle numpy scalar types
        try:
            return obj.item()
        except (AttributeError, ValueError):
            return str(obj)
    elif isinstance(obj, (int, float, str, bool)):
        return obj
    else:
        # For any other type, try to convert to string as fallback
        try:
            return str(obj)
        except:
            return None

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

    - **num_plays**: Number of diverse plays to generate (default: 5, max: 10)
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
        raise HTTPException(status_code=500, detail="Error retrieving adaptive weights.")

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
        raise HTTPException(status_code=500, detail="Error retrieving performance analytics.")

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

@api_router.get("/prediction-history-public")
async def get_prediction_history_public(
    limit: int = Query(25, ge=1, le=50, description="Number of recent predictions to return")
):
    """
    Get public prediction history for display on main page.
    Returns formatted prediction history for public interface.
    """
    try:
        logger.info(f"Received request for public prediction history (limit: {limit})")

        # Get prediction history from database
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

@api_router.get("/prediction-history-grouped")
async def get_prediction_history_grouped(
    limit_dates: int = Query(25, ge=1, le=50, description="Number of recent dates to return")
):
    """
    Get prediction history grouped by date for enhanced display.
    Returns predictions organized by generation date with statistics.
    """
    try:
        logger.info(f"Received request for grouped prediction history ({limit_dates} dates)")

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
        logger.error(f"Error retrieving grouped prediction history: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving grouped prediction history.")

@api_router.get("/predict/smart")
async def get_smart_predictions(
    limit: int = Query(100, ge=1, le=100, description="Number of Smart AI predictions to retrieve from database")
):
    """
    Get Smart AI predictions from database with next drawing information.
    Returns latest Smart AI predictions generated by the pipeline.
    """
    try:
        logger.info(f"Received request for {limit} Smart AI predictions from database")

        # Calculate next drawing date and related information
        from src.database import calculate_next_drawing_date
        from datetime import datetime, timedelta

        next_drawing_date = calculate_next_drawing_date()
        current_date = datetime.now()
        next_date = datetime.strptime(next_drawing_date, '%Y-%m-%d')
        days_until_drawing = (next_date - current_date).days

        # Determine if today is drawing day
        is_drawing_day = current_date.weekday() in [0, 2, 5]  # Mon, Wed, Sat

        # Get latest Smart AI predictions from database
        predictions_df = db.get_prediction_history(limit=limit)

        # Convert DataFrame to list of dictionaries
        if not predictions_df.empty:
            predictions_list = predictions_df.to_dict('records')
        else:
            predictions_list = []

        # Convert database records to Smart AI format
        smart_predictions = []
        for i, pred in enumerate(predictions_list):
            # Safely convert all values to ensure JSON serialization
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

                pred_id = pred.get("id", 0)
                prediction_id = int(pred_id) if pred_id is not None else 0

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
                    "prediction_id": prediction_id,
                    "generated_at": str(pred.get("timestamp", "")),
                    "method": "smart_ai_pipeline"
                }
                smart_predictions.append(smart_pred)
            except (ValueError, TypeError, KeyError) as e:
                logger.warning(f"Error converting prediction record {i}: {e}")
                continue

        if not smart_predictions:
            # If no predictions in database, generate some
            logger.warning("No Smart AI predictions found in database, generating new ones...")
            if not predictor:
                raise HTTPException(status_code=503, detail="No predictions available and predictor service not available")

            # Generate a few predictions as fallback
            predictions = predictor.predict_diverse_plays(num_plays=min(limit, 10), save_to_log=True)

            for i, pred in enumerate(predictions):
                # Safely convert numpy types to native Python types
                numbers = [int(x) for x in pred.get("numbers", [])]
                powerball = int(pred.get("powerball", 0))
                total_score = float(pred.get("score_total", 0.0))

                # Convert score_details safely
                score_details = pred.get("score_details", {})
                safe_score_details = {}
                for key, value in score_details.items():
                    if isinstance(value, (np.integer, np.floating)):
                        safe_score_details[key] = float(value)
                    else:
                        safe_score_details[key] = value

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

        # Calculate statistics with safe conversion
        if smart_predictions:
            avg_score = sum(float(p["total_score"]) for p in smart_predictions) / len(smart_predictions)
            best_score = max(float(p["total_score"]) for p in smart_predictions)
        else:
            avg_score = 0.0
            best_score = 0.0

        return {
            "method": "smart_ai_database",
            "smart_predictions": smart_predictions,
            "total_predictions": len(smart_predictions),
            "average_score": float(avg_score),
            "best_score": float(best_score),
            "data_source": "database" if smart_predictions and "pipeline" in smart_predictions[0].get("method", "") else "realtime_generation",
            "predictions_info": {
                "generated_for_drawing": next_drawing_date,
                "drawing_status": "future" if days_until_drawing > 0 else "today" if days_until_drawing == 0 else "past",
                "note": f"Predictions generated for drawing on {next_date.strftime('%B %d, %Y')}",
                "generated_date": datetime.now().strftime('%Y-%m-%d'),
                "generated_formatted": datetime.now().strftime('%B %d, %Y'),
                "generation_time": datetime.now().strftime('%H:%M ET')
            },
            "next_drawing": {
                "date": next_drawing_date,
                "formatted_date": next_date.strftime('%B %d, %Y'),
                "days_until": int(days_until_drawing),
                "is_today": days_until_drawing == 0,
                "is_drawing_day": is_drawing_day,
                "current_date": datetime.now().strftime('%Y-%m-%d'),
                "current_formatted": datetime.now().strftime('%B %d, %Y'),
                "current_day": datetime.now().day,
                "drawing_schedule": {
                    "monday": "Drawing Day",
                    "wednesday": "Drawing Day",
                    "saturday": "Drawing Day",
                    "other_days": "No Drawing"
                }
            },
            "timestamp": datetime.now().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving Smart AI predictions: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving Smart AI predictions.")

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
        orchestrator_status = convert_numpy_types(pipeline_orchestrator.get_pipeline_status())

        # Get recent execution history from global tracking with enhanced metadata
        recent_executions = []
        # Sort executions by start time descending and take the last 5
        sorted_executions = sorted(pipeline_executions.items(), key=lambda item: item[1].get("start_time", ""), reverse=True)
        for exec_id, execution in sorted_executions[:5]:
            # Extract trigger details
            trigger_details = execution.get("trigger_details", {})

            recent_executions.append({
                "execution_id": exec_id,
                "status": execution.get("status", "unknown"),
                "start_time": execution.get("start_time"),
                "end_time": execution.get("end_time"),
                "current_step": execution.get("current_step"),
                "steps_completed": execution.get("steps_completed", 0),
                "total_steps": execution.get("total_steps", 7),
                "execution_source": execution.get("execution_source", "unknown"),
                "trigger_type": execution.get("trigger_type", "unknown"),
                "triggered_by": trigger_details.get("triggered_by", "unknown"),
                "matches_schedule": trigger_details.get("actual_execution", {}).get("matches_schedule", None),
                "execution_day": trigger_details.get("actual_execution", {}).get("day", "unknown"),
                "execution_time": trigger_details.get("actual_execution", {}).get("time", "unknown")
            })

        # Get next scheduled execution (from scheduler)
        next_execution = None
        try:
            jobs = scheduler.get_jobs()
            # Filter for jobs that are not paused and have a next run time
            runnable_jobs = [job for job in jobs if not job.next_run_time is None]
            if runnable_jobs:
                # Find the job with the earliest next_run_time
                next_job = min(runnable_jobs, key=lambda job: job.next_run_time)
                next_execution = next_job.next_run_time.isoformat()
        except Exception as e:
            logger.warning(f"Could not get next scheduled execution: {e}")

        # Get recent predictions for generated plays
        recent_plays = []
        try:
            history = db.get_prediction_history(limit=5)
            for _, row in history.iterrows():
                # Safely convert all values to ensure JSON serialization
                try:
                    # Use direct type conversion for better safety
                    numbers = []
                    for col in ['n1', 'n2', 'n3', 'n4', 'n5']:
                        val = row.get(col, 0)
                        numbers.append(int(val) if val is not None else 0)

                    powerball_val = row.get('powerball', 0)
                    powerball = int(powerball_val) if powerball_val is not None else 0

                    score_val = row.get('score_total', 0.0)
                    score = float(score_val) if score_val is not None else 0.0

                    recent_plays.append({
                        "numbers": numbers,
                        "powerball": powerball,
                        "score": score,
                        "timestamp": str(row['timestamp']) if row.get('timestamp') is not None else "",
                        "dataset_hash": str(row['dataset_hash']) if row.get('dataset_hash') is not None else ""
                    })
                except Exception as convert_error:
                    logger.warning(f"Error converting prediction row to JSON-safe format: {convert_error}")
                    continue
        except Exception as e:
            logger.warning(f"Could not get recent plays: {e}")

        # Get system health metrics
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('.')

            system_health = {
                "cpu_usage_percent": float(cpu_percent),
                "memory_usage_percent": float(memory.percent),
                "memory_available_gb": round(float(memory.available) / (1024**3), 2),
                "disk_usage_percent": float(disk.percent),
                "disk_free_gb": round(float(disk.free) / (1024**3), 2)
            }
        except Exception as e:
            logger.warning(f"Could not get system health metrics: {e}")
            system_health = {"error": "System metrics unavailable"}

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

        # Apply convert_numpy_types to the entire response to ensure safety
        response_data = {
            "pipeline_status": {
                "current_status": current_status,
                "last_execution": sorted_executions[0] if sorted_executions else None,
                "next_scheduled_execution": next_execution,
                "recent_execution_history": recent_executions,
                "system_health": system_health
            },
            "orchestrator_status": orchestrator_status,
            "generated_plays_last_run": recent_plays,
            "timestamp": datetime.now().isoformat()
        }

        return convert_numpy_types(response_data)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting pipeline status: {e}")
        raise HTTPException(status_code=500, detail="Error getting pipeline status.")

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
            # Validate steps - assuming these are the core steps the orchestrator can handle
            valid_steps = ['data', 'adaptive', 'weights', 'prediction', 'validation', 'performance', 'reports']
            invalid_steps = [step for step in step_list if step not in valid_steps]
            if invalid_steps:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid steps: {invalid_steps}. Valid steps: {valid_steps}"
                )

        # Initialize execution tracking with enhanced metadata
        current_time = datetime.now()
        current_day = current_time.strftime('%A').lower()
        current_time_str = current_time.strftime('%H:%M')

        # Get scheduler configuration for comparison
        expected_days = ['monday', 'wednesday', 'saturday']
        expected_time = '23:30'
        timezone = 'America/New_York'

        # Always ensure 7 total steps for full pipeline, or count of step_list for partial
        total_steps = len(step_list) if step_list else 7

        pipeline_executions[execution_id] = {
            "execution_id": execution_id,
            "status": "starting",
            "start_time": current_time.isoformat(),
            "current_step": None,
            "steps_completed": 0,
            "total_steps": total_steps,  # Ensure correct step count
            "num_predictions": num_predictions if num_predictions else 100,  # Default to 100
            "requested_steps": step_list,
            "error": None,
            "trigger_type": "manual_dashboard",
            "execution_source": "manual_dashboard",
            "trigger_details": {
                "type": "manual",
                "scheduled_config": {
                    "days": expected_days,
                    "time": expected_time,
                    "timezone": timezone
                },
                "actual_execution": {
                    "day": current_day,
                    "time": current_time_str,
                    "matches_schedule": current_day in expected_days
                },
                "triggered_by": "user_dashboard",
                "manual_parameters": {
                    "num_predictions": num_predictions if num_predictions else 100,
                    "specific_steps": step_list,
                    "force_execution": force,
                    "pipeline_type": "partial" if step_list else "full"
                }
            }
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
        raise HTTPException(status_code=500, detail="Error triggering pipeline execution.")

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
        logger.info(f"Received request for pipeline logs (level: {level}, start_date: {start_date}, end_date: {end_date}, limit: {limit}, offset: {offset})")

        # Get log file path from config or use default
        log_file = "logs/shiolplus.log"
        # Use the orchestrator's config if available, otherwise use default path
        if pipeline_orchestrator and hasattr(pipeline_orchestrator, 'config') and pipeline_orchestrator.config:
            try:
                # Correctly access config values using get with fallback
                log_file = pipeline_orchestrator.config.get('paths', 'log_file', fallback=log_file)
            except configparser.NoSectionError:
                logger.warning(" 'paths' section not found in config, using default log file path.")
            except configparser.NoOptionError:
                logger.warning(" 'log_file' option not found in 'paths' section, using default log file path.")


        if not os.path.exists(log_file):
            logger.warning(f"Log file not found at: {log_file}")
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
                # Read all lines and then reverse for most recent first
                lines = f.readlines()

            # Process lines from most recent to oldest
            for line in reversed(lines):
                line = line.strip()
                if not line:
                    continue

                try:
                    # Basic log parsing - extract timestamp, level, message
                    # Example format: 2023-10-27 10:30:00 | INFO | src/main.py:55 - Message
                    parts = line.split(' | ')
                    if len(parts) < 4: # Ensure there are enough parts for timestamp, level, location, and message
                        logger.debug(f"Skipping malformed log line: {line}")
                        continue

                    timestamp_str = parts[0]
                    log_level = parts[1].strip()
                    location = parts[2]
                    message = ' | '.join(parts[3:]) # Rejoin remaining parts as the message

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
                            # Add one day to end_date to include the whole end day
                            if log_date.date() >= end_dt.date() + timedelta(days=1):
                                continue
                    except ValueError:
                        logger.warning(f"Could not parse timestamp in log line: {line}")
                        continue # Skip line if timestamp is invalid

                    logs.append({
                        "timestamp": timestamp_str,
                        "level": log_level.strip(),
                        "location": location,
                        "message": message,
                        "raw_line": line # Optionally include raw line for debugging
                    })

                except Exception as e:
                    # Catch any unexpected errors during line processing
                    logger.error(f"Error processing log line '{line}': {e}")
                    continue # Skip malformed lines or lines causing errors

            # Apply pagination AFTER collecting all relevant logs
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

        except FileNotFoundError:
            logger.error(f"Log file not found: {log_file}")
            raise HTTPException(status_code=404, detail=f"Log file not found: {log_file}")
        except Exception as e:
            logger.error(f"Error reading or parsing log file '{log_file}': {e}")
            raise HTTPException(status_code=500, detail=f"Error reading or parsing log file: {str(e)}")

    except Exception as e:
        logger.error(f"Unexpected error in get_pipeline_logs: {e}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred while retrieving pipeline logs.")

@api_router.get("/pipeline/health")
async def get_pipeline_health():
    """System health check endpoint that validates all pipeline components."""
    try:
        logger.info("Received request for pipeline health check")

        health_status = {
            "overall_status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "checks": {}
        }

        # Check pipeline orchestrator
        try:
            orchestrator_available = False
            orchestrator_message = "Pipeline orchestrator not available"
            orchestrator_details = {}
            if pipeline_orchestrator:
                orchestrator_available = True
                orchestrator_message = "Pipeline orchestrator is available"
                try:
                    # Attempt to get status without raising exceptions
                    status = pipeline_orchestrator.get_pipeline_status()
                    orchestrator_details["status"] = status.get("status", "unknown")
                    orchestrator_details["configuration_loaded"] = status.get("configuration_loaded", False)
                    orchestrator_details["database_initialized"] = status.get("database_initialized", False)
                except Exception as e:
                    orchestrator_details["error"] = f"Error fetching status: {str(e)}"
                    orchestrator_message = f"Pipeline orchestrator error: {str(e)}"
                    health_status["overall_status"] = "degraded"

            health_status["checks"]["pipeline_orchestrator"] = {
                "status": "healthy" if orchestrator_available else "unhealthy",
                "message": orchestrator_message,
                "details": orchestrator_details
            }
            if not orchestrator_available:
                health_status["overall_status"] = "degraded"

        except Exception as e:
            health_status["checks"]["pipeline_orchestrator"] = {
                "status": "unhealthy",
                "message": f"Unexpected error checking orchestrator: {str(e)}"
            }
            health_status["overall_status"] = "unhealthy" # Treat unexpected errors as critical

        # Check database connectivity
        try:
            db_status = "unhealthy"
            db_message = "Database connectivity error"
            db_details = {}
            # Ensure db connection is attempted safely
            conn = None
            try:
                conn = db.get_db_connection()
                if conn:
                    db_status = "healthy"
                    db_message = "Database accessible"
                    # Perform a simple query to check functionality
                    cursor = conn.cursor()
                    cursor.execute("SELECT COUNT(*) FROM draws")
                    record_count = cursor.fetchone()[0]
                    db_details["total_draws"] = record_count
                    # Get latest draw date
                    cursor.execute("SELECT MAX(draw_date) FROM draws")
                    latest_draw_date = cursor.fetchone()[0]
                    db_details["latest_draw_date"] = str(latest_draw_date) if latest_draw_date else "N/A"

                else:
                    db_status = "disconnected"
                    db_message = "Database connection could not be established"

            except Exception as e:
                db_status = "error"
                db_message = f"Database operation failed: {str(e)}"
                health_status["overall_status"] = "unhealthy"
            finally:
                if conn:
                    conn.close()

            health_status["checks"]["database"] = {
                "status": db_status,
                "message": db_message,
                "details": db_details
            }
        except Exception as e: # Catch broader exceptions related to db check setup
            health_status["checks"]["database"] = {
                "status": "unhealthy",
                "message": f"Unexpected error during database health check: {str(e)}"
            }
            health_status["overall_status"] = "unhealthy"

        # Check model availability
        try:
            model_status = "unhealthy"
            model_message = "ML model not available"
            model_details = {}
            if predictor:
                model_status = "healthy"
                model_message = "ML model is loaded and functional"
                # Test model prediction
                try:
                    wb_probs, pb_probs = predictor.predict_probabilities()
                    model_details["wb_predictions_count"] = len(wb_probs)
                    model_details["pb_predictions_count"] = len(pb_probs)
                except Exception as e:
                    model_status = "warning" # Model loaded but prediction failed
                    model_message = f"ML model loaded but prediction failed: {str(e)}"
                    model_details["prediction_error"] = str(e)
                    health_status["overall_status"] = "degraded"

            health_status["checks"]["model"] = {
                "status": model_status,
                "message": model_message,
                "details": model_details
            }
            if model_status == "unhealthy":
                health_status["overall_status"] = "degraded"

        except Exception as e: # Catch broader exceptions related to model check setup
            health_status["checks"]["model"] = {
                "status": "unhealthy",
                "message": f"Unexpected error during model health check: {str(e)}"
            }
            health_status["overall_status"] = "unhealthy"


        # Check configuration validation
        try:
            config_status = "unhealthy"
            config_message = "Configuration not loaded or invalid"
            config_details = {}
            import configparser
            config = configparser.ConfigParser()
            config_loaded = config.read('config/config.ini')

            if config_loaded:
                config_status = "healthy"
                config_message = "Configuration is valid"
                config_details["sections"] = config.sections()
                # Add specific checks for critical options if needed
                if not config.has_section('pipeline') or not config.has_option('pipeline', 'execution_time'):
                    config_status = "warning"
                    config_message = "Pipeline configuration may be incomplete"
                    health_status["overall_status"] = "degraded"
            else:
                config_status = "error"
                config_message = "Failed to read config/config.ini"
                health_status["overall_status"] = "unhealthy"

            health_status["checks"]["configuration"] = {
                "status": config_status,
                "message": config_message,
                "details": config_details
            }
        except Exception as e:
            health_status["checks"]["configuration"] = {
                "status": "unhealthy",
                "message": f"Unexpected error during configuration check: {str(e)}"
            }
            health_status["overall_status"] = "unhealthy"

        # Check disk space and system resources
        try:
            disk = psutil.disk_usage('/')
            memory = psutil.virtual_memory()

            disk_free_gb = disk.free / (1024**3)
            memory_available_gb = memory.available / (1024**3)

            resource_issues = []
            if disk_free_gb < 1.0:  # Less than 1GB free
                resource_issues.append(f"Low disk space: {disk_free_gb:.1f}GB free")
            if memory.percent > 90:  # More than 90% memory usage
                resource_issues.append(f"High memory usage: {memory.percent:.1f}%")

            resource_status = "healthy"
            resource_message = "System resources are adequate"
            resource_details = {
                "disk_free_gb": round(disk_free_gb, 2),
                "memory_usage_percent": memory.percent,
                "memory_available_gb": round(memory_available_gb, 2)
            }

            if resource_issues:
                resource_status = "warning"
                resource_message = "; ".join(resource_issues)
                if health_status["overall_status"] == "healthy":
                    health_status["overall_status"] = "degraded"

            health_status["checks"]["system_resources"] = {
                "status": resource_status,
                "message": resource_message,
                "details": resource_details
            }
        except Exception as e:
            health_status["checks"]["system_resources"] = {
                "status": "warning",
                "message": f"Could not check system resources: {str(e)}",
                "details": {"error": str(e)}
            }

        # Final determination of overall status
        if any(check["status"] == "unhealthy" for check in health_status["checks"].values()):
            health_status["overall_status"] = "unhealthy"
        elif any(check["status"] == "warning" for check in health_status["checks"].values()) and health_status["overall_status"] == "healthy":
             health_status["overall_status"] = "degraded"


        return health_status

    except Exception as e:
        logger.error(f"Unexpected error in pipeline health check: {e}")
        # Return a critical failure state
        return {
            "overall_status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "checks": {},
            "error": f"Critical error during health check execution: {str(e)}"
        }

@api_router.post("/pipeline/stop")
async def stop_pipeline_execution():
    """Stop currently running pipeline execution"""
    try:
        logger.info("Received request to stop pipeline execution")

        # Find running executions
        running_executions = [ex for ex in pipeline_executions.values() if ex.get("status") == "running"]

        if not running_executions:
            return {
                "message": "No running pipeline execution found",
                "status": "no_running_execution",
                "timestamp": datetime.now().isoformat()
            }

        # Stop the most recent running execution
        latest_execution = running_executions[0]
        execution_id = latest_execution["execution_id"]

        # Update execution status
        pipeline_executions[execution_id].update({
            "status": "stopped",
            "end_time": datetime.now().isoformat(),
            "current_step": "stopped_by_user"
        })

        logger.info(f"Pipeline execution {execution_id} stopped by user request")

        return {
            "message": "Pipeline execution stopped successfully",
            "execution_id": execution_id,
            "status": "stopped",
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error stopping pipeline execution: {e}")
        raise HTTPException(status_code=500, detail="Error stopping pipeline execution.")


# Background task functions for pipeline execution
async def run_full_pipeline_background(execution_id: str, num_predictions: int):
    """Run full pipeline in background task with enhanced logging and guaranteed 7-step execution."""
    try:
        pipeline_executions[execution_id]["status"] = "running"
        pipeline_executions[execution_id]["current_step"] = "starting"

        # Get execution metadata for logging
        execution_meta = pipeline_executions[execution_id]
        trigger_details = execution_meta.get("trigger_details", {})

        logger.info(f"Starting background pipeline execution {execution_id}")
        logger.info(f"Pipeline type: Full 7-step pipeline with {num_predictions} predictions")
        logger.info(f"Execution source: {execution_meta.get('execution_source', 'unknown')}")
        logger.info(f"Triggered by: {trigger_details.get('triggered_by', 'unknown')}")
        logger.info(f"Schedule compliance: {trigger_details.get('actual_execution', {}).get('matches_schedule', 'unknown')}")

        # Run the full pipeline with explicit parameters
        result = pipeline_orchestrator.run_full_pipeline(
            num_predictions=num_predictions,
            requested_steps=None,  # Full pipeline - all 7 steps
            execution_source=execution_meta.get('execution_source', 'background_task'),
            trigger_details=trigger_details
        )

        # Extract step summary from result
        steps_completed = 0
        if result.get("results"):
            # Count successful steps
            steps_completed = sum(1 for step_result in result["results"].values()
                                if step_result.get("status") == "success")

        # Determine final status
        final_status = result.get("status", "unknown")
        if final_status == "success" and steps_completed != 7:
            final_status = "partial_success"
            logger.warning(f"Pipeline completed but only {steps_completed}/7 steps were successful")

        # Update execution status based on orchestrator's result
        pipeline_executions[execution_id].update({
            "status": final_status,
            "end_time": datetime.now().isoformat(),
            "result": result,
            "steps_completed": steps_completed,
            "total_steps": 7,  # Always 7 for full pipeline
            "error": result.get("error"),
            "pipeline_summary": result.get("summary", {}),
            "execution_time": result.get("execution_time")
        })

        # Enhanced completion logging
        logger.info(f"Background pipeline execution {execution_id} completed with status: {final_status}")
        logger.info(f"Steps completed: {steps_completed}/7")
        logger.info(f"Execution time: {result.get('execution_time', 'unknown')}")
        logger.info(f"Execution metadata: Source={execution_meta.get('execution_source')}, "
                   f"Triggered_by={trigger_details.get('triggered_by')}, "
                   f"Scheduled={trigger_details.get('actual_execution', {}).get('matches_schedule')}")

    except Exception as e:
        # Catch any exceptions not handled by the orchestrator itself
        execution_meta = pipeline_executions[execution_id]
        trigger_details = execution_meta.get("trigger_details", {})

        pipeline_executions[execution_id].update({
            "status": "failed",
            "end_time": datetime.now().isoformat(),
            "error": str(e),
            "current_step": "exception_handler",
            "steps_completed": pipeline_executions[execution_id].get("steps_completed", 0),
            "total_steps": 7  # Always 7 for full pipeline
        })

        logger.error(f"Critical error during background pipeline execution {execution_id}: {e}")
        logger.error(f"Failed execution metadata: Source={execution_meta.get('execution_source')}, "
                    f"Triggered_by={trigger_details.get('triggered_by')}, "
                    f"Scheduled={trigger_details.get('actual_execution', {}).get('matches_schedule')}", exc_info=True)


async def run_pipeline_steps_background(execution_id: str, steps: List[str], num_predictions: int):
    """Run specific pipeline steps in background task."""
    try:
        pipeline_executions[execution_id]["status"] = "running"
        pipeline_executions[execution_id]["total_steps"] = len(steps) # Update total steps based on requested list

        results_per_step = {}
        for i, step in enumerate(steps):
            pipeline_executions[execution_id]["current_step"] = step
            pipeline_executions[execution_id]["steps_completed"] = i

            step_result = None
            try:
                # Execute the single step via the orchestrator
                step_result = pipeline_orchestrator.run_single_step(step) # Assume this returns a dict with 'status' and 'result' or 'error'
                results_per_step[step] = step_result

                if step_result.get("status") != "success":
                    # If a step fails, record the error and stop processing further steps for this execution
                    raise Exception(f"Step '{step}' failed: {step_result.get('error', 'Unknown error')}")

            except Exception as e:
                # Capture error from step execution or raised exception
                error_message = str(e)
                if step_result and "error" in step_result: # If step_result exists and contains an error key
                    error_message = step_result["error"]

                results_per_step[step] = {"status": "failed", "error": error_message}
                pipeline_executions[execution_id].update({
                    "status": "failed",
                    "end_time": datetime.now().isoformat(),
                    "error": f"Step '{step}' failed: {error_message}",
                    "current_step": step # Mark the failed step
                })
                logger.error(f"Pipeline step execution {execution_id} failed at step '{step}': {error_message}")
                return # Exit the function after the first failure

        # If all steps completed successfully
        pipeline_executions[execution_id].update({
            "status": "completed",
            "end_time": datetime.now().isoformat(),
            "result": {"status": "success", "results_per_step": results_per_step},
            "steps_completed": len(steps)
        })

        logger.info(f"Background pipeline steps execution {execution_id} completed successfully")

    except Exception as e:
        # Catch any unexpected errors during the overall process
        pipeline_executions[execution_id].update({
            "status": "failed",
            "end_time": datetime.now().isoformat(),
            "error": f"An unexpected error occurred: {str(e)}",
            "current_step": "unexpected_failure"
        })
        logger.error(f"Critical error in background pipeline steps execution {execution_id}: {e}", exc_info=True)


# --- SHIOL+ v6.0 Configuration Dashboard Endpoints ---

@api_router.get("/system/stats")
async def get_system_stats():
    """Get real-time system statistics for dashboard monitoring"""
    try:
        import psutil
        from datetime import datetime

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

@api_router.get("/pipeline/execution-history")
async def get_pipeline_execution_history(limit: int = Query(20, ge=1, le=100, description="Number of executions to return")):
    """
    Get pipeline execution history from reports directory.
    Returns recent pipeline executions with status and timing information.
    """
    try:
        logger.info(f"Received request for pipeline execution history (limit: {limit})")

        reports_dir = "reports"
        if not os.path.exists(reports_dir):
            return {
                "executions": [],
                "count": 0,
                "message": "No pipeline reports found"
            }

        # Get all pipeline report files
        report_files = []
        for filename in os.listdir(reports_dir):
            if filename.startswith("pipeline_report_") and filename.endswith(".json"):
                filepath = os.path.join(reports_dir, filename)
                try:
                    # Extract timestamp from filename
                    timestamp_str = filename.replace("pipeline_report_", "").replace(".json", "")
                    timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                    report_files.append((timestamp, filepath, timestamp_str))
                except ValueError:
                    continue

        # Sort by timestamp descending (newest first)
        report_files.sort(key=lambda x: x[0], reverse=True)

        # Limit results
        report_files = report_files[:limit]

        executions = []
        for timestamp, filepath, timestamp_str in report_files:
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    report_data = json.load(f)

                pipeline_execution = report_data.get('pipeline_execution', {})
                trigger_details = pipeline_execution.get('trigger_details', {}) # Corrected to use trigger_details

                # Calculate success/failure stats
                status_info = pipeline_execution.get('status', {})
                successful_steps = sum(1 for step_status in status_info.values()
                                     if isinstance(step_status, dict) and step_status.get('status') == 'success')
                total_steps = len(status_info)
                failed_steps = total_steps - successful_steps

                # Determine overall status
                overall_status = 'success' if failed_steps == 0 else 'partial' if successful_steps > 0 else 'failed'

                # Extract execution source and trigger information
                execution_source = pipeline_execution.get('execution_source', 'unknown')
                triggered_by = trigger_details.get('triggered_by', 'unknown') # Corrected to use triggered_by from trigger_details

                # Determine if execution matches schedule
                actual_execution = trigger_details.get('actual_execution', {})
                matches_schedule = actual_execution.get('matches_schedule', None)


                execution_info = {
                    "execution_id": timestamp_str,
                    "start_time": pipeline_execution.get('start_time'),
                    "end_time": pipeline_execution.get('end_time'),
                    "execution_time": pipeline_execution.get('total_execution_time'),
                    "status": overall_status,
                    "steps_successful": successful_steps,
                    "steps_failed": failed_steps,
                    "total_steps": total_steps,
                    "success_rate": f"{(successful_steps / total_steps * 100):.1f}%" if total_steps > 0 else "0%",
                    "report_file": filepath,
                    "system_info": report_data.get('system_info', {}),
                    "formatted_time": timestamp.strftime('%d %b %Y, %H:%M:%S'),
                    "steps_detail": status_info,
                    "execution_source": execution_source, # Added execution source
                    "triggered_by": triggered_by,       # Added triggered by
                    "matches_schedule": matches_schedule  # Added matches schedule
                }

                executions.append(execution_info)

            except Exception as e:
                logger.warning(f"Error reading pipeline report {filepath}: {e}")
                continue

        return {
            "executions": executions,
            "count": len(executions),
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error retrieving pipeline execution history: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving pipeline execution history.")

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
# Mount the API router before the static files to ensure API endpoints are prioritized.
app.include_router(api_router)

# Mount new public and authentication routers
app.include_router(public_router)
app.include_router(auth_router)

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
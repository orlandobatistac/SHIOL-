from fastapi import FastAPI, HTTPException, APIRouter, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from loguru import logger
import os

from src.predictor import Predictor
from src.intelligent_generator import IntelligentGenerator
from src.loader import update_database_from_source
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from contextlib import asynccontextmanager

# --- Scheduler and App Lifecycle ---
scheduler = AsyncIOScheduler()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # On startup
    logger.info("Application startup...")
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
    intelligent_generator = IntelligentGenerator()
    logger.info("Model and generator loaded successfully.")
except Exception as e:
    logger.critical(f"Fatal error during startup: Failed to load model. Error: {e}")
    predictor = None
    intelligent_generator = None

# --- API Router ---
api_router = APIRouter(prefix="/api/v1")

@api_router.get("/predict")
async def get_prediction():
    """
    Generates and returns a single Powerball prediction.
    """
    if not predictor or not intelligent_generator:
        logger.error("Endpoint /predict called, but model is not available.")
        raise HTTPException(
            status_code=500, detail="Model is not available. Please check server logs."
        )
    try:
        logger.info("Received request for prediction.")
        wb_probs, pb_probs = predictor.predict_probabilities()
        play_df = intelligent_generator.generate_plays(
            wb_probs, pb_probs, num_plays=1
        )
        prediction = play_df.iloc[0].astype(int).tolist()
        logger.info(f"Generated prediction: {prediction}")
        return {"prediction": prediction}
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
        return {"predictions": predictions}
    except Exception as e:
        logger.error(f"An error occurred during multiple prediction generation: {e}")
        raise HTTPException(status_code=500, detail="Model prediction failed.")

# --- Application Mounting ---
# Mount the API router before the static files to ensure API endpoints are prioritized.
app.include_router(api_router)

# Build an absolute path to the 'frontend' directory for robust file serving.
# This avoids issues with the current working directory.
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(APP_ROOT, "..", "frontend")

app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="static")
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime
from loguru import logger
from src.generative_predictor import GenerativePredictor
from src.rnn_predictor import RNNPredictor
from src.intelligent_generator import IntelligentGenerator
from src.adaptive_feedback import AdaptivePlayScorer
from src.loader import get_data_loader
from src.database import save_prediction_log

def train_and_predict() -> None:
    """
    Orchestrates training and prediction generation, including new and improved methods.
    """
    logger.info("Starting automatic training and prediction process...")

    # Load historical data
    data_loader = get_data_loader()
    historical_data = data_loader.load_historical_data()

    # 1. Generative Prediction (VAE)
    logger.info("Training and generating predictions with the generative model (VAE)...")
    generative_predictor = GenerativePredictor(historical_data)
    generative_predictor.train()
    generative_predictions = generative_predictor.predict(n_samples=10)

    # 2. Prediction with RNN
    logger.info("Training and generating predictions with the RNN model...")
    rnn_predictor = RNNPredictor(historical_data)
    rnn_predictor.train()
    rnn_predictions = rnn_predictor.predict(n_samples=10)

    # 3. Improve Traditional Prediction
    logger.info("Generating improved predictions with IntelligentGenerator...")
    intelligent_generator = IntelligentGenerator(historical_data)
    traditional_predictions = intelligent_generator.generate_plays(n_samples=10)

    # 4. Improved Adaptive Prediction
    logger.info("Generating improved adaptive predictions...")
    adaptive_scorer = AdaptivePlayScorer(historical_data)
    adaptive_predictions = adaptive_scorer.generate_adaptive_predictions(n_samples=10)

    # Save all predictions to the database
    all_predictions = {
        "generative": generative_predictions,
        "rnn": rnn_predictions,
        "traditional": traditional_predictions,
        "adaptive": adaptive_predictions,
    }

    for method, predictions in all_predictions.items():
        for prediction in predictions:
            save_prediction_log({
                "prediction": prediction,
                "timestamp": datetime.now().isoformat(),
                "method": method
            })

    logger.info("Training and prediction process completed.")


# Configure the scheduler
scheduler = BackgroundScheduler()
scheduler.add_job(train_and_predict, 'interval', hours=24)  # Execute every 24 hours
scheduler.start()

logger.info("Orchestrator started. Automatic tasks are scheduled.")

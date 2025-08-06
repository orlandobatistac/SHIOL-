from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime
from loguru import logger
from src.generative_predictor import GenerativePredictor
from src.rnn_predictor import RNNPredictor
from src.intelligent_generator import IntelligentGenerator
from src.adaptive_feedback import AdaptivePlayScorer
from src.loader import get_data_loader
from src.database import save_prediction_log
from src.ensemble_predictor import EnsemblePredictor, EnsembleMethod

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

    # 5. Multi-Model Ensemble Prediction
    logger.info("Generating ensemble predictions from multiple models...")
    ensemble_predictor = EnsemblePredictor(historical_data)
    
    # Try different ensemble methods
    ensemble_methods = [
        EnsembleMethod.PERFORMANCE_WEIGHTED,
        EnsembleMethod.WEIGHTED_AVERAGE,
        EnsembleMethod.MAJORITY_VOTING
    ]
    
    ensemble_predictions = []
    for method in ensemble_methods:
        try:
            ensemble_result = ensemble_predictor.predict_ensemble(method)
            if ensemble_result:
                # Convert to play format
                wb_probs = ensemble_result['white_ball_probabilities']
                pb_probs = ensemble_result['powerball_probabilities']
                
                # Generate plays from probabilities
                from src.intelligent_generator import IntelligentGenerator
                generator = IntelligentGenerator(historical_data)
                method_predictions = generator.generate_plays_from_probabilities(
                    wb_probs, pb_probs, n_samples=5
                )
                
                # Add method metadata
                for pred in method_predictions:
                    pred['ensemble_method'] = method.value
                    pred['models_used'] = ensemble_result.get('models_used', [])
                
                ensemble_predictions.extend(method_predictions)
                
        except Exception as e:
            logger.error(f"Error with ensemble method {method.value}: {e}")
            continue

    # Save all predictions to the database
    all_predictions = {
        "generative": generative_predictions,
        "rnn": rnn_predictions,
        "traditional": traditional_predictions,
        "adaptive": adaptive_predictions,
        "ensemble": ensemble_predictions,
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

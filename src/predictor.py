import configparser
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from loguru import logger
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from xgboost import XGBClassifier
from typing import Dict, List, Tuple, Any

from src.loader import get_data_loader
from src.intelligent_generator import FeatureEngineer, DeterministicGenerator
from src.database import save_prediction_log

def get_model_trainer():
    """Function to get model trainer for external validation"""
    return ModelTrainer("models/shiolplus.pkl")
# Assuming EnsemblePredictor and EnsembleMethod are defined elsewhere, e.g., in src.ensemble_predictor
# Placeholder import for EnsemblePredictor and EnsembleMethod
try:
    from src.ensemble_predictor import EnsemblePredictor, EnsembleMethod
except ImportError:
    logger.warning("Could not import EnsemblePredictor or EnsembleMethod. Ensemble functionality will be disabled.")
    # Define dummy classes if not found to prevent errors
    class EnsemblePredictor:
        def __init__(self, historical_data):
            logger.warning("Dummy EnsemblePredictor initialized.")
            self.historical_data = historical_data
            self.ensemble_method = "average" # Default method
            self.models = [] # Placeholder for other models

        def predict_ensemble(self) -> Dict[str, Any]:
            logger.warning("Dummy predict_ensemble called.")
            # Simulate a prediction
            return {
                'white_ball_probabilities': np.ones(69) / 69,
                'powerball_probabilities': np.ones(26) / 26,
                'total_models': 0,
                'ensemble_method': self.ensemble_method
            }

        def get_ensemble_summary(self) -> Dict[str, Any]:
            logger.warning("Dummy get_ensemble_summary called.")
            return {
                'ensemble_enabled': False,
                'reason': 'Dummy system'
            }

        def update_model_weights(self, performance_feedback: Dict[str, float]) -> None:
            logger.warning("Dummy update_model_weights called.")
            pass

    class EnsembleMethod:
        def __init__(self, method_name: str):
            self.method_name = method_name
        def __str__(self):
            return self.method_name

class ModelTrainer:
    def __init__(self, model_path):
        self.model_path = model_path
        self.config = configparser.ConfigParser()
        self.config.read("config/config.ini")
        self.model = None
        self.target_columns = None
        self.white_ball_columns = [f"n{i}" for i in range(1, 6)]
        self.pb_column = "pb"
        logger.info("ModelTrainer v2.0 initialized.")

    def create_target_variable(self, features_df):
        logger.info("Creating multi-label target variable 'y'...")

        white_ball_range = range(1, 70)
        pb_range = range(1, 27)

        wb_cols = [f"wb_{i}" for i in white_ball_range]
        pb_cols = [f"pb_{i}" for i in pb_range]

        y = pd.DataFrame(0, index=features_df.index, columns=wb_cols + pb_cols)

        for i in white_ball_range:
            y[f"wb_{i}"] = features_df[self.white_ball_columns].eq(i).any(axis=1).astype(int)

        for i in pb_range:
            y[f"pb_{i}"] = (features_df[self.pb_column] == i).astype(int)

        logger.info(f"Target variable 'y' created with shape: {y.shape}")
        return y

    def train(self, features_df):
        logger.info("Starting model training with multi-label classification objective...")

        X = self._get_feature_matrix(features_df)
        y = self.create_target_variable(features_df)
        self.target_columns = y.columns.tolist()

        combined = pd.concat([X, y], axis=1).dropna()
        X = combined[X.columns]
        y = combined[y.columns]

        if X.empty or y.empty:
           logger.error("Not enough data to train after dropping NaNs.")
           return None

        X_train, X_test, y_train, y_test = self._split_train_test_data(X, y)

        self.model = self._initialize_model()
        self.model.fit(X_train, y_train)
        logger.info("Model training complete.")

        self.evaluate_model(X_test, y_test)
        self.save_model()
        return self.model

    def _get_feature_matrix(self, features_df):
        feature_cols = [
            "even_count", "odd_count", "sum", "spread", "consecutive_count",
            "avg_delay", "max_delay", "min_delay",
            "dist_to_recent", "avg_dist_to_top_n", "dist_to_centroid",
            "time_weight", "increasing_trend_count", "decreasing_trend_count",
            "stable_trend_count"
        ]
        available_features = [col for col in feature_cols if col in features_df.columns]
        logger.info(f"Using features: {available_features}")
        return features_df[available_features]

    def _split_train_test_data(self, X, y):
        return train_test_split(
            X, y,
            test_size=float(self.config["model_params"]["test_size"]),
            random_state=int(self.config["model_params"]["random_state"])
        )

    def _initialize_model(self):
        base_classifier = XGBClassifier(
            n_estimators=int(self.config["model_params"]["n_estimators"]),
            learning_rate=float(self.config["model_params"].get("learning_rate", 0.1)),
            random_state=int(self.config["model_params"]["random_state"]),
            objective='binary:logistic',
            eval_metric='logloss'
        )
        return MultiOutputClassifier(estimator=base_classifier, n_jobs=-1)

    def evaluate_model(self, X_test, y_test):
        if self.model is None:
            logger.error("Model not trained. Cannot evaluate.")
            return None

        y_pred_proba = self.model.predict_proba(X_test)
        y_pred_proba_flat = np.array([arr[:, 1] for arr in y_pred_proba]).T

        try:
            auc = roc_auc_score(y_test, y_pred_proba_flat, average='macro')
            ll = log_loss(y_test, y_pred_proba_flat)

            logger.info("Model evaluation metrics:")
            logger.info(f"  Macro-Averaged AUC: {auc:.4f}")
            logger.info(f"  Log Loss: {ll:.4f}")

            return {"roc_auc_macro": auc, "log_loss": ll}
        except Exception as e:
            logger.error(f"Could not calculate evaluation metrics: {e}")
            return None

    def save_model(self):
        if self.model:
            model_bundle = {
                "model": self.model,
                "target_columns": self.target_columns
            }
            joblib.dump(model_bundle, self.model_path)
            logger.info(f"Model bundle saved to {self.model_path}")

    def load_model(self):
        try:
            model_bundle = joblib.load(self.model_path)
            self.model = model_bundle["model"]
            self.target_columns = model_bundle.get("target_columns")
            if self.target_columns is None:
                logger.warning("Loaded a model without target_columns. Prediction may fail.")
            logger.info(f"Model bundle loaded from {self.model_path}")
            return self.model
        except FileNotFoundError:
            logger.error(f"Model file not found at {self.model_path}. A new one will be created upon training.")
            return None
        except (KeyError, AttributeError):
             logger.warning(f"Failed to load model bundle from {self.model_path}, it might be an old version. Will create a new model.")
             return None

    def predict_probabilities(self, features_df):
        if self.model is None or self.target_columns is None:
            logger.error("Model or target columns not loaded. Cannot predict.")
            return None

        X = self._validate_prediction_features(features_df)
        if X is None:
            logger.error("Feature validation failed during prediction.")
            return None

        pred_probas = self.model.predict_proba(X)
        prob_class_1 = [p[:, 1] for p in pred_probas]

        prob_df = pd.DataFrame(np.array(prob_class_1).T, columns=self.target_columns, index=X.index)
        return prob_df

    def _validate_prediction_features(self, features_df):
        try:
            model_features = self.model.estimators_[0].get_booster().feature_names

            missing_features = [f for f in model_features if f not in features_df.columns]
            if missing_features:
                logger.warning(f"Missing features for prediction: {missing_features}")
                return None

            return features_df[model_features]
        except Exception as e:
            logger.error(f"Error validating prediction features: {e}")
            return None

class Predictor:
    """
    Handles the prediction of number probabilities using the trained model.
    """
    def __init__(self, config_path: str = "config/config.ini"):
        """
        Initialize the Predictor with configuration and feature engineering.

        Args:
            config_path: Path to the configuration file
        """
        logger.info("Initializing Predictor...")

        self.config = configparser.ConfigParser()
        self.config.read(config_path)

        # Initialize components
        self.data_loader = get_data_loader()
        self.model_trainer = ModelTrainer(self.config["paths"]["model_file"])

        # Load historical data for feature engineer and deterministic generator
        self.historical_data = self.data_loader.load_historical_data()

        self.feature_engineer = FeatureEngineer(self.historical_data)
        self.deterministic_generator = DeterministicGenerator(self.historical_data)

        # Model state
        self.model = None
        self.model_metadata = {}

        # Initialize ensemble system
        self.ensemble_predictor = None
        self.use_ensemble = self.config.getboolean("ensemble", "use_ensemble", fallback=True)  # Flag to enable/disable ensemble

        # Load existing model if available
        self.load_model()

        # Initialize ensemble system if enabled
        if self.use_ensemble:
            self._initialize_ensemble_system()

    def load_model(self):
        """Load the trained model from the specified path."""
        self.model = self.model_trainer.load_model()
        if self.model is None:
            logger.warning("No pre-trained model found. Model will be trained on first run if data is available.")
            # Raising an error here might be too strict if training is expected immediately.
            # Consider if this should be a hard error or a warning.
            # raise RuntimeError("Failed to load the model. Please train a model first using 'train' command.")
        else:
            logger.info("Model loaded successfully.")
            # You might want to load metadata related to the model here if available
            # self.model_metadata = self.model_trainer.load_model_metadata()

    def _initialize_ensemble_system(self) -> None:
        """Initialize the enhanced ensemble prediction system"""
        try:
            # Load historical data for ensemble
            historical_data = self.data_loader.load_historical_data()

            # Initialize enhanced ensemble predictor
            self.ensemble_predictor = EnsemblePredictor(historical_data, models_dir="models/")

            logger.info("Enhanced ensemble prediction system initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize ensemble system: {e}")
            self.use_ensemble = False

    def predict_probabilities(self, use_ensemble: bool = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate probability predictions for white balls and Powerball.

        Args:
            use_ensemble: Whether to use ensemble prediction (overrides default setting)

        Returns:
            Tuple of (white_ball_probabilities, powerball_probabilities)
        """
        # Determine if we should use ensemble
        should_use_ensemble = use_ensemble if use_ensemble is not None else self.use_ensemble

        # Try ensemble prediction first if enabled and available
        if should_use_ensemble and self.ensemble_predictor is not None:
            try:
                logger.info("Using ensemble prediction system")
                ensemble_result = self.ensemble_predictor.predict_ensemble()

                if ensemble_result and 'white_ball_probabilities' in ensemble_result:
                    wb_probs = ensemble_result['white_ball_probabilities']
                    pb_probs = ensemble_result['powerball_probabilities']

                    # Log ensemble details
                    logger.info(f"Ensemble prediction completed using {ensemble_result.get('total_models', 0)} models")
                    logger.info(f"Ensemble method: {ensemble_result.get('ensemble_method', 'unknown')}")

                    return wb_probs, pb_probs

            except Exception as e:
                logger.error(f"Ensemble prediction failed, falling back to single model: {e}")

        # Fallback to single model prediction
        if self.model is None:
            logger.error("Model not loaded. Cannot generate predictions.")
            # Return uniform probabilities as fallback
            wb_probs = np.ones(69) / 69
            pb_probs = np.ones(26) / 26
            return wb_probs, pb_probs

        try:
            logger.info("Using single model prediction")

            # Use already loaded historical data
            if self.historical_data.empty:
                self.historical_data = self.data_loader.load_historical_data()

            # Generate features
            features = self.feature_engineer.engineer_features(use_temporal_analysis=True)

            # Use the latest features for prediction
            latest_features = features.iloc[-1:].values

            # Get predictions from the model
            predictions = self.model.predict_proba(latest_features)

            # Extract white ball and Powerball probabilities
            if isinstance(predictions, list):
                # Multi-output classifier case
                wb_probs = np.array([pred[:, 1] if pred.shape[1] > 1 else pred.flatten() 
                                   for pred in predictions]).flatten()

                # Generate Powerball probabilities (simple approach for now)
                # This part might need a more sophisticated approach based on the model's output
                # For now, we'll use a placeholder or random generation if not directly available
                # A better approach would be to map specific model outputs to Powerball probabilities
                if len(wb_probs) > 69: # If the model predicts more than 69 outputs, assume last 26 are for PB
                    pb_probs = wb_probs[69:]
                    wb_probs = wb_probs[:69]
                else: # Placeholder for PB if not predicted by the main model
                    pb_probs = np.random.random(26)
            else:
                # Single output case - need to handle appropriately
                wb_probs = predictions.flatten()
                pb_probs = np.random.random(26) # Placeholder

            # Ensure correct dimensions and normalize
            wb_probs = wb_probs[:69] if len(wb_probs) >= 69 else np.pad(wb_probs, (0, 69 - len(wb_probs)))
            pb_probs = pb_probs[:26] if len(pb_probs) >= 26 else np.pad(pb_probs, (0, 26 - len(pb_probs)))

            wb_probs = wb_probs / wb_probs.sum() if wb_probs.sum() > 0 else wb_probs
            pb_probs = pb_probs / pb_probs.sum() if pb_probs.sum() > 0 else pb_probs

            return wb_probs, pb_probs

        except Exception as e:
            logger.error(f"Error generating predictions: {e}")
            # Return uniform probabilities as fallback
            wb_probs = np.ones(69) / 69
            pb_probs = np.ones(26) / 26
            return wb_probs, pb_probs


    def get_prediction_features(self):
        """
        Prepares the feature set for the next draw to be predicted.
        """
        logger.info("Preparing features for the next prediction...")

        if self.historical_data.empty:
            self.historical_data = self.data_loader.load_historical_data()

        if self.historical_data.empty:
            raise ValueError("Historical data is empty. Cannot prepare features for prediction.")

        all_features = self.feature_engineer.engineer_features(use_temporal_analysis=True)

        latest_features = all_features.iloc[-1:]

        logger.info("Successfully prepared features for one prediction row.")
        return latest_features

    def predict_deterministic(self, save_to_log: bool = True) -> Dict:
        """
        Genera una predicción determinística usando el sistema de scoring multi-criterio.

        Args:
            save_to_log: Si True, guarda la predicción en el log de base de datos

        Returns:
            Dict con la predicción determinística y sus detalles
        """
        logger.info("Generating deterministic prediction with multi-criteria scoring...")

        # Obtener probabilidades del modelo
        wb_probs, pb_probs = self.predict_probabilities()

        # Usar datos históricos ya cargados
        if self.historical_data.empty:
            self.historical_data = self.data_loader.load_historical_data()

        if self.historical_data.empty:
            raise ValueError("Historical data is empty. Cannot generate deterministic prediction.")

        # Usar generador determinístico ya inicializado
        deterministic_generator = self.deterministic_generator

        # Generar predicción top
        prediction = deterministic_generator.generate_top_prediction(wb_probs, pb_probs)

        # Guardar en log si se solicita
        if save_to_log:
            prediction_id = save_prediction_log(prediction)
            if prediction_id:
                prediction['log_id'] = prediction_id
                logger.info(f"Deterministic prediction saved to log with ID: {prediction_id}")
            else:
                logger.warning("Failed to save prediction to log")

        logger.info("Deterministic prediction generated successfully")
        return prediction

    def predict_diverse_plays(self, num_plays: int = 5, save_to_log: bool = True) -> List[Dict]:
        """
        Genera múltiples predicciones diversas de alta calidad para el próximo sorteo.

        Args:
            num_plays: Número de plays diversos a generar (default: 5)
            save_to_log: Si True, guarda las predicciones en el log de base de datos

        Returns:
            Lista de Dict con las predicciones diversas y sus detalles
        """
        logger.info(f"Generating {num_plays} diverse high-quality plays for next drawing...")

        # Obtener probabilidades del modelo
        wb_probs, pb_probs = self.predict_probabilities()

        # Usar datos históricos ya cargados
        if self.historical_data.empty:
            self.historical_data = self.data_loader.load_historical_data()

        if self.historical_data.empty:
            raise ValueError("Historical data is empty. Cannot generate diverse predictions.")

        # Usar generador determinístico ya inicializado
        deterministic_generator = self.deterministic_generator

        # Generar predicciones diversas
        diverse_predictions = deterministic_generator.generate_diverse_predictions(
            wb_probs, pb_probs, num_plays=num_plays
        )

        # Guardar en log si se solicita
        if save_to_log:
            saved_count = 0
            for i, prediction in enumerate(diverse_predictions):
                prediction_id = save_prediction_log(prediction)
                if prediction_id:
                    prediction['log_id'] = prediction_id
                    saved_count += 1
                else:
                    logger.warning(f"Failed to save prediction {i+1} to log")

            logger.info(f"Saved {saved_count}/{len(diverse_predictions)} diverse predictions to log")

        logger.info(f"Generated {len(diverse_predictions)} diverse plays successfully")
        return diverse_predictions

    def predict_syndicate_plays(self, num_plays: int = 100, save_to_log: bool = True) -> List[Dict]:
        """
        Genera un gran número de predicciones optimizadas para juego en sindicato.

        Args:
            num_plays: Número de jugadas para sindicato (default: 100)
            save_to_log: Si True, guarda las predicciones en el log

        Returns:
            Lista de predicciones optimizadas para sindicato
        """
        logger.info(f"Generating {num_plays} syndicate plays with advanced scoring...")

        # Obtener probabilidades del modelo
        wb_probs, pb_probs = self.predict_probabilities()

        # Usar datos históricos ya cargados
        if self.historical_data.empty:
            self.historical_data = self.data_loader.load_historical_data()

        if self.historical_data.empty:
            raise ValueError("Historical data is empty. Cannot generate syndicate predictions.")

        # Usar generador determinístico ya inicializado
        deterministic_generator = self.deterministic_generator

        # Generar un pool más grande de candidatos (5x el número solicitado)
        candidate_pool_size = max(5000, num_plays * 5)

        # Generar candidatos con scoring avanzado
        syndicate_predictions = deterministic_generator.generate_diverse_predictions(
            wb_probs, pb_probs, 
            num_plays=num_plays,
            num_candidates=candidate_pool_size
        )

        # Aplicar análisis adicional para sindicatos
        for i, prediction in enumerate(syndicate_predictions):
            prediction['syndicate_rank'] = i + 1
            prediction['syndicate_tier'] = self._classify_syndicate_tier(prediction['score_total'])
            prediction['expected_coverage'] = self._calculate_coverage_score(prediction, syndicate_predictions)

        # Guardar en log si se solicita
        if save_to_log:
            saved_count = 0
            for prediction in syndicate_predictions:
                prediction_id = save_prediction_log(prediction)
                if prediction_id:
                    prediction['log_id'] = prediction_id
                    saved_count += 1

            logger.info(f"Saved {saved_count}/{len(syndicate_predictions)} syndicate predictions to log")

        logger.info(f"Generated {len(syndicate_predictions)} syndicate plays successfully")
        return syndicate_predictions

    def _classify_syndicate_tier(self, score: float) -> str:
        """Clasifica las jugadas por tiers para sindicatos."""
        if score >= 0.8:
            return "Premium"
        elif score >= 0.6:
            return "High"
        elif score >= 0.4:
            return "Medium"
        else:
            return "Standard"

    def _calculate_coverage_score(self, prediction: Dict, all_predictions: List[Dict]) -> float:
        """Calcula qué tan bien una jugada complementa las otras en el sindicato."""
        numbers = set(prediction['numbers'] + [prediction['powerball']])

        coverage_scores = []
        for other_pred in all_predictions[:20]:  # Comparar con las top 20
            if other_pred == prediction:
                continue

            other_numbers = set(other_pred['numbers'] + [other_pred['powerball']])
            overlap = len(numbers.intersection(other_numbers))
            coverage = 1.0 - (overlap / 6.0)  # Menos overlap = mejor coverage
            coverage_scores.append(coverage)

        return np.mean(coverage_scores) if coverage_scores else 0.5

    def predict_ensemble_syndicate(self, num_plays: int = 100) -> List[Dict]:
        """
        Genera jugadas usando múltiples modelos y selecciona las mejores basado en ensemble scoring.

        Args:
            num_plays: Número final de jugadas a generar

        Returns:
            Lista de las mejores jugadas seleccionadas de múltiples modelos
        """
        logger.info(f"Generating ensemble syndicate predictions with multiple models...")

        # Obtener probabilidades base
        wb_probs, pb_probs = self.predict_probabilities()

        # Obtener datos históricos
        historical_data = self.data_loader.load_historical_data()

        all_candidates = []

        # 1. Generador Determinístico (60% del pool)
        deterministic_gen = DeterministicGenerator(historical_data)
        deterministic_candidates = deterministic_gen.generate_diverse_predictions(
            wb_probs, pb_probs, 
            num_plays=int(num_plays * 0.6),
            num_candidates=num_plays * 3
        )

        for candidate in deterministic_candidates:
            candidate['model_source'] = 'deterministic'
            candidate['ensemble_weight'] = 0.6

        all_candidates.extend(deterministic_candidates)

        # 2. Sistema Adaptativo (25% del pool)
        # Check if AdaptivePlayScorer is available and if it should be used
        try:
            from src.adaptive_feedback import AdaptivePlayScorer
            # Check if the ensemble_predictor is initialized and has adaptive capabilities if needed
            # For now, we assume if AdaptivePlayScorer is importable, it's usable.
            adaptive_scorer = AdaptivePlayScorer(historical_data)

            # Generate candidates with adaptive scoring
            adaptive_candidates = []
            for i in range(int(num_plays * 0.25)):
                # Use the deterministic generator but with adaptive scoring
                candidate = deterministic_gen.generate_top_prediction(wb_probs, pb_probs)

                # Recalculate score with adaptive system
                adaptive_scores = adaptive_scorer.calculate_total_score(
                    candidate['numbers'], candidate['powerball'], wb_probs, pb_probs
                )

                candidate['score_total'] = adaptive_scores['total']
                candidate['score_details'] = adaptive_scores
                candidate['model_source'] = 'adaptive'
                candidate['ensemble_weight'] = 0.25

                adaptive_candidates.append(candidate)

            all_candidates.extend(adaptive_candidates)
            logger.info("Adaptive model contributions included in ensemble.")

        except ImportError:
            logger.warning("AdaptivePlayScorer not found. Skipping adaptive model contribution.")
        except Exception as e:
            logger.warning(f"Error incorporating adaptive model: {e}")

        # 3. Generador Inteligente Híbrido (15% del pool)
        try:
            from src.intelligent_generator import IntelligentGenerator
            intelligent_gen = IntelligentGenerator(historical_data)

            intelligent_candidates = []
            # The number of plays for intelligent generator should be based on num_plays * 0.15
            # and num_candidates should be sufficiently large for diversity.
            hybrid_plays = intelligent_gen.generate_plays(
                num_plays=int(num_plays * 0.15), 
                num_candidates=num_plays * 2 # Adjust num_candidates as needed
            )

            for play in hybrid_plays:
                # Calculate scores for these plays
                # Ensure play is in the format expected by calculate_total_score
                # Assuming play is a list of numbers and the last element is powerball if applicable
                numbers = play[:5] if len(play) >= 5 else play
                powerball = play[5] if len(play) > 5 else 1 # Default powerball if not provided

                scores = deterministic_gen.scorer.calculate_total_score(
                    numbers, powerball, wb_probs, pb_probs
                )

                candidate = {
                    'numbers': numbers,
                    'powerball': powerball,
                    'score_total': scores['total'],
                    'score_details': scores,
                    'model_source': 'intelligent',
                    'ensemble_weight': 0.15,
                    'timestamp': datetime.now().isoformat()
                }

                intelligent_candidates.append(candidate)

            all_candidates.extend(intelligent_candidates)
            logger.info("Intelligent generator contributions included in ensemble.")

        except ImportError:
            logger.warning("IntelligentGenerator not found. Skipping intelligent generator contribution.")
        except Exception as e:
            logger.warning(f"Error incorporating intelligent generator: {e}")

        # Combine and rank all candidates
        ensemble_scores = []
        for candidate in all_candidates:
            # Ensemble score calculation: weighted sum of base score and diversity bonus
            base_score = candidate.get('score_total', 0) # Default to 0 if score_total is missing
            weight = candidate.get('ensemble_weight', 0.33) # Default weight if not specified
            diversity_bonus = self._calculate_ensemble_diversity(candidate, all_candidates)

            ensemble_score = (base_score * weight) + (diversity_bonus * 0.1) # Adjust diversity bonus impact

            candidate['ensemble_score'] = ensemble_score
            ensemble_scores.append(candidate)

        # Sort by ensemble score and take the top 'num_plays'
        ensemble_scores.sort(key=lambda x: x.get('ensemble_score', 0), reverse=True)
        final_predictions = ensemble_scores[:num_plays]

        # Add ensemble metadata
        for i, prediction in enumerate(final_predictions):
            prediction['ensemble_rank'] = i + 1
            prediction['method'] = 'ensemble_syndicate' # Indicate this prediction comes from ensemble syndicate

        logger.info(f"Generated {len(final_predictions)} ensemble syndicate predictions")
        return final_predictions

    def _calculate_ensemble_diversity(self, candidate: Dict, all_candidates: List[Dict]) -> float:
        """Calculates a diversity bonus for the ensemble candidate."""
        candidate_nums = set(candidate.get('numbers', []) + [candidate.get('powerball', 1)])

        diversity_scores = []
        # Compare against a subset of other candidates to avoid O(N^2) complexity, and focus on top candidates
        comparison_candidates = all_candidates[:min(len(all_candidates), 100)] # Compare with top 100

        for other in comparison_candidates:
            if other == candidate: # Skip self-comparison
                continue

            other_nums = set(other.get('numbers', []) + [other.get('powerball', 1)])

            # Jaccard index for set similarity, then convert to diversity (1 - similarity)
            intersection = len(candidate_nums.intersection(other_nums))
            union = len(candidate_nums.union(other_nums))

            if union == 0: # Handle cases with empty sets
                jaccard_similarity = 1.0 if len(candidate_nums) == 0 else 0.0
            else:
                jaccard_similarity = intersection / union

            diversity_score = 1.0 - jaccard_similarity
            diversity_scores.append(diversity_score)

        return np.mean(diversity_scores) if diversity_scores else 0.5 # Default diversity if no comparisons possible


    def compare_prediction_methods(self) -> Dict:
        """
        Compares the traditional method with the deterministic method for analysis.

        Returns:
            A dictionary containing the comparison results.
        """
        logger.info("Comparing traditional vs deterministic prediction methods...")

        # Get base probabilities from the model
        try:
            wb_probs, pb_probs = self.predict_probabilities()
        except ValueError as e:
            logger.error(f"Could not get probabilities for comparison: {e}")
            return {"error": str(e)}

        comparison_results = {}

        # 1. Traditional Method (using IntelligentGenerator)
        try:
            from src.intelligent_generator import IntelligentGenerator
            traditional_generator = IntelligentGenerator()
            # Assuming generate_plays can take probabilities and return a DataFrame or similar structure
            # The exact call might need adjustment based on IntelligentGenerator's signature
            traditional_plays_df = traditional_generator.generate_plays(wb_probs, pb_probs, num_plays=1)

            if not traditional_plays_df.empty:
                # Extract numbers and powerball from the first row
                # Adjust column names based on IntelligentGenerator output
                first_play = traditional_plays_df.iloc[0]
                comparison_results['traditional_method'] = {
                    'numbers': first_play[['n1', 'n2', 'n3', 'n4', 'n5']].tolist(),
                    'powerball': int(first_play['pb']),
                    'method': 'weighted_random_sampling', # Or whatever method IntelligentGenerator uses
                    'reproducible': False # Typically random methods are not reproducible without seeds
                }
            else:
                comparison_results['traditional_method'] = {"error": "IntelligentGenerator returned no plays."}
        except ImportError:
            logger.warning("IntelligentGenerator not available for comparison.")
            comparison_results['traditional_method'] = {"error": "IntelligentGenerator not found."}
        except Exception as e:
            logger.error(f"Error during traditional method comparison: {e}")
            comparison_results['traditional_method'] = {"error": str(e)}


        # 2. Deterministic Method
        try:
            historical_data = self.data_loader.load_historical_data()
            if historical_data.empty:
                raise ValueError("Historical data is empty for deterministic comparison.")

            deterministic_generator = DeterministicGenerator(historical_data)
            deterministic_prediction = deterministic_generator.generate_top_prediction(wb_probs, pb_probs)

            comparison_results['deterministic_method'] = {
                'numbers': deterministic_prediction.get('numbers', []),
                'powerball': deterministic_prediction.get('powerball', 1),
                'total_score': deterministic_prediction.get('score_total'),
                'score_details': deterministic_prediction.get('score_details'),
                'method': 'multi_criteria_scoring',
                'reproducible': True, # Deterministic methods should be reproducible
                'dataset_hash': deterministic_prediction.get('dataset_hash') # If hash is generated
            }
        except Exception as e:
            logger.error(f"Error during deterministic method comparison: {e}")
            comparison_results['deterministic_method'] = {"error": str(e)}

        comparison_results['comparison_timestamp'] = datetime.now().isoformat()

        logger.info("Method comparison completed.")
        return comparison_results

    def get_ensemble_summary(self) -> Dict[str, Any]:
        """Get summary of ensemble system status"""
        if not self.ensemble_predictor:
            return {
                'ensemble_enabled': False,
                'reason': 'Ensemble system not initialized'
            }

        try:
            summary = self.ensemble_predictor.get_ensemble_summary()
            summary['ensemble_enabled'] = True
            return summary

        except Exception as e:
            return {
                'ensemble_enabled': False,
                'reason': f'Error getting ensemble summary: {e}'
            }

    def update_ensemble_performance(self, performance_feedback: Dict[str, float]) -> None:
        """Update ensemble model performance based on feedback"""
        if self.ensemble_predictor:
            self.ensemble_predictor.update_model_weights(performance_feedback)
            logger.info("Ensemble performance updated")
        else:
            logger.warning("Ensemble system not available for performance update")

    def set_ensemble_method(self, method: str) -> bool:
        """Set the ensemble method to use"""
        if not self.ensemble_predictor:
            logger.warning("Ensemble system not available")
            return False

        return self.ensemble_predictor.set_ensemble_method(method)
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
from typing import Dict, List, Tuple, Any, Optional

from src.loader import get_data_loader, DataLoader # Assuming DataLoader is available for retraining
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
    def __init__(self, model_path_or_data):
        self.model_path = None
        self.config = configparser.ConfigParser()
        self.config.read("config/config.ini")

        if isinstance(model_path_or_data, str):
            self.model_path = model_path_or_data
            self.model = self.load_model()
            self.target_columns = self.load_target_columns()
        elif isinstance(model_path_or_data, pd.DataFrame):
            self.data = model_path_or_data
            self.model = None
            self.target_columns = None
            logger.info("ModelTrainer initialized with data for training.")
        else:
            raise TypeError("model_path_or_data must be a file path (str) or a pandas DataFrame.")

        # Handle both original and merged column names
        self.white_ball_columns = [f"n{i}" for i in range(1, 6)] # Original column names
        self.pb_column = "pb" # Original powerball column
        logger.info("ModelTrainer initialized.")

    def create_target_variable(self, features_df):
        """
        Creates a multi-label target variable for white balls and powerball.
        Handles both original and merged column names from pandas merge operations.
        """
        logger.info("Creating multi-label target variable 'y'...")

        # Define the possible range of numbers
        white_ball_range = range(1, 70)
        pb_range = range(1, 27)

        # Define column names for the target variable
        wb_cols = [f"wb_{i}" for i in white_ball_range]
        pb_cols = [f"pb_{i}" for i in pb_range]

        # Initialize target DataFrame with zeros
        y = pd.DataFrame(0, index=features_df.index, columns=wb_cols + pb_cols)

        # Detect actual column names (handle merged columns)
        available_cols = features_df.columns.tolist()
        logger.debug(f"Available columns: {available_cols}")
        
        # Find white ball columns (try original first, then _x, then _y suffixes)
        wb_columns_to_use = []
        pb_column_to_use = None
        
        for i in range(1, 6):
            col_found = False
            for suffix in ['', '_x', '_y']:
                col_name = f"n{i}{suffix}"
                if col_name in available_cols:
                    wb_columns_to_use.append(col_name)
                    col_found = True
                    break
            if not col_found:
                logger.error(f"Could not find column n{i} with any suffix")
                
        # Find powerball column
        for suffix in ['', '_x', '_y']:
            col_name = f"pb{suffix}"
            if col_name in available_cols:
                pb_column_to_use = col_name
                break
                
        if not pb_column_to_use:
            logger.error("Could not find powerball column with any suffix")
            
        logger.info(f"Using white ball columns: {wb_columns_to_use}")
        logger.info(f"Using powerball column: {pb_column_to_use}")

        if len(wb_columns_to_use) == 5 and pb_column_to_use:
            # Populate white ball targets
            for i in white_ball_range:
                # Check if number 'i' is present in any of the white ball columns for a given row
                y[f"wb_{i}"] = features_df[wb_columns_to_use].eq(i).any(axis=1).astype(int)

            # Populate powerball targets
            for i in pb_range:
                # Check if the powerball column matches number 'i'
                y[f"pb_{i}"] = (features_df[pb_column_to_use] == i).astype(int)
        else:
            logger.error("Could not find all required draw number columns. Creating empty targets.")

        logger.info(f"Target variable 'y' created with shape: {y.shape}")
        return y

    def train_model(self):
        """Train the model using the provided data."""
        if self.data is None:
            logger.error("No data provided for training. Call `load_data` first.")
            return False

        logger.info("Starting model training with multi-label classification objective...")

        X = self._get_feature_matrix(self.data)
        y = self.create_target_variable(self.data)
        self.target_columns = y.columns.tolist()

        # Ensure alignment and drop rows with NaNs in either features or target
        combined = pd.concat([X, y], axis=1).dropna()
        X = combined[X.columns]
        y = combined[y.columns]

        if X.empty or y.empty:
            logger.error("Not enough data to train after dropping NaNs. Please check data quality.")
            return False

        X_train, X_test, y_train, y_test = self._split_train_test_data(X, y)

        self.model = self._initialize_model()
        self.model.fit(X_train, y_train)
        logger.info("Model training complete.")

        # Evaluate and save the model
        self.evaluate_model(X_test, y_test)
        self.save_model()
        return True

    def _get_feature_matrix(self, features_df):
        """Extracts and selects relevant features for the model."""
        # Define the feature names that the model expects
        feature_cols = [
            "even_count", "odd_count", "sum", "spread", "consecutive_count",
            "avg_delay", "max_delay", "min_delay",
            "dist_to_recent", "avg_dist_to_top_n", "dist_to_centroid",
            "time_weight", "increasing_trend_count", "decreasing_trend_count",
            "stable_trend_count"
        ]
        # Filter for features that are actually present in the input DataFrame
        available_features = [col for col in feature_cols if col in features_df.columns]
        if len(available_features) != len(feature_cols):
            missing = set(feature_cols) - set(available_features)
            logger.warning(f"Missing expected features for training: {missing}. Proceeding with available features.")

        logger.info(f"Using features for training: {available_features}")
        return features_df[available_features]

    def _split_train_test_data(self, X, y):
        """Splits the data into training and testing sets."""
        test_size = float(self.config["model_params"]["test_size"])
        random_state = int(self.config["model_params"]["random_state"])
        logger.info(f"Splitting data into train/test sets (test_size={test_size}, random_state={random_state})...")
        return train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state
        )

    def _initialize_model(self):
        """Initializes the XGBoost model with specified parameters."""
        n_estimators = int(self.config["model_params"]["n_estimators"])
        learning_rate = float(self.config["model_params"].get("learning_rate", 0.1))
        random_state = int(self.config["model_params"]["random_state"])

        logger.info(f"Initializing XGBoost model: n_estimators={n_estimators}, learning_rate={learning_rate}")

        base_classifier = XGBClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=random_state,
            objective='binary:logistic',
            eval_metric='logloss'
        )
        # Wrap with MultiOutputClassifier for multi-label prediction
        return MultiOutputClassifier(estimator=base_classifier, n_jobs=-1)

    def evaluate_model(self, X_test, y_test):
        """Evaluates the trained model using AUC and Log Loss."""
        if self.model is None:
            logger.error("Model not trained. Cannot evaluate.")
            return None

        logger.info("Evaluating model performance...")
        try:
            # Predict probabilities for the test set
            y_pred_proba = self.model.predict_proba(X_test)
            # Extract probabilities for class 1 (positive class)
            y_pred_proba_flat = np.array([arr[:, 1] for arr in y_pred_proba]).T

            # Calculate metrics
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
        """Saves the trained model and target columns to a file."""
        if self.model:
            model_bundle = {
                "model": self.model,
                "target_columns": self.target_columns,
                "version": "v6.0", # Versioning for the model
                "last_trained": datetime.now().isoformat()
            }
            try:
                joblib.dump(model_bundle, self.model_path)
                logger.info(f"Model bundle saved to {self.model_path}")
            except Exception as e:
                logger.error(f"Failed to save model bundle to {self.model_path}: {e}")
        else:
            logger.warning("No model available to save.")

    def load_model(self):
        """Loads the model and target columns from a file."""
        if not self.model_path:
            logger.warning("Model path not set. Cannot load model.")
            return None
        try:
            model_bundle = joblib.load(self.model_path)
            self.model = model_bundle.get("model")
            self.target_columns = model_bundle.get("target_columns")
            self.feature_names = model_bundle.get("feature_names", []) # Store feature names if available

            if self.model is None or self.target_columns is None:
                logger.warning(f"Loaded model bundle from {self.model_path} is incomplete or corrupted. Retraining might be necessary.")
                return None

            logger.info(f"Model bundle loaded from {self.model_path}")
            return self.model
        except FileNotFoundError:
            logger.error(f"Model file not found at {self.model_path}. A new one will be created upon training.")
            return None
        except (KeyError, AttributeError, EOFError, joblib.externals.loky.process_executor.TerminatedWorkerError) as e:
             logger.warning(f"Failed to load model bundle from {self.model_path} due to error: {e}. It might be an old version or corrupted. Will attempt to create a new model.")
             return None
        except Exception as e:
            logger.error(f"An unexpected error occurred loading the model: {e}")
            return None

    def load_target_columns(self):
        """Loads the target columns from the model bundle."""
        if not self.model_path:
            return None
        try:
            model_bundle = joblib.load(self.model_path)
            return model_bundle.get("target_columns")
        except Exception as e:
            logger.warning(f"Could not load target columns from {self.model_path}: {e}")
            return None

    def predict_probabilities(self, features_df):
        """Predicts probabilities for white balls and powerball."""
        if self.model is None:
            logger.error("Model not loaded. Cannot predict probabilities.")
            return None
        if self.target_columns is None:
            logger.error("Target columns not loaded. Cannot predict probabilities.")
            return None

        X = self._validate_prediction_features(features_df)
        if X is None:
            logger.error("Feature validation failed during prediction.")
            return None

        try:
            pred_probas = self.model.predict_proba(X)
            # Assuming MultiOutputClassifier returns a list of probability arrays
            # We need to extract probabilities for class 1 for each target
            prob_class_1 = [p[:, 1] for p in pred_probas]

            # Create a DataFrame from the predictions, ensuring columns match target_columns
            prob_df = pd.DataFrame(np.array(prob_class_1).T, columns=self.target_columns, index=X.index)
            return prob_df
        except Exception as e:
            logger.error(f"Error during probability prediction: {e}")
            return None

    def _validate_prediction_features(self, features_df):
        """Validates features for prediction against model's expected features."""
        if self.model is None:
            logger.error("Model is not loaded, cannot validate features.")
            return None

        try:
            # Attempt to get feature names from the model's estimator
            # This assumes the first estimator in MultiOutputClassifier holds the feature names
            model_features = self.model.estimators_[0].get_booster().feature_names
        except (AttributeError, IndexError, TypeError) as e:
            logger.error(f"Could not retrieve feature names from the model: {e}. Using default feature list.")
            # Fallback to a predefined list if model introspection fails
            model_features = [
                "even_count", "odd_count", "sum", "spread", "consecutive_count",
                "avg_delay", "max_delay", "min_delay",
                "dist_to_recent", "avg_dist_to_top_n", "dist_to_centroid",
                "time_weight", "increasing_trend_count", "decreasing_trend_count",
                "stable_trend_count"
            ]

        # Check for missing features in the input DataFrame
        missing_features = [f for f in model_features if f not in features_df.columns]
        if missing_features:
            logger.warning(f"Input features are missing: {missing_features}. Predictions might be inaccurate.")
            # Optionally, add missing columns with default values (e.g., 0 or mean)
            # For now, we'll proceed but warn the user.

        # Return only the features expected by the model, in the correct order
        present_model_features = [f for f in model_features if f in features_df.columns]
        if len(present_model_features) != len(model_features):
             logger.warning(f"Some model features ({set(model_features) - set(present_model_features)}) are not in the input DataFrame.")

        # Ensure the order is correct
        ordered_features = [f for f in model_features if f in features_df.columns]

        # Check for unexpected features in input
        unexpected_features = [f for f in features_df.columns if f not in model_features]
        if unexpected_features:
            logger.warning(f"Input features contain unexpected columns: {unexpected_features}. These will be ignored.")

        return features_df[ordered_features]

class Predictor:
    """
    Handles the prediction of number probabilities using the trained model and ensemble methods.
    """
    def __init__(self, config_path: str = "config/config.ini"):
        """
        Initialize the Predictor with configuration and feature engineering.

        Args:
            config_path: Path to the configuration file
        """
        logger.info("Initializing Predictor v6.0...")

        self.config = configparser.ConfigParser()
        self.config.read(config_path)

        # Initialize components
        self.data_loader = DataLoader() # Use DataLoader for flexibility
        self.model_trainer = ModelTrainer(self.config["paths"]["model_file"]) # Initialize with model path

        # Load historical data for feature engineer and deterministic generator
        self.historical_data = self.data_loader.load_historical_data()

        self.feature_engineer = FeatureEngineer(self.historical_data)
        self.deterministic_generator = DeterministicGenerator(self.historical_data)

        # Model state
        self.model = None
        self.model_metadata = {}

        # Initialize ensemble system
        self.ensemble_predictor = None
        self.use_ensemble = self.config.getboolean("ensemble", "use_ensemble", fallback=True)

        # Load existing model
        self.load_model()

        # Initialize ensemble system if enabled
        if self.use_ensemble:
            self._initialize_ensemble_system()

    def load_model(self):
        """Load the trained model from the specified path."""
        self.model = self.model_trainer.load_model()
        if self.model is None:
            logger.info("No pre-trained model found. Training initial model with feature engineering...")
            # Try to train a model automatically if we have data
            try:
                if self.historical_data is not None and not self.historical_data.empty:
                    logger.info("🔧 Engineering features for model training...")

                    # Engineer features using FeatureEngineer
                    engineered_features = self.feature_engineer.engineer_features(use_temporal_analysis=True)

                    if engineered_features.empty:
                        logger.error("Feature engineering failed - cannot train model")
                        return

                    # Combine original data with engineered features
                    combined_data = self.historical_data.merge(
                        engineered_features,
                        left_index=True,
                        right_index=True,
                        how='inner'
                    )

                    logger.info(f"🤖 Training model with {combined_data.shape[1]} features...")

                    # Create new ModelTrainer with data for training
                    trainer = ModelTrainer(combined_data)
                    trainer.model_path = self.config["paths"]["model_file"]

                    if trainer.train_model():
                        # Update the current trainer and load the new model
                        self.model_trainer = trainer
                        self.model = self.model_trainer.load_model()
                        logger.info("✅ Initial model trained and loaded successfully.")
                    else:
                        logger.warning("❌ Initial model training failed.")
                else:
                    logger.warning("No historical data available for initial model training.")
            except Exception as e:
                logger.error(f"Error during automatic model training: {e}")
                logger.exception("Full traceback:")
                logger.warning("Model will be trained on first prediction request.")
        else:
            logger.info("✅ Model loaded successfully.")

    def _initialize_ensemble_system(self) -> None:
        """Initialize the enhanced ensemble prediction system"""
        try:
            # Load historical data for ensemble
            historical_data = self.data_loader.load_historical_data()

            # Initialize enhanced ensemble predictor
            # Assuming EnsemblePredictor expects models_dir to load other models
            self.ensemble_predictor = EnsemblePredictor(historical_data, models_dir="models/")

            logger.info("Enhanced ensemble prediction system initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize ensemble system: {e}")
            self.use_ensemble = False # Disable ensemble if initialization fails

    def predict_probabilities(self, use_ensemble: bool = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate probability predictions for white balls and Powerball.
        Prioritizes ensemble prediction if enabled and available.

        Args:
            use_ensemble: Boolean to explicitly use or ignore ensemble prediction.

        Returns:
            Tuple of (white_ball_probabilities, powerball_probabilities) or fallback values.
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

                    logger.info(f"Ensemble prediction completed using {ensemble_result.get('total_models', 0)} models")
                    logger.info(f"Ensemble method: {ensemble_result.get('ensemble_method', 'unknown')}")

                    return wb_probs, pb_probs

            except Exception as e:
                logger.error(f"Ensemble prediction failed, falling back to single model: {e}")

        # Fallback to single model prediction if ensemble is not used or failed
        if self.model is None:
            logger.error("Model not loaded. Cannot generate predictions. Returning uniform probabilities.")
            # Return uniform probabilities as fallback
            return np.ones(69) / 69, np.ones(26) / 26

        try:
            logger.info("Using single model prediction")

            # Use already loaded historical data or reload if empty
            if self.historical_data.empty:
                logger.debug("Reloading historical data for single model prediction.")
                self.historical_data = self.data_loader.load_historical_data()
                self.feature_engineer = FeatureEngineer(self.historical_data) # Re-initialize feature engineer

            # Generate features with proper validation for prediction
            features = self.feature_engineer.engineer_features(use_temporal_analysis=True)

            # Validate and prepare features for model
            prepared_features = self._prepare_features_for_model(features)
            if prepared_features is None:
                logger.error("Feature preparation failed for single model prediction.")
                return np.ones(69) / 69, np.ones(26) / 26

            # Get predictions from the model
            predictions = self.model_trainer.predict_probabilities(prepared_features) # Use model_trainer's predict method

            if predictions is None:
                logger.error("Model prediction returned None.")
                return np.ones(69) / 69, np.ones(26) / 26

            # Extract and normalize probabilities
            wb_probs, pb_probs = self._extract_and_normalize_probabilities(predictions)

            return wb_probs, pb_probs

        except Exception as e:
            logger.error(f"Error generating single model predictions: {e}")
            # Return uniform probabilities as fallback
            return np.ones(69) / 69, np.ones(26) / 26

    def _prepare_features_for_model(self, features_df: pd.DataFrame) -> Optional[np.ndarray]:
        """Prepares the latest features for model prediction with robust validation."""
        if features_df.empty:
            logger.error("Input features DataFrame is empty.")
            return None

        try:
            # Standard SHIOL+ features expected by the model, in order
            shiol_standard_features = [
                "even_count", "odd_count", "sum", "spread", "consecutive_count",
                "avg_delay", "max_delay", "min_delay",
                "dist_to_recent", "avg_dist_to_top_n", "dist_to_centroid",
                "time_weight", "increasing_trend_count", "decreasing_trend_count",
                "stable_trend_count"
            ]

            logger.debug(f"Preparing features for prediction. Expected: {shiol_standard_features}")
            logger.debug(f"Available features: {list(features_df.columns)}")

            # Use the latest row of features
            latest_row = features_df.iloc[-1]

            # Create feature array, filling with defaults if features are missing or invalid
            feature_values = np.zeros(len(shiol_standard_features))

            for i, feature_name in enumerate(shiol_standard_features):
                if feature_name in latest_row.index:
                    value = latest_row[feature_name]
                    if pd.notna(value) and np.isfinite(value):
                        feature_values[i] = float(value)
                    else:
                        # Use reasonable defaults for missing/invalid features
                        feature_values[i] = self._get_default_feature_value(feature_name)
                        logger.debug(f"Feature '{feature_name}' is invalid/missing, using default: {feature_values[i]}")
                else:
                    # Provide defaults for features entirely missing from the input
                    feature_values[i] = self._get_default_feature_value(feature_name)
                    logger.debug(f"Feature '{feature_name}' not found, using default: {feature_values[i]}")

            # Reshape to (1, num_features) for model input
            prepared_features = feature_values.reshape(1, -1)

            # Final check for non-finite values after processing
            if np.any(~np.isfinite(prepared_features)):
                logger.warning("Non-finite values detected in prepared features, cleaning...")
                prepared_features = np.nan_to_num(prepared_features, nan=0.0, posinf=1e6, neginf=-1e6) # Use large numbers for infinities

            logger.info(f"Successfully prepared features with shape: {prepared_features.shape}")
            logger.debug(f"Prepared feature values: {prepared_features.flatten()}")

            return prepared_features

        except Exception as e:
            logger.error(f"Error preparing features for model: {e}")
            return None

    def _get_default_feature_value(self, feature_name: str) -> float:
        """Provides default values for missing or invalid features."""
        defaults = {
            "even_count": 2.5, "odd_count": 2.5, "sum": 150.0, "spread": 40.0,
            "consecutive_count": 1.0, "avg_delay": 5.0, "max_delay": 10.0, "min_delay": 1.0,
            "dist_to_recent": 0.5, "avg_dist_to_top_n": 0.5, "dist_to_centroid": 0.5,
            "time_weight": 1.0, "increasing_trend_count": 1.0, "decreasing_trend_count": 1.0,
            "stable_trend_count": 1.0
        }
        return defaults.get(feature_name, 0.0) # Default to 0.0 for any unknown features

    def _extract_and_normalize_probabilities(self, predictions_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extracts and normalizes white ball and powerball probabilities from the prediction DataFrame.
        Ensures probabilities sum to 1 and are within valid range [0, 1].
        """
        try:
            wb_probs = np.zeros(69)
            pb_probs = np.zeros(26)
            epsilon = 1e-12 # Small value to prevent division by zero and zero probabilities

            # Ensure prediction DataFrame has expected columns
            wb_cols_present = [col for col in predictions_df.columns if col.startswith('wb_')]
            pb_cols_present = [col for col in predictions_df.columns if col.startswith('pb_')]

            if not wb_cols_present and not pb_cols_present:
                logger.error("Prediction DataFrame contains no recognized white ball or powerball probability columns.")
                return np.ones(69) / 69, np.ones(26) / 26

            # Extract white ball probabilities
            if wb_cols_present:
                wb_probs_raw = predictions_df[wb_cols_present].iloc[0].values # Assuming single prediction row
                wb_probs_raw = np.clip(wb_probs_raw, epsilon, 1.0 - epsilon) # Clip to avoid invalid values
                wb_sum = wb_probs_raw.sum()
                if wb_sum > epsilon and np.isfinite(wb_sum):
                    wb_probs = wb_probs_raw / wb_sum
                    # Ensure final sum is exactly 1.0
                    wb_probs = wb_probs / wb_probs.sum()
                else:
                    logger.warning(f"Invalid sum for white ball probabilities ({wb_sum}). Using uniform distribution.")
                    wb_probs = np.ones(69) / 69
            else:
                logger.warning("No white ball probability columns found in predictions. Using uniform distribution.")
                wb_probs = np.ones(69) / 69

            # Extract powerball probabilities
            if pb_cols_present:
                pb_probs_raw = predictions_df[pb_cols_present].iloc[0].values # Assuming single prediction row
                pb_probs_raw = np.clip(pb_probs_raw, epsilon, 1.0 - epsilon) # Clip to avoid invalid values
                pb_sum = pb_probs_raw.sum()
                if pb_sum > epsilon and np.isfinite(pb_sum):
                    pb_probs = pb_probs_raw / pb_sum
                    # Ensure final sum is exactly 1.0
                    pb_probs = pb_probs / pb_probs.sum()
                else:
                    logger.warning(f"Invalid sum for powerball probabilities ({pb_sum}). Using uniform distribution.")
                    pb_probs = np.ones(26) / 26
            else:
                logger.warning("No powerball probability columns found in predictions. Using uniform distribution.")
                pb_probs = np.ones(26) / 26

            # Final validation checks
            if not np.isclose(wb_probs.sum(), 1.0, atol=1e-8):
                logger.error(f"Final white ball probabilities do not sum to 1 ({wb_probs.sum()}). Re-normalizing.")
                wb_probs = wb_probs / wb_probs.sum()
            if not np.isclose(pb_probs.sum(), 1.0, atol=1e-8):
                logger.error(f"Final powerball probabilities do not sum to 1 ({pb_probs.sum()}). Re-normalizing.")
                pb_probs = pb_probs / pb_probs.sum()

            # Ensure all probabilities are within [0, 1]
            wb_probs = np.clip(wb_probs, 0, 1)
            pb_probs = np.clip(pb_probs, 0, 1)

            return wb_probs, pb_probs

        except Exception as e:
            logger.error(f"Error extracting and normalizing probabilities: {e}")
            # Return uniform probabilities as a safe fallback
            return np.ones(69) / 69, np.ones(26) / 26

    def get_prediction_features(self):
        """
        Prepares the feature set for the next draw to be predicted.
        Returns the features for the latest draw.
        """
        logger.info("Preparing features for the next prediction...")

        # Ensure historical data is loaded
        if self.historical_data.empty:
            logger.debug("Reloading historical data for feature preparation.")
            self.historical_data = self.data_loader.load_historical_data()
            # Re-initialize feature engineer if data was reloaded
            if not self.historical_data.empty:
                self.feature_engineer = FeatureEngineer(self.historical_data)
            else:
                raise ValueError("Historical data is empty. Cannot prepare features for prediction.")

        # Engineer features using the latest data
        all_features = self.feature_engineer.engineer_features(use_temporal_analysis=True)

        # Return the latest row of features
        latest_features = all_features.iloc[-1:]

        logger.info("Successfully prepared features for one prediction row.")
        return latest_features

    def predict_deterministic(self, save_to_log: bool = True) -> Dict:
        """
        Generates a deterministic prediction using the multi-criteria scoring system.

        Args:
            save_to_log: If True, saves the prediction to the database log.

        Returns:
            Dict containing the deterministic prediction and its details.
        """
        logger.info("Generating deterministic prediction with multi-criteria scoring...")

        # Get probability predictions from the model
        wb_probs, pb_probs = self.predict_probabilities()

        # Reload historical data if empty
        if self.historical_data.empty:
            logger.debug("Reloading historical data for deterministic prediction.")
            self.historical_data = self.data_loader.load_historical_data()
            if self.historical_data.empty:
                raise ValueError("Historical data is empty. Cannot generate deterministic prediction.")
            # Re-initialize deterministic generator if data was reloaded
            self.deterministic_generator = DeterministicGenerator(self.historical_data)

        # Use the deterministic generator to create the top prediction
        prediction = self.deterministic_generator.generate_top_prediction(wb_probs, pb_probs)

        # Save to log if requested
        if save_to_log:
            # Ensure prediction dictionary has the expected keys for logging
            prediction_data_for_log = {
                'timestamp': prediction.get('timestamp', datetime.now().isoformat()),
                'numbers': prediction.get('numbers'),
                'powerball': prediction.get('powerball'),
                'score_total': prediction.get('score_total'),
                'model_version': prediction.get('model_version', 'v6.0'), # Assuming v6.0 for new model
                'dataset_hash': prediction.get('dataset_hash') # Include hash if available
            }
            prediction_id = save_prediction_log(prediction_data_for_log)
            if prediction_id:
                prediction['log_id'] = prediction_id
                logger.info(f"Deterministic prediction saved to log with ID: {prediction_id}")
            else:
                logger.warning("Failed to save prediction to log")

        logger.info("Deterministic prediction generated successfully")
        return prediction

    def predict_diverse_plays(self, num_plays: int = 5, save_to_log: bool = False, target_draw_date: str = None) -> List[Dict[str, Any]]:
        """
        Generates multiple diverse, high-quality predictions for the next draw.

        Args:
            num_plays: Number of diverse plays to generate (default: 5).
            save_to_log: If True, saves the predictions to the database log.
            target_draw_date: The target draw date for the prediction (optional).

        Returns:
            List of Dicts, each representing a diverse prediction with its details.
        """
        logger.info(f"Generating {num_plays} diverse high-quality plays for next drawing...")

        # Get probability predictions
        wb_probs, pb_probs = self.predict_probabilities()

        # Ensure historical data is loaded
        if self.historical_data.empty:
            logger.debug("Reloading historical data for diverse plays generation.")
            self.historical_data = self.data_loader.load_historical_data()
            if self.historical_data.empty:
                raise ValueError("Historical data is empty. Cannot generate diverse predictions.")
            # Re-initialize deterministic generator if data was reloaded
            self.deterministic_generator = DeterministicGenerator(self.historical_data)

        # Generate diverse predictions using the deterministic generator
        diverse_predictions = self.deterministic_generator.generate_diverse_predictions(
            wb_probs, pb_probs, num_plays=num_plays
        )

        # Save predictions to log if requested
        if save_to_log:
            saved_count = 0
            for prediction in diverse_predictions:
                # Prepare data for logging
                prediction_data = {
                    'timestamp': prediction.get('timestamp', datetime.now().isoformat()),
                    'numbers': prediction.get('numbers'),
                    'powerball': prediction.get('powerball'),
                    'score_total': prediction.get('score_total'),
                    'model_version': prediction.get('model_version', 'v6.0'),
                    'dataset_hash': prediction.get('dataset_hash'),
                    'target_draw_date': target_draw_date # Include target date if provided
                }
                prediction_id = save_prediction_log(prediction_data)
                if prediction_id:
                    prediction['log_id'] = prediction_id
                    saved_count += 1
                else:
                    logger.warning("Failed to save one or more diverse predictions to log")

            logger.info(f"Saved {saved_count}/{len(diverse_predictions)} diverse predictions to log")

        logger.info(f"Generated {len(diverse_predictions)} diverse plays successfully")
        return diverse_predictions

    def predict_syndicate_plays(self, num_plays: int = 100, save_to_log: bool = True) -> List[Dict]:
        """
        Generates a large number of predictions optimized for syndicate play,
        considering diversity and scoring.

        Args:
            num_plays: Number of syndicate plays to generate (default: 100).
            save_to_log: If True, saves the predictions to the database log.

        Returns:
            List of predictions optimized for syndicate play.
        """
        logger.info(f"Generating {num_plays} syndicate plays with advanced scoring...")

        # Get probability predictions
        wb_probs, pb_probs = self.predict_probabilities()

        # Ensure historical data is loaded
        if self.historical_data.empty:
            logger.debug("Reloading historical data for syndicate plays generation.")
            self.historical_data = self.data_loader.load_historical_data()
            if self.historical_data.empty:
                raise ValueError("Historical data is empty. Cannot generate syndicate predictions.")
            # Re-initialize deterministic generator if data was reloaded
            self.deterministic_generator = DeterministicGenerator(self.historical_data)

        # Generate a larger pool of candidates for better selection
        candidate_pool_size = max(5000, num_plays * 5) # Ensure a substantial pool

        # Generate candidates with diverse scoring
        syndicate_predictions = self.deterministic_generator.generate_diverse_predictions(
            wb_probs, pb_probs,
            num_plays=num_plays,
            num_candidates=candidate_pool_size
        )

        # Apply syndicate-specific analysis and scoring
        for i, prediction in enumerate(syndicate_predictions):
            prediction['syndicate_rank'] = i + 1 # Rank based on generation order (implies quality)
            prediction['syndicate_tier'] = self._classify_syndicate_tier(prediction.get('score_total', 0))
            prediction['expected_coverage'] = self._calculate_coverage_score(prediction, syndicate_predictions) # Calculate diversity metric

        # Save predictions to log if requested
        if save_to_log:
            saved_count = 0
            for prediction in syndicate_predictions:
                prediction_data = {
                    'timestamp': prediction.get('timestamp', datetime.now().isoformat()),
                    'numbers': prediction.get('numbers'),
                    'powerball': prediction.get('powerball'),
                    'score_total': prediction.get('score_total'),
                    'model_version': prediction.get('model_version', 'v6.0'),
                    'dataset_hash': prediction.get('dataset_hash'),
                    'syndicate_rank': prediction.get('syndicate_rank'),
                    'syndicate_tier': prediction.get('syndicate_tier'),
                    'expected_coverage': prediction.get('expected_coverage')
                }
                prediction_id = save_prediction_log(prediction_data)
                if prediction_id:
                    prediction['log_id'] = prediction_id
                    saved_count += 1
                else:
                    logger.warning("Failed to save one or more syndicate predictions to log")

            logger.info(f"Saved {saved_count}/{len(syndicate_predictions)} syndicate predictions to log")

        logger.info(f"Generated {len(syndicate_predictions)} syndicate plays successfully")
        return syndicate_predictions

    def _classify_syndicate_tier(self, score: float) -> str:
        """Classifies plays into tiers based on their total score."""
        if score >= 0.8: return "Premium"
        elif score >= 0.6: return "High"
        elif score >= 0.4: return "Medium"
        else: return "Standard"

    def _calculate_coverage_score(self, prediction: Dict, all_predictions: List[Dict]) -> float:
        """Calculates how well a prediction complements others in the syndicate for diversity."""
        candidate_nums = set(prediction.get('numbers', []) + [prediction.get('powerball', 1)])

        coverage_scores = []
        # Compare against a subset of top predictions to estimate diversity impact
        comparison_candidates = all_predictions[:min(len(all_predictions), 100)] # Compare with top 100

        for other_pred in comparison_candidates:
            if other_pred == prediction: continue # Skip self-comparison

            other_nums = set(other_pred.get('numbers', []) + [other_pred.get('powerball', 1)])

            # Jaccard similarity calculation
            intersection = len(candidate_nums.intersection(other_nums))
            union = len(candidate_nums.union(other_nums))

            if union == 0: jaccard_similarity = 1.0 if len(candidate_nums) == 0 else 0.0
            else: jaccard_similarity = intersection / union

            # Diversity is 1 - similarity
            diversity_score = 1.0 - jaccard_similarity
            coverage_scores.append(diversity_score)

        # Return average diversity score, or a neutral value if no comparisons made
        return np.mean(coverage_scores) if coverage_scores else 0.5

    def predict_ensemble_syndicate(self, num_plays: int = 100) -> List[Dict]:
        """
        Generates ensemble syndicate predictions by combining results from multiple models
        (deterministic, adaptive, intelligent) and ranking them by an ensemble score.

        Args:
            num_plays: The final number of plays to generate.

        Returns:
            A list of the top 'num_plays' predictions selected from the ensemble.
        """
        logger.info("Generating ensemble syndicate predictions with multiple models for {num_plays} plays...")

        # Get base probability predictions
        wb_probs, pb_probs = self.predict_probabilities()

        # Load historical data if not already loaded
        if self.historical_data.empty:
            logger.debug("Reloading historical data for ensemble syndicate generation.")
            self.historical_data = self.data_loader.load_historical_data()
            if self.historical_data.empty:
                raise ValueError("Historical data is empty. Cannot generate ensemble syndicate predictions.")

        all_candidates = []

        # --- Model Contributions ---

        # 1. Deterministic Generator (e.g., 60% of pool weight)
        try:
            deterministic_gen = DeterministicGenerator(self.historical_data)
            num_deterministic = int(num_plays * 0.6)
            # Generate more candidates than needed to ensure diversity
            deterministic_candidates = deterministic_gen.generate_diverse_predictions(
                wb_probs, pb_probs,
                num_plays=num_deterministic,
                num_candidates=num_deterministic * 3
            )
            for candidate in deterministic_candidates:
                candidate['model_source'] = 'deterministic'
                candidate['ensemble_weight'] = 0.6
                # Use the base score as initial score
                candidate['ensemble_score'] = candidate.get('score_total', 0) * 0.6
            all_candidates.extend(deterministic_candidates)
            logger.info(f"Added {len(deterministic_candidates)} candidates from deterministic generator.")
        except Exception as e:
            logger.warning(f"Failed to add deterministic candidates: {e}")

        # 2. Adaptive Play Scorer (e.g., 25% of pool weight) - requires specific implementation
        try:
            from src.adaptive_feedback import AdaptivePlayScorer # Assuming this exists
            adaptive_scorer = AdaptivePlayScorer(self.historical_data)
            num_adaptive = int(num_plays * 0.25)

            # Use deterministic generator for base plays, then re-score adaptively
            deterministic_gen_for_adaptive = DeterministicGenerator(self.historical_data)
            adaptive_candidates_base = deterministic_gen_for_adaptive.generate_diverse_predictions(
                wb_probs, pb_probs,
                num_plays=num_adaptive,
                num_candidates=num_adaptive * 3
            )

            adaptive_candidates = []
            for candidate in adaptive_candidates_base:
                adaptive_scores = adaptive_scorer.calculate_total_score(
                    candidate.get('numbers', []), candidate.get('powerball', 1),
                    wb_probs, pb_probs
                )
                candidate['score_total'] = adaptive_scores['total'] # Update score
                candidate['score_details'] = adaptive_scores
                candidate['model_source'] = 'adaptive'
                candidate['ensemble_weight'] = 0.25
                candidate['ensemble_score'] = candidate.get('score_total', 0) * 0.25
                adaptive_candidates.append(candidate)

            all_candidates.extend(adaptive_candidates)
            logger.info(f"Added {len(adaptive_candidates)} candidates from adaptive scoring.")

        except ImportError:
            logger.warning("AdaptivePlayScorer not found. Skipping adaptive model contribution.")
        except Exception as e:
            logger.warning(f"Error incorporating adaptive model: {e}")

        # 3. Intelligent Generator (e.g., 15% of pool weight) - requires specific implementation
        try:
            from src.intelligent_generator import IntelligentGenerator # Assuming this exists
            intelligent_gen = IntelligentGenerator(self.historical_data)
            num_intelligent = int(num_plays * 0.15)

            # Generate plays using the intelligent generator
            intelligent_candidates_raw = intelligent_gen.generate_plays(
                 wb_probs, pb_probs, num_plays=num_intelligent, num_candidates=num_intelligent*3
            )

            # Need to convert raw output to a consistent format and score them
            intelligent_candidates = []
            # Assuming IntelligentGenerator output needs processing and scoring
            # This part heavily depends on the actual implementation of IntelligentGenerator
            # This section is highly speculative without the actual IntelligentGenerator code
            for i, raw_play in enumerate(intelligent_candidates_raw):
                # Placeholder: Assuming raw_play is a dict or can be converted
                # Need to ensure numbers and powerball are correctly extracted and scored
                try:
                    numbers = raw_play.get('numbers', raw_play.get('n1', [])[:5]) # Adapt based on actual output
                    powerball = raw_play.get('powerball', raw_play.get('pb', 1))

                    # Use deterministic scorer to get a comparable score
                    deterministic_gen_for_scoring = DeterministicGenerator(self.historical_data)
                    scores = deterministic_gen_for_scoring.scorer.calculate_total_score(
                        numbers, powerball, wb_probs, pb_probs
                    )

                    candidate = {
                        'numbers': numbers, 'powerball': powerball,
                        'score_total': scores['total'], 'score_details': scores,
                        'model_source': 'intelligent', 'ensemble_weight': 0.15,
                        'ensemble_score': scores['total'] * 0.15,
                        'timestamp': datetime.now().isoformat()
                    }
                    intelligent_candidates.append(candidate)
                except Exception as parse_err:
                    logger.warning(f"Failed to process intelligent candidate {i}: {parse_err}")

            all_candidates.extend(intelligent_candidates)
            logger.info(f"Added {len(intelligent_candidates)} candidates from intelligent generator.")

        except ImportError:
            logger.warning("IntelligentGenerator not found. Skipping intelligent generator contribution.")
        except Exception as e:
            logger.warning(f"Error incorporating intelligent generator: {e}")

        # --- Ensemble Scoring and Ranking ---
        final_ensemble_scores = []
        for candidate in all_candidates:
            # Calculate diversity bonus
            diversity_bonus = self._calculate_ensemble_diversity(candidate, all_candidates)

            # Combine weighted score and diversity bonus
            # Adjust contribution of diversity bonus (e.g., weight it less)
            ensemble_score = candidate.get('ensemble_score', 0) + (diversity_bonus * 0.1) # Example: diversity adds up to 0.1
            candidate['ensemble_score'] = ensemble_score
            final_ensemble_scores.append(candidate)

        # Sort all candidates by their final ensemble score
        final_ensemble_scores.sort(key=lambda x: x.get('ensemble_score', 0), reverse=True)

        # Select the top 'num_plays'
        final_predictions = final_ensemble_scores[:num_plays]

        # Add ensemble metadata to the final predictions
        for i, prediction in enumerate(final_predictions):
            prediction['ensemble_rank'] = i + 1
            prediction['method'] = 'ensemble_syndicate' # Clearly mark the method

        logger.info(f"Generated and ranked {len(final_predictions)} ensemble syndicate predictions.")
        return final_predictions

    def _calculate_ensemble_diversity(self, candidate: Dict, all_candidates: List[Dict]) -> float:
        """
        Calculates a diversity score for a candidate by comparing it against other candidates
        in the ensemble, aiming to reward unique combinations.
        """
        candidate_nums = set(candidate.get('numbers', []) + [candidate.get('powerball', 1)])

        diversity_scores = []
        # Compare against a limited set of other candidates for efficiency
        comparison_set_size = min(len(all_candidates), 100)
        comparison_candidates = all_candidates[:comparison_set_size]

        for other in comparison_candidates:
            if other == candidate: continue # Skip self-comparison

            other_nums = set(other.get('numbers', []) + [other.get('powerball', 1)])

            # Jaccard index for set similarity
            intersection = len(candidate_nums.intersection(other_nums))
            union = len(candidate_nums.union(other_nums))

            if union == 0: jaccard_similarity = 1.0 if len(candidate_nums) == 0 else 0.0
            else: jaccard_similarity = intersection / union

            # Diversity is 1 minus similarity
            diversity_score = 1.0 - jaccard_similarity
            diversity_scores.append(diversity_score)

        # Return average diversity score, or a default value if no comparisons were made
        return np.mean(diversity_scores) if diversity_scores else 0.5

    def compare_prediction_methods(self) -> Dict:
        """
        Compares the traditional (intelligent generator) vs. deterministic prediction methods,
        providing insights into their outputs and characteristics.

        Returns:
            A dictionary containing the comparison results.
        """
        logger.info("Comparing traditional vs. deterministic prediction methods...")

        # Get base probability predictions required for both methods
        try:
            wb_probs, pb_probs = self.predict_probabilities()
        except Exception as e:
            logger.error(f"Could not obtain probability predictions for method comparison: {e}")
            return {"error": f"Failed to get probabilities: {str(e)}"}

        comparison_results = {}

        # --- Traditional Method (Intelligent Generator) ---
        try:
            # Assuming IntelligentGenerator is available and can be used for comparison
            # This part requires the actual implementation of IntelligentGenerator
            # and understanding its output format for comparison.
            from src.intelligent_generator import IntelligentGenerator
            traditional_generator = IntelligentGenerator() # Assuming default initialization

            # Generate a single play for comparison
            # The exact parameters might need adjustment based on IntelligentGenerator's API
            traditional_play_list = traditional_generator.generate_plays(
                 wb_probs, pb_probs, num_plays=1
            )

            if traditional_play_list and isinstance(traditional_play_list, list) and len(traditional_play_list) > 0:
                # Process the output to fit the comparison structure
                # Assuming the output is a list of dicts or similar
                first_traditional_play = traditional_play_list[0]

                comparison_results['traditional_method'] = {
                    'numbers': first_traditional_play.get('numbers', []),
                    'powerball': first_traditional_play.get('powerball', 1),
                    'method': 'intelligent_generator', # Identify the method used
                    'reproducible': False, # Typically random or heuristic methods are not perfectly reproducible
                    'score': first_traditional_play.get('score_total'), # Include score if available
                    'dataset_hash': first_traditional_play.get('dataset_hash') # Include hash if available
                }
            else:
                comparison_results['traditional_method'] = {"error": "IntelligentGenerator returned no valid play for comparison."}
                logger.warning("IntelligentGenerator returned no plays for comparison.")

        except ImportError:
            comparison_results['traditional_method'] = {"error": "IntelligentGenerator module not found."}
            logger.warning("IntelligentGenerator module not available for comparison.")
        except Exception as e:
            comparison_results['traditional_method'] = {"error": f"Error processing IntelligentGenerator: {str(e)}"}
            logger.error(f"An error occurred during traditional method comparison: {str(e)}")

        # --- Deterministic Method ---
        try:
            # Ensure historical data is loaded for deterministic generation
            if self.historical_data.empty:
                logger.debug("Reloading historical data for deterministic method comparison.")
                self.historical_data = self.data_loader.load_historical_data()
                if self.historical_data.empty:
                    raise ValueError("Historical data is empty for deterministic comparison.")
                # Re-initialize deterministic generator if data was reloaded
                self.deterministic_generator = DeterministicGenerator(self.historical_data)

            deterministic_prediction = self.deterministic_generator.generate_top_prediction(wb_probs, pb_probs)

            comparison_results['deterministic_method'] = {
                'numbers': deterministic_prediction.get('numbers', []),
                'powerball': deterministic_prediction.get('powerball', 1),
                'total_score': deterministic_prediction.get('score_total'),
                'score_details': deterministic_prediction.get('score_details'),
                'method': 'multi_criteria_scoring',
                'reproducible': True, # Deterministic methods should be reproducible
                'dataset_hash': deterministic_prediction.get('dataset_hash')
            }
        except Exception as e:
            comparison_results['deterministic_method'] = {"error": f"Error processing DeterministicGenerator: {str(e)}"}
            logger.error(f"An error occurred during deterministic method comparison: {str(e)}")

        comparison_results['comparison_timestamp'] = datetime.now().isoformat()
        logger.info("Method comparison completed.")
        return comparison_results

    def get_model_info(self):
        """Get information about the current model"""
        if self.model:
            return {
                'loaded': True,
                'version': getattr(self.model_trainer, 'model_version', 'Unknown'), # Get version from trainer
                'features': self.model_trainer.feature_names if hasattr(self.model_trainer, 'feature_names') else [],
                'last_trained': getattr(self.model_trainer, 'last_trained', 'Unknown')
            }
        return {'loaded': False}

    def retrain(self):
        """
        Retrains the model using the latest available historical data.
        This method is part of the v6.0 dashboard functionality.

        Returns:
            True if retraining was successful, False otherwise.
        """
        try:
            logger.info("Starting model retraining process for SHIOL+ v6.0...")

            # Load fresh historical data
            logger.info("Loading latest historical data for retraining...")
            self.data_loader = DataLoader() # Ensure we use the latest loader
            data = self.data_loader.load_historical_data()

            if data is None or data.empty:
                logger.error("No historical data available for retraining. Please check data source.")
                return False

            logger.info(f"Successfully loaded {len(data)} records for retraining.")

            # Initialize ModelTrainer with the loaded data
            # The path for saving the new model needs to be configured or derived
            model_save_path = self.config.get("paths", "model_file", fallback="models/shiolplus_v6.pkl")
            trainer = ModelTrainer(model_save_path) # Pass path for saving the new model
            trainer.data = data # Set the data for training

            # Train the new model
            logger.info("Training the new model...")
            success = trainer.train_model()

            if success:
                logger.info("Model training completed successfully.")
                # Update the Predictor's model trainer to use the newly trained model
                self.model_trainer = trainer # Replace the trainer instance
                self.model = self.model_trainer.load_model()
                logger.info("Model retraining process finished successfully. Predictor updated.")
                return True
            else:
                logger.error("Model training process failed.")
                return False

        except Exception as e:
            logger.error(f"An error occurred during the model retraining process: {str(e)}")
            logger.exception("Retraining failed due to an unexpected error.") # Log traceback
            return False

    # --- Ensemble System Methods ---
    def get_ensemble_summary(self) -> Dict[str, Any]:
        """Retrieves a summary of the current ensemble system status."""
        if not self.ensemble_predictor:
            return {
                'ensemble_enabled': False,
                'reason': 'Ensemble system not initialized or disabled.'
            }

        try:
            summary = self.ensemble_predictor.get_ensemble_summary()
            summary['ensemble_enabled'] = True # Explicitly state it's enabled
            return summary
        except Exception as e:
            # Return error state if summary retrieval fails
            return {
                'ensemble_enabled': False,
                'reason': f'Error retrieving ensemble summary: {e}'
            }

    def update_ensemble_performance(self, performance_feedback: Dict[str, float]) -> None:
        """
        Updates the weights of models within the ensemble based on provided performance feedback.

        Args:
            performance_feedback: A dictionary mapping model identifiers to their performance metrics.
        """
        if self.ensemble_predictor:
            try:
                self.ensemble_predictor.update_model_weights(performance_feedback)
                logger.info("Ensemble model weights updated based on performance feedback.")
            except Exception as e:
                logger.error(f"Failed to update ensemble performance: {e}")
        else:
            logger.warning("Ensemble system is not active. Cannot update performance.")

    def set_ensemble_method(self, method: str) -> bool:
        """
        Sets the ensemble prediction method (e.g., 'average', 'weighted_average', 'stacking').

        Args:
            method: The name of the ensemble method to use.

        Returns:
            True if the method was set successfully, False otherwise.
        """
        if not self.ensemble_predictor:
            logger.warning("Ensemble system is not active. Cannot set ensemble method.")
            return False

        try:
            success = self.ensemble_predictor.set_ensemble_method(method)
            if success:
                logger.info(f"Ensemble prediction method set to: '{method}'")
            else:
                logger.warning(f"Failed to set ensemble method to '{method}'. Method may not be supported.")
            return success
        except Exception as e:
            logger.error(f"Error setting ensemble method '{method}': {e}")
            return False
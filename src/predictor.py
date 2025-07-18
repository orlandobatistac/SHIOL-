import configparser
import joblib
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from xgboost import XGBClassifier

from src.loader import get_data_loader
from src.intelligent_generator import FeatureEngineer

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
            use_label_encoder=False,
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
    def __init__(self):
        logger.info("Initializing Predictor...")
        self.model_trainer = get_model_trainer()
        self.model = self.model_trainer.load_model()
        if self.model is None:
            raise RuntimeError("Failed to load the model. Please train a model first using 'train' command.")

    def get_prediction_features(self):
        """
        Prepares the feature set for the next draw to be predicted.
        """
        logger.info("Preparing features for the next prediction...")
        data_loader = get_data_loader()
        historical_data = data_loader.load_historical_data()
        
        if historical_data.empty:
            raise ValueError("Historical data is empty. Cannot prepare features for prediction.")

        feature_engineer = FeatureEngineer(historical_data)
        all_features = feature_engineer.engineer_features(use_temporal_analysis=True)
        
        latest_features = all_features.iloc[-1:]
        
        logger.info("Successfully prepared features for one prediction row.")
        return latest_features

    def predict_probabilities(self):
        """
        Predicts the probability of each number appearing in the next draw.
        """
        logger.info("Predicting number probabilities for the next draw...")
        
        features_df = self.get_prediction_features()
        prob_df = self.model_trainer.predict_probabilities(features_df)
        
        if prob_df is None:
            raise RuntimeError("Prediction failed. The model did not return probabilities.")
            
        prob_series = prob_df.iloc[0]
        
        wb_probs = {int(col.split('_')[1]): prob for col, prob in prob_series.items() if col.startswith('wb_')}
        pb_probs = {int(col.split('_')[1]): prob for col, prob in prob_series.items() if col.startswith('pb_')}

        logger.info("Probabilities predicted successfully.")
        return wb_probs, pb_probs

def get_model_trainer():
    config = configparser.ConfigParser()
    config.read("config/config.ini")
    model_file = config["paths"]["model_file"]
    return ModelTrainer(model_file)

def retrain_existing_model(config_path="config/config.ini"):
    logger.info(f"Starting model retraining with config: {config_path}")
    try:
        trainer, _ = _setup_model_trainer(config_path)
        features = _prepare_training_data()
        if features is not None:
            trainer.train(features)
            logger.info(f"Successfully retrained and saved to {trainer.model_path}")
    except Exception as e:
        logger.error(f"Error during existing model retraining: {e}")
        raise

def _setup_model_trainer(config_path):
    config = configparser.ConfigParser()
    config.read(config_path)
    model_path = config["paths"]["model_file"]
    logger.info(f"Using model at: {model_path}")

    trainer = ModelTrainer(model_path)
    trainer.load_model()
    return trainer, trainer.model

def _prepare_training_data():
    data_loader = get_data_loader()
    historical_data = data_loader.load_historical_data()
    if historical_data.empty:
        logger.error("Failed to load historical data. Exiting.")
        return None
    logger.info(f"Successfully loaded {len(historical_data)} historical draws.")
    
    feature_engineer = FeatureEngineer(historical_data)
    features = feature_engineer.engineer_features()
    
    return features
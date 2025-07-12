import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib
from loguru import logger
import configparser

class ModelTrainer:
    def __init__(self, model_path):
        self.model_path = model_path
        self.config = configparser.ConfigParser()
        self.config.read('config/config.ini')
        self.model = None
        self.label_encoder = LabelEncoder()
        logger.info("ModelTrainer initialized.")

    def train(self, features_df):
        """
        Trains the XGBoost model on the feature-engineered data.
        """
        logger.info("Starting model training...")
        
        # Define features (X) and target (y)
        feature_cols = [
            'even_count', 'odd_count', 'sum', 'spread', 'consecutive_count',
            'avg_delay', 'max_delay', 'min_delay'
        ]
        X = features_df[feature_cols]
        
        # Use the new 'prize_tier' as the target
        y_categorical = features_df['prize_tier']
        y = self.label_encoder.fit_transform(y_categorical)
        logger.info(f"Target classes found: {self.label_encoder.classes_}")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=float(self.config['model_params']['test_size']),
            random_state=int(self.config['model_params']['random_state'])
        )

        self.model = XGBClassifier(
            n_estimators=int(self.config['model_params']['n_estimators']),
            random_state=int(self.config['model_params']['random_state']),
            use_label_encoder=False,
            eval_metric='logloss'
        )

        self.model.fit(X_train, y_train)
        logger.info("Model training complete.")

        # Evaluate model
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        logger.info(f"Model accuracy on test set: {accuracy:.2f}")

        self.save_model()
        return self.model

    def save_model(self):
        """
        Saves the trained model to the specified path.
        """
        if self.model:
            # Save both the model and the label encoder
            model_bundle = {'model': self.model, 'label_encoder': self.label_encoder}
            joblib.dump(model_bundle, self.model_path)
            logger.info(f"Model bundle saved to {self.model_path}")

    def load_model(self):
        """
        Loads a pre-trained model from the specified path.
        """
        try:
            model_bundle = joblib.load(self.model_path)
            self.model = model_bundle['model']
            self.label_encoder = model_bundle['label_encoder']
            logger.info(f"Model bundle loaded from {self.model_path}")
            return self.model
        except FileNotFoundError:
            logger.error(f"Model file not found at {self.model_path}")
            return None

def get_model_trainer():
    config = configparser.ConfigParser()
    config.read('config/config.ini')
    model_file = config['paths']['model_file']
    return ModelTrainer(model_file)
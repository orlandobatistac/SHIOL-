import pandas as pd
from sqlalchemy import create_engine
import configparser
from loguru import logger
import joblib
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

def train_model():
    """
    Loads data, trains a model, and saves it.
    """
    config = configparser.ConfigParser()
    config.read('config/config.ini')

    db_file = config['paths']['db_file']
    model_file = config['paths']['model_file']
    
    logger.info("Loading data from database for training...")
    engine = create_engine(f'sqlite:///{db_file}')
    try:
        df = pd.read_sql('historical_draws', engine)
    except Exception as e:
        logger.error(f"Could not read from database: {e}")
        return

    # This is a placeholder for actual model training logic.
    # A real implementation would require a target variable and more sophisticated feature selection.
    logger.info("Training model (placeholder)...")
    
    # Example: Predict if the sum of white balls will be high or low
    df['target'] = (df['sum_white'] > 175).astype(int)
    
    features = ['even_count', 'odd_count']
    target = 'target'

    X_train, X_test, y_train, y_test = train_test_split(
        df[features], df[target], 
        test_size=float(config['model_params']['test_size']), 
        random_state=int(config['model_params']['random_state'])
    )

    model = XGBClassifier(
        n_estimators=int(config['model_params']['n_estimators']),
        random_state=int(config['model_params']['random_state'])
    )
    model.fit(X_train, y_train)

    logger.info(f"Saving model to {model_file}")
    joblib.dump(model, model_file)
    logger.info("Model training complete.")

if __name__ == '__main__':
    logger.add("logs/shiolplus.log", rotation="10 MB")
    train_model()
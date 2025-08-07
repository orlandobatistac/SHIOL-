
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from predictor import ModelTrainer
from loader import DataLoader
from database import initialize_database
from loguru import logger

def initialize_model():
    """Initialize and train the model if it doesn't exist."""
    try:
        logger.info("Initializing SHIOL+ model...")
        
        # Initialize database
        initialize_database()
        
        # Load data
        data_loader = DataLoader()
        historical_data = data_loader.load_historical_data()
        
        if historical_data.empty:
            logger.error("No historical data available for model training")
            return False
        
        # Check if model exists
        model_path = "models/shiolplus.pkl"
        if os.path.exists(model_path):
            logger.info(f"Model already exists at {model_path}")
            return True
        
        # Create models directory
        os.makedirs("models", exist_ok=True)
        
        # Train model
        logger.info("Training initial model...")
        trainer = ModelTrainer(historical_data)
        trainer.data = historical_data
        
        if trainer.train_model():
            logger.info("Model trained successfully!")
            return True
        else:
            logger.error("Model training failed")
            return False
            
    except Exception as e:
        logger.error(f"Error during model initialization: {e}")
        return False

if __name__ == "__main__":
    success = initialize_model()
    if success:
        print("✅ Model initialization completed successfully")
    else:
        print("❌ Model initialization failed")
        sys.exit(1)

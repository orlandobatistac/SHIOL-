
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
        logger.info("🚀 Initializing SHIOL+ model with feature engineering...")
        
        # Initialize database
        initialize_database()
        
        # Load data
        data_loader = DataLoader()
        historical_data = data_loader.load_historical_data()
        
        if historical_data.empty:
            logger.error("No historical data available for model training")
            return False
        
        logger.info(f"Loaded {len(historical_data)} historical records")
        
        # Check if model exists
        model_path = "models/shiolplus.pkl"
        if os.path.exists(model_path):
            logger.info(f"Model already exists at {model_path}")
            return True
        
        # Create models directory
        os.makedirs("models", exist_ok=True)
        
        # CRITICAL: Engineer features BEFORE training
        logger.info("🔧 Engineering features for model training...")
        from intelligent_generator import FeatureEngineer
        
        feature_engineer = FeatureEngineer(historical_data)
        engineered_data = feature_engineer.engineer_features(use_temporal_analysis=True)
        
        if engineered_data.empty:
            logger.error("Feature engineering failed - no engineered features generated")
            return False
        
        logger.info(f"✅ Feature engineering complete: {engineered_data.shape[1]} features generated")
        
        # Combine original data with engineered features
        # Ensure we have the draw numbers and powerball for targets
        combined_data = historical_data.merge(engineered_data, left_index=True, right_index=True, how='inner')
        
        logger.info(f"Combined data shape: {combined_data.shape}")
        logger.info(f"Available columns: {list(combined_data.columns)}")
        
        # Train model with engineered features
        logger.info("🤖 Training model with engineered features...")
        trainer = ModelTrainer(model_path)  # Pass the path for saving
        trainer.data = combined_data
        
        if trainer.train_model():
            logger.info("✅ Model trained successfully!")
            return True
        else:
            logger.error("❌ Model training failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error during model initialization: {e}")
        logger.exception("Full traceback:")
        return False

if __name__ == "__main__":
    success = initialize_model()
    if success:
        print("✅ Model initialization completed successfully")
    else:
        print("❌ Model initialization failed")
        sys.exit(1)

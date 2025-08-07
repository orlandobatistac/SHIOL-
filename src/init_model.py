
#!/usr/bin/env python3

import os
import sys
import pandas as pd
from pathlib import Path

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.predictor import ModelTrainer
from src.loader import DataLoader
from src.database import initialize_database
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
        from src.intelligent_generator import FeatureEngineer
        
        feature_engineer = FeatureEngineer(historical_data)
        engineered_data = feature_engineer.engineer_features(use_temporal_analysis=True)
        
        if engineered_data.empty:
            logger.error("Feature engineering failed - no engineered features generated")
            return False
        
        logger.info(f"✅ Feature engineering complete: {engineered_data.shape[1]} features generated")
        
        # FIXED: Use concat instead of merge to avoid column renaming
        logger.info("🔗 Properly combining data to avoid column conflicts...")
        
        # Reset indices to ensure alignment
        historical_data_reset = historical_data.reset_index(drop=True)
        engineered_data_reset = engineered_data.reset_index(drop=True)
        
        # Use concat instead of merge to preserve original column names
        combined_data = pd.concat([historical_data_reset, engineered_data_reset], axis=1)
        
        logger.info(f"Combined data shape: {combined_data.shape}")
        logger.info(f"Available columns: {list(combined_data.columns)}")
        
        # Verify required columns exist
        required_cols = ['n1', 'n2', 'n3', 'n4', 'n5', 'pb']
        present_cols = [col for col in required_cols if col in combined_data.columns]
        missing_cols = [col for col in required_cols if col not in combined_data.columns]
        
        logger.info(f"Present required columns: {present_cols}")
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            return False
        
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

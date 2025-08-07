
#!/usr/bin/env python3
"""
Force Model Creation Utility for SHIOL+ v6.0
============================================

Utility to force creation of the model file when automatic training fails.
This addresses column naming conflicts and ensures proper model initialization.
"""

import os
import sys
import pandas as pd
from loguru import logger

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.predictor import ModelTrainer
from src.loader import DataLoader
from src.database import initialize_database
from src.intelligent_generator import FeatureEngineer

def force_create_model():
    """Force creation of the model with proper column handling."""
    try:
        logger.info("🚀 Force creating SHIOL+ model...")
        
        # Initialize database
        initialize_database()
        
        # Load data
        data_loader = DataLoader()
        historical_data = data_loader.load_historical_data()
        
        if historical_data.empty:
            logger.error("No historical data available for model training")
            return False
        
        logger.info(f"Loaded {len(historical_data)} historical records")
        
        # Create models directory
        os.makedirs("models", exist_ok=True)
        model_path = "models/shiolplus.pkl"
        
        # Engineer features
        logger.info("🔧 Engineering features for model training...")
        feature_engineer = FeatureEngineer(historical_data)
        engineered_data = feature_engineer.engineer_features(use_temporal_analysis=True)
        
        if engineered_data.empty:
            logger.error("Feature engineering failed")
            return False
        
        logger.info(f"✅ Feature engineering complete: {engineered_data.shape[1]} features")
        
        # CRITICAL: Properly combine data to avoid column conflicts
        logger.info("🔗 Combining original data with engineered features...")
        
        # Reset indices to ensure proper alignment
        historical_data_reset = historical_data.reset_index(drop=True)
        engineered_data_reset = engineered_data.reset_index(drop=True)
        
        # Combine using concat instead of merge to avoid column renaming
        combined_data = pd.concat([historical_data_reset, engineered_data_reset], axis=1)
        
        logger.info(f"Combined data shape: {combined_data.shape}")
        logger.info(f"Key columns present: {['n1' in combined_data.columns, 'n2' in combined_data.columns, 'pb' in combined_data.columns]}")
        
        # Verify we have the required columns
        required_cols = ['n1', 'n2', 'n3', 'n4', 'n5', 'pb']
        missing_cols = [col for col in required_cols if col not in combined_data.columns]
        
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            return False
        
        # Train model with proper data
        logger.info("🤖 Training model with combined features...")
        trainer = ModelTrainer(model_path)
        trainer.data = combined_data
        
        if trainer.train_model():
            logger.info("✅ Model trained and saved successfully!")
            
            # Verify model file was created
            if os.path.exists(model_path):
                file_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
                logger.info(f"✅ Model file created: {model_path} ({file_size:.2f} MB)")
                return True
            else:
                logger.error("❌ Model file was not created")
                return False
        else:
            logger.error("❌ Model training failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error during forced model creation: {e}")
        logger.exception("Full traceback:")
        return False

if __name__ == "__main__":
    success = force_create_model()
    if success:
        print("✅ Model creation completed successfully")
        print("You can now use the regular prediction functions")
    else:
        print("❌ Model creation failed")
        sys.exit(1)

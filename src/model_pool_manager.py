
"""
Multi-Model Pool Manager for SHIOL+ Smart AI

This module implements an intelligent system to automatically detect, load, and manage
multiple AI models for ensemble predictions.
"""

import os
import pickle
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from loguru import logger
from pathlib import Path
import hashlib
from datetime import datetime

class ModelInfo:
    """Information about a detected model"""
    def __init__(self, path: str, model_type: str, model_name: str, compatibility: bool):
        self.path = path
        self.model_type = model_type
        self.model_name = model_name
        self.compatibility = compatibility
        self.last_checked = datetime.now()
        self.performance_score = 0.0
        self.model_hash = self._calculate_hash()
    
    def _calculate_hash(self) -> str:
        """Calculate MD5 hash of the model file"""
        try:
            with open(self.path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception:
            return "unknown"

class ModelPoolManager:
    """
    Manages a pool of AI models for ensemble predictions
    """
    
    def __init__(self, models_dir: str = "models/"):
        self.models_dir = Path(models_dir)
        self.available_models: Dict[str, ModelInfo] = {}
        self.loaded_models: Dict[str, Any] = {}
        self.model_performances: Dict[str, float] = {}
        
        logger.info(f"Initializing ModelPoolManager with directory: {models_dir}")
        self._discover_models()
    
    def _discover_models(self) -> None:
        """Automatically discover and analyze models in the models directory"""
        logger.info("Discovering available models...")
        
        if not self.models_dir.exists():
            logger.warning(f"Models directory {self.models_dir} does not exist")
            return
        
        # Supported model file extensions
        supported_extensions = ['.pkl', '.joblib', '.h5', '.pt', '.pth']
        
        for file_path in self.models_dir.rglob('*'):
            if file_path.suffix in supported_extensions:
                self._analyze_model_file(file_path)
    
    def _analyze_model_file(self, file_path: Path) -> None:
        """Analyze a model file to determine its type and compatibility"""
        try:
            model_name = file_path.stem
            model_type = self._identify_model_type(file_path)
            compatibility = self._check_compatibility(file_path, model_type)
            
            model_info = ModelInfo(
                path=str(file_path),
                model_type=model_type,
                model_name=model_name,
                compatibility=compatibility
            )
            
            self.available_models[model_name] = model_info
            
            if compatibility:
                logger.info(f"✓ Compatible model found: {model_name} ({model_type})")
            else:
                logger.warning(f"✗ Incompatible model: {model_name} ({model_type})")
                
        except Exception as e:
            logger.error(f"Error analyzing model {file_path}: {e}")
    
    def _identify_model_type(self, file_path: Path) -> str:
        """Identify the type of model based on filename and content"""
        name_lower = file_path.name.lower()
        
        # Pattern matching based on filename
        if 'xgb' in name_lower or 'xgboost' in name_lower:
            return 'XGBoost'
        elif 'rf' in name_lower or 'randomforest' in name_lower:
            return 'RandomForest'
        elif 'lgb' in name_lower or 'lightgbm' in name_lower:
            return 'LightGBM'
        elif 'prophet' in name_lower:
            return 'Prophet'
        elif 'svm' in name_lower:
            return 'SVM'
        elif 'neural' in name_lower or 'nn' in name_lower or file_path.suffix == '.h5':
            return 'NeuralNetwork'
        elif 'ensemble' in name_lower:
            return 'Ensemble'
        else:
            return 'Unknown'
    
    def _check_compatibility(self, file_path: Path, model_type: str) -> bool:
        """Check if a model is compatible with our prediction interface"""
        try:
            # Try to load the model
            if file_path.suffix in ['.pkl', '.joblib']:
                model = joblib.load(file_path)
                
                # Check if model has predict_proba method (required for probability predictions)
                if hasattr(model, 'predict_proba'):
                    return True
                elif hasattr(model, 'predict'):
                    # Some models might only have predict but could be adapted
                    return True
                    
            elif file_path.suffix == '.h5':
                # Neural network models - would need tensorflow/keras
                return False  # For now, skip neural networks
                
            return False
            
        except Exception as e:
            logger.debug(f"Compatibility check failed for {file_path}: {e}")
            return False
    
    def load_compatible_models(self) -> Dict[str, Any]:
        """Load all compatible models into memory"""
        logger.info("Loading compatible models...")
        
        for model_name, model_info in self.available_models.items():
            if model_info.compatibility and model_name not in self.loaded_models:
                try:
                    model = joblib.load(model_info.path)
                    self.loaded_models[model_name] = model
                    logger.info(f"✓ Loaded model: {model_name}")
                    
                except Exception as e:
                    logger.error(f"Failed to load model {model_name}: {e}")
                    # Mark as incompatible
                    self.available_models[model_name].compatibility = False
        
        logger.info(f"Successfully loaded {len(self.loaded_models)} models")
        return self.loaded_models
    
    def get_model_predictions(self, features: np.ndarray) -> Dict[str, Dict[str, np.ndarray]]:
        """Get predictions from all loaded models"""
        predictions = {}
        
        for model_name, model in self.loaded_models.items():
            try:
                # Get white ball predictions (1-69)
                wb_features = features  # Adjust based on your feature engineering
                
                if hasattr(model, 'predict_proba'):
                    # For multi-class classification models
                    wb_probs = model.predict_proba(wb_features)
                    
                    # Handle different model output formats
                    if isinstance(wb_probs, list):
                        # Multi-output classifier
                        wb_probs = np.array([prob[:, 1] if prob.shape[1] > 1 else prob.flatten() 
                                           for prob in wb_probs]).T
                    
                    # Generate Powerball probabilities (1-26)
                    # This is a simplified approach - you might need model-specific logic
                    pb_probs = np.random.random(26)  # Placeholder - implement proper logic
                    
                else:
                    # For regression models or models without predict_proba
                    predictions_raw = model.predict(wb_features)
                    
                    # Convert to probabilities (normalize)
                    wb_probs = np.abs(predictions_raw)
                    wb_probs = wb_probs / np.sum(wb_probs) if np.sum(wb_probs) > 0 else wb_probs
                    
                    pb_probs = np.random.random(26)  # Placeholder
                
                predictions[model_name] = {
                    'white_ball_probs': wb_probs[:69] if len(wb_probs) >= 69 else np.pad(wb_probs, (0, 69-len(wb_probs))),
                    'powerball_probs': pb_probs[:26] if len(pb_probs) >= 26 else np.pad(pb_probs, (0, 26-len(pb_probs))),
                    'model_type': self.available_models[model_name].model_type,
                    'confidence': self._calculate_confidence(model_name)
                }
                
                logger.debug(f"Got predictions from model: {model_name}")
                
            except Exception as e:
                logger.error(f"Error getting predictions from {model_name}: {e}")
                continue
        
        return predictions
    
    def _calculate_confidence(self, model_name: str) -> float:
        """Calculate confidence score for a model based on historical performance"""
        # Use stored performance or default confidence
        return self.model_performances.get(model_name, 0.5)
    
    def update_model_performance(self, model_name: str, performance_score: float) -> None:
        """Update performance tracking for a model"""
        self.model_performances[model_name] = performance_score
        if model_name in self.available_models:
            self.available_models[model_name].performance_score = performance_score
        
        logger.info(f"Updated performance for {model_name}: {performance_score:.4f}")
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get summary of all discovered models"""
        return {
            'total_models': len(self.available_models),
            'compatible_models': len([m for m in self.available_models.values() if m.compatibility]),
            'loaded_models': len(self.loaded_models),
            'models_detail': {
                name: {
                    'type': info.model_type,
                    'compatible': info.compatibility,
                    'loaded': name in self.loaded_models,
                    'performance': info.performance_score,
                    'path': info.path
                }
                for name, info in self.available_models.items()
            }
        }

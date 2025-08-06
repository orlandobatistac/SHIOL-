
"""
Ensemble Predictor for SHIOL+ Smart AI

This module implements intelligent ensemble methods to combine predictions
from multiple AI models for optimal Powerball predictions.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from loguru import logger
from enum import Enum
import json
from datetime import datetime

from src.model_pool_manager import ModelPoolManager
from src.intelligent_generator import FeatureEngineer

class EnsembleMethod(Enum):
    """Available ensemble methods"""
    WEIGHTED_AVERAGE = "weighted_average"
    MAJORITY_VOTING = "majority_voting"
    PERFORMANCE_WEIGHTED = "performance_weighted"
    DYNAMIC_SELECTION = "dynamic_selection"
    STACKED_ENSEMBLE = "stacked_ensemble"

class EnsemblePredictor:
    """
    Intelligent ensemble predictor that combines multiple models
    """
    
    def __init__(self, historical_data: pd.DataFrame, models_dir: str = "models/"):
        self.historical_data = historical_data
        self.model_pool = ModelPoolManager(models_dir)
        self.feature_engineer = FeatureEngineer()
        self.ensemble_method = EnsembleMethod.PERFORMANCE_WEIGHTED
        self.model_weights: Dict[str, float] = {}
        
        logger.info("Initializing EnsemblePredictor...")
        self._initialize_ensemble()
    
    def _initialize_ensemble(self) -> None:
        """Initialize the ensemble system"""
        # Load all compatible models
        loaded_models = self.model_pool.load_compatible_models()
        
        if not loaded_models:
            logger.warning("No compatible models found for ensemble")
            return
        
        # Initialize equal weights for all models
        num_models = len(loaded_models)
        for model_name in loaded_models.keys():
            self.model_weights[model_name] = 1.0 / num_models
        
        logger.info(f"Ensemble initialized with {num_models} models")
    
    def predict_ensemble(self, method: Optional[EnsembleMethod] = None) -> Dict[str, Any]:
        """
        Generate ensemble predictions using specified method
        """
        if method:
            self.ensemble_method = method
        
        logger.info(f"Generating ensemble predictions using {self.ensemble_method.value}")
        
        # Prepare features for prediction
        features = self._prepare_features()
        
        if features is None:
            logger.error("Failed to prepare features for prediction")
            return {}
        
        # Get predictions from all models
        model_predictions = self.model_pool.get_model_predictions(features)
        
        if not model_predictions:
            logger.error("No model predictions available")
            return {}
        
        # Apply ensemble method
        ensemble_result = self._apply_ensemble_method(model_predictions)
        
        # Add metadata
        ensemble_result.update({
            'ensemble_method': self.ensemble_method.value,
            'models_used': list(model_predictions.keys()),
            'model_weights': self.model_weights.copy(),
            'timestamp': datetime.now().isoformat(),
            'total_models': len(model_predictions)
        })
        
        return ensemble_result
    
    def _prepare_features(self) -> Optional[np.ndarray]:
        """Prepare features for model prediction"""
        try:
            # Use the latest draws for feature engineering
            recent_data = self.historical_data.tail(100)  # Last 100 draws
            
            # Generate features using the existing feature engineering system
            features = self.feature_engineer.generate_features(recent_data)
            
            # Prepare for single prediction (latest state)
            latest_features = features.iloc[-1:].values
            
            return latest_features
            
        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            return None
    
    def _apply_ensemble_method(self, model_predictions: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, Any]:
        """Apply the selected ensemble method to combine predictions"""
        
        if self.ensemble_method == EnsembleMethod.WEIGHTED_AVERAGE:
            return self._weighted_average_ensemble(model_predictions)
        
        elif self.ensemble_method == EnsembleMethod.MAJORITY_VOTING:
            return self._majority_voting_ensemble(model_predictions)
        
        elif self.ensemble_method == EnsembleMethod.PERFORMANCE_WEIGHTED:
            return self._performance_weighted_ensemble(model_predictions)
        
        elif self.ensemble_method == EnsembleMethod.DYNAMIC_SELECTION:
            return self._dynamic_selection_ensemble(model_predictions)
        
        else:
            # Default to weighted average
            return self._weighted_average_ensemble(model_predictions)
    
    def _weighted_average_ensemble(self, model_predictions: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, Any]:
        """Combine predictions using weighted average"""
        wb_probs = np.zeros(69)
        pb_probs = np.zeros(26)
        total_weight = 0.0
        
        for model_name, predictions in model_predictions.items():
            weight = self.model_weights.get(model_name, 1.0)
            total_weight += weight
            
            wb_probs += predictions['white_ball_probs'] * weight
            pb_probs += predictions['powerball_probs'] * weight
        
        # Normalize
        if total_weight > 0:
            wb_probs /= total_weight
            pb_probs /= total_weight
        
        return {
            'white_ball_probabilities': wb_probs,
            'powerball_probabilities': pb_probs,
            'method_details': {
                'total_weight': total_weight,
                'individual_weights': self.model_weights
            }
        }
    
    def _performance_weighted_ensemble(self, model_predictions: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, Any]:
        """Combine predictions weighted by historical performance"""
        wb_probs = np.zeros(69)
        pb_probs = np.zeros(26)
        total_weight = 0.0
        
        for model_name, predictions in model_predictions.items():
            # Use performance score as weight
            performance = self.model_pool.model_performances.get(model_name, 0.5)
            confidence = predictions.get('confidence', 0.5)
            
            # Combined weight from performance and confidence
            weight = performance * confidence
            total_weight += weight
            
            wb_probs += predictions['white_ball_probs'] * weight
            pb_probs += predictions['powerball_probs'] * weight
        
        # Normalize
        if total_weight > 0:
            wb_probs /= total_weight
            pb_probs /= total_weight
        
        return {
            'white_ball_probabilities': wb_probs,
            'powerball_probabilities': pb_probs,
            'method_details': {
                'total_weight': total_weight,
                'performance_scores': {
                    name: self.model_pool.model_performances.get(name, 0.5)
                    for name in model_predictions.keys()
                }
            }
        }
    
    def _majority_voting_ensemble(self, model_predictions: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, Any]:
        """Combine predictions using majority voting on top predictions"""
        # Convert probabilities to top N selections for voting
        top_n = 10  # Consider top 10 numbers from each model
        
        wb_votes = np.zeros(69)
        pb_votes = np.zeros(26)
        
        for model_name, predictions in model_predictions.items():
            # Get top white ball numbers
            wb_top_indices = np.argsort(predictions['white_ball_probs'])[-top_n:]
            wb_votes[wb_top_indices] += 1
            
            # Get top powerball numbers
            pb_top_indices = np.argsort(predictions['powerball_probs'])[-5:]  # Top 5 for powerball
            pb_votes[pb_top_indices] += 1
        
        # Convert votes back to probabilities
        wb_probs = wb_votes / np.sum(wb_votes) if np.sum(wb_votes) > 0 else wb_votes
        pb_probs = pb_votes / np.sum(pb_votes) if np.sum(pb_votes) > 0 else pb_votes
        
        return {
            'white_ball_probabilities': wb_probs,
            'powerball_probabilities': pb_probs,
            'method_details': {
                'top_n_considered': top_n,
                'vote_counts': {
                    'white_ball_votes': wb_votes.tolist(),
                    'powerball_votes': pb_votes.tolist()
                }
            }
        }
    
    def _dynamic_selection_ensemble(self, model_predictions: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, Any]:
        """Dynamically select best performing models for current prediction"""
        # For now, select top 3 performing models
        performance_scores = {
            name: self.model_pool.model_performances.get(name, 0.5)
            for name in model_predictions.keys()
        }
        
        # Sort by performance and select top models
        top_models = sorted(performance_scores.items(), key=lambda x: x[1], reverse=True)[:3]
        selected_models = [name for name, _ in top_models]
        
        # Apply weighted average on selected models only
        filtered_predictions = {
            name: predictions for name, predictions in model_predictions.items()
            if name in selected_models
        }
        
        result = self._weighted_average_ensemble(filtered_predictions)
        result['method_details']['selected_models'] = selected_models
        result['method_details']['selection_criteria'] = 'top_3_performance'
        
        return result
    
    def update_model_weights(self, performance_feedback: Dict[str, float]) -> None:
        """Update model weights based on performance feedback"""
        for model_name, performance in performance_feedback.items():
            if model_name in self.model_weights:
                # Update pool manager performance tracking
                self.model_pool.update_model_performance(model_name, performance)
                
                # Update ensemble weights (simple approach)
                self.model_weights[model_name] = performance
        
        # Normalize weights
        total_weight = sum(self.model_weights.values())
        if total_weight > 0:
            for model_name in self.model_weights:
                self.model_weights[model_name] /= total_weight
        
        logger.info("Model weights updated based on performance feedback")
    
    def get_ensemble_summary(self) -> Dict[str, Any]:
        """Get summary of ensemble configuration"""
        model_summary = self.model_pool.get_model_summary()
        
        return {
            'ensemble_method': self.ensemble_method.value,
            'model_weights': self.model_weights,
            'model_pool_summary': model_summary,
            'available_methods': [method.value for method in EnsembleMethod]
        }

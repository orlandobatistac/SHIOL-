
"""
Model Pool Manager for SHIOL+ Ensemble System

This module manages a pool of AI models for ensemble predictions,
providing model discovery, loading, validation, and performance tracking.
"""

import os
import pickle
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from loguru import logger
from datetime import datetime
import configparser

class ModelPoolManager:
    """
    Manages a pool of machine learning models for ensemble predictions
    """
    
    def __init__(self, models_dir: str = "models/"):
        self.models_dir = models_dir
        self.loaded_models: Dict[str, Any] = {}
        self.model_metadata: Dict[str, Dict] = {}
        self.model_performances: Dict[str, float] = {}
        self.compatible_models: List[str] = []
        
        logger.info(f"Initializing ModelPoolManager with directory: {models_dir}")
        self._discover_models()
    
    def _discover_models(self) -> None:
        """Discover available models in the models directory"""
        logger.info("Discovering available models...")
        
        if not os.path.exists(self.models_dir):
            logger.warning(f"Models directory {self.models_dir} does not exist")
            return
        
        for file in os.listdir(self.models_dir):
            if file.endswith(('.pkl', '.joblib', '.model')):
                model_path = os.path.join(self.models_dir, file)
                model_name = os.path.splitext(file)[0]
                
                try:
                    model_info = self._analyze_model_file(model_path)
                    if model_info['compatible']:
                        self.compatible_models.append(model_name)
                        self.model_metadata[model_name] = model_info
                        logger.info(f"✓ Compatible model found: {model_name} ({model_info['type']})")
                    else:
                        logger.warning(f"✗ Incompatible model: {model_name} ({model_info['type']})")
                        
                except Exception as e:
                    logger.error(f"Error analyzing model {model_name}: {e}")
    
    def _analyze_model_file(self, model_path: str) -> Dict[str, Any]:
        """Analyze a model file to determine compatibility and type"""
        try:
            # Try to load with joblib first (most common)
            model_data = joblib.load(model_path)
            
            model_info = {
                'path': model_path,
                'compatible': False,
                'type': 'Unknown',
                'has_predict_proba': False,
                'input_features': None,
                'output_format': None
            }
            
            # Check if it's a model bundle (like SHIOL+ format)
            if isinstance(model_data, dict) and 'model' in model_data:
                model = model_data['model']
                model_info['type'] = 'SHIOL+ Bundle'
                model_info['target_columns'] = model_data.get('target_columns', [])
                
                # Check if model has required methods
                if hasattr(model, 'predict_proba'):
                    model_info['has_predict_proba'] = True
                    model_info['compatible'] = True
                    model_info['output_format'] = 'multi_output_probabilities'
                
            # Check if it's a direct model
            elif hasattr(model_data, 'predict_proba'):
                model_info['type'] = type(model_data).__name__
                model_info['has_predict_proba'] = True
                model_info['compatible'] = True
                model_info['output_format'] = 'direct_probabilities'
            
            return model_info
            
        except Exception as e:
            return {
                'path': model_path,
                'compatible': False,
                'type': 'Unknown',
                'error': str(e)
            }
    
    def load_compatible_models(self) -> Dict[str, Any]:
        """Load all compatible models into memory"""
        logger.info("Loading compatible models...")
        
        for model_name in self.compatible_models:
            try:
                model_info = self.model_metadata[model_name]
                model_data = joblib.load(model_info['path'])
                
                if model_info['type'] == 'SHIOL+ Bundle':
                    self.loaded_models[model_name] = {
                        'model': model_data['model'],
                        'target_columns': model_data.get('target_columns', []),
                        'metadata': model_info
                    }
                else:
                    self.loaded_models[model_name] = {
                        'model': model_data,
                        'metadata': model_info
                    }
                
                # Initialize performance score
                self.model_performances[model_name] = 0.5  # Default neutral performance
                
                logger.info(f"Loaded model: {model_name}")
                
            except Exception as e:
                logger.error(f"Failed to load model {model_name}: {e}")
                if model_name in self.compatible_models:
                    self.compatible_models.remove(model_name)
        
        logger.info(f"Successfully loaded {len(self.loaded_models)} models")
        return self.loaded_models
    
    def get_model_predictions(self, features: np.ndarray) -> Dict[str, Dict[str, np.ndarray]]:
        """Get predictions from all loaded models"""
        predictions = {}
        
        for model_name, model_data in self.loaded_models.items():
            try:
                model = model_data['model']
                metadata = model_data['metadata']
                
                # Get predictions based on model type
                if metadata['type'] == 'SHIOL+ Bundle':
                    pred_probas = model.predict_proba(features)
                    
                    # Extract white ball and powerball probabilities
                    wb_probs, pb_probs = self._extract_probabilities_from_bundle(
                        pred_probas, model_data.get('target_columns', [])
                    )
                else:
                    # Direct model prediction
                    pred_probas = model.predict_proba(features)
                    wb_probs, pb_probs = self._extract_probabilities_direct(pred_probas)
                
                # Calculate prediction confidence
                confidence = self._calculate_prediction_confidence(wb_probs, pb_probs)
                
                predictions[model_name] = {
                    'white_ball_probs': wb_probs,
                    'powerball_probs': pb_probs,
                    'confidence': confidence,
                    'model_type': metadata['type']
                }
                
            except Exception as e:
                logger.error(f"Error getting predictions from {model_name}: {e}")
                continue
        
        return predictions
    
    def _extract_probabilities_from_bundle(self, pred_probas: List, target_columns: List) -> Tuple[np.ndarray, np.ndarray]:
        """Extract white ball and powerball probabilities from SHIOL+ bundle predictions"""
        try:
            # Convert list of predictions to probability arrays
            prob_class_1 = [p[:, 1] if p.shape[1] > 1 else p.flatten() for p in pred_probas]
            all_probs = np.array(prob_class_1).flatten()
            
            # Separate white ball and powerball probabilities based on target columns
            wb_probs = np.zeros(69)
            pb_probs = np.zeros(26)
            
            for i, col in enumerate(target_columns):
                if col.startswith('wb_') and i < len(all_probs):
                    wb_idx = int(col.split('_')[1]) - 1  # wb_1 -> index 0
                    if 0 <= wb_idx < 69:
                        wb_probs[wb_idx] = all_probs[i]
                elif col.startswith('pb_') and i < len(all_probs):
                    pb_idx = int(col.split('_')[1]) - 1  # pb_1 -> index 0
                    if 0 <= pb_idx < 26:
                        pb_probs[pb_idx] = all_probs[i]
            
            # Normalize probabilities
            wb_probs = wb_probs / wb_probs.sum() if wb_probs.sum() > 0 else np.ones(69) / 69
            pb_probs = pb_probs / pb_probs.sum() if pb_probs.sum() > 0 else np.ones(26) / 26
            
            return wb_probs, pb_probs
            
        except Exception as e:
            logger.error(f"Error extracting probabilities from bundle: {e}")
            return np.ones(69) / 69, np.ones(26) / 26
    
    def _extract_probabilities_direct(self, pred_probas) -> Tuple[np.ndarray, np.ndarray]:
        """Extract probabilities from direct model predictions"""
        try:
            if isinstance(pred_probas, list):
                # Multi-output case
                all_probs = np.array([p[:, 1] if p.shape[1] > 1 else p.flatten() 
                                    for p in pred_probas]).flatten()
            else:
                all_probs = pred_probas.flatten()
            
            # Split probabilities (assuming first 69 are white balls, rest are powerball)
            if len(all_probs) >= 95:  # 69 + 26
                wb_probs = all_probs[:69]
                pb_probs = all_probs[69:95]
            else:
                # Fallback: use available probabilities and pad with uniform
                wb_probs = all_probs[:69] if len(all_probs) >= 69 else np.ones(69) / 69
                pb_probs = np.ones(26) / 26
            
            # Normalize
            wb_probs = wb_probs / wb_probs.sum() if wb_probs.sum() > 0 else np.ones(69) / 69
            pb_probs = pb_probs / pb_probs.sum() if pb_probs.sum() > 0 else np.ones(26) / 26
            
            return wb_probs, pb_probs
            
        except Exception as e:
            logger.error(f"Error extracting direct probabilities: {e}")
            return np.ones(69) / 69, np.ones(26) / 26
    
    def _calculate_prediction_confidence(self, wb_probs: np.ndarray, pb_probs: np.ndarray) -> float:
        """Calculate confidence score for predictions"""
        try:
            # Confidence based on entropy (lower entropy = higher confidence)
            wb_entropy = -np.sum(wb_probs * np.log(wb_probs + 1e-10))
            pb_entropy = -np.sum(pb_probs * np.log(pb_probs + 1e-10))
            
            # Normalize entropy to [0, 1] and invert for confidence
            max_wb_entropy = np.log(69)
            max_pb_entropy = np.log(26)
            
            wb_confidence = 1.0 - (wb_entropy / max_wb_entropy)
            pb_confidence = 1.0 - (pb_entropy / max_pb_entropy)
            
            # Weighted average (white balls weight more)
            overall_confidence = 0.8 * wb_confidence + 0.2 * pb_confidence
            
            return max(0.0, min(1.0, overall_confidence))
            
        except Exception:
            return 0.5  # Default neutral confidence
    
    def update_model_performance(self, model_name: str, performance_score: float) -> None:
        """Update performance score for a model"""
        if model_name in self.model_performances:
            # Use exponential moving average for performance updates
            alpha = 0.3  # Learning rate
            old_score = self.model_performances[model_name]
            new_score = alpha * performance_score + (1 - alpha) * old_score
            self.model_performances[model_name] = new_score
            
            logger.info(f"Updated performance for {model_name}: {old_score:.3f} -> {new_score:.3f}")
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get summary of model pool status"""
        return {
            'total_models_discovered': len(self.model_metadata),
            'compatible_models': len(self.compatible_models),
            'loaded_models': len(self.loaded_models),
            'model_performances': self.model_performances.copy(),
            'models_by_type': {
                name: data['metadata']['type'] 
                for name, data in self.loaded_models.items()
            }
        }

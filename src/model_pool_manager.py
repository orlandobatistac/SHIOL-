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

        try:
            model_files = [f for f in os.listdir(self.models_dir) 
                          if f.endswith(('.pkl', '.joblib', '.model'))]
            
            if not model_files:
                logger.info("No model files found in models directory")
                return

            for file in model_files:
                model_path = os.path.join(self.models_dir, file)
                model_name = os.path.splitext(file)[0]

                try:
                    # Check if file is accessible and not corrupted
                    if not os.access(model_path, os.R_OK):
                        logger.warning(f"Cannot read model file: {model_path}")
                        continue

                    model_info = self._analyze_model_file(model_path)
                    if model_info.get('compatible', False):
                        self.compatible_models.append(model_name)
                        self.model_metadata[model_name] = model_info
                        logger.info(f"✓ Compatible model found: {model_name} ({model_info.get('type', 'Unknown')})")
                    else:
                        logger.debug(f"✗ Incompatible model: {model_name} ({model_info.get('type', 'Unknown')})")

                except Exception as e:
                    logger.error(f"Error analyzing model {model_name}: {e}")
                    
        except OSError as e:
            logger.error(f"Error accessing models directory {self.models_dir}: {e}")

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
                model_info['expected_features'] = 15  # Standard SHIOL+ feature count

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

    def get_model_predictions(self, features: np.ndarray) -> Dict[str, Dict[str, Any]]:
        """Get predictions from all loaded models with robust feature compatibility"""
        predictions = {}

        for model_name, model_data in self.loaded_models.items():
            try:
                model = model_data['model']
                metadata = model_data['metadata']

                # Check if model expects different number of features
                expected_features = metadata.get('expected_features', 15)

                # Robust feature processing
                processed_features = self._process_features_for_model(features, model_name, expected_features)
                if processed_features is None:
                    continue

                # Get raw prediction probabilities
                raw_predictions = model.predict_proba(processed_features)

                # Process and normalize predictions
                processed_preds = self._process_model_output(raw_predictions, metadata)

                predictions[model_name] = processed_preds

            except Exception as e:
                logger.error(f"Error getting predictions from {model_name}: {e}")
                continue

        return predictions

    def _process_features_for_model(self, features: np.ndarray, model_name: str, expected_features: int) -> Optional[np.ndarray]:
        """Process and validate features for a specific model"""
        try:
            # Convert to numpy array if not already
            if hasattr(features, 'select_dtypes'):
                # If it's a DataFrame, select only numeric columns
                numeric_features = features.select_dtypes(include=[np.number]).values
            elif isinstance(features, np.ndarray):
                numeric_features = features
            else:
                # Try to convert to DataFrame first
                try:
                    df_features = pd.DataFrame(features)
                    numeric_features = df_features.select_dtypes(include=[np.number]).values
                except:
                    logger.error(f"Cannot convert features to numeric format for {model_name}")
                    return None

            # Ensure features are 2D
            if numeric_features.ndim == 1:
                numeric_features = numeric_features.reshape(1, -1)

            # Handle feature count mismatch
            current_feature_count = numeric_features.shape[1]
            
            if current_feature_count != expected_features:
                logger.warning(f"Feature shape mismatch for {model_name}, expected: {expected_features}, got: {current_feature_count}")

                if current_feature_count > expected_features:
                    # Use the standard SHIOL+ features in order
                    standard_features = [
                        "even_count", "odd_count", "sum", "spread", "consecutive_count",
                        "avg_delay", "max_delay", "min_delay",
                        "dist_to_recent", "avg_dist_to_top_n", "dist_to_centroid",
                        "time_weight", "increasing_trend_count", "decreasing_trend_count",
                        "stable_trend_count"
                    ]
                    
                    # Take only the first N features that match expected count
                    adjusted_features = numeric_features[:, :expected_features]
                    logger.info(f"Adjusted features for {model_name}: {adjusted_features.shape}")
                    return adjusted_features
                    
                elif current_feature_count < expected_features:
                    # Pad with zeros or mean values
                    padding_needed = expected_features - current_feature_count
                    padding = np.zeros((numeric_features.shape[0], padding_needed))
                    adjusted_features = np.hstack([numeric_features, padding])
                    logger.info(f"Padded features for {model_name}: {adjusted_features.shape}")
                    return adjusted_features
            
            return numeric_features

        except Exception as e:
            logger.error(f"Error processing features for {model_name}: {e}")
            return None

    def _process_model_output(self, raw_predictions: Any, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Process raw model output into structured predictions"""
        if metadata['type'] == 'SHIOL+ Bundle':
            # Extract white ball and powerball probabilities
            wb_probs, pb_probs = self._extract_probabilities_from_bundle(
                raw_predictions, metadata.get('target_columns', [])
            )
        else:
            # Direct model prediction
            wb_probs, pb_probs = self._extract_probabilities_direct(raw_predictions)

        # Calculate prediction confidence
        confidence = self._calculate_prediction_confidence(wb_probs, pb_probs)

        return {
            'white_ball_probs': wb_probs,
            'powerball_probs': pb_probs,
            'confidence': confidence,
            'model_type': metadata['type']
        }

    def _extract_probabilities_from_bundle(self, pred_probas: List, target_columns: List) -> Tuple[np.ndarray, np.ndarray]:
        """Extract white ball and powerball probabilities from SHIOL+ bundle predictions"""
        try:
            # Convert list of predictions to probability arrays with proper validation
            prob_class_1 = []
            for p in pred_probas:
                if hasattr(p, 'shape') and p.shape[1] > 1:
                    # Use class 1 probabilities and ensure they sum to 1
                    class_1_probs = p[:, 1]
                    # Normalize if needed
                    if not np.isclose(class_1_probs.sum(), 1.0, rtol=1e-3):
                        class_1_probs = class_1_probs / class_1_probs.sum()
                    prob_class_1.append(class_1_probs)
                else:
                    flattened = p.flatten()
                    # Normalize flattened probabilities
                    if flattened.sum() > 0:
                        flattened = flattened / flattened.sum()
                    prob_class_1.append(flattened)
            
            all_probs = np.array(prob_class_1).flatten()

            # Separate white ball and powerball probabilities based on target columns
            wb_probs = np.zeros(69)
            pb_probs = np.zeros(26)

            for i, col in enumerate(target_columns):
                if i >= len(all_probs):
                    break
                    
                if col.startswith('wb_'):
                    try:
                        wb_idx = int(col.split('_')[1]) - 1  # wb_1 -> index 0
                        if 0 <= wb_idx < 69:
                            wb_probs[wb_idx] = max(0.0, min(1.0, all_probs[i]))  # Clamp to [0,1]
                    except (ValueError, IndexError):
                        continue
                        
                elif col.startswith('pb_'):
                    try:
                        pb_idx = int(col.split('_')[1]) - 1  # pb_1 -> index 0
                        if 0 <= pb_idx < 26:
                            pb_probs[pb_idx] = max(0.0, min(1.0, all_probs[i]))  # Clamp to [0,1]
                    except (ValueError, IndexError):
                        continue

            # Robust normalization with epsilon for numerical stability
            epsilon = 1e-10
            wb_sum = wb_probs.sum()
            pb_sum = pb_probs.sum()
            
            if wb_sum > epsilon:
                wb_probs = wb_probs / wb_sum
            else:
                wb_probs = np.ones(69) / 69  # Uniform distribution fallback
                
            if pb_sum > epsilon:
                pb_probs = pb_probs / pb_sum
            else:
                pb_probs = np.ones(26) / 26  # Uniform distribution fallback

            # Final validation - ensure probabilities sum to 1
            if not np.isclose(wb_probs.sum(), 1.0, rtol=1e-6):
                wb_probs = wb_probs / wb_probs.sum()
            if not np.isclose(pb_probs.sum(), 1.0, rtol=1e-6):
                pb_probs = pb_probs / pb_probs.sum()

            return wb_probs, pb_probs

        except Exception as e:
            logger.error(f"Error extracting probabilities from bundle: {e}")
            # Return uniform distributions as safe fallback
            return np.ones(69) / 69, np.ones(26) / 26

    def _extract_probabilities_direct(self, pred_probas) -> Tuple[np.ndarray, np.ndarray]:
        """Extract probabilities from direct model predictions with robust normalization"""
        try:
            if isinstance(pred_probas, list):
                # Multi-output case with proper probability validation
                processed_probs = []
                for p in pred_probas:
                    if hasattr(p, 'shape') and p.shape[1] > 1:
                        # Use class 1 probabilities
                        class_1_probs = p[:, 1]
                        # Ensure probabilities are valid
                        class_1_probs = np.clip(class_1_probs, 0.0, 1.0)
                        processed_probs.append(class_1_probs)
                    else:
                        flattened = p.flatten()
                        flattened = np.clip(flattened, 0.0, 1.0)
                        processed_probs.append(flattened)
                
                all_probs = np.concatenate(processed_probs)
            else:
                all_probs = pred_probas.flatten()
                all_probs = np.clip(all_probs, 0.0, 1.0)

            # Split probabilities with better fallback handling
            if len(all_probs) >= 95:  # 69 + 26
                wb_probs = all_probs[:69]
                pb_probs = all_probs[69:95]
            elif len(all_probs) >= 69:
                wb_probs = all_probs[:69]
                pb_probs = np.ones(26) / 26  # Uniform fallback for powerball
            else:
                # Not enough probabilities - use uniform distributions
                wb_probs = np.ones(69) / 69
                pb_probs = np.ones(26) / 26

            # Robust normalization with numerical stability
            epsilon = 1e-10
            
            # Normalize white ball probabilities
            wb_sum = wb_probs.sum()
            if wb_sum > epsilon:
                wb_probs = wb_probs / wb_sum
            else:
                wb_probs = np.ones(69) / 69
            
            # Normalize powerball probabilities
            pb_sum = pb_probs.sum()
            if pb_sum > epsilon:
                pb_probs = pb_probs / pb_sum
            else:
                pb_probs = np.ones(26) / 26

            # Final validation to ensure probabilities sum to 1
            if not np.isclose(wb_probs.sum(), 1.0, rtol=1e-6):
                wb_probs = wb_probs / wb_probs.sum()
            if not np.isclose(pb_probs.sum(), 1.0, rtol=1e-6):
                pb_probs = pb_probs / pb_probs.sum()

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
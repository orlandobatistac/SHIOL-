"""
Enhanced Ensemble Predictor for SHIOL+ Smart AI

This module implements intelligent ensemble methods to combine predictions
from multiple AI models for optimal Powerball predictions with advanced
strategies and adaptive performance monitoring.
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
    """Available ensemble methods with enhanced options"""
    WEIGHTED_AVERAGE = "weighted_average"
    PERFORMANCE_WEIGHTED = "performance_weighted"
    DYNAMIC_SELECTION = "dynamic_selection"
    MAJORITY_VOTING = "majority_voting"
    CONFIDENCE_WEIGHTED = "confidence_weighted"
    ADAPTIVE_HYBRID = "adaptive_hybrid"

class EnsemblePredictor:
    """
    Advanced ensemble predictor that intelligently combines multiple models
    """

    def __init__(self, historical_data: pd.DataFrame, models_dir: str = "models/"):
        self.historical_data = historical_data
        self.model_pool = ModelPoolManager(models_dir)
        self.feature_engineer = FeatureEngineer(historical_data)
        self.ensemble_method = EnsembleMethod.ADAPTIVE_HYBRID
        self.model_weights: Dict[str, float] = {}
        self.performance_history: Dict[str, List[float]] = {}
        self.prediction_count = 0

        logger.info("Initializing Enhanced EnsemblePredictor...")
        self._initialize_ensemble()

    def _initialize_ensemble(self) -> None:
        """Initialize the enhanced ensemble system"""
        # Load all compatible models
        loaded_models = self.model_pool.load_compatible_models()

        if not loaded_models:
            logger.warning("No compatible models found for ensemble")
            return

        # Initialize adaptive weights based on model types
        self._initialize_adaptive_weights(loaded_models)

        logger.info(f"Enhanced ensemble initialized with {len(loaded_models)} models")
        logger.info(f"Available methods: {[method.value for method in EnsembleMethod]}")

    def _initialize_adaptive_weights(self, loaded_models: Dict[str, Any]) -> None:
        """Initialize adaptive weights based on model characteristics"""
        num_models = len(loaded_models)

        if num_models == 0:
            return

        # Initialize performance history
        for model_name in loaded_models.keys():
            self.performance_history[model_name] = [0.5]  # Start with neutral performance

        # Set initial weights based on model types and characteristics
        for model_name, model_data in loaded_models.items():
            model_type = model_data['metadata']['type']

            # Weight assignment based on model sophistication
            if 'SHIOL+' in model_type:
                base_weight = 0.4  # Higher weight for SHIOL+ models
            elif 'XGBoost' in model_type or 'Boost' in model_type:
                base_weight = 0.3  # Good weight for gradient boosting
            elif 'Random' in model_type:
                base_weight = 0.2  # Lower weight for random forest
            else:
                base_weight = 0.1  # Lowest weight for unknown models

            self.model_weights[model_name] = base_weight

        # Normalize weights
        self._normalize_weights()

    def _normalize_weights(self) -> None:
        """Normalize model weights to sum to 1"""
        total_weight = sum(self.model_weights.values())
        if total_weight > 0:
            for model_name in self.model_weights:
                self.model_weights[model_name] /= total_weight

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
            return self._fallback_prediction()

        # Get predictions from all models
        model_predictions = self.model_pool.get_model_predictions(features)

        if not model_predictions:
            logger.error("No model predictions available")
            return self._fallback_prediction()

        # Apply ensemble method
        ensemble_result = self._apply_ensemble_method(model_predictions)

        # Update prediction count and performance tracking
        self.prediction_count += 1

        # Add comprehensive metadata
        ensemble_result.update({
            'ensemble_method': self.ensemble_method.value,
            'models_used': list(model_predictions.keys()),
            'model_weights': self.model_weights.copy(),
            'timestamp': datetime.now().isoformat(),
            'total_models': len(model_predictions),
            'prediction_count': self.prediction_count,
            'ensemble_version': '2.0'
        })

        return ensemble_result

    def _prepare_features(self) -> Optional[pd.DataFrame]:
        """Prepare features for model prediction"""
        try:
            # Use the feature engineer to get latest features
            latest_features = self.feature_engineer.engineer_features(use_temporal_analysis=True)

            if latest_features.empty:
                logger.error("No features generated for ensemble prediction")
                return self._fallback_prediction()

            # Get the most recent feature row as DataFrame for pool manager processing
            latest_features_df = latest_features.iloc[-1:]

            return latest_features_df

        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            return None

    def _apply_ensemble_method(self, model_predictions: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, Any]:
        """Apply the selected ensemble method to combine predictions"""

        method_map = {
            EnsembleMethod.WEIGHTED_AVERAGE: self._weighted_average_ensemble,
            EnsembleMethod.PERFORMANCE_WEIGHTED: self._performance_weighted_ensemble,
            EnsembleMethod.DYNAMIC_SELECTION: self._dynamic_selection_ensemble,
            EnsembleMethod.MAJORITY_VOTING: self._majority_voting_ensemble,
            EnsembleMethod.CONFIDENCE_WEIGHTED: self._confidence_weighted_ensemble,
            EnsembleMethod.ADAPTIVE_HYBRID: self._adaptive_hybrid_ensemble
        }

        ensemble_func = method_map.get(self.ensemble_method, self._weighted_average_ensemble)
        return ensemble_func(model_predictions)

    def _weighted_average_ensemble(self, model_predictions: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, Any]:
        """Combine predictions using weighted average"""
        wb_probs = np.zeros(69)
        pb_probs = np.zeros(26)
        total_weight = 0.0

        for model_name, predictions in model_predictions.items():
            weight = self.model_weights.get(model_name, 1.0 / len(model_predictions))
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
                'individual_weights': {name: self.model_weights.get(name, 0) for name in model_predictions.keys()}
            }
        }

    def _performance_weighted_ensemble(self, model_predictions: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, Any]:
        """Combine predictions weighted by historical performance"""
        wb_probs = np.zeros(69)
        pb_probs = np.zeros(26)
        total_weight = 0.0
        performance_weights = {}

        for model_name, predictions in model_predictions.items():
            # Use recent performance average
            recent_performance = np.mean(self.performance_history.get(model_name, [0.5])[-5:])
            confidence = predictions.get('confidence', 0.5)

            # Combined weight from performance and confidence
            weight = (recent_performance ** 2) * confidence  # Square to emphasize good performers
            performance_weights[model_name] = weight
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
                'performance_weights': performance_weights,
                'recent_performances': {
                    name: np.mean(self.performance_history.get(name, [0.5])[-5:])
                    for name in model_predictions.keys()
                }
            }
        }

    def _dynamic_selection_ensemble(self, model_predictions: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, Any]:
        """Dynamically select best performing models for current prediction"""
        # Get performance scores and select top performers
        performance_scores = {}
        for model_name in model_predictions.keys():
            recent_perf = np.mean(self.performance_history.get(model_name, [0.5])[-3:])
            confidence = model_predictions[model_name].get('confidence', 0.5)
            combined_score = 0.7 * recent_perf + 0.3 * confidence
            performance_scores[model_name] = combined_score

        # Select top 60% of models
        sorted_models = sorted(performance_scores.items(), key=lambda x: x[1], reverse=True)
        top_count = max(1, int(len(sorted_models) * 0.6))
        selected_models = [name for name, _ in sorted_models[:top_count]]

        # Apply weighted average on selected models only
        filtered_predictions = {
            name: predictions for name, predictions in model_predictions.items()
            if name in selected_models
        }

        result = self._performance_weighted_ensemble(filtered_predictions)
        result['method_details'].update({
            'selected_models': selected_models,
            'selection_criteria': 'top_60_percent_performance',
            'performance_scores': performance_scores
        })

        return result

    def _majority_voting_ensemble(self, model_predictions: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, Any]:
        """Combine predictions using majority voting on top predictions"""
        top_n_wb = 15  # Consider top 15 white ball numbers from each model
        top_n_pb = 8   # Consider top 8 powerball numbers from each model

        wb_votes = np.zeros(69)
        pb_votes = np.zeros(26)

        for model_name, predictions in model_predictions.items():
            # Weight votes by model performance
            model_weight = self.model_weights.get(model_name, 1.0)

            # Get top white ball numbers
            wb_top_indices = np.argsort(predictions['white_ball_probs'])[-top_n_wb:]
            wb_votes[wb_top_indices] += model_weight

            # Get top powerball numbers
            pb_top_indices = np.argsort(predictions['powerball_probs'])[-top_n_pb:]
            pb_votes[pb_top_indices] += model_weight

        # Convert votes back to probabilities
        wb_probs = wb_votes / np.sum(wb_votes) if np.sum(wb_votes) > 0 else np.ones(69) / 69
        pb_probs = pb_votes / np.sum(pb_votes) if np.sum(pb_votes) > 0 else np.ones(26) / 26

        return {
            'white_ball_probabilities': wb_probs,
            'powerball_probabilities': pb_probs,
            'method_details': {
                'top_n_wb_considered': top_n_wb,
                'top_n_pb_considered': top_n_pb,
                'total_votes_wb': np.sum(wb_votes),
                'total_votes_pb': np.sum(pb_votes)
            }
        }

    def _confidence_weighted_ensemble(self, model_predictions: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, Any]:
        """Weight predictions by model confidence scores"""
        wb_probs = np.zeros(69)
        pb_probs = np.zeros(26)
        total_confidence = 0.0
        confidence_weights = {}

        for model_name, predictions in model_predictions.items():
            confidence = predictions.get('confidence', 0.5)
            # Apply exponential weighting to emphasize high confidence
            exp_confidence = np.exp(2 * confidence)  # Exponential amplification
            confidence_weights[model_name] = exp_confidence
            total_confidence += exp_confidence

            wb_probs += predictions['white_ball_probs'] * exp_confidence
            pb_probs += predictions['powerball_probs'] * exp_confidence

        # Normalize
        if total_confidence > 0:
            wb_probs /= total_confidence
            pb_probs /= total_confidence

        return {
            'white_ball_probabilities': wb_probs,
            'powerball_probabilities': pb_probs,
            'method_details': {
                'total_confidence': total_confidence,
                'confidence_weights': confidence_weights,
                'raw_confidences': {
                    name: predictions.get('confidence', 0.5)
                    for name, predictions in model_predictions.items()
                }
            }
        }

    def _adaptive_hybrid_ensemble(self, model_predictions: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, Any]:
        """Advanced hybrid approach that adapts based on prediction context"""
        # Analyze prediction characteristics
        context = self._analyze_prediction_context(model_predictions)

        # Choose strategy based on context
        if context['high_agreement']:
            # High agreement -> use confidence weighting
            result = self._confidence_weighted_ensemble(model_predictions)
            strategy_used = 'confidence_weighted'
        elif context['performance_disparity']:
            # Performance disparity -> use dynamic selection
            result = self._dynamic_selection_ensemble(model_predictions)
            strategy_used = 'dynamic_selection'
        else:
            # Default -> use performance weighted
            result = self._performance_weighted_ensemble(model_predictions)
            strategy_used = 'performance_weighted'

        result['method_details'].update({
            'hybrid_strategy_used': strategy_used,
            'context_analysis': context,
            'adaptation_reason': self._get_adaptation_reason(context)
        })

        return result

    def _analyze_prediction_context(self, model_predictions: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, Any]:
        """Analyze the context of current predictions to choose optimal strategy"""
        if len(model_predictions) < 2:
            return {'high_agreement': False, 'performance_disparity': False}

        # Calculate agreement between models
        wb_predictions = [pred['white_ball_probs'] for pred in model_predictions.values()]
        pb_predictions = [pred['powerball_probs'] for pred in model_predictions.values()]

        # Measure agreement using correlation
        wb_correlations = []
        pb_correlations = []

        for i in range(len(wb_predictions)):
            for j in range(i + 1, len(wb_predictions)):
                wb_corr = np.corrcoef(wb_predictions[i], wb_predictions[j])[0, 1]
                pb_corr = np.corrcoef(pb_predictions[i], pb_predictions[j])[0, 1]

                if not np.isnan(wb_corr):
                    wb_correlations.append(wb_corr)
                if not np.isnan(pb_corr):
                    pb_correlations.append(pb_corr)

        avg_wb_correlation = np.mean(wb_correlations) if wb_correlations else 0
        avg_pb_correlation = np.mean(pb_correlations) if pb_correlations else 0
        overall_agreement = (avg_wb_correlation + avg_pb_correlation) / 2

        # Check performance disparity
        performances = [
            np.mean(self.performance_history.get(name, [0.5])[-3:])
            for name in model_predictions.keys()
        ]
        performance_std = np.std(performances)

        return {
            'high_agreement': overall_agreement > 0.7,
            'performance_disparity': performance_std > 0.2,
            'agreement_score': overall_agreement,
            'performance_std': performance_std,
            'avg_wb_correlation': avg_wb_correlation,
            'avg_pb_correlation': avg_pb_correlation
        }

    def _get_adaptation_reason(self, context: Dict[str, Any]) -> str:
        """Get human-readable reason for strategy adaptation"""
        if context['high_agreement']:
            return f"High model agreement (score: {context['agreement_score']:.3f}) - using confidence weighting"
        elif context['performance_disparity']:
            return f"High performance disparity (std: {context['performance_std']:.3f}) - using dynamic selection"
        else:
            return "Balanced conditions - using performance weighting"

    def _fallback_prediction(self) -> Dict[str, Any]:
        """Fallback prediction when ensemble fails"""
        logger.warning("Using fallback uniform predictions")
        return {
            'white_ball_probabilities': np.ones(69) / 69,
            'powerball_probabilities': np.ones(26) / 26,
            'ensemble_method': 'fallback_uniform',
            'models_used': [],
            'total_models': 0
        }

    def update_model_weights(self, performance_feedback: Dict[str, float]) -> None:
        """Update model weights based on performance feedback"""
        for model_name, performance in performance_feedback.items():
            if model_name in self.model_weights:
                # Update performance history
                if model_name not in self.performance_history:
                    self.performance_history[model_name] = []

                self.performance_history[model_name].append(performance)

                # Keep only last 20 performance scores
                if len(self.performance_history[model_name]) > 20:
                    self.performance_history[model_name] = self.performance_history[model_name][-20:]

                # Update pool manager performance tracking
                self.model_pool.update_model_performance(model_name, performance)

                # Update ensemble weights using exponential moving average
                recent_avg = np.mean(self.performance_history[model_name][-5:])
                old_weight = self.model_weights[model_name]

                # Adaptive learning rate based on performance consistency
                consistency = 1.0 - np.std(self.performance_history[model_name][-5:])
                learning_rate = 0.1 + 0.2 * consistency  # More consistent models get higher learning rate

                new_weight = old_weight + learning_rate * (recent_avg - old_weight)
                self.model_weights[model_name] = max(0.01, min(1.0, new_weight))  # Clamp to reasonable range

        # Normalize weights after updates
        self._normalize_weights()

        logger.info("Model weights updated based on performance feedback")
        logger.debug(f"Updated weights: {self.model_weights}")

    def get_ensemble_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of ensemble configuration"""
        model_summary = self.model_pool.get_model_summary()

        return {
            'ensemble_method': self.ensemble_method.value,
            'model_weights': self.model_weights.copy(),
            'performance_history_length': {
                name: len(history) for name, history in self.performance_history.items()
            },
            'recent_performance_averages': {
                name: np.mean(history[-5:]) if history else 0.5
                for name, history in self.performance_history.items()
            },
            'model_pool_summary': model_summary,
            'available_methods': [method.value for method in EnsembleMethod],
            'prediction_count': self.prediction_count,
            'ensemble_version': '2.0'
        }

    def set_ensemble_method(self, method: str) -> bool:
        """Set the ensemble method to use"""
        try:
            new_method = EnsembleMethod(method)
            old_method = self.ensemble_method
            self.ensemble_method = new_method
            logger.info(f"Ensemble method changed from {old_method.value} to {new_method.value}")
            return True
        except ValueError:
            logger.error(f"Invalid ensemble method: {method}")
            logger.info(f"Available methods: {[method.value for method in EnsembleMethod]}")
            return False
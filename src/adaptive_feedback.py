"""
Phase 4: Adaptive Feedback System for SHIOL+

This module implements the adaptive feedback system that learns from prediction performance
and continuously improves the scoring system through various optimization algorithms.

Key Components:
- AdaptiveValidator: Extends BasicValidator with adaptive capabilities
- ModelFeedbackEngine: Core adaptive learning engine
- WeightOptimizer: Optimizes scoring component weights
- PatternAnalyzer: Identifies winning patterns
- ReliablePlayTracker: Monitors high-performing combinations
- AdaptivePlayScorer: Extends PlayScorer with adaptive weights
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from loguru import logger
import json
from scipy.optimize import minimize, differential_evolution
from sklearn.metrics import accuracy_score, precision_score, recall_score
import hashlib

from src.basic_validator import BasicValidator
from src.intelligent_generator import PlayScorer
from src.evaluator import Evaluator
import src.database as db


class AdaptiveValidator(BasicValidator):
    """
    Extends BasicValidator with adaptive feedback capabilities.
    Tracks prediction performance and feeds back into the learning system.
    """
    
    def __init__(self):
        super().__init__()
        self.feedback_engine = None
        logger.info("AdaptiveValidator initialized for Phase 4 adaptive feedback system")
    
    def set_feedback_engine(self, feedback_engine: 'ModelFeedbackEngine'):
        """Sets the feedback engine for adaptive learning."""
        self.feedback_engine = feedback_engine
        logger.info("Feedback engine connected to AdaptiveValidator")
    
    def adaptive_validate_predictions(self, enable_learning: bool = True) -> str:
        """
        Enhanced validation that includes adaptive learning feedback.
        
        Args:
            enable_learning: Whether to enable adaptive learning from results
            
        Returns:
            Path to the validation results CSV file
        """
        logger.info("Starting adaptive validation with learning feedback...")
        
        try:
            # Run basic validation first
            csv_path = self.basic_validate_predictions()
            
            if not csv_path or not enable_learning:
                return csv_path
            
            # Load validation results for learning
            validation_df = pd.read_csv(csv_path)
            
            # Process results for adaptive learning
            self._process_validation_results(validation_df)
            
            logger.info("Adaptive validation completed with learning feedback")
            return csv_path
            
        except Exception as e:
            logger.error(f"Error during adaptive validation: {e}")
            raise
    
    def _process_validation_results(self, validation_df: pd.DataFrame):
        """
        Processes validation results and feeds back into the learning system.
        
        Args:
            validation_df: DataFrame with validation results
        """
        try:
            if self.feedback_engine is None:
                logger.warning("No feedback engine available for learning")
                return
            
            # Get predictions with their IDs for tracking
            predictions = self._get_predictions_with_ids()
            
            for _, result in validation_df.iterrows():
                # Find matching prediction
                matching_pred = self._find_matching_prediction(result, predictions)
                
                if matching_pred:
                    # Save performance tracking
                    self._save_performance_tracking(matching_pred, result)
                    
                    # Feed results to learning engine
                    self.feedback_engine.process_prediction_result(
                        prediction_id=matching_pred['id'],
                        actual_result=result,
                        prediction_data=matching_pred
                    )
            
            logger.info(f"Processed {len(validation_df)} validation results for adaptive learning")
            
        except Exception as e:
            logger.error(f"Error processing validation results for learning: {e}")
    
    def _get_predictions_with_ids(self) -> List[Dict]:
        """Retrieves predictions with their database IDs."""
        try:
            predictions_df = db.get_prediction_history(limit=100)
            return predictions_df.to_dict('records')
        except Exception as e:
            logger.error(f"Error retrieving predictions with IDs: {e}")
            return []
    
    def _find_matching_prediction(self, result: pd.Series, predictions: List[Dict]) -> Optional[Dict]:
        """Finds the prediction that matches the validation result."""
        try:
            result_numbers = [int(x) for x in result['numbers'].split('-')]
            result_pb = int(result['powerball'])
            
            for pred in predictions:
                pred_numbers = [pred['n1'], pred['n2'], pred['n3'], pred['n4'], pred['n5']]
                if pred_numbers == result_numbers and pred['powerball'] == result_pb:
                    return pred
            
            return None
            
        except Exception as e:
            logger.warning(f"Error matching prediction: {e}")
            return None
    
    def _save_performance_tracking(self, prediction: Dict, result: pd.Series):
        """Saves performance tracking data to the database."""
        try:
            actual_numbers = [int(x) for x in result['draw_numbers'].split('-')]
            actual_pb = int(result['draw_powerball'])
            
            # Calculate score accuracy (placeholder - would need actual score comparison)
            score_accuracy = 0.5  # This would be calculated based on actual vs predicted scores
            
            # Component accuracy (placeholder)
            component_accuracy = {
                'probability': 0.6,
                'diversity': 0.4,
                'historical': 0.5,
                'risk_adjusted': 0.3
            }
            
            db.save_performance_tracking(
                prediction_id=prediction['id'],
                draw_date=result['prediction_date'],
                actual_numbers=actual_numbers,
                actual_pb=actual_pb,
                matches_main=result['match_main'],
                matches_pb=result['match_pb'],
                prize_tier=result['prize_category'],
                score_accuracy=score_accuracy,
                component_accuracy=component_accuracy
            )
            
        except Exception as e:
            logger.error(f"Error saving performance tracking: {e}")


class ModelFeedbackEngine:
    """
    Core adaptive learning engine that processes prediction results and optimizes the system.
    """
    
    def __init__(self, historical_data: pd.DataFrame):
        self.historical_data = historical_data
        self.weight_optimizer = WeightOptimizer()
        self.pattern_analyzer = PatternAnalyzer(historical_data)
        self.reliable_play_tracker = ReliablePlayTracker()
        self.learning_rate = 0.1
        self.min_samples_for_learning = 10
        
        logger.info("ModelFeedbackEngine initialized for adaptive learning")
    
    def process_prediction_result(self, prediction_id: int, actual_result: pd.Series, 
                                prediction_data: Dict):
        """
        Processes a single prediction result for adaptive learning.
        
        Args:
            prediction_id: Database ID of the prediction
            actual_result: Actual draw result
            prediction_data: Original prediction data
        """
        try:
            # Update reliable play tracking
            self.reliable_play_tracker.update_play_performance(
                numbers=[prediction_data['n1'], prediction_data['n2'], prediction_data['n3'],
                        prediction_data['n4'], prediction_data['n5']],
                powerball=prediction_data['powerball'],
                result=actual_result
            )
            
            # Analyze patterns
            self.pattern_analyzer.analyze_result_pattern(actual_result, prediction_data)
            
            # Check if we have enough data for weight optimization
            performance_data = self._get_recent_performance_data()
            if len(performance_data) >= self.min_samples_for_learning:
                self._trigger_weight_optimization(performance_data)
            
            logger.debug(f"Processed prediction result for ID {prediction_id}")
            
        except Exception as e:
            logger.error(f"Error processing prediction result: {e}")
    
    def _get_recent_performance_data(self, days_back: int = 30) -> List[Dict]:
        """Retrieves recent performance data for analysis."""
        try:
            analytics = db.get_performance_analytics(days_back)
            return analytics
        except Exception as e:
            logger.error(f"Error retrieving performance data: {e}")
            return []
    
    def _trigger_weight_optimization(self, performance_data: Dict):
        """Triggers weight optimization based on performance data."""
        try:
            if performance_data.get('total_predictions', 0) < self.min_samples_for_learning:
                return
            
            # Get current weights
            current_weights = db.get_active_adaptive_weights()
            if not current_weights:
                current_weights = {
                    'weights': {'probability': 0.4, 'diversity': 0.25, 'historical': 0.2, 'risk_adjusted': 0.15}
                }
            
            # Optimize weights based on performance
            optimized_weights = self.weight_optimizer.optimize_weights(
                current_weights['weights'],
                performance_data,
                algorithm='differential_evolution'
            )
            
            if optimized_weights:
                # Save optimized weights
                weight_set_name = f"adaptive_weights_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                performance_score = performance_data.get('avg_accuracy', 0.0)
                
                db.save_adaptive_weights(
                    weight_set_name=weight_set_name,
                    weights=optimized_weights,
                    performance_score=performance_score,
                    optimization_algorithm='differential_evolution',
                    dataset_hash=self._calculate_dataset_hash(),
                    is_active=True
                )
                
                logger.info(f"Weight optimization completed: {optimized_weights}")
            
        except Exception as e:
            logger.error(f"Error in weight optimization: {e}")
    
    def _calculate_dataset_hash(self) -> str:
        """Calculates hash of the current dataset."""
        try:
            dataset_str = str(len(self.historical_data)) + str(self.historical_data.iloc[-1].to_dict())
            return hashlib.sha256(dataset_str.encode()).hexdigest()[:16]
        except Exception as e:
            logger.warning(f"Error calculating dataset hash: {e}")
            return "unknown_hash"
    
    def generate_feedback_report(self) -> Dict:
        """Generates a comprehensive feedback report."""
        try:
            # Get performance analytics
            analytics = db.get_performance_analytics(30)
            
            # Get pattern analysis results
            patterns = self.pattern_analyzer.get_recent_patterns()
            
            # Get reliable plays
            reliable_plays = db.get_reliable_plays(limit=10)
            
            # Get current weights
            current_weights = db.get_active_adaptive_weights()
            
            report = {
                'timestamp': datetime.now().isoformat(),
                'performance_analytics': analytics,
                'identified_patterns': patterns,
                'reliable_plays_count': len(reliable_plays),
                'current_weights': current_weights,
                'learning_status': {
                    'total_predictions_analyzed': analytics.get('total_predictions', 0),
                    'learning_active': analytics.get('total_predictions', 0) >= self.min_samples_for_learning,
                    'last_optimization': current_weights.get('weight_set_name', 'None') if current_weights else 'None'
                }
            }
            
            logger.info("Generated comprehensive feedback report")
            return report
            
        except Exception as e:
            logger.error(f"Error generating feedback report: {e}")
            return {}


class WeightOptimizer:
    """
    Optimizes scoring component weights using various optimization algorithms.
    """
    
    def __init__(self):
        self.algorithms = {
            'differential_evolution': self._differential_evolution_optimize,
            'scipy_minimize': self._scipy_minimize_optimize,
            'grid_search': self._grid_search_optimize
        }
        logger.info("WeightOptimizer initialized with multiple algorithms")
    
    def optimize_weights(self, current_weights: Dict[str, float], performance_data: Dict,
                        algorithm: str = 'differential_evolution') -> Optional[Dict[str, float]]:
        """
        Optimizes weights using the specified algorithm.
        
        Args:
            current_weights: Current weight configuration
            performance_data: Performance data for optimization
            algorithm: Optimization algorithm to use
            
        Returns:
            Optimized weights or None if optimization failed
        """
        try:
            if algorithm not in self.algorithms:
                logger.error(f"Unknown optimization algorithm: {algorithm}")
                return None
            
            optimizer_func = self.algorithms[algorithm]
            optimized_weights = optimizer_func(current_weights, performance_data)
            
            if optimized_weights:
                logger.info(f"Weight optimization successful using {algorithm}")
                return optimized_weights
            else:
                logger.warning(f"Weight optimization failed using {algorithm}")
                return None
                
        except Exception as e:
            logger.error(f"Error in weight optimization: {e}")
            return None
    
    def _differential_evolution_optimize(self, current_weights: Dict[str, float], 
                                       performance_data: Dict) -> Optional[Dict[str, float]]:
        """Optimizes weights using differential evolution."""
        try:
            # Define bounds for each weight (must sum to 1.0)
            bounds = [(0.1, 0.7), (0.1, 0.5), (0.1, 0.4), (0.05, 0.3)]  # prob, div, hist, risk
            
            def objective_function(weights):
                # Normalize weights to sum to 1.0
                normalized_weights = weights / np.sum(weights)
                
                # Calculate objective based on performance metrics
                # Higher accuracy and win rate are better
                accuracy_score = performance_data.get('avg_accuracy', 0.0)
                win_rate = performance_data.get('win_rate', 0.0)
                
                # Penalize extreme weight distributions
                weight_entropy = -np.sum(normalized_weights * np.log(normalized_weights + 1e-10))
                
                # Objective: maximize accuracy and win rate, encourage balanced weights
                objective = -(accuracy_score * 0.6 + win_rate * 0.3 + weight_entropy * 0.1)
                return objective
            
            # Run optimization
            result = differential_evolution(
                objective_function,
                bounds,
                maxiter=50,
                popsize=10,
                seed=42
            )
            
            if result.success:
                optimized = result.x / np.sum(result.x)  # Normalize
                return {
                    'probability': float(optimized[0]),
                    'diversity': float(optimized[1]),
                    'historical': float(optimized[2]),
                    'risk_adjusted': float(optimized[3])
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error in differential evolution optimization: {e}")
            return None
    
    def _scipy_minimize_optimize(self, current_weights: Dict[str, float], 
                               performance_data: Dict) -> Optional[Dict[str, float]]:
        """Optimizes weights using scipy minimize."""
        try:
            # Convert current weights to array
            x0 = np.array([current_weights['probability'], current_weights['diversity'],
                          current_weights['historical'], current_weights['risk_adjusted']])
            
            def objective_function(weights):
                normalized_weights = weights / np.sum(weights)
                accuracy_score = performance_data.get('avg_accuracy', 0.0)
                win_rate = performance_data.get('win_rate', 0.0)
                return -(accuracy_score * 0.7 + win_rate * 0.3)
            
            # Constraint: weights must sum to 1
            constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}
            bounds = [(0.05, 0.8) for _ in range(4)]
            
            result = minimize(
                objective_function,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
            
            if result.success:
                return {
                    'probability': float(result.x[0]),
                    'diversity': float(result.x[1]),
                    'historical': float(result.x[2]),
                    'risk_adjusted': float(result.x[3])
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error in scipy minimize optimization: {e}")
            return None
    
    def _grid_search_optimize(self, current_weights: Dict[str, float], 
                            performance_data: Dict) -> Optional[Dict[str, float]]:
        """Optimizes weights using grid search."""
        try:
            best_weights = None
            best_score = -float('inf')
            
            # Define grid ranges
            prob_range = np.arange(0.2, 0.7, 0.1)
            div_range = np.arange(0.1, 0.4, 0.1)
            hist_range = np.arange(0.1, 0.4, 0.1)
            
            for prob_w in prob_range:
                for div_w in div_range:
                    for hist_w in hist_range:
                        risk_w = 1.0 - prob_w - div_w - hist_w
                        
                        if risk_w < 0.05 or risk_w > 0.4:
                            continue
                        
                        weights = {
                            'probability': prob_w,
                            'diversity': div_w,
                            'historical': hist_w,
                            'risk_adjusted': risk_w
                        }
                        
                        # Calculate score for this weight combination
                        score = self._evaluate_weight_combination(weights, performance_data)
                        
                        if score > best_score:
                            best_score = score
                            best_weights = weights
            
            return best_weights
            
        except Exception as e:
            logger.error(f"Error in grid search optimization: {e}")
            return None
    
    def _evaluate_weight_combination(self, weights: Dict[str, float], 
                                   performance_data: Dict) -> float:
        """Evaluates a weight combination based on performance data."""
        try:
            accuracy = performance_data.get('avg_accuracy', 0.0)
            win_rate = performance_data.get('win_rate', 0.0)
            
            # Simple scoring function
            score = accuracy * 0.7 + win_rate * 0.3
            
            # Bonus for balanced weights
            weight_values = list(weights.values())
            weight_std = np.std(weight_values)
            balance_bonus = max(0, 0.1 - weight_std)  # Bonus for lower standard deviation
            
            return score + balance_bonus
            
        except Exception as e:
            logger.warning(f"Error evaluating weight combination: {e}")
            return 0.0


class PatternAnalyzer:
    """
    Identifies and analyzes winning patterns in lottery data.
    """
    
    def __init__(self, historical_data: pd.DataFrame):
        self.historical_data = historical_data
        self.pattern_types = [
            'consecutive_numbers',
            'parity_distribution',
            'range_distribution',
            'sum_patterns',
            'frequency_patterns'
        ]
        logger.info("PatternAnalyzer initialized for pattern identification")
    
    def analyze_result_pattern(self, actual_result: pd.Series, prediction_data: Dict):
        """
        Analyzes patterns in a single result.
        
        Args:
            actual_result: Actual draw result
            prediction_data: Original prediction data
        """
        try:
            actual_numbers = [int(x) for x in actual_result['draw_numbers'].split('-')]
            actual_pb = int(actual_result['draw_powerball'])
            
            # Analyze each pattern type
            for pattern_type in self.pattern_types:
                pattern_data = self._analyze_pattern_type(pattern_type, actual_numbers, actual_pb)
                
                if pattern_data:
                    # Save pattern analysis
                    db.save_pattern_analysis(
                        pattern_type=pattern_type,
                        pattern_description=pattern_data['description'],
                        pattern_data=pattern_data['data'],
                        success_rate=pattern_data['success_rate'],
                        frequency=pattern_data['frequency'],
                        confidence_score=pattern_data['confidence'],
                        date_range_start=actual_result['prediction_date'],
                        date_range_end=actual_result['prediction_date']
                    )
            
        except Exception as e:
            logger.error(f"Error analyzing result pattern: {e}")
    
    def _analyze_pattern_type(self, pattern_type: str, numbers: List[int], 
                            powerball: int) -> Optional[Dict]:
        """Analyzes a specific pattern type."""
        try:
            if pattern_type == 'consecutive_numbers':
                return self._analyze_consecutive_pattern(numbers)
            elif pattern_type == 'parity_distribution':
                return self._analyze_parity_pattern(numbers)
            elif pattern_type == 'range_distribution':
                return self._analyze_range_pattern(numbers)
            elif pattern_type == 'sum_patterns':
                return self._analyze_sum_pattern(numbers)
            elif pattern_type == 'frequency_patterns':
                return self._analyze_frequency_pattern(numbers, powerball)
            
            return None
            
        except Exception as e:
            logger.warning(f"Error analyzing pattern type {pattern_type}: {e}")
            return None
    
    def _analyze_consecutive_pattern(self, numbers: List[int]) -> Dict:
        """Analyzes consecutive number patterns."""
        sorted_numbers = sorted(numbers)
        consecutive_count = 0
        
        for i in range(len(sorted_numbers) - 1):
            if sorted_numbers[i + 1] - sorted_numbers[i] == 1:
                consecutive_count += 1
        
        return {
            'description': f"Contains {consecutive_count} consecutive number pairs",
            'data': {'consecutive_pairs': consecutive_count, 'numbers': sorted_numbers},
            'success_rate': 0.5,  # Would be calculated from historical data
            'frequency': consecutive_count,
            'confidence': 0.7
        }
    
    def _analyze_parity_pattern(self, numbers: List[int]) -> Dict:
        """Analyzes even/odd distribution patterns."""
        even_count = sum(1 for num in numbers if num % 2 == 0)
        odd_count = len(numbers) - even_count
        
        return {
            'description': f"Even/Odd distribution: {even_count}E-{odd_count}O",
            'data': {'even_count': even_count, 'odd_count': odd_count},
            'success_rate': 0.6,
            'frequency': max(even_count, odd_count),
            'confidence': 0.8
        }
    
    def _analyze_range_pattern(self, numbers: List[int]) -> Dict:
        """Analyzes number range distribution patterns."""
        range1 = sum(1 for num in numbers if 1 <= num <= 23)
        range2 = sum(1 for num in numbers if 24 <= num <= 46)
        range3 = sum(1 for num in numbers if 47 <= num <= 69)
        
        return {
            'description': f"Range distribution: {range1}-{range2}-{range3}",
            'data': {'low_range': range1, 'mid_range': range2, 'high_range': range3},
            'success_rate': 0.55,
            'frequency': max(range1, range2, range3),
            'confidence': 0.6
        }
    
    def _analyze_sum_pattern(self, numbers: List[int]) -> Dict:
        """Analyzes sum patterns."""
        total_sum = sum(numbers)
        
        return {
            'description': f"Sum total: {total_sum}",
            'data': {'sum': total_sum, 'avg': total_sum / len(numbers)},
            'success_rate': 0.5,
            'frequency': 1,
            'confidence': 0.4
        }
    
    def _analyze_frequency_pattern(self, numbers: List[int], powerball: int) -> Dict:
        """Analyzes frequency patterns based on historical data."""
        # Calculate historical frequencies
        freq_data = {}
        for num in numbers:
            freq_data[num] = self._get_historical_frequency(num, 'white_ball')
        
        freq_data[f'pb_{powerball}'] = self._get_historical_frequency(powerball, 'powerball')
        
        avg_frequency = np.mean(list(freq_data.values()))
        
        return {
            'description': f"Average historical frequency: {avg_frequency:.2f}",
            'data': freq_data,
            'success_rate': 0.4,
            'frequency': int(avg_frequency),
            'confidence': 0.5
        }
    
    def _get_historical_frequency(self, number: int, ball_type: str) -> int:
        """Gets historical frequency of a number."""
        try:
            if ball_type == 'white_ball':
                cols = ['n1', 'n2', 'n3', 'n4', 'n5']
                count = 0
                for col in cols:
                    if col in self.historical_data.columns:
                        count += (self.historical_data[col] == number).sum()
                return count
            elif ball_type == 'powerball':
                if 'pb' in self.historical_data.columns:
                    return (self.historical_data['pb'] == number).sum()
            
            return 0
            
        except Exception as e:
            logger.warning(f"Error getting historical frequency: {e}")
            return 0
    
    def get_recent_patterns(self, days_back: int = 30) -> List[Dict]:
        """Retrieves recent pattern analysis results."""
        try:
            # This would query the pattern_analysis table
            # For now, return a placeholder
            return [
                {
                    'pattern_type': 'consecutive_numbers',
                    'success_rate': 0.6,
                    'frequency': 15,
                    'confidence': 0.7
                },
                {
                    'pattern_type': 'parity_distribution',
                    'success_rate': 0.8,
                    'frequency': 25,
                    'confidence': 0.9
                }
            ]
            
        except Exception as e:
            logger.error(f"Error retrieving recent patterns: {e}")
            return []


class ReliablePlayTracker:
    """
    Monitors and tracks high-performing play combinations.
    """
    
    def __init__(self):
        self.min_reliability_threshold = 0.6
        self.tracking_window_days = 90
        logger.info("ReliablePlayTracker initialized for monitoring high-performing plays")
    
    def update_play_performance(self, numbers: List[int], powerball: int, result: pd.Series):
        """
        Updates performance tracking for a play combination.
        
        Args:
            numbers: List of 5 main numbers
            powerball: Powerball number
            result: Actual draw result
        """
        try:
            # Calculate performance metrics
            win_rate = 1.0 if result['prize_category'] != 'Non-winning' else 0.0
            reliability_score = self._calculate_reliability_score(numbers, powerball, result)
            
            # Create performance history entry
            performance_history = {
                'date': result['prediction_date'],
                'matches_main': result['match_main'],
                'matches_pb': result['match_pb'],
                'prize_tier': result['prize_category'],
                'win_rate': win_rate
            }
            
            # Calculate average score (placeholder)
            avg_score = 0.5  # This would be calculated from actual scoring data
            
            # Save or update reliable play
            db.save_reliable_play(
                numbers=numbers,
                powerball=powerball,
                reliability_score=reliability_score,
                performance_history=performance_history,
                win_rate=win_rate,
                avg_score=avg_score
            )
            
            logger.debug(f"Updated performance for play: {numbers} + {powerball}")
            
        except Exception as e:
            logger.error(f"Error updating play performance: {e}")
    
    def _calculate_reliability_score(self, numbers: List[int], powerball: int, 
                                   result: pd.Series) -> float:
        """
        Calculates reliability score for a play combination.
        
        Args:
            numbers: List of main numbers
            powerball: Powerball number
            result: Draw result
            
        Returns:
            Reliability score between 0.0 and 1.0
        """
        try:
            score_components = []
            
            # Component 1: Match performance (40%)
            match_score = (result['match_main'] / 5.0) * 0.8 + (result['match_pb'] / 1.0) * 0.2
            score_components.append(match_score * 0.4)
            
            # Component 2: Prize tier achievement (30%)
            prize_scores = {
                'Jackpot': 1.0, 'Match 5': 0.9, 'Match 4 + PB': 0.8, 'Match 4': 0.6,
                'Match 3 + PB': 0.5, 'Match 3': 0.3, 'Match 2 + PB': 0.2,
                'Match 1 + PB': 0.1, 'Match PB': 0.05, 'Non-winning': 0.0
            }
            prize_score = prize_scores.get(result['prize_category'], 0.0)
            score_components.append(prize_score * 0.3)
            
            # Component 3: Number diversity (20%)
            diversity_score = self._calculate_diversity_score(numbers)
            score_components.append(diversity_score * 0.2)
            
            # Component 4: Historical performance (10%)
            historical_score = self._calculate_historical_score(numbers, powerball)
            score_components.append(historical_score * 0.1)
            
            reliability_score = sum(score_components)
            return min(1.0, max(0.0, reliability_score))
            
        except Exception as e:
            logger.warning(f"Error calculating reliability score: {e}")
            return 0.5
    
    def _calculate_diversity_score(self, numbers: List[int]) -> float:
        """Calculates diversity score for number combination."""
        try:
            # Even/odd balance
            even_count = sum(1 for num in numbers if num % 2 == 0)
            parity_score = 1.0 - abs(even_count - 2.5) / 2.5
            
            # Range distribution
            ranges = [0, 0, 0]  # low, mid, high
            for num in numbers:
                if num <= 23:
                    ranges[0] += 1
                elif num <= 46:
                    ranges[1] += 1
                else:
                    ranges[2] += 1
            
            range_score = 1.0 - max(ranges) / 5.0  # Penalize concentration
            
            # Spread
            spread = max(numbers) - min(numbers)
            spread_score = min(1.0, spread / 50.0)
            
            return (parity_score + range_score + spread_score) / 3.0
            
        except Exception as e:
            logger.warning(f"Error calculating diversity score: {e}")
            return 0.5
    
    def _calculate_historical_score(self, numbers: List[int], powerball: int) -> float:
        """Calculates historical performance score."""
        try:
            # This would analyze historical performance of similar combinations
            # For now, return a placeholder score
            return 0.5
            
        except Exception as e:
            logger.warning(f"Error calculating historical score: {e}")
            return 0.5
    
    def get_top_reliable_plays(self, limit: int = 10) -> pd.DataFrame:
        """
        Retrieves top reliable plays.
        
        Args:
            limit: Maximum number of plays to return
            
        Returns:
            DataFrame with top reliable plays
        """
        try:
            return db.get_reliable_plays(limit=limit, min_reliability_score=self.min_reliability_threshold)
            
        except Exception as e:
            logger.error(f"Error retrieving top reliable plays: {e}")
            return pd.DataFrame()


class AdaptivePlayScorer(PlayScorer):
    """
    Extends PlayScorer with adaptive weight capabilities.
    Uses dynamically optimized weights instead of fixed weights.
    """
    
    def __init__(self, historical_data: pd.DataFrame):
        super().__init__(historical_data)
        self.adaptive_weights = None
        self.load_adaptive_weights()
        logger.info("AdaptivePlayScorer initialized with adaptive weight system")
    
    def load_adaptive_weights(self):
        """Loads the currently active adaptive weights."""
        try:
            active_weights = db.get_active_adaptive_weights()
            if active_weights:
                self.adaptive_weights = active_weights['weights']
                logger.info(f"Loaded adaptive weights: {self.adaptive_weights}")
            else:
                logger.info("No active adaptive weights found, using default weights")
                self.adaptive_weights = self.weights.copy()
                
        except Exception as e:
            logger.error(f"Error loading adaptive weights: {e}")
            self.adaptive_weights = self.weights.copy()
    
    def calculate_total_score(self, white_balls: List[int], powerball: int,
                            wb_probs: Dict[int, float], pb_probs: Dict[int, float]) -> Dict:
        """
        Calculates total score using adaptive weights.
        
        Args:
            white_balls: List of 5 numbers blancos
            powerball: Número del powerball
            wb_probs: Probabilidades de números blancos
            pb_probs: Probabilidades de powerball
            
        Returns:
            Dict con scores individuales y total usando pesos adaptativos
        """
        try:
            # Use adaptive weights if available
            current_weights = self.adaptive_weights if self.adaptive_weights else self.weights
            
            scores = {}
            
            # Calculate each scoring component (same as parent class)
            scores['probability'] = self._calculate_probability_score(white_balls, powerball, wb_probs, pb_probs)
            scores['diversity'] = self._calculate_diversity_score(white_balls, powerball)
            scores['historical'] = self._calculate_historical_score(white_balls, powerball)
            scores['risk_adjusted'] = self._calculate_risk_adjusted_score(white_balls, powerball)
            
            # Calculate total score using adaptive weights
            total_score = sum(scores[component] * current_weights[component]
                            for component in scores.keys())
            
            scores['total'] = total_score
            scores['weights_used'] = current_weights.copy()
            scores['adaptive_mode'] = self.adaptive_weights is not None
            
            return scores
            
        except Exception as e:
            logger.error(f"Error calculating adaptive total score: {e}")
            # Fallback to parent class method
            return super().calculate_total_score(white_balls, powerball, wb_probs, pb_probs)
    
    def update_weights(self, new_weights: Dict[str, float]):
        """
        Updates the adaptive weights.
        
        Args:
            new_weights: New weight configuration
        """
        try:
            self.adaptive_weights = new_weights.copy()
            logger.info(f"Updated adaptive weights: {new_weights}")
            
        except Exception as e:
            logger.error(f"Error updating adaptive weights: {e}")
    
    def get_weight_performance_analysis(self) -> Dict:
        """
        Analyzes the performance of current weights.
        
        Returns:
            Dict with weight performance analysis
        """
        try:
            current_weights = self.adaptive_weights if self.adaptive_weights else self.weights
            
            # Get recent performance data
            performance_data = db.get_performance_analytics(30)
            
            analysis = {
                'current_weights': current_weights,
                'performance_metrics': performance_data,
                'weight_balance': {
                    'max_weight': max(current_weights.values()),
                    'min_weight': min(current_weights.values()),
                    'weight_std': np.std(list(current_weights.values())),
                    'is_balanced': max(current_weights.values()) - min(current_weights.values()) < 0.4
                },
                'recommendations': self._generate_weight_recommendations(current_weights, performance_data)
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing weight performance: {e}")
            return {}
    
    def _generate_weight_recommendations(self, weights: Dict[str, float],
                                       performance_data: Dict) -> List[str]:
        """Generates recommendations for weight adjustments."""
        try:
            recommendations = []
            
            # Check if performance is below threshold
            avg_accuracy = performance_data.get('avg_accuracy', 0.0)
            win_rate = performance_data.get('win_rate', 0.0)
            
            if avg_accuracy < 0.3:
                recommendations.append("Consider increasing probability weight for better accuracy")
            
            if win_rate < 0.05:
                recommendations.append("Consider adjusting diversity and historical weights")
            
            # Check weight balance
            weight_values = list(weights.values())
            if max(weight_values) - min(weight_values) > 0.5:
                recommendations.append("Weights are highly imbalanced, consider rebalancing")
            
            if not recommendations:
                recommendations.append("Current weight configuration appears optimal")
            
            return recommendations
            
        except Exception as e:
            logger.warning(f"Error generating weight recommendations: {e}")
            return ["Unable to generate recommendations due to error"]


# Utility functions for the adaptive feedback system

def initialize_adaptive_system(historical_data: pd.DataFrame) -> Dict[str, Any]:
    """
    Initializes the complete adaptive feedback system.
    
    Args:
        historical_data: Historical lottery data
        
    Returns:
        Dict with initialized system components
    """
    try:
        logger.info("Initializing Phase 4 Adaptive Feedback System...")
        
        # Initialize core components
        adaptive_validator = AdaptiveValidator()
        feedback_engine = ModelFeedbackEngine(historical_data)
        adaptive_scorer = AdaptivePlayScorer(historical_data)
        
        # Connect components
        adaptive_validator.set_feedback_engine(feedback_engine)
        
        # Initialize database tables
        db.initialize_database()
        
        system_components = {
            'adaptive_validator': adaptive_validator,
            'feedback_engine': feedback_engine,
            'adaptive_scorer': adaptive_scorer,
            'weight_optimizer': feedback_engine.weight_optimizer,
            'pattern_analyzer': feedback_engine.pattern_analyzer,
            'reliable_play_tracker': feedback_engine.reliable_play_tracker
        }
        
        logger.info("Adaptive feedback system initialized successfully")
        return system_components
        
    except Exception as e:
        logger.error(f"Error initializing adaptive feedback system: {e}")
        raise


def run_adaptive_analysis(days_back: int = 30) -> Dict:
    """
    Runs a comprehensive adaptive analysis.
    
    Args:
        days_back: Number of days to analyze
        
    Returns:
        Dict with analysis results
    """
    try:
        logger.info(f"Running adaptive analysis for {days_back} days...")
        
        # Get performance analytics
        performance_data = db.get_performance_analytics(days_back)
        
        # Get reliable plays
        reliable_plays = db.get_reliable_plays(limit=20)
        
        # Get current adaptive weights
        current_weights = db.get_active_adaptive_weights()
        
        analysis_results = {
            'analysis_period': f"{days_back} days",
            'timestamp': datetime.now().isoformat(),
            'performance_summary': performance_data,
            'reliable_plays_count': len(reliable_plays),
            'top_reliable_plays': reliable_plays.head(5).to_dict('records') if not reliable_plays.empty else [],
            'current_adaptive_weights': current_weights,
            'system_status': {
                'adaptive_learning_active': current_weights is not None,
                'total_predictions_analyzed': performance_data.get('total_predictions', 0),
                'learning_threshold_met': performance_data.get('total_predictions', 0) >= 10
            }
        }
        
        logger.info("Adaptive analysis completed successfully")
        return analysis_results
        
    except Exception as e:
        logger.error(f"Error running adaptive analysis: {e}")
        return {'error': str(e)}


class AdaptivePlayScorer:
    """
    Extends PlayScorer with adaptive weight capabilities.
    Uses dynamically optimized weights instead of fixed weights.
    """
    
    def __init__(self, historical_data: pd.DataFrame):
        super().__init__(historical_data)
        self.adaptive_weights = None
        self.load_adaptive_weights()
        logger.info("AdaptivePlayScorer initialized with adaptive weight system")
    
    def load_adaptive_weights(self):
        """Loads the currently active adaptive weights."""
        try:
            active_weights = db.get_active_adaptive_weights()
            if active_weights:
                self.adaptive_weights = active_weights['weights']
                logger.info(f"Loaded adaptive weights: {self.adaptive_weights}")
            else:
                logger.info("No active adaptive weights found, using default weights")
                self.adaptive_weights = self.weights.copy()
                
        except Exception as e:
            logger.error(f"Error loading adaptive weights: {e}")
            self.adaptive_weights = self.weights.copy()
    
    def calculate_total_score(self, white_balls: List[int], powerball: int,
                            wb_probs: Dict[int, float], pb_probs: Dict[int, float]) -> Dict:
        """
        Calculates total score using adaptive weights.
        
        Args:
            white_balls: List of 5 numbers blancos
            powerball: Número del powerball
            wb_probs: Probabilidades de números blancos
            pb_probs: Probabilidades de powerball
            
        Returns:
            Dict con scores individuales y total usando pesos adaptativos
        """
        try:
            # Use adaptive weights if available
            current_weights = self.adaptive_weights if self.adaptive_weights else self.weights
            
            scores = {}
            
            # Calculate each scoring component (same as parent class)
            scores['probability'] = self._calculate_probability_score(white_balls, powerball, wb_probs, pb_probs)
            scores['diversity'] = self._calculate_diversity_score(white_balls, powerball)
            scores['historical'] = self._calculate_historical_score(white_balls, powerball)
            scores['risk_adjusted'] = self._calculate_risk_adjusted_score(white_balls, powerball)
            
            # Calculate total score using adaptive weights
            total_score = sum(scores[component] * current_weights[component]
                            for component in scores.keys())
            
            scores['total'] = total_score
            scores['weights_used'] = current_weights.copy()
            scores['adaptive_mode'] = self.adaptive_weights is not None
            
            return scores
            
        except Exception as e:
            logger.error(f"Error calculating adaptive total score: {e}")
            # Fallback to parent class method
            return super().calculate_total_score(white_balls, powerball, wb_probs, pb_probs)
    
    def update_weights(self, new_weights: Dict[str, float]):
        """
        Updates the adaptive weights.
        
        Args:
            new_weights: New weight configuration
        """
        try:
            self.adaptive_weights = new_weights.copy()
            logger.info(f"Updated adaptive weights: {new_weights}")
            
        except Exception as e:
            logger.error(f"Error updating adaptive weights: {e}")
    
    def get_weight_performance_analysis(self) -> Dict:
        """
        Analyzes the performance of current weights.
        
        Returns:
            Dict with weight performance analysis
        """
        try:
            current_weights = self.adaptive_weights if self.adaptive_weights else self.weights
            
            # Get recent performance data
            performance_data = db.get_performance_analytics(30)
            
            analysis = {
                'current_weights': current_weights,
                'performance_metrics': performance_data,
                'weight_balance': {
                    'max_weight': max(current_weights.values()),
                    'min_weight': min(current_weights.values()),
                    'weight_std': np.std(list(current_weights.values())),
                    'is_balanced': max(current_weights.values()) - min(current_weights.values()) < 0.4
                },
                'recommendations': self._generate_weight_recommendations(current_weights, performance_data)
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing weight performance: {e}")
            return {}
    
    def _generate_weight_recommendations(self, weights: Dict[str, float],
                                       performance_data: Dict) -> List[str]:
        """Generates recommendations for weight adjustments."""
        try:
            recommendations = []
            
            # Check if performance is below threshold
            avg_accuracy = performance_data.get('avg_accuracy', 0.0)
            win_rate = performance_data.get('win_rate', 0.0)
            
            if avg_accuracy < 0.3:
                recommendations.append("Consider increasing probability weight for better accuracy")
            
            if win_rate < 0.05:
                recommendations.append("Consider adjusting diversity and historical weights")
            
            # Check weight balance
            weight_values = list(weights.values())
            if max(weight_values) - min(weight_values) > 0.5:
                recommendations.append("Weights are highly imbalanced, consider rebalancing")
            
            if not recommendations:
                recommendations.append("Current weight configuration appears optimal")
            
            return recommendations
            
        except Exception as e:
            logger.warning(f"Error generating weight recommendations: {e}")
            return ["Unable to generate recommendations due to error"]

    def generate_adaptive_predictions(self, n_samples: int) -> List[List[int]]:
        """
        Genera combinaciones utilizando pesos adaptativos optimizados.
        """
        logger.info("Generando combinaciones adaptativas con AdaptivePlayScorer...")
        # Implementar lógica adaptativa mejorada aquí
        return [[7, 8, 9, 10, 11, 12] for _ in range(n_samples)]  # Ejemplo


# Utility functions for the adaptive feedback system

def initialize_adaptive_system(historical_data: pd.DataFrame) -> Dict[str, Any]:
    """
    Initializes the complete adaptive feedback system.
    
    Args:
        historical_data: Historical lottery data
        
    Returns:
        Dict with initialized system components
    """
    try:
        logger.info("Initializing Phase 4 Adaptive Feedback System...")
        
        # Initialize core components
        adaptive_validator = AdaptiveValidator()
        feedback_engine = ModelFeedbackEngine(historical_data)
        adaptive_scorer = AdaptivePlayScorer(historical_data)
        
        # Connect components
        adaptive_validator.set_feedback_engine(feedback_engine)
        
        # Initialize database tables
        db.initialize_database()
        
        system_components = {
            'adaptive_validator': adaptive_validator,
            'feedback_engine': feedback_engine,
            'adaptive_scorer': adaptive_scorer,
            'weight_optimizer': feedback_engine.weight_optimizer,
            'pattern_analyzer': feedback_engine.pattern_analyzer,
            'reliable_play_tracker': feedback_engine.reliable_play_tracker
        }
        
        logger.info("Adaptive feedback system initialized successfully")
        return system_components
        
    except Exception as e:
        logger.error(f"Error initializing adaptive feedback system: {e}")
        raise


def run_adaptive_analysis(days_back: int = 30) -> Dict:
    """
    Runs a comprehensive adaptive analysis.
    
    Args:
        days_back: Number of days to analyze
        
    Returns:
        Dict with analysis results
    """
    try:
        logger.info(f"Running adaptive analysis for {days_back} days...")
        
        # Get performance analytics
        performance_data = db.get_performance_analytics(days_back)
        
        # Get reliable plays
        reliable_plays = db.get_reliable_plays(limit=20)
        
        # Get current adaptive weights
        current_weights = db.get_active_adaptive_weights()
        
        analysis_results = {
            'analysis_period': f"{days_back} days",
            'timestamp': datetime.now().isoformat(),
            'performance_summary': performance_data,
            'reliable_plays_count': len(reliable_plays),
            'top_reliable_plays': reliable_plays.head(5).to_dict('records') if not reliable_plays.empty else [],
            'current_adaptive_weights': current_weights,
            'system_status': {
                'adaptive_learning_active': current_weights is not None,
                'total_predictions_analyzed': performance_data.get('total_predictions', 0),
                'learning_threshold_met': performance_data.get('total_predictions', 0) >= 10
            }
        }
        
        logger.info("Adaptive analysis completed successfully")
        return analysis_results
        
    except Exception as e:
        logger.error(f"Error running adaptive analysis: {e}")
        return {'error': str(e)}
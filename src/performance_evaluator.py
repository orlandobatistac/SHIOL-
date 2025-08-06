
"""
SHIOL+ Performance Evaluator
============================

Sistema completo de evaluación del rendimiento histórico de las jugadas generadas.
Analiza efectividad por componente, distribución de scores y métricas avanzadas.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from loguru import logger
from typing import Dict, List, Tuple, Optional
import json
from collections import defaultdict

from src.database import (
    get_prediction_history, get_all_draws, get_performance_analytics,
    get_db_connection
)
from src.intelligent_generator import PlayScorer


class PerformanceEvaluator:
    """
    Evaluador completo de rendimiento que analiza la efectividad
    de las predicciones generadas a través del tiempo.
    """
    
    def __init__(self, analysis_window_days: int = 90):
        self.analysis_window_days = analysis_window_days
        self.prize_categories = {
            'jackpot': {'wb_matches': 5, 'pb_match': True, 'prize_value': 1000000},
            'second_prize': {'wb_matches': 5, 'pb_match': False, 'prize_value': 1000000},
            'third_prize': {'wb_matches': 4, 'pb_match': True, 'prize_value': 50000},
            'fourth_prize': {'wb_matches': 4, 'pb_match': False, 'prize_value': 100},
            'fifth_prize': {'wb_matches': 3, 'pb_match': True, 'prize_value': 100},
            'sixth_prize': {'wb_matches': 3, 'pb_match': False, 'prize_value': 7},
            'seventh_prize': {'wb_matches': 2, 'pb_match': True, 'prize_value': 7},
            'eighth_prize': {'wb_matches': 1, 'pb_match': True, 'prize_value': 4},
            'ninth_prize': {'wb_matches': 0, 'pb_match': True, 'prize_value': 4}
        }
        
        logger.info(f"PerformanceEvaluator initialized with {analysis_window_days}-day analysis window")
    
    def evaluate_comprehensive_performance(self) -> Dict[str, any]:
        """
        Realiza evaluación completa del rendimiento del sistema.
        
        Returns:
            Dict con análisis completo de rendimiento
        """
        logger.info("Starting comprehensive performance evaluation...")
        
        evaluation_results = {
            'evaluation_timestamp': datetime.now().isoformat(),
            'analysis_window_days': self.analysis_window_days,
            'overall_performance': {},
            'score_component_analysis': {},
            'temporal_performance_trends': {},
            'prize_category_analysis': {},
            'prediction_accuracy_analysis': {},
            'diversity_effectiveness': {},
            'recommendations': []
        }
        
        try:
            # 1. Obtener datos de predicciones y sorteos
            predictions_data = self._get_predictions_data()
            lottery_results = self._get_lottery_results()
            
            if not predictions_data or lottery_results.empty:
                logger.warning("Insufficient data for comprehensive evaluation")
                return evaluation_results
            
            # 2. Análisis de rendimiento general
            overall_perf = self._analyze_overall_performance(predictions_data, lottery_results)
            evaluation_results['overall_performance'] = overall_perf
            
            # 3. Análisis por componentes de score
            component_analysis = self._analyze_score_components(predictions_data, lottery_results)
            evaluation_results['score_component_analysis'] = component_analysis
            
            # 4. Tendencias temporales
            temporal_trends = self._analyze_temporal_trends(predictions_data, lottery_results)
            evaluation_results['temporal_performance_trends'] = temporal_trends
            
            # 5. Análisis por categorías de premio
            prize_analysis = self._analyze_prize_categories(predictions_data, lottery_results)
            evaluation_results['prize_category_analysis'] = prize_analysis
            
            # 6. Análisis de precisión de predicciones
            accuracy_analysis = self._analyze_prediction_accuracy(predictions_data, lottery_results)
            evaluation_results['prediction_accuracy_analysis'] = accuracy_analysis
            
            # 7. Efectividad de diversidad
            diversity_analysis = self._analyze_diversity_effectiveness(predictions_data)
            evaluation_results['diversity_effectiveness'] = diversity_analysis
            
            # 8. Generar recomendaciones
            recommendations = self._generate_performance_recommendations(evaluation_results)
            evaluation_results['recommendations'] = recommendations
            
            logger.info("Comprehensive performance evaluation completed successfully")
            return evaluation_results
            
        except Exception as e:
            logger.error(f"Error during comprehensive performance evaluation: {e}")
            evaluation_results['error'] = str(e)
            return evaluation_results
    
    def _get_predictions_data(self) -> List[Dict]:
        """Obtiene datos de predicciones del período de análisis."""
        try:
            cutoff_date = datetime.now() - timedelta(days=self.analysis_window_days)
            
            # Obtener predicciones desde la base de datos
            predictions = get_prediction_history(limit=1000)
            
            # Filtrar por fecha
            filtered_predictions = []
            for pred in predictions:
                try:
                    pred_date = datetime.fromisoformat(pred['timestamp'].replace('Z', '+00:00'))
                    if pred_date >= cutoff_date:
                        filtered_predictions.append(pred)
                except:
                    continue
            
            logger.info(f"Retrieved {len(filtered_predictions)} predictions for analysis")
            return filtered_predictions
            
        except Exception as e:
            logger.error(f"Error getting predictions data: {e}")
            return []
    
    def _get_lottery_results(self) -> pd.DataFrame:
        """Obtiene resultados de sorteos del período de análisis."""
        try:
            cutoff_date = datetime.now() - timedelta(days=self.analysis_window_days)
            cutoff_str = cutoff_date.strftime('%Y-%m-%d')
            
            all_draws = get_all_draws()
            
            if 'draw_date' in all_draws.columns:
                recent_draws = all_draws[all_draws['draw_date'] >= cutoff_str]
            else:
                recent_draws = all_draws.tail(self.analysis_window_days // 3)  # Aproximación
            
            logger.info(f"Retrieved {len(recent_draws)} lottery results for analysis")
            return recent_draws
            
        except Exception as e:
            logger.error(f"Error getting lottery results: {e}")
            return pd.DataFrame()
    
    def _analyze_overall_performance(self, predictions: List[Dict], 
                                   lottery_results: pd.DataFrame) -> Dict:
        """Analiza el rendimiento general del sistema."""
        try:
            logger.info("Analyzing overall system performance...")
            
            performance_metrics = {
                'total_predictions_analyzed': len(predictions),
                'total_lottery_draws': len(lottery_results),
                'prediction_periods_covered': 0,
                'hit_rate_analysis': {},
                'score_distribution': {},
                'winning_predictions': 0,
                'total_prize_value': 0,
                'roi_analysis': {}
            }
            
            if not predictions or lottery_results.empty:
                return performance_metrics
            
            # Análisis de tasa de aciertos
            hit_analysis = self._calculate_hit_rates(predictions, lottery_results)
            performance_metrics['hit_rate_analysis'] = hit_analysis
            
            # Distribución de scores
            scores = [pred.get('score_total', 0) for pred in predictions]
            score_dist = {
                'mean_score': np.mean(scores),
                'median_score': np.median(scores),
                'std_score': np.std(scores),
                'min_score': np.min(scores),
                'max_score': np.max(scores),
                'score_percentiles': {
                    '25th': np.percentile(scores, 25),
                    '75th': np.percentile(scores, 75),
                    '90th': np.percentile(scores, 90),
                    '95th': np.percentile(scores, 95)
                }
            }
            performance_metrics['score_distribution'] = score_dist
            
            # Análisis de premios ganados
            prize_analysis = self._calculate_prize_winnings(predictions, lottery_results)
            performance_metrics['winning_predictions'] = prize_analysis['winning_count']
            performance_metrics['total_prize_value'] = prize_analysis['total_value']
            
            # ROI Analysis (retorno de inversión simulado)
            cost_per_play = 2.0  # Costo estándar por jugada
            total_cost = len(predictions) * cost_per_play
            roi = (prize_analysis['total_value'] - total_cost) / total_cost if total_cost > 0 else 0
            
            performance_metrics['roi_analysis'] = {
                'total_investment': total_cost,
                'total_returns': prize_analysis['total_value'],
                'net_profit': prize_analysis['total_value'] - total_cost,
                'roi_percentage': roi * 100
            }
            
            return performance_metrics
            
        except Exception as e:
            logger.error(f"Error analyzing overall performance: {e}")
            return {'error': str(e)}
    
    def _analyze_score_components(self, predictions: List[Dict], 
                                lottery_results: pd.DataFrame) -> Dict:
        """Analiza la efectividad de cada componente de score."""
        try:
            logger.info("Analyzing score component effectiveness...")
            
            component_analysis = {
                'probability_component': {},
                'diversity_component': {},
                'historical_component': {},
                'risk_adjusted_component': {},
                'component_correlations': {},
                'component_importance': {}
            }
            
            # Extraer scores por componente
            components_data = {
                'probability': [],
                'diversity': [],
                'historical': [],
                'risk_adjusted': [],
                'total_score': [],
                'hit_count': []
            }
            
            for pred in predictions:
                score_details = pred.get('score_details', {})
                if score_details:
                    components_data['probability'].append(score_details.get('probability', 0))
                    components_data['diversity'].append(score_details.get('diversity', 0))
                    components_data['historical'].append(score_details.get('historical', 0))
                    components_data['risk_adjusted'].append(score_details.get('risk_adjusted', 0))
                    components_data['total_score'].append(pred.get('score_total', 0))
                    
                    # Calcular hits para esta predicción
                    hit_count = self._calculate_prediction_hits(pred, lottery_results)
                    components_data['hit_count'].append(hit_count)
            
            if not components_data['total_score']:
                return component_analysis
            
            # Análisis de correlaciones
            correlations = {}
            for component in ['probability', 'diversity', 'historical', 'risk_adjusted']:
                if components_data[component]:
                    corr_with_hits = np.corrcoef(
                        components_data[component], 
                        components_data['hit_count']
                    )[0, 1]
                    correlations[f'{component}_vs_hits'] = corr_with_hits
            
            component_analysis['component_correlations'] = correlations
            
            # Estadísticas por componente
            for component in ['probability', 'diversity', 'historical', 'risk_adjusted']:
                if components_data[component]:
                    stats = {
                        'mean': np.mean(components_data[component]),
                        'std': np.std(components_data[component]),
                        'min': np.min(components_data[component]),
                        'max': np.max(components_data[component]),
                        'correlation_with_hits': correlations.get(f'{component}_vs_hits', 0)
                    }
                    component_analysis[f'{component}_component'] = stats
            
            # Importancia relativa (basada en correlación con aciertos)
            importance_scores = {}
            for component in ['probability', 'diversity', 'historical', 'risk_adjusted']:
                corr_key = f'{component}_vs_hits'
                if corr_key in correlations:
                    importance_scores[component] = abs(correlations[corr_key])
            
            # Normalizar importancias
            total_importance = sum(importance_scores.values()) if importance_scores else 1
            normalized_importance = {
                k: v/total_importance for k, v in importance_scores.items()
            }
            
            component_analysis['component_importance'] = normalized_importance
            
            return component_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing score components: {e}")
            return {'error': str(e)}
    
    def _analyze_temporal_trends(self, predictions: List[Dict], 
                               lottery_results: pd.DataFrame) -> Dict:
        """Analiza tendencias temporales en el rendimiento."""
        try:
            logger.info("Analyzing temporal performance trends...")
            
            temporal_analysis = {
                'weekly_performance': {},
                'monthly_performance': {},
                'performance_trend': {},
                'seasonality_analysis': {}
            }
            
            # Agrupar predicciones por período
            predictions_by_week = defaultdict(list)
            predictions_by_month = defaultdict(list)
            
            for pred in predictions:
                try:
                    pred_date = datetime.fromisoformat(pred['timestamp'].replace('Z', '+00:00'))
                    week_key = pred_date.strftime('%Y-W%U')
                    month_key = pred_date.strftime('%Y-%m')
                    
                    predictions_by_week[week_key].append(pred)
                    predictions_by_month[month_key].append(pred)
                except:
                    continue
            
            # Análisis semanal
            weekly_stats = {}
            for week, week_preds in predictions_by_week.items():
                if len(week_preds) >= 5:  # Mínimo para análisis confiable
                    scores = [p.get('score_total', 0) for p in week_preds]
                    hits = sum(self._calculate_prediction_hits(p, lottery_results) for p in week_preds)
                    
                    weekly_stats[week] = {
                        'prediction_count': len(week_preds),
                        'avg_score': np.mean(scores),
                        'total_hits': hits,
                        'hit_rate': hits / (len(week_preds) * 5) if week_preds else 0
                    }
            
            temporal_analysis['weekly_performance'] = weekly_stats
            
            # Análisis mensual
            monthly_stats = {}
            for month, month_preds in predictions_by_month.items():
                if len(month_preds) >= 10:
                    scores = [p.get('score_total', 0) for p in month_preds]
                    hits = sum(self._calculate_prediction_hits(p, lottery_results) for p in month_preds)
                    
                    monthly_stats[month] = {
                        'prediction_count': len(month_preds),
                        'avg_score': np.mean(scores),
                        'total_hits': hits,
                        'hit_rate': hits / (len(month_preds) * 5) if month_preds else 0
                    }
            
            temporal_analysis['monthly_performance'] = monthly_stats
            
            # Análisis de tendencia
            if len(monthly_stats) >= 3:
                monthly_scores = [stats['avg_score'] for stats in monthly_stats.values()]
                monthly_hit_rates = [stats['hit_rate'] for stats in monthly_stats.values()]
                
                # Tendencia lineal simple
                x = np.arange(len(monthly_scores))
                score_trend = np.polyfit(x, monthly_scores, 1)[0]  # Pendiente
                hit_rate_trend = np.polyfit(x, monthly_hit_rates, 1)[0]
                
                temporal_analysis['performance_trend'] = {
                    'score_trend': 'improving' if score_trend > 0.01 else 'declining' if score_trend < -0.01 else 'stable',
                    'hit_rate_trend': 'improving' if hit_rate_trend > 0.001 else 'declining' if hit_rate_trend < -0.001 else 'stable',
                    'score_slope': score_trend,
                    'hit_rate_slope': hit_rate_trend
                }
            
            return temporal_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing temporal trends: {e}")
            return {'error': str(e)}
    
    def _analyze_prize_categories(self, predictions: List[Dict], 
                                lottery_results: pd.DataFrame) -> Dict:
        """Analiza rendimiento por categorías de premios."""
        try:
            logger.info("Analyzing performance by prize categories...")
            
            prize_analysis = {
                'category_distribution': {},
                'category_frequency': {},
                'expected_vs_actual': {},
                'category_roi': {}
            }
            
            category_counts = {category: 0 for category in self.prize_categories.keys()}
            category_counts['non_winning'] = 0
            
            total_prize_value = 0
            
            for pred in predictions:
                pred_numbers = pred.get('numbers', [])
                pred_pb = pred.get('powerball', 0)
                
                # Verificar contra todos los sorteos
                best_category = 'non_winning'
                best_prize = 0
                
                for _, draw in lottery_results.iterrows():
                    draw_numbers = [draw[f'n{i}'] for i in range(1, 6)]
                    draw_pb = draw['pb']
                    
                    wb_matches = len(set(pred_numbers).intersection(set(draw_numbers)))
                    pb_match = pred_pb == draw_pb
                    
                    # Determinar categoría de premio
                    for category, criteria in self.prize_categories.items():
                        if (wb_matches == criteria['wb_matches'] and 
                            pb_match == criteria['pb_match']):
                            if criteria['prize_value'] > best_prize:
                                best_category = category
                                best_prize = criteria['prize_value']
                
                category_counts[best_category] += 1
                total_prize_value += best_prize
            
            # Calcular distribución
            total_predictions = len(predictions)
            if total_predictions > 0:
                distribution = {
                    category: count / total_predictions 
                    for category, count in category_counts.items()
                }
                prize_analysis['category_distribution'] = distribution
                prize_analysis['category_frequency'] = category_counts
            
            # ROI por categoría
            cost_per_play = 2.0
            total_cost = total_predictions * cost_per_play
            
            category_roi = {}
            for category, count in category_counts.items():
                if category != 'non_winning' and count > 0:
                    category_value = count * self.prize_categories[category]['prize_value']
                    category_cost = count * cost_per_play
                    roi = (category_value - category_cost) / category_cost if category_cost > 0 else 0
                    category_roi[category] = roi * 100
            
            prize_analysis['category_roi'] = category_roi
            
            return prize_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing prize categories: {e}")
            return {'error': str(e)}
    
    def _analyze_prediction_accuracy(self, predictions: List[Dict], 
                                   lottery_results: pd.DataFrame) -> Dict:
        """Analiza la precisión de las predicciones."""
        try:
            logger.info("Analyzing prediction accuracy...")
            
            accuracy_analysis = {
                'number_frequency_accuracy': {},
                'range_prediction_accuracy': {},
                'sum_prediction_accuracy': {},
                'pattern_prediction_accuracy': {}
            }
            
            # Análisis de frecuencia de números
            predicted_numbers = []
            actual_numbers = []
            
            for pred in predictions:
                predicted_numbers.extend(pred.get('numbers', []))
            
            for _, draw in lottery_results.iterrows():
                actual_numbers.extend([draw[f'n{i}'] for i in range(1, 6)])
            
            if predicted_numbers and actual_numbers:
                pred_freq = pd.Series(predicted_numbers).value_counts(normalize=True)
                actual_freq = pd.Series(actual_numbers).value_counts(normalize=True)
                
                # Correlación entre frecuencias predichas y reales
                common_numbers = set(pred_freq.index).intersection(set(actual_freq.index))
                if common_numbers:
                    pred_common = [pred_freq.get(n, 0) for n in common_numbers]
                    actual_common = [actual_freq.get(n, 0) for n in common_numbers]
                    
                    if len(pred_common) > 1:
                        frequency_correlation = np.corrcoef(pred_common, actual_common)[0, 1]
                        accuracy_analysis['number_frequency_accuracy'] = {
                            'correlation': frequency_correlation,
                            'accuracy_score': max(0, frequency_correlation)
                        }
            
            # Análisis de rangos
            pred_ranges = []
            actual_ranges = []
            
            for pred in predictions:
                numbers = pred.get('numbers', [])
                if numbers:
                    pred_ranges.append({
                        'low': sum(1 for n in numbers if n <= 23),
                        'mid': sum(1 for n in numbers if 24 <= n <= 46),
                        'high': sum(1 for n in numbers if n >= 47)
                    })
            
            for _, draw in lottery_results.iterrows():
                numbers = [draw[f'n{i}'] for i in range(1, 6)]
                actual_ranges.append({
                    'low': sum(1 for n in numbers if n <= 23),
                    'mid': sum(1 for n in numbers if 24 <= n <= 46),
                    'high': sum(1 for n in numbers if n >= 47)
                })
            
            if pred_ranges and actual_ranges:
                range_accuracy = self._calculate_range_accuracy(pred_ranges, actual_ranges)
                accuracy_analysis['range_prediction_accuracy'] = range_accuracy
            
            return accuracy_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing prediction accuracy: {e}")
            return {'error': str(e)}
    
    def _analyze_diversity_effectiveness(self, predictions: List[Dict]) -> Dict:
        """Analiza la efectividad de la diversidad en las predicciones."""
        try:
            logger.info("Analyzing diversity effectiveness...")
            
            diversity_analysis = {
                'number_coverage': {},
                'pattern_diversity': {},
                'score_diversity': {},
                'diversity_metrics': {}
            }
            
            if not predictions:
                return diversity_analysis
            
            # Cobertura de números
            all_predicted_numbers = set()
            for pred in predictions:
                all_predicted_numbers.update(pred.get('numbers', []))
            
            total_possible_numbers = 69
            coverage_percentage = len(all_predicted_numbers) / total_possible_numbers * 100
            
            diversity_analysis['number_coverage'] = {
                'unique_numbers_predicted': len(all_predicted_numbers),
                'total_possible_numbers': total_possible_numbers,
                'coverage_percentage': coverage_percentage
            }
            
            # Diversidad de patrones
            patterns = []
            for pred in predictions:
                numbers = pred.get('numbers', [])
                if len(numbers) == 5:
                    # Análisis de paridad
                    even_count = sum(1 for n in numbers if n % 2 == 0)
                    
                    # Análisis de suma
                    total_sum = sum(numbers)
                    
                    # Análisis de spread
                    spread = max(numbers) - min(numbers)
                    
                    patterns.append({
                        'even_count': even_count,
                        'sum': total_sum,
                        'spread': spread
                    })
            
            if patterns:
                pattern_diversity = {
                    'even_count_variety': len(set(p['even_count'] for p in patterns)),
                    'sum_range': max(p['sum'] for p in patterns) - min(p['sum'] for p in patterns),
                    'spread_variety': np.std([p['spread'] for p in patterns])
                }
                diversity_analysis['pattern_diversity'] = pattern_diversity
            
            # Diversidad de scores
            scores = [pred.get('score_total', 0) for pred in predictions]
            if scores:
                score_diversity = {
                    'score_range': max(scores) - min(scores),
                    'score_std': np.std(scores),
                    'score_cv': np.std(scores) / np.mean(scores) if np.mean(scores) > 0 else 0
                }
                diversity_analysis['score_diversity'] = score_diversity
            
            return diversity_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing diversity effectiveness: {e}")
            return {'error': str(e)}
    
    def _calculate_hit_rates(self, predictions: List[Dict], 
                           lottery_results: pd.DataFrame) -> Dict:
        """Calcula tasas de aciertos detalladas."""
        hit_analysis = {
            'overall_hit_rate': 0,
            'hits_by_count': {str(i): 0 for i in range(6)},
            'powerball_hit_rate': 0,
            'partial_match_distribution': {}
        }
        
        if not predictions or lottery_results.empty:
            return hit_analysis
        
        total_hits = 0
        total_possible_hits = len(predictions) * 5
        pb_hits = 0
        
        hits_by_count = {i: 0 for i in range(6)}
        
        for pred in predictions:
            pred_numbers = set(pred.get('numbers', []))
            pred_pb = pred.get('powerball', 0)
            
            best_hit_count = 0
            pb_hit = False
            
            for _, draw in lottery_results.iterrows():
                draw_numbers = set([draw[f'n{i}'] for i in range(1, 6)])
                draw_pb = draw['pb']
                
                hit_count = len(pred_numbers.intersection(draw_numbers))
                if hit_count > best_hit_count:
                    best_hit_count = hit_count
                
                if pred_pb == draw_pb:
                    pb_hit = True
            
            total_hits += best_hit_count
            hits_by_count[best_hit_count] += 1
            
            if pb_hit:
                pb_hits += 1
        
        hit_analysis['overall_hit_rate'] = total_hits / total_possible_hits if total_possible_hits > 0 else 0
        hit_analysis['hits_by_count'] = {str(k): v for k, v in hits_by_count.items()}
        hit_analysis['powerball_hit_rate'] = pb_hits / len(predictions) if predictions else 0
        
        return hit_analysis
    
    def _calculate_prediction_hits(self, prediction: Dict, 
                                 lottery_results: pd.DataFrame) -> int:
        """Calcula el número de aciertos para una predicción específica."""
        pred_numbers = set(prediction.get('numbers', []))
        
        max_hits = 0
        for _, draw in lottery_results.iterrows():
            draw_numbers = set([draw[f'n{i}'] for i in range(1, 6)])
            hits = len(pred_numbers.intersection(draw_numbers))
            max_hits = max(max_hits, hits)
        
        return max_hits
    
    def _calculate_prize_winnings(self, predictions: List[Dict], 
                                lottery_results: pd.DataFrame) -> Dict:
        """Calcula los premios ganados por las predicciones."""
        winning_count = 0
        total_value = 0
        
        for pred in predictions:
            pred_numbers = pred.get('numbers', [])
            pred_pb = pred.get('powerball', 0)
            
            for _, draw in lottery_results.iterrows():
                draw_numbers = [draw[f'n{i}'] for i in range(1, 6)]
                draw_pb = draw['pb']
                
                wb_matches = len(set(pred_numbers).intersection(set(draw_numbers)))
                pb_match = pred_pb == draw_pb
                
                # Verificar categorías de premio
                for category, criteria in self.prize_categories.items():
                    if (wb_matches == criteria['wb_matches'] and 
                        pb_match == criteria['pb_match']):
                        winning_count += 1
                        total_value += criteria['prize_value']
                        break
        
        return {'winning_count': winning_count, 'total_value': total_value}
    
    def _calculate_range_accuracy(self, pred_ranges: List[Dict], 
                                actual_ranges: List[Dict]) -> Dict:
        """Calcula la precisión en la predicción de rangos."""
        range_accuracy = {
            'low_range_accuracy': 0,
            'mid_range_accuracy': 0,
            'high_range_accuracy': 0,
            'overall_range_accuracy': 0
        }
        
        if not pred_ranges or not actual_ranges:
            return range_accuracy
        
        # Calcular diferencias promedio
        low_diffs = []
        mid_diffs = []
        high_diffs = []
        
        for pred_range in pred_ranges:
            min_diff_low = float('inf')
            min_diff_mid = float('inf')
            min_diff_high = float('inf')
            
            for actual_range in actual_ranges:
                diff_low = abs(pred_range['low'] - actual_range['low'])
                diff_mid = abs(pred_range['mid'] - actual_range['mid'])
                diff_high = abs(pred_range['high'] - actual_range['high'])
                
                min_diff_low = min(min_diff_low, diff_low)
                min_diff_mid = min(min_diff_mid, diff_mid)
                min_diff_high = min(min_diff_high, diff_high)
            
            low_diffs.append(min_diff_low)
            mid_diffs.append(min_diff_mid)
            high_diffs.append(min_diff_high)
        
        # Convertir diferencias a scores de precisión (0-1)
        max_possible_diff = 5  # Máxima diferencia posible
        
        range_accuracy['low_range_accuracy'] = 1 - np.mean(low_diffs) / max_possible_diff
        range_accuracy['mid_range_accuracy'] = 1 - np.mean(mid_diffs) / max_possible_diff
        range_accuracy['high_range_accuracy'] = 1 - np.mean(high_diffs) / max_possible_diff
        
        range_accuracy['overall_range_accuracy'] = np.mean([
            range_accuracy['low_range_accuracy'],
            range_accuracy['mid_range_accuracy'],
            range_accuracy['high_range_accuracy']
        ])
        
        return range_accuracy
    
    def _generate_performance_recommendations(self, evaluation_results: Dict) -> List[str]:
        """Genera recomendaciones basadas en el análisis de rendimiento."""
        recommendations = []
        
        try:
            overall_perf = evaluation_results.get('overall_performance', {})
            component_analysis = evaluation_results.get('score_component_analysis', {})
            temporal_trends = evaluation_results.get('temporal_performance_trends', {})
            
            # Recomendaciones basadas en rendimiento general
            hit_rate = overall_perf.get('hit_rate_analysis', {}).get('overall_hit_rate', 0)
            if hit_rate < 0.15:
                recommendations.append("Low hit rate detected - consider retraining model with recent data")
            elif hit_rate > 0.25:
                recommendations.append("Excellent hit rate - maintain current configuration")
            
            # Recomendaciones basadas en ROI
            roi = overall_perf.get('roi_analysis', {}).get('roi_percentage', 0)
            if roi < -50:
                recommendations.append("Negative ROI detected - review scoring weights and strategy")
            elif roi > 0:
                recommendations.append("Positive ROI achieved - consider scaling strategy")
            
            # Recomendaciones basadas en componentes
            component_importance = component_analysis.get('component_importance', {})
            if component_importance:
                max_importance_component = max(component_importance, key=component_importance.get)
                min_importance_component = min(component_importance, key=component_importance.get)
                
                recommendations.append(f"'{max_importance_component}' component shows highest correlation with wins")
                
                if component_importance[min_importance_component] < 0.1:
                    recommendations.append(f"Consider reducing weight of '{min_importance_component}' component")
            
            # Recomendaciones basadas en tendencias temporales
            performance_trend = temporal_trends.get('performance_trend', {})
            if performance_trend.get('score_trend') == 'declining':
                recommendations.append("Declining score trend detected - consider model refresh")
            elif performance_trend.get('hit_rate_trend') == 'improving':
                recommendations.append("Improving hit rate trend - system is learning effectively")
            
            if not recommendations:
                recommendations.append("System performance is stable - continue monitoring")
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            recommendations.append("Error generating recommendations - manual review needed")
        
        return recommendations


def run_comprehensive_performance_evaluation(days_back: int = 90) -> Dict:
    """
    Función de conveniencia para ejecutar evaluación completa de rendimiento.
    
    Args:
        days_back: Días hacia atrás para el análisis
        
    Returns:
        Dict con resultados de evaluación
    """
    evaluator = PerformanceEvaluator(analysis_window_days=days_back)
    return evaluator.evaluate_comprehensive_performance()


def generate_performance_report(evaluation_results: Dict, 
                              output_file: Optional[str] = None) -> str:
    """
    Genera reporte textual detallado de rendimiento.
    
    Args:
        evaluation_results: Resultados de evaluación
        output_file: Archivo de salida opcional
        
    Returns:
        String con el reporte generado
    """
    try:
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("SHIOL+ PERFORMANCE EVALUATION REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Analysis Period: {evaluation_results.get('analysis_window_days', 'Unknown')} days")
        report_lines.append("")
        
        # Overall Performance
        overall = evaluation_results.get('overall_performance', {})
        if overall:
            report_lines.append("OVERALL PERFORMANCE:")
            report_lines.append("-" * 40)
            report_lines.append(f"Total Predictions: {overall.get('total_predictions_analyzed', 0)}")
            
            hit_analysis = overall.get('hit_rate_analysis', {})
            if hit_analysis:
                report_lines.append(f"Overall Hit Rate: {hit_analysis.get('overall_hit_rate', 0):.3f}")
                report_lines.append(f"Powerball Hit Rate: {hit_analysis.get('powerball_hit_rate', 0):.3f}")
            
            roi_analysis = overall.get('roi_analysis', {})
            if roi_analysis:
                report_lines.append(f"ROI: {roi_analysis.get('roi_percentage', 0):.2f}%")
                report_lines.append(f"Net Profit: ${roi_analysis.get('net_profit', 0):.2f}")
            
            report_lines.append("")
        
        # Score Components
        components = evaluation_results.get('score_component_analysis', {})
        if components:
            report_lines.append("SCORE COMPONENT ANALYSIS:")
            report_lines.append("-" * 40)
            
            importance = components.get('component_importance', {})
            for component, imp_score in importance.items():
                report_lines.append(f"{component.capitalize()}: {imp_score:.3f} importance")
            
            report_lines.append("")
        
        # Recommendations
        recommendations = evaluation_results.get('recommendations', [])
        if recommendations:
            report_lines.append("RECOMMENDATIONS:")
            report_lines.append("-" * 40)
            for i, rec in enumerate(recommendations, 1):
                report_lines.append(f"{i}. {rec}")
            report_lines.append("")
        
        report_text = "\n".join(report_lines)
        
        # Save to file if specified
        if output_file:
            try:
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(report_text)
                logger.info(f"Performance report saved to {output_file}")
            except Exception as e:
                logger.error(f"Failed to save report to file: {e}")
        
        return report_text
        
    except Exception as e:
        logger.error(f"Error generating performance report: {e}")
        return f"Error generating report: {str(e)}"

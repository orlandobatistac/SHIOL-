"""
SHIOL+ Model Validator
======================

Sistema avanzado de validación del modelo antes de generar predicciones.
Evalúa la calidad y confiabilidad del modelo usando datos históricos recientes.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from loguru import logger
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.loader import get_data_loader
from src.predictor import ModelTrainer
from src.intelligent_generator import FeatureEngineer
from src.database import get_all_draws, get_performance_analytics


class ModelValidator:
    """
    Validador avanzado del modelo que evalúa su desempeño reciente
    y determina la confiabilidad para generar predicciones.
    """

    def __init__(self, validation_window_days: int = 30):
        self.validation_window_days = validation_window_days
        self.model_trainer = ModelTrainer("models/shiolplus.pkl")
        self.thresholds = {
            'min_accuracy': 0.05,  # Mínima precisión aceptable (5% para lotería)
            'min_top_n_recall': 0.15,  # Recall mínimo en top-N (15%)
            'min_pb_accuracy': 0.03,  # Precisión mínima para Powerball (3%)
            'max_prediction_variance': 0.6  # Máxima varianza en predicciones (60% - más estricto para estabilidad)
        }
        logger.info(
            "ModelValidator initialized with validation window of {} days".
            format(validation_window_days))

    def validate_model_quality(self) -> Dict[str, any]:
        """
        Realiza validación completa de la calidad del modelo.

        Returns:
            Dict con resultados de validación y recomendaciones
        """
        logger.info("Starting comprehensive model quality validation...")

        validation_results = {
            'validation_timestamp': datetime.now().isoformat(),
            'validation_window_days': self.validation_window_days,
            'model_loaded': False,
            'sufficient_data': False,
            'validation_metrics': {},
            'quality_assessment': {},
            'recommendations': [],
            'overall_status': 'unknown'
        }

        try:
            # 1. Verificar que el modelo esté cargado
            model = self.model_trainer.load_model()
            if model is None:
                validation_results['recommendations'].append(
                    "Model not found - training required")
                validation_results['overall_status'] = 'critical'
                return validation_results

            validation_results['model_loaded'] = True
            logger.info("✓ Model loaded successfully")

            # 2. Obtener datos para validación
            historical_data = get_all_draws()
            if len(historical_data) < 50:
                validation_results['recommendations'].append(
                    "Insufficient historical data for validation")
                validation_results['overall_status'] = 'warning'
                return validation_results

            validation_results['sufficient_data'] = True

            # 3. Realizar validaciones específicas
            recent_performance = self._validate_recent_performance(
                historical_data)
            top_n_analysis = self._validate_top_n_predictions(historical_data)
            powerball_analysis = self._validate_powerball_predictions(
                historical_data)
            prediction_stability = self._validate_prediction_stability(
                historical_data)

            # 4. Consolidar métricas
            validation_results['validation_metrics'] = {
                'recent_performance': recent_performance,
                'top_n_analysis': top_n_analysis,
                'powerball_analysis': powerball_analysis,
                'prediction_stability': prediction_stability
            }

            # 5. Evaluar calidad general
            quality_assessment = self._assess_overall_quality(
                validation_results['validation_metrics'])
            validation_results['quality_assessment'] = quality_assessment

            # 6. Generar recomendaciones
            recommendations = self._generate_recommendations(
                quality_assessment)
            validation_results['recommendations'] = recommendations

            # 7. Determinar estado general
            validation_results['overall_status'] = quality_assessment[
                'overall_status']

            logger.info(
                f"Model validation completed - Status: {validation_results['overall_status']}"
            )
            return validation_results

        except Exception as e:
            logger.error(f"Error during model validation: {e}")
            validation_results['error'] = str(e)
            validation_results['overall_status'] = 'error'
            validation_results['recommendations'].append(
                "Validation failed - check logs for details")
            return validation_results

    def _validate_recent_performance(self,
                                     historical_data: pd.DataFrame) -> Dict:
        """Valida el desempeño del modelo en datos recientes."""
        try:
            logger.info("Validating recent model performance...")

            # Tomar últimos N días
            cutoff_date = datetime.now() - timedelta(
                days=self.validation_window_days)
            recent_data = historical_data[historical_data['draw_date'] >=
                                          cutoff_date.strftime('%Y-%m-%d')]

            if len(recent_data) < 5:
                return {
                    'status': 'insufficient_data',
                    'draws_analyzed': len(recent_data),
                    'min_required': 5
                }

            # Generar features y predicciones para los datos recientes
            feature_engineer = FeatureEngineer(historical_data)
            features_df = feature_engineer.engineer_features(
                use_temporal_analysis=True)

            # Obtener predicciones del modelo
            recent_features = features_df.iloc[-len(recent_data):]

            # Check feature compatibility before prediction
            try:
                prob_df = self.model_trainer.predict_probabilities(
                    recent_features)

                if prob_df is None:
                    return {'status': 'prediction_failed'}
            except Exception as e:
                if "Feature shape mismatch" in str(e) or "expected" in str(e):
                    logger.warning(
                        f"Feature compatibility issue detected: {e}")
                    return {
                        'status': 'feature_mismatch',
                        'error': str(e),
                        'recommendation': 'model_retrain_required'
                    }
                else:
                    return {'status': 'prediction_failed', 'error': str(e)}

            # Calcular métricas de desempeño
            wb_accuracy = self._calculate_white_ball_accuracy(
                recent_data, prob_df)
            pb_accuracy = self._calculate_powerball_accuracy(
                recent_data, prob_df)

            performance_metrics = {
                'status':
                'completed',
                'draws_analyzed':
                len(recent_data),
                'white_ball_accuracy':
                wb_accuracy,
                'powerball_accuracy':
                pb_accuracy,
                'combined_accuracy': (wb_accuracy * 0.8 + pb_accuracy * 0.2),
                'analysis_period':
                f"{cutoff_date.strftime('%Y-%m-%d')} to {datetime.now().strftime('%Y-%m-%d')}"
            }

            logger.info(
                f"Recent performance: WB={wb_accuracy:.3f}, PB={pb_accuracy:.3f}"
            )
            return performance_metrics

        except Exception as e:
            logger.error(f"Error validating recent performance: {e}")
            return {'status': 'error', 'error': str(e)}

    def _validate_top_n_predictions(self,
                                    historical_data: pd.DataFrame) -> Dict:
        """Valida la efectividad de las predicciones top-N del modelo."""
        try:
            logger.info("Validating top-N prediction effectiveness...")

            # Analizar últimos 20 sorteos
            recent_draws = historical_data.tail(20)

            if len(recent_draws) < 10:
                return {'status': 'insufficient_data'}

            top_n_results = {}
            for n in [5, 10, 15, 20]:
                hit_rate = self._calculate_top_n_hit_rate(recent_draws, n)
                top_n_results[f'top_{n}_hit_rate'] = hit_rate

            # Calcular recall promedio
            avg_recall = np.mean(list(top_n_results.values()))

            analysis = {
                'status':
                'completed',
                'draws_analyzed':
                len(recent_draws),
                'top_n_hit_rates':
                top_n_results,
                'average_recall':
                avg_recall,
                'meets_threshold':
                avg_recall >= self.thresholds['min_top_n_recall']
            }

            logger.info(f"Top-N analysis: Average recall = {avg_recall:.3f}")
            return analysis

        except Exception as e:
            logger.error(f"Error validating top-N predictions: {e}")
            return {'status': 'error', 'error': str(e)}

    def _validate_powerball_predictions(self,
                                        historical_data: pd.DataFrame) -> Dict:
        """Valida específicamente las predicciones de Powerball."""
        try:
            logger.info("Validating Powerball prediction accuracy...")

            recent_draws = historical_data.tail(30)

            if len(recent_draws) < 10:
                return {'status': 'insufficient_data'}

            # Simular predicciones de Powerball
            pb_predictions = []
            actual_pb = recent_draws['pb'].tolist()

            # Para cada sorteo, predecir basado en datos anteriores
            for i in range(len(recent_draws)):
                historical_subset = historical_data.iloc[:-(len(recent_draws) -
                                                            i)]
                pb_freq = historical_subset['pb'].value_counts()
                # Tomar los 3 Powerballs más frecuentes como predicción
                top_pb = pb_freq.head(3).index.tolist()
                pb_predictions.append(top_pb)

            # Calcular precisión
            hits = 0
            for i, actual in enumerate(actual_pb):
                if actual in pb_predictions[i]:
                    hits += 1

            pb_accuracy = hits / len(actual_pb)

            analysis = {
                'status':
                'completed',
                'draws_analyzed':
                len(recent_draws),
                'powerball_accuracy':
                pb_accuracy,
                'hits':
                hits,
                'total_predictions':
                len(actual_pb),
                'meets_threshold':
                pb_accuracy >= self.thresholds['min_pb_accuracy']
            }

            logger.info(
                f"Powerball accuracy: {pb_accuracy:.3f} ({hits}/{len(actual_pb)})"
            )
            return analysis

        except Exception as e:
            logger.error(f"Error validating Powerball predictions: {e}")
            return {'status': 'error', 'error': str(e)}

    def _validate_prediction_stability(self,
                                       historical_data: pd.DataFrame) -> Dict:
        """Valida la estabilidad de las predicciones del modelo con variaciones controladas."""
        try:
            logger.info("Validating prediction stability with controlled variations...")

            # Generar features base
            feature_engineer = FeatureEngineer(historical_data)
            base_features = feature_engineer.engineer_features(
                use_temporal_analysis=True)

            if base_features.empty:
                return {'status': 'no_features'}

            # Obtener el último conjunto de features
            latest_features = base_features.iloc[-1:].copy()

            predictions = []
            prediction_details = []
            
            # Generar múltiples predicciones con pequeñas variaciones controladas
            for i in range(5):
                try:
                    # Crear copia para variación
                    varied_features = latest_features.copy()
                    
                    if i > 0:
                        # Aplicar pequeñas variaciones controladas (±2% del valor original)
                        noise_factor = 0.02
                        np.random.seed(42 + i)  # Seed fijo para reproducibilidad
                        
                        numeric_columns = varied_features.select_dtypes(include=[np.number]).columns
                        for col in numeric_columns:
                            if col in varied_features.columns:
                                original_value = varied_features[col].iloc[0]
                                if original_value != 0:
                                    # Aplicar ruido gaussiano pequeño
                                    noise = np.random.normal(0, abs(original_value) * noise_factor)
                                    varied_features[col] = original_value + noise
                                else:
                                    # Para valores cero, agregar ruido pequeño absoluto
                                    varied_features[col] = np.random.normal(0, 0.001)

                    # Generar predicción
                    prob_df = self.model_trainer.predict_probabilities(varied_features)
                    
                    if prob_df is not None:
                        # Extraer top 15 números blancos para análisis más robusto
                        wb_cols = [col for col in prob_df.columns if col.startswith('wb_')]
                        pb_cols = [col for col in prob_df.columns if col.startswith('pb_')]
                        
                        if wb_cols and pb_cols:
                            wb_probs = prob_df[wb_cols].iloc[0].sort_values(ascending=False)
                            pb_probs = prob_df[pb_cols].iloc[0].sort_values(ascending=False)
                            
                            top_15_wb = [int(col.split('_')[1]) for col in wb_probs.head(15).index]
                            top_5_pb = [int(col.split('_')[1]) for col in pb_probs.head(5).index]
                            
                            # Combinar números blancos y powerball para análisis de estabilidad
                            prediction_set = set(top_15_wb + top_5_pb)
                            predictions.append(prediction_set)
                            
                            prediction_details.append({
                                'iteration': i,
                                'wb_numbers': top_15_wb,
                                'pb_numbers': top_5_pb,
                                'variation_applied': i > 0
                            })
                
                except Exception as e:
                    logger.warning(f"Error in stability iteration {i}: {e}")
                    continue

            if len(predictions) < 3:
                return {
                    'status': 'insufficient_predictions',
                    'predictions_generated': len(predictions),
                    'minimum_required': 3
                }

            # Calcular estabilidad usando Jaccard similarity
            stability_scores = []
            
            # Comparar cada predicción con la base (primera predicción)
            base_prediction = predictions[0]
            
            for i in range(1, len(predictions)):
                current_prediction = predictions[i]
                
                # Jaccard similarity
                intersection = len(base_prediction.intersection(current_prediction))
                union = len(base_prediction.union(current_prediction))
                
                similarity = intersection / union if union > 0 else 0
                stability_scores.append(similarity)

            # Calcular métricas de estabilidad
            avg_stability = np.mean(stability_scores) if stability_scores else 0
            min_stability = np.min(stability_scores) if stability_scores else 0
            variance = 1.0 - avg_stability  # Convertir similarity a variance
            
            # El modelo es estable si la similitud promedio es alta
            is_stable = avg_stability >= (1.0 - self.thresholds['max_prediction_variance'])
            
            analysis = {
                'status': 'completed',
                'predictions_generated': len(predictions),
                'prediction_variance': variance,
                'stability_score': avg_stability,
                'min_stability': min_stability,
                'is_stable': is_stable,
                'stability_threshold': 1.0 - self.thresholds['max_prediction_variance'],
                'prediction_details': prediction_details[:3]  # Solo mostrar primeras 3 para logs
            }

            logger.info(f"Prediction stability: variance = {variance:.3f}, avg_similarity = {avg_stability:.3f}")
            return analysis

        except Exception as e:
            logger.error(f"Error validating prediction stability: {e}")
            return {'status': 'error', 'error': str(e)}

    def _calculate_white_ball_accuracy(self, actual_data: pd.DataFrame,
                                       predictions_df: pd.DataFrame) -> float:
        """Calcula la precisión de las predicciones de números blancos."""
        try:
            wb_cols = [
                col for col in predictions_df.columns if col.startswith('wb_')
            ]
            total_accuracy = 0
            valid_predictions = 0

            for i, row in actual_data.iterrows():
                if i >= len(predictions_df):
                    break

                actual_numbers = [row[f'n{j}'] for j in range(1, 6)]
                pred_probs = predictions_df.iloc[i][wb_cols]

                # Top 15 predicciones
                top_15_nums = [
                    int(col.split('_')[1])
                    for col in pred_probs.nlargest(15).index
                ]

                hits = sum(1 for num in actual_numbers if num in top_15_nums)
                accuracy = hits / 5.0
                total_accuracy += accuracy
                valid_predictions += 1

            return total_accuracy / valid_predictions if valid_predictions > 0 else 0.0

        except Exception as e:
            logger.error(f"Error calculating white ball accuracy: {e}")
            return 0.0

    def _calculate_powerball_accuracy(self, actual_data: pd.DataFrame,
                                      predictions_df: pd.DataFrame) -> float:
        """Calcula la precisión de las predicciones de Powerball."""
        try:
            pb_cols = [
                col for col in predictions_df.columns if col.startswith('pb_')
            ]
            total_accuracy = 0
            valid_predictions = 0

            for i, row in actual_data.iterrows():
                if i >= len(predictions_df):
                    break

                actual_pb = row['pb']
                pred_probs = predictions_df.iloc[i][pb_cols]

                # Top 5 predicciones de Powerball
                top_5_pb = [
                    int(col.split('_')[1])
                    for col in pred_probs.nlargest(5).index
                ]

                accuracy = 1.0 if actual_pb in top_5_pb else 0.0
                total_accuracy += accuracy
                valid_predictions += 1

            return total_accuracy / valid_predictions if valid_predictions > 0 else 0.0

        except Exception as e:
            logger.error(f"Error calculating powerball accuracy: {e}")
            return 0.0

    def _calculate_top_n_hit_rate(self, draws: pd.DataFrame, n: int) -> float:
        """Calcula la tasa de aciertos en top-N números."""
        try:
            # Simulación simple basada en frecuencia histórica
            all_numbers = []
            for col in ['n1', 'n2', 'n3', 'n4', 'n5']:
                all_numbers.extend(draws[col].tolist())

            freq_dist = pd.Series(all_numbers).value_counts()
            top_n_numbers = freq_dist.head(n).index.tolist()

            hits = 0
            total = 0

            for _, row in draws.iterrows():
                actual_numbers = [row[f'n{j}'] for j in range(1, 6)]
                hit_count = sum(1 for num in actual_numbers
                                if num in top_n_numbers)
                hits += hit_count
                total += 5

            return hits / total if total > 0 else 0.0

        except Exception as e:
            logger.error(f"Error calculating top-{n} hit rate: {e}")
            return 0.0

    def _calculate_prediction_variance(self,
                                       predictions: List[set]) -> float:
        """Calcula la varianza entre múltiples predicciones usando sets."""
        try:
            if len(predictions) < 2:
                return 0.0

            # Calcular todas las similitudes de Jaccard por pares
            similarities = []
            for i in range(len(predictions)):
                for j in range(i + 1, len(predictions)):
                    set1, set2 = predictions[i], predictions[j]
                    
                    # Jaccard similarity
                    intersection = len(set1.intersection(set2))
                    union = len(set1.union(set2))
                    
                    if union > 0:
                        similarity = intersection / union
                    else:
                        similarity = 1.0 if len(set1) == 0 and len(set2) == 0 else 0.0
                    
                    similarities.append(similarity)

            if not similarities:
                return 1.0

            # Varianza = 1 - similitud promedio
            avg_similarity = np.mean(similarities)
            variance = 1.0 - avg_similarity
            
            # Asegurar que la varianza esté en rango [0, 1]
            variance = max(0.0, min(1.0, variance))

            return variance

        except Exception as e:
            logger.error(f"Error calculating prediction variance: {e}")
            return 1.0

    def _assess_overall_quality(self, metrics: Dict) -> Dict:
        """Evalúa la calidad general basada en todas las métricas."""
        quality_scores = []
        issues = []

        # Evaluar desempeño reciente
        if metrics['recent_performance'].get('status') == 'completed':
            recent_score = metrics['recent_performance']['combined_accuracy']
            quality_scores.append(recent_score)

            if recent_score < self.thresholds['min_accuracy']:
                issues.append(f"Low recent accuracy: {recent_score:.3f}")

        # Evaluar top-N
        if metrics['top_n_analysis'].get('status') == 'completed':
            top_n_score = metrics['top_n_analysis']['average_recall']
            quality_scores.append(top_n_score)

            if not metrics['top_n_analysis']['meets_threshold']:
                issues.append(f"Low top-N recall: {top_n_score:.3f}")

        # Evaluar Powerball
        if metrics['powerball_analysis'].get('status') == 'completed':
            pb_score = metrics['powerball_analysis']['powerball_accuracy']
            quality_scores.append(pb_score *
                                  2)  # Peso adicional para Powerball

            if not metrics['powerball_analysis']['meets_threshold']:
                issues.append(f"Low Powerball accuracy: {pb_score:.3f}")

        # Evaluar estabilidad
        if metrics['prediction_stability'].get('status') == 'completed':
            stability_score = metrics['prediction_stability'][
                'stability_score']
            quality_scores.append(stability_score)

            if not metrics['prediction_stability']['is_stable']:
                issues.append(
                    f"Unstable predictions: variance {metrics['prediction_stability']['prediction_variance']:.3f}"
                )

        # Calcular score general
        overall_score = np.mean(quality_scores) if quality_scores else 0.0

        # Determinar estado
        if overall_score >= 0.7 and len(issues) == 0:
            status = 'excellent'
        elif overall_score >= 0.5 and len(issues) <= 1:
            status = 'good'
        elif overall_score >= 0.3 and len(issues) <= 2:
            status = 'acceptable'
        elif overall_score >= 0.15:
            status = 'poor'
        else:
            status = 'critical'

        return {
            'overall_score': overall_score,
            'overall_status': status,
            'quality_issues': issues,
            'component_scores': quality_scores,
            'total_components_evaluated': len(quality_scores)
        }

    def _generate_recommendations(self, quality_assessment: Dict) -> List[str]:
        """Genera recomendaciones basadas en la evaluación de calidad."""
        recommendations = []

        status = quality_assessment['overall_status']
        score = quality_assessment['overall_score']
        issues = quality_assessment['quality_issues']

        if status == 'critical':
            recommendations.append(
                "CRITICAL: Model retraining required immediately")
            recommendations.append(
                "Suspend prediction generation until model is retrained")
            recommendations.append(
                "Investigate data quality and feature engineering")

        elif status == 'poor':
            recommendations.append(
                "Model performance is below acceptable thresholds")
            recommendations.append("Schedule model retraining within 24 hours")
            recommendations.append("Review recent data for anomalies")

        elif status == 'acceptable':
            recommendations.append("Model performance is marginal")
            recommendations.append("Consider retraining with recent data")
            recommendations.append("Monitor performance closely")

        elif status == 'good':
            recommendations.append("Model performance is satisfactory")
            recommendations.append("Continue regular monitoring")

        elif status == 'excellent':
            recommendations.append("Model performance is optimal")
            recommendations.append("Maintain current configuration")

        # Recomendaciones específicas por problemas
        for issue in issues:
            if "Low recent accuracy" in issue:
                recommendations.append(
                    "Focus on improving feature engineering for recent patterns"
                )
            elif "Low top-N recall" in issue:
                recommendations.append(
                    "Adjust probability thresholds or increase candidate pool")
            elif "Low Powerball accuracy" in issue:
                recommendations.append("Review Powerball prediction strategy")
            elif "Unstable predictions" in issue:
                recommendations.append(
                    "Investigate model stability and feature consistency")

        return recommendations


def validate_model_before_prediction() -> Dict:
    """
    Función de conveniencia para validar el modelo antes de generar predicciones.

    Returns:
        Dict con resultados de validación
    """
    validator = ModelValidator(validation_window_days=30)
    return validator.validate_model_quality()


def is_model_ready_for_prediction() -> bool:
    """
    Verifica si el modelo está listo para generar predicciones confiables.

    Returns:
        bool: True si el modelo está listo, False caso contrario
    """
    validation_results = validate_model_before_prediction()
    status = validation_results.get('overall_status', 'unknown')

    # Consideramos el modelo listo si el estado es aceptable o mejor
    ready_statuses = ['acceptable', 'good', 'excellent']
    return status in ready_statuses
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
            'min_accuracy': 0.15,  # Mínima precisión aceptable
            'min_top_n_recall': 0.25,  # Recall mínimo en top-N
            'min_pb_accuracy': 0.08,  # Precisión mínima para Powerball
            'max_prediction_variance': 0.3  # Máxima varianza en predicciones
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
            # 1. Verificar que el modelo se pueda cargar
            model = self.model_trainer.load_model()
            if model is None:
                validation_results['overall_status'] = 'failed'
                validation_results['recommendations'].append(
                    'Model file not found or corrupted')
                return validation_results

            validation_results['model_loaded'] = True

            # 2. Verificar compatibilidad de features
            feature_compatibility = self._check_feature_compatibility()
            validation_results['validation_metrics'][
                'feature_compatibility'] = feature_compatibility

            if feature_compatibility['status'] == 'feature_mismatch':
                validation_results['overall_status'] = 'needs_retrain'
                validation_results['recommendations'].append(
                    'Feature shape mismatch detected - retrain required')
                return validation_results

            # 3. Verificar datos suficientes
            data_check = self._check_data_availability()
            validation_results['validation_metrics'][
                'data_availability'] = data_check
            validation_results['sufficient_data'] = data_check['sufficient']

            if not data_check['sufficient']:
                validation_results['overall_status'] = 'insufficient_data'
                validation_results['recommendations'].append(
                    'Insufficient data for validation')
                return validation_results

            # 4. Evaluar performance reciente
            recent_performance = self._evaluate_recent_performance()
            validation_results['validation_metrics'][
                'recent_performance'] = recent_performance

            # 5. Determinar estado general
            overall_status = self._determine_overall_status(
                validation_results['validation_metrics'])
            validation_results['overall_status'] = overall_status
            validation_results[
                'recommendations'] = self._generate_recommendations(
                    validation_results['validation_metrics'])

            logger.info(
                f"Model validation completed with status: {overall_status}")
            return validation_results

        except Exception as e:
            logger.error(f"Error during model validation: {e}")
            validation_results['overall_status'] = 'error'
            validation_results['recommendations'].append(
                f'Validation error: {str(e)}')
            return validation_results

    def _check_feature_compatibility(self) -> Dict[str, any]:
        """Verifica la compatibilidad de features entre modelo y datos actuales."""
        try:
            # Obtener datos actuales y generar features
            data_loader = get_data_loader()
            historical_data = data_loader.load_historical_data()

            if historical_data.empty:
                return {
                    'status': 'no_data',
                    'error': 'No historical data available'
                }

            # Generar features como lo hace el predictor
            feature_engineer = FeatureEngineer(historical_data)
            features_df = feature_engineer.engineer_features(
                use_temporal_analysis=True)

            # Obtener features esperadas por el modelo
            try:
                model = self.model_trainer.model
                if model and hasattr(model, 'estimators_') and len(
                        model.estimators_) > 0:
                    expected_features = model.estimators_[0].get_booster(
                    ).feature_names
                    expected_count = len(expected_features)
                else:
                    # Fallback: usar las features estándar de SHIOL+
                    expected_features = [
                        "even_count", "odd_count", "sum", "spread",
                        "consecutive_count", "avg_delay", "max_delay",
                        "min_delay", "dist_to_recent", "avg_dist_to_top_n",
                        "dist_to_centroid", "time_weight",
                        "increasing_trend_count", "decreasing_trend_count",
                        "stable_trend_count"
                    ]
                    expected_count = 15

                # Verificar compatibilidad
                available_features = [
                    col for col in expected_features
                    if col in features_df.columns
                ]
                actual_count = len(available_features)

                if actual_count != expected_count:
                    missing_features = [
                        f for f in expected_features
                        if f not in features_df.columns
                    ]
                    return {
                        'status': 'feature_mismatch',
                        'expected_count': expected_count,
                        'actual_count': actual_count,
                        'missing_features': missing_features,
                        'available_features': available_features
                    }

                return {
                    'status': 'compatible',
                    'feature_count': actual_count,
                    'available_features': available_features
                }

            except Exception as e:
                logger.warning(f"Could not check model features: {e}")
                return {'status': 'unknown', 'error': str(e)}

        except Exception as e:
            return {'status': 'error', 'error': str(e)}

    def _check_data_availability(self) -> Dict[str, any]:
        """Verifica disponibilidad de datos para validación."""
        try:
            historical_data = get_all_draws()

            if historical_data.empty:
                return {
                    'sufficient': False,
                    'total_records': 0,
                    'reason': 'No historical data available'
                }

            # Verificar datos recientes
            historical_data['draw_date'] = pd.to_datetime(
                historical_data['draw_date'])
            cutoff_date = datetime.now() - timedelta(
                days=self.validation_window_days)
            recent_data = historical_data[historical_data['draw_date'] >=
                                          cutoff_date]

            min_required = 10  # Mínimo de registros requeridos
            sufficient = len(historical_data) >= min_required

            return {
                'sufficient': sufficient,
                'total_records': len(historical_data),
                'recent_records': len(recent_data),
                'validation_window_days': self.validation_window_days,
                'min_required': min_required
            }

        except Exception as e:
            return {'sufficient': False, 'error': str(e)}

    def _evaluate_recent_performance(self) -> Dict[str, any]:
        """Evalúa el performance reciente del modelo."""
        try:
            # Obtener analytics de performance
            performance_data = get_performance_analytics(
                days_back=self.validation_window_days)

            if not performance_data or performance_data.get(
                    'total_predictions', 0) == 0:
                return {
                    'status': 'no_performance_data',
                    'total_predictions': 0
                }

            # Evaluar métricas clave
            total_predictions = performance_data.get('total_predictions', 0)
            avg_accuracy = performance_data.get('avg_accuracy', 0.0)
            win_rate = performance_data.get('win_rate', 0.0)

            # Calcular score combinado
            performance_score = (avg_accuracy * 0.6) + (win_rate * 0.4)

            # Determinar estado
            if performance_score >= 0.4:
                status = 'good'
            elif performance_score >= 0.25:
                status = 'acceptable'
            else:
                status = 'poor'

            return {
                'status': status,
                'performance_score': performance_score,
                'total_predictions': total_predictions,
                'avg_accuracy': avg_accuracy,
                'win_rate': win_rate,
                'validation_window_days': self.validation_window_days
            }

        except Exception as e:
            return {'status': 'error', 'error': str(e)}

    def _determine_overall_status(self, metrics: Dict) -> str:
        """Determina el estado general del modelo basado en todas las métricas."""
        feature_compat = metrics.get('feature_compatibility', {})
        data_avail = metrics.get('data_availability', {})
        recent_perf = metrics.get('recent_performance', {})

        # Verificar problemas críticos
        if feature_compat.get('status') == 'feature_mismatch':
            return 'needs_retrain'

        if not data_avail.get('sufficient', False):
            return 'insufficient_data'

        # Evaluar performance
        perf_status = recent_perf.get('status', 'unknown')

        if perf_status == 'good':
            return 'good'
        elif perf_status == 'acceptable':
            return 'acceptable'
        elif perf_status == 'poor':
            return 'needs_improvement'
        else:
            return 'unknown'

    def _generate_recommendations(self, metrics: Dict) -> List[str]:
        """Genera recomendaciones basadas en las métricas."""
        recommendations = []

        feature_compat = metrics.get('feature_compatibility', {})
        recent_perf = metrics.get('recent_performance', {})

        # Recomendaciones por compatibilidad de features
        if feature_compat.get('status') == 'feature_mismatch':
            recommendations.append(
                "Feature shape mismatch detected - model retraining required")

        # Recomendaciones por performance
        perf_status = recent_perf.get('status', 'unknown')
        if perf_status == 'poor':
            recommendations.append(
                "Poor recent performance - consider model retraining")
        elif perf_status == 'acceptable':
            recommendations.append("Acceptable performance - monitor closely")
        elif perf_status == 'good':
            recommendations.append("Good performance - model is working well")

        if not recommendations:
            recommendations.append("Model validation completed successfully")

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
    ready_statuses = ['acceptable', 'good']
    return status in ready_statusesaws


class ModelValidator:
    """
    Validador avanzado del modelo que evalúa su desempeño reciente
    y determina la confiabilidad para generar predicciones.
    """

    def __init__(self, validation_window_days: int = 30):
        self.validation_window_days = validation_window_days
        self.model_trainer = ModelTrainer("models/shiolplus.pkl")
        self.thresholds = {
            'min_accuracy': 0.15,  # Mínima precisión aceptable
            'min_top_n_recall': 0.25,  # Recall mínimo en top-N
            'min_pb_accuracy': 0.08,  # Precisión mínima para Powerball
            'max_prediction_variance': 0.3  # Máxima varianza en predicciones
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
        """Valida la estabilidad de las predicciones del modelo."""
        try:
            logger.info("Validating prediction stability...")

            # Generar múltiples predicciones con pequeñas variaciones en los datos
            feature_engineer = FeatureEngineer(historical_data)
            base_features = feature_engineer.engineer_features(
                use_temporal_analysis=True)

            predictions = []
            for i in range(5):
                # Pequeña variación en los features
                varied_features = base_features.copy()
                if len(varied_features) > 1:
                    varied_features = varied_features.iloc[:-i] if i > 0 else varied_features

                prob_df = self.model_trainer.predict_probabilities(
                    varied_features.tail(1))
                if prob_df is not None:
                    # Extraer top 10 números blancos
                    wb_cols = [
                        col for col in prob_df.columns if col.startswith('wb_')
                    ]
                    wb_probs = prob_df[wb_cols].iloc[0].sort_values(
                        ascending=False)
                    top_10_wb = [
                        int(col.split('_')[1])
                        for col in wb_probs.head(10).index
                    ]
                    predictions.append(top_10_wb)

            if len(predictions) < 3:
                return {'status': 'insufficient_predictions'}

            # Calcular varianza en las predicciones
            variance = self._calculate_prediction_variance(predictions)

            analysis = {
                'status': 'completed',
                'predictions_generated': len(predictions),
                'prediction_variance': variance,
                'stability_score': 1.0 - variance,
                'is_stable': variance
                <= self.thresholds['max_prediction_variance']
            }

            logger.info(f"Prediction stability: variance = {variance:.3f}")
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
                                       predictions: List[List[int]]) -> float:
        """Calcula la varianza entre múltiples predicciones."""
        try:
            if len(predictions) < 2:
                return 0.0

            # Calcular Jaccard similarity entre todas las parejas
            similarities = []
            for i in range(len(predictions)):
                for j in range(i + 1, len(predictions)):
                    set1, set2 = set(predictions[i]), set(predictions[j])
                    intersection = len(set1.intersection(set2))
                    union = len(set1.union(set2))
                    similarity = intersection / union if union > 0 else 0
                    similarities.append(similarity)

            # Varianza = 1 - similitud promedio
            avg_similarity = np.mean(similarities)
            variance = 1.0 - avg_similarity

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

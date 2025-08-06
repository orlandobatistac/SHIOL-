
"""
SHIOL+ Auto Retrainer
=====================

Sistema automático de reentrenamiento del modelo basado en nuevos datos
y evaluación de performance. Actualiza model_version y dataset_hash automáticamente.
"""

import os
import shutil
import hashlib
from datetime import datetime, timedelta
from loguru import logger
from typing import Dict, Optional, Tuple
import pandas as pd
import numpy as np

from src.loader import get_data_loader, update_database_from_source
from src.predictor import get_model_trainer, ModelTrainer
from src.intelligent_generator import FeatureEngineer
from src.database import get_all_draws, get_performance_analytics
from src.model_validator import ModelValidator


class AutoRetrainer:
    """
    Sistema automático de reentrenamiento que determina cuándo es necesario
    reentrenar el modelo y ejecuta el proceso completo.
    """
    
    def __init__(self, config_path: str = "config/config.ini"):
        self.config_path = config_path
        self.model_trainer = get_model_trainer()
        self.validator = ModelValidator()
        
        # Criterios para reentrenamiento
        self.retrain_criteria = {
            'min_new_draws': 10,  # Mínimo número de sorteos nuevos
            'max_days_without_retrain': 30,  # Máximo días sin reentrenar
            'min_performance_threshold': 0.3,  # Performance mínimo aceptable
            'data_change_threshold': 0.15  # Cambio mínimo en datos (%)
        }
        
        # Configuración de respaldo
        self.backup_config = {
            'max_backups': 5,
            'backup_dir': 'models/backups'
        }
        
        logger.info("AutoRetrainer initialized with smart retraining criteria")
    
    def should_retrain_model(self) -> Tuple[bool, List[str]]:
        """
        Determina si el modelo necesita ser reentrenado.
        
        Returns:
            Tuple[bool, List[str]]: (needs_retrain, reasons)
        """
        logger.info("Evaluating if model retraining is needed...")
        
        reasons = []
        needs_retrain = False
        
        try:
            # 1. Verificar edad del modelo
            model_age_check, model_age_reason = self._check_model_age()
            if model_age_check:
                needs_retrain = True
                reasons.append(model_age_reason)
            
            # 2. Verificar nuevos datos disponibles
            new_data_check, new_data_reason = self._check_new_data_availability()
            if new_data_check:
                needs_retrain = True
                reasons.append(new_data_reason)
            
            # 3. Verificar performance del modelo
            performance_check, performance_reason = self._check_model_performance()
            if performance_check:
                needs_retrain = True
                reasons.append(performance_reason)
            
            # 4. Verificar cambios significativos en datos
            data_change_check, data_change_reason = self._check_data_drift()
            if data_change_check:
                needs_retrain = True
                reasons.append(data_change_reason)
            
            logger.info(f"Retrain evaluation: {'NEEDED' if needs_retrain else 'NOT NEEDED'}")
            if reasons:
                logger.info(f"Retrain reasons: {', '.join(reasons)}")
            
            return needs_retrain, reasons
            
        except Exception as e:
            logger.error(f"Error evaluating retrain necessity: {e}")
            return True, [f"Error in evaluation: {str(e)} - Recommending retrain for safety"]
    
    def execute_automatic_retrain(self, force: bool = False) -> Dict:
        """
        Ejecuta el reentrenamiento automático del modelo.
        
        Args:
            force: Si True, fuerza el reentrenamiento sin verificar criterios
            
        Returns:
            Dict con resultados del reentrenamiento
        """
        logger.info("Starting automatic model retraining process...")
        
        retrain_results = {
            'retrain_timestamp': datetime.now().isoformat(),
            'force_retrain': force,
            'retrain_needed': False,
            'retrain_executed': False,
            'backup_created': False,
            'new_model_version': None,
            'new_dataset_hash': None,
            'performance_comparison': {},
            'error': None,
            'execution_time_seconds': 0
        }
        
        start_time = datetime.now()
        
        try:
            # 1. Verificar si es necesario reentrenar
            if not force:
                needs_retrain, reasons = self.should_retrain_model()
                retrain_results['retrain_needed'] = needs_retrain
                retrain_results['retrain_reasons'] = reasons
                
                if not needs_retrain:
                    logger.info("Model retraining not needed based on current criteria")
                    return retrain_results
            else:
                retrain_results['retrain_needed'] = True
                retrain_results['retrain_reasons'] = ["Force retrain requested"]
            
            # 2. Actualizar datos desde la fuente
            logger.info("Updating data from source before retraining...")
            total_rows = update_database_from_source()
            retrain_results['total_data_rows'] = total_rows
            
            # 3. Crear backup del modelo actual
            backup_success = self._create_model_backup()
            retrain_results['backup_created'] = backup_success
            
            if not backup_success:
                logger.warning("Failed to create model backup, but continuing with retrain")
            
            # 4. Validar modelo actual para comparación
            pre_retrain_validation = self.validator.validate_model_quality()
            retrain_results['pre_retrain_validation'] = pre_retrain_validation
            
            # 5. Preparar nuevos datos para entrenamiento
            logger.info("Preparing fresh training data...")
            training_data = self._prepare_training_data()
            
            if training_data is None or training_data.empty:
                raise ValueError("Failed to prepare training data")
            
            # 6. Ejecutar reentrenamiento
            logger.info("Executing model retraining...")
            new_model = self._execute_retrain(training_data)
            
            if new_model is None:
                raise ValueError("Model retraining failed")
            
            retrain_results['retrain_executed'] = True
            
            # 7. Generar nueva versión y hash
            new_version = self._generate_new_model_version()
            new_hash = self._calculate_dataset_hash(training_data)
            
            retrain_results['new_model_version'] = new_version
            retrain_results['new_dataset_hash'] = new_hash
            
            # 8. Validar nuevo modelo
            logger.info("Validating retrained model...")
            post_retrain_validation = self.validator.validate_model_quality()
            retrain_results['post_retrain_validation'] = post_retrain_validation
            
            # 9. Comparar performance
            performance_comparison = self._compare_model_performance(
                pre_retrain_validation, post_retrain_validation
            )
            retrain_results['performance_comparison'] = performance_comparison
            
            # 10. Decidir si mantener el nuevo modelo
            if performance_comparison['improvement_significant']:
                logger.info("✓ Retrained model shows significant improvement - keeping new model")
                self._update_model_metadata(new_version, new_hash)
            else:
                logger.warning("⚠ Retrained model shows no significant improvement - reverting to backup")
                self._restore_model_backup()
                retrain_results['reverted_to_backup'] = True
            
            execution_time = datetime.now() - start_time
            retrain_results['execution_time_seconds'] = execution_time.total_seconds()
            
            logger.info(f"Automatic retraining completed successfully in {execution_time}")
            return retrain_results
            
        except Exception as e:
            execution_time = datetime.now() - start_time
            retrain_results['execution_time_seconds'] = execution_time.total_seconds()
            retrain_results['error'] = str(e)
            
            logger.error(f"Error during automatic retraining: {e}")
            
            # Intentar restaurar backup si algo salió mal
            if retrain_results['backup_created']:
                try:
                    self._restore_model_backup()
                    logger.info("Successfully restored model backup after error")
                except Exception as restore_error:
                    logger.error(f"Failed to restore backup after error: {restore_error}")
            
            return retrain_results
    
    def _check_model_age(self) -> Tuple[bool, str]:
        """Verifica la edad del modelo actual."""
        try:
            model_path = self.model_trainer.model_path
            
            if not os.path.exists(model_path):
                return True, "No existing model found"
            
            model_mtime = datetime.fromtimestamp(os.path.getmtime(model_path))
            days_old = (datetime.now() - model_mtime).days
            
            if days_old > self.retrain_criteria['max_days_without_retrain']:
                return True, f"Model is {days_old} days old (max: {self.retrain_criteria['max_days_without_retrain']})"
            
            return False, f"Model age acceptable: {days_old} days"
            
        except Exception as e:
            logger.error(f"Error checking model age: {e}")
            return True, f"Error checking model age: {str(e)}"
    
    def _check_new_data_availability(self) -> Tuple[bool, str]:
        """Verifica si hay nuevos datos disponibles."""
        try:
            # Obtener fecha del último modelo
            model_path = self.model_trainer.model_path
            
            if not os.path.exists(model_path):
                return True, "No existing model - new data available"
            
            model_date = datetime.fromtimestamp(os.path.getmtime(model_path))
            
            # Obtener datos más recientes que el modelo
            historical_data = get_all_draws()
            
            if historical_data.empty:
                return False, "No historical data available"
            
            # Contar sorteos nuevos desde la última actualización del modelo
            historical_data['draw_date'] = pd.to_datetime(historical_data['draw_date'])
            new_draws = historical_data[historical_data['draw_date'] > model_date]
            
            if len(new_draws) >= self.retrain_criteria['min_new_draws']:
                return True, f"{len(new_draws)} new draws available (min: {self.retrain_criteria['min_new_draws']})"
            
            return False, f"Only {len(new_draws)} new draws available"
            
        except Exception as e:
            logger.error(f"Error checking new data availability: {e}")
            return False, f"Error checking data: {str(e)}"
    
    def _check_model_performance(self) -> Tuple[bool, str]:
        """Verifica el performance actual del modelo."""
        try:
            # Obtener métricas de performance recientes
            performance_analytics = get_performance_analytics(7)  # Últimos 7 días
            
            if performance_analytics.get('total_predictions', 0) == 0:
                return False, "No recent predictions to evaluate"
            
            # Verificar métricas clave
            avg_accuracy = performance_analytics.get('avg_accuracy', 0)
            win_rate = performance_analytics.get('win_rate', 0)
            
            # Calcular score combinado
            combined_score = (avg_accuracy * 0.7) + (win_rate * 0.3 / 100)  # win_rate en %
            
            if combined_score < self.retrain_criteria['min_performance_threshold']:
                return True, f"Low performance: {combined_score:.3f} (threshold: {self.retrain_criteria['min_performance_threshold']})"
            
            return False, f"Performance acceptable: {combined_score:.3f}"
            
        except Exception as e:
            logger.error(f"Error checking model performance: {e}")
            return True, f"Error checking performance: {str(e)}"
    
    def _check_data_drift(self) -> Tuple[bool, str]:
        """Verifica si hay cambios significativos en los patrones de datos."""
        try:
            historical_data = get_all_draws()
            
            if len(historical_data) < 100:
                return False, "Insufficient data for drift analysis"
            
            # Comparar distribuciones recientes vs históricas
            recent_data = historical_data.tail(30)  # Últimos 30 sorteos
            older_data = historical_data.iloc[-90:-30]  # 30 sorteos anteriores
            
            if len(older_data) < 20:
                return False, "Insufficient historical data for comparison"
            
            # Analizar distribución de números
            recent_numbers = []
            older_numbers = []
            
            for col in ['n1', 'n2', 'n3', 'n4', 'n5']:
                recent_numbers.extend(recent_data[col].tolist())
                older_numbers.extend(older_data[col].tolist())
            
            # Calcular diferencia en distribuciones
            recent_dist = pd.Series(recent_numbers).value_counts(normalize=True)
            older_dist = pd.Series(older_numbers).value_counts(normalize=True)
            
            # Asegurar que ambas distribuciones tengan los mismos números
            all_numbers = set(recent_numbers + older_numbers)
            recent_probs = [recent_dist.get(num, 0) for num in all_numbers]
            older_probs = [older_dist.get(num, 0) for num in all_numbers]
            
            # Calcular divergencia KL
            kl_divergence = self._calculate_kl_divergence(recent_probs, older_probs)
            
            if kl_divergence > self.retrain_criteria['data_change_threshold']:
                return True, f"Significant data drift detected: KL={kl_divergence:.3f}"
            
            return False, f"Data distribution stable: KL={kl_divergence:.3f}"
            
        except Exception as e:
            logger.error(f"Error checking data drift: {e}")
            return False, f"Error checking drift: {str(e)}"
    
    def _calculate_kl_divergence(self, p: list, q: list) -> float:
        """Calcula la divergencia KL entre dos distribuciones."""
        try:
            p = np.array(p) + 1e-10  # Evitar log(0)
            q = np.array(q) + 1e-10
            
            # Normalizar
            p = p / np.sum(p)
            q = q / np.sum(q)
            
            # KL divergence
            kl = np.sum(p * np.log(p / q))
            return kl
            
        except Exception as e:
            logger.error(f"Error calculating KL divergence: {e}")
            return 0.0
    
    def _create_model_backup(self) -> bool:
        """Crea backup del modelo actual."""
        try:
            model_path = self.model_trainer.model_path
            
            if not os.path.exists(model_path):
                logger.warning("No existing model to backup")
                return True  # No es error si no hay modelo
            
            # Crear directorio de backup
            backup_dir = self.backup_config['backup_dir']
            os.makedirs(backup_dir, exist_ok=True)
            
            # Nombre del backup con timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"shiolplus_backup_{timestamp}.pkl"
            backup_path = os.path.join(backup_dir, backup_name)
            
            # Copiar modelo
            shutil.copy2(model_path, backup_path)
            
            # Limpiar backups antiguos
            self._cleanup_old_backups()
            
            logger.info(f"Model backup created: {backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating model backup: {e}")
            return False
    
    def _cleanup_old_backups(self):
        """Limpia backups antiguos manteniendo solo los más recientes."""
        try:
            backup_dir = self.backup_config['backup_dir']
            
            if not os.path.exists(backup_dir):
                return
            
            # Obtener lista de backups
            backups = [f for f in os.listdir(backup_dir) if f.startswith('shiolplus_backup_')]
            backups.sort(reverse=True)  # Más recientes primero
            
            # Eliminar backups excedentes
            max_backups = self.backup_config['max_backups']
            for backup in backups[max_backups:]:
                backup_path = os.path.join(backup_dir, backup)
                os.remove(backup_path)
                logger.info(f"Removed old backup: {backup}")
                
        except Exception as e:
            logger.error(f"Error cleaning up old backups: {e}")
    
    def _prepare_training_data(self) -> Optional[pd.DataFrame]:
        """Prepara datos actualizados para entrenamiento."""
        try:
            # Cargar datos históricos más recientes
            data_loader = get_data_loader()
            historical_data = data_loader.load_historical_data()
            
            if historical_data.empty:
                logger.error("No historical data available for training")
                return None
            
            # Generar features con análisis temporal
            feature_engineer = FeatureEngineer(historical_data)
            features_df = feature_engineer.engineer_features(use_temporal_analysis=True)
            
            logger.info(f"Training data prepared: {len(features_df)} samples with features")
            return features_df
            
        except Exception as e:
            logger.error(f"Error preparing training data: {e}")
            return None
    
    def _execute_retrain(self, training_data: pd.DataFrame):
        """Ejecuta el reentrenamiento del modelo."""
        try:
            logger.info("Starting model retraining...")
            
            # Crear nuevo trainer para el reentrenamiento
            trainer = ModelTrainer(self.model_trainer.model_path)
            
            # Entrenar modelo
            new_model = trainer.train(training_data)
            
            if new_model is None:
                raise ValueError("Model training returned None")
            
            logger.info("Model retraining completed successfully")
            return new_model
            
        except Exception as e:
            logger.error(f"Error during model retraining: {e}")
            raise
    
    def _generate_new_model_version(self) -> str:
        """Genera nueva versión del modelo."""
        try:
            # Formato: v1.0.YYYYMMDD_HHMMSS
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            return f"v1.0.{timestamp}"
            
        except Exception as e:
            logger.error(f"Error generating model version: {e}")
            return f"v1.0.unknown_{datetime.now().strftime('%Y%m%d')}"
    
    def _calculate_dataset_hash(self, training_data: pd.DataFrame) -> str:
        """Calcula hash del dataset de entrenamiento."""
        try:
            # Crear string representativo del dataset
            data_string = ""
            
            # Incluir forma y columnas
            data_string += f"shape:{training_data.shape}"
            data_string += f"columns:{','.join(training_data.columns)}"
            
            # Incluir muestra de datos
            sample_data = training_data.head(10).to_string()
            data_string += sample_data
            
            # Calcular hash SHA256
            hash_obj = hashlib.sha256(data_string.encode('utf-8'))
            dataset_hash = hash_obj.hexdigest()[:16]
            
            return dataset_hash
            
        except Exception as e:
            logger.error(f"Error calculating dataset hash: {e}")
            return f"hash_error_{datetime.now().strftime('%Y%m%d')}"
    
    def _compare_model_performance(self, pre_validation: Dict, post_validation: Dict) -> Dict:
        """Compara el rendimiento antes y después del reentrenamiento."""
        try:
            pre_score = pre_validation.get('quality_assessment', {}).get('overall_score', 0)
            post_score = post_validation.get('quality_assessment', {}).get('overall_score', 0)
            
            improvement = post_score - pre_score
            improvement_percent = (improvement / pre_score * 100) if pre_score > 0 else 0
            
            # Considerar mejora significativa si es > 5%
            significant = improvement_percent > 5.0
            
            comparison = {
                'pre_retrain_score': pre_score,
                'post_retrain_score': post_score,
                'improvement': improvement,
                'improvement_percent': improvement_percent,
                'improvement_significant': significant,
                'recommendation': 'keep_new' if significant else 'revert_to_backup'
            }
            
            logger.info(f"Performance comparison: {improvement:+.3f} ({improvement_percent:+.1f}%)")
            return comparison
            
        except Exception as e:
            logger.error(f"Error comparing model performance: {e}")
            return {
                'improvement_significant': False,
                'recommendation': 'revert_to_backup',
                'error': str(e)
            }
    
    def _restore_model_backup(self) -> bool:
        """Restaura el backup más reciente del modelo."""
        try:
            backup_dir = self.backup_config['backup_dir']
            
            if not os.path.exists(backup_dir):
                logger.error("No backup directory found")
                return False
            
            # Encontrar backup más reciente
            backups = [f for f in os.listdir(backup_dir) if f.startswith('shiolplus_backup_')]
            
            if not backups:
                logger.error("No backups found")
                return False
            
            latest_backup = sorted(backups)[-1]
            backup_path = os.path.join(backup_dir, latest_backup)
            
            # Restaurar backup
            shutil.copy2(backup_path, self.model_trainer.model_path)
            
            logger.info(f"Model restored from backup: {latest_backup}")
            return True
            
        except Exception as e:
            logger.error(f"Error restoring model backup: {e}")
            return False
    
    def _update_model_metadata(self, version: str, dataset_hash: str):
        """Actualiza metadatos del modelo."""
        try:
            # Crear archivo de metadatos
            metadata = {
                'model_version': version,
                'dataset_hash': dataset_hash,
                'retrain_timestamp': datetime.now().isoformat(),
                'retrain_method': 'automatic'
            }
            
            metadata_path = self.model_trainer.model_path.replace('.pkl', '_metadata.json')
            
            import json
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Model metadata updated: version={version}, hash={dataset_hash}")
            
        except Exception as e:
            logger.error(f"Error updating model metadata: {e}")


def execute_automatic_retrain_if_needed() -> Dict:
    """
    Función de conveniencia para ejecutar reentrenamiento automático si es necesario.
    
    Returns:
        Dict con resultados del proceso
    """
    retrainer = AutoRetrainer()
    return retrainer.execute_automatic_retrain(force=False)


def force_model_retrain() -> Dict:
    """
    Función para forzar reentrenamiento del modelo.
    
    Returns:
        Dict con resultados del reentrenamiento
    """
    retrainer = AutoRetrainer()
    return retrainer.execute_automatic_retrain(force=True)
"""
SHIOL+ Auto-Retrainer Module
============================

Sistema automático de reentrenamiento del modelo cuando la calidad baja.
"""

import os
from datetime import datetime
from loguru import logger
from typing import Dict, Any

from src.predictor import ModelTrainer
from src.loader import get_data_loader
from src.intelligent_generator import FeatureEngineer


def execute_automatic_retrain_if_needed() -> Dict[str, Any]:
    """
    Ejecuta reentrenamiento automático del modelo si es necesario.
    
    Returns:
        Dict con resultados del reentrenamiento
    """
    logger.info("Checking if automatic model retraining is needed...")
    
    try:
        # Verificar si existe un modelo actual
        model_path = "models/shiolplus.pkl"
        if not os.path.exists(model_path):
            logger.info("No existing model found, training new model...")
            return _train_new_model()
        
        # Por ahora, no reentrenamos automáticamente
        # En el futuro se puede implementar lógica más sofisticada
        logger.info("Automatic retraining not triggered - model exists")
        
        return {
            'retrain_executed': False,
            'reason': 'model_exists_no_trigger',
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in automatic retraining check: {e}")
        return {
            'retrain_executed': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }


def _train_new_model() -> Dict[str, Any]:
    """Entrena un nuevo modelo desde cero."""
    try:
        logger.info("Starting new model training...")
        
        # Cargar datos
        data_loader = get_data_loader()
        historical_data = data_loader.load_historical_data()
        
        if historical_data.empty:
            raise ValueError("No historical data available for training")
        
        # Generar features
        feature_engineer = FeatureEngineer(historical_data)
        features_df = feature_engineer.engineer_features(use_temporal_analysis=True)
        
        # Entrenar modelo
        model_trainer = ModelTrainer("models/shiolplus.pkl")
        model = model_trainer.train(features_df)
        
        if model is not None:
            logger.info("New model trained successfully")
            return {
                'retrain_executed': True,
                'reason': 'new_model_trained',
                'model_path': 'models/shiolplus.pkl',
                'training_samples': len(features_df),
                'timestamp': datetime.now().isoformat()
            }
        else:
            logger.error("Model training failed")
            return {
                'retrain_executed': False,
                'error': 'training_failed',
                'timestamp': datetime.now().isoformat()
            }
            
    except Exception as e:
        logger.error(f"Error training new model: {e}")
        return {
            'retrain_executed': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }


def check_model_health() -> Dict[str, Any]:
    """
    Verifica la salud del modelo actual.
    
    Returns:
        Dict con estado de salud del modelo
    """
    try:
        model_path = "models/shiolplus.pkl"
        
        health_status = {
            'model_exists': os.path.exists(model_path),
            'model_path': model_path,
            'check_timestamp': datetime.now().isoformat()
        }
        
        if health_status['model_exists']:
            # Verificar que el modelo se puede cargar
            model_trainer = ModelTrainer(model_path)
            model = model_trainer.load_model()
            
            health_status['model_loadable'] = model is not None
            health_status['model_size_mb'] = os.path.getsize(model_path) / (1024 * 1024)
        else:
            health_status['model_loadable'] = False
            health_status['model_size_mb'] = 0
        
        return health_status
        
    except Exception as e:
        logger.error(f"Error checking model health: {e}")
        return {
            'model_exists': False,
            'model_loadable': False,
            'error': str(e),
            'check_timestamp': datetime.now().isoformat()
        }

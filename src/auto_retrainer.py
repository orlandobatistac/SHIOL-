"""
SHIOL+ Auto-Retrainer Module
============================

Sistema automático de reentrenamiento del modelo cuando la calidad baja.
"""

import configparser
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Tuple, Dict, Any, List, Optional

# Add the project root to Python path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import joblib
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
from sklearn.model_selection import cross_val_score

from src.database import get_all_draws, get_performance_analytics
from src.loader import get_data_loader
from src.intelligent_generator import FeatureEngineer
# Removed unavailable import: from src.predictor import get_model_trainer


class AutoRetrainer:
    """
    Sistema automático de reentrenamiento del modelo basado en criterios
    de performance y antigüedad.
    """

    def __init__(self, config_path: str = "config/config.ini"):
        self.config_path = config_path
        self.config = configparser.ConfigParser()
        self.config.read(self.config_path)

        # Criterios para reentrenamiento
        self.retrain_criteria = {
            'min_new_draws': self.config.getint("retrain_criteria", "min_new_draws", fallback=10),
            'max_days_without_retrain': self.config.getint("retrain_criteria", "max_days_without_retrain", fallback=30),
            'min_performance_threshold': self.config.getfloat("retrain_criteria", "min_performance_threshold", fallback=0.3),
            'data_change_threshold': self.config.getfloat("retrain_criteria", "data_change_threshold", fallback=0.15)
        }

        # Configuración de respaldo
        self.backup_config = {
            'max_backups': self.config.getint("backup", "max_backups", fallback=5),
            'backup_dir': self.config.get("backup", "backup_dir", fallback="models/backups")
        }
        os.makedirs(self.backup_config['backup_dir'], exist_ok=True)

        logger.info("AutoRetrainer initialized.")
        logger.debug(f"Retrain criteria: {self.retrain_criteria}")
        logger.debug(f"Backup config: {self.backup_config}")

    def should_retrain_model(self) -> Tuple[bool, List[str]]:
        """
        Determina si el modelo necesita ser reentrenado.

        Returns:
            Tuple[bool, List[str]]: (needs_retrain, reasons)
        """
        logger.info("Evaluating if model retraining is needed...")

        reasons = []
        should_retrain = False

        try:
            # 1. Verificar si existe un modelo
            model_path = self.config.get("paths", "model_file", fallback="models/shiolplus.pkl")
            if not os.path.exists(model_path):
                logger.warning("No existing model file found.")
                reasons.append("No existing model file")
                should_retrain = True
            else:
                # Load and validate model
                try:
                    from src.predictor import ModelTrainer
                    model_trainer = ModelTrainer(model_path)

                    if not model_trainer.load_model():
                        issues.append("Model file not found or corrupted")
                        should_retrain = True
                except Exception as e:
                    logger.error(f"Error loading model: {e}")
                    issues.append(f"Model loading error: {str(e)}")
                    should_retrain = True

            # 2. Verificar edad del modelo
            model_age_check, model_age_reason = self._check_model_age()
            if model_age_check:
                should_retrain = True
                reasons.append(model_age_reason)

            # 3. Verificar nuevos datos disponibles
            new_data_check, new_data_reason = self._check_new_data_availability()
            if new_data_check:
                should_retrain = True
                reasons.append(new_data_reason)

            # 4. Verificar performance del modelo
            performance_check, performance_reason = self._check_model_performance()
            if performance_check:
                should_retrain = True
                reasons.append(performance_reason)

            # 5. Verificar cambios significativos en datos (data drift)
            data_change_check, data_change_reason = self._check_data_drift()
            if data_change_check:
                should_retrain = True
                reasons.append(data_change_reason)

            logger.info(f"Retrain evaluation: {'NEEDED' if should_retrain else 'NOT NEEDED'}")
            if reasons:
                logger.info(f"Retrain reasons: {', '.join(reasons)}")

            return should_retrain, reasons

        except Exception as e:
            logger.error(f"Error evaluating retrain necessity: {e}")
            return True, [f"Error in evaluation: {str(e)} - Recommending retrain for safety"]

    def retrain_model(self) -> Dict[str, Any]:
        """
        Entrena un nuevo modelo con los datos más recientes.

        Returns:
            Dict con resultados del reentrenamiento
        """
        try:
            logger.info("Starting automatic model retrain...")

            # Load fresh data
            data_loader = get_data_loader()
            historical_data = data_loader.load_historical_data()

            if historical_data.empty:
                return {
                    'retrain_executed': False,
                    'error': 'No historical data available for retraining',
                    'timestamp': datetime.now().isoformat()
                }

            # Engineer features
            feature_engineer = FeatureEngineer(historical_data)
            features_df = feature_engineer.engineer_features(use_temporal_analysis=True)

            # Train new model
            from src.predictor import ModelTrainer
            model_path = self.config.get("paths", "model_file", fallback="models/shiolplus.pkl")
            model_trainer = ModelTrainer(model_path)
            new_model = model_trainer.train(features_df)

            if new_model is not None:
                logger.info("New model trained successfully.")
                return {
                    'retrain_executed': True,
                    'model_path': model_path,
                    'training_samples': len(features_df),
                    'timestamp': datetime.now().isoformat()
                }
            else:
                logger.error("Model training failed.")
                return {
                    'retrain_executed': False,
                    'error': 'model_training_failed',
                    'timestamp': datetime.now().isoformat()
                }

        except Exception as e:
            logger.error(f"Error during model retraining: {e}")
            return {
                'retrain_executed': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    def validate_model_performance(self) -> Dict[str, Any]:
        """
        Valida el performance del modelo actual.

        Returns:
            Dict con la validación del modelo
        """
        try:
            # Load model
            from src.predictor import ModelTrainer
            model_path = self.config.get("paths", "model_file", fallback="models/shiolplus.pkl")
            model_trainer = ModelTrainer(model_path)
            model = model_trainer.load_model()

            if not model:
                return {
                    'performance_score': 0.0,
                    'issues': ['Model not available for validation'],
                    'validation_successful': False
                }

            # Get recent performance analytics
            # We'll use a simple approach for now, assuming get_performance_analytics provides relevant metrics
            performance_data = get_performance_analytics(days=7) # Last 7 days

            if not performance_data or performance_data.get('total_predictions', 0) == 0:
                return {
                    'performance_score': 0.0,
                    'issues': ['No recent performance data available'],
                    'validation_successful': False
                }

            # Calculate a simple performance score (e.g., weighted accuracy and win rate)
            avg_accuracy = performance_data.get('avg_accuracy', 0.0)
            win_rate = performance_data.get('win_rate', 0.0) # Assuming win_rate is a percentage

            # Example: weighting accuracy higher
            performance_score = (avg_accuracy * 0.6) + (win_rate * 0.4)

            issues = []
            if performance_score < self.retrain_criteria['min_performance_threshold']:
                issues.append(f"Performance score ({performance_score:.3f}) is below threshold ({self.retrain_criteria['min_performance_threshold']})")

            return {
                'performance_score': performance_score,
                'issues': issues,
                'validation_successful': True
            }

        except Exception as e:
            logger.error(f"Error validating model performance: {e}")
            return {
                'performance_score': 0.0,
                'issues': [f"An error occurred during validation: {str(e)}"],
                'validation_successful': False
            }

    def _check_model_age(self) -> Tuple[bool, str]:
        """Verifica la edad del modelo actual."""
        try:
            model_path = self.config.get("paths", "model_file", fallback="models/shiolplus.pkl")

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
            # Obtener fecha del último modelo (o una fecha de referencia si no hay modelo)
            model_path = self.config.get("paths", "model_file", fallback="models/shiolplus.pkl")
            reference_date = datetime.now() # Default to now if no model exists

            if os.path.exists(model_path):
                reference_date = datetime.fromtimestamp(os.path.getmtime(model_path))
            else:
                # If no model, we consider all data as "new" if it meets the minimum draw count
                pass

            # Obtener datos históricos
            historical_data = get_all_draws()

            if historical_data.empty:
                return False, "No historical data available"

            # Contar sorteos nuevos desde la fecha de referencia
            historical_data['draw_date'] = pd.to_datetime(historical_data['draw_date'])
            new_draws = historical_data[historical_data['draw_date'] > reference_date]

            if len(new_draws) >= self.retrain_criteria['min_new_draws']:
                return True, f"{len(new_draws)} new draws available (min: {self.retrain_criteria['min_new_draws']})"

            return False, f"Only {len(new_draws)} new draws available"

        except Exception as e:
            logger.error(f"Error checking new data availability: {e}")
            return False, f"Error checking data: {str(e)}"

    def _check_model_performance(self) -> Tuple[bool, str]:
        """Verifica el performance actual del modelo."""
        try:
            validation_results = self.validate_model_performance()

            if not validation_results['validation_successful']:
                return True, f"Performance check failed: {validation_results.get('issues', ['Unknown error'])[0]}"

            score = validation_results['performance_score']

            if score < self.retrain_criteria['min_performance_threshold']:
                return True, f"Low performance: {score:.3f} (threshold: {self.retrain_criteria['min_performance_threshold']})"

            return False, f"Performance acceptable: {score:.3f}"

        except Exception as e:
            logger.error(f"Error checking model performance: {e}")
            return True, f"Error checking performance: {str(e)}"

    def _check_data_drift(self) -> Tuple[bool, str]:
        """Verifica si hay cambios significativos en los patrones de datos."""
        try:
            historical_data = get_all_draws()

            if len(historical_data) < 100: # Need sufficient data for comparison
                return False, "Insufficient data for drift analysis"

            # Use tail for recent data and slice for older data
            recent_data = historical_data.tail(30)  # Last 30 draws
            older_data = historical_data.iloc[-90:-30]  # 30 draws prior to the recent ones

            if len(older_data) < 20: # Need sufficient older data for comparison
                return False, "Insufficient historical data for comparison"

            # Analyze distribution of numbers (example for 'n1' to 'n5')
            num_cols = [col for col in historical_data.columns if col.startswith('n')]
            recent_numbers = recent_data[num_cols].values.flatten()
            older_numbers = older_data[num_cols].values.flatten()

            if len(recent_numbers) == 0 or len(older_numbers) == 0:
                return False, "No numeric data found for drift analysis"

            # Calculate frequency distributions
            recent_dist = pd.Series(recent_numbers).value_counts(normalize=True)
            older_dist = pd.Series(older_numbers).value_counts(normalize=True)

            # Ensure both distributions cover the same set of numbers
            all_numbers = sorted(list(set(recent_numbers) | set(older_numbers)))
            recent_probs = [recent_dist.get(num, 0) for num in all_numbers]
            older_probs = [older_dist.get(num, 0) for num in all_numbers]

            # Calculate KL divergence
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

            # Normalize to ensure they sum to 1
            p = p / np.sum(p)
            q = q / np.sum(q)

            # KL divergence: sum(p * log(p / q))
            kl = np.sum(p * np.log(p / q))
            return kl

        except Exception as e:
            logger.error(f"Error calculating KL divergence: {e}")
            return 0.0 # Return 0 if calculation fails

    def _create_model_backup(self) -> bool:
        """Crea backup del modelo actual."""
        try:
            model_path = self.config.get("paths", "model_file", fallback="models/shiolplus.pkl")

            if not os.path.exists(model_path):
                logger.warning("No existing model to backup")
                return True  # Not an error if no model exists

            backup_dir = self.backup_config['backup_dir']
            os.makedirs(backup_dir, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"shiolplus_backup_{timestamp}.pkl"
            backup_path = os.path.join(backup_dir, backup_name)

            shutil.copy2(model_path, backup_path)

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

            backups = [f for f in os.listdir(backup_dir) if f.startswith('shiolplus_backup_')]
            backups.sort(reverse=True)  # Sort by date (newest first)

            max_backups = self.backup_config['max_backups']
            for backup in backups[max_backups:]:
                backup_path = os.path.join(backup_dir, backup)
                os.remove(backup_path)
                logger.info(f"Removed old backup: {backup}")

        except Exception as e:
            logger.error(f"Error cleaning up old backups: {e}")

    def _restore_model_backup(self) -> bool:
        """Restaura el backup más reciente del modelo."""
        try:
            backup_dir = self.backup_config['backup_dir']
            model_path = self.config.get("paths", "model_file", fallback="models/shiolplus.pkl")

            if not os.path.exists(backup_dir):
                logger.error("No backup directory found.")
                return False

            backups = [f for f in os.listdir(backup_dir) if f.startswith('shiolplus_backup_')]
            if not backups:
                logger.error("No backups found to restore.")
                return False

            latest_backup = sorted(backups)[-1] # Get the most recent backup
            backup_path = os.path.join(backup_dir, latest_backup)

            shutil.copy2(backup_path, model_path)

            logger.info(f"Model restored from backup: {latest_backup}")
            return True

        except Exception as e:
            logger.error(f"Error restoring model backup: {e}")
            return False

    def _update_model_metadata(self, version: str, dataset_hash: str):
        """Actualiza metadatos del modelo."""
        try:
            metadata = {
                'model_version': version,
                'dataset_hash': dataset_hash,
                'retrain_timestamp': datetime.now().isoformat(),
                'retrain_method': 'automatic' # Or could be 'manual' if forced
            }

            model_path = self.config.get("paths", "model_file", fallback="models/shiolplus.pkl")
            metadata_path = model_path.replace('.pkl', '_metadata.json')

            import json
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            logger.info(f"Model metadata updated: version={version}, hash={dataset_hash}")

        except Exception as e:
            logger.error(f"Error updating model metadata: {e}")


def execute_automatic_retrain_if_needed() -> Dict[str, Any]:
    """
    Ejecuta reentrenamiento automático del modelo si es necesario.

    Returns:
        Dict con resultados del reentrenamiento
    """
    logger.info("Checking if automatic model retraining is needed...")
    retrainer = AutoRetrainer()

    needs_retrain, reasons = retrainer.should_retrain_model()

    if needs_retrain:
        logger.info("Automatic retraining is required.")
        retrain_results = retrainer.retrain_model()
        # Optionally, update metadata and perform validation after retraining
        # For now, just returning the retraining results.
        return retrain_results
    else:
        logger.info("Automatic retraining is not required at this time.")
        return {
            'retrain_executed': False,
            'reason': 'criteria_not_met',
            'timestamp': datetime.now().isoformat()
        }


def force_model_retrain() -> Dict[str, Any]:
    """
    Fuerza el reentrenamiento del modelo, independientemente de los criterios.

    Returns:
        Dict con resultados del reentrenamiento
    """
    logger.info("Forcing model retraining...")
    retrainer = AutoRetrainer()
    retrain_results = retrainer.retrain_model()
    # Optionally, update metadata and perform validation after retraining
    return retrain_results


def check_model_health() -> Dict[str, Any]:
    """
    Verifica la salud del modelo actual.

    Returns:
        Dict con estado de salud del modelo
    """
    try:
        retrainer = AutoRetrainer()
        model_path = retrainer.config.get("paths", "model_file", fallback="models/shiolplus.pkl")

        health_status = {
            'model_exists': os.path.exists(model_path),
            'model_path': model_path,
            'check_timestamp': datetime.now().isoformat()
        }

        if health_status['model_exists']:
            # Verificar que el modelo se puede cargar
            model_trainer = ModelTrainer(model_path) # Use the correct ModelTrainer class
            model = model_trainer.load_model()

            health_status['model_loadable'] = model is not None
            if model is not None:
                health_status['model_size_mb'] = os.path.getsize(model_path) / (1024 * 1024)
            else:
                health_status['model_size_mb'] = 0
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
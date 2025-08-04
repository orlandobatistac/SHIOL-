import configparser
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from loguru import logger
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from xgboost import XGBClassifier
from typing import Dict, List

from src.loader import get_data_loader
from src.intelligent_generator import FeatureEngineer, DeterministicGenerator
from src.database import save_prediction_log

class ModelTrainer:
    def __init__(self, model_path):
        self.model_path = model_path
        self.config = configparser.ConfigParser()
        self.config.read("config/config.ini")
        self.model = None
        self.target_columns = None
        self.white_ball_columns = [f"n{i}" for i in range(1, 6)]
        self.pb_column = "pb"
        logger.info("ModelTrainer v2.0 initialized.")

    def create_target_variable(self, features_df):
        logger.info("Creating multi-label target variable 'y'...")
        
        white_ball_range = range(1, 70)
        pb_range = range(1, 27)

        wb_cols = [f"wb_{i}" for i in white_ball_range]
        pb_cols = [f"pb_{i}" for i in pb_range]
        
        y = pd.DataFrame(0, index=features_df.index, columns=wb_cols + pb_cols)

        for i in white_ball_range:
            y[f"wb_{i}"] = features_df[self.white_ball_columns].eq(i).any(axis=1).astype(int)

        for i in pb_range:
            y[f"pb_{i}"] = (features_df[self.pb_column] == i).astype(int)
        
        logger.info(f"Target variable 'y' created with shape: {y.shape}")
        return y

    def train(self, features_df):
        logger.info("Starting model training with multi-label classification objective...")

        X = self._get_feature_matrix(features_df)
        y = self.create_target_variable(features_df)
        self.target_columns = y.columns.tolist()

        combined = pd.concat([X, y], axis=1).dropna()
        X = combined[X.columns]
        y = combined[y.columns]
        
        if X.empty or y.empty:
           logger.error("Not enough data to train after dropping NaNs.")
           return None

        X_train, X_test, y_train, y_test = self._split_train_test_data(X, y)

        self.model = self._initialize_model()
        self.model.fit(X_train, y_train)
        logger.info("Model training complete.")

        self.evaluate_model(X_test, y_test)
        self.save_model()
        return self.model

    def _get_feature_matrix(self, features_df):
        feature_cols = [
            "even_count", "odd_count", "sum", "spread", "consecutive_count",
            "avg_delay", "max_delay", "min_delay",
            "dist_to_recent", "avg_dist_to_top_n", "dist_to_centroid",
            "time_weight", "increasing_trend_count", "decreasing_trend_count",
            "stable_trend_count"
        ]
        available_features = [col for col in feature_cols if col in features_df.columns]
        logger.info(f"Using features: {available_features}")
        return features_df[available_features]

    def _split_train_test_data(self, X, y):
        return train_test_split(
            X, y,
            test_size=float(self.config["model_params"]["test_size"]),
            random_state=int(self.config["model_params"]["random_state"])
        )

    def _initialize_model(self):
        base_classifier = XGBClassifier(
            n_estimators=int(self.config["model_params"]["n_estimators"]),
            learning_rate=float(self.config["model_params"].get("learning_rate", 0.1)),
            random_state=int(self.config["model_params"]["random_state"]),
            objective='binary:logistic',
            use_label_encoder=False,
            eval_metric='logloss'
        )
        return MultiOutputClassifier(estimator=base_classifier, n_jobs=-1)

    def evaluate_model(self, X_test, y_test):
        if self.model is None:
            logger.error("Model not trained. Cannot evaluate.")
            return None

        y_pred_proba = self.model.predict_proba(X_test)
        y_pred_proba_flat = np.array([arr[:, 1] for arr in y_pred_proba]).T

        try:
            auc = roc_auc_score(y_test, y_pred_proba_flat, average='macro')
            ll = log_loss(y_test, y_pred_proba_flat)
            
            logger.info("Model evaluation metrics:")
            logger.info(f"  Macro-Averaged AUC: {auc:.4f}")
            logger.info(f"  Log Loss: {ll:.4f}")

            return {"roc_auc_macro": auc, "log_loss": ll}
        except Exception as e:
            logger.error(f"Could not calculate evaluation metrics: {e}")
            return None

    def save_model(self):
        if self.model:
            model_bundle = {
                "model": self.model,
                "target_columns": self.target_columns
            }
            joblib.dump(model_bundle, self.model_path)
            logger.info(f"Model bundle saved to {self.model_path}")

    def load_model(self):
        try:
            model_bundle = joblib.load(self.model_path)
            self.model = model_bundle["model"]
            self.target_columns = model_bundle.get("target_columns")
            if self.target_columns is None:
                logger.warning("Loaded a model without target_columns. Prediction may fail.")
            logger.info(f"Model bundle loaded from {self.model_path}")
            return self.model
        except FileNotFoundError:
            logger.error(f"Model file not found at {self.model_path}. A new one will be created upon training.")
            return None
        except (KeyError, AttributeError):
             logger.warning(f"Failed to load model bundle from {self.model_path}, it might be an old version. Will create a new model.")
             return None

    def predict_probabilities(self, features_df):
        if self.model is None or self.target_columns is None:
            logger.error("Model or target columns not loaded. Cannot predict.")
            return None

        X = self._validate_prediction_features(features_df)
        if X is None:
            logger.error("Feature validation failed during prediction.")
            return None

        pred_probas = self.model.predict_proba(X)
        prob_class_1 = [p[:, 1] for p in pred_probas]
        
        prob_df = pd.DataFrame(np.array(prob_class_1).T, columns=self.target_columns, index=X.index)
        return prob_df
        
    def _validate_prediction_features(self, features_df):
        try:
            model_features = self.model.estimators_[0].get_booster().feature_names
            
            missing_features = [f for f in model_features if f not in features_df.columns]
            if missing_features:
                logger.warning(f"Missing features for prediction: {missing_features}")
                return None
            
            return features_df[model_features]
        except Exception as e:
            logger.error(f"Error validating prediction features: {e}")
            return None

class Predictor:
    """
    Handles the prediction of number probabilities using the trained model.
    """
    def __init__(self):
        logger.info("Initializing Predictor...")
        self.model_trainer = get_model_trainer()
        self.model = self.model_trainer.load_model()
        if self.model is None:
            raise RuntimeError("Failed to load the model. Please train a model first using 'train' command.")

    def get_prediction_features(self):
        """
        Prepares the feature set for the next draw to be predicted.
        """
        logger.info("Preparing features for the next prediction...")
        data_loader = get_data_loader()
        historical_data = data_loader.load_historical_data()
        
        if historical_data.empty:
            raise ValueError("Historical data is empty. Cannot prepare features for prediction.")

        feature_engineer = FeatureEngineer(historical_data)
        all_features = feature_engineer.engineer_features(use_temporal_analysis=True)
        
        latest_features = all_features.iloc[-1:]
        
        logger.info("Successfully prepared features for one prediction row.")
        return latest_features

    def predict_probabilities(self):
        """
        Predicts the probability of each number appearing in the next draw.
        """
        logger.info("Predicting number probabilities for the next draw...")
        
        features_df = self.get_prediction_features()
        prob_df = self.model_trainer.predict_probabilities(features_df)
        
        if prob_df is None:
            raise RuntimeError("Prediction failed. The model did not return probabilities.")
            
        prob_series = prob_df.iloc[0]
        
        wb_probs = {int(col.split('_')[1]): prob for col, prob in prob_series.items() if col.startswith('wb_')}
        pb_probs = {int(col.split('_')[1]): prob for col, prob in prob_series.items() if col.startswith('pb_')}

        logger.info("Probabilities predicted successfully.")
        return wb_probs, pb_probs

    def predict_deterministic(self, save_to_log: bool = True) -> Dict:
        """
        Genera una predicción determinística usando el sistema de scoring multi-criterio.
        
        Args:
            save_to_log: Si True, guarda la predicción en el log de base de datos
            
        Returns:
            Dict con la predicción determinística y sus detalles
        """
        logger.info("Generating deterministic prediction with multi-criteria scoring...")
        
        # Obtener probabilidades del modelo
        wb_probs, pb_probs = self.predict_probabilities()
        
        # Obtener datos históricos para el generador determinístico
        data_loader = get_data_loader()
        historical_data = data_loader.load_historical_data()
        
        if historical_data.empty:
            raise ValueError("Historical data is empty. Cannot generate deterministic prediction.")
        
        # Crear generador determinístico
        deterministic_generator = DeterministicGenerator(historical_data)
        
        # Generar predicción top
        prediction = deterministic_generator.generate_top_prediction(wb_probs, pb_probs)
        
        # Guardar en log si se solicita
        if save_to_log:
            prediction_id = save_prediction_log(prediction)
            if prediction_id:
                prediction['log_id'] = prediction_id
                logger.info(f"Deterministic prediction saved to log with ID: {prediction_id}")
            else:
                logger.warning("Failed to save prediction to log")
        
        logger.info("Deterministic prediction generated successfully")
        return prediction
    
    def predict_diverse_plays(self, num_plays: int = 5, save_to_log: bool = True) -> List[Dict]:
        """
        Genera múltiples predicciones diversas de alta calidad para el próximo sorteo.
        
        Args:
            num_plays: Número de plays diversos a generar (default: 5)
            save_to_log: Si True, guarda las predicciones en el log de base de datos
            
        Returns:
            Lista de Dict con las predicciones diversas y sus detalles
        """
        logger.info(f"Generating {num_plays} diverse high-quality plays for next drawing...")
        
        # Obtener probabilidades del modelo
        wb_probs, pb_probs = self.predict_probabilities()
        
        # Obtener datos históricos para el generador determinístico
        data_loader = get_data_loader()
        historical_data = data_loader.load_historical_data()
        
        if historical_data.empty:
            raise ValueError("Historical data is empty. Cannot generate diverse predictions.")
        
        # Crear generador determinístico
        deterministic_generator = DeterministicGenerator(historical_data)
        
        # Generar predicciones diversas
        diverse_predictions = deterministic_generator.generate_diverse_predictions(
            wb_probs, pb_probs, num_plays=num_plays
        )
        
        # Guardar en log si se solicita
        if save_to_log:
            saved_count = 0
            for i, prediction in enumerate(diverse_predictions):
                prediction_id = save_prediction_log(prediction)
                if prediction_id:
                    prediction['log_id'] = prediction_id
                    saved_count += 1
                else:
                    logger.warning(f"Failed to save prediction {i+1} to log")
            
            logger.info(f"Saved {saved_count}/{len(diverse_predictions)} diverse predictions to log")
        
        logger.info(f"Generated {len(diverse_predictions)} diverse plays successfully")
        return diverse_predictions

    def predict_syndicate_plays(self, num_plays: int = 100, save_to_log: bool = True) -> List[Dict]:
        """
        Genera un gran número de predicciones optimizadas para juego en sindicato.
        
        Args:
            num_plays: Número de jugadas para sindicato (default: 100)
            save_to_log: Si True, guarda las predicciones en el log
            
        Returns:
            Lista de predicciones optimizadas para sindicato
        """
        logger.info(f"Generating {num_plays} syndicate plays with advanced scoring...")
        
        # Obtener probabilidades del modelo
        wb_probs, pb_probs = self.predict_probabilities()
        
        # Obtener datos históricos
        data_loader = get_data_loader()
        historical_data = data_loader.load_historical_data()
        
        if historical_data.empty:
            raise ValueError("Historical data is empty. Cannot generate syndicate predictions.")
        
        # Crear generador con más candidatos para mayor diversidad
        deterministic_generator = DeterministicGenerator(historical_data)
        
        # Generar un pool más grande de candidatos (5x el número solicitado)
        candidate_pool_size = max(5000, num_plays * 5)
        
        # Generar candidatos con scoring avanzado
        syndicate_predictions = deterministic_generator.generate_diverse_predictions(
            wb_probs, pb_probs, 
            num_plays=num_plays,
            num_candidates=candidate_pool_size
        )
        
        # Aplicar análisis adicional para sindicatos
        for i, prediction in enumerate(syndicate_predictions):
            prediction['syndicate_rank'] = i + 1
            prediction['syndicate_tier'] = self._classify_syndicate_tier(prediction['score_total'])
            prediction['expected_coverage'] = self._calculate_coverage_score(prediction, syndicate_predictions)
        
        # Guardar en log si se solicita
        if save_to_log:
            saved_count = 0
            for prediction in syndicate_predictions:
                prediction_id = save_prediction_log(prediction)
                if prediction_id:
                    prediction['log_id'] = prediction_id
                    saved_count += 1
            
            logger.info(f"Saved {saved_count}/{len(syndicate_predictions)} syndicate predictions to log")
        
        logger.info(f"Generated {len(syndicate_predictions)} syndicate plays successfully")
        return syndicate_predictions
    
    def _classify_syndicate_tier(self, score: float) -> str:
        """Clasifica las jugadas por tiers para sindicatos."""
        if score >= 0.8:
            return "Premium"
        elif score >= 0.6:
            return "High"
        elif score >= 0.4:
            return "Medium"
        else:
            return "Standard"
    
    def _calculate_coverage_score(self, prediction: Dict, all_predictions: List[Dict]) -> float:
        """Calcula qué tan bien una jugada complementa las otras en el sindicato."""
        numbers = set(prediction['numbers'] + [prediction['powerball']])
        
        coverage_scores = []
        for other_pred in all_predictions[:20]:  # Comparar con las top 20
            if other_pred == prediction:
                continue
            
            other_numbers = set(other_pred['numbers'] + [other_pred['powerball']])
            overlap = len(numbers.intersection(other_numbers))
            coverage = 1.0 - (overlap / 6.0)  # Menos overlap = mejor coverage
            coverage_scores.append(coverage)
        
        return np.mean(coverage_scores) if coverage_scores else 0.5

    def predict_ensemble_syndicate(self, num_plays: int = 100) -> List[Dict]:
        """
        Genera jugadas usando múltiples modelos y selecciona las mejores basado en ensemble scoring.
        
        Args:
            num_plays: Número final de jugadas a generar
            
        Returns:
            Lista de las mejores jugadas seleccionadas de múltiples modelos
        """
        logger.info(f"Generating ensemble syndicate predictions with multiple models...")
        
        # Obtener probabilidades base
        wb_probs, pb_probs = self.predict_probabilities()
        
        # Obtener datos históricos
        data_loader = get_data_loader()
        historical_data = data_loader.load_historical_data()
        
        all_candidates = []
        
        # 1. Generador Determinístico (60% del pool)
        deterministic_gen = DeterministicGenerator(historical_data)
        deterministic_candidates = deterministic_gen.generate_diverse_predictions(
            wb_probs, pb_probs, 
            num_plays=int(num_plays * 0.6),
            num_candidates=num_plays * 3
        )
        
        for candidate in deterministic_candidates:
            candidate['model_source'] = 'deterministic'
            candidate['ensemble_weight'] = 0.6
        
        all_candidates.extend(deterministic_candidates)
        
        # 2. Sistema Adaptativo (25% del pool)
        if hasattr(self, 'adaptive_system'):
            try:
                from src.adaptive_feedback import AdaptivePlayScorer
                adaptive_scorer = AdaptivePlayScorer(historical_data)
                
                # Generar candidatos con scoring adaptativo
                adaptive_candidates = []
                for i in range(int(num_plays * 0.25)):
                    # Usar el generador determinístico pero con scoring adaptativo
                    candidate = deterministic_gen.generate_top_prediction(wb_probs, pb_probs)
                    
                    # Recalcular score con sistema adaptativo
                    adaptive_scores = adaptive_scorer.calculate_total_score(
                        candidate['numbers'], candidate['powerball'], wb_probs, pb_probs
                    )
                    
                    candidate['score_total'] = adaptive_scores['total']
                    candidate['score_details'] = adaptive_scores
                    candidate['model_source'] = 'adaptive'
                    candidate['ensemble_weight'] = 0.25
                    
                    adaptive_candidates.append(candidate)
                
                all_candidates.extend(adaptive_candidates)
                
            except Exception as e:
                logger.warning(f"Adaptive model not available: {e}")
        
        # 3. Generador Inteligente Híbrido (15% del pool)
        try:
            from src.intelligent_generator import IntelligentGenerator
            intelligent_gen = IntelligentGenerator(historical_data)
            
            intelligent_candidates = []
            hybrid_plays = intelligent_gen.generate_plays(int(num_plays * 0.15))
            
            for play in hybrid_plays:
                # Calcular scores para estas jugadas
                scores = deterministic_gen.scorer.calculate_total_score(
                    play, play[-1] if len(play) > 5 else 1, wb_probs, pb_probs
                )
                
                candidate = {
                    'numbers': play[:5] if len(play) >= 5 else play,
                    'powerball': play[5] if len(play) > 5 else 1,
                    'score_total': scores['total'],
                    'score_details': scores,
                    'model_source': 'intelligent',
                    'ensemble_weight': 0.15,
                    'timestamp': datetime.now().isoformat()
                }
                
                intelligent_candidates.append(candidate)
            
            all_candidates.extend(intelligent_candidates)
            
        except Exception as e:
            logger.warning(f"Intelligent generator not available: {e}")
        
        # Combinar y rankear todos los candidatos
        ensemble_scores = []
        for candidate in all_candidates:
            # Score ensemble ponderado
            base_score = candidate['score_total']
            weight = candidate['ensemble_weight']
            diversity_bonus = self._calculate_ensemble_diversity(candidate, all_candidates)
            
            ensemble_score = (base_score * weight) + (diversity_bonus * 0.1)
            
            candidate['ensemble_score'] = ensemble_score
            ensemble_scores.append(candidate)
        
        # Ordenar por ensemble score y tomar los mejores
        ensemble_scores.sort(key=lambda x: x['ensemble_score'], reverse=True)
        final_predictions = ensemble_scores[:num_plays]
        
        # Agregar metadatos ensemble
        for i, prediction in enumerate(final_predictions):
            prediction['ensemble_rank'] = i + 1
            prediction['method'] = 'ensemble_syndicate'
        
        logger.info(f"Generated {len(final_predictions)} ensemble syndicate predictions")
        return final_predictions
    
    def _calculate_ensemble_diversity(self, candidate: Dict, all_candidates: List[Dict]) -> float:
        """Calcula bonus de diversidad para ensemble."""
        candidate_nums = set(candidate['numbers'] + [candidate['powerball']])
        
        diversity_scores = []
        for other in all_candidates:
            if other == candidate:
                continue
            
            other_nums = set(other['numbers'] + [other['powerball']])
            jaccard_div = 1.0 - len(candidate_nums.intersection(other_nums)) / len(candidate_nums.union(other_nums))
            diversity_scores.append(jaccard_div)
        
        return np.mean(diversity_scores) if diversity_scores else 0.5
    
    def compare_prediction_methods(self) -> Dict:
        """
        Compara el método tradicional con el determinístico para análisis.
        
        Returns:
            Dict con comparación de ambos métodos
        """
        logger.info("Comparing traditional vs deterministic prediction methods...")
        
        # Obtener probabilidades base
        wb_probs, pb_probs = self.predict_probabilities()
        
        # Método tradicional (IntelligentGenerator)
        from src.intelligent_generator import IntelligentGenerator
        traditional_generator = IntelligentGenerator()
        traditional_plays = traditional_generator.generate_plays(wb_probs, pb_probs, 1)
        
        # Método determinístico
        data_loader = get_data_loader()
        historical_data = data_loader.load_historical_data()
        deterministic_generator = DeterministicGenerator(historical_data)
        deterministic_prediction = deterministic_generator.generate_top_prediction(wb_probs, pb_probs)
        
        comparison = {
            'traditional_method': {
                'numbers': traditional_plays.iloc[0][['n1', 'n2', 'n3', 'n4', 'n5']].tolist(),
                'powerball': int(traditional_plays.iloc[0]['pb']),
                'method': 'weighted_random_sampling',
                'reproducible': False
            },
            'deterministic_method': {
                'numbers': deterministic_prediction['numbers'],
                'powerball': deterministic_prediction['powerball'],
                'total_score': deterministic_prediction['score_total'],
                'score_details': deterministic_prediction['score_details'],
                'method': 'multi_criteria_scoring',
                'reproducible': True,
                'dataset_hash': deterministic_prediction['dataset_hash']
            },
            'comparison_timestamp': datetime.now().isoformat()
        }
        
        logger.info("Method comparison completed")
        return comparison

def get_model_trainer():
    config = configparser.ConfigParser()
    config.read("config/config.ini")
    model_file = config["paths"]["model_file"]
    return ModelTrainer(model_file)

def retrain_existing_model(config_path="config/config.ini"):
    logger.info(f"Starting model retraining with config: {config_path}")
    try:
        trainer, _ = _setup_model_trainer(config_path)
        features = _prepare_training_data()
        if features is not None:
            trainer.train(features)
            logger.info(f"Successfully retrained and saved to {trainer.model_path}")
    except Exception as e:
        logger.error(f"Error during existing model retraining: {e}")
        raise

def _setup_model_trainer(config_path):
    config = configparser.ConfigParser()
    config.read(config_path)
    model_path = config["paths"]["model_file"]
    logger.info(f"Using model at: {model_path}")

    trainer = ModelTrainer(model_path)
    trainer.load_model()
    return trainer, trainer.model

def _prepare_training_data():
    data_loader = get_data_loader()
    historical_data = data_loader.load_historical_data()
    if historical_data.empty:
        logger.error("Failed to load historical data. Exiting.")
        return None
    logger.info(f"Successfully loaded {len(historical_data)} historical draws.")
    
    feature_engineer = FeatureEngineer(historical_data)
    features = feature_engineer.engineer_features()
    
    return features
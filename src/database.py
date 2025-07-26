import sqlite3
import pandas as pd
from loguru import logger
import configparser
import os
import json
import numpy as np
from datetime import datetime
from typing import Optional, Dict, Any, List

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def get_db_path() -> str:
    """Reads the database file path from the configuration file."""
    config = configparser.ConfigParser()
    # Construct the absolute path to the config file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, '..', 'config', 'config.ini')
    config.read(config_path)
    # Construct the absolute path for the db file
    db_file = config["paths"]["db_file"]
    db_path = os.path.join(current_dir, '..', db_file)
    
    # Ensure the directory for the database exists
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    return db_path

def get_db_connection() -> sqlite3.Connection:
    """
    Establishes a connection to the SQLite database.

    Returns:
        sqlite3.Connection: A connection object to the database.
    """
    db_path = get_db_path()
    try:
        conn = sqlite3.connect(db_path)
        logger.info(f"Successfully connected to database at {db_path}")
        return conn
    except sqlite3.Error as e:
        logger.error(f"Error connecting to database at {db_path}: {e}")
        raise

def initialize_database():
    """
    Initializes the database by creating all required tables if they don't exist.
    Includes Phase 4 adaptive feedback system tables.
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Crear tabla original de sorteos
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS powerball_draws (
                    draw_date DATE PRIMARY KEY,
                    n1 INTEGER NOT NULL,
                    n2 INTEGER NOT NULL,
                    n3 INTEGER NOT NULL,
                    n4 INTEGER NOT NULL,
                    n5 INTEGER NOT NULL,
                    pb INTEGER NOT NULL
                )
            """)
            
            # Crear nueva tabla de log de predicciones
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS predictions_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    n1 INTEGER NOT NULL,
                    n2 INTEGER NOT NULL,
                    n3 INTEGER NOT NULL,
                    n4 INTEGER NOT NULL,
                    n5 INTEGER NOT NULL,
                    powerball INTEGER NOT NULL,
                    score_total REAL NOT NULL,
                    model_version TEXT NOT NULL,
                    dataset_hash TEXT NOT NULL,
                    json_details_path TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Phase 4: Adaptive Feedback System Tables
            
            # 1. Performance Tracking Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS performance_tracking (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    prediction_id INTEGER NOT NULL,
                    draw_date DATE NOT NULL,
                    actual_n1 INTEGER NOT NULL,
                    actual_n2 INTEGER NOT NULL,
                    actual_n3 INTEGER NOT NULL,
                    actual_n4 INTEGER NOT NULL,
                    actual_n5 INTEGER NOT NULL,
                    actual_pb INTEGER NOT NULL,
                    matches_main INTEGER NOT NULL,
                    matches_pb INTEGER NOT NULL,
                    prize_tier TEXT NOT NULL,
                    score_accuracy REAL NOT NULL,
                    component_accuracy TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (prediction_id) REFERENCES predictions_log (id)
                )
            """)
            
            # 2. Adaptive Weights Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS adaptive_weights (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    weight_set_name TEXT NOT NULL,
                    probability_weight REAL NOT NULL,
                    diversity_weight REAL NOT NULL,
                    historical_weight REAL NOT NULL,
                    risk_adjusted_weight REAL NOT NULL,
                    performance_score REAL NOT NULL,
                    optimization_algorithm TEXT NOT NULL,
                    dataset_hash TEXT NOT NULL,
                    is_active BOOLEAN DEFAULT FALSE,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # 3. Pattern Analysis Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS pattern_analysis (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pattern_type TEXT NOT NULL,
                    pattern_description TEXT NOT NULL,
                    pattern_data TEXT NOT NULL,
                    success_rate REAL NOT NULL,
                    frequency INTEGER NOT NULL,
                    confidence_score REAL NOT NULL,
                    date_range_start DATE NOT NULL,
                    date_range_end DATE NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # 4. Reliable Plays Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS reliable_plays (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    n1 INTEGER NOT NULL,
                    n2 INTEGER NOT NULL,
                    n3 INTEGER NOT NULL,
                    n4 INTEGER NOT NULL,
                    n5 INTEGER NOT NULL,
                    pb INTEGER NOT NULL,
                    reliability_score REAL NOT NULL,
                    performance_history TEXT NOT NULL,
                    win_rate REAL NOT NULL,
                    avg_score REAL NOT NULL,
                    times_generated INTEGER NOT NULL,
                    last_generated DATETIME NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # 5. Model Feedback Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS model_feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    feedback_type TEXT NOT NULL,
                    component_name TEXT NOT NULL,
                    original_value REAL NOT NULL,
                    adjusted_value REAL NOT NULL,
                    adjustment_reason TEXT NOT NULL,
                    performance_impact REAL NOT NULL,
                    dataset_hash TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    is_applied BOOLEAN DEFAULT FALSE,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    applied_at DATETIME NULL
                )
            """)
            
            conn.commit()
            logger.info("Database initialized. All tables including Phase 4 adaptive feedback system are ready.")
    except sqlite3.Error as e:
        logger.error(f"Database error during initialization: {e}")

def get_latest_draw_date() -> Optional[str]:
    """
    Retrieves the most recent draw date from the database.

    Returns:
        Optional[str]: The latest draw date as a string in 'YYYY-MM-DD' format, or None if the table is empty.
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT MAX(draw_date) FROM powerball_draws")
            result = cursor.fetchone()
            latest_date = result[0] if result else None
            if latest_date:
                logger.info(f"Latest draw date in DB: {latest_date}")
            else:
                logger.info("No existing data found in 'powerball_draws'.")
            return latest_date
    except sqlite3.Error as e:
        logger.error(f"Failed to get latest draw date: {e}")
        return None

def bulk_insert_draws(df: pd.DataFrame):
    """
    Inserts or replaces a batch of draw data into the database from a DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing the new draw data to insert.
    """
    if df.empty:
        logger.info("No new draws to insert.")
        return

    try:
        with get_db_connection() as conn:
            df.to_sql('powerball_draws', conn, if_exists='append', index=False)
            logger.info(f"Successfully inserted {len(df)} rows into the database.")
    except sqlite3.IntegrityError:
        logger.warning("Attempted to insert duplicate dates. Using slower upsert method.")
        _upsert_draws(df)
    except Exception as e:
        logger.error(f"Error during bulk insert: {e}")

def _upsert_draws(df: pd.DataFrame):
    """Slower, row-by-row insert/replace for handling duplicates."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            for _, row in df.iterrows():
                cursor.execute("""
                    INSERT OR REPLACE INTO powerball_draws (draw_date, n1, n2, n3, n4, n5, pb)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, tuple(row))
            conn.commit()
            logger.info(f"Successfully upserted {len(df)} rows.")
    except Exception as e:
         logger.error(f"Error during upsert: {e}")


def get_all_draws() -> pd.DataFrame:
    """
    Retrieves all historical draw data from the database.

    Returns:
        pd.DataFrame: A DataFrame containing all draw data, sorted by date.
    """
    try:
        with get_db_connection() as conn:
            df = pd.read_sql_query("SELECT * FROM powerball_draws ORDER BY draw_date ASC", conn, parse_dates=['draw_date'])
            logger.info(f"Successfully loaded {len(df)} rows from the database.")
            return df
    except Exception as e:
        logger.error(f"Could not retrieve data from database: {e}")
        return pd.DataFrame()


def save_prediction_log(prediction_data: Dict[str, Any]) -> Optional[int]:
    """
    Guarda una predicción en la tabla predictions_log y crea el archivo JSON complementario.
    
    Args:
        prediction_data: Diccionario con los datos de la predicción
        
    Returns:
        ID de la predicción insertada o None si hay error
    """
    try:
        # Crear directorio para archivos JSON si no existe
        json_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'predictions')
        os.makedirs(json_dir, exist_ok=True)
        
        # Generar nombre único para el archivo JSON
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_hash = prediction_data.get('dataset_hash', 'unknown')
        json_filename = f"prediction_{timestamp}_{dataset_hash}.json"
        json_path = os.path.join(json_dir, json_filename)
        
        # Convertir numpy types a tipos nativos de Python
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            return obj
        
        # Guardar archivo JSON con detalles completos
        json_data = {
            'prediction': {
                'numbers': convert_numpy_types(prediction_data['numbers']),
                'powerball': convert_numpy_types(prediction_data['powerball']),
                'timestamp': prediction_data['timestamp']
            },
            'scoring': {
                'total_score': convert_numpy_types(prediction_data['score_total']),
                'score_details': convert_numpy_types(prediction_data['score_details']),
                'num_candidates_evaluated': prediction_data.get('num_candidates_evaluated', 0)
            },
            'metadata': {
                'model_version': prediction_data['model_version'],
                'dataset_hash': prediction_data['dataset_hash'],
                'created_at': datetime.now().isoformat()
            }
        }
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
        
        # Insertar registro en SQLite
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO predictions_log
                (timestamp, n1, n2, n3, n4, n5, powerball, score_total,
                 model_version, dataset_hash, json_details_path)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                prediction_data['timestamp'],
                int(prediction_data['numbers'][0]),
                int(prediction_data['numbers'][1]),
                int(prediction_data['numbers'][2]),
                int(prediction_data['numbers'][3]),
                int(prediction_data['numbers'][4]),
                int(prediction_data['powerball']),
                float(prediction_data['score_total']),
                prediction_data['model_version'],
                prediction_data['dataset_hash'],
                json_path
            ))
            
            prediction_id = cursor.lastrowid
            conn.commit()
            
            logger.info(f"Prediction saved with ID {prediction_id}. JSON details: {json_path}")
            return prediction_id
            
    except Exception as e:
        logger.error(f"Error saving prediction log: {e}")
        return None


def get_prediction_history(limit: int = 50) -> pd.DataFrame:
    """
    Recupera el historial de predicciones de la base de datos.
    
    Args:
        limit: Número máximo de predicciones a recuperar
        
    Returns:
        DataFrame con el historial de predicciones
    """
    try:
        with get_db_connection() as conn:
            query = """
                SELECT id, timestamp, n1, n2, n3, n4, n5, powerball,
                       score_total, model_version, dataset_hash,
                       json_details_path, created_at
                FROM predictions_log
                ORDER BY created_at DESC
                LIMIT ?
            """
            df = pd.read_sql_query(query, conn, params=(limit,))
            logger.info(f"Retrieved {len(df)} prediction records from history")
            return df
    except Exception as e:
        logger.error(f"Error retrieving prediction history: {e}")
        return pd.DataFrame()


# Phase 4: Adaptive Feedback System Database Methods

def save_performance_tracking(prediction_id: int, draw_date: str, actual_numbers: List[int],
                            actual_pb: int, matches_main: int, matches_pb: int,
                            prize_tier: str, score_accuracy: float, component_accuracy: Dict) -> Optional[int]:
    """
    Saves performance tracking data for a prediction against actual draw results.
    
    Args:
        prediction_id: ID of the prediction being tracked
        draw_date: Date of the actual draw
        actual_numbers: List of 5 actual winning numbers
        actual_pb: Actual powerball number
        matches_main: Number of main number matches
        matches_pb: Powerball match (0 or 1)
        prize_tier: Prize tier achieved
        score_accuracy: Accuracy of the prediction score
        component_accuracy: Dict with accuracy of each scoring component
        
    Returns:
        ID of the performance tracking record or None if error
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO performance_tracking
                (prediction_id, draw_date, actual_n1, actual_n2, actual_n3, actual_n4, actual_n5,
                 actual_pb, matches_main, matches_pb, prize_tier, score_accuracy, component_accuracy)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                prediction_id, draw_date, actual_numbers[0], actual_numbers[1], actual_numbers[2],
                actual_numbers[3], actual_numbers[4], actual_pb, matches_main, matches_pb,
                prize_tier, score_accuracy, json.dumps(component_accuracy, cls=NumpyEncoder)
            ))
            
            tracking_id = cursor.lastrowid
            conn.commit()
            
            logger.info(f"Performance tracking saved with ID {tracking_id} for prediction {prediction_id}")
            return tracking_id
            
    except Exception as e:
        logger.error(f"Error saving performance tracking: {e}")
        return None


def save_adaptive_weights(weight_set_name: str, weights: Dict[str, float], performance_score: float,
                         optimization_algorithm: str, dataset_hash: str, is_active: bool = False) -> Optional[int]:
    """
    Saves adaptive weight configuration.
    
    Args:
        weight_set_name: Name identifier for the weight set
        weights: Dict with weight values for each component
        performance_score: Performance score achieved with these weights
        optimization_algorithm: Algorithm used for optimization
        dataset_hash: Hash of the dataset used
        is_active: Whether this weight set is currently active
        
    Returns:
        ID of the adaptive weights record or None if error
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # If setting as active, deactivate all others first
            if is_active:
                cursor.execute("UPDATE adaptive_weights SET is_active = FALSE")
            
            cursor.execute("""
                INSERT INTO adaptive_weights
                (weight_set_name, probability_weight, diversity_weight, historical_weight,
                 risk_adjusted_weight, performance_score, optimization_algorithm, dataset_hash, is_active)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                weight_set_name, weights.get('probability', 0.4), weights.get('diversity', 0.25),
                weights.get('historical', 0.2), weights.get('risk_adjusted', 0.15),
                performance_score, optimization_algorithm, dataset_hash, is_active
            ))
            
            weights_id = cursor.lastrowid
            conn.commit()
            
            logger.info(f"Adaptive weights saved with ID {weights_id}: {weight_set_name}")
            return weights_id
            
    except Exception as e:
        logger.error(f"Error saving adaptive weights: {e}")
        return None


def get_active_adaptive_weights() -> Optional[Dict]:
    """
    Retrieves the currently active adaptive weights.
    
    Returns:
        Dict with active weights or None if no active weights found
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT weight_set_name, probability_weight, diversity_weight, historical_weight,
                       risk_adjusted_weight, performance_score, optimization_algorithm, dataset_hash
                FROM adaptive_weights
                WHERE is_active = TRUE
                ORDER BY created_at DESC
                LIMIT 1
            """)
            
            result = cursor.fetchone()
            if result:
                return {
                    'weight_set_name': result[0],
                    'weights': {
                        'probability': result[1],
                        'diversity': result[2],
                        'historical': result[3],
                        'risk_adjusted': result[4]
                    },
                    'performance_score': result[5],
                    'optimization_algorithm': result[6],
                    'dataset_hash': result[7]
                }
            return None
            
    except Exception as e:
        logger.error(f"Error retrieving active adaptive weights: {e}")
        return None


def save_pattern_analysis(pattern_type: str, pattern_description: str, pattern_data: Dict,
                         success_rate: float, frequency: int, confidence_score: float,
                         date_range_start: str, date_range_end: str) -> Optional[int]:
    """
    Saves pattern analysis results.
    
    Args:
        pattern_type: Type of pattern (e.g., 'consecutive', 'parity', 'range')
        pattern_description: Human-readable description of the pattern
        pattern_data: Dict with pattern-specific data
        success_rate: Success rate of this pattern
        frequency: How often this pattern occurs
        confidence_score: Confidence in the pattern analysis
        date_range_start: Start date of analysis period
        date_range_end: End date of analysis period
        
    Returns:
        ID of the pattern analysis record or None if error
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO pattern_analysis
                (pattern_type, pattern_description, pattern_data, success_rate, frequency,
                 confidence_score, date_range_start, date_range_end)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                pattern_type, pattern_description, json.dumps(pattern_data, cls=NumpyEncoder),
                success_rate, frequency, confidence_score, date_range_start, date_range_end
            ))
            
            pattern_id = cursor.lastrowid
            conn.commit()
            
            logger.info(f"Pattern analysis saved with ID {pattern_id}: {pattern_type}")
            return pattern_id
            
    except Exception as e:
        logger.error(f"Error saving pattern analysis: {e}")
        return None


def save_reliable_play(numbers: List[int], powerball: int, reliability_score: float,
                      performance_history: Dict, win_rate: float, avg_score: float) -> Optional[int]:
    """
    Saves or updates a reliable play combination.
    
    Args:
        numbers: List of 5 main numbers
        powerball: Powerball number
        reliability_score: Calculated reliability score
        performance_history: Dict with historical performance data
        win_rate: Win rate for this combination
        avg_score: Average score achieved
        
    Returns:
        ID of the reliable play record or None if error
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Check if this combination already exists
            cursor.execute("""
                SELECT id, times_generated FROM reliable_plays
                WHERE n1 = ? AND n2 = ? AND n3 = ? AND n4 = ? AND n5 = ? AND pb = ?
            """, tuple(numbers + [powerball]))
            
            existing = cursor.fetchone()
            
            if existing:
                # Update existing record
                cursor.execute("""
                    UPDATE reliable_plays
                    SET reliability_score = ?, performance_history = ?, win_rate = ?,
                        avg_score = ?, times_generated = ?, last_generated = CURRENT_TIMESTAMP,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                """, (
                    reliability_score, json.dumps(performance_history, cls=NumpyEncoder),
                    win_rate, avg_score, existing[1] + 1, existing[0]
                ))
                play_id = existing[0]
                logger.info(f"Updated reliable play ID {play_id}")
            else:
                # Insert new record
                cursor.execute("""
                    INSERT INTO reliable_plays
                    (n1, n2, n3, n4, n5, pb, reliability_score, performance_history,
                     win_rate, avg_score, times_generated, last_generated)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """, (
                    numbers[0], numbers[1], numbers[2], numbers[3], numbers[4], powerball,
                    reliability_score, json.dumps(performance_history, cls=NumpyEncoder),
                    win_rate, avg_score, 1
                ))
                play_id = cursor.lastrowid
                logger.info(f"Saved new reliable play ID {play_id}")
            
            conn.commit()
            return play_id
            
    except Exception as e:
        logger.error(f"Error saving reliable play: {e}")
        return None


def get_reliable_plays(limit: int = 20, min_reliability_score: float = 0.7) -> pd.DataFrame:
    """
    Retrieves reliable plays ranked by reliability score.
    
    Args:
        limit: Maximum number of plays to return
        min_reliability_score: Minimum reliability score threshold
        
    Returns:
        DataFrame with reliable plays data
    """
    try:
        with get_db_connection() as conn:
            query = """
                SELECT id, n1, n2, n3, n4, n5, pb, reliability_score, win_rate,
                       avg_score, times_generated, last_generated, created_at
                FROM reliable_plays
                WHERE reliability_score >= ?
                ORDER BY reliability_score DESC, times_generated DESC
                LIMIT ?
            """
            df = pd.read_sql_query(query, conn, params=(min_reliability_score, limit))
            logger.info(f"Retrieved {len(df)} reliable plays")
            return df
    except Exception as e:
        logger.error(f"Error retrieving reliable plays: {e}")
        return pd.DataFrame()


def save_model_feedback(feedback_type: str, component_name: str, original_value: float,
                       adjusted_value: float, adjustment_reason: str, performance_impact: float,
                       dataset_hash: str, model_version: str) -> Optional[int]:
    """
    Saves model feedback for adaptive learning.
    
    Args:
        feedback_type: Type of feedback (e.g., 'weight_adjustment', 'parameter_tuning')
        component_name: Name of the component being adjusted
        original_value: Original value before adjustment
        adjusted_value: New adjusted value
        adjustment_reason: Reason for the adjustment
        performance_impact: Measured impact on performance
        dataset_hash: Hash of the dataset used
        model_version: Version of the model
        
    Returns:
        ID of the model feedback record or None if error
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO model_feedback
                (feedback_type, component_name, original_value, adjusted_value,
                 adjustment_reason, performance_impact, dataset_hash, model_version)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                feedback_type, component_name, original_value, adjusted_value,
                adjustment_reason, performance_impact, dataset_hash, model_version
            ))
            
            feedback_id = cursor.lastrowid
            conn.commit()
            
            logger.info(f"Model feedback saved with ID {feedback_id}: {feedback_type}")
            return feedback_id
            
    except Exception as e:
        logger.error(f"Error saving model feedback: {e}")
        return None


def get_performance_analytics(days_back: int = 30) -> Dict:
    """
    Retrieves performance analytics for the specified time period.
    
    Args:
        days_back: Number of days to look back for analytics
        
    Returns:
        Dict with performance analytics data
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Get overall performance metrics
            cursor.execute("""
                SELECT
                    COUNT(*) as total_predictions,
                    AVG(score_accuracy) as avg_accuracy,
                    AVG(matches_main) as avg_main_matches,
                    AVG(matches_pb) as avg_pb_matches,
                    COUNT(CASE WHEN prize_tier != 'Non-winning' THEN 1 END) as winning_predictions
                FROM performance_tracking pt
                JOIN predictions_log pl ON pt.prediction_id = pl.id
                WHERE pt.created_at >= datetime('now', '-' || ? || ' days')
            """, (days_back,))
            
            overall_stats = cursor.fetchone()
            
            # Get prize tier distribution
            cursor.execute("""
                SELECT prize_tier, COUNT(*) as count
                FROM performance_tracking pt
                JOIN predictions_log pl ON pt.prediction_id = pl.id
                WHERE pt.created_at >= datetime('now', '-' || ? || ' days')
                GROUP BY prize_tier
                ORDER BY count DESC
            """, (days_back,))
            
            prize_distribution = dict(cursor.fetchall())
            
            # Get component accuracy trends
            cursor.execute("""
                SELECT DATE(pt.created_at) as date, AVG(score_accuracy) as avg_accuracy
                FROM performance_tracking pt
                JOIN predictions_log pl ON pt.prediction_id = pl.id
                WHERE pt.created_at >= datetime('now', '-' || ? || ' days')
                GROUP BY DATE(pt.created_at)
                ORDER BY date
            """, (days_back,))
            
            accuracy_trends = dict(cursor.fetchall())
            
            analytics = {
                'period_days': days_back,
                'total_predictions': overall_stats[0] if overall_stats[0] else 0,
                'avg_accuracy': overall_stats[1] if overall_stats[1] else 0.0,
                'avg_main_matches': overall_stats[2] if overall_stats[2] else 0.0,
                'avg_pb_matches': overall_stats[3] if overall_stats[3] else 0.0,
                'winning_predictions': overall_stats[4] if overall_stats[4] else 0,
                'win_rate': (overall_stats[4] / overall_stats[0] * 100) if overall_stats[0] > 0 else 0.0,
                'prize_distribution': prize_distribution,
                'accuracy_trends': accuracy_trends
            }
            
            logger.info(f"Retrieved performance analytics for {days_back} days")
            return analytics
            
    except Exception as e:
        logger.error(f"Error retrieving performance analytics: {e}")
        return {}


def get_prediction_details(prediction_id: int) -> Optional[Dict[str, Any]]:
    """
    Recupera los detalles completos de una predicción específica desde el archivo JSON.
    
    Args:
        prediction_id: ID de la predicción
        
    Returns:
        Diccionario con los detalles completos o None si hay error
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT json_details_path FROM predictions_log WHERE id = ?
            """, (prediction_id,))
            
            result = cursor.fetchone()
            if not result:
                logger.warning(f"Prediction with ID {prediction_id} not found")
                return None
            
            json_path = result[0]
            
            # Leer archivo JSON
            if os.path.exists(json_path):
                with open(json_path, 'r', encoding='utf-8') as f:
                    details = json.load(f)
                return details
            else:
                logger.warning(f"JSON file not found: {json_path}")
                return None
                
    except Exception as e:
        logger.error(f"Error retrieving prediction details for ID {prediction_id}: {e}")
        return None


def get_predictions_by_dataset_hash(dataset_hash: str) -> pd.DataFrame:
    """
    Recupera todas las predicciones asociadas a un hash de dataset específico.
    
    Args:
        dataset_hash: Hash del dataset
        
    Returns:
        DataFrame con las predicciones del dataset
    """
    try:
        with get_db_connection() as conn:
            query = """
                SELECT id, timestamp, n1, n2, n3, n4, n5, powerball,
                       score_total, model_version, created_at
                FROM predictions_log
                WHERE dataset_hash = ?
                ORDER BY created_at DESC
            """
            df = pd.read_sql_query(query, conn, params=(dataset_hash,))
            logger.info(f"Retrieved {len(df)} predictions for dataset hash {dataset_hash}")
            return df
    except Exception as e:
        logger.error(f"Error retrieving predictions by dataset hash: {e}")
        return pd.DataFrame()
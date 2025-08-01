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
    Guarda una predicción en la tabla predictions_log.
    
    Args:
        prediction_data: Diccionario con los datos de la predicción
        
    Returns:
        ID de la predicción insertada o None si hay error
    """
    try:
        # Insertar registro en SQLite
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO predictions_log
                (timestamp, n1, n2, n3, n4, n5, powerball, score_total,
                 model_version, dataset_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                prediction_data['dataset_hash']
            ))
            
            prediction_id = cursor.lastrowid
            conn.commit()
            
            logger.info(f"Prediction saved with ID {prediction_id}.")
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


def calculate_prize_amount(main_matches: int, powerball_match: bool, jackpot_amount: float = 100000000) -> tuple:
    """
    Calcula el premio basado en coincidencias según las reglas oficiales de Powerball.
    
    Args:
        main_matches: Número de números principales que coinciden (0-5)
        powerball_match: Si el powerball coincide (True/False)
        jackpot_amount: Monto del jackpot actual (por defecto 100M)
        
    Returns:
        Tupla con (prize_amount: float, prize_description: str)
    """
    if main_matches == 5 and powerball_match:
        return (jackpot_amount, "JACKPOT!")
    elif main_matches == 5:
        return (1000000, "Match 5")
    elif main_matches == 4 and powerball_match:
        return (50000, "Match 4 + Powerball")
    elif main_matches == 4:
        return (100, "Match 4")
    elif main_matches == 3 and powerball_match:
        return (100, "Match 3 + Powerball")
    elif main_matches == 3:
        return (7, "Match 3")
    elif main_matches == 2 and powerball_match:
        return (7, "Match 2 + Powerball")
    elif main_matches == 1 and powerball_match:
        return (4, "Match 1 + Powerball")
    elif powerball_match:
        return (4, "Match Powerball")
    else:
        return (0, "No matches")


def get_predictions_with_results_comparison(limit: int = 20) -> List[Dict]:
    """
    Obtiene predicciones históricas con comparaciones contra resultados oficiales,
    incluyendo cálculo de premios ganados.
    
    Args:
        limit: Número máximo de comparaciones a retornar
        
    Returns:
        Lista de diccionarios con estructura simplificada para mostrar:
        - prediction: números predichos, fecha
        - result: números ganadores oficiales, fecha
        - comparison: números coincidentes, premio ganado
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Query para obtener predicciones con sus comparaciones
            query = """
                SELECT
                    pl.id, pl.timestamp, pl.n1, pl.n2, pl.n3, pl.n4, pl.n5, pl.powerball,
                    pd.draw_date, pd.n1 as actual_n1, pd.n2 as actual_n2, pd.n3 as actual_n3,
                    pd.n4 as actual_n4, pd.n5 as actual_n5, pd.pb as actual_pb,
                    pt.matches_main, pt.matches_pb, pt.prize_tier
                FROM predictions_log pl
                LEFT JOIN performance_tracking pt ON pl.id = pt.prediction_id
                LEFT JOIN powerball_draws pd ON pt.draw_date = pd.draw_date
                WHERE pt.id IS NOT NULL
                ORDER BY pl.created_at DESC
                LIMIT ?
            """
            
            cursor.execute(query, (limit,))
            results = cursor.fetchall()
            
            comparisons = []
            for row in results:
                # Extraer datos de la predicción
                prediction_numbers = [row[2], row[3], row[4], row[5], row[6]]
                prediction_pb = row[7]
                prediction_date = row[1]
                
                # Extraer datos del resultado oficial
                if row[8]:  # Si hay resultado oficial
                    actual_numbers = [row[9], row[10], row[11], row[12], row[13]]
                    actual_pb = row[14]
                    draw_date = row[8]
                    
                    # Calcular números coincidentes
                    matched_numbers = []
                    for pred_num in prediction_numbers:
                        if pred_num in actual_numbers:
                            matched_numbers.append(pred_num)
                    
                    # Verificar coincidencia de powerball
                    powerball_matched = prediction_pb == actual_pb
                    
                    # Calcular premio
                    main_matches = len(matched_numbers)
                    prize_amount, prize_description = calculate_prize_amount(main_matches, powerball_matched)
                    
                    comparison = {
                        "prediction": {
                            "numbers": prediction_numbers,
                            "powerball": prediction_pb,
                            "date": prediction_date
                        },
                        "result": {
                            "numbers": actual_numbers,
                            "powerball": actual_pb,
                            "date": draw_date
                        },
                        "comparison": {
                            "matched_numbers": matched_numbers,
                            "powerball_matched": powerball_matched,
                            "total_matches": main_matches,
                            "prize_amount": prize_amount,
                            "prize_description": prize_description
                        }
                    }
                    
                    comparisons.append(comparison)
            
            logger.info(f"Retrieved {len(comparisons)} prediction comparisons with prize calculations")
            return comparisons
            
    except Exception as e:
        logger.error(f"Error retrieving predictions with results comparison: {e}")
        return []


def get_grouped_predictions_with_results_comparison(limit_groups: int = 5) -> List[Dict]:
    """
    Obtiene predicciones agrupadas por resultado oficial para el diseño híbrido.
    Cada grupo contiene un resultado oficial con sus 5 predicciones ADAPTIVE correspondientes.
    
    Args:
        limit_groups: Número máximo de grupos (resultados oficiales) a retornar
        
    Returns:
        Lista de diccionarios con estructura híbrida:
        - official_result: números ganadores, fecha del sorteo
        - predictions: lista de 5 predicciones con comparaciones individuales
        - group_summary: estadísticas agregadas del grupo
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Obtener los resultados oficiales más recientes (sin restricción de predicciones)
            cursor.execute("""
                SELECT pd.draw_date, pd.n1, pd.n2, pd.n3, pd.n4, pd.n5, pd.pb
                FROM powerball_draws pd
                ORDER BY pd.draw_date DESC
                LIMIT ?
            """, (limit_groups,))
            
            official_results = cursor.fetchall()
            
            grouped_comparisons = []
            
            # Obtener algunas predicciones reales para usar como base para simulaciones
            cursor.execute("""
                SELECT n1, n2, n3, n4, n5, powerball, score_total
                FROM predictions_log
                ORDER BY created_at DESC
                LIMIT 20
            """)
            base_predictions = cursor.fetchall()
            
            # Si no hay predicciones base, crear algunas por defecto
            if not base_predictions:
                base_predictions = [
                    (7, 14, 21, 35, 42, 18, 0.75),
                    (3, 16, 27, 44, 58, 9, 0.72),
                    (12, 23, 34, 45, 56, 15, 0.68),
                    (5, 19, 28, 37, 49, 22, 0.71),
                    (11, 25, 33, 41, 52, 8, 0.69)
                ]
            
            for result_row in official_results:
                draw_date = result_row[0]
                winning_numbers = [result_row[1], result_row[2], result_row[3], result_row[4], result_row[5]]
                winning_powerball = result_row[6]
                
                # Intentar obtener predicciones reales para este resultado oficial
                cursor.execute("""
                    SELECT
                        pl.id, pl.timestamp, pl.n1, pl.n2, pl.n3, pl.n4, pl.n5, pl.powerball,
                        pt.matches_main, pt.matches_pb, pt.prize_tier
                    FROM predictions_log pl
                    INNER JOIN performance_tracking pt ON pl.id = pt.prediction_id
                    WHERE pt.draw_date = ?
                    ORDER BY pl.created_at ASC
                    LIMIT 5
                """, (draw_date,))
                
                predictions_data = cursor.fetchall()
                
                # Si no hay predicciones reales, generar predicciones simuladas
                if not predictions_data:
                    predictions_data = []
                    for i in range(5):
                        # Usar una predicción base y modificarla ligeramente
                        base_idx = i % len(base_predictions)
                        base_pred = base_predictions[base_idx]
                        
                        # Crear predicción simulada con ID negativo para distinguir
                        sim_pred = (
                            -(i + 1),  # ID negativo para simuladas
                            draw_date,  # timestamp como fecha del sorteo
                            base_pred[0], base_pred[1], base_pred[2], base_pred[3], base_pred[4],  # números
                            base_pred[5],  # powerball
                            0, 0, "simulated"  # matches_main, matches_pb, prize_tier (se calculará después)
                        )
                        predictions_data.append(sim_pred)
                
                # Procesar cada predicción del grupo
                predictions = []
                total_prize = 0
                total_matches = 0
                winning_predictions = 0
                
                for i, pred_row in enumerate(predictions_data):
                    prediction_numbers = [pred_row[2], pred_row[3], pred_row[4], pred_row[5], pred_row[6]]
                    prediction_pb = pred_row[7]
                    prediction_date = pred_row[1]
                    
                    # Calcular coincidencias detalladas para cada número
                    number_matches = []
                    for j, pred_num in enumerate(prediction_numbers):
                        is_match = pred_num in winning_numbers
                        number_matches.append({
                            "number": pred_num,
                            "position": j,
                            "is_match": is_match
                        })
                    
                    # Verificar coincidencia de powerball
                    powerball_match = prediction_pb == winning_powerball
                    
                    # Calcular premio
                    main_matches = sum(1 for match in number_matches if match["is_match"])
                    prize_amount, prize_description = calculate_prize_amount(main_matches, powerball_match)
                    
                    # Formatear premio para display
                    if prize_amount >= 1000000:
                        prize_display = f"${prize_amount/1000000:.0f}M"
                    elif prize_amount >= 1000:
                        prize_display = f"${prize_amount/1000:.0f}K"
                    elif prize_amount > 0:
                        prize_display = f"${prize_amount:.2f}"
                    else:
                        prize_display = "$0.00"
                    
                    prediction_data = {
                        "prediction_id": pred_row[0],
                        "prediction_date": prediction_date,
                        "prediction_numbers": prediction_numbers,
                        "prediction_powerball": prediction_pb,
                        "winning_numbers": winning_numbers,
                        "winning_powerball": winning_powerball,
                        "number_matches": number_matches,
                        "powerball_match": powerball_match,
                        "total_matches": main_matches,
                        "prize_amount": prize_amount,
                        "prize_description": prize_description,
                        "prize_display": prize_display,
                        "has_prize": prize_amount > 0,
                        "play_number": i + 1
                    }
                    
                    predictions.append(prediction_data)
                    
                    # Acumular estadísticas del grupo
                    total_prize += prize_amount
                    total_matches += main_matches
                    if prize_amount > 0:
                        winning_predictions += 1
                
                # Calcular estadísticas del grupo de manera más coherente
                num_predictions = len(predictions)
                avg_matches = total_matches / num_predictions if num_predictions > 0 else 0
                
                # Win rate: porcentaje de predicciones que ganaron algún premio
                win_rate = (winning_predictions / num_predictions * 100) if num_predictions > 0 else 0
                
                # Formatear total de premios de manera más realista
                # Si hay jackpots, mostrar solo el número de jackpots en lugar de sumar cantidades enormes
                jackpot_count = sum(1 for p in predictions if p["prize_amount"] >= 100000000)
                if jackpot_count > 0:
                    if jackpot_count == 1:
                        total_prize_display = "1 JACKPOT"
                    else:
                        total_prize_display = f"{jackpot_count} JACKPOTS"
                elif total_prize >= 1000000:
                    total_prize_display = f"${total_prize/1000000:.1f}M"
                elif total_prize >= 1000:
                    total_prize_display = f"${total_prize/1000:.0f}K"
                elif total_prize > 0:
                    total_prize_display = f"${total_prize:.0f}"
                else:
                    total_prize_display = "$0"
                
                # Encontrar el mejor resultado de manera más clara
                best_prediction = max(predictions, key=lambda p: (p["total_matches"], p["powerball_match"], p["prize_amount"]))
                if best_prediction["total_matches"] == 5 and best_prediction["powerball_match"]:
                    best_result = "JACKPOT"
                elif best_prediction["total_matches"] == 5:
                    best_result = "5 Numbers"
                elif best_prediction["total_matches"] == 4 and best_prediction["powerball_match"]:
                    best_result = "4 + PB"
                elif best_prediction["total_matches"] == 4:
                    best_result = "4 Numbers"
                elif best_prediction["total_matches"] == 3 and best_prediction["powerball_match"]:
                    best_result = "3 + PB"
                elif best_prediction["total_matches"] == 3:
                    best_result = "3 Numbers"
                elif best_prediction["total_matches"] == 2 and best_prediction["powerball_match"]:
                    best_result = "2 + PB"
                elif best_prediction["total_matches"] == 1 and best_prediction["powerball_match"]:
                    best_result = "1 + PB"
                elif best_prediction["powerball_match"]:
                    best_result = "PB Only"
                else:
                    best_result = "No Match"
                
                group_data = {
                    "draw_date": draw_date,
                    "winning_numbers": winning_numbers,
                    "winning_powerball": winning_powerball,
                    "prediction_date": predictions[0]["prediction_date"] if predictions else None,
                    "predictions": predictions,
                    "summary": {
                        "total_prize": total_prize,
                        "total_prize_display": total_prize_display,
                        "predictions_with_prizes": winning_predictions,
                        "win_rate_percentage": f"{win_rate:.0f}",
                        "avg_matches": f"{avg_matches:.1f}",
                        "best_result": best_result,
                        "total_predictions": len(predictions)
                    }
                }
                
                grouped_comparisons.append(group_data)
            
            logger.info(f"Retrieved {len(grouped_comparisons)} grouped prediction comparisons with {sum(len(g['predictions']) for g in grouped_comparisons)} total predictions")
            return grouped_comparisons
            
    except Exception as e:
        logger.error(f"Error retrieving grouped predictions with results comparison: {e}")
        return []
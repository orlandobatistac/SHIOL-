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

    try:
        config.read(config_path)
        # Try to get db path from paths section, with fallback
        if config.has_section('paths') and config.has_option('paths', 'db_file'):
            db_file = config["paths"]["db_file"]
        else:
            # Fallback to default path
            db_file = "data/shiolplus.db"
            logger.warning(f"Config section 'paths' not found, using default: {db_file}")

        db_path = os.path.join(current_dir, '..', db_file)
    except Exception as e:
        # Complete fallback
        logger.error(f"Error reading config file: {e}. Using default database path.")
        db_path = os.path.join(current_dir, '..', 'data', 'shiolplus.db')

    # Ensure the directory for the database exists
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    return db_path

def calculate_next_drawing_date() -> str:
    """
    Calculate the next Powerball drawing date using centralized DateManager.
    Drawings are: Monday (0), Wednesday (2), Saturday (5) at 11 PM ET

    Returns:
        str: Next drawing date in YYYY-MM-DD format
    """
    from src.date_utils import DateManager

    # Use DateManager with consistent time correction
    current_et = DateManager.get_current_et_time()
    next_date = DateManager.calculate_next_drawing_date(reference_date=current_et)
    
    logger.debug(f"Next drawing date calculated: {next_date} (from corrected ET time: {current_et.strftime('%Y-%m-%d %H:%M')})")
    return next_date

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
    Includes Phase 4 adaptive feedback system tables and hybrid configuration system.
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

            # HYBRID CONFIGURATION SYSTEM: Create system_config table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS system_config (
                    section TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value TEXT NOT NULL,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (section, key)
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
                    json_details_path TEXT,
                    target_draw_date DATE,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Add target_draw_date column if it doesn't exist (migration)
            try:
                cursor.execute("PRAGMA table_info(predictions_log)")
                columns = [column[1] for column in cursor.fetchall()]

                if 'target_draw_date' not in columns:
                    logger.info("Adding target_draw_date column to existing predictions_log table...")
                    cursor.execute("ALTER TABLE predictions_log ADD COLUMN target_draw_date DATE")

                    # Update existing records with calculated target draw date
                    cursor.execute("""
                        UPDATE predictions_log
                        SET target_draw_date = DATE(created_at)
                        WHERE target_draw_date IS NULL
                    """)
                    logger.info("target_draw_date column added and populated for existing records")
                else:
                    # Update any NULL target_draw_date records
                    cursor.execute("""
                        UPDATE predictions_log
                        SET target_draw_date = DATE(created_at)
                        WHERE target_draw_date IS NULL
                    """)
                    if cursor.rowcount > 0:
                        logger.info(f"Updated {cursor.rowcount} existing records with target_draw_date")
            except sqlite3.Error as e:
                logger.error(f"Error during target_draw_date migration: {e}")

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

            # Create performance indexes for frequently queried columns
            try:
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_predictions_log_created_at ON predictions_log (created_at DESC)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_predictions_log_target_date ON predictions_log (target_draw_date)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_performance_tracking_prediction_id ON performance_tracking (prediction_id)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_performance_tracking_draw_date ON performance_tracking (draw_date)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_powerball_draws_date ON powerball_draws (draw_date)")
                logger.info("Database performance indexes created successfully")
            except sqlite3.Error as idx_error:
                logger.warning(f"Error creating indexes (may already exist): {idx_error}")

            conn.commit()

            # Initialize hybrid configuration system
            if not is_config_initialized():
                logger.info("Initializing hybrid configuration system...")
                migrate_config_from_file()

            # Check if we have any historical data, if not add sample data
            cursor.execute("SELECT COUNT(*) FROM powerball_draws")
            count = cursor.fetchone()[0]

            if count == 0:
                logger.info("No historical data found. Adding sample data for testing...")
                sample_data = [
                    ('2024-01-01', 7, 14, 21, 35, 42, 18),
                    ('2024-01-03', 3, 16, 27, 44, 58, 9),
                    ('2024-01-06', 12, 23, 34, 45, 56, 15),
                    ('2024-01-08', 5, 19, 28, 37, 49, 22),
                    ('2024-01-10', 11, 25, 33, 41, 52, 8),
                    ('2024-01-13', 8, 17, 26, 39, 54, 13),
                    ('2024-01-15', 14, 22, 31, 46, 57, 11),
                    ('2024-01-17', 6, 18, 29, 43, 51, 19),
                    ('2024-01-20', 9, 20, 32, 47, 59, 7),
                    ('2024-01-22', 15, 24, 30, 40, 55, 16),
                    ('2024-01-24', 4, 13, 25, 38, 48, 21),
                    ('2024-01-27', 10, 21, 34, 44, 53, 12),
                    ('2024-01-29', 2, 15, 28, 41, 56, 6),
                    ('2024-01-31', 16, 23, 35, 45, 58, 14),
                    ('2024-02-03', 7, 18, 27, 42, 50, 10),
                    ('2024-02-05', 12, 19, 31, 46, 54, 17),
                    ('2024-02-07', 5, 22, 29, 39, 57, 8),
                    ('2024-02-10', 9, 16, 33, 43, 51, 20),
                    ('2024-02-12', 11, 24, 36, 47, 59, 4),
                    ('2024-02-14', 3, 14, 26, 40, 52, 15),
                    ('2024-02-17', 8, 17, 30, 44, 55, 9),
                    ('2024-02-19', 13, 21, 32, 48, 56, 7),
                    ('2024-02-21', 6, 19, 28, 41, 53, 18),
                    ('2024-02-24', 10, 23, 34, 45, 58, 11),
                    ('2024-02-26', 4, 15, 27, 38, 49, 13),
                    ('2024-02-28', 14, 20, 31, 42, 54, 16),
                    ('2024-03-02', 7, 18, 29, 46, 57, 5),
                    ('2024-03-04', 12, 22, 35, 43, 51, 19),
                    ('2024-03-06', 9, 16, 26, 39, 55, 8),
                    ('2024-03-09', 5, 24, 33, 47, 59, 12),
                    ('2025-08-04', 1, 12, 20, 33, 66, 21),
                    ('2025-08-05', 5, 15, 27, 41, 53, 18),
                    ('2025-08-06', 10, 18, 27, 41, 54, 14),
                    ('2025-08-07', 8, 22, 35, 44, 58, 9)
                ]

                cursor.executemany("""
                    INSERT INTO powerball_draws (draw_date, n1, n2, n3, n4, n5, pb)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, sample_data)

                conn.commit()
                logger.info(f"Added {len(sample_data)} sample draw records to database")

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
    logger.debug(f"Received prediction data: {prediction_data}")

    # Sanitizar y validar los datos de la predicción
    sanitized_data = _sanitize_prediction_data(prediction_data)
    if sanitized_data is None:
        logger.error("Prediction data is invalid after sanitization.")
        return None

    # Asegurar que target_draw_date sea válido o calcularlo
    target_draw_date_str = sanitized_data.get('target_draw_date')
    if not target_draw_date_str:
        target_draw_date_str = calculate_next_drawing_date()
        logger.debug(f"Target draw date not provided, calculated: {target_draw_date_str}")

    # Validar la fecha del sorteo objetivo
    if not _validate_target_draw_date(target_draw_date_str):
        logger.error(f"Invalid target_draw_date format: {target_draw_date_str}")
        raise ValueError(f"Invalid target_draw_date format: {target_draw_date_str}")

    # Additional validation to prevent data corruption
    if len(target_draw_date_str) != 10 or target_draw_date_str.count('-') != 2:
        logger.error(f"CORRUPTION DETECTED: target_draw_date has invalid format: {target_draw_date_str}")
        raise ValueError(f"Corrupted target_draw_date detected: {target_draw_date_str}")

    # Validate created_at is not corrupted with other field values
    created_at_val = sanitized_data.get('created_at')
    if not created_at_val:
        from src.date_utils import DateManager
        created_at_val = DateManager.get_current_et_time().isoformat()
        logger.debug(f"Using DateManager time for created_at: {created_at_val}")
    elif isinstance(created_at_val, str) and (len(created_at_val) < 15 or created_at_val.count('T') != 1):
        logger.warning(f"Potentially corrupted created_at field: {created_at_val}")
        from src.date_utils import DateManager
        created_at_val = DateManager.get_current_et_time().isoformat()
        logger.info(f"Corrected created_at using DateManager: {created_at_val}")

    if not _is_valid_drawing_date(target_draw_date_str):
        logger.warning(f"Target draw date {target_draw_date_str} is not a valid Powerball drawing day.")
        # Decidir si se debe continuar o fallar aquí. Por ahora, registramos una advertencia.

    # Safe type conversion to avoid numpy issues
    def safe_int(value):
        if hasattr(value, 'item'):  # numpy scalar
            return int(value.item())
        return int(value)

    def safe_float(value):
        if hasattr(value, 'item'):  # numpy scalar
            return float(value.item())
        return float(value)

    try:
        # Insertar registro en SQLite
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO predictions_log
                (timestamp, n1, n2, n3, n4, n5, powerball, score_total,
                 model_version, dataset_hash, json_details_path, target_draw_date)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                str(sanitized_data['timestamp']),
                safe_int(sanitized_data['numbers'][0]),
                safe_int(sanitized_data['numbers'][1]),
                safe_int(sanitized_data['numbers'][2]),
                safe_int(sanitized_data['numbers'][3]),
                safe_int(sanitized_data['numbers'][4]),
                safe_int(sanitized_data['powerball']),
                safe_float(sanitized_data['score_total']),
                str(sanitized_data['model_version']),
                str(sanitized_data['dataset_hash']),
                sanitized_data.get('json_details_path', None),
                target_draw_date_str # Usar la fecha validada
            ))

            prediction_id = cursor.lastrowid
            conn.commit()

            logger.info(f"Prediction saved with ID {prediction_id}.")
            return prediction_id

    except Exception as e:
        logger.error(f"Error saving prediction log: {e}")
        return None


def get_prediction_history(limit: int = 50):
    """
    Recupera el historial de predicciones de la base de datos.

    Args:
        limit: Número máximo de predicciones a recuperar

    Returns:
        DataFrame o Lista de diccionarios con el historial de predicciones
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

            # The original request implied returning a DataFrame by default
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


def get_predictions_grouped_by_date(limit_dates: int = 25) -> List[Dict]:
    """
    Obtiene predicciones agrupadas por fecha del sorteo objetivo para mostrar historial.
    CORREGIDO: Ahora agrupa por target_draw_date (fecha del sorteo) en lugar de created_at.

    Args:
        limit_dates: Número máximo de fechas a retornar (default: 25)

    Returns:
        Lista de diccionarios con estructura:
        - date: fecha del sorteo objetivo
        - formatted_date: fecha formateada en español
        - total_plays: total de predicciones para ese sorteo
        - winning_plays: número de predicciones que ganaron algún premio
        - best_prize: mejor premio obtenido
        - total_prize_amount: suma total de premios
        - predictions: lista de predicciones detalladas
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()

            # Get unique target drawing dates with prediction counts
            cursor.execute("""
                SELECT
                    COALESCE(target_draw_date, DATE(created_at)) as target_date,
                    COUNT(*) as total_predictions
                FROM predictions_log
                WHERE created_at IS NOT NULL
                GROUP BY COALESCE(target_draw_date, DATE(created_at))
                ORDER BY target_date DESC
                LIMIT ?
            """, (limit_dates,))

            date_groups = cursor.fetchall()

            grouped_results = []

            # Spanish month names for formatting
            spanish_months = {
                1: 'Ene', 2: 'Feb', 3: 'Mar', 4: 'Abr', 5: 'May', 6: 'Jun',
                7: 'Jul', 8: 'Ago', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dic'
            }

            for date_row in date_groups:
                target_date = date_row[0]
                total_predictions = date_row[1]

                # Get all predictions for this target drawing date
                cursor.execute("""
                    SELECT
                        id, timestamp, n1, n2, n3, n4, n5, powerball,
                        score_total, model_version, dataset_hash,
                        COALESCE(target_draw_date, DATE(created_at)) as target_draw_date,
                        created_at
                    FROM predictions_log
                    WHERE COALESCE(target_draw_date, DATE(created_at)) = ?
                    ORDER BY score_total DESC
                """, (target_date,))

                predictions_data = cursor.fetchall()

                # Process predictions and calculate statistics
                predictions = []
                total_prize = 0.0
                winning_predictions = 0
                best_prize_amount = 0.0
                best_prize_description = "No matches"

                for pred_row in predictions_data:
                    prediction_numbers = [pred_row[2], pred_row[3], pred_row[4], pred_row[5], pred_row[6]]
                    prediction_pb = pred_row[7]
                    prediction_target_date = pred_row[11] or target_date  # target_draw_date is now index 11
                    prediction_created_at = pred_row[12]  # created_at is now index 12

                    # Find matching official result for the TARGET drawing date (not creation date)
                    cursor.execute("""
                        SELECT n1, n2, n3, n4, n5, pb
                        FROM powerball_draws
                        WHERE draw_date = ?
                    """, (prediction_target_date,))

                    official_result = cursor.fetchone()

                    # Calculate matches and prizes
                    matches_main = 0
                    powerball_match = False
                    prize_amount = 0.0
                    prize_description = "No matches"

                    if official_result:
                        winning_numbers = [official_result[0], official_result[1], official_result[2],
                                         official_result[3], official_result[4]]
                        winning_pb = official_result[5]

                        # Count main number matches
                        matches_main = len(set(prediction_numbers) & set(winning_numbers))
                        powerball_match = prediction_pb == winning_pb

                        # Calculate prize
                        prize_amount, prize_description = calculate_prize_amount(matches_main, powerball_match)

                    # Update statistics
                    total_prize += prize_amount
                    if prize_amount > 0:
                        winning_predictions += 1

                    if prize_amount > best_prize_amount:
                        best_prize_amount = prize_amount
                        best_prize_description = prize_description

                    # Add prediction to list
                    prediction_detail = {
                        "prediction_id": pred_row[0],
                        "timestamp": pred_row[1],
                        "numbers": prediction_numbers,
                        "powerball": prediction_pb,
                        "score": float(pred_row[8]),
                        "model_version": pred_row[9],  # model_version is correct at index 9
                        "dataset_hash": pred_row[10],  # dataset_hash is correct at index 10
                        "target_draw_date": prediction_target_date,
                        "created_at": prediction_created_at,
                        "matches_main": matches_main,
                        "powerball_match": powerball_match,
                        "prize_amount": prize_amount,
                        "prize_description": prize_description,
                        "has_prize": prize_amount > 0
                    }

                    predictions.append(prediction_detail)

                # Format date in Spanish
                try:
                    date_obj = datetime.strptime(target_date, '%Y-%m-%d')
                    formatted_date = f"{date_obj.day} {spanish_months[date_obj.month]} {date_obj.year}"
                except:
                    formatted_date = target_date

                # Format total prize display
                if best_prize_amount >= 100000000:
                    total_prize_display = "JACKPOT!"
                elif total_prize >= 1000000:
                    total_prize_display = f"${total_prize/1000000:.1f}M"
                elif total_prize >= 1000:
                    total_prize_display = f"${total_prize/1000:.0f}K"
                elif total_prize > 0:
                    total_prize_display = f"${total_prize:.0f}"
                else:
                    total_prize_display = "$0"

                # Calculate win rate
                win_rate = (winning_predictions / total_predictions * 100) if total_predictions > 0 else 0.0

                grouped_result = {
                    "date": target_date,
                    "formatted_date": formatted_date,
                    "total_plays": total_predictions,
                    "winning_plays": winning_predictions,
                    "win_rate_percentage": f"{win_rate:.1f}%",
                    "best_prize": best_prize_description,
                    "best_prize_amount": best_prize_amount,
                    "total_prize_amount": total_prize,
                    "total_prize_display": total_prize_display,
                    "predictions": predictions,
                    "is_target_draw_date": True,
                    "context": f"Predictions generated for drawing on {formatted_date}"
                }

                grouped_results.append(grouped_result)

            logger.info(f"Retrieved {len(grouped_results)} grouped prediction dates with {sum(g['total_plays'] for g in grouped_results)} total predictions")
            return grouped_results

    except Exception as e:
        logger.error(f"Error retrieving grouped predictions by date: {e}")
        return []


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


# ========================================================================
# HYBRID CONFIGURATION SYSTEM - SIMPLE & ROBUST
# ========================================================================

def migrate_config_from_file() -> bool:
    """
    Migra la configuración desde config.ini a la base de datos.
    Solo se ejecuta en la primera inicialización del sistema.

    Returns:
        bool: True si la migración fue exitosa
    """
    try:
        # Verificar si ya hay configuración en DB
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM system_config")
            config_count = cursor.fetchone()[0]

            if config_count > 0:
                logger.info("Configuration already exists in database, skipping migration")
                return True

        # Leer configuración del archivo
        config = configparser.ConfigParser()
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, '..', 'config', 'config.ini')

        if not os.path.exists(config_path):
            logger.warning(f"Config file not found at {config_path}, using default values")
            return _create_default_config_in_db()

        config.read(config_path)

        # Migrar cada sección del archivo a la DB
        with get_db_connection() as conn:
            cursor = conn.cursor()

            for section_name in config.sections():
                for key, value in config.items(section_name):
                    cursor.execute("""
                        INSERT OR REPLACE INTO system_config (section, key, value)
                        VALUES (?, ?, ?)
                    """, (section_name, key, value))

            # Marcar migración como completada
            cursor.execute("""
                INSERT OR REPLACE INTO system_config (section, key, value)
                VALUES ('system', 'config_migrated', 'true')
            """)

            conn.commit()

        logger.info("Configuration successfully migrated from config.ini to database")
        return True

    except Exception as e:
        logger.error(f"Error migrating configuration from file: {e}")
        return False


def _create_default_config_in_db() -> bool:
    """Crea configuración por defecto en la base de datos"""
    try:
        default_config = {
            'paths': {
                'db_file': 'data/shiolplus.db',
                'model_file': 'models/shiolplus.pkl'
            },
            'pipeline': {
                'execution_days_monday': 'True',
                'execution_days_wednesday': 'True',
                'execution_days_saturday': 'True',
                'execution_time': '02:00',
                'timezone': 'America/New_York',
                'auto_execution': 'True'
            },
            'predictions': {
                'count': '50',
                'method': 'deterministic'
            },
            'weights': {
                'probability': '50',
                'diversity': '20',
                'historical': '20',
                'risk': '10'
            }
        }

        with get_db_connection() as conn:
            cursor = conn.cursor()

            for section_name, section_data in default_config.items():
                for key, value in section_data.items():
                    cursor.execute("""
                        INSERT OR REPLACE INTO system_config (section, key, value)
                        VALUES (?, ?, ?)
                    """, (section_name, key, value))

            # Marcar como configuración por defecto
            cursor.execute("""
                INSERT OR REPLACE INTO system_config (section, key, value)
                VALUES ('system', 'config_migrated', 'default')
            """)

            conn.commit()

        logger.info("Default configuration created in database")
        return True

    except Exception as e:
        logger.error(f"Error creating default configuration: {e}")
        return False


def load_config_from_db() -> Dict[str, Any]:
    """
    Carga la configuración desde la base de datos.
    Implementa fallback automático al archivo config.ini si falla.

    Returns:
        Dict con la configuración del sistema
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT section, key, value FROM system_config")
            config_rows = cursor.fetchall()

            if not config_rows:
                logger.warning("No configuration found in database, using file fallback")
                return _load_config_from_file()

            # Convertir a estructura de diccionario
            config_dict = {}
            for section, key, value in config_rows:
                if section not in config_dict:
                    config_dict[section] = {}
                config_dict[section][key] = value

            logger.info("Configuration loaded successfully from database")
            return config_dict

    except Exception as e:
        logger.error(f"Error loading configuration from database: {e}")
        logger.info("Falling back to config file")
        return _load_config_from_file()


def _load_config_from_file() -> Dict[str, Any]:
    """Fallback: carga configuración desde archivo"""
    try:
        config = configparser.ConfigParser()
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, '..', 'config', 'config.ini')

        config.read(config_path)

        config_dict = {}
        for section in config.sections():
            config_dict[section] = dict(config.items(section))

        logger.info(f"Configuration loaded from {config_path}")
        return config_dict

    except Exception as e:
        logger.error(f"Error loading configuration from file: {e}")
        return {}


def save_config_to_db(config_data: Dict[str, Any]) -> bool:
    """
    Guarda la configuración en la base de datos.
    Esta función es llamada desde el dashboard.

    Args:
        config_data: Diccionario con la configuración

    Returns:
        bool: True si se guardó exitosamente
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()

            # Guardar cada valor en la tabla
            for section_name, section_data in config_data.items():
                if isinstance(section_data, dict):
                    for key, value in section_data.items():
                        cursor.execute("""
                            INSERT OR REPLACE INTO system_config (section, key, value, updated_at)
                            VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                        """, (section_name, key, str(value)))
                else:
                    # Valor directo (no anidado)
                    cursor.execute("""
                        INSERT OR REPLACE INTO system_config (section, key, value, updated_at)
                        VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                    """, ('general', section_name, str(section_data)))

            conn.commit()

        logger.info("Configuration saved successfully to database")
        return True

    except Exception as e:
        logger.error(f"Error saving configuration to database: {e}")
        return False


def get_config_value(section: str, key: str, default: Any = None) -> Any:
    """
    Obtiene un valor específico de configuración con fallback automático.

    Args:
        section: Sección de configuración
        key: Clave específica
        default: Valor por defecto si no se encuentra

    Returns:
        Valor de configuración o default
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT value FROM system_config
                WHERE section = ? AND key = ?
            """, (section, key))

            result = cursor.fetchone()
            if result:
                return result[0]

        # Fallback a archivo si no está en DB
        config = configparser.ConfigParser()
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, '..', 'config', 'config.ini')

        if os.path.exists(config_path):
            config.read(config_path)
            if config.has_section(section) and config.has_option(section, key):
                return config.get(section, key)

        return default

    except Exception as e:
        logger.error(f"Error getting config value {section}.{key}: {e}")
        return default


def is_config_initialized() -> bool:
    """
    Verifica si el sistema de configuración híbrido está inicializado.

    Returns:
        bool: True si la configuración está inicializada
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT value FROM system_config
                WHERE section = 'system' AND key = 'config_migrated'
            """)

            result = cursor.fetchone()
            return result is not None

    except Exception as e:
        logger.error(f"Error checking config initialization: {e}")
        return False


# ========================================================================
# FASE 2: FUNCIONES DE VALIDACIÓN DE FECHAS
# ========================================================================

def _validate_target_draw_date(date_str: str) -> bool:
    """
    Valida que target_draw_date tenga el formato correcto usando DateManager centralizado.

    Args:
        date_str: String de fecha a validar

    Returns:
        True si la fecha es válida
    """
    from src.date_utils import DateManager

    logger.debug(f"Validating target draw date using centralized DateManager: {date_str}")
    return DateManager.validate_date_format(date_str)


def _is_valid_drawing_date(date_str: str) -> bool:
    """
    Verifica que una fecha corresponda a un día de sorteo usando DateManager centralizado.
    Los sorteos son: Lunes (0), Miércoles (2), Sábado (5)

    Args:
        date_str: Fecha en formato YYYY-MM-DD

    Returns:
        True si es un día de sorteo válido
    """
    from src.date_utils import DateManager

    logger.debug(f"Validating drawing date using centralized DateManager: {date_str}")
    return DateManager.is_valid_drawing_date(date_str)


def _sanitize_prediction_data(prediction_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sanitiza y valida todos los campos de una predicción antes de guardar.

    Args:
        prediction_data: Datos de predicción originales

    Returns:
        Datos de predicción sanitizados y validados
    """
    sanitized_data = prediction_data.copy()

    # Validar y corregir timestamp
    if 'timestamp' not in sanitized_data or not sanitized_data['timestamp']:
        sanitized_data['timestamp'] = datetime.now().isoformat()

    # Validar números principales (deben estar entre 1-69)
    if 'numbers' in sanitized_data:
        numbers = sanitized_data['numbers']
        if isinstance(numbers, list) and len(numbers) == 5:
            valid_numbers = []
            for num in numbers:
                try:
                    num_int = int(num)
                    if 1 <= num_int <= 69:
                        valid_numbers.append(num_int)
                    else:
                        logger.warning(f"Number {num} is outside valid range 1-69")
                        return None  # Datos inválidos
                except (ValueError, TypeError):
                    logger.error(f"Invalid number format: {num}")
                    return None
            sanitized_data['numbers'] = valid_numbers
        else:
            logger.error(f"Invalid numbers format: {numbers}")
            return None

    # Validar Powerball (debe estar entre 1-26)
    if 'powerball' in sanitized_data:
        try:
            pb = int(sanitized_data['powerball'])
            if not (1 <= pb <= 26):
                logger.error(f"Powerball {pb} is outside valid range 1-26")
                return None
            sanitized_data['powerball'] = pb
        except (ValueError, TypeError):
            logger.error(f"Invalid powerball format: {sanitized_data['powerball']}")
            return None

    # Validar score_total
    if 'score_total' in sanitized_data:
        try:
            score = float(sanitized_data['score_total'])
            if score < 0 or score > 1:
                logger.warning(f"Score {score} is outside typical range 0-1")
            sanitized_data['score_total'] = score
        except (ValueError, TypeError):
            logger.error(f"Invalid score format: {sanitized_data['score_total']}")
            return None

    # Validar model_version
    if 'model_version' not in sanitized_data or not sanitized_data['model_version']:
        sanitized_data['model_version'] = '1.0.0'  # Default version

    # Validar dataset_hash
    if 'dataset_hash' not in sanitized_data or not sanitized_data['dataset_hash']:
        # Generar hash por defecto basado en timestamp
        import hashlib
        timestamp_str = str(sanitized_data.get('timestamp', datetime.now().isoformat()))
        default_hash = hashlib.md5(timestamp_str.encode()).hexdigest()[:16]
        sanitized_data['dataset_hash'] = default_hash

    return sanitized_data
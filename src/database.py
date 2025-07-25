import sqlite3
import pandas as pd
from loguru import logger
import configparser
import os
from typing import Optional

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
    Initializes the database by creating the powerball_draws table if it doesn't exist.
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
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
            conn.commit()
            logger.info("Database initialized. 'powerball_draws' table is ready.")
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
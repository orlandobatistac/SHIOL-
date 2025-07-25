import os
from typing import Optional

import pandas as pd
import requests
from loguru import logger

# Import the new database module
from src.database import (
    bulk_insert_draws,
    get_all_draws,
    get_latest_draw_date,
    initialize_database,
)


class DataLoader:
    """
    Loads historical Powerball data from the SQLite database.
    This class acts as an abstraction layer for data retrieval.
    """

    def __init__(self):
        """Initializes the DataLoader."""
        logger.info("DataLoader initialized to read from SQLite database.")

    def load_historical_data(self) -> pd.DataFrame:
        """
        Loads all historical Powerball data from the database.

        Returns:
            pd.DataFrame: A DataFrame with all historical data, sorted by date.
                         Returns an empty DataFrame if an error occurs.
        """
        try:
            # The database module now handles the details of the query.
            df = get_all_draws()
            if df.empty:
                logger.warning(
                    "Historical data is empty. The database may need to be updated."
                )
            return df
        except Exception as e:
            logger.error(
                f"An unexpected error occurred while loading data from the database: {e}"
            )
            return pd.DataFrame()


def get_data_loader() -> DataLoader:
    """
    Factory function to get an instance of DataLoader.
    """
    return DataLoader()


def update_database_from_source() -> int:
    """
    Downloads the latest Powerball data, compares it with the existing database,
    and inserts only the new draws. Initializes the database if it's empty.

    Returns:
        int: Total number of rows in the database after the update.
    """
    logger.info("Starting data update process...")
    # 1. Ensure the database and table exist
    initialize_database()

    # 2. Download the latest data from the source
    url = "https://nclottery.com/powerball-download"
    temp_csv_path = "data/temp_powerball_data.csv"

    downloaded_df = _download_data(url, temp_csv_path)
    if downloaded_df is None:
        logger.error("Update process failed: could not download data.")
        return len(get_all_draws())

    # 3. Transform and standardize the downloaded data
    transformed_df = _transform_data(downloaded_df)
    if transformed_df is None:
        logger.error("Update process failed: could not transform data.")
        return len(get_all_draws())

    # 4. Get the latest date from our database
    latest_date_in_db = get_latest_draw_date()

    # 5. Filter for new draws
    if latest_date_in_db:
        # Convert to datetime to ensure correct comparison
        transformed_df["draw_date"] = pd.to_datetime(transformed_df["draw_date"])
        latest_date_in_db_dt = pd.to_datetime(latest_date_in_db)

        new_draws_df = transformed_df[
            transformed_df["draw_date"] > latest_date_in_db_dt
        ].copy()
        logger.info(f"Found {len(new_draws_df)} new draws to add.")
    else:
        # If database is empty, all transformed data is new
        new_draws_df = transformed_df.copy()
        logger.info(f"Database is empty. Populating with {len(new_draws_df)} draws.")

    # 6. Insert new draws into the database
    if not new_draws_df.empty:
        # Convert date back to string for SQLite storage
        new_draws_df["draw_date"] = new_draws_df["draw_date"].dt.strftime("%Y-%m-%d")
        bulk_insert_draws(new_draws_df)

    # 7. Clean up the temporary file
    try:
        if os.path.exists(temp_csv_path):
            os.remove(temp_csv_path)
            logger.info(f"Removed temporary file: {temp_csv_path}")
    except OSError as e:
        logger.error(f"Error removing temporary file {temp_csv_path}: {e}")

    # 8. Return the total number of rows in the table
    final_df = get_all_draws()
    return len(final_df)


def _download_data(url: str, output_path: str) -> Optional[pd.DataFrame]:
    """Downloads data and returns a DataFrame."""
    try:
        logger.info(f"Downloading Powerball data from {url}")
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(url, headers=headers, allow_redirects=True, timeout=30)
        response.raise_for_status()

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(response.content)

        df = pd.read_csv(output_path)
        logger.info(f"Successfully loaded {len(df)} rows from downloaded file.")
        return df
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error occurred while downloading data: {e}")
        return None
    except pd.errors.ParserError as e:
        logger.error(f"Error parsing CSV file: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error in _download_data: {e}")
        return None


def _transform_data(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Transforms the raw DataFrame from the download source."""
    try:
        logger.info("Transforming downloaded data.")

        # Standardize column names from 'Date', 'Number 1', etc. to 'draw_date', 'n1', etc.
        column_mapping = {
            "Date": "draw_date",
            "Number 1": "n1",
            "Number 2": "n2",
            "Number 3": "n3",
            "Number 4": "n4",
            "Number 5": "n5",
            "Powerball": "pb",
        }
        # Check if expected columns are present
        if not all(col in df.columns for col in column_mapping.keys()):
            logger.error(
                f"Downloaded CSV does not have the expected columns. Got: {df.columns.tolist()}"
            )
            return None

        df.rename(columns=column_mapping, inplace=True)

        # Filter out rows that don't have a valid date format (like a disclaimer)
        date_pattern = r"^\d{2}/\d{2}/\d{4}$"
        df = df[df["draw_date"].str.match(date_pattern, na=False)].copy()

        # Convert date column to datetime objects for sorting
        df["draw_date"] = pd.to_datetime(df["draw_date"], format="%m/%d/%Y")

        # Sort by date
        df.sort_values(by="draw_date", ascending=True, inplace=True)

        # Select and reorder columns
        required_cols = ["draw_date", "n1", "n2", "n3", "n4", "n5", "pb"]
        df = df[required_cols]

        # Drop duplicates just in case
        df.drop_duplicates(subset=["draw_date"], inplace=True)

        logger.info(f"Transformation complete. {len(df)} valid rows processed.")
        return df
    except Exception as e:
        logger.error(f"An error occurred during data transformation: {e}")
        return None

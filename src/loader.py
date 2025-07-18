import configparser
import os

import pandas as pd
import requests
from loguru import logger
from typing import Dict, List, Optional, Tuple, Union

class ColumnStandardizer:
    """
    A utility class for standardizing column names in data frames.

    This class helps ensure that data frames have consistent column names
    regardless of the source or original naming convention. It provides
    methods to detect column naming patterns, standardize columns based on
    predefined mappings, and validate that required columns are present.

    Attributes:
        standard_column_names (Dict[str, str]): Dictionary mapping standard column names
            to their descriptions
        column_mappings (Dict[str, Dict[str, str]]): Dictionary of column mapping patterns
            for different data sources
    """

    def __init__(self):
        """Initialize the ColumnStandardizer with predefined column mappings."""
        # Define standard column names and their descriptions
        self.standard_column_names = {
            "draw_date": "Date of the lottery draw",
            "n1": "First white ball number",
            "n2": "Second white ball number",
            "n3": "Third white ball number",
            "n4": "Fourth white ball number",
            "n5": "Fifth white ball number",
            "pb": "Powerball number",
        }

        # Define column mapping patterns for different data sources
        self.column_mappings = {
            "original": {
                "Date": "draw_date",
                "Number 1": "n1",
                "Number 2": "n2",
                "Number 3": "n3",
                "Number 4": "n4",
                "Number 5": "n5",
                "Powerball": "pb",
            },
            "transformed": {
                "white_ball_1": "n1",
                "white_ball_2": "n2",
                "white_ball_3": "n3",
                "white_ball_4": "n4",
                "white_ball_5": "n5",
                "powerball": "pb",
            },
            "alternative": {
                "date": "draw_date",
                "ball1": "n1",
                "ball2": "n2",
                "ball3": "n3",
                "ball4": "n4",
                "ball5": "n5",
                "power_ball": "pb",
            },
        }

        logger.info("ColumnStandardizer initialized with predefined column mappings")

    def detect_column_pattern(self, df: pd.DataFrame) -> str:
        """
        Detect the column naming pattern in the given DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame to analyze

        Returns:
            str: The detected pattern name ('original', 'transformed', 'alternative', or 'unknown')
        """
        column_set = set(df.columns)

        # Check for each known pattern
        for pattern_name, mapping in self.column_mappings.items():
            # If all or most of the source columns in this pattern are present
            source_columns = set(mapping.keys())
            if (
                len(source_columns.intersection(column_set))
                >= len(source_columns) * 0.7
            ):
                logger.info(f"Detected column pattern: {pattern_name}")
                return pattern_name

        # If no known pattern is detected
        logger.warning(f"Unknown column pattern detected: {', '.join(column_set)}")
        return "unknown"

    def standardize_columns(
        self, df: pd.DataFrame, pattern: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Standardize column names in the given DataFrame based on the detected or specified pattern.

        Args:
            df (pd.DataFrame): The DataFrame to standardize
            pattern (Optional[str]): The column pattern to use for standardization.
                                    If None, the pattern will be auto-detected.

        Returns:
            pd.DataFrame: A new DataFrame with standardized column names
        """
        # Make a copy to avoid modifying the original
        standardized_df = df.copy()

        # Detect pattern if not specified
        if pattern is None or pattern not in self.column_mappings:
            pattern = self.detect_column_pattern(df)

        # If pattern is unknown, try to make educated guesses
        if pattern == "unknown":
            logger.warning("Using heuristic column mapping for unknown pattern")
            return self._apply_heuristic_mapping(standardized_df)

        # Apply the mapping for the detected pattern
        mapping = self.column_mappings[pattern]
        rename_dict = {}

        for source_col, target_col in mapping.items():
            if source_col in standardized_df.columns:
                rename_dict[source_col] = target_col

        if rename_dict:
            standardized_df.rename(columns=rename_dict, inplace=True)
            logger.info(
                f"Standardized {len(rename_dict)} columns using '{pattern}' pattern"
            )

        return standardized_df

    def _apply_heuristic_mapping(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply heuristic mapping for unknown column patterns.

        This method tries to map columns based on common patterns and naming conventions
        when the exact pattern cannot be determined.

        Args:
            df (pd.DataFrame): The DataFrame to standardize

        Returns:
            pd.DataFrame: A DataFrame with heuristically standardized column names
        """
        # Make a copy to avoid modifying the original
        result_df = df.copy()

        # Common date column patterns
        date_patterns = [
            "date",
            "draw_date",
            "draw date",
            "drawdate",
            "Date",
            "dt",
            "draw_dt",
        ]
        for col in result_df.columns:
            # Check for date columns
            if any(pattern in col.lower() for pattern in ["date", "day", "time", "dt"]):
                result_df.rename(columns={col: "draw_date"}, inplace=True)
                logger.info(f"Heuristically mapped '{col}' to 'draw_date'")

            # Check for number columns with explicit handling for 'num#' pattern
            elif (
                "number" in col.lower() or "ball" in col.lower() or "num" in col.lower()
            ):
                # First check for direct 'num#' pattern (like num1, num2, etc.)
                if col.startswith("num") and len(col) > 3 and col[3:].isdigit():
                    num = int(col[3:])
                    if 1 <= num <= 5:
                        result_df.rename(columns={col: f"n{num}"}, inplace=True)
                        logger.info(f"Heuristically mapped '{col}' to 'n{num}'")
                        continue

                # Try to extract the ball number for other patterns
                for i in range(1, 6):
                    if str(i) in col or f"_{i}" in col or f"-{i}" in col:
                        result_df.rename(columns={col: f"n{i}"}, inplace=True)
                        logger.info(f"Heuristically mapped '{col}' to 'n{i}'")
                        break

            # Check for powerball column
            elif "power" in col.lower() or "pb" in col.lower():
                result_df.rename(columns={col: "pb"}, inplace=True)
                logger.info(f"Heuristically mapped '{col}' to 'pb'")

        return result_df

    def validate_required_columns(
        self, df: pd.DataFrame, required_columns: Optional[List[str]] = None
    ) -> Tuple[bool, List[str]]:
        """
        Validate that the DataFrame contains all required columns.

        Args:
            df (pd.DataFrame): The DataFrame to validate
            required_columns (Optional[List[str]]): List of required column names.
                                                  If None, all standard columns are required.

        Returns:
            Tuple[bool, List[str]]: A tuple containing:
                - A boolean indicating if all required columns are present
                - A list of missing column names
        """
        if required_columns is None:
            required_columns = list(self.standard_column_names.keys())

        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            logger.warning(f"Missing required columns: {', '.join(missing_columns)}")
            return False, missing_columns

        logger.info("All required columns are present")
        return True, []

    def ensure_draw_date_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure that the DataFrame has a 'draw_date' column.

        This method specifically addresses the 'Missing draw_date Column' warning
        by ensuring that a properly formatted draw_date column exists.

        Args:
            df (pd.DataFrame): The DataFrame to process

        Returns:
            pd.DataFrame: A DataFrame with a guaranteed 'draw_date' column
        """
        result_df = df.copy()

        # Check if draw_date column exists
        if "draw_date" not in result_df.columns:
            # Look for alternative date columns
            date_alternatives = ["Date", "date", "DrawDate", "draw date", "drawdate"]
            found = False

            for alt in date_alternatives:
                if alt in result_df.columns:
                    result_df.rename(columns={alt: "draw_date"}, inplace=True)
                    logger.info(f"Renamed '{alt}' column to 'draw_date'")
                    found = True
                    break

            if not found:
                logger.warning(
                    "No date column found. Creating a placeholder 'draw_date' column."
                )
                # Create a placeholder date column if none exists
                result_df["draw_date"] = pd.NaT

        # Ensure draw_date is in datetime format
        if not pd.api.types.is_datetime64_any_dtype(result_df["draw_date"]):
            try:
                result_df["draw_date"] = pd.to_datetime(result_df["draw_date"])
                logger.info("Converted 'draw_date' column to datetime format")
            except Exception as e:
                logger.error(f"Failed to convert 'draw_date' to datetime: {e}")

        return result_df

    def add_column_mapping(self, pattern_name: str, mapping: Dict[str, str]) -> None:
        """
        Add a new column mapping pattern.

        Args:
            pattern_name (str): The name of the new pattern
            mapping (Dict[str, str]): Dictionary mapping source column names to standard column names
        """
        if pattern_name in self.column_mappings:
            logger.warning(
                f"Overwriting existing column mapping pattern: {pattern_name}"
            )

        self.column_mappings[pattern_name] = mapping
        logger.info(f"Added new column mapping pattern: {pattern_name}")


class DataLoader:
    def __init__(self, data_file_path):
        self.data_file_path = data_file_path
        self.column_standardizer = ColumnStandardizer()
        self.required_cols = ["draw_date", "n1", "n2", "n3", "n4", "n5", "pb"]
        logger.info(f"DataLoader initialized for file: {data_file_path}")

    def load_historical_data(self):
        """
        Loads and cleans the historical Powerball data from the CSV file.

        Returns:
            pd.DataFrame: A DataFrame with standardized column names and properly
                         formatted data, or an empty DataFrame if an error occurs.
        """
        try:
            # Load the data
            df = pd.read_csv(self.data_file_path)
            initial_row_count = len(df)
            logger.info(
                f"Successfully loaded {initial_row_count} rows from "
                f"{self.data_file_path}"
            )

            # Process the dataframe through a single pipeline to reduce try/except
            # blocks
            df = self._process_dataframe(df)

            if df.empty:
                return df

            # Final row count
            final_row_count = len(df)
            logger.info(
                f"Data cleaning and structuring complete. "
                f"Processed {final_row_count} rows."
            )
            return df

        except FileNotFoundError:
            logger.error(f"Data file not found at {self.data_file_path}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"An error occurred while loading data: {e}")
            return pd.DataFrame()

    def _process_dataframe(self, df):
        """
        Process the dataframe through standardization, validation, and cleaning.

        Args:
            df (pd.DataFrame): The dataframe to process

        Returns:
            pd.DataFrame: The processed dataframe, or empty dataframe if processing
                fails
        """
        try:
            # Standardize column names
            df = self.column_standardizer.standardize_columns(df)
            logger.info(f"Standardized column names: {', '.join(df.columns)}")

            # Ensure draw_date column exists and is in the correct format
            df = self.column_standardizer.ensure_draw_date_column(df)
            logger.info("Ensured draw_date column is present and properly formatted")

            # Validate that all required columns are present
            is_valid, missing_cols = self.column_standardizer.validate_required_columns(
                df, self.required_cols
            )

            if not is_valid:
                logger.error(f"Missing required columns: {', '.join(missing_cols)}")
                return pd.DataFrame()

            # Select only the necessary columns
            df = df[self.required_cols]

            # Remove any rows with invalid data
            original_count = len(df)
            df = df.dropna(subset=["draw_date"])
            if len(df) < original_count:
                logger.warning(
                    f"Removed {original_count - len(df)} rows with missing "
                    f"draw_date values"
                )

            return df

        except Exception as e:
            logger.error(f"Error during dataframe processing: {e}")
            return pd.DataFrame()


def get_data_loader():
    """
    Factory function to get an instance of DataLoader.
    """
    config = configparser.ConfigParser()
    config.read("config/config.ini")
    data_file = config["paths"]["data_file"]
    return DataLoader(data_file)


def update_powerball_data_from_csv():
    """
    Downloads the latest historical Powerball results from the North Carolina Lottery,
    saves it to data/NCELPowerball.csv, and performs necessary transformations.

    Returns:
        int: Total number of rows in the updated dataset
    """
    url = "https://nclottery.com/powerball-download"
    output_path = "data/NCELPowerball.csv"

    # Download the data
    df = _download_powerball_data(url, output_path)
    if df is None or df.empty:
        return 0

    # Transform the data
    df = _transform_powerball_data(df, output_path)
    if df is None or df.empty:
        return 0

    row_count = len(df)
    logger.info(f"Data transformation complete. Total rows saved: {row_count}")
    return row_count


def _download_powerball_data(url, output_path):
    """
    Downloads Powerball data from the specified URL and saves it to the output path.

    Args:
        url (str): The URL to download the data from
        output_path (str): The path to save the downloaded data to

    Returns:
        pd.DataFrame: The loaded DataFrame, or None if download fails
    """
    try:
        logger.info(f"Downloading Powerball data from {url}")

        # Set up headers to mimic a browser request
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/91.0.4472.124 Safari/537.36"
        }

        # Download the CSV file
        response = requests.get(url, headers=headers, allow_redirects=True, timeout=30)

        # Check if the request was successful
        if response.status_code != 200:
            logger.error(
                f"Failed to download data: HTTP status code {response.status_code}"
            )
            return None

        # Ensure the directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save the downloaded file
        with open(output_path, "wb") as f:
            f.write(response.content)

        logger.info(f"Downloaded data saved to {output_path}")

        # Load the CSV into a DataFrame
        df = pd.read_csv(output_path)
        logger.info(f"Loaded {len(df)} rows from CSV file")
        return df

    except requests.exceptions.RequestException as e:
        logger.error(f"Network error occurred while downloading data: {e}")
        return None
    except pd.errors.ParserError as e:
        logger.error(f"Error parsing CSV file: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error in _download_powerball_data: {e}")
        return None


def _transform_powerball_data(df, output_path):
    """
    Transforms the Powerball data by standardizing columns, formatting dates,
    and ensuring the required columns are present.

    Args:
        df (pd.DataFrame): The DataFrame to transform
        output_path (str): The path to save the transformed data to

    Returns:
        pd.DataFrame: The transformed DataFrame, or None if transformation fails
    """
    try:
        logger.info("Transforming the data")
        initial_row_count = len(df)

        # Filter out rows that don't have a valid date format (like the disclaimer)
        date_pattern = r"^\d{2}/\d{2}/\d{4}$"
        df = df[df["Date"].str.match(date_pattern, na=False)]
        filtered_count = initial_row_count - len(df)
        if filtered_count > 0:
            logger.info(f"Filtered out {filtered_count} rows with invalid date format")

        # Use ColumnStandardizer to standardize column names
        column_standardizer = ColumnStandardizer()
        logger.info(f"Processing {len(df)} rows of data")

        # Standardize columns using the original mapping pattern
        df = column_standardizer.standardize_columns(df, pattern="original")

        # Ensure draw_date column exists and is properly formatted
        df = column_standardizer.ensure_draw_date_column(df)

        # Format draw_date to YYYY-MM-DD format
        df["draw_date"] = df["draw_date"].dt.strftime("%Y-%m-%d")
        logger.info(f"Successfully formatted {len(df)} draw_date entries")

        # Sort by draw_date in ascending order
        df = df.sort_values(by="draw_date", ascending=True)
        logger.info("Sorted data by draw_date in ascending order")

        # Remove duplicate rows
        original_count = len(df)
        df = df.drop_duplicates()
        duplicate_count = original_count - len(df)
        if duplicate_count > 0:
            logger.info(f"Removed {duplicate_count} duplicate rows")

        # Keep only the required columns in the correct order
        required_cols = ["draw_date", "n1", "n2", "n3", "n4", "n5", "pb"]
        df = df[required_cols]

        # Validate that all required columns are present
        is_valid, missing_cols = column_standardizer.validate_required_columns(
            df, required_cols
        )
        if not is_valid:
            logger.warning(
                f"Missing required columns after transformation: "
                f"{', '.join(missing_cols)}"
            )

        # Save the transformed data back to the CSV file
        df.to_csv(output_path, index=False)
        logger.info(f"Successfully saved transformed data to {output_path}")

        return df

    except Exception as e:
        logger.error(f"Error in _transform_powerball_data: {e}")
        return None

"""
Data Processing and Analysis Module for SHIOLPlus.

This module is responsible for loading the historical lottery data,
cleaning it, validating it, and performing an initial frequency analysis.
"""
import pandas as pd
from typing import Dict, Any, List, Set

def load_and_analyze_data(file_path: str) -> Dict[str, Any]:
    """
    Loads and analyzes historical lottery data from a CSV file.

    Args:
        file_path (str): The path to the CSV file containing the data.

    Returns:
        Dict[str, Any]: A dictionary containing:
            - 'main_freq' (pd.Series): Frequency of each main number (1-69).
            - 'powerball_freq' (pd.Series): Frequency of each Powerball number (1-26).
            - 'historical_winners' (List[Set[int]]): A list of all
              historical winning combinations as sets of 5 numbers.
            - 'historical_draws' (List[Dict]): A list of historical draws,
              each with 'Winning_Set' and 'Powerball'.
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: Data file not found at path: {file_path}")
        return {}

    # Define number columns
    main_number_cols = [f'Number {i}' for i in range(1, 6)]
    powerball_col = 'Powerball'
    all_number_cols = main_number_cols + [powerball_col]

    # Data cleaning and validation
    df.dropna(subset=all_number_cols, inplace=True)
    for col in all_number_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce', downcast='integer')
    df.dropna(subset=all_number_cols, inplace=True)

    # Frequency analysis
    main_numbers_flat = df[main_number_cols].values.flatten()
    main_freq = pd.Series(main_numbers_flat).value_counts()

    powerball_freq = df[powerball_col].value_counts()

    # Extract historical winning combinations
    df['Winning_Set'] = df[main_number_cols].apply(lambda row: set(row), axis=1)
    historical_winners: List[Set[int]] = df['Winning_Set'].tolist()
    
    historical_draws = df[['Winning_Set', powerball_col]].to_dict('records')

    print("Data analysis complete.")
    print(f"Total historical draws analyzed: {len(df)}")

    return {
        "main_freq": main_freq,
        "powerball_freq": powerball_freq,
        "historical_winners": historical_winners,
        "historical_draws": historical_draws
    }

if __name__ == '__main__':
    # Example usage and module test
    from src.config import DATA_FILE_PATH
    analysis = load_and_analyze_data(DATA_FILE_PATH)
    if analysis:
        print("\n--- Analysis Results ---")
        print("Frequency of the 10 most common main numbers:")
        print(analysis['main_freq'].head(10))
        print("\nFrequency of the 5 most common Powerballs:")
        print(analysis['powerball_freq'].head(5))
        print(f"\nTotal historical combinations stored: {len(analysis['historical_winners'])}")
"""
Final Assembly and Export Module for SHIOLPlus.

This module takes the final high-performing combinations, selects the
most frequent Powerballs, assembles the complete plays, and exports
them to a CSV file. It also presents a summary to the console.
"""
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any

from src.config import NUM_POWERBALLS_TO_SELECT, OUTPUT_DIR

def export_final_plays(
    final_combinations: List[List[int]],
    powerball_analysis: pd.Series,
    num_powerballs: int = NUM_POWERBALLS_TO_SELECT
) -> str:
    """
    Assembles and exports the final plays to a CSV file.

    Args:
        final_combinations (List[List[int]]): The N selected final
                                              combinations.
        powerball_analysis (pd.Series): The frequency analysis of
                                        Powerball numbers.
        num_powerballs (int): The number of most frequent Powerballs to use.

    Returns:
        str: The path of the generated output file.
    """
    if not final_combinations:
        print("Error: No final combinations to export.")
        return ""

    # 1. Select the most frequent Powerballs
    top_powerballs = powerball_analysis.head(num_powerballs).index.tolist()
    print(f"\nSelected Powerballs (the {num_powerballs} most frequent): {top_powerballs}")

    # 2. Assemble the final plays
    final_plays = []
    for combo in final_combinations:
        for pb in top_powerballs:
            play = sorted(combo) + [pb]
            final_plays.append(play)

    # 3. Create DataFrame and export to CSV
    df_export = pd.DataFrame(final_plays, columns=['Num_1', 'Num_2', 'Num_3', 'Num_4', 'Num_5', 'Powerball'])
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    output_filename = f"SHIOLPlus_plays_{timestamp}.csv"
    output_path = f"{OUTPUT_DIR}/{output_filename}"

    try:
        df_export.to_csv(output_path, index=False)
        print(f"\nSuccess! {len(final_plays)} final plays exported to:")
        print(output_path)
        return output_path
    except IOError as e:
        print(f"Error writing the output file: {e}")
        return ""

if __name__ == '__main__':
    # Example usage and module test
    print("--- Exporter Module Test ---")
    
    # Sample data
    sample_final_combos = [
        [8, 16, 24, 32, 40],
        [1, 2, 3, 4, 5]
    ]
    sample_pb_analysis = pd.Series([10, 5, 3], index=[19, 4, 25]) # Frequencies, Indices are the numbers

    export_path = export_final_plays(sample_final_combos, sample_pb_analysis, num_powerballs=2)

    if export_path:
        print(f"\nTest finished. Example file created at: {export_path}")
        # Optional: read the file to verify
        # df_check = pd.read_csv(export_path)
        # print("\nContents of the example file:")
        # print(df_check)
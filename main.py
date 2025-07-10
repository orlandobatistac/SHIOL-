"""
SHIOLPlus - Strategic Lottery Optimization System

Main entry point to orchestrate the entire lottery analysis workflow,
from data ingestion to the export of optimized plays.
"""
import time

# Import functions from modules and configuration parameters
from src.config import DATA_FILE_PATH, NUM_FINAL_COMBINATIONS, NUM_POWERBALLS_TO_SELECT
from src.data_processor import load_and_analyze_data
from src.base_number_selector import select_strategic_base_numbers
from src.combination_generator import generate_and_filter_combinations
from src.performance_evaluator import evaluate_and_select_final_combinations
from src.output_exporter import export_final_plays

def main():
    """
    Executes the complete SHIOLPlus workflow.
    """
    start_time = time.time()
    print("=============================================")
    print("===   Starting SHIOLPlus Process   ===")
    print("=============================================\n")

    # --- Step 1: Data Loading and Analysis ---
    print("--- [Step 1/5] Loading and analyzing historical data...")
    analysis_results = load_and_analyze_data(DATA_FILE_PATH)
    if not analysis_results:
        print("\nProcess terminated due to an error in data loading.")
        return
    print("---------------------------------------------\n")

    # --- Step 2: Base Number Selection ---
    print("--- [Step 2/5] Selecting strategic base numbers...")
    base_numbers = select_strategic_base_numbers(analysis_results)
    if not base_numbers:
        print("\nProcess terminated. Could not select base numbers.")
        return
    print(f"Selected Base Numbers: {base_numbers}")
    print("---------------------------------------------\n")

    # --- Step 3: Combination Generation and Filtering ---
    print("--- [Step 3/5] Generating and filtering combinations...")
    filtered_combinations = generate_and_filter_combinations(
        base_numbers,
        analysis_results['historical_winners']
    )
    if not filtered_combinations:
        print("\nProcess terminated. No combinations passed the filters.")
        return
    print("---------------------------------------------\n")

    # --- Step 4: Performance Evaluation and Final Selection ---
    print("--- [Step 4/5] Evaluating performance via backtesting...")
    final_combinations = evaluate_and_select_final_combinations(
        filtered_combinations,
        analysis_results['historical_draws'],
        num_final_combinations=NUM_FINAL_COMBINATIONS
    )
    if not final_combinations:
        print("\nProcess terminated. Could not select final combinations.")
        return
    print("---------------------------------------------\n")

    # --- Step 5: Final Assembly and Export ---
    print("--- [Step 5/5] Assembling and exporting final plays...")
    output_file = export_final_plays(
        final_combinations,
        analysis_results['powerball_freq'],
        num_powerballs=NUM_POWERBALLS_TO_SELECT
    )
    
    end_time = time.time()
    total_time = end_time - start_time

    print("\n=============================================")
    print("===      SHIOLPlus Process Complete      ===")
    print("=============================================")
    print(f"Total execution time: {total_time:.2f} seconds")
    print(f"Final plays saved in: {output_file}")
    print("=============================================\n")


if __name__ == '__main__':
    main()
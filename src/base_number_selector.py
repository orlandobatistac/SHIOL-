"""
Strategic Base Number Selection (SBNS) Module for SHIOLPlus.

This module implements the logic to select a strategic set of 10 base
numbers, balancing historical frequency, range distribution, parity,
and separation rules.
"""
import pandas as pd
from typing import List, Dict, Any

from src.config import (
    LOW_RANGE_DEFINITION, MID_RANGE_DEFINITION, HIGH_RANGE_DEFINITION,
    LOW_RANGE_COUNT, MID_RANGE_COUNT, HIGH_RANGE_COUNT,
    EVEN_COUNT, ODD_COUNT, BASE_NUMBERS_COUNT, MIN_SEPARATION
)

def _is_balanced(numbers: List[int]) -> bool:
    """Checks if a set of numbers meets all balance criteria."""
    # Criterion 1: Range Distribution
    low_count = sum(1 for n in numbers if LOW_RANGE_DEFINITION[0] <= n <= LOW_RANGE_DEFINITION[1])
    mid_count = sum(1 for n in numbers if MID_RANGE_DEFINITION[0] <= n <= MID_RANGE_DEFINITION[1])
    high_count = sum(1 for n in numbers if HIGH_RANGE_DEFINITION[0] <= n <= HIGH_RANGE_DEFINITION[1])
    if not (low_count == LOW_RANGE_COUNT and mid_count == MID_RANGE_COUNT and high_count == HIGH_RANGE_COUNT):
        return False

    # Criterion 2: Even/Odd Balance
    even_count = sum(1 for n in numbers if n % 2 == 0)
    odd_count = len(numbers) - even_count
    if not (even_count == EVEN_COUNT and odd_count == ODD_COUNT):
        return False

    # Criterion 3: Separation Rules
    sorted_nums = sorted(numbers)
    consecutive_pairs = 0
    for i in range(len(sorted_nums) - 1):
        if sorted_nums[i+1] - sorted_nums[i] == 1:
            consecutive_pairs += 1
        # Check minimum separation for non-consecutive numbers
        if i < len(sorted_nums) - 2 and sorted_nums[i+1] - sorted_nums[i] > 1:
            if sorted_nums[i+1] - sorted_nums[i] < MIN_SEPARATION:
                return False # Does not meet minimum separation
    
    # No more than one consecutive pair and no 3 consecutive numbers
    if consecutive_pairs > 1:
        for i in range(len(sorted_nums) - 2):
            if sorted_nums[i+1] - sorted_nums[i] == 1 and sorted_nums[i+2] - sorted_nums[i+1] == 1:
                return False # Three consecutive
        if consecutive_pairs > 2: # More than one pair (e.g., 5,6 and 10,11)
             return False

    return True

def select_strategic_base_numbers(analysis_dict: Dict[str, Any]) -> List[int]:
    """
    Selects a set of 10 strategic base numbers.

    Args:
        analysis_dict (Dict[str, Any]): The analysis dictionary from the
                                        data_processor module.

    Returns:
        List[int]: An ordered list of 10 integers (the base numbers).
    """
    if not analysis_dict or "main_freq" not in analysis_dict:
        print("Error: Invalid or empty analysis dictionary.")
        return []

    # Weight numbers by frequency (higher frequency = higher probability)
    freq_series = analysis_dict["main_freq"]
    
    # This is a simplified approach. A more robust one would use search 
    # algorithms or a Constraint Satisfaction Problem (CSP) solver.
    # For now, we iterate through combinations of the most frequent numbers.
    from itertools import combinations

    # We try combinations from a larger pool of frequent numbers
    pool_size = 30 # Take the 30 most frequent to search for combinations
    number_pool = freq_series.head(pool_size).index.tolist()

    print(f"Searching for a balanced set of {BASE_NUMBERS_COUNT} numbers from a pool of {pool_size}...")

    for combo in combinations(number_pool, BASE_NUMBERS_COUNT):
        if _is_balanced(list(combo)):
            selected_set = sorted(list(combo))
            print(f"Balanced set found: {selected_set}")
            return selected_set
            
    print("Warning: A perfectly balanced set could not be found with the initial pool.")
    print("Returning the set of the 10 most frequent numbers as a fallback.")
    return sorted(freq_series.head(BASE_NUMBERS_COUNT).index.tolist())


if __name__ == '__main__':
    # Example usage and module test
    from src.data_processor import load_and_analyze_data
    from src.config import DATA_FILE_PATH
    
    analysis = load_and_analyze_data(DATA_FILE_PATH)
    if analysis:
        base_numbers = select_strategic_base_numbers(analysis)
        if base_numbers:
            print("\n--- Base Number Selection ---")
            print(f"Base numbers selected: {base_numbers}")
            print(f"Total numbers: {len(base_numbers)}")
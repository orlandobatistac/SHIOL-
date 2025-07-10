"""
Combination Generator and Filter Module for SHIOLPlus.

This module takes the selected base numbers, generates all possible
5-number combinations, and applies a series of heuristic filters
to discard the least probable ones.
"""
from itertools import combinations
from typing import List, Set

from src.config import (
    MIN_SUM_FILTER, MAX_SUM_FILTER
)

def _passes_sum_filter(combination: List[int]) -> bool:
    """Checks if the combination's sum is within the valid range."""
    return MIN_SUM_FILTER <= sum(combination) <= MAX_SUM_FILTER

def _passes_parity_filter(combination: List[int]) -> bool:
    """Checks for even/odd balance (2E/3O or 3E/2O)."""
    evens = sum(1 for n in combination if n % 2 == 0)
    odds = len(combination) - evens
    return (evens == 2 and odds == 3) or (evens == 3 and odds == 2)

def _passes_consecutive_filter(combination: List[int]) -> bool:
    """Discards combinations with 3 or more consecutive numbers."""
    sorted_combo = sorted(combination)
    for i in range(len(sorted_combo) - 2):
        if sorted_combo[i+1] - sorted_combo[i] == 1 and sorted_combo[i+2] - sorted_combo[i+1] == 1:
            return False
    return True

def generate_and_filter_combinations(
    base_numbers: List[int],
    historical_winners: List[Set[int]]
) -> List[List[int]]:
    """
    Generates and filters 5-number combinations from the base numbers.

    Args:
        base_numbers (List[int]): The list of 10 base numbers.
        historical_winners (List[Set[int]]): A list of historical winning
                                              combinations for filtering.

    Returns:
        List[List[int]]: A list of the combinations that passed all filters.
    """
    if not base_numbers or len(base_numbers) < 5:
        print("Error: At least 5 base numbers are required to generate combinations.")
        return []

    initial_combinations = list(combinations(base_numbers, 5))
    num_initial = len(initial_combinations)
    print(f"Generated {num_initial} initial combinations from the base numbers.")

    # Apply filters
    filtered_combinations = [
        list(combo) for combo in initial_combinations
        if _passes_sum_filter(list(combo)) and
           _passes_parity_filter(list(combo)) and
           _passes_consecutive_filter(list(combo)) and
           set(combo) not in historical_winners
    ]
    
    num_filtered = len(filtered_combinations)
    print(f"After applying filters, {num_filtered} combinations remain.")
    print(f"{num_initial - num_filtered} combinations were discarded.")

    return filtered_combinations

if __name__ == '__main__':
    # Example usage and module test
    from src.data_processor import load_and_analyze_data
    from src.base_number_selector import select_strategic_base_numbers
    from src.config import DATA_FILE_PATH

    analysis = load_and_analyze_data(DATA_FILE_PATH)
    if analysis:
        base_nums = select_strategic_base_numbers(analysis)
        if base_nums:
            filtered_combos = generate_and_filter_combinations(
                base_nums,
                analysis['historical_winners']
            )
            if filtered_combos:
                print("\n--- Combination Generator and Filter ---")
                print(f"Example of 5 filtered combinations:")
                for combo in filtered_combos[:5]:
                    print(f"- {sorted(combo)}, Sum: {sum(combo)}")
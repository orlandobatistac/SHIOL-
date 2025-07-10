"""
Performance Evaluation and Final Selection Module for SHIOLPlus.

This module backtests the filtered combinations against historical data,
calculates a "fitness" score for each, and selects the final set of
top-performing combinations.
"""
from typing import List, Dict, Any, Set

from src.config import NUM_FINAL_COMBINATIONS, FITNESS_SCORES

def evaluate_and_select_final_combinations(
    filtered_combinations: List[List[int]],
    historical_draws: List[Dict[str, Any]],
    num_final_combinations: int = NUM_FINAL_COMBINATIONS
) -> List[List[int]]:
    """
    Evaluates combinations via backtesting and selects the best ones.

    Args:
        filtered_combinations (List[List[int]]): The list of combinations
                                                 that passed the filters.
        historical_draws (List[Dict[str, Any]]): A list of dictionaries,
            where each represents a historical draw with 'Winning_Set'
            and 'Powerball'.
        num_final_combinations (int): The number of final combinations to
                                      select.

    Returns:
        List[List[int]]: The list of the N top-performing combinations.
    """
    if not filtered_combinations:
        print("Warning: No filtered combinations to evaluate.")
        return []

    print(f"Starting backtesting for {len(filtered_combinations)} combinations...")
    
    fitness_scores: Dict[tuple, int] = {tuple(sorted(c)): 0 for c in filtered_combinations}

    for draw in historical_draws:
        winning_set: Set[int] = draw['Winning_Set']
        for combo_list in filtered_combinations:
            combo_tuple = tuple(sorted(combo_list))
            matches = len(set(combo_list).intersection(winning_set))
            
            if matches in FITNESS_SCORES:
                fitness_scores[combo_tuple] += FITNESS_SCORES[matches]

    # Sort combinations by their fitness score
    sorted_combinations = sorted(
        fitness_scores.items(),
        key=lambda item: item[1],
        reverse=True
    )

    # Select the top N
    final_combinations_tuples = sorted_combinations[:num_final_combinations]
    final_combinations = [list(c) for c, score in final_combinations_tuples]

    print(f"Selected the top {len(final_combinations)} combinations after backtesting.")
    
    # Print a small report of the best ones
    print("\n--- Top 5 Combinations by Fitness Score ---")
    for combo, score in sorted_combinations[:5]:
        print(f"Combination: {list(combo)}, Fitness: {score}")

    return final_combinations

if __name__ == '__main__':
    # Example usage and module test
    # This is harder to test in isolation as it requires data from previous modules.
    # Sample data is used for demonstration.
    print("--- Performance Evaluator Module Test ---")
    
    # Sample data
    sample_filtered_combos = [
        [1, 2, 3, 4, 5], [10, 20, 30, 40, 50], [7, 14, 21, 28, 35],
        [5, 15, 25, 35, 45], [8, 16, 24, 32, 40]
    ]
    sample_historical_draws = [
        {'Winning_Set': {1, 2, 3, 10, 11}, 'Powerball': 5},
        {'Winning_Set': {10, 20, 30, 12, 13}, 'Powerball': 6},
        {'Winning_Set': {8, 16, 24, 32, 40}, 'Powerball': 7} # 5 matches
    ]

    final_selection = evaluate_and_select_final_combinations(
        sample_filtered_combos,
        sample_historical_draws,
        num_final_combinations=3
    )

    if final_selection:
        print("\n--- Example Final Selection ---")
        print(f"Selected combinations: {final_selection}")
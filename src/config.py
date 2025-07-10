"""
Configuration file for SHIOLPlus.

This file contains all the key parameters for the lottery analysis and
combination generation process. Centralizing these parameters makes it
easy to fine-tune the strategy without modifying the core logic.
"""
from typing import List, Dict

# --- Data Source ---
DATA_FILE_PATH: str = "data/NCELPowerball.csv"

# --- Base Number Selection (SSINB) ---
BASE_NUMBERS_COUNT: int = 10

# Range distribution for base numbers
LOW_RANGE_DEFINITION: List[int] = [1, 20]
MID_RANGE_DEFINITION: List[int] = [21, 40]
HIGH_RANGE_DEFINITION: List[int] = [41, 69]

LOW_RANGE_COUNT: int = 3
MID_RANGE_COUNT: int = 4
HIGH_RANGE_COUNT: int = 3

# Parity balance for base numbers
EVEN_COUNT: int = 5
ODD_COUNT: int = 5

# Separation rules
MIN_SEPARATION: int = 2

# --- Combination Generation & Filtering ---
# Sum filter for 5-number combinations
MIN_SUM_FILTER: int = 110
MAX_SUM_FILTER: int = 190

# --- Performance Evaluation ---
# Number of final combinations to select after backtesting
NUM_FINAL_COMBINATIONS: int = 20

# Fitness scores for backtesting
FITNESS_SCORES: Dict[int, int] = {
    3: 1,    # Score for 3 matches
    4: 10,   # Score for 4 matches
    5: 100   # Score for 5 matches
}

# --- Output Export ---
# Number of most frequent Powerballs to select
NUM_POWERBALLS_TO_SELECT: int = 3
OUTPUT_DIR: str = "outputs"
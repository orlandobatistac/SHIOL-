import numpy as np
import pandas as pd
from loguru import logger
from src.evaluator import Evaluator

class MonteCarloSimulator:
    def __init__(self, num_simulations=100000, play_cost=2):
        self.num_simulations = num_simulations
        self.play_cost = play_cost
        self.evaluator = Evaluator()
        self.prize_map = {
            "Jackpot": 20_000_000,  # Placeholder, actual value varies
            "Match 5": 1_000_000,
            "Match 4 + PB": 50_000,
            "Match 4": 100,
            "Match 3 + PB": 100,
            "Match 3": 7,
            "Match 2 + PB": 7,
            "Match 1 + PB": 4,
            "Match PB": 4,
            "Non-winning": 0
        }
        logger.info(f"MonteCarloSimulator initialized for {num_simulations} simulations.")

    def _generate_synthetic_draws(self):
        """Generates a large number of random synthetic draws."""
        logger.info(f"Generating {self.num_simulations} synthetic draws for simulation...")
        draws = []
        for _ in range(self.num_simulations):
            white_balls = np.random.choice(range(1, 70), 5, replace=False)
            pb = np.random.randint(1, 27)
            draws.append((set(white_balls), pb))
        return draws

    def run_simulation(self, plays_df):
        """
        Runs the simulation for a given DataFrame of plays.
        :param plays_df: DataFrame containing the plays to simulate.
        :return: DataFrame with added simulation metrics.
        """
        if plays_df.empty:
            logger.warning("Plays DataFrame is empty, skipping simulation.")
            return plays_df

        synthetic_draws = self._generate_synthetic_draws()
        simulation_results = []

        for index, play in plays_df.iterrows():
            total_winnings = 0
            hit_counts = {tier: 0 for tier in self.prize_map}
            
            play_numbers = {play['n1'], play['n2'], play['n3'], play['n4'], play['n5']}
            play_pb = play['pb']

            for winning_numbers, winning_pb in synthetic_draws:
                hits_white = len(play_numbers.intersection(winning_numbers))
                hits_powerball = 1 if play_pb == winning_pb else 0
                prize_tier = self.evaluator._get_prize_tier(hits_white, hits_powerball)
                
                total_winnings += self.prize_map[prize_tier]
                hit_counts[prize_tier] += 1

            total_cost = self.play_cost * self.num_simulations
            expected_roi = ((total_winnings - total_cost) / total_cost) * 100 if total_cost > 0 else 0
            hit_frequency = (1 - (hit_counts["Non-winning"] / self.num_simulations)) * 100 if self.num_simulations > 0 else 0

            simulation_results.append({
                'expected_roi_percent': round(expected_roi, 4),
                'hit_frequency_percent': round(hit_frequency, 4)
            })
        
        results_df = pd.DataFrame(simulation_results, index=plays_df.index)
        return plays_df.join(results_df)
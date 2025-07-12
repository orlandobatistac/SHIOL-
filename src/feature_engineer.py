import pandas as pd
import numpy as np
from loguru import logger

class FeatureEngineer:
    def __init__(self, historical_data):
        self.data = historical_data.copy()
        logger.info("FeatureEngineer initialized.")

    def engineer_features(self):
        """
        Engineers a rich set of features from the historical data.
        """
        logger.info("Starting feature engineering...")
        
        white_ball_cols = ['n1', 'n2', 'n3', 'n4', 'n5']
        
        # Parity
        self.data['even_count'] = self.data[white_ball_cols].apply(lambda row: sum(1 for x in row if x % 2 == 0), axis=1)
        self.data['odd_count'] = 5 - self.data['even_count']
        
        # Sum
        self.data['sum'] = self.data[white_ball_cols].sum(axis=1)
        
        # Spread
        self.data['spread'] = self.data[white_ball_cols].max(axis=1) - self.data[white_ball_cols].min(axis=1)
        
        # Consecutive Numbers
        def count_consecutive(row):
            numbers = sorted(row)
            consecutive_count = 0
            for i in range(len(numbers) - 1):
                if numbers[i+1] - numbers[i] == 1:
                    consecutive_count += 1
            return consecutive_count
        self.data['consecutive_count'] = self.data[white_ball_cols].apply(count_consecutive, axis=1)

        # Low/High Balance (assuming range 1-69)
        def low_high_balance(row):
            low_count = sum(1 for x in row if x <= 35)
            return f"{low_count}L-{5-low_count}H"
        self.data['low_high_balance'] = self.data[white_ball_cols].apply(low_high_balance, axis=1)

        # Remove placeholder frequency map
        # self.data.drop(columns=['frequency_map'], inplace=True, errors='ignore')

        self._calculate_recency()
        self._add_prize_tier()

        logger.info("Feature engineering complete.")
        return self.data

    def _add_prize_tier(self):
        """
        Classifies each historical draw into a tier based on its features.
        This creates the target variable for the ML model.
        """
        logger.info("Classifying historical draws into prize tiers...")
        
        conditions = [
            (self.data['sum'].between(120, 240)) & (self.data['even_count'].isin([2, 3])) & (self.data['consecutive_count'] <= 1),
            (self.data['sum'].between(100, 260)) & (self.data['even_count'].isin([1, 2, 3, 4]))
        ]
        choices = ['TopTier', 'MidTier']
        
        self.data['prize_tier'] = np.select(conditions, choices, default='LowTier')

    def _calculate_recency(self):
        """
        Calculates the delay for each number in a draw, defined as the number
        of draws since it last appeared.
        """
        logger.info("Calculating recency (delay) features...")
        
        # This calculation is only valid for historical data with dates
        if 'draw_date' not in self.data.columns:
            logger.warning("'draw_date' column not found. Skipping recency calculation.")
            return

        # Ensure data is sorted by date
        self.data.sort_values(by='draw_date', inplace=True)
        self.data.reset_index(drop=True, inplace=True)

        last_seen = {}
        delays = []

        for index, row in self.data.iterrows():
            current_delays = []
            numbers_in_draw = [row['n1'], row['n2'], row['n3'], row['n4'], row['n5']]
            for num in numbers_in_draw:
                delay = index - last_seen.get(num, index) # If not seen, delay is 0 relative to current
                current_delays.append(delay)
                last_seen[num] = index
            delays.append(current_delays)

        delay_df = pd.DataFrame(delays, columns=[f'delay_n{i+1}' for i in range(5)])
        
        self.data['avg_delay'] = delay_df.mean(axis=1)
        self.data['max_delay'] = delay_df.max(axis=1)
        self.data['min_delay'] = delay_df.min(axis=1)
import pandas as pd
import numpy as np
from loguru import logger
import configparser
from src.feature_engineer import FeatureEngineer
from src.simulation_engine import MonteCarloSimulator
from src.evolutionary_engine import GeneticAlgorithmOptimizer

class PlayGenerator:
    def __init__(self, model, label_encoder, historical_data):
        self.model = model
        self.label_encoder = label_encoder
        self.historical_data = historical_data
        self.config = configparser.ConfigParser()
        self.config.read('config/config.ini')
        logger.info("PlayGenerator initialized.")

    def generate_plays(self):
        """
        Generates and ranks lottery plays using a two-stage process.
        """
        logger.info("Starting play generation...")
        
        # 1. Candidate Generation
        pool_size = int(self.config['generation_params']['candidate_pool_size'])
        candidates = self._generate_candidates(pool_size)
        logger.info(f"Generated {len(candidates)} candidate plays.")

        # 2. Hybrid Filtering & Ranking
        ranked_plays = self._filter_and_rank(candidates)
        logger.info("Filtering and ranking complete.")

        # Separate into personal and syndicate plays
        personal_plays_count = int(self.config['generation_params']['personal_plays'])
        syndicate_plays_count = int(self.config['generation_params']['syndicate_plays'])
        
        personal_plays = ranked_plays.head(personal_plays_count)
        syndicate_plays = ranked_plays.head(syndicate_plays_count) # For now, just take the top N

        # 3. Evolve the syndicate plays for further optimization
        logger.info("Evolving syndicate plays using Genetic Algorithm...")
        evo_config = self.config['Evolutionary']
        optimizer = GeneticAlgorithmOptimizer(
            num_generations=int(evo_config['num_generations']),
            population_size=syndicate_plays_count,
            mutation_rate=float(evo_config['mutation_rate']),
            tournament_size=int(evo_config['tournament_size'])
        )
        evolved_syndicate_plays = optimizer.evolve(syndicate_plays)
        
        # The evolved plays need to be re-evaluated and re-ranked
        # For now, we will just use the evolved set and simulate them.
        logger.info("Re-evaluating and ranking evolved syndicate plays...")
        evolved_syndicate_plays_ranked = self._filter_and_rank(evolved_syndicate_plays)


        # 4. Run simulation on the top plays
        logger.info("Running Monte Carlo simulation on selected plays...")
        simulator = MonteCarloSimulator()
        
        # Combine plays to run simulation only once
        all_top_plays = pd.concat([personal_plays, syndicate_plays]).drop_duplicates().reset_index(drop=True)
        
        simulated_all_plays = simulator.run_simulation(all_top_plays)
        
        # Separate them back out
        simulated_personal_plays = simulated_all_plays.head(len(personal_plays))
        simulated_syndicate_plays = simulated_all_plays.head(len(syndicate_plays))

        logger.info("Simulation complete.")

        return simulated_personal_plays, simulated_syndicate_plays

    def _generate_candidates(self, num_candidates):
        """
        Generates a pool of candidate plays weighted by historical frequency.
        """
        logger.info("Calculating historical number frequencies for weighted generation...")
        white_ball_cols = ['n1', 'n2', 'n3', 'n4', 'n5']
        
        # Calculate frequencies
        all_numbers = pd.melt(self.historical_data[white_ball_cols])['value']
        frequencies = all_numbers.value_counts(normalize=True)
        
        # Prepare numbers and their corresponding probabilities for np.random.choice
        numbers = frequencies.index.values
        probabilities = frequencies.values

        # Generate candidates
        candidates = []
        for _ in range(num_candidates):
            white_balls = sorted(np.random.choice(numbers, 5, replace=False, p=probabilities))
            pb = np.random.randint(1, 27) # Powerball generation remains random for now
            candidates.append(white_balls + [pb])
        
        df = pd.DataFrame(candidates, columns=['n1', 'n2', 'n3', 'n4', 'n5', 'pb'])
        return df

    def _filter_and_rank(self, candidates_df):
        """
        Uses the ML model and heuristics to filter and rank candidates.
        """
        # Engineer features for the candidate plays
        feature_engineer = FeatureEngineer(candidates_df)
        candidates_with_features = feature_engineer.engineer_features()

        # Use the model to predict the "likeliness" score
        # Add placeholder recency features for candidate prediction
        # A more advanced implementation would calculate this based on the last historical draw
        if 'avg_delay' not in candidates_with_features.columns:
            logger.warning("Recency features not calculated for candidates. Using historical mean as placeholder.")
            # To avoid re-calculating, we can pass the featured historical data
            hist_feature_engineer = FeatureEngineer(self.historical_data)
            hist_featured_data = hist_feature_engineer.engineer_features()
            
            candidates_with_features['avg_delay'] = hist_featured_data['avg_delay'].mean()
            candidates_with_features['max_delay'] = hist_featured_data['max_delay'].mean()
            candidates_with_features['min_delay'] = hist_featured_data['min_delay'].mean()


        feature_cols = [
            'even_count', 'odd_count', 'sum', 'spread', 'consecutive_count',
            'avg_delay', 'max_delay', 'min_delay'
        ]
        
        # Ensure all feature columns are present
        for col in feature_cols:
            if col not in candidates_with_features.columns:
                raise ValueError(f"Missing required feature column for prediction: {col}")

        X_candidates = candidates_with_features[feature_cols]
        
        # Get the index for the 'TopTier' class from the label encoder
        try:
            top_tier_index = list(self.label_encoder.classes_).index('TopTier')
        except ValueError:
            logger.error("'TopTier' class not found in model's label encoder. Defaulting to first class.")
            top_tier_index = 0
            
        # predict_proba returns probabilities for each class
        # We want the probability of the 'TopTier' class
        scores = self.model.predict_proba(X_candidates)[:, top_tier_index]
        
        candidates_df['likeliness_score'] = scores

        # Apply final heuristic filters from config
        min_sum = int(self.config['heuristic_filters']['min_sum'])
        max_sum = int(self.config['heuristic_filters']['max_sum'])
        
        filtered_df = candidates_df[
            (candidates_with_features['sum'] >= min_sum) &
            (candidates_with_features['sum'] <= max_sum)
        ].copy()

        # Rank by the model's score
        ranked_df = filtered_df.sort_values(by='likeliness_score', ascending=False)
        
        return ranked_df
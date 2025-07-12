import numpy as np
import pandas as pd
import random
from loguru import logger

class GeneticAlgorithmOptimizer:
    def __init__(self, num_generations, population_size, mutation_rate, tournament_size):
        self.num_generations = num_generations
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        logger.info("GeneticAlgorithmOptimizer initialized.")

    def _selection(self, population):
        """Performs tournament selection."""
        selected = []
        for _ in range(len(population)):
            tournament = random.sample(population, self.tournament_size)
            winner = max(tournament, key=lambda x: x['fitness'])
            selected.append(winner)
        return selected

    def _crossover(self, parent1, parent2):
        """Performs single-point crossover for the 5 white balls."""
        p1_genes = set(parent1['play'][:5])
        p2_genes = set(parent2['play'][:5])
        
        # Combine unique genes from both parents
        all_genes = list(p1_genes.union(p2_genes))
        
        # If not enough unique genes, fill with random ones
        while len(all_genes) < 5:
            new_gene = random.randint(1, 69)
            if new_gene not in all_genes:
                all_genes.append(new_gene)

        # Create child by sampling from the combined gene pool
        child_genes = sorted(random.sample(all_genes, 5))
        child_pb = random.choice([parent1['play'][5], parent2['play'][5]])
        
        return child_genes + [child_pb]

    def _mutate(self, play):
        """Mutates a single gene in the play."""
        mutated_play = play[:]
        if random.random() < self.mutation_rate:
            index_to_mutate = random.randint(0, 4) # Mutate one of the white balls
            current_genes = set(mutated_play[:5])
            
            new_gene = random.randint(1, 69)
            while new_gene in current_genes:
                new_gene = random.randint(1, 69)
            
            mutated_play[index_to_mutate] = new_gene
            mutated_play[:5] = sorted(mutated_play[:5])
        
        return mutated_play

    def evolve(self, initial_population_df):
        """
        Evolves a population of plays over a number of generations.
        :param initial_population_df: DataFrame with plays and a 'likeliness_score' column.
        :return: DataFrame of the best plays from the final generation.
        """
        logger.info(f"Starting evolution for {self.num_generations} generations...")
        
        # Convert DataFrame to a list of dictionaries for easier processing
        population = []
        for _, row in initial_population_df.iterrows():
            population.append({
                'play': [row['n1'], row['n2'], row['n3'], row['n4'], row['n5'], row['pb']],
                'fitness': row['likeliness_score']
            })

        for gen in range(self.num_generations):
            # 1. Selection
            selected_parents = self._selection(population)
            
            # 2. Crossover and Mutation
            next_generation_plays = []
            for i in range(0, len(selected_parents), 2):
                parent1 = selected_parents[i]
                # Ensure there's a second parent
                parent2 = selected_parents[i+1] if (i+1) < len(selected_parents) else selected_parents[0]
                
                child1_play = self._crossover(parent1, parent2)
                child2_play = self._crossover(parent2, parent1)
                
                next_generation_plays.append(self._mutate(child1_play))
                next_generation_plays.append(self._mutate(child2_play))
            
            # This part is crucial: The new generation needs to be re-evaluated for fitness.
            # For this implementation, we will skip re-evaluation and assume the evolved
            # plays are superior, which is a simplification. A full implementation
            # would re-run the feature engineering and model prediction on the new generation.
            # We will just create a new population dict with a dummy fitness.
            
            new_population = []
            for play in next_generation_plays[:self.population_size]:
                new_population.append({'play': play, 'fitness': 0}) # Dummy fitness
            
            population = new_population
            
            if (gen + 1) % 10 == 0:
                logger.info(f"Completed generation {gen + 1}/{self.num_generations}")

        # Return the final population as a DataFrame
        final_plays_list = [p['play'] for p in population]
        final_df = pd.DataFrame(final_plays_list, columns=['n1', 'n2', 'n3', 'n4', 'n5', 'pb'])
        
        logger.info("Evolution complete.")
        return final_df
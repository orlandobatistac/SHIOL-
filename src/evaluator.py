import pandas as pd
from loguru import logger

class Evaluator:
    def __init__(self):
        logger.info("Evaluator initialized.")

    def evaluate_plays(self, plays_df, winning_numbers, winning_pb):
        """
        Evaluates a DataFrame of plays against a set of winning numbers.
        
        :param plays_df: DataFrame with columns ['n1', 'n2', 'n3', 'n4', 'n5', 'pb']
        :param winning_numbers: A set of the 5 winning white ball numbers.
        :param winning_pb: The winning Powerball number.
        :return: DataFrame with added evaluation columns.
        """
        logger.info("Evaluating plays...")
        
        if plays_df.empty:
            logger.warning("Plays DataFrame is empty, nothing to evaluate.")
            return plays_df

        eval_results = []
        for _, play in plays_df.iterrows():
            play_numbers = {play['n1'], play['n2'], play['n3'], play['n4'], play['n5']}
            
            hits_white = len(play_numbers.intersection(winning_numbers))
            hits_powerball = 1 if play['pb'] == winning_pb else 0
            
            prize_tier = self._get_prize_tier(hits_white, hits_powerball)
            
            eval_results.append({
                'hits_white': hits_white,
                'hits_powerball': hits_powerball,
                'prize_tier': prize_tier
            })
        
        results_df = pd.DataFrame(eval_results)
        
        # Merge results back into the original DataFrame
        plays_df_evaluated = plays_df.reset_index(drop=True).join(results_df)
        
        logger.info("Evaluation complete.")
        return plays_df_evaluated

    def _get_prize_tier(self, hits_white, hits_powerball):
        """
        Determines the prize tier based on the number of hits.
        (Based on Powerball prize rules)
        """
        if hits_white == 5 and hits_powerball == 1: return "Jackpot"
        if hits_white == 5 and hits_powerball == 0: return "Match 5"
        if hits_white == 4 and hits_powerball == 1: return "Match 4 + PB"
        if hits_white == 4 and hits_powerball == 0: return "Match 4"
        if hits_white == 3 and hits_powerball == 1: return "Match 3 + PB"
        if hits_white == 3 and hits_powerball == 0: return "Match 3"
        if hits_white == 2 and hits_powerball == 1: return "Match 2 + PB"
        if hits_white == 1 and hits_powerball == 1: return "Match 1 + PB"
        if hits_white == 0 and hits_powerball == 1: return "Match PB"
        return "Non-winning"
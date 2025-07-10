import pandas as pd
from sqlalchemy import create_engine, text
import configparser
from loguru import logger

def evaluate_results(draw_date_str: str, winning_numbers: list, winning_pb: int):
    """
    Compares generated plays with actual results and updates the database.
    
    :param draw_date_str: The date of the draw to evaluate (e.g., '2023-10-28')
    :param winning_numbers: A list of the 5 winning white ball numbers.
    :param winning_pb: The winning Powerball number.
    """
    config = configparser.ConfigParser()
    config.read('config/config.ini')

    db_file = config['paths']['db_file']
    engine = create_engine(f'sqlite:///{db_file}')

    logger.info(f"Evaluating results for draw date: {draw_date_str}")

    query = text("""
        SELECT id, n1, n2, n3, n4, n5, pb 
        FROM generated_plays 
        WHERE target_draw_date = :date AND is_evaluated = 0
    """)
    
    with engine.connect() as connection:
        plays_to_evaluate = pd.read_sql(query, connection, params={'date': draw_date_str})

        if plays_to_evaluate.empty:
            logger.warning(f"No unevaluated plays found for {draw_date_str}")
            return

        for index, play in plays_to_evaluate.iterrows():
            play_numbers = {play['n1'], play['n2'], play['n3'], play['n4'], play['n5']}
            
            hits_white = len(play_numbers.intersection(winning_numbers))
            hits_powerball = 1 if play['pb'] == winning_pb else 0
            
            # Placeholder for prize tier logic
            prize_tier = "Non-winning"
            if hits_white == 5 and hits_powerball == 1:
                prize_tier = "Jackpot"
            elif hits_white == 5 and hits_powerball == 0:
                prize_tier = "Match 5"
            # ... add other prize tiers

            update_query = text("""
                UPDATE generated_plays
                SET hits_white = :hits_w,
                    hits_powerball = :hits_pb,
                    prize_tier = :prize,
                    is_evaluated = 1
                WHERE id = :play_id
            """)
            
            connection.execute(update_query, {
                'hits_w': hits_white,
                'hits_pb': hits_powerball,
                'prize': prize_tier,
                'play_id': play['id']
            })
        
        # Commit the transaction
        connection.commit()


    logger.info("Results evaluation complete.")

if __name__ == '__main__':
    # This is an example of how to run the evaluator
    # In a real scenario, this would be triggered after a draw.
    logger.add("logs/shiolplus.log", rotation="10 MB")
    
    # Example usage:
    # evaluate_results('2023-10-28', [1, 2, 3, 4, 5], 6)
    print("This script should be called with a specific draw date and winning numbers.")
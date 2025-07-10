from loguru import logger
import argparse
from src.data_manager import update_db
from src.model_trainer import train_model
from src.play_generator import generate_plays
from src.results_evaluator import evaluate_results

def main():
    """
    Main orchestrator for the SHIOLPlus system.
    """
    logger.add("logs/shiolplus.log", rotation="10 MB", level="INFO")
    logger.info("SHIOLPlus v1.2 - Starting weekly run.")

    parser = argparse.ArgumentParser(description="SHIOLPlus Powerball Analysis Tool")
    parser.add_argument(
        "--evaluate", 
        nargs=2, 
        metavar=('DRAW_DATE', 'WINNING_NUMBERS'),
        help="Evaluate results for a given draw. WINNING_NUMBERS should be a comma-separated string, e.g., '1,2,3,4,5,6'"
    )
    parser.add_argument(
        "--full-run",
        action="store_true",
        help="Perform a full run: update DB, retrain model, and generate new plays."
    )

    args = parser.parse_args()

    if args.full_run:
        logger.info("--- Phase 1: Updating Database ---")
        update_db()
        
        logger.info("--- Phase 2: Training Model ---")
        train_model()
        
        logger.info("--- Phase 3: Generating Plays ---")
        generate_plays()

    elif args.evaluate:
        logger.info("--- Phase 4: Evaluating Results ---")
        draw_date = args.evaluate[0]
        try:
            numbers = [int(n) for n in args.evaluate[1].split(',')]
            if len(numbers) != 6:
                raise ValueError("Exactly 6 numbers are required (5 white + 1 PB).")
            winning_numbers = numbers[:5]
            winning_pb = numbers[5]
            evaluate_results(draw_date, winning_numbers, winning_pb)
        except ValueError as e:
            logger.error(f"Invalid format for winning numbers: {e}")
            print(f"Error: {e}")
            print("Please provide numbers as a comma-separated string, e.g., '1,2,3,4,5,6'")

    else:
        print("No action specified. Use --full-run to generate plays or --evaluate to check results.")
        parser.print_help()

    logger.info("SHIOLPlus run finished.")

if __name__ == "__main__":
    main()
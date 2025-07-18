import argparse
from loguru import logger
import sys
import os

# Add project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Configure logging
logger.add(sys.stderr, format="{time} {level} {message}", filter="my_module", level="INFO")
logger.add("logs/shiol_v2.log", rotation="10 MB", level="INFO")

def main():
    """Main entry point for the SHIOL+ v2.0 CLI."""
    parser = argparse.ArgumentParser(description="SHIOL+ v2.0 - AI Lottery Pattern Analysis")
    subparsers = parser.add_subparsers(dest='command', required=True)

    # --- Train Command ---
    parser_train = subparsers.add_parser('train', help="Train the prediction model on historical data.")
    parser_train.set_defaults(func=train_model_command)

    # --- Predict Command ---
    parser_predict = subparsers.add_parser('predict', help="Generate new plays based on the trained model.")
    parser_predict.add_argument('--count', type=int, help="Number of plays to generate.")
    parser_predict.set_defaults(func=predict_plays_command)

    # --- Backtest Command ---
    parser_backtest = subparsers.add_parser('backtest', help="Backtest a generated strategy against historical data.")
    parser_backtest.add_argument('--count', type=int, help="Number of plays to generate and test.")
    parser_backtest.set_defaults(func=backtest_strategy_command)

    # --- Update Command ---
    parser_update = subparsers.add_parser('update', help="Download the latest Powerball data.")
    parser_update.set_defaults(func=update_data_command)

    # Load defaults from config and then parse args
    defaults = load_cli_defaults()
    parser.set_defaults(**defaults)

    args = parser.parse_args()
    args.func(args)

def predict_plays_command(args):
    """Handles the 'predict' command."""
    logger.info(f"Received 'predict' command. Generating {args.count} plays...")
    try:
        from src.predictor import Predictor
        from src.intelligent_generator import IntelligentGenerator

        predictor = Predictor()
        wb_probs, pb_probs = predictor.predict_probabilities()

        generator = IntelligentGenerator()
        plays_df = generator.generate_plays(wb_probs, pb_probs, args.count)
        
        logger.info("Generated plays:\n" + plays_df.to_string())
        print("\n--- Generated Plays ---")
        print(plays_df.to_string(index=False))
        print("-----------------------\n")

    except Exception as e:
        logger.error(f"An error occurred during prediction: {e}")
        sys.exit(1)
    
def backtest_strategy_command(args):
    """Handles the 'backtest' command."""
    logger.info(f"Received 'backtest' command for {args.count} plays...")
    try:
        from src.predictor import Predictor
        from src.intelligent_generator import IntelligentGenerator
        from src.evaluator import Evaluator
        from src.loader import get_data_loader
        import json

        # 1. Generate plays
        logger.info("Generating plays for the backtest...")
        predictor = Predictor()
        wb_probs, pb_probs = predictor.predict_probabilities()
        generator = IntelligentGenerator()
        plays_df = generator.generate_plays(wb_probs, pb_probs, args.count)
        logger.info(f"{len(plays_df)} plays generated.")

        # 2. Load historical data
        logger.info("Loading historical data for backtesting...")
        data_loader = get_data_loader()
        historical_data = data_loader.load_historical_data()
        
        # 3. Run the backtest
        logger.info("Running backtest simulation...")
        evaluator = Evaluator()
        report = evaluator.run_backtest(plays_df, historical_data)

        # 4. Display the report
        logger.info("Backtest complete. Displaying report.")
        print("\n--- Backtest Report ---")
        print(json.dumps(report, indent=2))
        print("-----------------------\n")

    except Exception as e:
        logger.error(f"An error occurred during backtest: {e}")
        sys.exit(1)

def train_model_command(args):
    """Handles the 'train' command."""
    logger.info("Received 'train' command. Initializing model training process...")
    try:
        from src.predictor import retrain_existing_model
        
        # The config file path is now hardcoded in the function, which is fine for this specialized tool.
        retrain_existing_model()
        
        logger.info("Model training process completed successfully.")
    except Exception as e:
        logger.error(f"An error occurred during model training: {e}")
        sys.exit(1)

def load_cli_defaults():
    """Loads default values for CLI arguments from config.ini."""
    try:
        import configparser
        config = configparser.ConfigParser()
        # Correct path to config.ini from the project root
        config_path = os.path.join(project_root, 'config', 'config.ini')
        config.read(config_path)
        
        defaults = {}
        if 'cli_defaults' in config:
            for key, value in config['cli_defaults'].items():
                # Convert to integer if possible, otherwise use as string
                try:
                    defaults[key] = int(value)
                except ValueError:
                    defaults[key] = value
        return defaults
    except Exception as e:
        logger.warning(f"Could not load CLI defaults from config.ini: {e}")
        return {}

def update_data_command(args):
    """Handles the 'update' command."""
    logger.info("Received 'update' command. Downloading latest data...")
    try:
        from src.loader import update_powerball_data_from_csv
        
        rows_updated = update_powerball_data_from_csv()
        
        if rows_updated > 0:
            logger.info(f"Data update complete. Total rows in dataset: {rows_updated}")
            print(f"\nData update successful. The dataset now contains {rows_updated} rows.\n")
        else:
            logger.warning("Data update did not return any rows. The dataset might be empty or unchanged.")
            print("\nData update process finished, but no new data was returned.\n")

    except Exception as e:
        logger.error(f"An error occurred during data update: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
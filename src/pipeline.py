from loguru import logger
import configparser
import pandas as pd

from src.data_loader import get_data_loader
from src.feature_engineer import FeatureEngineer
from src.model_trainer import get_model_trainer
from src.play_generator import PlayGenerator
from src.persistence_manager import get_persistence_manager

def generate_new_plays():
    """
    Runs the full pipeline to generate new plays.
    """
    logger.info("--- Starting SHIOLPlus v1.3: Generate New Plays ---")
    
    # 1. Load historical data
    data_loader = get_data_loader()
    historical_data = data_loader.load_historical_data()
    if historical_data.empty:
        logger.error("Halting pipeline: No historical data loaded.")
        return

    # 2. Engineer features
    feature_engineer = FeatureEngineer(historical_data)
    featured_data = feature_engineer.engineer_features()

    # 3. Train model
    model_trainer = get_model_trainer()
    model = model_trainer.train(featured_data)
    if model is None:
        logger.error("Halting pipeline: Model training failed.")
        return

    # 4. Generate plays
    play_generator = PlayGenerator(model, model_trainer.label_encoder, historical_data)
    personal_plays, syndicate_plays = play_generator.generate_plays()
    logger.info(f"Generated {len(personal_plays)} personal plays and {len(syndicate_plays)} syndicate plays.")

    # 5. Save plays
    config = configparser.ConfigParser()
    config.read('config/config.ini')
    personal_file = config['paths']['personal_predictions_file']
    syndicate_file = config['paths']['syndicate_predictions_file']
    
    personal_plays.to_csv(personal_file, index=False)
    syndicate_plays.to_csv(syndicate_file, index=False)
    logger.info(f"Saved personal plays to {personal_file}")
    logger.info(f"Saved syndicate plays to {syndicate_file}")
    
    logger.info("--- SHIOLPlus v1.3: Play Generation Complete ---")


def update_with_new_results(file_path):
    """
    Ingests new real draw results, updates the main data file, and triggers a full regeneration.
    """
    logger.info(f"--- Starting SHIOLPlus v1.3: Update with New Results from {file_path} ---")
    
    config = configparser.ConfigParser()
    config.read('config/config.ini')
    main_data_file = config['paths']['data_file']

    try:
        # 1. Load existing and new data
        main_df = pd.read_csv(main_data_file)
        new_df = pd.read_csv(file_path)
        
        # Convert dates to a consistent format for comparison
        main_df['Date'] = pd.to_datetime(main_df['Date'], format='%m/%d/%Y')
        new_df['Date'] = pd.to_datetime(new_df['Date'], format='%m/%d/%Y')

        # 2. Identify and filter out duplicate draws
        existing_dates = set(main_df['Date'])
        unique_new_draws = new_df[~new_df['Date'].isin(existing_dates)]

        if unique_new_draws.empty:
            logger.info("No new, unique draw results found in the provided file. No update needed.")
            return

        logger.info(f"Found {len(unique_new_draws)} new draws to add.")

        # 3. Append new unique data to the main CSV
        unique_new_draws['Date'] = unique_new_draws['Date'].dt.strftime('%m/%d/%Y')
        unique_new_draws.to_csv(main_data_file, mode='a', header=False, index=False)
        logger.info(f"Successfully appended new data to {main_data_file}.")

        # 4. Trigger the full generation pipeline to retrain on new data
        logger.info("Triggering pipeline to retrain model and generate new plays...")
        generate_new_plays()

    except FileNotFoundError:
        logger.error(f"File not found at {file_path} or {main_data_file}. Please check paths.")
    except Exception as e:
        logger.error(f"An error occurred during the update process: {e}")

    logger.info("--- SHIOLPlus v1.3: Update Process Finished ---")
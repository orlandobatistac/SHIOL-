import argparse
from loguru import logger

from src.pipeline import generate_new_plays, update_with_new_results
from src.gui import launch_gui

def setup_logging():
    logger.add("logs/shiolplus.log", rotation="10 MB", level="INFO")

def main():
    setup_logging()
    parser = argparse.ArgumentParser(description="SHIOLPlus v1.3 - Lottery Analysis and Prediction System")
    subparsers = parser.add_subparsers(dest='command', required=True)

    # 'generate' command
    parser_generate = subparsers.add_parser('generate', help="Run the full pipeline to generate new plays.")
    parser_generate.set_defaults(func=lambda args: generate_new_plays())

    # 'update' command
    parser_update = subparsers.add_parser('update', help="Ingest new real draw results and retrain the model.")
    parser_update.add_argument('--file', required=True, help="Path to the CSV file with new draw results.")
    parser_update.set_defaults(func=lambda args: update_with_new_results(args.file))

    # 'gui' command
    parser_gui = subparsers.add_parser('gui', help="Launch the graphical user interface.")
    parser_gui.set_defaults(func=lambda args: launch_gui())

    args = parser.parse_args()
    
    # Call the function associated with the chosen command
    args.func(args)

if __name__ == "__main__":
    main()
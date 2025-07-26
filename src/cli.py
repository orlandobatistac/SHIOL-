import argparse
from loguru import logger
import sys
import os
import json
import numpy as np

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
    # default value set to avoid NoneType error
    parser_predict.add_argument('--count', type=int, default=5, help="Number of plays to generate.")
    parser_predict.set_defaults(func=predict_plays_command)

    # --- Predict Deterministic Command ---
    parser_predict_det = subparsers.add_parser('predict-deterministic',
                                              help="Generate deterministic prediction using multi-criteria scoring.")
    parser_predict_det.add_argument('--save-log', action='store_true', default=True,
                                   help="Save prediction to database log (default: True)")
    parser_predict_det.set_defaults(func=predict_deterministic_command)

    # --- Compare Methods Command ---
    parser_compare = subparsers.add_parser('compare-methods',
                                          help="Compare traditional vs deterministic prediction methods.")
    parser_compare.set_defaults(func=compare_methods_command)

    # --- Validate Command ---
    parser_validate = subparsers.add_parser('validate', help="Validate predictions against actual draw results.")
    parser_validate.set_defaults(func=validate_predictions_command)

    # --- Phase 4: Adaptive Feedback System Commands ---
    
    # --- Analyze Feedback Command ---
    parser_analyze_feedback = subparsers.add_parser('analyze-feedback',
                                                   help="Analyze adaptive feedback system performance and patterns.")
    parser_analyze_feedback.add_argument('--days', type=int, default=30,
                                       help="Number of days to analyze (default: 30)")
    parser_analyze_feedback.add_argument('--detailed', action='store_true',
                                       help="Show detailed analysis including patterns and weights")
    parser_analyze_feedback.set_defaults(func=analyze_feedback_command)

    # --- Rank Reliable Plays Command ---
    parser_rank_plays = subparsers.add_parser('rank-reliable-plays',
                                            help="Rank and display most reliable play combinations.")
    parser_rank_plays.add_argument('--limit', type=int, default=10,
                                 help="Number of top plays to show (default: 10)")
    parser_rank_plays.add_argument('--min-score', type=float, default=0.7,
                                 help="Minimum reliability score (default: 0.7)")
    parser_rank_plays.add_argument('--export', type=str,
                                 help="Export results to CSV file")
    parser_rank_plays.set_defaults(func=rank_reliable_plays_command)

    # --- Adaptive Validate Command ---
    parser_adaptive_validate = subparsers.add_parser('adaptive-validate',
                                                   help="Run adaptive validation with learning feedback.")
    parser_adaptive_validate.add_argument('--no-learning', action='store_true',
                                        help="Disable adaptive learning from results")
    parser_adaptive_validate.set_defaults(func=adaptive_validate_command)

    # --- Optimize Weights Command ---
    parser_optimize = subparsers.add_parser('optimize-weights',
                                          help="Optimize scoring component weights using performance data.")
    parser_optimize.add_argument('--algorithm', type=str, default='differential_evolution',
                               choices=['differential_evolution', 'scipy_minimize', 'grid_search'],
                               help="Optimization algorithm to use")
    parser_optimize.add_argument('--days', type=int, default=30,
                               help="Days of performance data to use")
    parser_optimize.set_defaults(func=optimize_weights_command)

    # --- Backtest Command ---
    parser_backtest = subparsers.add_parser('backtest', help="Backtest a generated strategy against historical data.")
    # default value set to avoid NoneType error
    parser_backtest.add_argument('--count', type=int, default=100, help="Number of plays to generate and test.")
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
    # Defensive check for count to ensure it's a valid integer.
    count = args.count if args.count is not None else 5
    logger.info(f"Received 'predict' command. Generating {count} plays...")
    try:
        from src.predictor import Predictor
        from src.intelligent_generator import IntelligentGenerator

        predictor = Predictor()
        wb_probs, pb_probs = predictor.predict_probabilities()

        generator = IntelligentGenerator()
        plays_df = generator.generate_plays(wb_probs, pb_probs, count)
        
        logger.info("Generated plays:\n" + plays_df.to_string())
        print("\n--- Generated Plays ---")
        print(plays_df.to_string(index=False))
        print("-----------------------\n")

    except Exception as e:
        logger.error(f"An error occurred during prediction: {e}")
        sys.exit(1)
    
def backtest_strategy_command(args):
    """Handles the 'backtest' command."""
    # Defensive check for count to ensure it's a valid integer.
    count = args.count if args.count is not None else 100
    logger.info(f"Received 'backtest' command for {count} plays...")
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
        plays_df = generator.generate_plays(wb_probs, pb_probs, count)
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
    logger.info("Received 'update' command. Updating database from source...")
    try:
        from src.loader import update_database_from_source
        
        total_rows = update_database_from_source()
        
        if total_rows > 0:
            logger.info(f"Database update complete. Total rows in database: {total_rows}")
            print(f"\nDatabase update successful. The database now contains {total_rows} rows.\n")
        else:
            logger.warning("Database update did not add new rows. The database might be empty or already up-to-date.")
            print("\nDatabase update process finished, but no new data was added.\n")

    except Exception as e:
        logger.error(f"An error occurred during data update: {e}")
        sys.exit(1)

def predict_deterministic_command(args):
    """Handles the 'predict-deterministic' command."""
    save_log = args.save_log if hasattr(args, 'save_log') else True
    logger.info(f"Received 'predict-deterministic' command. Save to log: {save_log}")
    try:
        from src.predictor import Predictor
        from src.database import get_prediction_history
        import json

        predictor = Predictor()
        prediction = predictor.predict_deterministic(save_to_log=save_log)
        
        # Mostrar la predicción
        print("\n--- Deterministic Prediction ---")
        print(f"Numbers: {prediction['numbers']}")
        print(f"Powerball: {prediction['powerball']}")
        print(f"Total Score: {prediction['score_total']:.4f}")
        print(f"Model Version: {prediction['model_version']}")
        print(f"Dataset Hash: {prediction['dataset_hash']}")
        
        # Mostrar detalles de scoring
        print("\n--- Scoring Details ---")
        score_details = prediction['score_details']
        print(f"Probability Score (40%): {score_details['probability']:.4f}")
        print(f"Diversity Score (25%): {score_details['diversity']:.4f}")
        print(f"Historical Score (20%): {score_details['historical']:.4f}")
        print(f"Risk Adjusted Score (15%): {score_details['risk_adjusted']:.4f}")
        
        if 'log_id' in prediction:
            print(f"\nPrediction saved to database with ID: {prediction['log_id']}")
        
        print("--------------------------------\n")
        
        logger.info("Deterministic prediction completed successfully.")

    except Exception as e:
        logger.error(f"An error occurred during deterministic prediction: {e}")
        sys.exit(1)

def compare_methods_command(args):
    """Handles the 'compare-methods' command."""
    logger.info("Received 'compare-methods' command. Comparing prediction methods...")
    try:
        from src.predictor import Predictor
        from src.database import NumpyEncoder

        predictor = Predictor()
        comparison = predictor.compare_prediction_methods()
        
        print("\n--- Method Comparison ---")
        print(json.dumps(comparison, indent=2, cls=NumpyEncoder))
        print("-------------------------\n")
        
        # Mostrar resumen comparativo
        traditional = comparison['traditional_method']
        deterministic = comparison['deterministic_method']
        
        print("--- Summary ---")
        print(f"Traditional Method:")
        print(f"  Numbers: {[int(x) for x in traditional['numbers']]}")
        print(f"  Powerball: {int(traditional['powerball'])}")
        print(f"  Reproducible: {traditional['reproducible']}")
        
        print(f"\nDeterministic Method:")
        print(f"  Numbers: {[int(x) for x in deterministic['numbers']]}")
        print(f"  Powerball: {int(deterministic['powerball'])}")
        print(f"  Total Score: {float(deterministic['total_score']):.4f}")
        print(f"  Reproducible: {deterministic['reproducible']}")
        print(f"  Dataset Hash: {deterministic['dataset_hash']}")
        print("---------------\n")
        
        logger.info("Method comparison completed successfully.")

    except Exception as e:
        logger.error(f"An error occurred during method comparison: {e}")
        sys.exit(1)

def validate_predictions_command(args):
    """Handles the 'validate' command."""
    logger.info("Received 'validate' command. Starting prediction validation...")
    try:
        from src.basic_validator import basic_validate_predictions
        
        csv_path = basic_validate_predictions()
        
        if csv_path:
            logger.info(f"Validation completed successfully. Results saved to: {csv_path}")
            print(f"\n--- Validation Complete ---")
            print(f"Results saved to: {csv_path}")
            
            # Show a summary of the results
            try:
                import pandas as pd
                df = pd.read_csv(csv_path)
                total_predictions = len(df)
                winners = len(df[df['prize_category'] != 'Non-winning'])
                
                print(f"Total predictions validated: {total_predictions}")
                print(f"Winning predictions: {winners}")
                print(f"Win rate: {winners/total_predictions*100:.1f}%" if total_predictions > 0 else "Win rate: 0%")
                
                if winners > 0:
                    print("\nWinning predictions breakdown:")
                    prize_counts = df[df['prize_category'] != 'Non-winning']['prize_category'].value_counts()
                    for prize, count in prize_counts.items():
                        print(f"  {prize}: {count}")
                        
            except Exception as e:
                logger.warning(f"Could not generate summary: {e}")
                
            print("---------------------------\n")
        else:
            print("\nValidation completed but no results were generated.\n")
            
    except Exception as e:
        logger.error(f"An error occurred during validation: {e}")
        sys.exit(1)


def analyze_feedback_command(args):
    """Handles the 'analyze-feedback' command."""
    days = args.days if hasattr(args, 'days') else 30
    detailed = args.detailed if hasattr(args, 'detailed') else False
    
    logger.info(f"Received 'analyze-feedback' command for {days} days (detailed: {detailed})")
    
    try:
        from src.adaptive_feedback import run_adaptive_analysis
        import json
        
        # Run adaptive analysis
        analysis_results = run_adaptive_analysis(days_back=days)
        
        if 'error' in analysis_results:
            print(f"\nError in adaptive analysis: {analysis_results['error']}\n")
            return
        
        print(f"\n--- Adaptive Feedback Analysis ({days} days) ---")
        print(f"Analysis Period: {analysis_results['analysis_period']}")
        print(f"Timestamp: {analysis_results['timestamp']}")
        
        # Performance Summary
        performance = analysis_results.get('performance_summary', {})
        print(f"\n--- Performance Summary ---")
        print(f"Total Predictions: {performance.get('total_predictions', 0)}")
        print(f"Average Accuracy: {performance.get('avg_accuracy', 0.0):.3f}")
        print(f"Average Main Matches: {performance.get('avg_main_matches', 0.0):.2f}")
        print(f"Average PB Matches: {performance.get('avg_pb_matches', 0.0):.2f}")
        print(f"Winning Predictions: {performance.get('winning_predictions', 0)}")
        print(f"Win Rate: {performance.get('win_rate', 0.0):.2f}%")
        
        # System Status
        system_status = analysis_results.get('system_status', {})
        print(f"\n--- System Status ---")
        print(f"Adaptive Learning Active: {'Yes' if system_status.get('adaptive_learning_active', False) else 'No'}")
        print(f"Learning Threshold Met: {'Yes' if system_status.get('learning_threshold_met', False) else 'No'}")
        print(f"Reliable Plays Count: {analysis_results.get('reliable_plays_count', 0)}")
        
        # Current Weights
        current_weights = analysis_results.get('current_adaptive_weights')
        if current_weights:
            print(f"\n--- Current Adaptive Weights ---")
            weights = current_weights.get('weights', {})
            print(f"Probability: {weights.get('probability', 0.4):.3f}")
            print(f"Diversity: {weights.get('diversity', 0.25):.3f}")
            print(f"Historical: {weights.get('historical', 0.2):.3f}")
            print(f"Risk Adjusted: {weights.get('risk_adjusted', 0.15):.3f}")
            print(f"Performance Score: {current_weights.get('performance_score', 0.0):.3f}")
        else:
            print(f"\n--- Weights Status ---")
            print("No adaptive weights configured - using default weights")
        
        # Detailed analysis
        if detailed:
            print(f"\n--- Detailed Analysis ---")
            
            # Top reliable plays
            top_plays = analysis_results.get('top_reliable_plays', [])
            if top_plays:
                print(f"\nTop Reliable Plays:")
                for i, play in enumerate(top_plays, 1):
                    numbers = [play['n1'], play['n2'], play['n3'], play['n4'], play['n5']]
                    print(f"  {i}. {numbers} + {play['pb']} (Score: {play['reliability_score']:.3f})")
            
            # Prize distribution
            prize_dist = performance.get('prize_distribution', {})
            if prize_dist:
                print(f"\nPrize Distribution:")
                for prize, count in prize_dist.items():
                    print(f"  {prize}: {count}")
        
        print("----------------------------------------\n")
        
        logger.info("Adaptive feedback analysis completed successfully")
        
    except Exception as e:
        logger.error(f"An error occurred during adaptive feedback analysis: {e}")
        sys.exit(1)


def rank_reliable_plays_command(args):
    """Handles the 'rank-reliable-plays' command."""
    limit = args.limit if hasattr(args, 'limit') else 10
    min_score = args.min_score if hasattr(args, 'min_score') else 0.7
    export_path = args.export if hasattr(args, 'export') else None
    
    logger.info(f"Received 'rank-reliable-plays' command (limit: {limit}, min_score: {min_score})")
    
    try:
        import src.database as db
        import pandas as pd
        
        # Get reliable plays
        reliable_plays = db.get_reliable_plays(limit=limit, min_reliability_score=min_score)
        
        if reliable_plays.empty:
            print(f"\nNo reliable plays found with minimum score {min_score}\n")
            return
        
        print(f"\n--- Top {len(reliable_plays)} Reliable Plays (Min Score: {min_score}) ---")
        print(f"{'Rank':<4} {'Numbers':<20} {'PB':<3} {'Score':<6} {'Win Rate':<8} {'Times Gen':<9} {'Last Generated'}")
        print("-" * 80)
        
        for i, (_, play) in enumerate(reliable_plays.iterrows(), 1):
            numbers = f"{play['n1']}-{play['n2']}-{play['n3']}-{play['n4']}-{play['n5']}"
            print(f"{i:<4} {numbers:<20} {play['pb']:<3} {play['reliability_score']:<6.3f} "
                  f"{play['win_rate']:<8.3f} {play['times_generated']:<9} {play['last_generated']}")
        
        print("-" * 80)
        print(f"Total plays found: {len(reliable_plays)}")
        
        # Export to CSV if requested
        if export_path:
            try:
                reliable_plays.to_csv(export_path, index=False)
                print(f"Results exported to: {export_path}")
            except Exception as e:
                logger.error(f"Error exporting to CSV: {e}")
                print(f"Error exporting to CSV: {e}")
        
        print()
        
        logger.info("Reliable plays ranking completed successfully")
        
    except Exception as e:
        logger.error(f"An error occurred during reliable plays ranking: {e}")
        sys.exit(1)


def adaptive_validate_command(args):
    """Handles the 'adaptive-validate' command."""
    enable_learning = not (args.no_learning if hasattr(args, 'no_learning') else False)
    
    logger.info(f"Received 'adaptive-validate' command (learning: {enable_learning})")
    
    try:
        from src.adaptive_feedback import initialize_adaptive_system
        from src.loader import DataLoader
        
        # Initialize adaptive system
        data_loader = DataLoader()
        historical_data = data_loader.load_historical_data()
        adaptive_system = initialize_adaptive_system(historical_data)
        
        # Run adaptive validation
        adaptive_validator = adaptive_system['adaptive_validator']
        csv_path = adaptive_validator.adaptive_validate_predictions(enable_learning=enable_learning)
        
        if not csv_path:
            print("\nAdaptive validation failed - no results generated\n")
            return
        
        print(f"\n--- Adaptive Validation Complete ---")
        print(f"Results saved to: {csv_path}")
        print(f"Learning enabled: {'Yes' if enable_learning else 'No'}")
        
        # Show summary
        try:
            import pandas as pd
            validation_df = pd.read_csv(csv_path)
            total_predictions = len(validation_df)
            winners = len(validation_df[validation_df['prize_category'] != 'Non-winning'])
            win_rate = (winners / total_predictions * 100) if total_predictions > 0 else 0.0
            
            print(f"\nValidation Summary:")
            print(f"Total predictions validated: {total_predictions}")
            print(f"Winning predictions: {winners}")
            print(f"Win rate: {win_rate:.1f}%")
            
            if winners > 0:
                print(f"\nWinning predictions breakdown:")
                prize_counts = validation_df[validation_df['prize_category'] != 'Non-winning']['prize_category'].value_counts()
                for prize, count in prize_counts.items():
                    print(f"  {prize}: {count}")
                    
        except Exception as e:
            logger.warning(f"Could not generate validation summary: {e}")
        
        if enable_learning:
            print(f"\nAdaptive learning feedback has been processed.")
            print(f"The system has learned from these validation results.")
        
        print("----------------------------------\n")
        
        logger.info("Adaptive validation completed successfully")
        
    except Exception as e:
        logger.error(f"An error occurred during adaptive validation: {e}")
        sys.exit(1)


def optimize_weights_command(args):
    """Handles the 'optimize-weights' command."""
    algorithm = args.algorithm if hasattr(args, 'algorithm') else 'differential_evolution'
    days = args.days if hasattr(args, 'days') else 30
    
    logger.info(f"Received 'optimize-weights' command (algorithm: {algorithm}, days: {days})")
    
    try:
        from src.adaptive_feedback import initialize_adaptive_system
        from src.loader import DataLoader
        import src.database as db
        
        # Get performance data
        performance_data = db.get_performance_analytics(days)
        
        if performance_data.get('total_predictions', 0) < 10:
            print(f"\nInsufficient data for optimization.")
            print(f"Found {performance_data.get('total_predictions', 0)} predictions, need at least 10.")
            print("Generate more predictions and try again.\n")
            return
        
        # Initialize adaptive system
        data_loader = DataLoader()
        historical_data = data_loader.load_historical_data()
        adaptive_system = initialize_adaptive_system(historical_data)
        
        # Get current weights
        current_weights = db.get_active_adaptive_weights()
        if not current_weights:
            current_weights = {
                'weights': {'probability': 0.4, 'diversity': 0.25, 'historical': 0.2, 'risk_adjusted': 0.15}
            }
        
        print(f"\n--- Weight Optimization ---")
        print(f"Algorithm: {algorithm}")
        print(f"Performance data: {days} days ({performance_data.get('total_predictions', 0)} predictions)")
        print(f"Current accuracy: {performance_data.get('avg_accuracy', 0.0):.3f}")
        print(f"Current win rate: {performance_data.get('win_rate', 0.0):.2f}%")
        
        print(f"\nCurrent weights:")
        for component, weight in current_weights['weights'].items():
            print(f"  {component}: {weight:.3f}")
        
        # Perform optimization
        weight_optimizer = adaptive_system['weight_optimizer']
        optimized_weights = weight_optimizer.optimize_weights(
            current_weights['weights'],
            performance_data,
            algorithm
        )
        
        if not optimized_weights:
            print(f"\nWeight optimization failed using {algorithm}")
            print("Try a different algorithm or ensure sufficient performance data.\n")
            return
        
        print(f"\nOptimized weights:")
        for component, weight in optimized_weights.items():
            print(f"  {component}: {weight:.3f}")
        
        # Calculate improvement
        print(f"\nWeight changes:")
        for component in optimized_weights:
            old_weight = current_weights['weights'][component]
            new_weight = optimized_weights[component]
            change = new_weight - old_weight
            print(f"  {component}: {old_weight:.3f} -> {new_weight:.3f} ({change:+.3f})")
        
        # Save optimized weights
        from datetime import datetime
        weight_set_name = f"cli_optimized_{algorithm}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        performance_score = performance_data.get('avg_accuracy', 0.0)
        
        weights_id = db.save_adaptive_weights(
            weight_set_name=weight_set_name,
            weights=optimized_weights,
            performance_score=performance_score,
            optimization_algorithm=algorithm,
            dataset_hash="cli_optimization",
            is_active=True
        )
        
        print(f"\nOptimized weights saved with ID: {weights_id}")
        print(f"Weight set name: {weight_set_name}")
        print(f"These weights are now active and will be used for future predictions.")
        print("----------------------------\n")
        
        logger.info("Weight optimization completed successfully")
        
    except Exception as e:
        logger.error(f"An error occurred during weight optimization: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
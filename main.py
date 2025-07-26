#!/usr/bin/env python3
"""
SHIOL+ Phase 5 Pipeline Orchestrator
====================================

Main pipeline orchestrator that coordinates all 7 pipeline steps:
1. Data Update
2. Adaptive Analysis  
3. Weight Optimization
4. Prediction Generation
5. Historical Validation
6. Performance Analysis
7. Notifications & Reports

Usage:
    python main.py                    # Run full pipeline
    python main.py --step data        # Run specific step
    python main.py --status           # Check pipeline status
    python main.py --help             # Show help
"""

import argparse
import configparser
import os
import sys
import traceback
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path

# Configure logging before importing other modules
from loguru import logger
import pandas as pd

# Import SHIOL+ modules
from src.loader import update_database_from_source, get_data_loader
from src.adaptive_feedback import (
    run_adaptive_analysis, 
    initialize_adaptive_system,
    WeightOptimizer,
    AdaptiveValidator
)
from src.intelligent_generator import DeterministicGenerator, FeatureEngineer
from src.database import (
    initialize_database,
    get_performance_analytics,
    save_prediction_log,
    get_all_draws
)
from src.predictor import Predictor


class PipelineOrchestrator:
    """
    Main pipeline orchestrator that coordinates all SHIOL+ Phase 5 pipeline steps.
    Handles execution, error recovery, logging, and status tracking.
    """
    
    def __init__(self, config_path: str = "config/config.ini"):
        """
        Initialize the pipeline orchestrator.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config = self._load_configuration()
        self.pipeline_status = {}
        self.execution_start_time = None
        self.historical_data = None
        self.adaptive_system = None
        
        # Setup logging
        self._setup_logging()
        
        # Initialize database
        self._initialize_database()
        
        logger.info("SHIOL+ Phase 5 Pipeline Orchestrator initialized")
    
    def _load_configuration(self) -> configparser.ConfigParser:
        """Load configuration from config.ini file."""
        try:
            config = configparser.ConfigParser()
            if not os.path.exists(self.config_path):
                raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
            
            config.read(self.config_path)
            logger.info(f"Configuration loaded from {self.config_path}")
            return config
            
        except Exception as e:
            print(f"ERROR: Failed to load configuration: {e}")
            sys.exit(1)
    
    def _setup_logging(self):
        """Setup logging configuration."""
        try:
            # Remove default logger
            logger.remove()
            
            # Get log file path from config
            log_file = self.config.get("paths", "log_file", fallback="logs/shiolplus.log")
            
            # Ensure log directory exists
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            
            # Add console logger
            logger.add(
                sys.stdout,
                format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
                level="INFO"
            )
            
            # Add file logger
            logger.add(
                log_file,
                format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
                level="DEBUG",
                rotation="10 MB",
                retention="30 days"
            )
            
            logger.info("Logging system initialized")
            
        except Exception as e:
            print(f"ERROR: Failed to setup logging: {e}")
            sys.exit(1)
    
    def _initialize_database(self):
        """Initialize database and ensure all tables exist."""
        try:
            initialize_database()
            logger.info("Database initialization completed")
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise
    
    def run_full_pipeline(self) -> Dict[str, Any]:
        """
        Execute the complete SHIOL+ Phase 5 pipeline.
        
        Returns:
            Dict with pipeline execution results and status
        """
        logger.info("=" * 60)
        logger.info("STARTING SHIOL+ PHASE 5 FULL PIPELINE EXECUTION")
        logger.info("=" * 60)
        
        self.execution_start_time = datetime.now()
        pipeline_results = {}
        
        try:
            # Step 1: Data Update
            logger.info("STEP 1/7: Data Update")
            pipeline_results['data_update'] = self._execute_step('data_update', self.step_data_update)
            
            # Step 2: Adaptive Analysis
            logger.info("STEP 2/7: Adaptive Analysis")
            pipeline_results['adaptive_analysis'] = self._execute_step('adaptive_analysis', self.step_adaptive_analysis)
            
            # Step 3: Weight Optimization
            logger.info("STEP 3/7: Weight Optimization")
            pipeline_results['weight_optimization'] = self._execute_step('weight_optimization', self.step_weight_optimization)
            
            # Step 4: Prediction Generation
            logger.info("STEP 4/7: Prediction Generation")
            pipeline_results['prediction_generation'] = self._execute_step('prediction_generation', self.step_prediction_generation)
            
            # Step 5: Historical Validation
            logger.info("STEP 5/7: Historical Validation")
            pipeline_results['historical_validation'] = self._execute_step('historical_validation', self.step_historical_validation)
            
            # Step 6: Performance Analysis
            logger.info("STEP 6/7: Performance Analysis")
            pipeline_results['performance_analysis'] = self._execute_step('performance_analysis', self.step_performance_analysis)
            
            # Step 7: Notifications & Reports
            logger.info("STEP 7/7: Notifications & Reports")
            pipeline_results['notifications_reports'] = self._execute_step('notifications_reports', self.step_notifications_reports)
            
            # Calculate execution time
            execution_time = datetime.now() - self.execution_start_time
            
            # Generate final summary
            pipeline_summary = self._generate_pipeline_summary(pipeline_results, execution_time)
            
            logger.info("=" * 60)
            logger.info("SHIOL+ PHASE 5 PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
            logger.info(f"Total execution time: {execution_time}")
            logger.info("=" * 60)
            
            return {
                'status': 'success',
                'execution_time': str(execution_time),
                'results': pipeline_results,
                'summary': pipeline_summary
            }
            
        except Exception as e:
            execution_time = datetime.now() - self.execution_start_time if self.execution_start_time else None
            error_msg = f"Pipeline execution failed: {str(e)}"
            logger.error(error_msg)
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            return {
                'status': 'failed',
                'error': error_msg,
                'execution_time': str(execution_time) if execution_time else None,
                'results': pipeline_results,
                'traceback': traceback.format_exc()
            }
    
    def _execute_step(self, step_name: str, step_function) -> Dict[str, Any]:
        """
        Execute a single pipeline step with error handling.
        
        Args:
            step_name: Name of the step
            step_function: Function to execute
            
        Returns:
            Dict with step execution results
        """
        step_start_time = datetime.now()
        
        try:
            logger.info(f"Executing {step_name}...")
            result = step_function()
            
            execution_time = datetime.now() - step_start_time
            
            self.pipeline_status[step_name] = {
                'status': 'success',
                'execution_time': str(execution_time),
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"✓ {step_name} completed successfully in {execution_time}")
            
            return {
                'status': 'success',
                'execution_time': str(execution_time),
                'result': result,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            execution_time = datetime.now() - step_start_time
            error_msg = f"Step {step_name} failed: {str(e)}"
            
            self.pipeline_status[step_name] = {
                'status': 'failed',
                'error': error_msg,
                'execution_time': str(execution_time),
                'timestamp': datetime.now().isoformat()
            }
            
            logger.error(f"✗ {error_msg}")
            logger.error(f"Step traceback: {traceback.format_exc()}")
            
            return {
                'status': 'failed',
                'error': error_msg,
                'execution_time': str(execution_time),
                'timestamp': datetime.now().isoformat(),
                'traceback': traceback.format_exc()
            }
    
    def step_data_update(self) -> Dict[str, Any]:
        """
        Step 1: Data Update - Update database from source.
        
        Returns:
            Dict with data update results
        """
        try:
            # Update database from source
            total_rows = update_database_from_source()
            
            # Load updated historical data
            data_loader = get_data_loader()
            self.historical_data = data_loader.load_historical_data()
            
            result = {
                'total_rows_in_database': total_rows,
                'historical_data_loaded': len(self.historical_data),
                'latest_draw_date': self.historical_data['draw_date'].max().strftime('%Y-%m-%d') if not self.historical_data.empty else None
            }
            
            logger.info(f"Data update completed: {total_rows} total rows, {len(self.historical_data)} historical records")
            return result
            
        except Exception as e:
            logger.error(f"Data update step failed: {e}")
            raise
    
    def step_adaptive_analysis(self) -> Dict[str, Any]:
        """
        Step 2: Adaptive Analysis - Run adaptive analysis on recent data.
        
        Returns:
            Dict with adaptive analysis results
        """
        try:
            # Ensure we have historical data
            if self.historical_data is None or self.historical_data.empty:
                self.historical_data = get_all_draws()
            
            # Initialize adaptive system if not already done
            if self.adaptive_system is None:
                self.adaptive_system = initialize_adaptive_system(self.historical_data)
            
            # Run adaptive analysis
            analysis_results = run_adaptive_analysis(days_back=30)
            
            logger.info(f"Adaptive analysis completed: {analysis_results.get('total_predictions_analyzed', 0)} predictions analyzed")
            return analysis_results
            
        except Exception as e:
            logger.error(f"Adaptive analysis step failed: {e}")
            raise
    
    def step_weight_optimization(self) -> Dict[str, Any]:
        """
        Step 3: Weight Optimization - Optimize scoring weights based on performance.
        
        Returns:
            Dict with weight optimization results
        """
        try:
            # Ensure adaptive system is initialized
            if self.adaptive_system is None:
                if self.historical_data is None:
                    self.historical_data = get_all_draws()
                self.adaptive_system = initialize_adaptive_system(self.historical_data)
            
            # Get performance data for optimization
            performance_data = get_performance_analytics(30)
            
            # Check if we have enough data for optimization
            if performance_data.get('total_predictions', 0) < 10:
                logger.warning("Insufficient data for weight optimization (need at least 10 predictions)")
                return {
                    'status': 'skipped',
                    'reason': 'insufficient_data',
                    'predictions_available': performance_data.get('total_predictions', 0),
                    'minimum_required': 10
                }
            
            # Get weight optimizer
            weight_optimizer = self.adaptive_system['weight_optimizer']
            
            # Get current weights (default if none exist)
            current_weights = {
                'probability': 0.40,
                'diversity': 0.25,
                'historical': 0.20,
                'risk_adjusted': 0.15
            }
            
            # Optimize weights
            optimized_weights = weight_optimizer.optimize_weights(
                current_weights=current_weights,
                performance_data=performance_data,
                algorithm='differential_evolution'
            )
            
            result = {
                'optimization_performed': optimized_weights is not None,
                'current_weights': current_weights,
                'optimized_weights': optimized_weights,
                'performance_data_used': performance_data,
                'algorithm_used': 'differential_evolution'
            }
            
            if optimized_weights:
                logger.info(f"Weight optimization completed: {optimized_weights}")
            else:
                logger.warning("Weight optimization failed to find better weights")
            
            return result
            
        except Exception as e:
            logger.error(f"Weight optimization step failed: {e}")
            raise
    
    def step_prediction_generation(self) -> Dict[str, Any]:
        """
        Step 4: Prediction Generation - Generate 5 diverse high-quality plays for next drawing.
        
        Returns:
            Dict with prediction generation results
        """
        try:
            # Initialize predictor (it loads data internally)
            predictor = Predictor()
            
            # Generate 5 diverse predictions for the next drawing (includes saving to log)
            diverse_predictions = predictor.predict_diverse_plays(num_plays=5, save_to_log=True)
            
            # Prepare result with all 5 plays
            plays_info = []
            for i, prediction in enumerate(diverse_predictions):
                play_info = {
                    'play_number': i + 1,
                    'prediction_id': prediction.get('log_id'),
                    'numbers': prediction['numbers'],
                    'powerball': prediction['powerball'],
                    'total_score': prediction['score_total'],
                    'score_details': prediction['score_details'],
                    'play_rank': prediction.get('play_rank', i + 1)
                }
                plays_info.append(play_info)
            
            result = {
                'predictions_generated': True,
                'num_plays_generated': len(diverse_predictions),
                'plays': plays_info,
                'model_version': diverse_predictions[0]['model_version'],
                'dataset_hash': diverse_predictions[0]['dataset_hash'],
                'candidates_evaluated': diverse_predictions[0]['num_candidates_evaluated'],
                'generation_method': 'diverse_deterministic',
                'diversity_algorithm': 'intelligent_selection'
            }
            
            # Log summary of generated plays
            logger.info(f"Generated {len(diverse_predictions)} diverse plays for next drawing:")
            for i, prediction in enumerate(diverse_predictions):
                logger.info(f"  Play {i+1}: {prediction['numbers']} + {prediction['powerball']} (Score: {prediction['score_total']:.4f})")
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction generation step failed: {e}")
            raise
    
    def step_historical_validation(self) -> Dict[str, Any]:
        """
        Step 5: Historical Validation - Validate predictions against historical data.
        
        Returns:
            Dict with historical validation results
        """
        try:
            # Ensure adaptive system is initialized
            if self.adaptive_system is None:
                if self.historical_data is None:
                    self.historical_data = get_all_draws()
                self.adaptive_system = initialize_adaptive_system(self.historical_data)
            
            # Get adaptive validator
            adaptive_validator = self.adaptive_system['adaptive_validator']
            
            # Run adaptive validation with learning enabled
            validation_csv_path = adaptive_validator.adaptive_validate_predictions(enable_learning=True)
            
            result = {
                'validation_completed': validation_csv_path is not None,
                'validation_file': validation_csv_path,
                'learning_enabled': True
            }
            
            if validation_csv_path:
                # Try to get validation statistics
                try:
                    validation_df = pd.read_csv(validation_csv_path)
                    result.update({
                        'total_validations': len(validation_df),
                        'winning_predictions': len(validation_df[validation_df['prize_category'] != 'Non-winning']),
                        'validation_summary': validation_df['prize_category'].value_counts().to_dict()
                    })
                except Exception as e:
                    logger.warning(f"Could not read validation statistics: {e}")
            
            logger.info(f"Historical validation completed: {validation_csv_path}")
            return result
            
        except Exception as e:
            logger.error(f"Historical validation step failed: {e}")
            raise
    
    def step_performance_analysis(self) -> Dict[str, Any]:
        """
        Step 6: Performance Analysis - Analyze system performance metrics.
        
        Returns:
            Dict with performance analysis results
        """
        try:
            # Get performance analytics for different time periods
            analytics_30d = get_performance_analytics(30)
            analytics_7d = get_performance_analytics(7)
            analytics_1d = get_performance_analytics(1)
            
            result = {
                'analytics_30_days': analytics_30d,
                'analytics_7_days': analytics_7d,
                'analytics_1_day': analytics_1d,
                'analysis_timestamp': datetime.now().isoformat()
            }
            
            # Generate performance insights
            insights = []
            
            if analytics_30d.get('total_predictions', 0) > 0:
                win_rate = analytics_30d.get('win_rate', 0)
                avg_accuracy = analytics_30d.get('avg_accuracy', 0)
                
                if win_rate > 5:
                    insights.append(f"Good win rate: {win_rate:.1f}% over 30 days")
                elif win_rate > 0:
                    insights.append(f"Low win rate: {win_rate:.1f}% over 30 days")
                else:
                    insights.append("No wins recorded in the last 30 days")
                
                if avg_accuracy > 0.5:
                    insights.append(f"High prediction accuracy: {avg_accuracy:.1%}")
                else:
                    insights.append(f"Low prediction accuracy: {avg_accuracy:.1%}")
            else:
                insights.append("No performance data available for analysis")
            
            result['performance_insights'] = insights
            
            logger.info(f"Performance analysis completed: {len(insights)} insights generated")
            return result
            
        except Exception as e:
            logger.error(f"Performance analysis step failed: {e}")
            raise
    
    def step_notifications_reports(self) -> Dict[str, Any]:
        """
        Step 7: Notifications & Reports - Generate reports and notifications.
        
        Returns:
            Dict with notifications and reports results
        """
        try:
            # Generate pipeline execution report
            report = self._generate_execution_report()
            
            # Save report to file
            report_dir = "reports"
            os.makedirs(report_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = os.path.join(report_dir, f"pipeline_report_{timestamp}.json")
            
            import json
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False, default=str)
            
            result = {
                'report_generated': True,
                'report_file': report_file,
                'report_timestamp': timestamp,
                'pipeline_status': self.pipeline_status,
                'notifications_sent': 0  # Basic implementation - no actual notifications yet
            }
            
            logger.info(f"Reports and notifications completed: {report_file}")
            return result
            
        except Exception as e:
            logger.error(f"Notifications and reports step failed: {e}")
            raise
    
    def _generate_execution_report(self) -> Dict[str, Any]:
        """Generate comprehensive execution report."""
        return {
            'pipeline_execution': {
                'start_time': self.execution_start_time.isoformat() if self.execution_start_time else None,
                'end_time': datetime.now().isoformat(),
                'total_execution_time': str(datetime.now() - self.execution_start_time) if self.execution_start_time else None,
                'status': self.pipeline_status
            },
            'system_info': {
                'config_file': self.config_path,
                'historical_data_records': len(self.historical_data) if self.historical_data is not None else 0,
                'adaptive_system_initialized': self.adaptive_system is not None
            },
            'generated_at': datetime.now().isoformat()
        }
    
    def _generate_pipeline_summary(self, pipeline_results: Dict[str, Any], execution_time) -> Dict[str, Any]:
        """Generate pipeline execution summary."""
        successful_steps = sum(1 for result in pipeline_results.values() if result.get('status') == 'success')
        total_steps = len(pipeline_results)
        
        return {
            'total_steps': total_steps,
            'successful_steps': successful_steps,
            'failed_steps': total_steps - successful_steps,
            'success_rate': f"{(successful_steps / total_steps * 100):.1f}%" if total_steps > 0 else "0%",
            'total_execution_time': str(execution_time),
            'pipeline_health': 'healthy' if successful_steps == total_steps else 'degraded' if successful_steps > 0 else 'failed'
        }
    
    def run_single_step(self, step_name: str) -> Dict[str, Any]:
        """
        Run a single pipeline step.
        
        Args:
            step_name: Name of the step to run
            
        Returns:
            Dict with step execution results
        """
        step_mapping = {
            'data': self.step_data_update,
            'data_update': self.step_data_update,
            'adaptive': self.step_adaptive_analysis,
            'adaptive_analysis': self.step_adaptive_analysis,
            'weights': self.step_weight_optimization,
            'weight_optimization': self.step_weight_optimization,
            'prediction': self.step_prediction_generation,
            'prediction_generation': self.step_prediction_generation,
            'validation': self.step_historical_validation,
            'historical_validation': self.step_historical_validation,
            'performance': self.step_performance_analysis,
            'performance_analysis': self.step_performance_analysis,
            'reports': self.step_notifications_reports,
            'notifications_reports': self.step_notifications_reports
        }
        
        if step_name not in step_mapping:
            available_steps = list(step_mapping.keys())
            raise ValueError(f"Unknown step '{step_name}'. Available steps: {available_steps}")
        
        logger.info(f"Running single step: {step_name}")
        self.execution_start_time = datetime.now()
        
        return self._execute_step(step_name, step_mapping[step_name])
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """
        Get current pipeline status and health information.
        
        Returns:
            Dict with pipeline status information
        """
        try:
            # Get basic system status
            status = {
                'timestamp': datetime.now().isoformat(),
                'database_initialized': True,  # We initialize in __init__
                'configuration_loaded': self.config is not None,
                'historical_data_available': self.historical_data is not None and not self.historical_data.empty,
                'adaptive_system_initialized': self.adaptive_system is not None,
                'recent_execution_status': self.pipeline_status
            }
            
            # Get database statistics
            try:
                historical_data = get_all_draws()
                performance_analytics = get_performance_analytics(7)
                
                status.update({
                    'database_records': len(historical_data),
                    'latest_draw_date': historical_data['draw_date'].max().strftime('%Y-%m-%d') if not historical_data.empty else None,
                    'recent_predictions': performance_analytics.get('total_predictions', 0),
                    'recent_win_rate': f"{performance_analytics.get('win_rate', 0):.1f}%"
                })
            except Exception as e:
                logger.warning(f"Could not retrieve database statistics: {e}")
                status['database_error'] = str(e)
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting pipeline status: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'status': 'error'
            }


def main():
    """Main entry point for the pipeline orchestrator."""
    parser = argparse.ArgumentParser(
        description="SHIOL+ Phase 5 Pipeline Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Run full pipeline
  python main.py --step data        # Run data update step only
  python main.py --step prediction  # Run prediction generation only
  python main.py --status           # Check pipeline status
  python main.py --help             # Show this help message

Available steps:
  data, adaptive, weights, prediction, validation, performance, reports
        """
    )
    
    parser.add_argument(
        '--step',
        type=str,
        help='Run a specific pipeline step only'
    )
    
    parser.add_argument(
        '--status',
        action='store_true',
        help='Check pipeline status and exit'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.ini',
        help='Path to configuration file (default: config/config.ini)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging output'
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize pipeline orchestrator
        orchestrator = PipelineOrchestrator(config_path=args.config)
        
        # Handle status check
        if args.status:
            status = orchestrator.get_pipeline_status()
            print("\n" + "=" * 50)
            print("SHIOL+ PIPELINE STATUS")
            print("=" * 50)
            
            for key, value in status.items():
                if key != 'recent_execution_status':
                    print(f"{key.replace('_', ' ').title()}: {value}")
            
            if status.get('recent_execution_status'):
                print("\nRecent Execution Status:")
                for step, step_status in status['recent_execution_status'].items():
                    status_symbol = "✓" if step_status.get('status') == 'success' else "✗"
                    print(f"  {status_symbol} {step}: {step_status.get('status', 'unknown')}")
            
            print("=" * 50)
            return
        
        # Handle single step execution
        if args.step:
            result = orchestrator.run_single_step(args.step)
            
            print(f"\nStep '{args.step}' execution result:")
            print(f"Status: {result.get('status', 'unknown')}")
            print(f"Execution time: {result.get('execution_time', 'unknown')}")
            
            if result.get('status') == 'failed':
                print(f"Error: {result.get('error', 'unknown error')}")
                sys.exit(1)
            else:
                print("Step completed successfully!")
            return
        
        # Run full pipeline
        result = orchestrator.run_full_pipeline()
        
        # Print results summary
        print("\n" + "=" * 60)
        print("PIPELINE EXECUTION SUMMARY")
        print("=" * 60)
        print(f"Status: {result.get('status', 'unknown')}")
        print(f"Execution time: {result.get('execution_time', 'unknown')}")
        
        if result.get('summary'):
            summary = result['summary']
            print(f"Steps completed: {summary.get('successful_steps', 0)}/{summary.get('total_steps', 0)}")
            print(f"Success rate: {summary.get('success_rate', '0%')}")
            print(f"Pipeline health: {summary.get('pipeline_health', 'unknown')}")
        
        if result.get('status') == 'failed':
            print(f"Error: {result.get('error', 'unknown error')}")
            sys.exit(1)
        else:
            print("Pipeline execution completed successfully!")
        
    except KeyboardInterrupt:
        print("\nPipeline execution interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        if args.verbose:
            print(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
SHIOL+ Phase 5 Pipeline Orchestrator
====================================

Main pipeline orchestrator that coordinates all 7 pipeline steps:
1. Data Update
2. Adaptive Analysis  
3. Prediction Generation
4. Weight Optimization
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
import subprocess
import requests
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
        Execute the complete SHIOL+ Phase 5 pipeline with optimized flow.
        
        FLUJO OPTIMIZADO:
        1. Data Update & Sorteo Detection
        2. Validation of Previous Predictions (if drawing day)
        3. Adaptive Learning from Results
        4. Weight Optimization (if sufficient data)
        5. Prediction Generation for Next Drawing
        6. Performance Analysis & Insights
        7. Reports & Next Drawing Schedule

        Returns:
            Dict with pipeline execution results and status
        """
        logger.info("=" * 60)
        logger.info("STARTING SHIOL+ PHASE 5 OPTIMIZED PIPELINE EXECUTION")
        logger.info("=" * 60)

        self.execution_start_time = datetime.now()
        pipeline_results = {}
        current_date = datetime.now()
        
        # Determine if today is a drawing day (Monday=0, Wednesday=2, Saturday=5)
        is_drawing_day = current_date.weekday() in [0, 2, 5]
        hours_after_drawing = current_date.hour >= 23  # After 11 PM ET
        
        logger.info(f"Pipeline execution context: Drawing day: {is_drawing_day}, After 11PM: {hours_after_drawing}")

        try:
            # STEP 1: Data Update & Drawing Detection
            logger.info("STEP 1/7: Data Update & Drawing Detection")
            pipeline_results['data_update'] = self._execute_step('data_update', self.step_data_update)
            
            # STEP 2: Historical Validation (PRIORITY on drawing days after 11 PM)
            if is_drawing_day and hours_after_drawing:
                logger.info("STEP 2/7: Historical Validation (Drawing Day Priority)")
                pipeline_results['historical_validation'] = self._execute_step('historical_validation', self.step_historical_validation)
                
                # STEP 3: Adaptive Analysis (Enhanced learning from fresh results)
                logger.info("STEP 3/7: Adaptive Analysis (Post-Drawing Learning)")
                pipeline_results['adaptive_analysis'] = self._execute_step('adaptive_analysis', self.step_adaptive_analysis)
                
                # STEP 4: Weight Optimization (Triggered by new results)
                logger.info("STEP 4/7: Weight Optimization (Results-Based)")
                pipeline_results['weight_optimization'] = self._execute_step('weight_optimization', self.step_weight_optimization)
            else:
                # STEP 2: Adaptive Analysis (Regular maintenance)
                logger.info("STEP 2/7: Adaptive Analysis (Maintenance Mode)")
                pipeline_results['adaptive_analysis'] = self._execute_step('adaptive_analysis', self.step_adaptive_analysis)
                
                # STEP 3: Weight Optimization (Regular optimization)
                logger.info("STEP 3/7: Weight Optimization (Scheduled)")
                pipeline_results['weight_optimization'] = self._execute_step('weight_optimization', self.step_weight_optimization)
                
                # STEP 4: Historical Validation (Maintenance validation)
                logger.info("STEP 4/7: Historical Validation (Maintenance)")
                pipeline_results['historical_validation'] = self._execute_step('historical_validation', self.step_historical_validation)

            # STEP 5: Prediction Generation (ALWAYS generate for next drawing)
            logger.info("STEP 5/7: Prediction Generation (Next Drawing)")
            pipeline_results['prediction_generation'] = self._execute_step('prediction_generation', self.step_prediction_generation)

            # STEP 6: Performance Analysis
            logger.info("STEP 6/7: Performance Analysis")
            pipeline_results['performance_analysis'] = self._execute_step('performance_analysis', self.step_performance_analysis)

            # STEP 7: Notifications & Reports
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


    def _calculate_next_drawing_date(self) -> str:
        """
        Calculate the next Powerball drawing date.
        Drawings are: Monday (0), Wednesday (2), Saturday (5)
        
        Returns:
            str: Next drawing date in YYYY-MM-DD format
        """
        from datetime import datetime, timedelta
        
        current_date = datetime.now()
        current_weekday = current_date.weekday()
        
        # Drawing days: Monday=0, Wednesday=2, Saturday=5
        drawing_days = [0, 2, 5]
        
        # If today is a drawing day and it's before 11 PM, the drawing is today
        if current_weekday in drawing_days and current_date.hour < 23:
            return current_date.strftime('%Y-%m-%d')
        
        # Otherwise, find the next drawing day
        days_ahead = 0
        for i in range(1, 8):  # Check next 7 days
            next_date = current_date + timedelta(days=i)
            if next_date.weekday() in drawing_days:
                days_ahead = i
                break
        
        next_drawing_date = current_date + timedelta(days=days_ahead)
        return next_drawing_date.strftime('%Y-%m-%d')

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
        Step 4: Weight Optimization - Optimize scoring weights based on performance.

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
            total_predictions = performance_data.get('total_predictions', 0)
            
            # If we just generated predictions in step 3, check the database directly
            if total_predictions < 10:
                try:
                    from src.database import get_prediction_history
                    recent_predictions = get_prediction_history(limit=50)
                    total_predictions = len(recent_predictions)
                    logger.info(f"Found {total_predictions} total predictions in database for optimization")
                except Exception as e:
                    logger.warning(f"Could not check recent predictions: {e}")
            
            if total_predictions < 5:  # Reduced threshold since we just generated predictions
                logger.warning(f"Still insufficient data for weight optimization (have {total_predictions}, need at least 5)")
                return {
                    'status': 'skipped',
                    'reason': 'insufficient_data',
                    'predictions_available': total_predictions,
                    'minimum_required': 5
                }
            
            logger.info(f"Proceeding with weight optimization using {total_predictions} predictions")

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
        Step 5: Prediction Generation - Generate 100 Smart AI predictions for next drawing.
        
        FUNCIONALIDAD OPTIMIZADA:
        - Valida calidad del modelo antes de generar predicciones
        - Ejecuta reentrenamiento automático si es necesario
        - Calcula fecha del próximo sorteo (Lunes, Miércoles, Sábado)
        - Genera 100 predicciones Smart AI con fecha objetivo
        - Aplica pesos adaptativos optimizados

        Returns:
            Dict with prediction generation results including next drawing date
        """
        try:
            # NUEVA FUNCIONALIDAD: Validar modelo antes de generar predicciones
            try:
                from src.model_validator import validate_model_before_prediction, is_model_ready_for_prediction
                from src.auto_retrainer import execute_automatic_retrain_if_needed
                
                logger.info("Validating model quality before prediction generation...")
                model_validation = validate_model_before_prediction()
                
                # Check for specific issues that require retraining
                needs_retrain = False
                retrain_reason = "model_quality_acceptable"
                
                # Check recent performance metrics for feature mismatch
                if isinstance(model_validation.get('validation_metrics'), dict):
                    recent_perf = model_validation['validation_metrics'].get('recent_performance', {})
                    if recent_perf.get('status') == 'feature_mismatch':
                        logger.warning("Feature shape mismatch detected - forcing model retrain...")
                        needs_retrain = True
                        retrain_reason = "feature_compatibility_issue"
                
                # Check overall model readiness
                if not needs_retrain and not is_model_ready_for_prediction():
                    logger.warning("Model quality below acceptable threshold - attempting automatic retrain...")
                    needs_retrain = True
                    retrain_reason = "quality_below_threshold"
                
                if needs_retrain:
                    retrain_results = execute_automatic_retrain_if_needed()
                    
                    if retrain_results.get('retrain_executed', False):
                        logger.info(f"Model successfully retrained due to: {retrain_reason}")
                    else:
                        logger.warning(f"Model retrain not executed despite {retrain_reason} - proceeding with caution")
                else:
                    retrain_results = {'retrain_executed': False, 'reason': retrain_reason}
                    
            except ImportError as e:
                logger.warning(f"Model validation not available: {e}")
                model_validation = {'validation_available': False}
                retrain_results = {'retrain_executed': False, 'error': 'validation_unavailable'}
            except Exception as e:
                logger.error(f"Error during model validation: {e}")
                model_validation = {'validation_error': str(e)}
                retrain_results = {'retrain_executed': False, 'error': f'validation_error: {str(e)}'}
            
            # Calculate next drawing date (Monday=0, Wednesday=2, Saturday=5)
            next_drawing_date = self._calculate_next_drawing_date()
            logger.info(f"Generating predictions for next drawing date: {next_drawing_date}")
            
            # Initialize predictor (it loads data internally)
            predictor = Predictor()

            # Generate 100 Smart AI predictions for the next drawing (includes saving to log)
            logger.info("Generating 100 Smart AI predictions with optimized weights...")
            smart_predictions = predictor.predict_diverse_plays(num_plays=100, save_to_log=True)

            # Prepare result with all 100 plays
            plays_info = []
            for i, prediction in enumerate(smart_predictions):
                play_info = {
                    'play_number': i + 1,
                    'prediction_id': prediction.get('log_id'),
                    'numbers': prediction['numbers'],
                    'powerball': prediction['powerball'],
                    'total_score': prediction['score_total'],
                    'score_details': prediction['score_details'],
                    'play_rank': prediction.get('play_rank', i + 1),
                    'method': 'smart_ai'
                }
                plays_info.append(play_info)

            # Calculate statistics
            avg_score = sum(p['score_total'] for p in smart_predictions) / len(smart_predictions)
            top_10_avg = sum(p['score_total'] for p in smart_predictions[:10]) / 10

            result = {
                'predictions_generated': True,
                'method': 'smart_ai',
                'num_plays_generated': len(smart_predictions),
                'target_drawing_date': next_drawing_date,  # NUEVA: Fecha del próximo sorteo
                'plays': plays_info,
                'statistics': {
                    'average_score': avg_score,
                    'top_10_average_score': top_10_avg,
                    'best_score': smart_predictions[0]['score_total'],
                    'worst_score': smart_predictions[-1]['score_total']
                },
                'model_version': smart_predictions[0]['model_version'],
                'dataset_hash': smart_predictions[0]['dataset_hash'],
                'candidates_evaluated': smart_predictions[0]['num_candidates_evaluated'],
                'generation_method': 'smart_ai_diverse_deterministic',
                'diversity_algorithm': 'intelligent_selection_100_plays',
                'drawing_schedule': {
                    'next_drawing_date': next_drawing_date,
                    'is_drawing_day': datetime.now().weekday() in [0, 2, 5],
                    'drawing_days': ['Monday', 'Wednesday', 'Saturday']
                },
                # Validación y reentrenamiento automático
                'model_validation': model_validation,
                'retrain_executed': retrain_results.get('retrain_executed', False) if 'retrain_results' in locals() else False
            }

            # Log summary of generated plays
            logger.info(f"Generated {len(smart_predictions)} Smart AI predictions for next drawing")
            logger.info(f"Average score: {avg_score:.4f}")
            logger.info(f"Top 10 average score: {top_10_avg:.4f}")
            logger.info(f"Best prediction: {smart_predictions[0]['numbers']} + {smart_predictions[0]['powerball']} (Score: {smart_predictions[0]['score_total']:.4f})")
            logger.info("All 100 Smart AI predictions have been saved to the database")

            return result

        except Exception as e:
            logger.error(f"Smart AI prediction generation step failed: {e}")
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


def get_public_ip() -> Optional[str]:
    """
    Detect the public IP address of the server automatically.

    Returns:
        str: Public IP address or None if detection fails
    """
    services = [
        'https://api.ipify.org',
        'https://ipinfo.io/ip',
        'https://icanhazip.com',
        'https://ident.me'
    ]

    for service in services:
        try:
            logger.debug(f"Trying to get public IP from {service}")
            response = requests.get(service, timeout=5)
            if response.status_code == 200:
                ip = response.text.strip()
                # Basic IP validation
                if ip and '.' in ip and len(ip.split('.')) == 4:
                    logger.info(f"Public IP detected: {ip}")
                    return ip
        except Exception as e:
            logger.debug(f"Failed to get IP from {service}: {e}")
            continue

    logger.warning("Could not detect public IP address")
    return None


def start_api_server(host: str = "0.0.0.0", port: int = 8000, auto_detect_ip: bool = True):
    """
    Start the API server optimized for VPN access.

    Args:
        host: Host to bind to (default: 0.0.0.0 for external access)
        port: Port to bind to (default: 8000)
        auto_detect_ip: Whether to auto-detect and display public IP
    """
    print("🚀 Starting SHIOL+ API Server...")
    print("=" * 50)

    # Verify that we're in the correct directory
    if not os.path.exists("src/api.py"):
        print("❌ Error: src/api.py not found")
        print("   Make sure to run this script from the project root directory")
        sys.exit(1)

    # Check if virtual environment is activated
    if not hasattr(sys, 'real_prefix') and not (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("⚠️  Warning: No virtual environment detected")
        print("   It's recommended to activate the virtual environment first:")
        print("   source venv/bin/activate  # Linux/Mac")
        print("   .\\venv\\Scripts\\activate  # Windows")
        print()

    # Server configuration
    print(f"📡 Server Configuration:")
    print(f"   Host: {host} (allows external connections)")
    print(f"   Port: {port}")
    print(f"   CORS: Enabled for all origins")
    print()

    # Display access URLs
    print("🌐 Access URLs:")
    print(f"   Local: http://127.0.0.1:{port}")

    if auto_detect_ip:
        public_ip = get_public_ip()
        if public_ip:
            print(f"   External/VPN: http://{public_ip}:{port}")
            print()
            print("📱 For mobile/remote access:")
            print(f"   Use: http://{public_ip}:{port}")
        else:
            print("   External/VPN: Could not detect public IP")
            print("   Check your network configuration or use manual IP")
    print()

    print("🔧 Starting uvicorn server...")
    print("   Press Ctrl+C to stop the server")
    print("=" * 50)

    try:
        # Command to start uvicorn
        cmd = [
            "uvicorn",
            "src.api:app",
            "--host", host,
            "--port", str(port),
            "--reload",  # Auto-reload in development
            "--access-log",  # Access logs
            "--log-level", "info"
        ]

        # Execute the server
        subprocess.run(cmd, check=True)

    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error starting server: {e}")
        print("\n🔍 Possible solutions:")
        print("1. Install uvicorn: pip install uvicorn")
        print("2. Install dependencies: pip install -r requirements.txt")
        print("3. Check that the port is not in use")
    except FileNotFoundError:
        print("\n❌ Error: uvicorn not found")
        print("   Install with: pip install uvicorn")


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
  python main.py --server           # Start API server for VPN access
  python main.py --api --port 8080  # Start API server on custom port
  python main.py --help             # Show this help message

Available steps:
  data, adaptive, weights, prediction, validation, performance, reports

Server mode:
  --server or --api starts the web API server optimized for VPN access
  Automatically detects public IP and configures CORS for external access
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

    parser.add_argument(
        '--server',
        action='store_true',
        help='Start API server optimized for VPN access'
    )

    parser.add_argument(
        '--api',
        action='store_true',
        help='Alias for --server (start API server)'
    )

    parser.add_argument(
        '--host',
        type=str,
        default='0.0.0.0',
        help='Host to bind server to (default: 0.0.0.0 for external access)'
    )

    parser.add_argument(
        '--port',
        type=int,
        default=8000,
        help='Port to bind server to (default: 8000)'
    )

    args = parser.parse_args()

    try:
        # Handle server mode
        if args.server or args.api:
            start_api_server(host=args.host, port=args.port, auto_detect_ip=True)
            return

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

from fastapi import APIRouter, HTTPException
from typing import Dict
from loguru import logger
from datetime import datetime
from pathlib import Path
import os
import sqlite3

import src.database as db

database_router = APIRouter(prefix="/database", tags=["Database Management"])

@database_router.get("/stats")
async def get_database_stats():
    """Get database statistics for dashboard"""
    try:
        conn = db.get_db_connection()
        cursor = conn.cursor()

        # Get total records from main tables
        cursor.execute("SELECT COUNT(*) FROM powerball_draws")
        draws_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM predictions_log")
        predictions_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM performance_tracking")
        validations_count = cursor.fetchone()[0]

        # Get database file size
        db_path = db.get_db_path() # Use centralized configuration
        try:
            db_size_bytes = os.path.getsize(db_path)
            db_size_mb = round(db_size_bytes / (1024 * 1024), 2)
        except FileNotFoundError:
            db_size_mb = 0
            logger.warning(f"Database file not found at expected path: {db_path}")
        except Exception as e:
            db_size_mb = 0
            logger.error(f"Could not get database file size for {db_path}: {e}")

        conn.close()

        return {
            "total_records": draws_count + predictions_count + validations_count,
            "draws_count": draws_count,
            "predictions_count": predictions_count,
            "validations_count": validations_count,
            "size_mb": db_size_mb,
            "last_updated": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting database stats: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting database stats: {str(e)}")

@database_router.post("/cleanup")
async def cleanup_database(cleanup_options: Dict[str, bool]):
    """Clean database based on selected options"""
    from src.api import pipeline_executions, pipeline_logs
    try:
        logger.info(f"Starting database cleanup with options: {cleanup_options}")

        conn = db.get_db_connection()
        cursor = conn.cursor()
        results = []

        if cleanup_options.get('predictions', False):
            try:
                cursor.execute("DELETE FROM predictions_log")
                deleted_predictions = cursor.rowcount
                results.append(f"Deleted {deleted_predictions} predictions")
                logger.info(f"Deleted {deleted_predictions} prediction records")
            except Exception as e:
                logger.error(f"Error deleting predictions: {e}")
                results.append(f"Error deleting predictions: {str(e)}")

        if cleanup_options.get('validations', False):
            try:
                cursor.execute("DELETE FROM performance_tracking")
                deleted_performance = cursor.rowcount
                results.append(f"Deleted {deleted_performance} performance tracking records")
                logger.info(f"Deleted {deleted_performance} performance tracking records")
            except Exception as e:
                logger.error(f"Error deleting validations: {e}")
                results.append(f"Error deleting validations: {str(e)}")

        if cleanup_options.get('logs', False):
            # Clear log files
            log_files_cleared = 0
            logs_dir = Path('logs')
            if logs_dir.exists():
                for log_file in logs_dir.glob('*.log'):
                    log_file.unlink()
                    log_files_cleared += 1
            results.append(f"Cleared {log_files_cleared} log files")
            logger.info(f"Cleared {log_files_cleared} log files")

        if cleanup_options.get('pipeline_logs', False):
            # Clear pipeline reports
            pipeline_files_cleared = 0

            # Clear pipeline reports
            reports_dir = Path('reports')
            if reports_dir.exists():
                for report_file in reports_dir.glob('pipeline_report_*.json'):
                    report_file.unlink()
                    pipeline_files_cleared += 1

            # Clear system logs
            logs_dir = Path('logs')
            if logs_dir.exists():
                for log_file in logs_dir.glob('*.log'):
                    log_file.unlink()
                    pipeline_files_cleared += 1

            # Clear global pipeline execution tracking
            pipeline_executions.clear()
            pipeline_logs.clear()

            results.append(f"Cleared {pipeline_files_cleared} pipeline log files and execution history")
            logger.info(f"Cleared {pipeline_files_cleared} pipeline log files and execution history")

        if cleanup_options.get('models', False):
            # Reset AI models data - using safe table operations
            safe_model_tables = {
                'adaptive_weights': 'DELETE FROM adaptive_weights',
                'model_feedback': 'DELETE FROM model_feedback', 
                'reliable_plays': 'DELETE FROM reliable_plays'
            }
            
            deleted_weights = deleted_feedback = deleted_plays = 0
            
            for table_name, safe_query in safe_model_tables.items():
                try:
                    cursor.execute(safe_query)
                    deleted_count = cursor.rowcount
                    if table_name == 'adaptive_weights':
                        deleted_weights = deleted_count
                    elif table_name == 'model_feedback':
                        deleted_feedback = deleted_count
                    elif table_name == 'reliable_plays':
                        deleted_plays = deleted_count
                    logger.info(f"Safely cleared {deleted_count} records from {table_name}")
                except sqlite3.Error as e:
                    logger.error(f"Error clearing {table_name}: {e}")
                    
            results.append(f"Reset AI models: deleted {deleted_weights} weight sets, {deleted_feedback} feedback records, {deleted_plays} reliable plays")
            logger.info(f"Reset AI models data")

        if cleanup_options.get('complete_reset', False):
            # Complete system reset - using predefined safe queries
            safe_reset_operations = {
                'predictions_log': 'DELETE FROM predictions_log',
                'performance_tracking': 'DELETE FROM performance_tracking',
                'adaptive_weights': 'DELETE FROM adaptive_weights',
                'pattern_analysis': 'DELETE FROM pattern_analysis',
                'reliable_plays': 'DELETE FROM reliable_plays',
                'model_feedback': 'DELETE FROM model_feedback'
            }
            total_cleared = 0

            for table_name, safe_query in safe_reset_operations.items():
                try:
                    cursor.execute(safe_query)
                    deleted_count = cursor.rowcount
                    total_cleared += deleted_count
                    logger.info(f"Safely cleared {deleted_count} records from {table_name}")
                except sqlite3.Error as e:
                    logger.error(f"Error clearing {table_name}: {e}")
                    results.append(f"Error clearing {table_name}: {str(e)}")

            # Clear all log files
            logs_dir = Path('logs')
            if logs_dir.exists():
                for log_file in logs_dir.glob('*'):
                    if log_file.is_file():
                        log_file.unlink()

            # Clear pipeline reports
            reports_dir = Path('reports')
            if reports_dir.exists():
                for report_file in reports_dir.glob('pipeline_report_*.json'):
                    report_file.unlink()

            # Clear global pipeline execution tracking
            pipeline_executions.clear()
            pipeline_logs.clear()

            results.append(f"Complete system reset: cleared {total_cleared} total records, all log files, and pipeline execution history (kept historical draw data and configuration)")
            logger.info(f"Complete system reset performed")

        # Commit all changes
        conn.commit()
        conn.close()

        if not results:
            results.append("No cleanup options selected")

        logger.info(f"Database cleanup completed successfully: {results}")

        return {
            "success": True,
            "message": "Cleanup completed successfully",
            "details": results
        }

    except Exception as e:
        logger.error(f"Error during database cleanup: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error during cleanup: {str(e)}")

@database_router.get("/status")
async def get_database_status():
    """Get detailed database status and record counts"""
    try:
        conn = db.get_db_connection()
        cursor = conn.cursor()

        # Get counts from all main tables
        table_counts = {}

        # Safe table count operations using predefined queries
        safe_table_queries = {
            'powerball_draws': 'SELECT COUNT(*) FROM powerball_draws',
            'predictions_log': 'SELECT COUNT(*) FROM predictions_log',
            'performance_tracking': 'SELECT COUNT(*) FROM performance_tracking',
            'adaptive_weights': 'SELECT COUNT(*) FROM adaptive_weights',
            'pattern_analysis': 'SELECT COUNT(*) FROM pattern_analysis',
            'reliable_plays': 'SELECT COUNT(*) FROM reliable_plays',
            'model_feedback': 'SELECT COUNT(*) FROM model_feedback',
            'system_config': 'SELECT COUNT(*) FROM system_config'
        }

        for table_name, safe_query in safe_table_queries.items():
            try:
                cursor.execute(safe_query)
                count = cursor.fetchone()[0]
                table_counts[table_name] = count
            except sqlite3.Error as e:
                table_counts[table_name] = f"Error: {str(e)}"

        # Check if database is "empty" (only has essential data)
        is_empty = (
            table_counts.get('predictions_log', 0) == 0 and
            table_counts.get('performance_tracking', 0) == 0 and
            table_counts.get('adaptive_weights', 0) == 0 and
            table_counts.get('model_feedback', 0) == 0
        )

        conn.close()

        return {
            "database_status": "empty" if is_empty else "has_data",
            "table_counts": table_counts,
            "total_predictions": table_counts.get('predictions_log', 0),
            "total_validations": table_counts.get('performance_tracking', 0),
            "has_historical_data": table_counts.get('powerball_draws', 0) > 0,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error getting database status: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting database status: {str(e)}")

@database_router.post("/backup")
async def backup_database():
    """Create database backup"""
    try:
        from shutil import copy2
        from datetime import datetime

        db_path = db.get_db_path() # Use centralized configuration
        backup_dir = os.path.join(os.path.dirname(db_path), "backups")

        os.makedirs(backup_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = os.path.join(backup_dir, f"shiolplus_backup_{timestamp}.db")

        copy2(db_path, backup_path)

        logger.info(f"Database backup created: {backup_path}")
        return {"message": "Database backup created successfully", "backup_file": backup_path}

    except FileNotFoundError:
        logger.error(f"Database file not found for backup: {db_path}")
        raise HTTPException(status_code=404, detail=f"Database file not found at {db_path}")
    except Exception as e:
        logger.error(f"Error creating database backup: {e}")
        raise HTTPException(status_code=500, detail=f"Error creating database backup: {str(e)}")

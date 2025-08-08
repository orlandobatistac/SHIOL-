
from fastapi import APIRouter, HTTPException
from loguru import logger
from datetime import datetime
import psutil

import src.database as db

analytics_router = APIRouter(prefix="/analytics", tags=["Analytics"])

@analytics_router.get("/performance")
async def get_performance_analytics():
    """Get performance analytics for dashboard"""
    try:
        # Get performance data from database without passing connection
        performance_data = db.get_performance_analytics(days_back=30)

        # Calculate key metrics
        total_predictions = performance_data.get('total_predictions', 0)
        winning_predictions = performance_data.get('winning_predictions', 0)
        win_rate = (winning_predictions / total_predictions * 100) if total_predictions > 0 else 0

        return {
            "win_rate": round(win_rate, 1),
            "avg_score": round(performance_data.get('avg_score', 0), 3),
            "best_method": performance_data.get('best_method', 'smart_ai'),
            "total_predictions": total_predictions,
            "winning_predictions": winning_predictions,
            "total_prize_amount": performance_data.get('total_prize_amount', 0),
            "roi_percentage": round(performance_data.get('roi_percentage', 0), 2)
        }
    except Exception as e:
        logger.error(f"Error getting performance analytics: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting performance analytics: {str(e)}")

@analytics_router.get("/system/stats")
async def get_system_stats():
    """Get real-time system statistics for dashboard monitoring"""
    try:
        from fastapi import Request
        from src.api import app
        
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')

        # Get pipeline status from orchestrator if available
        pipeline_status = "ready"
        last_execution = "Never"

        if hasattr(app.state, 'orchestrator') and app.state.orchestrator:
            try:
                status_info = app.state.orchestrator.get_pipeline_status()
                pipeline_status = status_info.get('current_status', 'ready')
                last_execution_info = status_info.get('last_execution')
                if last_execution_info and isinstance(last_execution_info, dict) and last_execution_info.get('start_time'):
                    last_execution = last_execution_info['start_time']
                elif isinstance(last_execution_info, str):
                    last_execution = last_execution_info
            except Exception as e:
                logger.warning(f"Could not retrieve pipeline status: {e}")
                pipeline_status = "Error"

        return {
            "cpu_usage": round(cpu_percent, 1),
            "memory_usage": round(memory.percent, 1),
            "disk_usage": round((disk.used / disk.total) * 100, 1),
            "memory_total_gb": round(memory.total / (1024**3), 2),
            "disk_total_gb": round(disk.total / (1024**3), 2),
            "pipeline_status": pipeline_status,
            "last_execution": last_execution,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting system stats: {e}")
        # Return default stats instead of raising exception
        return {
            "cpu_usage": 0.0,
            "memory_usage": 0.0,
            "disk_usage": 0.0,
            "memory_total_gb": 0.0,
            "disk_total_gb": 0.0,
            "pipeline_status": "error",
            "last_execution": "Error",
            "timestamp": datetime.now().isoformat()
        }

@analytics_router.get("/logs")
async def get_system_logs():
    """Get recent system logs"""
    try:
        import os
        log_dir = os.path.join(os.path.dirname(__file__), "..", "logs")
        log_files = []

        if os.path.exists(log_dir):
            log_files = [f for f in os.listdir(log_dir) if f.endswith('.log')]

        recent_logs = []

        # Read recent log entries (last 100 lines)
        if log_files:
            # Find the most recently modified log file
            latest_log_file_path = max(
                [os.path.join(log_dir, f) for f in log_files],
                key=os.path.getctime
            )

            try:
                with open(latest_log_file_path, 'r') as f:
                    lines = f.readlines()[-100:]  # Last 100 lines

                for line in lines:
                    if 'ERROR' in line:
                        level = 'error'
                    elif 'WARNING' in line:
                        level = 'warning'
                    else:
                        level = 'info'

                    recent_logs.append({
                        'level': level,
                        'message': line.strip(),
                        'timestamp': datetime.now().isoformat() # Use current timestamp for log entry if specific timestamp not parsed
                    })
            except FileNotFoundError:
                logger.warning(f"Latest log file not found for reading: {latest_log_file_path}")
            except Exception as e:
                logger.error(f"Error reading log file {latest_log_file_path}: {e}")

        return recent_logs

    except Exception as e:
        logger.error(f"Error getting logs: {e}")
        return []

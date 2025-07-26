"""
SHIOL+ Pipeline Logger
=====================

Comprehensive pipeline logging system for enhanced monitoring and debugging capabilities.
Provides structured JSON logging, database integration, performance metrics collection,
and seamless integration with the existing loguru logging system.

Features:
- Structured JSON logging with consistent format
- Database integration for persistent log storage
- Performance metrics collection and monitoring
- Asynchronous logging for non-blocking operations
- Log rotation and retention management
- Pipeline execution tracking with unique IDs
- System health monitoring and alerts
- API-ready log retrieval methods

Author: SHIOL+ Development Team
Version: 1.0.0
"""

import asyncio
import configparser
import json
import os
import psutil
import sqlite3
import threading
import time
import traceback
import uuid
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from loguru import logger

from src.database import get_db_connection, get_db_path


class PipelineLogger:
    """
    Comprehensive pipeline logging system with structured JSON format,
    database integration, and performance monitoring capabilities.
    """
    
    def __init__(self, config_path: str = "config/config.ini"):
        """
        Initialize the PipelineLogger with configuration and database setup.
        
        Args:
            config_path: Path to the configuration file
        """
        self.config_path = config_path
        self.config = self._load_configuration()
        
        # Initialize logging configuration
        self.log_level = self.config.get("pipeline", "pipeline_log_level", fallback="INFO")
        self.log_retention_days = self.config.getint("pipeline", "log_retention_days", fallback=90)
        self.collect_performance_metrics = self.config.getboolean("pipeline", "collect_performance_metrics", fallback=True)
        self.max_log_file_size_mb = self.config.getint("pipeline", "max_log_file_size_mb", fallback=100)
        self.max_log_files = self.config.getint("pipeline", "max_log_files", fallback=10)
        
        # Initialize internal state
        self.execution_id = None
        self.pipeline_start_time = None
        self.step_start_times = {}
        self.performance_data = {}
        self.log_buffer = deque(maxlen=1000)  # In-memory buffer for recent logs
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="PipelineLogger")
        
        # Setup database and logging
        self._initialize_database()
        self._setup_structured_logging()
        
        # Start background tasks
        self._start_background_tasks()
        
        logger.info("PipelineLogger initialized successfully")
    
    def _load_configuration(self) -> configparser.ConfigParser:
        """Load configuration from config.ini file."""
        config = configparser.ConfigParser()
        if os.path.exists(self.config_path):
            config.read(self.config_path)
        else:
            logger.warning(f"Configuration file not found: {self.config_path}. Using defaults.")
        return config
    
    def _initialize_database(self):
        """Initialize database tables for pipeline logging."""
        try:
            with get_db_connection() as conn:
                cursor = conn.cursor()
                
                # Create pipeline_logs table for persistent log storage
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS pipeline_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        execution_id TEXT,
                        level TEXT NOT NULL,
                        category TEXT NOT NULL,
                        message TEXT NOT NULL,
                        context TEXT,
                        step_name TEXT,
                        duration_ms INTEGER,
                        performance_metrics TEXT,
                        error_details TEXT,
                        stack_trace TEXT,
                        user_info TEXT,
                        system_info TEXT,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        INDEX(timestamp),
                        INDEX(execution_id),
                        INDEX(level),
                        INDEX(category)
                    )
                """)
                
                # Create pipeline_executions table for tracking pipeline runs
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS pipeline_executions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        execution_id TEXT UNIQUE NOT NULL,
                        start_time TEXT NOT NULL,
                        end_time TEXT,
                        status TEXT NOT NULL,
                        total_steps INTEGER,
                        completed_steps INTEGER,
                        current_step TEXT,
                        error_message TEXT,
                        performance_summary TEXT,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        INDEX(execution_id),
                        INDEX(status),
                        INDEX(start_time)
                    )
                """)
                
                # Create system_metrics table for performance monitoring
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS system_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        execution_id TEXT,
                        cpu_percent REAL,
                        memory_percent REAL,
                        memory_available_gb REAL,
                        disk_percent REAL,
                        disk_free_gb REAL,
                        process_memory_mb REAL,
                        active_threads INTEGER,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        INDEX(timestamp),
                        INDEX(execution_id)
                    )
                """)
                
                conn.commit()
                logger.info("Pipeline logging database tables initialized")
                
        except Exception as e:
            logger.error(f"Failed to initialize pipeline logging database: {e}")
            raise
    
    def _setup_structured_logging(self):
        """Setup structured logging with JSON format and rotation."""
        try:
            # Get log file path from config
            log_file = self.config.get("paths", "log_file", fallback="logs/shiolplus.log")
            pipeline_log_file = log_file.replace(".log", "_pipeline.log")
            
            # Ensure log directory exists
            os.makedirs(os.path.dirname(pipeline_log_file), exist_ok=True)
            
            # Add structured pipeline logger
            logger.add(
                pipeline_log_file,
                format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {extra[execution_id]} | {extra[category]} | {message}",
                level=self.log_level,
                rotation=f"{self.max_log_file_size_mb} MB",
                retention=f"{self.log_retention_days} days",
                compression="gz",
                serialize=True,  # Enable JSON serialization
                backtrace=True,
                diagnose=True,
                enqueue=True,  # Enable async logging
                filter=lambda record: record["extra"].get("pipeline_logger", False)
            )
            
            logger.info("Structured pipeline logging configured")
            
        except Exception as e:
            logger.error(f"Failed to setup structured logging: {e}")
            raise
    
    def _start_background_tasks(self):
        """Start background tasks for log processing and cleanup."""
        # Start periodic cleanup task
        cleanup_thread = threading.Thread(
            target=self._periodic_cleanup,
            daemon=True,
            name="PipelineLogger-Cleanup"
        )
        cleanup_thread.start()
        
        # Start performance monitoring if enabled
        if self.collect_performance_metrics:
            metrics_thread = threading.Thread(
                target=self._periodic_metrics_collection,
                daemon=True,
                name="PipelineLogger-Metrics"
            )
            metrics_thread.start()
    
    def _periodic_cleanup(self):
        """Periodic cleanup of old logs and database records."""
        while True:
            try:
                time.sleep(3600)  # Run every hour
                self.cleanup_old_logs()
            except Exception as e:
                logger.error(f"Error in periodic cleanup: {e}")
    
    def _periodic_metrics_collection(self):
        """Periodic collection of system performance metrics."""
        while True:
            try:
                time.sleep(60)  # Collect every minute
                if self.execution_id:  # Only collect during pipeline execution
                    self._collect_system_metrics()
            except Exception as e:
                logger.error(f"Error in metrics collection: {e}")
    
    def _collect_system_metrics(self):
        """Collect current system performance metrics."""
        try:
            # Get system metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('.')
            process = psutil.Process()
            
            metrics = {
                "timestamp": datetime.now().isoformat(),
                "execution_id": self.execution_id,
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available_gb": round(memory.available / (1024**3), 2),
                "disk_percent": disk.percent,
                "disk_free_gb": round(disk.free / (1024**3), 2),
                "process_memory_mb": round(process.memory_info().rss / (1024**2), 2),
                "active_threads": threading.active_count()
            }
            
            # Store in database asynchronously
            self.executor.submit(self._store_system_metrics, metrics)
            
            # Store in performance data for current execution
            if self.execution_id:
                if "system_metrics" not in self.performance_data:
                    self.performance_data["system_metrics"] = []
                self.performance_data["system_metrics"].append(metrics)
            
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
    
    def _store_system_metrics(self, metrics: Dict[str, Any]):
        """Store system metrics in database."""
        try:
            with get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO system_metrics
                    (timestamp, execution_id, cpu_percent, memory_percent, memory_available_gb,
                     disk_percent, disk_free_gb, process_memory_mb, active_threads)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    metrics["timestamp"], metrics["execution_id"], metrics["cpu_percent"],
                    metrics["memory_percent"], metrics["memory_available_gb"],
                    metrics["disk_percent"], metrics["disk_free_gb"],
                    metrics["process_memory_mb"], metrics["active_threads"]
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to store system metrics: {e}")
    
    def _create_log_entry(self, level: str, category: str, message: str, 
                         context: Optional[Dict[str, Any]] = None,
                         step_name: Optional[str] = None,
                         duration_ms: Optional[int] = None,
                         error_details: Optional[str] = None,
                         stack_trace: Optional[str] = None) -> Dict[str, Any]:
        """Create a structured log entry."""
        timestamp = datetime.now().isoformat()
        
        # Get system info if collecting performance metrics
        system_info = None
        if self.collect_performance_metrics:
            try:
                memory = psutil.virtual_memory()
                system_info = {
                    "cpu_percent": psutil.cpu_percent(),
                    "memory_percent": memory.percent,
                    "memory_available_gb": round(memory.available / (1024**3), 2),
                    "active_threads": threading.active_count()
                }
            except Exception:
                system_info = {"error": "Failed to collect system info"}
        
        log_entry = {
            "timestamp": timestamp,
            "execution_id": self.execution_id,
            "level": level,
            "category": category,
            "message": message,
            "context": json.dumps(context) if context else None,
            "step_name": step_name,
            "duration_ms": duration_ms,
            "error_details": error_details,
            "stack_trace": stack_trace,
            "user_info": json.dumps({"user": "system", "source": "pipeline"}),
            "system_info": json.dumps(system_info) if system_info else None
        }
        
        return log_entry
    
    def _log_to_database(self, log_entry: Dict[str, Any]):
        """Store log entry in database asynchronously."""
        def store_log():
            try:
                with get_db_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        INSERT INTO pipeline_logs
                        (timestamp, execution_id, level, category, message, context,
                         step_name, duration_ms, error_details, stack_trace, user_info, system_info)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        log_entry["timestamp"], log_entry["execution_id"], log_entry["level"],
                        log_entry["category"], log_entry["message"], log_entry["context"],
                        log_entry["step_name"], log_entry["duration_ms"], log_entry["error_details"],
                        log_entry["stack_trace"], log_entry["user_info"], log_entry["system_info"]
                    ))
                    conn.commit()
            except Exception as e:
                logger.error(f"Failed to store log entry in database: {e}")
        
        self.executor.submit(store_log)
    
    def _log_structured(self, level: str, category: str, message: str, **kwargs):
        """Log a structured message to both loguru and database."""
        # Create structured log entry
        log_entry = self._create_log_entry(level, category, message, **kwargs)
        
        # Add to in-memory buffer
        self.log_buffer.append(log_entry)
        
        # Log to loguru with structured format
        logger.bind(
            pipeline_logger=True,
            execution_id=self.execution_id or "none",
            category=category,
            step_name=kwargs.get("step_name"),
            duration_ms=kwargs.get("duration_ms")
        ).log(level, message)
        
        # Store in database asynchronously
        self._log_to_database(log_entry)
    
    def log_pipeline_start(self, total_steps: int = 7, context: Optional[Dict[str, Any]] = None):
        """
        Log pipeline execution start with timing and context.
        
        Args:
            total_steps: Total number of steps in the pipeline
            context: Additional context information
        """
        self.execution_id = str(uuid.uuid4())[:8]
        self.pipeline_start_time = datetime.now()
        self.performance_data = {"start_time": self.pipeline_start_time.isoformat()}
        
        # Store pipeline execution record
        try:
            with get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO pipeline_executions
                    (execution_id, start_time, status, total_steps, completed_steps, current_step)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    self.execution_id, self.pipeline_start_time.isoformat(),
                    "running", total_steps, 0, "starting"
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to store pipeline execution record: {e}")
        
        self._log_structured(
            "INFO", "pipeline_execution", 
            f"Pipeline execution started with ID: {self.execution_id}",
            context=context or {},
            step_name="pipeline_start"
        )
        
        logger.info(f"Pipeline execution started: {self.execution_id}")
    
    def log_pipeline_end(self, status: str = "success", results: Optional[Dict[str, Any]] = None,
                        error: Optional[str] = None):
        """
        Log pipeline completion with results and performance summary.
        
        Args:
            status: Pipeline execution status (success, failed, partial)
            results: Pipeline execution results
            error: Error message if pipeline failed
        """
        if not self.execution_id or not self.pipeline_start_time:
            logger.warning("Pipeline end logged without corresponding start")
            return
        
        end_time = datetime.now()
        execution_time = end_time - self.pipeline_start_time
        
        # Update performance data
        self.performance_data.update({
            "end_time": end_time.isoformat(),
            "execution_time_seconds": execution_time.total_seconds(),
            "status": status,
            "results": results
        })
        
        # Update pipeline execution record
        try:
            with get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE pipeline_executions
                    SET end_time = ?, status = ?, error_message = ?, 
                        performance_summary = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE execution_id = ?
                """, (
                    end_time.isoformat(), status, error,
                    json.dumps(self.performance_data), self.execution_id
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to update pipeline execution record: {e}")
        
        self._log_structured(
            "INFO" if status == "success" else "ERROR",
            "pipeline_execution",
            f"Pipeline execution {status} in {execution_time}",
            context={
                "execution_time_seconds": execution_time.total_seconds(),
                "results": results or {},
                "performance_summary": self.performance_data
            },
            step_name="pipeline_end",
            duration_ms=int(execution_time.total_seconds() * 1000),
            error_details=error
        )
        
        logger.info(f"Pipeline execution completed: {self.execution_id} ({status})")
        
        # Reset execution state
        self.execution_id = None
        self.pipeline_start_time = None
        self.step_start_times.clear()
    
    def log_step_start(self, step_name: str, context: Optional[Dict[str, Any]] = None):
        """
        Log individual step start with context.
        
        Args:
            step_name: Name of the pipeline step
            context: Additional context information
        """
        step_start_time = datetime.now()
        self.step_start_times[step_name] = step_start_time
        
        # Update current step in execution record
        if self.execution_id:
            try:
                with get_db_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        UPDATE pipeline_executions
                        SET current_step = ?, updated_at = CURRENT_TIMESTAMP
                        WHERE execution_id = ?
                    """, (step_name, self.execution_id))
                    conn.commit()
            except Exception as e:
                logger.error(f"Failed to update current step: {e}")
        
        self._log_structured(
            "INFO", "step_execution",
            f"Step '{step_name}' started",
            context=context or {},
            step_name=step_name
        )
    
    def log_step_end(self, step_name: str, status: str = "success", 
                    metrics: Optional[Dict[str, Any]] = None,
                    error: Optional[str] = None):
        """
        Log step completion with metrics and timing.
        
        Args:
            step_name: Name of the pipeline step
            status: Step execution status
            metrics: Step-specific metrics
            error: Error message if step failed
        """
        end_time = datetime.now()
        duration_ms = None
        
        if step_name in self.step_start_times:
            duration = end_time - self.step_start_times[step_name]
            duration_ms = int(duration.total_seconds() * 1000)
            del self.step_start_times[step_name]
        
        # Update completed steps count
        if self.execution_id and status == "success":
            try:
                with get_db_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        UPDATE pipeline_executions
                        SET completed_steps = completed_steps + 1, updated_at = CURRENT_TIMESTAMP
                        WHERE execution_id = ?
                    """, (self.execution_id,))
                    conn.commit()
            except Exception as e:
                logger.error(f"Failed to update completed steps: {e}")
        
        self._log_structured(
            "INFO" if status == "success" else "ERROR",
            "step_execution",
            f"Step '{step_name}' {status}" + (f" in {duration_ms}ms" if duration_ms else ""),
            context={
                "step_metrics": metrics or {},
                "step_status": status
            },
            step_name=step_name,
            duration_ms=duration_ms,
            error_details=error
        )
    
    def log_error(self, error: Exception, context: Optional[Dict[str, Any]] = None,
                 step_name: Optional[str] = None):
        """
        Log errors with full stack traces and context.
        
        Args:
            error: Exception object
            context: Additional context information
            step_name: Name of the step where error occurred
        """
        error_details = str(error)
        stack_trace = traceback.format_exc()
        
        self._log_structured(
            "ERROR", "error",
            f"Error occurred: {error_details}",
            context=context or {},
            step_name=step_name,
            error_details=error_details,
            stack_trace=stack_trace
        )
        
        logger.error(f"Pipeline error in {step_name or 'unknown step'}: {error_details}")
    
    def log_performance_metrics(self, metrics: Dict[str, Any], 
                              category: str = "performance",
                              step_name: Optional[str] = None):
        """
        Log system performance data and metrics.
        
        Args:
            metrics: Performance metrics dictionary
            category: Metrics category
            step_name: Associated step name
        """
        self._log_structured(
            "INFO", category,
            f"Performance metrics collected: {len(metrics)} metrics",
            context={"performance_metrics": metrics},
            step_name=step_name
        )
        
        # Store metrics in performance data
        if self.execution_id:
            if "custom_metrics" not in self.performance_data:
                self.performance_data["custom_metrics"] = []
            self.performance_data["custom_metrics"].append({
                "timestamp": datetime.now().isoformat(),
                "step_name": step_name,
                "category": category,
                "metrics": metrics
            })
    
    def get_recent_logs(self, limit: int = 100, level: Optional[str] = None,
                       category: Optional[str] = None, 
                       execution_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Retrieve recent logs for API endpoints with filtering.
        
        Args:
            limit: Maximum number of logs to return
            level: Filter by log level
            category: Filter by category
            execution_id: Filter by execution ID
            
        Returns:
            List of log entries
        """
        try:
            with get_db_connection() as conn:
                query = """
                    SELECT timestamp, execution_id, level, category, message, context,
                           step_name, duration_ms, error_details, system_info
                    FROM pipeline_logs
                    WHERE 1=1
                """
                params = []
                
                if level:
                    query += " AND level = ?"
                    params.append(level)
                
                if category:
                    query += " AND category = ?"
                    params.append(category)
                
                if execution_id:
                    query += " AND execution_id = ?"
                    params.append(execution_id)
                
                query += " ORDER BY timestamp DESC LIMIT ?"
                params.append(limit)
                
                cursor = conn.cursor()
                cursor.execute(query, params)
                
                logs = []
                for row in cursor.fetchall():
                    log_entry = {
                        "timestamp": row[0],
                        "execution_id": row[1],
                        "level": row[2],
                        "category": row[3],
                        "message": row[4],
                        "context": json.loads(row[5]) if row[5] else None,
                        "step_name": row[6],
                        "duration_ms": row[7],
                        "error_details": row[8],
                        "system_info": json.loads(row[9]) if row[9] else None
                    }
                    logs.append(log_entry)
                
                return logs
                
        except Exception as e:
            logger.error(f"Failed to retrieve recent logs: {e}")
            return []
    
    def get_execution_logs(self, execution_id: str) -> Dict[str, Any]:
        """
        Get all logs for a specific pipeline execution.
        
        Args:
            execution_id: Pipeline execution ID
            
        Returns:
            Dictionary with execution details and logs
        """
        try:
            with get_db_connection() as conn:
                cursor = conn.cursor()
                
                # Get execution details
                cursor.execute("""
                    SELECT execution_id, start_time, end_time, status, total_steps,
                           completed_steps, current_step, error_message, performance_summary
                    FROM pipeline_executions
                    WHERE execution_id = ?
                """, (execution_id,))
                
                execution_row = cursor.fetchone()
                if not execution_row:
                    return {"error": f"Execution {execution_id} not found"}
                
                execution_details = {
                    "execution_id": execution_row[0],
                    "start_time": execution_row[1],
                    "end_time": execution_row[2],
                    "status": execution_row[3],
                    "total_steps": execution_row[4],
                    "completed_steps": execution_row[5],
                    "current_step": execution_row[6],
                    "error_message": execution_row[7],
                    "performance_summary": json.loads(execution_row[8]) if execution_row[8] else None
                }
                
                # Get all logs for this execution
                logs = self.get_recent_logs(limit=1000, execution_id=execution_id)
                
                # Get system metrics for this execution
                cursor.execute("""
                    SELECT timestamp, cpu_percent, memory_percent, memory_available_gb,
                           disk_percent, disk_free_gb, process_memory_mb, active_threads
                    FROM system_metrics
                    WHERE execution_id = ?
                    ORDER BY timestamp
                """, (execution_id,))
                
                metrics = []
                for row in cursor.fetchall():
                    metrics.append({
                        "timestamp": row[0],
                        "cpu_percent": row[1],
                        "memory_percent": row[2],
                        "memory_available_gb": row[3],
                        "disk_percent": row[4],
                        "disk_free_gb": row[5],
                        "process_memory_mb": row[6],
                        "active_threads": row[7]
                    })
                
                return {
                    "execution_details": execution_details,
                    "logs": logs,
                    "system_metrics": metrics,
                    "log_count": len(logs),
                    "metrics_count": len(metrics)
                }
                
        except Exception as e:
            logger.error(f"Failed to get execution logs: {e}")
            return {"error": str(e)}
    
    def cleanup_old_logs(self):
        """Remove old logs based on retention policy."""
        try:
            cutoff_date = datetime.now() - timedelta(days=self.log_retention_days)
            cutoff_str = cutoff_date.isoformat()
            
            with get_db_connection() as conn:
                cursor = conn.cursor()
                
                # Clean up old pipeline logs
                cursor.execute("DELETE FROM pipeline_logs WHERE timestamp < ?", (cutoff_str,))
                logs_deleted = cursor.rowcount
                
                # Clean up old pipeline executions
                cursor.execute("DELETE FROM pipeline_executions WHERE start_time < ?", (cutoff_str,))
                executions_deleted = cursor.rowcount
                
                # Clean up old system metrics
                cursor.execute("DELETE FROM system_metrics WHERE timestamp < ?", (cutoff_str,))
                metrics_deleted = cursor.rowcount
                
                conn.commit()
                
                if logs_deleted > 0 or executions_deleted > 0 or metrics_deleted > 0:
                    logger.info(f"Cleaned up old logs: {logs_deleted} logs, {executions_deleted} executions, {metrics_deleted} metrics")
                
        except Exception as e:
            logger.error(f"Failed to cleanup old logs: {e}")
    
    def get_pipeline_statistics(self, days_back: int = 30) -> Dict[str, Any]:
        """
        Get pipeline execution statistics for the specified period.
        
        Args:
            days_back: Number of days to analyze
            
        Returns:
            Dictionary with pipeline statistics
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=days_back)
            cutoff_str = cutoff_date.isoformat()
            
            with get_db_connection() as conn:
                cursor = conn.cursor()
                
                # Get execution statistics
                cursor.execute("""
                    SELECT status, COUNT(*) as count,
                           AVG(CASE WHEN end_time IS NOT NULL THEN
                               (julianday(end_time) - julianday(start_time)) * 24 * 60 * 60
                           END) as avg_duration_seconds
                    FROM pipeline_executions
                    WHERE start_time >= ?
                    GROUP BY status
                """, (cutoff_str,))
                
                execution_stats = {}
                for row in cursor.fetchall():
                    execution_stats[row[0]] = {
                        "count": row[1],
                        "avg_duration_seconds": row[2] if row[2] else 0
                    }
                
                # Get log level distribution
                cursor.execute("""
                    SELECT level, COUNT(*) as count
                    FROM pipeline_logs
                    WHERE timestamp >= ?
                    GROUP BY level
                """, (cutoff_str,))
                
                log_level_stats = dict(cursor.fetchall())
                
                # Get step performance
                cursor.execute("""
                    SELECT step_name, COUNT(*) as count,
                           AVG(duration_ms) as avg_duration_ms,
                           MIN(duration_ms) as min_duration_ms,
                           MAX(duration_ms) as max_duration_ms
                    FROM pipeline_logs
                    WHERE timestamp >= ? AND step_name IS NOT NULL AND duration_ms IS NOT NULL
                    GROUP BY step_name
                """, (cutoff_str,))
                
                step_stats = {}
                for row in cursor.fetchall():
                    step_stats[row[0]] = {
                        "count": row[1],
                        "avg_duration_ms": row[2] if row[2] else 0,
                        "min_duration_ms": row[3] if row[3] else 0,
                        "max_duration_ms": row[4] if row[4] else 0
                    }
                
                # Get error statistics
                cursor.execute("""
                    SELECT COUNT(*) as total_errors,
                           COUNT(DISTINCT execution_id) as executions_with_errors
                    FROM pipeline_logs
                    WHERE timestamp >= ? AND level = 'ERROR'
                """, (cutoff_str,))
                
                error_row = cursor.fetchone()
                error_stats = {
                    "total_errors": error_row[0] if error_row else 0,
                    "executions_with_errors": error_row[1] if error_row else 0
                }
                
                return {
                    "period_days": days_back,
                    "execution_statistics": execution_stats,
                    "log_level_distribution": log_level_stats,
                    "step_performance": step_stats,
                    "error_statistics": error_stats,
                    "generated_at": datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Failed to get pipeline statistics: {e}")
            return {"error": str(e)}
    
    def get_system_health(self) -> Dict[str, Any]:
        """
        Get current system health status and metrics.
        
        Returns:
            Dictionary with system health information
        """
        try:
            # Get current system metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('.')
            
            # Get database size
            db_path = get_db_path()
            db_size_mb = os.path.getsize(db_path) / (1024**2) if os.path.exists(db_path) else 0
            
            # Get log buffer status
            buffer_usage = len(self.log_buffer) / self.log_buffer.maxlen * 100
            
            # Determine health status
            health_issues = []
            if cpu_percent > 80:
                health_issues.append(f"High CPU usage: {cpu_percent:.1f}%")
            if memory.percent > 85:
                health_issues.append(f"High memory usage: {memory.percent:.1f}%")
            if disk.percent > 90:
                health_issues.append(f"Low disk space: {disk.percent:.1f}% used")
            if buffer_usage > 90:
                health_issues.append(f"Log buffer nearly full: {buffer_usage:.1f}%")
            
            overall_status = "healthy"
            if len(health_issues) > 2:
                overall_status = "critical"
            elif len(health_issues) > 0:
                overall_status = "warning"
            
            return {
                "overall_status": overall_status,
                "health_issues": health_issues,
                "system_metrics": {
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory.percent,
                    "memory_available_gb": round(memory.available / (1024**3), 2),
                    "disk_percent": disk.percent,
                    "disk_free_gb": round(disk.free / (1024**3), 2)
                },
                "pipeline_status": {
                    "current_execution_id": self.execution_id,
                    "is_running": self.execution_id is not None,
                    "log_buffer_usage_percent": round(buffer_usage, 1),
                    "active_threads": threading.active_count()
                },
                "database_info": {
                    "size_mb": round(db_size_mb, 2),
                    "path": db_path
                },
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get system health: {e}")
            return {
                "overall_status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def export_logs(self, execution_id: Optional[str] = None,
                   start_date: Optional[str] = None,
                   end_date: Optional[str] = None,
                   format: str = "json") -> Optional[str]:
        """
        Export logs to file for analysis or backup.
        
        Args:
            execution_id: Specific execution ID to export
            start_date: Start date filter (ISO format)
            end_date: End date filter (ISO format)
            format: Export format ('json' or 'csv')
            
        Returns:
            Path to exported file or None if failed
        """
        try:
            # Build query
            query = "SELECT * FROM pipeline_logs WHERE 1=1"
            params = []
            
            if execution_id:
                query += " AND execution_id = ?"
                params.append(execution_id)
            
            if start_date:
                query += " AND timestamp >= ?"
                params.append(start_date)
            
            if end_date:
                query += " AND timestamp <= ?"
                params.append(end_date)
            
            query += " ORDER BY timestamp"
            
            # Execute query
            with get_db_connection() as conn:
                df = pd.read_sql_query(query, conn, params=params)
            
            if df.empty:
                logger.warning("No logs found for export criteria")
                return None
            
            # Create export directory
            export_dir = "exports"
            os.makedirs(export_dir, exist_ok=True)
            
            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename_parts = ["pipeline_logs", timestamp]
            if execution_id:
                filename_parts.append(f"exec_{execution_id}")
            
            if format.lower() == "csv":
                filename = "_".join(filename_parts) + ".csv"
                filepath = os.path.join(export_dir, filename)
                df.to_csv(filepath, index=False)
            else:
                filename = "_".join(filename_parts) + ".json"
                filepath = os.path.join(export_dir, filename)
                df.to_json(filepath, orient="records", indent=2)
            
            logger.info(f"Logs exported to: {filepath} ({len(df)} records)")
            return filepath
            
        except Exception as e:
            logger.error(f"Failed to export logs: {e}")
            return None
    
    def __del__(self):
        """Cleanup resources when logger is destroyed."""
        try:
            if hasattr(self, 'executor'):
                self.executor.shutdown(wait=True)
        except Exception:
            pass


# Global pipeline logger instance
_pipeline_logger_instance = None


def get_pipeline_logger(config_path: str = "config/config.ini") -> PipelineLogger:
    """
    Get or create the global pipeline logger instance.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        PipelineLogger instance
    """
    global _pipeline_logger_instance
    
    if _pipeline_logger_instance is None:
        _pipeline_logger_instance = PipelineLogger(config_path)
    
    return _pipeline_logger_instance


# Convenience functions for easy integration
def log_pipeline_start(total_steps: int = 7, context: Optional[Dict[str, Any]] = None):
    """Convenience function to log pipeline start."""
    get_pipeline_logger().log_pipeline_start(total_steps, context)


def log_pipeline_end(status: str = "success", results: Optional[Dict[str, Any]] = None,
                    error: Optional[str] = None):
    """Convenience function to log pipeline end."""
    get_pipeline_logger().log_pipeline_end(status, results, error)


def log_step_start(step_name: str, context: Optional[Dict[str, Any]] = None):
    """Convenience function to log step start."""
    get_pipeline_logger().log_step_start(step_name, context)


def log_step_end(step_name: str, status: str = "success",
                metrics: Optional[Dict[str, Any]] = None,
                error: Optional[str] = None):
    """Convenience function to log step end."""
    get_pipeline_logger().log_step_end(step_name, status, metrics, error)


def log_error(error: Exception, context: Optional[Dict[str, Any]] = None,
             step_name: Optional[str] = None):
    """Convenience function to log errors."""
    get_pipeline_logger().log_error(error, context, step_name)


def log_performance_metrics(metrics: Dict[str, Any],
                          category: str = "performance",
                          step_name: Optional[str] = None):
    """Convenience function to log performance metrics."""
    get_pipeline_logger().log_performance_metrics(metrics, category, step_name)
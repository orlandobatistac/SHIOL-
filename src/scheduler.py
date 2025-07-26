#!/usr/bin/env python3
"""
SHIOL+ WeeklyScheduler Component
===============================

WeeklyScheduler class with APScheduler integration for automatic pipeline execution.
Provides weekly scheduling, timezone support, manual schedule management, and 
seamless integration with the PipelineOrchestrator.

Features:
- APScheduler integration for weekly execution
- Configuration loading from config/config.ini [pipeline] section
- Timezone support (configurable via config)
- Automatic pipeline execution on schedule
- Manual schedule management (start, stop, reschedule)
- Error handling and retry logic
- Status information and monitoring
"""

import configparser
import os
import sys
import traceback
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Callable
from pathlib import Path

# APScheduler imports
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.events import EVENT_JOB_EXECUTED, EVENT_JOB_ERROR, EVENT_JOB_MISSED
from apscheduler.executors.pool import ThreadPoolExecutor
from apscheduler.jobstores.memory import MemoryJobStore
import pytz

# Logging
from loguru import logger


class WeeklyScheduler:
    """
    WeeklyScheduler class for automatic pipeline execution using APScheduler.
    
    Provides weekly scheduling with timezone support, configuration management,
    error handling, and integration with PipelineOrchestrator.
    """
    
    def __init__(self, config_path: str = "config/config.ini", pipeline_orchestrator=None):
        """
        Initialize the WeeklyScheduler.
        
        Args:
            config_path: Path to configuration file
            pipeline_orchestrator: PipelineOrchestrator instance for pipeline execution
        """
        self.config_path = config_path
        self.config = self._load_configuration()
        self.pipeline_orchestrator = pipeline_orchestrator
        self.scheduler = None
        self.job_id = "weekly_pipeline_execution"
        self.is_scheduler_running = False
        self.last_execution_result = None
        self.execution_history = []
        self.retry_count = 0
        
        # Load scheduler configuration
        self._load_scheduler_config()
        
        # Initialize APScheduler
        self._initialize_scheduler()
        
        logger.info("WeeklyScheduler initialized successfully")
    
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
            logger.error(f"Failed to load configuration: {e}")
            raise
    
    def _load_scheduler_config(self):
        """Load scheduler-specific configuration from config.ini."""
        try:
            pipeline_section = self.config['pipeline']
            
            # Weekly execution settings
            self.weekly_execution_day = pipeline_section.getint('weekly_execution_day', 0)  # 0=Monday
            self.execution_time = pipeline_section.get('execution_time', '02:00')
            self.timezone_str = pipeline_section.get('timezone', 'America/New_York')
            self.auto_execution_enabled = pipeline_section.getboolean('auto_execution_enabled', True)
            
            # Retry settings
            self.max_retry_attempts = pipeline_section.getint('max_retry_attempts', 3)
            self.retry_delay_minutes = pipeline_section.getint('retry_delay_minutes', 30)
            self.retry_backoff_multiplier = pipeline_section.getfloat('retry_backoff_multiplier', 2.0)
            
            # Timeout settings
            self.pipeline_timeout_seconds = pipeline_section.getint('pipeline_timeout_seconds', 3600)
            
            # Parse execution time
            time_parts = self.execution_time.split(':')
            self.execution_hour = int(time_parts[0])
            self.execution_minute = int(time_parts[1]) if len(time_parts) > 1 else 0
            
            # Setup timezone
            try:
                self.timezone = pytz.timezone(self.timezone_str)
            except pytz.exceptions.UnknownTimeZoneError:
                logger.warning(f"Unknown timezone '{self.timezone_str}', using UTC")
                self.timezone = pytz.UTC
                self.timezone_str = 'UTC'
            
            logger.info(f"Scheduler config loaded: Day={self.weekly_execution_day}, Time={self.execution_time}, TZ={self.timezone_str}")
            
        except Exception as e:
            logger.error(f"Failed to load scheduler configuration: {e}")
            # Set defaults
            self.weekly_execution_day = 0
            self.execution_hour = 2
            self.execution_minute = 0
            self.timezone = pytz.timezone('America/New_York')
            self.timezone_str = 'America/New_York'
            self.auto_execution_enabled = True
            self.max_retry_attempts = 3
            self.retry_delay_minutes = 30
            self.retry_backoff_multiplier = 2.0
            self.pipeline_timeout_seconds = 3600
    
    def _initialize_scheduler(self):
        """Initialize APScheduler with proper configuration."""
        try:
            # Configure job stores and executors
            jobstores = {
                'default': MemoryJobStore()
            }
            
            executors = {
                'default': ThreadPoolExecutor(max_workers=1)  # Single thread for pipeline execution
            }
            
            job_defaults = {
                'coalesce': True,  # Combine multiple pending executions into one
                'max_instances': 1,  # Only one instance of the job can run at a time
                'misfire_grace_time': 300  # 5 minutes grace time for missed jobs
            }
            
            # Create scheduler
            self.scheduler = BackgroundScheduler(
                jobstores=jobstores,
                executors=executors,
                job_defaults=job_defaults,
                timezone=self.timezone
            )
            
            # Add event listeners
            self.scheduler.add_listener(self._job_executed_listener, EVENT_JOB_EXECUTED)
            self.scheduler.add_listener(self._job_error_listener, EVENT_JOB_ERROR)
            self.scheduler.add_listener(self._job_missed_listener, EVENT_JOB_MISSED)
            
            logger.info("APScheduler initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize APScheduler: {e}")
            raise
    
    def start_scheduler(self) -> Dict[str, Any]:
        """
        Start the weekly scheduler.
        
        Returns:
            Dict with scheduler start results
        """
        try:
            if not self.auto_execution_enabled:
                logger.warning("Auto execution is disabled in configuration")
                return {
                    'status': 'disabled',
                    'message': 'Auto execution is disabled in configuration',
                    'auto_execution_enabled': False
                }
            
            if self.is_scheduler_running:
                logger.warning("Scheduler is already running")
                return {
                    'status': 'already_running',
                    'message': 'Scheduler is already running',
                    'next_run_time': self.get_next_run_time()
                }
            
            # Start the scheduler
            self.scheduler.start()
            
            # Add the weekly job
            self._schedule_weekly_job()
            
            self.is_scheduler_running = True
            next_run = self.get_next_run_time()
            
            logger.info(f"WeeklyScheduler started successfully. Next run: {next_run}")
            
            return {
                'status': 'started',
                'message': 'WeeklyScheduler started successfully',
                'next_run_time': next_run,
                'weekly_execution_day': self.weekly_execution_day,
                'execution_time': self.execution_time,
                'timezone': self.timezone_str
            }
            
        except Exception as e:
            logger.error(f"Failed to start scheduler: {e}")
            return {
                'status': 'error',
                'message': f'Failed to start scheduler: {str(e)}',
                'error': str(e)
            }
    
    def stop_scheduler(self) -> Dict[str, Any]:
        """
        Stop the scheduler.
        
        Returns:
            Dict with scheduler stop results
        """
        try:
            if not self.is_scheduler_running:
                logger.warning("Scheduler is not running")
                return {
                    'status': 'not_running',
                    'message': 'Scheduler is not running'
                }
            
            # Shutdown the scheduler
            self.scheduler.shutdown(wait=True)
            self.is_scheduler_running = False
            
            logger.info("WeeklyScheduler stopped successfully")
            
            return {
                'status': 'stopped',
                'message': 'WeeklyScheduler stopped successfully'
            }
            
        except Exception as e:
            logger.error(f"Failed to stop scheduler: {e}")
            return {
                'status': 'error',
                'message': f'Failed to stop scheduler: {str(e)}',
                'error': str(e)
            }
    
    def _schedule_weekly_job(self):
        """Schedule the weekly pipeline execution job."""
        try:
            # Remove existing job if it exists
            if self.scheduler.get_job(self.job_id):
                self.scheduler.remove_job(self.job_id)
            
            # Create cron trigger for weekly execution
            trigger = CronTrigger(
                day_of_week=self.weekly_execution_day,
                hour=self.execution_hour,
                minute=self.execution_minute,
                timezone=self.timezone
            )
            
            # Add the job
            self.scheduler.add_job(
                func=self._execute_pipeline_job,
                trigger=trigger,
                id=self.job_id,
                name="Weekly Pipeline Execution",
                replace_existing=True
            )
            
            logger.info(f"Weekly job scheduled: {self._get_day_name(self.weekly_execution_day)} at {self.execution_time} ({self.timezone_str})")
            
        except Exception as e:
            logger.error(f"Failed to schedule weekly job: {e}")
            raise
    
    def get_next_run_time(self) -> Optional[str]:
        """
        Get next scheduled execution time.
        
        Returns:
            Next run time as ISO string or None if not scheduled
        """
        try:
            if not self.is_scheduler_running or not self.scheduler:
                return None
            
            job = self.scheduler.get_job(self.job_id)
            if job and job.next_run_time:
                return job.next_run_time.isoformat()
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get next run time: {e}")
            return None
    
    def reschedule(self, weekly_execution_day: Optional[int] = None, 
                   execution_time: Optional[str] = None,
                   timezone_str: Optional[str] = None) -> Dict[str, Any]:
        """
        Update schedule with new parameters.
        
        Args:
            weekly_execution_day: Day of week (0=Monday, 6=Sunday)
            execution_time: Time in HH:MM format
            timezone_str: Timezone string (e.g., 'America/New_York')
            
        Returns:
            Dict with reschedule results
        """
        try:
            # Update configuration if provided
            if weekly_execution_day is not None:
                self.weekly_execution_day = weekly_execution_day
            
            if execution_time is not None:
                self.execution_time = execution_time
                time_parts = execution_time.split(':')
                self.execution_hour = int(time_parts[0])
                self.execution_minute = int(time_parts[1]) if len(time_parts) > 1 else 0
            
            if timezone_str is not None:
                try:
                    self.timezone = pytz.timezone(timezone_str)
                    self.timezone_str = timezone_str
                except pytz.exceptions.UnknownTimeZoneError:
                    logger.warning(f"Unknown timezone '{timezone_str}', keeping current timezone")
            
            # Reschedule if scheduler is running
            if self.is_scheduler_running:
                self._schedule_weekly_job()
            
            next_run = self.get_next_run_time()
            
            logger.info(f"Schedule updated: {self._get_day_name(self.weekly_execution_day)} at {self.execution_time} ({self.timezone_str})")
            
            return {
                'status': 'rescheduled',
                'message': 'Schedule updated successfully',
                'weekly_execution_day': self.weekly_execution_day,
                'execution_time': self.execution_time,
                'timezone': self.timezone_str,
                'next_run_time': next_run
            }
            
        except Exception as e:
            logger.error(f"Failed to reschedule: {e}")
            return {
                'status': 'error',
                'message': f'Failed to reschedule: {str(e)}',
                'error': str(e)
            }
    
    def is_running(self) -> bool:
        """
        Check if scheduler is active.
        
        Returns:
            True if scheduler is running, False otherwise
        """
        return self.is_scheduler_running and self.scheduler is not None and self.scheduler.running
    
    def get_job_status(self) -> Dict[str, Any]:
        """
        Get current job status.
        
        Returns:
            Dict with job status information
        """
        try:
            status = {
                'scheduler_running': self.is_running(),
                'auto_execution_enabled': self.auto_execution_enabled,
                'weekly_execution_day': self.weekly_execution_day,
                'weekly_execution_day_name': self._get_day_name(self.weekly_execution_day),
                'execution_time': self.execution_time,
                'timezone': self.timezone_str,
                'next_run_time': self.get_next_run_time(),
                'last_execution_result': self.last_execution_result,
                'retry_count': self.retry_count,
                'max_retry_attempts': self.max_retry_attempts,
                'execution_history_count': len(self.execution_history)
            }
            
            # Add job details if scheduler is running
            if self.is_running():
                job = self.scheduler.get_job(self.job_id)
                if job:
                    status.update({
                        'job_id': job.id,
                        'job_name': job.name,
                        'job_next_run_time': job.next_run_time.isoformat() if job.next_run_time else None
                    })
            
            return status
            
        except Exception as e:
            logger.error(f"Failed to get job status: {e}")
            return {
                'error': str(e),
                'scheduler_running': False
            }
    
    def _execute_pipeline_job(self):
        """Execute the pipeline job (called by APScheduler)."""
        execution_start_time = datetime.now()
        
        try:
            logger.info("=" * 60)
            logger.info("SCHEDULED PIPELINE EXECUTION STARTED")
            logger.info("=" * 60)
            
            # Check if pipeline orchestrator is available
            if self.pipeline_orchestrator is None:
                raise RuntimeError("PipelineOrchestrator not available for scheduled execution")
            
            # Execute the full pipeline
            result = self.pipeline_orchestrator.run_full_pipeline()
            
            # Record execution result
            execution_time = datetime.now() - execution_start_time
            self.last_execution_result = {
                'status': result.get('status', 'unknown'),
                'execution_time': str(execution_time),
                'timestamp': execution_start_time.isoformat(),
                'result': result
            }
            
            # Add to execution history
            self.execution_history.append(self.last_execution_result)
            
            # Keep only last 10 executions in history
            if len(self.execution_history) > 10:
                self.execution_history = self.execution_history[-10:]
            
            # Reset retry count on success
            if result.get('status') == 'success':
                self.retry_count = 0
                logger.info(f"✓ Scheduled pipeline execution completed successfully in {execution_time}")
            else:
                logger.error(f"✗ Scheduled pipeline execution failed: {result.get('error', 'Unknown error')}")
                self._handle_execution_failure(result)
            
        except Exception as e:
            execution_time = datetime.now() - execution_start_time
            error_msg = f"Scheduled pipeline execution failed: {str(e)}"
            
            self.last_execution_result = {
                'status': 'error',
                'error': error_msg,
                'execution_time': str(execution_time),
                'timestamp': execution_start_time.isoformat(),
                'traceback': traceback.format_exc()
            }
            
            self.execution_history.append(self.last_execution_result)
            if len(self.execution_history) > 10:
                self.execution_history = self.execution_history[-10:]
            
            logger.error(error_msg)
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            self._handle_execution_failure({'error': error_msg})
        
        finally:
            logger.info("=" * 60)
            logger.info("SCHEDULED PIPELINE EXECUTION COMPLETED")
            logger.info("=" * 60)
    
    def _handle_execution_failure(self, result: Dict[str, Any]):
        """Handle pipeline execution failure with retry logic."""
        try:
            self.retry_count += 1
            
            if self.retry_count <= self.max_retry_attempts:
                # Calculate retry delay with backoff
                delay_minutes = self.retry_delay_minutes * (self.retry_backoff_multiplier ** (self.retry_count - 1))
                retry_time = datetime.now() + timedelta(minutes=delay_minutes)
                
                logger.warning(f"Pipeline execution failed (attempt {self.retry_count}/{self.max_retry_attempts})")
                logger.info(f"Scheduling retry in {delay_minutes:.1f} minutes at {retry_time.strftime('%Y-%m-%d %H:%M:%S')}")
                
                # Schedule retry
                self.scheduler.add_job(
                    func=self._execute_pipeline_job,
                    trigger='date',
                    run_date=retry_time,
                    id=f"{self.job_id}_retry_{self.retry_count}",
                    name=f"Pipeline Retry {self.retry_count}",
                    replace_existing=True
                )
            else:
                logger.error(f"Pipeline execution failed after {self.max_retry_attempts} attempts. No more retries.")
                self.retry_count = 0  # Reset for next scheduled execution
                
                # TODO: Integration point for future NotificationEngine
                # This is where we would send failure notifications
                logger.warning("Future enhancement: Send failure notification to NotificationEngine")
                
        except Exception as e:
            logger.error(f"Failed to handle execution failure: {e}")
    
    def _job_executed_listener(self, event):
        """Handle job executed events."""
        logger.debug(f"Job executed: {event.job_id}")
    
    def _job_error_listener(self, event):
        """Handle job error events."""
        logger.error(f"Job error: {event.job_id} - {event.exception}")
    
    def _job_missed_listener(self, event):
        """Handle job missed events."""
        logger.warning(f"Job missed: {event.job_id} - scheduled for {event.scheduled_run_time}")
    
    def _get_day_name(self, day_number: int) -> str:
        """Get day name from day number (0=Monday)."""
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        return days[day_number] if 0 <= day_number <= 6 else f'Day{day_number}'
    
    def set_pipeline_orchestrator(self, pipeline_orchestrator):
        """
        Set the pipeline orchestrator for scheduled execution.
        
        Args:
            pipeline_orchestrator: PipelineOrchestrator instance
        """
        self.pipeline_orchestrator = pipeline_orchestrator
        logger.info("PipelineOrchestrator set for scheduled execution")
    
    def get_execution_history(self) -> List[Dict[str, Any]]:
        """
        Get execution history.
        
        Returns:
            List of execution results
        """
        return self.execution_history.copy()
    
    def force_execution(self) -> Dict[str, Any]:
        """
        Force immediate pipeline execution (manual trigger).
        
        Returns:
            Dict with execution results
        """
        try:
            logger.info("Manual pipeline execution triggered")
            
            if self.pipeline_orchestrator is None:
                raise RuntimeError("PipelineOrchestrator not available for manual execution")
            
            # Execute pipeline in current thread
            result = self.pipeline_orchestrator.run_full_pipeline()
            
            # Record execution
            execution_result = {
                'status': result.get('status', 'unknown'),
                'execution_time': result.get('execution_time', 'unknown'),
                'timestamp': datetime.now().isoformat(),
                'trigger': 'manual',
                'result': result
            }
            
            self.execution_history.append(execution_result)
            if len(self.execution_history) > 10:
                self.execution_history = self.execution_history[-10:]
            
            logger.info(f"Manual pipeline execution completed: {result.get('status', 'unknown')}")
            
            return {
                'status': 'executed',
                'message': 'Manual pipeline execution completed',
                'execution_result': execution_result
            }
            
        except Exception as e:
            error_msg = f"Manual pipeline execution failed: {str(e)}"
            logger.error(error_msg)
            
            return {
                'status': 'error',
                'message': error_msg,
                'error': str(e)
            }


def create_weekly_scheduler(config_path: str = "config/config.ini", 
                          pipeline_orchestrator=None) -> WeeklyScheduler:
    """
    Factory function to create a WeeklyScheduler instance.
    
    Args:
        config_path: Path to configuration file
        pipeline_orchestrator: PipelineOrchestrator instance
        
    Returns:
        WeeklyScheduler instance
    """
    try:
        scheduler = WeeklyScheduler(config_path=config_path, pipeline_orchestrator=pipeline_orchestrator)
        logger.info("WeeklyScheduler created successfully")
        return scheduler
        
    except Exception as e:
        logger.error(f"Failed to create WeeklyScheduler: {e}")
        raise


# Example usage and testing
if __name__ == "__main__":
    """Example usage of WeeklyScheduler."""
    import time
    
    # Create scheduler
    scheduler = create_weekly_scheduler()
    
    # Print status
    status = scheduler.get_job_status()
    print("Scheduler Status:")
    for key, value in status.items():
        print(f"  {key}: {value}")
    
    # Start scheduler
    print("\nStarting scheduler...")
    start_result = scheduler.start_scheduler()
    print(f"Start result: {start_result}")
    
    # Wait a bit
    print("\nWaiting 5 seconds...")
    time.sleep(5)
    
    # Check status again
    status = scheduler.get_job_status()
    print("\nUpdated Status:")
    for key, value in status.items():
        print(f"  {key}: {value}")
    
    # Stop scheduler
    print("\nStopping scheduler...")
    stop_result = scheduler.stop_scheduler()
    print(f"Stop result: {stop_result}")
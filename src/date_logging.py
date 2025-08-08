
"""
Enhanced Date Logging Utilities - SHIOL+ Phase 3
=================================================

Specialized logging utilities for date operations with detailed tracking
and debugging information for date-related operations across the system.
"""

import functools
from datetime import datetime
from typing import Any, Callable, Optional, Union
from loguru import logger
from src.date_utils import DateManager


class DateOperationLogger:
    """
    Enhanced logger specifically for date operations with context tracking.
    """
    
    @staticmethod
    def log_date_operation(operation_name: str, 
                          input_date: Optional[Union[str, datetime]] = None,
                          output_date: Optional[str] = None,
                          additional_context: Optional[dict] = None,
                          level: str = "info") -> None:
        """
        Log a date operation with detailed context.
        
        Args:
            operation_name: Name of the operation being performed
            input_date: Input date for the operation
            output_date: Output date from the operation
            additional_context: Additional context information
            level: Log level (info, debug, warning, error)
        """
        # Use Eastern Time for all timestamps
        current_et = DateManager.get_current_et_time()
        log_data = {
            "operation": operation_name,
            "timestamp": current_et.isoformat(),
            "current_et_time": current_et.isoformat(),
            "server_timezone": "America/New_York"
        }
        
        if input_date:
            if isinstance(input_date, datetime):
                log_data["input_date"] = input_date.isoformat()
                log_data["input_timezone"] = str(input_date.tzinfo) if input_date.tzinfo else "naive"
            else:
                log_data["input_date"] = str(input_date)
        
        if output_date:
            log_data["output_date"] = output_date
            log_data["output_is_drawing_date"] = DateManager.is_valid_drawing_date(output_date)
        
        if additional_context:
            log_data.update(additional_context)
        
        # Format log message
        message = f"Date Operation: {operation_name}"
        if input_date and output_date:
            message += f" | {input_date} -> {output_date}"
        elif input_date:
            message += f" | Input: {input_date}"
        elif output_date:
            message += f" | Output: {output_date}"
        
        # Log with appropriate level
        log_func = getattr(logger, level.lower(), logger.info)
        log_func(f"{message} | Context: {log_data}")
    
    @staticmethod
    def log_date_validation(date_str: str, 
                           is_valid: bool, 
                           validation_type: str = "general",
                           error_details: Optional[str] = None) -> None:
        """
        Log date validation results with detailed information.
        
        Args:
            date_str: Date string being validated
            is_valid: Whether the date is valid
            validation_type: Type of validation (format, drawing_date, etc.)
            error_details: Error details if validation failed
        """
        validation_context = {
            "date_string": date_str,
            "validation_type": validation_type,
            "is_valid": is_valid,
            "timestamp": DateManager.get_current_et_time().isoformat(),
            "server_timezone": "America/New_York"
        }
        
        if error_details:
            validation_context["error_details"] = error_details
        
        level = "debug" if is_valid else "warning"
        status = "PASSED" if is_valid else "FAILED"
        message = f"Date Validation {status}: {date_str} ({validation_type})"
        
        if error_details:
            message += f" - {error_details}"
        
        log_func = getattr(logger, level)
        log_func(f"{message} | Context: {validation_context}")
    
    @staticmethod
    def log_timezone_conversion(original_dt: Union[str, datetime], 
                               converted_dt: datetime,
                               target_timezone: str = "America/New_York") -> None:
        """
        Log timezone conversion operations.
        
        Args:
            original_dt: Original datetime or string
            converted_dt: Converted datetime object
            target_timezone: Target timezone name
        """
        conversion_context = {
            "original_value": str(original_dt),
            "converted_value": converted_dt.isoformat(),
            "target_timezone": target_timezone,
            "original_timezone": str(original_dt.tzinfo) if hasattr(original_dt, 'tzinfo') and original_dt.tzinfo else "unknown",
            "timestamp": DateManager.get_current_et_time().isoformat(),
            "server_timezone": "America/New_York"
        }
        
        message = f"Timezone Conversion: {original_dt} -> {converted_dt.isoformat()} ({target_timezone})"
        logger.debug(f"{message} | Context: {conversion_context}")


def log_date_operation(operation_name: str):
    """
    Decorator to automatically log date operations with input/output tracking.
    
    Args:
        operation_name: Name of the operation to log
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Extract date-related arguments for logging
            input_dates = []
            for arg in args:
                if isinstance(arg, (str, datetime)):
                    input_dates.append(arg)
            
            for key, value in kwargs.items():
                if isinstance(value, (str, datetime)) and ('date' in key.lower() or 'time' in key.lower()):
                    input_dates.append(value)
            
            # Log operation start
            DateOperationLogger.log_date_operation(
                operation_name=f"{operation_name}_START",
                input_date=input_dates[0] if input_dates else None,
                additional_context={
                    "function": func.__name__,
                    "args_count": len(args),
                    "kwargs": list(kwargs.keys())
                },
                level="debug"
            )
            
            try:
                # Execute function
                result = func(*args, **kwargs)
                
                # Log successful completion
                DateOperationLogger.log_date_operation(
                    operation_name=f"{operation_name}_SUCCESS",
                    input_date=input_dates[0] if input_dates else None,
                    output_date=result if isinstance(result, str) and len(result) == 10 else None,
                    additional_context={
                        "function": func.__name__,
                        "result_type": type(result).__name__
                    },
                    level="debug"
                )
                
                return result
                
            except Exception as e:
                # Log error
                DateOperationLogger.log_date_operation(
                    operation_name=f"{operation_name}_ERROR",
                    input_date=input_dates[0] if input_dates else None,
                    additional_context={
                        "function": func.__name__,
                        "error": str(e),
                        "error_type": type(e).__name__
                    },
                    level="error"
                )
                raise
        
        return wrapper
    return decorator


def log_date_calculation(func: Callable) -> Callable:
    """
    Decorator specifically for date calculation functions.
    """
    return log_date_operation("DATE_CALCULATION")(func)


def log_date_validation(func: Callable) -> Callable:
    """
    Decorator specifically for date validation functions.
    """
    return log_date_operation("DATE_VALIDATION")(func)


def log_timezone_operation(func: Callable) -> Callable:
    """
    Decorator specifically for timezone conversion functions.
    """
    return log_date_operation("TIMEZONE_OPERATION")(func)


# Initialize date logging system
logger.info("Enhanced date logging utilities initialized")
logger.debug("Available decorators: @log_date_operation, @log_date_calculation, @log_date_validation, @log_timezone_operation")

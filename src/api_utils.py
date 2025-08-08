
"""
SHIOL+ API Utilities
===================

Utility functions for API operations including type conversion and common helpers.
"""

import numpy as np
from typing import Any, Dict, List, Union
from loguru import logger


def convert_numpy_types(obj: Any) -> Any:
    """Convert numpy types to native Python types for JSON serialization."""
    if obj is None:
        return None
    elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    elif isinstance(obj, dict):
        return {str(key): convert_numpy_types(value) for key, value in obj.items()}
    elif hasattr(obj, 'item'):  # Handle numpy scalar types
        try:
            return obj.item()
        except (AttributeError, ValueError):
            return str(obj)
    elif isinstance(obj, (int, float, str, bool)):
        return obj
    else:
        # For any other type, try to convert to string as fallback
        try:
            return str(obj)
        except:
            return None


def format_prediction_response(prediction_data: Dict[str, Any], method: str = "smart_ai") -> Dict[str, Any]:
    """Format prediction data for API response."""
    return {
        "prediction": convert_numpy_types(prediction_data.get("numbers", [])) + [convert_numpy_types(prediction_data.get("powerball", 0))],
        "method": method,
        "score_total": convert_numpy_types(prediction_data.get("score_total", 0.0)),
        "score_details": convert_numpy_types(prediction_data.get("score_details", {})),
        "model_version": prediction_data.get("model_version", "1.0.0"),
        "dataset_hash": prediction_data.get("dataset_hash", ""),
        "timestamp": prediction_data.get("timestamp", ""),
        "prediction_id": prediction_data.get("prediction_id")
    }


def safe_database_operation(operation_func, error_message: str = "Database operation failed"):
    """Safely execute database operations with error handling."""
    try:
        return operation_func()
    except Exception as e:
        logger.error(f"{error_message}: {e}")
        raise Exception(f"{error_message}: {str(e)}")

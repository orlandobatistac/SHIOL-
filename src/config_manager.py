
"""
SHIOL+ v6.0 Configuration Manager - HYBRID SYSTEM
Handles system configuration with database-first approach and file fallback
"""

import os
import json
import configparser
from typing import Dict, Any, Optional
from datetime import datetime
import psutil
import logging
from src.database import (
    load_config_from_db, 
    save_config_to_db, 
    get_config_value,
    migrate_config_from_file,
    is_config_initialized
)

logger = logging.getLogger(__name__)

class ConfigurationManager:
    def __init__(self, config_file="config/config.ini"):
        self.config_file = config_file
        self.config = {}
        self.initialize_hybrid_system()
        
    def initialize_hybrid_system(self):
        """Initialize the hybrid configuration system"""
        try:
            # Check if configuration is already in database
            if not is_config_initialized():
                logger.info("Initializing hybrid configuration system...")
                migrate_config_from_file()
            
            # Load configuration from database
            self.load_configuration()
            
        except Exception as e:
            logger.error(f"Error initializing hybrid configuration system: {e}")
            self.config = self._get_default_config()
        
    def load_configuration(self) -> Dict[str, Any]:
        """Load configuration from database with file fallback"""
        try:
            self.config = load_config_from_db()
            return self.config
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            return self._get_default_config()
    
    def save_configuration(self, config_data: Dict[str, Any]) -> bool:
        """Save configuration to database (dashboard calls this)"""
        try:
            success = save_config_to_db(config_data)
            if success:
                self.config = config_data
                logger.info("Configuration saved successfully via hybrid system")
            return success
        except Exception as e:
            logger.error(f"Error saving configuration via hybrid system: {e}")
            return False
    
    def get_config_value(self, section: str, key: str, default: Any = None) -> Any:
        """Get specific configuration value with fallback"""
        return get_config_value(section, key, default)
    
    def get_configuration_profiles(self) -> Dict[str, Dict[str, Any]]:
        """Get predefined configuration profiles"""
        return {
            "conservative": {
                "pipeline": {
                    "execution_days": {"monday": True, "wednesday": True, "saturday": False},
                    "execution_time": "02:00",
                    "auto_execution": True
                },
                "predictions": {
                    "count": 50,
                    "method": "deterministic"
                },
                "weights": {
                    "probability": 50,
                    "diversity": 20,
                    "historical": 20,
                    "risk": 10
                }
            },
            "aggressive": {
                "pipeline": {
                    "execution_days": {"monday": True, "wednesday": True, "saturday": True},
                    "execution_time": "01:00",
                    "auto_execution": True
                },
                "predictions": {
                    "count": 500,
                    "method": "ensemble"
                },
                "weights": {
                    "probability": 30,
                    "diversity": 35,
                    "historical": 20,
                    "risk": 15
                }
            },
            "balanced": {
                "pipeline": {
                    "execution_days": {"monday": True, "wednesday": True, "saturday": True},
                    "execution_time": "02:00",
                    "auto_execution": True
                },
                "predictions": {
                    "count": 100,
                    "method": "smart_ai"
                },
                "weights": {
                    "probability": 40,
                    "diversity": 25,
                    "historical": 20,
                    "risk": 15
                }
            }
        }
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get current system statistics"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            return {
                "cpu_usage": round(cpu_percent, 1),
                "memory_usage": round(memory.percent, 1),
                "disk_usage": round((disk.used / disk.total) * 100, 1),
                "memory_total": round(memory.total / (1024**3), 2),  # GB
                "disk_total": round(disk.total / (1024**3), 2),  # GB
                "timestamp": datetime.now().isoformat(),
                "config_source": "database" if is_config_initialized() else "file_fallback"
            }
        except Exception as e:
            logger.error(f"Error getting system stats: {e}")
            return {
                "cpu_usage": 0,
                "memory_usage": 0,
                "disk_usage": 0,
                "memory_total": 0,
                "disk_total": 0,
                "timestamp": datetime.now().isoformat(),
                "config_source": "unknown"
            }
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            "pipeline": {
                "execution_days_monday": "True",
                "execution_days_wednesday": "True",
                "execution_days_saturday": "True",
                "execution_time": "02:00",
                "timezone": "America/New_York",
                "auto_execution": "True"
            },
            "predictions": {
                "count": "100",
                "method": "smart_ai"
            },
            "weights": {
                "probability": "40",
                "diversity": "25",
                "historical": "20",
                "risk": "15"
            },
            "paths": {
                "db_file": "data/shiolplus.db",
                "model_file": "models/shiolplus.pkl"
            }
        }

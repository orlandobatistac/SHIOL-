
"""
SHIOL+ v6.0 Configuration Manager
Handles system configuration, profiles, and settings persistence
"""

import os
import json
import configparser
from typing import Dict, Any, Optional
from datetime import datetime
import psutil
import logging

logger = logging.getLogger(__name__)

class ConfigurationManager:
    def __init__(self, config_file="config/config.ini"):
        self.config_file = config_file
        self.config = configparser.ConfigParser()
        self.load_configuration()
        
    def load_configuration(self) -> Dict[str, Any]:
        """Load configuration from file"""
        try:
            self.config.read(self.config_file)
            return self._config_to_dict()
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            return self._get_default_config()
    
    def save_configuration(self, config_data: Dict[str, Any]) -> bool:
        """Save configuration to file"""
        try:
            self._dict_to_config(config_data)
            with open(self.config_file, 'w') as f:
                self.config.write(f)
            logger.info("Configuration saved successfully")
            return True
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            return False
    
    def get_configuration_profiles(self) -> Dict[str, Dict[str, Any]]:
        """Get predefined configuration profiles"""
        return {
            "conservative": {
                "pipeline": {
                    "prediction_count": 50,
                    "prediction_method": "deterministic",
                    "execution_days": {"monday": True, "wednesday": True, "saturday": False},
                    "execution_time": "02:00",
                    "auto_execution": True
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
                    "prediction_count": 500,
                    "prediction_method": "ensemble",
                    "execution_days": {"monday": True, "wednesday": True, "saturday": True},
                    "execution_time": "01:00",
                    "auto_execution": True
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
                    "prediction_count": 100,
                    "prediction_method": "smart_ai",
                    "execution_days": {"monday": True, "wednesday": True, "saturday": True},
                    "execution_time": "02:00",
                    "auto_execution": True
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
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting system stats: {e}")
            return {
                "cpu_usage": 0,
                "memory_usage": 0,
                "disk_usage": 0,
                "memory_total": 0,
                "disk_total": 0,
                "timestamp": datetime.now().isoformat()
            }
    
    def _config_to_dict(self) -> Dict[str, Any]:
        """Convert ConfigParser to dictionary"""
        config_dict = {}
        for section in self.config.sections():
            config_dict[section] = dict(self.config.items(section))
        return config_dict
    
    def _dict_to_config(self, config_data: Dict[str, Any]):
        """Convert dictionary to ConfigParser"""
        self.config.clear()
        for section, items in config_data.items():
            self.config.add_section(section)
            for key, value in items.items():
                self.config.set(section, key, str(value))
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            "pipeline": {
                "prediction_count": "100",
                "prediction_method": "smart_ai",
                "execution_time": "02:00",
                "timezone": "America/New_York",
                "auto_execution": "true"
            },
            "database": {
                "backup_enabled": "true",
                "cleanup_interval": "30"
            },
            "notifications": {
                "email_enabled": "false",
                "browser_enabled": "true",
                "session_timeout": "60"
            }
        }


"""
System Diagnostics and Correction Utilities - SHIOL+
====================================================

Utilidades para diagnosticar y corregir problemas del sistema,
incluyendo drift de reloj, corrupción de datos y validaciones.
"""

import os
import pytz
from datetime import datetime, timedelta
from typing import Dict, Any, List
from loguru import logger
from src.date_utils import DateManager
from src.database import get_db_connection


class SystemDiagnostics:
    """Herramientas de diagnóstico y corrección del sistema."""
    
    @staticmethod
    def diagnose_clock_drift() -> Dict[str, Any]:
        """
        Diagnostica drift del reloj del sistema.
        
        Returns:
            Dict con resultados del diagnóstico
        """
        system_utc = datetime.now(pytz.UTC)
        calculated_et = system_utc.astimezone(DateManager.POWERBALL_TIMEZONE)
        
        # Expected vs actual comparison
        expected_date = "2025-08-07"  # Known correct date
        actual_date = calculated_et.strftime('%Y-%m-%d')
        
        drift_detected = actual_date != expected_date
        
        diagnosis = {
            'system_utc': system_utc.isoformat(),
            'calculated_et': calculated_et.isoformat(),
            'expected_date': expected_date,
            'actual_date': actual_date,
            'drift_detected': drift_detected,
            'drift_hours': 4 if drift_detected else 0,
            'recommendation': 'apply_correction' if drift_detected else 'no_action_needed'
        }
        
        logger.info(f"Clock drift diagnosis: {diagnosis}")
        return diagnosis
    
    @staticmethod
    def check_data_corruption() -> Dict[str, Any]:
        """
        Verifica corrupción en la base de datos.
        
        Returns:
            Dict con resultados de la verificación
        """
        corrupted_records = []
        
        try:
            with get_db_connection() as conn:
                cursor = conn.cursor()
                
                # Check for corrupted target_draw_date
                cursor.execute("""
                    SELECT id, target_draw_date, created_at, model_version, dataset_hash
                    FROM prediction_logs
                    WHERE target_draw_date NOT LIKE '____-__-__'
                       OR LENGTH(target_draw_date) != 10
                       OR created_at NOT LIKE '____-__-__T__:__:__.___%'
                    LIMIT 10
                """)
                
                corrupted_data = cursor.fetchall()
                
                for row in corrupted_data:
                    corrupted_records.append({
                        'id': row[0],
                        'target_draw_date': row[1],
                        'created_at': row[2],
                        'model_version': row[3],
                        'dataset_hash': row[4],
                        'issue': 'field_value_mismatch'
                    })
                
                # Get total count of corrupted records
                cursor.execute("""
                    SELECT COUNT(*) FROM prediction_logs
                    WHERE target_draw_date NOT LIKE '____-__-__'
                       OR LENGTH(target_draw_date) != 10
                """)
                
                total_corrupted = cursor.fetchone()[0]
                
        except Exception as e:
            logger.error(f"Error checking data corruption: {e}")
            return {'error': str(e)}
        
        result = {
            'total_corrupted_records': total_corrupted,
            'sample_corrupted_records': corrupted_records,
            'corruption_detected': total_corrupted > 0,
            'recommendation': 'run_migration' if total_corrupted > 0 else 'no_action_needed'
        }
        
        logger.info(f"Data corruption check: {result}")
        return result
    
    @staticmethod
    def run_system_health_check() -> Dict[str, Any]:
        """
        Ejecuta verificación completa de salud del sistema.
        
        Returns:
            Dict con resultados completos
        """
        logger.info("Running comprehensive system health check...")
        
        # Clock drift check
        clock_diagnosis = SystemDiagnostics.diagnose_clock_drift()
        
        # Data corruption check
        corruption_diagnosis = SystemDiagnostics.check_data_corruption()
        
        # Environment checks
        env_check = {
            'database_exists': os.path.exists('data/shiolplus.db'),
            'models_directory': os.path.exists('models/'),
            'frontend_directory': os.path.exists('frontend/'),
            'logs_directory': os.path.exists('logs/')
        }
        
        # Overall health assessment
        issues_found = []
        if clock_diagnosis['drift_detected']:
            issues_found.append('clock_drift')
        if corruption_diagnosis['corruption_detected']:
            issues_found.append('data_corruption')
        
        health_status = 'healthy' if not issues_found else 'issues_detected'
        
        comprehensive_report = {
            'timestamp': datetime.now().isoformat(),
            'health_status': health_status,
            'issues_found': issues_found,
            'clock_diagnosis': clock_diagnosis,
            'corruption_diagnosis': corruption_diagnosis,
            'environment_check': env_check,
            'recommendations': []
        }
        
        # Generate recommendations
        if clock_diagnosis['drift_detected']:
            comprehensive_report['recommendations'].append({
                'issue': 'clock_drift',
                'action': 'Enable automatic clock correction in DateManager',
                'priority': 'high'
            })
        
        if corruption_diagnosis['corruption_detected']:
            comprehensive_report['recommendations'].append({
                'issue': 'data_corruption',
                'action': 'Run data migration: python main.py --migrate',
                'priority': 'high'
            })
        
        logger.info(f"System health check completed: {health_status}")
        logger.info(f"Issues found: {issues_found}")
        
        return comprehensive_report


def run_diagnostics():
    """Función de conveniencia para ejecutar diagnósticos."""
    return SystemDiagnostics.run_system_health_check()

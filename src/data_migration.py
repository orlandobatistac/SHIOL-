
"""
Data Migration Script - SHIOL+ Phase 2
=======================================

Script para limpiar y corregir datos corruptos en la base de datos,
específicamente problemas con fechas incorrectas en predictions_log.
"""

import sqlite3
import pandas as pd
from datetime import datetime, timedelta
from loguru import logger
from typing import List, Dict, Any
import pytz

from src.database import get_db_connection, calculate_next_drawing_date


class DataMigrationManager:
    """
    Administrador para ejecutar migraciones de datos y correcciones.
    """
    
    def __init__(self):
        """Inicializa el administrador de migración."""
        self.migration_log = []
        logger.info("DataMigrationManager initialized")
    
    def execute_date_correction_migration(self) -> Dict[str, Any]:
        """
        Ejecuta la migración completa para corregir fechas incorrectas.
        
        Returns:
            Dict con resultados de la migración
        """
        logger.info("Starting date correction migration...")
        
        migration_results = {
            'start_time': datetime.now().isoformat(),
            'corrupted_records_found': 0,
            'records_corrected': 0,
            'validation_failures': 0,
            'errors': [],
            'migration_steps': []
        }
        
        try:
            # Paso 1: Identificar predicciones con fechas incorrectas
            corrupted_records = self._identify_corrupted_date_records()
            migration_results['corrupted_records_found'] = len(corrupted_records)
            migration_results['migration_steps'].append("✓ Identified corrupted records")
            
            if len(corrupted_records) == 0:
                logger.info("No corrupted date records found")
                migration_results['status'] = 'no_action_needed'
                return migration_results
            
            logger.info(f"Found {len(corrupted_records)} records with corrupted dates")
            
            # Paso 2: Recalcular target_draw_date basado en created_at
            corrected_records = self._recalculate_target_draw_dates(corrupted_records)
            migration_results['migration_steps'].append("✓ Recalculated target draw dates")
            
            # Paso 3: Aplicar correcciones a la base de datos
            correction_results = self._apply_date_corrections(corrected_records)
            migration_results['records_corrected'] = correction_results['records_updated']
            migration_results['validation_failures'] = correction_results['validation_failures']
            migration_results['migration_steps'].append("✓ Applied corrections to database")
            
            # Paso 4: Validar integridad después de migración
            validation_results = self._validate_date_integrity()
            migration_results['migration_steps'].append("✓ Validated date integrity")
            migration_results['final_validation'] = validation_results
            
            migration_results['status'] = 'completed'
            migration_results['end_time'] = datetime.now().isoformat()
            
            logger.info(f"Date correction migration completed successfully")
            logger.info(f"Records corrected: {migration_results['records_corrected']}")
            
            return migration_results
            
        except Exception as e:
            error_msg = f"Migration failed: {str(e)}"
            logger.error(error_msg)
            migration_results['status'] = 'failed'
            migration_results['errors'].append(error_msg)
            migration_results['end_time'] = datetime.now().isoformat()
            return migration_results
    
    def _identify_corrupted_date_records(self) -> List[Dict]:
        """
        Identifica registros con fechas incorrectas o corruptas.
        
        Returns:
            Lista de registros corruptos
        """
        corrupted_records = []
        
        try:
            with get_db_connection() as conn:
                cursor = conn.cursor()
                
                # Buscar registros donde target_draw_date no es una fecha válida
                cursor.execute("""
                    SELECT id, timestamp, target_draw_date, created_at, model_version, dataset_hash
                    FROM predictions_log
                    WHERE target_draw_date IS NULL 
                       OR target_draw_date = ''
                       OR target_draw_date = model_version
                       OR target_draw_date = dataset_hash
                       OR LENGTH(target_draw_date) != 10
                       OR target_draw_date NOT LIKE '____-__-__'
                    ORDER BY id
                """)
                
                results = cursor.fetchall()
                
                for row in results:
                    record = {
                        'id': row[0],
                        'timestamp': row[1],
                        'target_draw_date': row[2],
                        'created_at': row[3],
                        'model_version': row[4],
                        'dataset_hash': row[5]
                    }
                    corrupted_records.append(record)
                
                logger.info(f"Identified {len(corrupted_records)} corrupted date records")
                
                # Log algunos ejemplos para debugging
                if corrupted_records:
                    logger.debug("Sample corrupted records:")
                    for i, record in enumerate(corrupted_records[:3]):
                        logger.debug(f"  {i+1}. ID {record['id']}: target_draw_date='{record['target_draw_date']}'")
                
                return corrupted_records
                
        except Exception as e:
            logger.error(f"Error identifying corrupted records: {e}")
            return []
    
    def _recalculate_target_draw_dates(self, corrupted_records: List[Dict]) -> List[Dict]:
        """
        Recalcula target_draw_date basado en created_at para registros corruptos.
        
        Args:
            corrupted_records: Lista de registros corruptos
            
        Returns:
            Lista de registros con fechas corregidas
        """
        corrected_records = []
        
        # Timezone de Eastern Time para cálculos de Powerball
        et_tz = pytz.timezone('America/New_York')
        
        for record in corrupted_records:
            try:
                # Extraer fecha de created_at o timestamp
                base_date = record.get('created_at') or record.get('timestamp')
                
                if not base_date:
                    logger.warning(f"Record ID {record['id']} has no valid timestamp")
                    continue
                
                # Parsear fecha base
                if isinstance(base_date, str):
                    try:
                        # Intentar diferentes formatos de fecha
                        if 'T' in base_date:
                            parsed_date = datetime.fromisoformat(base_date.replace('Z', '+00:00'))
                        else:
                            parsed_date = datetime.strptime(base_date[:19], '%Y-%m-%d %H:%M:%S')
                    except ValueError:
                        parsed_date = datetime.strptime(base_date[:10], '%Y-%m-%d')
                else:
                    parsed_date = base_date
                
                # Convertir a ET si no tiene timezone
                if parsed_date.tzinfo is None:
                    parsed_date = et_tz.localize(parsed_date)
                else:
                    parsed_date = parsed_date.astimezone(et_tz)
                
                # Calcular próxima fecha de sorteo desde la fecha de creación
                target_draw_date = self._calculate_next_drawing_from_date(parsed_date)
                
                corrected_record = record.copy()
                corrected_record['corrected_target_draw_date'] = target_draw_date
                corrected_records.append(corrected_record)
                
                logger.debug(f"Record ID {record['id']}: {base_date} -> {target_draw_date}")
                
            except Exception as e:
                logger.error(f"Error calculating date for record ID {record['id']}: {e}")
                continue
        
        logger.info(f"Successfully calculated corrected dates for {len(corrected_records)} records")
        return corrected_records
    
    def _calculate_next_drawing_from_date(self, reference_date: datetime) -> str:
        """
        Calcula la próxima fecha de sorteo desde una fecha de referencia.
        
        Args:
            reference_date: Fecha de referencia (con timezone)
            
        Returns:
            Fecha del próximo sorteo en formato YYYY-MM-DD
        """
        # Días de sorteo: Lunes=0, Miércoles=2, Sábado=5
        drawing_days = [0, 2, 5]
        current_weekday = reference_date.weekday()
        
        # Si es día de sorteo y es antes de las 11 PM ET, el sorteo es ese día
        if current_weekday in drawing_days and reference_date.hour < 23:
            return reference_date.strftime('%Y-%m-%d')
        
        # Encontrar el próximo día de sorteo
        for i in range(1, 8):
            next_date = reference_date + timedelta(days=i)
            if next_date.weekday() in drawing_days:
                return next_date.strftime('%Y-%m-%d')
        
        # Fallback (no debería ocurrir)
        return (reference_date + timedelta(days=1)).strftime('%Y-%m-%d')
    
    def _apply_date_corrections(self, corrected_records: List[Dict]) -> Dict[str, int]:
        """
        Aplica las correcciones de fecha a la base de datos.
        
        Args:
            corrected_records: Lista de registros con fechas corregidas
            
        Returns:
            Dict con estadísticas de actualización
        """
        results = {
            'records_updated': 0,
            'validation_failures': 0
        }
        
        try:
            with get_db_connection() as conn:
                cursor = conn.cursor()
                
                for record in corrected_records:
                    try:
                        # Validar fecha antes de aplicar
                        corrected_date = record['corrected_target_draw_date']
                        if not self._validate_date_format(corrected_date):
                            logger.warning(f"Invalid corrected date format: {corrected_date}")
                            results['validation_failures'] += 1
                            continue
                        
                        # Aplicar corrección
                        cursor.execute("""
                            UPDATE predictions_log 
                            SET target_draw_date = ?
                            WHERE id = ?
                        """, (corrected_date, record['id']))
                        
                        if cursor.rowcount > 0:
                            results['records_updated'] += 1
                            logger.debug(f"Updated record ID {record['id']} with date {corrected_date}")
                        
                    except Exception as e:
                        logger.error(f"Error updating record ID {record['id']}: {e}")
                        results['validation_failures'] += 1
                        continue
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error applying date corrections: {e}")
            raise
        
        logger.info(f"Applied corrections: {results['records_updated']} updated, {results['validation_failures']} failed")
        return results
    
    def _validate_date_format(self, date_str: str) -> bool:
        """
        Valida que una fecha tenga el formato correcto YYYY-MM-DD.
        
        Args:
            date_str: String de fecha a validar
            
        Returns:
            True si la fecha es válida
        """
        if not isinstance(date_str, str) or len(date_str) != 10:
            return False
        
        try:
            datetime.strptime(date_str, '%Y-%m-%d')
            return True
        except ValueError:
            return False
    
    def _validate_date_integrity(self) -> Dict[str, Any]:
        """
        Valida la integridad de fechas después de la migración.
        
        Returns:
            Dict con resultados de validación
        """
        validation_results = {
            'total_records': 0,
            'valid_target_dates': 0,
            'invalid_target_dates': 0,
            'null_target_dates': 0,
            'date_format_errors': 0
        }
        
        try:
            with get_db_connection() as conn:
                cursor = conn.cursor()
                
                # Contar total de registros
                cursor.execute("SELECT COUNT(*) FROM predictions_log")
                validation_results['total_records'] = cursor.fetchone()[0]
                
                # Contar fechas válidas
                cursor.execute("""
                    SELECT COUNT(*) FROM predictions_log 
                    WHERE target_draw_date IS NOT NULL 
                      AND LENGTH(target_draw_date) = 10
                      AND target_draw_date LIKE '____-__-__'
                """)
                validation_results['valid_target_dates'] = cursor.fetchone()[0]
                
                # Contar fechas nulas
                cursor.execute("SELECT COUNT(*) FROM predictions_log WHERE target_draw_date IS NULL")
                validation_results['null_target_dates'] = cursor.fetchone()[0]
                
                # Calcular fechas inválidas
                validation_results['invalid_target_dates'] = (
                    validation_results['total_records'] - 
                    validation_results['valid_target_dates'] - 
                    validation_results['null_target_dates']
                )
                
                # Calcular porcentaje de éxito
                if validation_results['total_records'] > 0:
                    success_rate = (validation_results['valid_target_dates'] / 
                                  validation_results['total_records'] * 100)
                    validation_results['success_rate'] = f"{success_rate:.1f}%"
                else:
                    validation_results['success_rate'] = "0%"
                
                logger.info(f"Date integrity validation completed: {validation_results['success_rate']} success rate")
                
        except Exception as e:
            logger.error(f"Error during date integrity validation: {e}")
            validation_results['error'] = str(e)
        
        return validation_results


def run_date_correction_migration() -> Dict[str, Any]:
    """
    Función principal para ejecutar la migración de corrección de fechas.
    
    Returns:
        Dict con resultados de la migración
    """
    logger.info("=" * 60)
    logger.info("INICIANDO MIGRACIÓN DE CORRECCIÓN DE FECHAS - FASE 2")
    logger.info("=" * 60)
    
    migration_manager = DataMigrationManager()
    results = migration_manager.execute_date_correction_migration()
    
    logger.info("=" * 60)
    logger.info("MIGRACIÓN DE CORRECCIÓN DE FECHAS COMPLETADA")
    logger.info(f"Estado: {results.get('status', 'unknown')}")
    logger.info(f"Registros corruptos encontrados: {results.get('corrupted_records_found', 0)}")
    logger.info(f"Registros corregidos: {results.get('records_corrected', 0)}")
    logger.info("=" * 60)
    
    return results


if __name__ == "__main__":
    # Ejecutar migración si se ejecuta directamente
    results = run_date_correction_migration()
    
    if results.get('status') == 'completed':
        print("✓ Migración completada exitosamente")
    elif results.get('status') == 'no_action_needed':
        print("✓ No se encontraron datos corruptos")
    else:
        print("✗ Migración falló")
        if results.get('errors'):
            for error in results['errors']:
                print(f"  Error: {error}")

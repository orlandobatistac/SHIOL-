import os
import csv
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from loguru import logger

from src.database import get_db_connection
from src.evaluator import Evaluator


class BasicValidator:
    """
    Validador básico para predicciones SHIOL+ Phase 2A MVP.
    
    Valida predicciones contra resultados reales de sorteos y calcula métricas de precisión.
    """
    
    def __init__(self):
        """Inicializa el validador básico con conexión a base de datos y evaluador."""
        self.evaluator = Evaluator()
        logger.info("BasicValidator initialized for Phase 2A MVP")
    
    def basic_validate_predictions(self) -> str:
        """
        Función principal que valida todas las predicciones contra sorteos reales.
        
        Returns:
            str: Ruta del archivo CSV generado con los resultados
        """
        logger.info("Starting basic validation of all predictions...")
        
        try:
            # Obtener todas las predicciones y sorteos
            predictions = self._get_all_predictions()
            draws = self._get_all_draws()
            
            if not predictions:
                logger.warning("No predictions found in database")
                return None
            
            if not draws:
                logger.warning("No draws found in database")
                return None
            
            logger.info(f"Found {len(predictions)} predictions and {len(draws)} draws")
            
            # Validar cada predicción
            validation_results = []
            for prediction in predictions:
                result = self._validate_single_prediction(prediction, draws)
                if result:
                    validation_results.append(result)
            
            logger.info(f"Successfully validated {len(validation_results)} predictions")
            
            # Exportar resultados a CSV
            csv_path = self._export_to_csv(validation_results)
            
            logger.info(f"Validation complete. Results saved to: {csv_path}")
            return csv_path
            
        except Exception as e:
            logger.error(f"Error during basic validation: {e}")
            raise
    
    def _get_all_predictions(self) -> List[Dict]:
        """
        Obtiene todas las predicciones de la tabla predictions_log.
        
        Returns:
            List[Dict]: Lista de predicciones con sus datos
        """
        try:
            with get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, timestamp, n1, n2, n3, n4, n5, powerball, 
                           score_total, model_version, dataset_hash
                    FROM predictions_log
                    ORDER BY timestamp
                """)
                
                columns = [description[0] for description in cursor.description]
                predictions = []
                
                for row in cursor.fetchall():
                    prediction = dict(zip(columns, row))
                    # Extraer fecha de timestamp para matching
                    prediction['prediction_date'] = self._extract_date_from_timestamp(prediction['timestamp'])
                    predictions.append(prediction)
                
                logger.info(f"Retrieved {len(predictions)} predictions from database")
                return predictions
                
        except Exception as e:
            logger.error(f"Error retrieving predictions: {e}")
            return []
    
    def _get_all_draws(self) -> Dict[str, Dict]:
        """
        Obtiene todos los sorteos de la tabla powerball_draws.
        
        Returns:
            Dict[str, Dict]: Diccionario con fecha como clave y datos del sorteo como valor
        """
        try:
            with get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT draw_date, n1, n2, n3, n4, n5, pb
                    FROM powerball_draws
                    ORDER BY draw_date
                """)
                
                draws = {}
                for row in cursor.fetchall():
                    draw_date = row[0]
                    draws[draw_date] = {
                        'draw_date': draw_date,
                        'n1': row[1], 'n2': row[2], 'n3': row[3], 
                        'n4': row[4], 'n5': row[5], 'pb': row[6]
                    }
                
                logger.info(f"Retrieved {len(draws)} draws from database")
                return draws
                
        except Exception as e:
            logger.error(f"Error retrieving draws: {e}")
            return {}
    
    def _extract_date_from_timestamp(self, timestamp: str) -> str:
        """
        Extrae la fecha (YYYY-MM-DD) de un timestamp.
        
        Args:
            timestamp: Timestamp en formato ISO
            
        Returns:
            str: Fecha en formato YYYY-MM-DD
        """
        try:
            # Manejar diferentes formatos de timestamp
            if 'T' in timestamp:
                date_part = timestamp.split('T')[0]
            else:
                date_part = timestamp.split(' ')[0]
            
            return date_part
        except Exception as e:
            logger.warning(f"Error extracting date from timestamp {timestamp}: {e}")
            return timestamp[:10]  # Fallback: tomar primeros 10 caracteres
    
    def _validate_single_prediction(self, prediction: Dict, draws: Dict[str, Dict]) -> Optional[Dict]:
        """
        Valida una predicción individual contra el sorteo correspondiente.
        
        Args:
            prediction: Datos de la predicción
            draws: Diccionario de todos los sorteos
            
        Returns:
            Optional[Dict]: Resultado de validación o None si no hay sorteo correspondiente
        """
        prediction_date = prediction['prediction_date']
        
        # Buscar sorteo correspondiente por fecha
        matching_draw = draws.get(prediction_date)
        if not matching_draw:
            logger.debug(f"No matching draw found for prediction date {prediction_date}")
            return None
        
        # Extraer números predichos y reales
        predicted_numbers = [prediction['n1'], prediction['n2'], prediction['n3'], 
                           prediction['n4'], prediction['n5']]
        predicted_powerball = prediction['powerball']
        
        actual_numbers = [matching_draw['n1'], matching_draw['n2'], matching_draw['n3'],
                         matching_draw['n4'], matching_draw['n5']]
        actual_powerball = matching_draw['pb']
        
        # Calcular coincidencias
        match_main, match_pb = self._calculate_matches(
            predicted_numbers, predicted_powerball,
            actual_numbers, actual_powerball
        )
        
        # Determinar categoría de premio usando lógica existente del evaluador
        prize_category = self.evaluator._get_prize_tier(match_main, match_pb)
        
        # Determinar etiqueta de resultado
        result_label = self._get_result_label(match_main, match_pb, prize_category)
        
        # Crear resultado de validación
        validation_result = {
            'prediction_date': prediction_date,
            'numbers': f"{predicted_numbers[0]}-{predicted_numbers[1]}-{predicted_numbers[2]}-{predicted_numbers[3]}-{predicted_numbers[4]}",
            'powerball': predicted_powerball,
            'draw_numbers': f"{actual_numbers[0]}-{actual_numbers[1]}-{actual_numbers[2]}-{actual_numbers[3]}-{actual_numbers[4]}",
            'draw_powerball': actual_powerball,
            'match_main': match_main,
            'match_pb': match_pb,
            'prize_category': prize_category,
            'result_label': result_label,
            # Campos adicionales para análisis
            'prediction_score': prediction.get('score_total', 0.0),
            'model_version': prediction.get('model_version', 'unknown'),
            'dataset_hash': prediction.get('dataset_hash', 'unknown')
        }
        
        return validation_result
    
    def _calculate_matches(self, predicted_numbers: List[int], predicted_pb: int,
                          actual_numbers: List[int], actual_pb: int) -> Tuple[int, int]:
        """
        Calcula las coincidencias entre números predichos y reales.
        
        Args:
            predicted_numbers: Lista de números principales predichos
            predicted_pb: Powerball predicho
            actual_numbers: Lista de números principales reales
            actual_pb: Powerball real
            
        Returns:
            Tuple[int, int]: (coincidencias_principales, coincidencia_powerball)
        """
        # Convertir a sets para comparación eficiente
        predicted_set = set(predicted_numbers)
        actual_set = set(actual_numbers)
        
        # Calcular coincidencias principales
        match_main = len(predicted_set.intersection(actual_set))
        
        # Calcular coincidencia powerball
        match_pb = 1 if predicted_pb == actual_pb else 0
        
        return match_main, match_pb
    
    def _get_result_label(self, match_main: int, match_pb: int, prize_category: str) -> str:
        """
        Genera una etiqueta descriptiva del resultado.
        
        Args:
            match_main: Número de coincidencias principales
            match_pb: Coincidencia powerball (0 o 1)
            prize_category: Categoría de premio
            
        Returns:
            str: Etiqueta descriptiva del resultado
        """
        if prize_category == "Non-winning":
            return f"No Prize ({match_main} main + {'PB' if match_pb else 'no PB'})"
        else:
            return f"Winner: {prize_category}"
    
    def _export_to_csv(self, validation_results: List[Dict]) -> str:
        """
        Exporta los resultados de validación a un archivo CSV.
        
        Args:
            validation_results: Lista de resultados de validación
            
        Returns:
            str: Ruta del archivo CSV generado
        """
        # Crear directorio si no existe
        os.makedirs('data/validations', exist_ok=True)
        
        # Generar nombre de archivo con timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = f"data/validations/validation_results_{timestamp}.csv"
        
        # Definir campos del CSV según especificación
        fieldnames = [
            'prediction_date', 'numbers', 'powerball', 'draw_numbers', 'draw_powerball',
            'match_main', 'match_pb', 'prize_category', 'result_label'
        ]
        
        try:
            with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                for result in validation_results:
                    # Filtrar solo los campos requeridos para el CSV
                    csv_row = {field: result.get(field, '') for field in fieldnames}
                    writer.writerow(csv_row)
            
            logger.info(f"Validation results exported to {csv_path}")
            return csv_path
            
        except Exception as e:
            logger.error(f"Error exporting to CSV: {e}")
            raise
    
    def get_validation_summary(self, validation_results: List[Dict]) -> Dict:
        """
        Genera un resumen estadístico de los resultados de validación.
        
        Args:
            validation_results: Lista de resultados de validación
            
        Returns:
            Dict: Resumen estadístico
        """
        if not validation_results:
            return {}
        
        total_predictions = len(validation_results)
        winning_predictions = sum(1 for r in validation_results if r['prize_category'] != 'Non-winning')
        
        # Contar por categorías de premio
        prize_categories = {}
        match_distribution = {}
        
        for result in validation_results:
            category = result['prize_category']
            prize_categories[category] = prize_categories.get(category, 0) + 1
            
            match_key = f"{result['match_main']}+{result['match_pb']}"
            match_distribution[match_key] = match_distribution.get(match_key, 0) + 1
        
        summary = {
            'total_predictions': total_predictions,
            'winning_predictions': winning_predictions,
            'win_rate': winning_predictions / total_predictions if total_predictions > 0 else 0,
            'prize_categories': prize_categories,
            'match_distribution': match_distribution,
            'validation_date': datetime.now().isoformat()
        }
        
        return summary


def basic_validate_predictions() -> str:
    """
    Función principal para validación básica de predicciones (Phase 2A MVP).
    
    Esta función:
    1. Lee todas las predicciones de predictions_log
    2. Encuentra sorteos correspondientes en powerball_draws por fecha
    3. Compara números predichos vs reales
    4. Calcula coincidencias y categorías de premio
    5. Guarda resultados en CSV
    
    Returns:
        str: Ruta del archivo CSV con resultados de validación
    """
    validator = BasicValidator()
    return validator.basic_validate_predictions()


if __name__ == "__main__":
    # Ejecutar validación básica si se ejecuta directamente
    try:
        csv_path = basic_validate_predictions()
        if csv_path:
            print(f"Validation completed successfully. Results saved to: {csv_path}")
        else:
            print("Validation completed but no results were generated.")
    except Exception as e:
        print(f"Error during validation: {e}")
        logger.error(f"Error during validation: {e}")
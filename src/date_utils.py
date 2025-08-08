
"""
Date Utilities - SHIOL+ Phase 3
===============================

Utilidad centralizada para manejo de fechas con timezone estandarizado
y logging detallado para tracking y debugging.
"""

import pytz
from datetime import datetime, timedelta
from typing import Optional, Union, List
from loguru import logger


class DateManager:
    """
    Administrador centralizado para todas las operaciones de fecha en SHIOL+.
    
    Características:
    - Timezone estandarizado (America/New_York)
    - Logging detallado para tracking
    - Validaciones consistentes
    - Cálculos de fechas de sorteo
    """
    
    # Timezone estándar para todo el proyecto
    POWERBALL_TIMEZONE = pytz.timezone('America/New_York')
    
    # Días de sorteo de Powerball (Miércoles=2, Sábado=5)
    DRAWING_DAYS = [2, 5]
    
    # Hora de sorteo (11 PM ET)
    DRAWING_HOUR = 23
    
    def __init__(self):
        """Inicializa el administrador de fechas."""
        logger.debug("DateManager initialized with timezone: America/New_York")
    
    @classmethod
    def get_current_et_time(cls) -> datetime:
        """
        Obtiene la fecha y hora actual en Eastern Time sin correcciones.
        
        Returns:
            datetime: Fecha y hora actual en ET con timezone
        """
        # Get system time in UTC
        system_utc = datetime.now(pytz.UTC)
        
        # Convert to Eastern Time
        current_time = system_utc.astimezone(cls.POWERBALL_TIMEZONE)
        
        logger.debug(f"System UTC: {system_utc.isoformat()}")
        logger.debug(f"ET time: {current_time.isoformat()}")
        
        return current_time
    
    @classmethod
    def convert_to_et(cls, dt: Union[datetime, str]) -> datetime:
        """
        Convierte cualquier fecha/hora a Eastern Time.
        
        Args:
            dt: Fecha como datetime o string ISO
            
        Returns:
            datetime: Fecha convertida a ET
        """
        if isinstance(dt, str):
            try:
                # Intentar parsear string ISO
                if 'T' in dt:
                    parsed_dt = datetime.fromisoformat(dt.replace('Z', '+00:00'))
                else:
                    parsed_dt = datetime.strptime(dt[:19], '%Y-%m-%d %H:%M:%S')
                    
                logger.debug(f"Parsed string date: {dt} -> {parsed_dt}")
            except ValueError as e:
                logger.warning(f"Failed to parse date string '{dt}': {e}")
                # Fallback: solo fecha
                parsed_dt = datetime.strptime(dt[:10], '%Y-%m-%d')
                logger.debug(f"Fallback parsed date: {dt} -> {parsed_dt}")
        else:
            parsed_dt = dt
        
        # Convertir a ET
        if parsed_dt.tzinfo is None:
            # Sin timezone, asumir que es local y convertir a ET
            et_time = cls.POWERBALL_TIMEZONE.localize(parsed_dt)
            logger.debug(f"Localized naive datetime to ET: {parsed_dt} -> {et_time}")
        else:
            # Ya tiene timezone, convertir a ET
            et_time = parsed_dt.astimezone(cls.POWERBALL_TIMEZONE)
            logger.debug(f"Converted timezone to ET: {parsed_dt} -> {et_time}")
        
        return et_time
    
    @classmethod
    def calculate_next_drawing_date(cls, reference_date: Optional[datetime] = None) -> str:
        """
        Calcula la próxima fecha de sorteo desde una fecha de referencia.
        
        Args:
            reference_date: Fecha de referencia (opcional, usa fecha actual si no se provee)
            
        Returns:
            str: Fecha del próximo sorteo en formato YYYY-MM-DD
        """
        if reference_date is None:
            reference_date = cls.get_current_et_time()
        else:
            reference_date = cls.convert_to_et(reference_date)
        
        current_weekday = reference_date.weekday()
        
        logger.info(f"Calculating next drawing date from: {reference_date.isoformat()}")
        logger.debug(f"Reference weekday: {current_weekday} ({'Monday' if current_weekday == 0 else 'Wednesday' if current_weekday == 2 else 'Saturday' if current_weekday == 5 else 'Other'})")
        
        # Si es día de sorteo y es antes de las 11 PM ET, el sorteo es ese día
        if current_weekday in cls.DRAWING_DAYS and reference_date.hour < cls.DRAWING_HOUR:
            next_draw_date = reference_date.strftime('%Y-%m-%d')
            logger.info(f"Drawing day before cutoff time - next drawing today: {next_draw_date}")
            return next_draw_date
        
        # Encontrar el próximo día de sorteo
        for i in range(1, 8):
            next_date = reference_date + timedelta(days=i)
            if next_date.weekday() in cls.DRAWING_DAYS:
                next_draw_date = next_date.strftime('%Y-%m-%d')
                logger.info(f"Next drawing date found: {next_draw_date} (in {i} days)")
                return next_draw_date
        
        # Fallback (no debería ocurrir)
        fallback_date = (reference_date + timedelta(days=1)).strftime('%Y-%m-%d')
        logger.warning(f"Fallback to next day: {fallback_date}")
        return fallback_date
    
    @classmethod
    def is_valid_drawing_date(cls, date_str: str) -> bool:
        """
        Verifica si una fecha corresponde a un día de sorteo válido.
        
        Args:
            date_str: Fecha en formato YYYY-MM-DD
            
        Returns:
            bool: True si es un día de sorteo válido
        """
        try:
            date_obj = datetime.strptime(date_str, '%Y-%m-%d')
            is_valid = date_obj.weekday() in cls.DRAWING_DAYS
            
            logger.debug(f"Drawing date validation: {date_str} -> weekday {date_obj.weekday()} -> valid: {is_valid}")
            
            if not is_valid:
                weekday_name = date_obj.strftime('%A')
                logger.warning(f"Invalid drawing date: {date_str} ({weekday_name}) - not a drawing day")
            
            return is_valid
            
        except ValueError as e:
            logger.error(f"Invalid date format for drawing validation: {date_str} - {e}")
            return False
    
    @classmethod
    def validate_date_format(cls, date_str: str) -> bool:
        """
        Valida que una fecha tenga el formato correcto YYYY-MM-DD.
        
        Args:
            date_str: String de fecha a validar
            
        Returns:
            bool: True si la fecha es válida
        """
        if not isinstance(date_str, str):
            logger.error(f"Date validation failed: not a string - {type(date_str)}")
            return False
        
        if len(date_str) != 10:
            logger.error(f"Date validation failed: incorrect length {len(date_str)} (expected 10)")
            return False
        
        if date_str.count('-') != 2:
            logger.error(f"Date validation failed: incorrect format (expected YYYY-MM-DD)")
            return False
        
        try:
            date_obj = datetime.strptime(date_str, '%Y-%m-%d')
            
            # Verificar rango razonable (no muy antigua o muy futura)
            current_year = datetime.now().year
            if date_obj.year < (current_year - 2) or date_obj.year > (current_year + 3):
                logger.warning(f"Date outside reasonable range: {date_str} (year {date_obj.year})")
                return False
            
            logger.debug(f"Date format validation passed: {date_str}")
            return True
            
        except ValueError as e:
            logger.error(f"Date format validation failed: {date_str} - {e}")
            return False
    
    @classmethod
    def format_date_for_display(cls, date_str: str, language: str = 'es') -> str:
        """
        Formatea una fecha para mostrar en la interfaz.
        
        Args:
            date_str: Fecha en formato YYYY-MM-DD
            language: Idioma ('es' para español, 'en' para inglés)
            
        Returns:
            str: Fecha formateada para mostrar
        """
        try:
            date_obj = datetime.strptime(date_str, '%Y-%m-%d')
            
            if language == 'es':
                spanish_months = {
                    1: 'Ene', 2: 'Feb', 3: 'Mar', 4: 'Abr', 5: 'May', 6: 'Jun',
                    7: 'Jul', 8: 'Ago', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dic'
                }
                formatted_date = f"{date_obj.day} {spanish_months[date_obj.month]} {date_obj.year}"
            else:
                formatted_date = date_obj.strftime('%b %d, %Y')
            
            logger.debug(f"Date formatted for display: {date_str} -> {formatted_date} ({language})")
            return formatted_date
            
        except Exception as e:
            logger.error(f"Error formatting date for display: {date_str} - {e}")
            return date_str
    
    @classmethod
    def get_drawing_days_info(cls) -> dict:
        """
        Obtiene información detallada sobre los días de sorteo.
        
        Returns:
            dict: Información sobre días de sorteo
        """
        info = {
            'drawing_days': cls.DRAWING_DAYS,
            'drawing_days_names': ['Wednesday', 'Saturday'],
            'drawing_days_spanish': ['Miércoles', 'Sábado'],
            'drawing_hour_et': cls.DRAWING_HOUR,
            'timezone': str(cls.POWERBALL_TIMEZONE),
            'next_drawing_date': cls.calculate_next_drawing_date()
        }
        
        logger.debug(f"Drawing days info requested: {info}")
        return info
    
    @classmethod
    def days_until_next_drawing(cls, reference_date: Optional[datetime] = None) -> int:
        """
        Calcula cuántos días faltan para el próximo sorteo.
        
        Args:
            reference_date: Fecha de referencia (opcional)
            
        Returns:
            int: Días hasta el próximo sorteo
        """
        if reference_date is None:
            reference_date = cls.get_current_et_time()
        else:
            reference_date = cls.convert_to_et(reference_date)
        
        next_drawing_str = cls.calculate_next_drawing_date(reference_date)
        next_drawing = datetime.strptime(next_drawing_str, '%Y-%m-%d')
        next_drawing = cls.POWERBALL_TIMEZONE.localize(next_drawing.replace(hour=cls.DRAWING_HOUR))
        
        # Calcular diferencia en días
        time_diff = next_drawing - reference_date
        days_until = time_diff.days
        
        # Si es el mismo día pero antes de la hora del sorteo, son 0 días
        if days_until == 0 and reference_date.hour < cls.DRAWING_HOUR:
            days_until = 0
        elif time_diff.total_seconds() < 0:
            days_until = 0
        
        logger.debug(f"Days until next drawing: {days_until} (from {reference_date.date()} to {next_drawing.date()})")
        return days_until
    
    @classmethod
    def get_recent_drawing_dates(cls, count: int = 10) -> List[str]:
        """
        Obtiene las fechas de los sorteos más recientes.
        
        Args:
            count: Número de fechas a obtener
            
        Returns:
            List[str]: Lista de fechas de sorteo en formato YYYY-MM-DD
        """
        current_date = cls.get_current_et_time()
        drawing_dates = []
        
        # Buscar hacia atrás desde la fecha actual
        check_date = current_date - timedelta(days=1)  # Empezar desde ayer
        
        while len(drawing_dates) < count:
            if check_date.weekday() in cls.DRAWING_DAYS:
                drawing_dates.append(check_date.strftime('%Y-%m-%d'))
            check_date -= timedelta(days=1)
        
        drawing_dates.reverse()  # Orden cronológico (más antigua primero)
        
        logger.debug(f"Recent drawing dates retrieved: {drawing_dates}")
        return drawing_dates
    
    @classmethod
    def get_current_date_info(cls) -> Dict[str, Any]:
        """
        Obtiene información completa de la fecha actual en ET.
        
        Returns:
            Dict: Información de fecha actual
        """
        current_time = cls.get_current_et_time()
        
        return {
            "date": current_time.strftime('%Y-%m-%d'),
            "formatted_date": current_time.strftime('%B %d, %Y'),
            "day": current_time.day,
            "month": current_time.month,
            "year": current_time.year,
            "weekday": current_time.weekday(),
            "weekday_name": current_time.strftime('%A'),
            "time": current_time.strftime('%H:%M ET'),
            "is_drawing_day": current_time.weekday() in cls.DRAWING_DAYS,
            "iso": current_time.isoformat()
        }


def get_current_et_time() -> datetime:
    """Función de conveniencia para obtener la hora actual en ET."""
    return DateManager.get_current_et_time()


def calculate_next_drawing_date(reference_date: Optional[datetime] = None) -> str:
    """Función de conveniencia para calcular la próxima fecha de sorteo."""
    return DateManager.calculate_next_drawing_date(reference_date)


def is_valid_drawing_date(date_str: str) -> bool:
    """Función de conveniencia para validar fecha de sorteo."""
    return DateManager.is_valid_drawing_date(date_str)


def validate_date_format(date_str: str) -> bool:
    """Función de conveniencia para validar formato de fecha."""
    return DateManager.validate_date_format(date_str)


def convert_to_et(dt: Union[datetime, str]) -> datetime:
    """Función de conveniencia para convertir a ET."""
    return DateManager.convert_to_et(dt)


# Logging de inicialización del módulo
logger.info("Date utilities module loaded - centralized date management initialized")
logger.debug(f"Standard timezone: {DateManager.POWERBALL_TIMEZONE}")
logger.debug(f"Drawing days: {DateManager.DRAWING_DAYS} (Monday, Wednesday, Saturday)")
logger.debug(f"Drawing hour: {DateManager.DRAWING_HOUR}:00 ET")

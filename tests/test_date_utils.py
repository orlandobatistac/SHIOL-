
"""
Tests for Date Utilities - SHIOL+ Phase 3
==========================================

Comprehensive test suite for date management functions
with timezone handling and validation testing.
"""

import pytest
import pytz
from datetime import datetime, timedelta
from src.date_utils import DateManager, calculate_next_drawing_date, is_valid_drawing_date, validate_date_format


class TestDateManager:
    """Test suite for DateManager class."""
    
    def test_get_current_et_time(self):
        """Test getting current ET time."""
        current_et = DateManager.get_current_et_time()
        
        assert current_et.tzinfo is not None
        assert current_et.tzinfo.zone == 'America/New_York'
        assert isinstance(current_et, datetime)
    
    def test_convert_to_et_string(self):
        """Test converting string dates to ET."""
        # Test ISO string
        iso_string = "2025-08-08T14:30:00"
        et_time = DateManager.convert_to_et(iso_string)
        
        assert et_time.tzinfo.zone == 'America/New_York'
        assert et_time.hour == 14
        assert et_time.minute == 30
        
        # Test date-only string
        date_string = "2025-08-08"
        et_time = DateManager.convert_to_et(date_string)
        
        assert et_time.tzinfo.zone == 'America/New_York'
        assert et_time.year == 2025
        assert et_time.month == 8
        assert et_time.day == 8
    
    def test_convert_to_et_datetime(self):
        """Test converting datetime objects to ET."""
        # Test naive datetime
        naive_dt = datetime(2025, 8, 8, 15, 30)
        et_time = DateManager.convert_to_et(naive_dt)
        
        assert et_time.tzinfo.zone == 'America/New_York'
        assert et_time.hour == 15
        
        # Test datetime with different timezone
        utc_tz = pytz.UTC
        utc_dt = utc_tz.localize(datetime(2025, 8, 8, 19, 30))  # 19:30 UTC
        et_time = DateManager.convert_to_et(utc_dt)
        
        assert et_time.tzinfo.zone == 'America/New_York'
        # 19:30 UTC should be 15:30 ET (EDT) in August
        assert et_time.hour in [14, 15]  # Account for DST
    
    def test_calculate_next_drawing_date_monday(self):
        """Test calculating next drawing date from Monday."""
        # Monday before 11 PM - should be same day
        monday_early = DateManager.POWERBALL_TIMEZONE.localize(
            datetime(2025, 8, 4, 10, 0)  # Monday 10 AM
        )
        next_date = DateManager.calculate_next_drawing_date(monday_early)
        assert next_date == "2025-08-04"  # Same day
        
        # Monday after 11 PM - should be Wednesday
        monday_late = DateManager.POWERBALL_TIMEZONE.localize(
            datetime(2025, 8, 4, 23, 30)  # Monday 11:30 PM
        )
        next_date = DateManager.calculate_next_drawing_date(monday_late)
        assert next_date == "2025-08-06"  # Wednesday
    
    def test_calculate_next_drawing_date_tuesday(self):
        """Test calculating next drawing date from Tuesday."""
        # Tuesday - should be Wednesday
        tuesday = DateManager.POWERBALL_TIMEZONE.localize(
            datetime(2025, 8, 5, 15, 0)  # Tuesday 3 PM
        )
        next_date = DateManager.calculate_next_drawing_date(tuesday)
        assert next_date == "2025-08-06"  # Wednesday
    
    def test_calculate_next_drawing_date_wednesday(self):
        """Test calculating next drawing date from Wednesday."""
        # Wednesday before 11 PM - should be same day
        wednesday_early = DateManager.POWERBALL_TIMEZONE.localize(
            datetime(2025, 8, 6, 10, 0)  # Wednesday 10 AM
        )
        next_date = DateManager.calculate_next_drawing_date(wednesday_early)
        assert next_date == "2025-08-06"  # Same day
        
        # Wednesday after 11 PM - should be Saturday
        wednesday_late = DateManager.POWERBALL_TIMEZONE.localize(
            datetime(2025, 8, 6, 23, 30)  # Wednesday 11:30 PM
        )
        next_date = DateManager.calculate_next_drawing_date(wednesday_late)
        assert next_date == "2025-08-09"  # Saturday
    
    def test_calculate_next_drawing_date_saturday(self):
        """Test calculating next drawing date from Saturday."""
        # Saturday before 11 PM - should be same day
        saturday_early = DateManager.POWERBALL_TIMEZONE.localize(
            datetime(2025, 8, 9, 10, 0)  # Saturday 10 AM
        )
        next_date = DateManager.calculate_next_drawing_date(saturday_early)
        assert next_date == "2025-08-09"  # Same day
        
        # Saturday after 11 PM - should be Monday
        saturday_late = DateManager.POWERBALL_TIMEZONE.localize(
            datetime(2025, 8, 9, 23, 30)  # Saturday 11:30 PM
        )
        next_date = DateManager.calculate_next_drawing_date(saturday_late)
        assert next_date == "2025-08-11"  # Monday
    
    def test_is_valid_drawing_date(self):
        """Test drawing date validation."""
        # Valid drawing dates (Monday, Wednesday, Saturday)
        assert DateManager.is_valid_drawing_date("2025-08-04") == True   # Monday
        assert DateManager.is_valid_drawing_date("2025-08-06") == True   # Wednesday
        assert DateManager.is_valid_drawing_date("2025-08-09") == True   # Saturday
        
        # Invalid drawing dates
        assert DateManager.is_valid_drawing_date("2025-08-05") == False  # Tuesday
        assert DateManager.is_valid_drawing_date("2025-08-07") == False  # Thursday
        assert DateManager.is_valid_drawing_date("2025-08-08") == False  # Friday
        assert DateManager.is_valid_drawing_date("2025-08-10") == False  # Sunday
        
        # Invalid date format
        assert DateManager.is_valid_drawing_date("invalid-date") == False
        assert DateManager.is_valid_drawing_date("2025/08/04") == False
    
    def test_validate_date_format(self):
        """Test date format validation."""
        # Valid formats
        assert DateManager.validate_date_format("2025-08-08") == True
        assert DateManager.validate_date_format("2024-01-01") == True
        assert DateManager.validate_date_format("2026-12-31") == True
        
        # Invalid formats
        assert DateManager.validate_date_format("2025/08/08") == False
        assert DateManager.validate_date_format("08-08-2025") == False
        assert DateManager.validate_date_format("2025-8-8") == False
        assert DateManager.validate_date_format("invalid") == False
        assert DateManager.validate_date_format("") == False
        assert DateManager.validate_date_format("2025-13-01") == False  # Invalid month
        assert DateManager.validate_date_format("2025-01-32") == False  # Invalid day
        
        # Edge cases
        assert DateManager.validate_date_format("2020-08-08") == False  # Too old
        assert DateManager.validate_date_format("2030-08-08") == False  # Too future
        assert DateManager.validate_date_format(None) == False
        assert DateManager.validate_date_format(123) == False
    
    def test_format_date_for_display(self):
        """Test date formatting for display."""
        # Spanish formatting
        spanish_date = DateManager.format_date_for_display("2025-08-08", "es")
        assert spanish_date == "8 Ago 2025"
        
        # English formatting
        english_date = DateManager.format_date_for_display("2025-08-08", "en")
        assert "Aug" in english_date
        assert "2025" in english_date
        
        # Invalid date should return original
        invalid_result = DateManager.format_date_for_display("invalid-date", "es")
        assert invalid_result == "invalid-date"
    
    def test_get_drawing_days_info(self):
        """Test getting drawing days information."""
        info = DateManager.get_drawing_days_info()
        
        assert info['drawing_days'] == [0, 2, 5]
        assert info['drawing_days_names'] == ['Monday', 'Wednesday', 'Saturday']
        assert info['drawing_days_spanish'] == ['Lunes', 'Miércoles', 'Sábado']
        assert info['drawing_hour_et'] == 23
        assert 'America/New_York' in info['timezone']
        assert info['next_drawing_date'] is not None
        assert DateManager.validate_date_format(info['next_drawing_date'])
    
    def test_days_until_next_drawing(self):
        """Test calculating days until next drawing."""
        # Test from Monday morning (should be 0 days)
        monday_morning = DateManager.POWERBALL_TIMEZONE.localize(
            datetime(2025, 8, 4, 10, 0)  # Monday 10 AM
        )
        days = DateManager.days_until_next_drawing(monday_morning)
        assert days == 0
        
        # Test from Tuesday (should be 1 day to Wednesday)
        tuesday = DateManager.POWERBALL_TIMEZONE.localize(
            datetime(2025, 8, 5, 15, 0)  # Tuesday 3 PM
        )
        days = DateManager.days_until_next_drawing(tuesday)
        assert days == 1
    
    def test_get_recent_drawing_dates(self):
        """Test getting recent drawing dates."""
        recent_dates = DateManager.get_recent_drawing_dates(5)
        
        assert len(recent_dates) == 5
        
        # All should be valid dates
        for date_str in recent_dates:
            assert DateManager.validate_date_format(date_str)
            assert DateManager.is_valid_drawing_date(date_str)
        
        # Should be in chronological order
        for i in range(1, len(recent_dates)):
            prev_date = datetime.strptime(recent_dates[i-1], '%Y-%m-%d')
            curr_date = datetime.strptime(recent_dates[i], '%Y-%m-%d')
            assert curr_date > prev_date


class TestConvenienceFunctions:
    """Test suite for convenience functions."""
    
    def test_calculate_next_drawing_date_function(self):
        """Test the convenience function."""
        next_date = calculate_next_drawing_date()
        assert validate_date_format(next_date)
        assert is_valid_drawing_date(next_date)
    
    def test_is_valid_drawing_date_function(self):
        """Test the convenience function."""
        assert is_valid_drawing_date("2025-08-09") == True   # Saturday
        assert is_valid_drawing_date("2025-08-08") == False  # Friday
    
    def test_validate_date_format_function(self):
        """Test the convenience function."""
        assert validate_date_format("2025-08-08") == True
        assert validate_date_format("invalid") == False


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"])

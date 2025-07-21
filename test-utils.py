"""
Tests for utils module
"""

import pytest
from datetime import datetime, date
import pandas as pd
import numpy as np
import logging
from decimal import Decimal

from src.utils.validators import (
    validate_probability, validate_odds, validate_stake,
    validate_team_name, validate_date, validate_league_id,
    validate_season, validate_features, validate_prediction_data,
    ValidationError, sanitize_input
)
from src.utils.formatters import (
    format_currency, format_percentage, format_odds,
    format_date, format_time, format_duration, format_number,
    format_team_name, format_confidence_level
)
from src.utils.logger import setup_logging, get_logger, log_performance
from src.utils.exceptions import (
    APIError, RateLimitError, ModelError, ValidationError as ValError,
    handle_exception
)


class TestValidators:
    """Test cases for validators"""
    
    def test_validate_probability(self):
        """Test probability validation"""
        # Valid probabilities
        assert validate_probability(0.5) == 0.5
        assert validate_probability(0) == 0.0
        assert validate_probability(1) == 1.0
        
        # Invalid probabilities
        with pytest.raises(ValidationError):
            validate_probability(-0.1)
        with pytest.raises(ValidationError):
            validate_probability(1.1)
        with pytest.raises(ValidationError):
            validate_probability("not a number")
    
    def test_validate_odds(self):
        """Test odds validation"""
        # Valid odds
        assert validate_odds(1.5) == 1.5
        assert validate_odds(100) == 100.0
        
        # Invalid odds
        with pytest.raises(ValidationError):
            validate_odds(0.5)
        with pytest.raises(ValidationError):
            validate_odds(101)
        with pytest.raises(ValidationError):
            validate_odds("invalid")
    
    def test_validate_stake(self):
        """Test stake validation"""
        # Valid stakes
        assert validate_stake(10) == 10.0
        assert validate_stake(1000.50) == 1000.50
        
        # Invalid stakes
        with pytest.raises(ValidationError):
            validate_stake(0)
        with pytest.raises(ValidationError):
            validate_stake(10001)
    
    def test_validate_team_name(self):
        """Test team name validation"""
        # Valid names
        assert validate_team_name("Manchester United") == "Manchester United"
        assert validate_team_name("  Arsenal  ") == "Arsenal"
        assert validate_team_name("1. FC Köln") == "1. FC Köln"
        
        # Invalid names
        with pytest.raises(ValidationError):
            validate_team_name("")
        with pytest.raises(ValidationError):
            validate_team_name("A")  # Too short
        with pytest.raises(ValidationError):
            validate_team_name("Team@#$%")  # Invalid characters
    
    def test_validate_date(self):
        """Test date validation"""
        # Valid dates
        assert isinstance(validate_date("2024-01-15"), datetime)
        assert isinstance(validate_date(datetime.now()), datetime)
        assert isinstance(validate_date(date.today()), datetime)
        
        # Invalid dates
        with pytest.raises(ValidationError):
            validate_date("invalid-date")
        with pytest.raises(ValidationError):
            validate_date(12345)
    
    def test_validate_league_id(self):
        """Test league ID validation"""
        # Valid IDs
        assert validate_league_id(39) == 39
        assert validate_league_id("135") == 135
        
        # Invalid IDs
        with pytest.raises(ValidationError):
            validate_league_id(0)
        with pytest.raises(ValidationError):
            validate_league_id(-1)
        with pytest.raises(ValidationError):
            validate_league_id("invalid")
    
    def test_validate_season(self):
        """Test season validation"""
        current_year = datetime.now().year
        
        # Valid seasons
        assert validate_season(2023) == 2023
        assert validate_season(current_year) == current_year
        
        # Invalid seasons
        with pytest.raises(ValidationError):
            validate_season(1999)
        with pytest.raises(ValidationError):
            validate_season(current_year + 2)
    
    def test_validate_features(self):
        """Test feature validation"""
        features = pd.DataFrame({
            'home_goals_avg': [1.5, 2.0],
            'away_goals_avg': [1.3, 1.8],
            'over25_rate': [0.6, 0.7]
        })
        
        # Valid features
        assert validate_features(features, ['home_goals_avg', 'away_goals_avg'])
        
        # Missing features
        assert not validate_features(features, ['missing_feature'])
        
        # Invalid values
        features.loc[0, 'over25_rate'] = 1.5  # Out of range
        assert not validate_features(features, ['over25_rate'])
    
    def test_sanitize_input(self):
        """Test input sanitization"""
        # Text sanitization
        assert sanitize_input("<script>alert('xss')</script>") == "scriptalert('xss')/script"
        
        # Numeric sanitization
        assert sanitize_input("123.45abc", "numeric") == "123.45"
        
        # Alphanumeric sanitization
        assert sanitize_input("Team@123!", "alphanumeric") == "Team123"


class TestFormatters:
    """Test cases for formatters"""
    
    def test_format_currency(self):
        """Test currency formatting"""
        assert format_currency(1234.56) == "€1,234.56"
        assert format_currency(1000) == "€1,000.00"
        assert format_currency(-500.50) == "-€500.50"
        assert format_currency(0) == "€0.00"
        
        # Custom settings
        assert format_currency(1234.56, "$", 0) == "$1,235"
        assert format_currency(1234.56, "£", 2, ".", ",") == "£1.234,56"
    
    def test_format_percentage(self):
        """Test percentage formatting"""
        assert format_percentage(0.756) == "75.6%"
        assert format_percentage(0.5, 0) == "50%"
        assert format_percentage(1.0) == "100.0%"
        assert format_percentage(0.756, 2, False) == "75.60"
    
    def test_format_odds(self):
        """Test odds formatting"""
        # Decimal odds
        assert format_odds(1.85) == "@1.85"
        assert format_odds(2.50, include_prefix=False) == "2.50"
        
        # Fractional odds
        assert format_odds(1.5, "fractional") == "@1/2"
        assert format_odds(2.5, "fractional") == "@3/2"
        
        # American odds
        assert format_odds(2.0, "american") == "@+100"
        assert format_odds(1.5, "american") == "@-200"
    
    def test_format_date(self):
        """Test date formatting"""
        test_date = datetime(2024, 1, 15, 10, 30)
        
        assert format_date(test_date) == "2024-01-15"
        assert format_date(test_date, "%d/%m/%Y") == "15/01/2024"
        
        # Relative formatting
        today = datetime.now()
        assert format_date(today, relative=True) == "Today"
        
        tomorrow = today + pd.Timedelta(days=1)
        assert format_date(tomorrow, relative=True) == "Tomorrow"
    
    def test_format_time(self):
        """Test time formatting"""
        test_time = datetime(2024, 1, 15, 14, 30, 45)
        
        assert format_time(test_time) == "14:30"
        assert format_time(test_time, include_seconds=True) == "14:30:45"
    
    def test_format_duration(self):
        """Test duration formatting"""
        assert format_duration(3665) == "1h 1m 5s"
        assert format_duration(3665, "long") == "1 hour 1 minute 5 seconds"
        assert format_duration(45) == "45s"
        assert format_duration(7200) == "2h"
    
    def test_format_number(self):
        """Test number formatting"""
        assert format_number(1234567) == "1,234,567"
        assert format_number(1234.567, 2) == "1,234.57"
        
        # Compact notation
        assert format_number(1234, compact=True) == "1.2K"
        assert format_number(1234567, compact=True) == "1.2M"
        assert format_number(1234567890, compact=True) == "1.2B"
    
    def test_format_team_name(self):
        """Test team name formatting"""
        assert format_team_name("Manchester United", abbreviate=True) == "Man Utd"
        assert format_team_name("Real Madrid", max_length=8) == "Real Mad..."
        assert format_team_name("PSG") == "PSG"
    
    def test_format_confidence_level(self):
        """Test confidence level formatting"""
        assert format_confidence_level("High") == "High"
        assert format_confidence_level("High", "emoji") == "🟢 High"
        assert format_confidence_level(0.85, "text") == "High"
        assert format_confidence_level(0.65, "bar") == "█████ Medium"


class TestLogger:
    """Test cases for logger"""
    
    def test_setup_logging(self, tmp_path):
        """Test logging setup"""
        # Modify config to use temp directory
        config = {
            "version": 1,
            "handlers": {
                "file": {
                    "class": "logging.FileHandler",
                    "filename": str(tmp_path / "test.log"),
                    "level": "DEBUG"
                }
            },
            "loggers": {
                "test": {
                    "level": "DEBUG",
                    "handlers": ["file"]
                }
            }
        }
        
        logger = setup_logging(config)
        assert logger is not None
        
        # Test logging
        test_logger = get_logger("test")
        test_logger.info("Test message")
        
        # Check log file exists
        log_file = tmp_path / "test.log"
        assert log_file.exists()
    
    def test_get_logger(self):
        """Test getting logger with context"""
        context = {"user_id": "123", "session": "abc"}
        logger = get_logger("test.module", context)
        
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test.module"
    
    @pytest.mark.asyncio
    async def test_log_performance_async(self, caplog):
        """Test performance logging decorator for async functions"""
        @log_performance
        async def slow_function():
            await asyncio.sleep(0.1)
            return "done"
        
        result = await slow_function()
        
        assert result == "done"
        assert "completed in" in caplog.text
    
    def test_log_performance_sync(self, caplog):
        """Test performance logging decorator for sync functions"""
        @log_performance
        def fast_function():
            return "done"
        
        result = fast_function()
        
        assert result == "done"
        assert "completed in" in caplog.text


class TestExceptions:
    """Test cases for custom exceptions"""
    
    def test_api_error(self):
        """Test API error"""
        error = APIError("API failed", {"status": 500})
        assert str(error) == "API failed (status=500)"
        assert error.details["status"] == 500
    
    def test_rate_limit_error(self):
        """Test rate limit error"""
        error = RateLimitError("Rate limit exceeded", retry_after=60)
        assert error.retry_after == 60
        assert "Rate limit exceeded" in str(error)
    
    def test_model_error(self):
        """Test model error"""
        error = ModelError("Model failed")
        assert str(error) == "Model failed"
    
    def test_handle_exception(self):
        """Test exception handler"""
        # Rate limit error
        exc = RateLimitError("Limited", retry_after=30)
        msg = handle_exception(exc)
        assert "try again in 30 seconds" in msg
        
        # Generic error
        exc = Exception("Something went wrong")
        msg = handle_exception(exc)
        assert "unexpected error" in msg.lower()
        
        # Custom message
        exc = ValueError("Bad value")
        msg = handle_exception(exc, user_message="Invalid input provided")
        assert msg == "Invalid input provided"
    
    def test_exception_handler_context(self):
        """Test exception handler context manager"""
        # Suppress exception
        with ExceptionHandler(reraise=False) as handler:
            raise ValueError("Test error")
        
        assert handler.exception is not None
        assert isinstance(handler.exception, ValueError)
        
        # Reraise exception
        with pytest.raises(ValueError):
            with ExceptionHandler(reraise=True):
                raise ValueError("Test error")
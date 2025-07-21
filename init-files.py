# src/__init__.py
"""
ProFootballAI - Professional Over 2.5 Goals Prediction Suite
"""

__version__ = "2.0.0"
__author__ = "ProFootballAI Team"
__email__ = "support@profootball-ai.com"

# src/api/__init__.py
"""API integration modules"""

from .football_api import FootballAPIClient
from .rate_limiter import RateLimiter

__all__ = ['FootballAPIClient', 'RateLimiter']

# src/models/__init__.py
"""Machine Learning models"""

from .predictor import Over25Predictor, PredictionResult
from .bet_optimizer import BetOptimizer, Match, BettingSlip
from .feature_engineering import FeatureEngineer

__all__ = [
    'Over25Predictor', 'PredictionResult',
    'BetOptimizer', 'Match', 'BettingSlip',
    'FeatureEngineer'
]

# src/data/__init__.py
"""Data management modules"""

from .database import DatabaseManager
from .cache_manager import CacheManager

__all__ = ['DatabaseManager', 'CacheManager']

# src/ui/__init__.py
"""User Interface modules"""

from .theme import apply_theme, create_metric_card, create_prediction_card

__all__ = ['apply_theme', 'create_metric_card', 'create_prediction_card']

# src/ui/pages/__init__.py
"""UI Pages"""

from . import dashboard, predictions, betting_slips, analytics

__all__ = ['dashboard', 'predictions', 'betting_slips', 'analytics']

# src/utils/__init__.py
"""Utility modules"""

from .validators import (
    validate_probability, validate_odds, validate_stake,
    validate_team_name, validate_date, ValidationError
)
from .formatters import (
    format_currency, format_percentage, format_odds,
    format_date, format_number
)
from .logger import setup_logging, get_logger, log_performance
from .exceptions import (
    ProFootballAIError, APIError, ModelError,
    ValidationError, handle_exception
)

__all__ = [
    # Validators
    'validate_probability', 'validate_odds', 'validate_stake',
    'validate_team_name', 'validate_date', 'ValidationError',
    
    # Formatters
    'format_currency', 'format_percentage', 'format_odds',
    'format_date', 'format_number',
    
    # Logger
    'setup_logging', 'get_logger', 'log_performance',
    
    # Exceptions
    'ProFootballAIError', 'APIError', 'ModelError',
    'handle_exception'
]
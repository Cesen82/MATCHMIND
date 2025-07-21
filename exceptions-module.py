"""
Custom exceptions for ProFootballAI
"""

from typing import Optional, Dict, Any


class ProFootballAIError(Exception):
    """Base exception for ProFootballAI"""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}
        
    def __str__(self):
        if self.details:
            details_str = ', '.join(f"{k}={v}" for k, v in self.details.items())
            return f"{self.message} ({details_str})"
        return self.message


# API Related Exceptions
class APIError(ProFootballAIError):
    """Base API exception"""
    pass


class RateLimitError(APIError):
    """Rate limit exceeded"""
    
    def __init__(self, message: str, retry_after: Optional[int] = None, **kwargs):
        super().__init__(message, kwargs)
        self.retry_after = retry_after


class AuthenticationError(APIError):
    """API authentication failed"""
    pass


class APITimeoutError(APIError):
    """API request timeout"""
    pass


class InvalidAPIResponseError(APIError):
    """Invalid API response format"""
    pass


# Data Related Exceptions
class DataError(ProFootballAIError):
    """Base data exception"""
    pass


class InvalidDataError(DataError):
    """Invalid data format or content"""
    pass


class MissingDataError(DataError):
    """Required data is missing"""
    pass


class DataIntegrityError(DataError):
    """Data integrity check failed"""
    pass


# Model Related Exceptions
class ModelError(ProFootballAIError):
    """Base model exception"""
    pass


class ModelNotTrainedError(ModelError):
    """Model not trained"""
    
    def __init__(self, model_name: str = "model"):
        super().__init__(f"{model_name} is not trained. Please train the model first.")
        self.model_name = model_name


class InvalidFeaturesError(ModelError):
    """Invalid features for model"""
    
    def __init__(self, message: str, missing_features: Optional[list] = None, **kwargs):
        super().__init__(message, kwargs)
        self.missing_features = missing_features or []


class PredictionError(ModelError):
    """Error during prediction"""
    pass


# Database Related Exceptions
class DatabaseError(ProFootballAIError):
    """Base database exception"""
    pass


class ConnectionError(DatabaseError):
    """Database connection failed"""
    pass


class QueryError(DatabaseError):
    """Database query failed"""
    pass


class TransactionError(DatabaseError):
    """Database transaction failed"""
    pass


# Betting Related Exceptions
class BettingError(ProFootballAIError):
    """Base betting exception"""
    pass


class InvalidOddsError(BettingError):
    """Invalid betting odds"""
    
    def __init__(self, odds: float, min_odds: float = 1.01, max_odds: float = 100.0):
        super().__init__(
            f"Invalid odds: {odds}. Must be between {min_odds} and {max_odds}",
            {'odds': odds, 'min': min_odds, 'max': max_odds}
        )
        self.odds = odds
        self.min_odds = min_odds
        self.max_odds = max_odds


class InvalidStakeError(BettingError):
    """Invalid stake amount"""
    
    def __init__(self, stake: float, min_stake: float = 1, max_stake: float = 10000):
        super().__init__(
            f"Invalid stake: {stake}. Must be between {min_stake} and {max_stake}",
            {'stake': stake, 'min': min_stake, 'max': max_stake}
        )
        self.stake = stake
        self.min_stake = min_stake
        self.max_stake = max_stake


class InsufficientBankrollError(BettingError):
    """Insufficient bankroll for bet"""
    
    def __init__(self, required: float, available: float):
        super().__init__(
            f"Insufficient bankroll. Required: {required}, Available: {available}",
            {'required': required, 'available': available}
        )
        self.required = required
        self.available = available


# Configuration Related Exceptions
class ConfigurationError(ProFootballAIError):
    """Base configuration exception"""
    pass


class MissingConfigError(ConfigurationError):
    """Required configuration missing"""
    
    def __init__(self, config_key: str):
        super().__init__(f"Missing required configuration: {config_key}")
        self.config_key = config_key


class InvalidConfigError(ConfigurationError):
    """Invalid configuration value"""
    
    def __init__(self, config_key: str, value: Any, reason: str):
        super().__init__(
            f"Invalid configuration for {config_key}: {value}. {reason}",
            {'key': config_key, 'value': value, 'reason': reason}
        )
        self.config_key = config_key
        self.value = value
        self.reason = reason


# Cache Related Exceptions
class CacheError(ProFootballAIError):
    """Base cache exception"""
    pass


class CacheExpiredError(CacheError):
    """Cache entry expired"""
    pass


class CacheLimitError(CacheError):
    """Cache size limit exceeded"""
    pass


# Validation Related Exceptions
class ValidationError(ProFootballAIError):
    """Base validation exception"""
    pass


class InputValidationError(ValidationError):
    """User input validation failed"""
    
    def __init__(self, field: str, value: Any, reason: str):
        super().__init__(
            f"Invalid input for {field}: {value}. {reason}",
            {'field': field, 'value': value, 'reason': reason}
        )
        self.field = field
        self.value = value
        self.reason = reason


# UI Related Exceptions
class UIError(ProFootballAIError):
    """Base UI exception"""
    pass


class RenderError(UIError):
    """UI rendering failed"""
    pass


class ComponentError(UIError):
    """UI component error"""
    pass


# Utility function to handle exceptions gracefully
def handle_exception(exc: Exception, logger=None, user_message: Optional[str] = None) -> str:
    """
    Handle exception and return user-friendly message
    
    Args:
        exc: The exception to handle
        logger: Optional logger instance
        user_message: Optional custom user message
        
    Returns:
        User-friendly error message
    """
    
    if logger:
        logger.error(f"Exception occurred: {type(exc).__name__}: {str(exc)}", exc_info=True)
    
    # Return custom message if provided
    if user_message:
        return user_message
    
    # Handle specific exceptions
    if isinstance(exc, RateLimitError):
        if exc.retry_after:
            return f"API rate limit reached. Please try again in {exc.retry_after} seconds."
        return "API rate limit reached. Please try again later."
    
    elif isinstance(exc, AuthenticationError):
        return "Authentication failed. Please check your API key."
    
    elif isinstance(exc, APITimeoutError):
        return "Request timed out. Please try again."
    
    elif isinstance(exc, ModelNotTrainedError):
        return "The prediction model is not ready. Please wait for initialization."
    
    elif isinstance(exc, InvalidOddsError):
        return f"Invalid odds value. Please use odds between {exc.min_odds} and {exc.max_odds}."
    
    elif isinstance(exc, InsufficientBankrollError):
        return f"Insufficient funds. You need €{exc.required:.2f} but only have €{exc.available:.2f}."
    
    elif isinstance(exc, ValidationError):
        return f"Validation error: {exc.message}"
    
    elif isinstance(exc, DatabaseError):
        return "Database error occurred. Please try again or contact support."
    
    elif isinstance(exc, ProFootballAIError):
        return f"Error: {exc.message}"
    
    # Generic error
    return "An unexpected error occurred. Please try again or contact support."


# Exception context manager for better error handling
class ExceptionHandler:
    """Context manager for exception handling"""
    
    def __init__(self, logger=None, default_message: str = None, reraise: bool = False):
        self.logger = logger
        self.default_message = default_message
        self.reraise = reraise
        self.exception = None
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_val:
            self.exception = exc_val
            message = handle_exception(exc_val, self.logger, self.default_message)
            
            if self.logger:
                self.logger.error(message)
            
            if self.reraise:
                return False
            
            # Suppress exception
            return True
        
        return False
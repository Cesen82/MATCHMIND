"""
Data validation utilities for ProFootballAI
"""

import re
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, date
import pandas as pd
import numpy as np
import logging

from config import VALIDATION_RULES, MODEL_CONFIG

logger = logging.getLogger("profootball.validators")


class ValidationError(Exception):
    """Custom validation error"""
    pass


def validate_probability(value: float, field_name: str = "probability") -> float:
    """Validate probability value (0-1)"""
    rules = VALIDATION_RULES['probability']
    
    if not isinstance(value, (int, float)):
        raise ValidationError(f"{field_name} must be numeric, got {type(value)}")
        
    if value < rules['min'] or value > rules['max']:
        raise ValidationError(
            f"{field_name} must be between {rules['min']} and {rules['max']}, got {value}"
        )
        
    return float(value)


def validate_odds(value: float, field_name: str = "odds") -> float:
    """Validate betting odds"""
    rules = VALIDATION_RULES['odds']
    
    if not isinstance(value, (int, float)):
        raise ValidationError(f"{field_name} must be numeric, got {type(value)}")
        
    if value < rules['min'] or value > rules['max']:
        raise ValidationError(
            f"{field_name} must be between {rules['min']} and {rules['max']}, got {value}"
        )
        
    return float(value)


def validate_stake(value: Union[int, float], field_name: str = "stake") -> float:
    """Validate stake amount"""
    rules = VALIDATION_RULES['stake']
    
    if not isinstance(value, (int, float)):
        raise ValidationError(f"{field_name} must be numeric, got {type(value)}")
        
    if value < rules['min'] or value > rules['max']:
        raise ValidationError(
            f"{field_name} must be between {rules['min']} and {rules['max']}, got {value}"
        )
        
    return float(value)


def validate_team_name(name: str) -> str:
    """Validate team name"""
    if not isinstance(name, str):
        raise ValidationError(f"Team name must be string, got {type(name)}")
        
    name = name.strip()
    
    if len(name) < 2:
        raise ValidationError(f"Team name too short: {name}")
        
    if len(name) > 100:
        raise ValidationError(f"Team name too long: {name}")
        
    # Check for valid characters (letters, numbers, spaces, common punctuation)
    if not re.match(r'^[\w\s\-\.\'&]+$', name, re.UNICODE):
        raise ValidationError(f"Team name contains invalid characters: {name}")
        
    return name


def validate_date(value: Union[str, datetime, date], field_name: str = "date") -> datetime:
    """Validate and parse date"""
    if isinstance(value, datetime):
        return value
        
    if isinstance(value, date):
        return datetime.combine(value, datetime.min.time())
        
    if isinstance(value, str):
        # Try common date formats
        formats = [
            '%Y-%m-%d',
            '%Y-%m-%d %H:%M:%S',
            '%Y-%m-%dT%H:%M:%S',
            '%Y-%m-%dT%H:%M:%SZ',
            '%d/%m/%Y',
            '%d-%m-%Y'
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(value, fmt)
            except ValueError:
                continue
                
        raise ValidationError(f"Invalid date format for {field_name}: {value}")
        
    raise ValidationError(f"{field_name} must be string, datetime or date, got {type(value)}")


def validate_league_id(league_id: int) -> int:
    """Validate league ID"""
    if not isinstance(league_id, int):
        try:
            league_id = int(league_id)
        except (ValueError, TypeError):
            raise ValidationError(f"League ID must be integer, got {type(league_id)}")
            
    if league_id <= 0:
        raise ValidationError(f"League ID must be positive, got {league_id}")
        
    return league_id


def validate_season(season: int) -> int:
    """Validate season year"""
    if not isinstance(season, int):
        try:
            season = int(season)
        except (ValueError, TypeError):
            raise ValidationError(f"Season must be integer, got {type(season)}")
            
    current_year = datetime.now().year
    
    if season < 2000 or season > current_year + 1:
        raise ValidationError(f"Season must be between 2000 and {current_year + 1}, got {season}")
        
    return season


def validate_features(features: Union[Dict[str, Any], pd.DataFrame, List], 
                     expected_features: List[str]) -> bool:
    """Validate feature set for model input"""
    
    # Convert to DataFrame if needed
    if isinstance(features, dict):
        features = pd.DataFrame([features])
    elif isinstance(features, list):
        features = pd.DataFrame(features)
        
    if not isinstance(features, pd.DataFrame):
        logger.error(f"Features must be dict, list or DataFrame, got {type(features)}")
        return False
        
    # Check for missing features
    missing_features = set(expected_features) - set(features.columns)
    if missing_features:
        logger.error(f"Missing required features: {missing_features}")
        return False
        
    # Validate individual features
    for feature in expected_features:
        if feature not in features.columns:
            continue
            
        values = features[feature]
        
        # Check for NaN values
        if values.isna().any():
            logger.warning(f"Feature '{feature}' contains NaN values")
            
        # Type-specific validation
        if 'rate' in feature or 'probability' in feature:
            if not ((values >= 0) & (values <= 1)).all():
                logger.error(f"Feature '{feature}' should be between 0 and 1")
                return False
                
        elif 'goals' in feature and 'avg' in feature:
            if not ((values >= 0) & (values <= 10)).all():
                logger.warning(f"Feature '{feature}' has unusual values (outside 0-10 range)")
                
        elif 'form' in feature:
            if not ((values >= 0) & (values <= 15)).all():
                logger.warning(f"Feature '{feature}' has unusual values (outside 0-15 range)")
                
    return True


def validate_prediction_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate prediction data structure"""
    required_fields = [
        'fixture_id', 'home_team', 'away_team', 'league',
        'match_date', 'prediction', 'probability', 'confidence',
        'confidence_score', 'odds', 'expected_value'
    ]
    
    # Check required fields
    missing_fields = set(required_fields) - set(data.keys())
    if missing_fields:
        raise ValidationError(f"Missing required fields: {missing_fields}")
        
    # Validate individual fields
    validated = {}
    
    validated['fixture_id'] = str(data['fixture_id'])
    validated['home_team'] = validate_team_name(data['home_team'])
    validated['away_team'] = validate_team_name(data['away_team'])
    validated['league'] = str(data['league'])
    validated['match_date'] = validate_date(data['match_date'])
    
    validated['prediction'] = int(data['prediction'])
    if validated['prediction'] not in [0, 1]:
        raise ValidationError(f"Prediction must be 0 or 1, got {validated['prediction']}")
        
    validated['probability'] = validate_probability(data['probability'])
    
    validated['confidence'] = str(data['confidence'])
    if validated['confidence'] not in ['High', 'Medium', 'Low']:
        raise ValidationError(f"Invalid confidence level: {validated['confidence']}")
        
    validated['confidence_score'] = validate_probability(data['confidence_score'])
    validated['odds'] = validate_odds(data['odds'])
    
    validated['expected_value'] = float(data['expected_value'])
    
    # Optional fields
    if 'features' in data:
        validated['features'] = dict(data['features'])
        
    if 'feature_importance' in data:
        validated['feature_importance'] = dict(data['feature_importance'])
        
    return validated


def validate_betting_slip_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate betting slip data"""
    required_fields = [
        'slip_type', 'total_odds', 'combined_probability',
        'expected_value', 'kelly_stake', 'recommended_stake',
        'risk_level', 'confidence_score', 'diversification_score'
    ]
    
    # Check required fields
    missing_fields = set(required_fields) - set(data.keys())
    if missing_fields:
        raise ValidationError(f"Missing required fields: {missing_fields}")
        
    # Validate individual fields
    validated = {}
    
    validated['slip_type'] = str(data['slip_type'])
    if validated['slip_type'] not in ['single', 'double', 'triple', 'quadruple', 'quintuple']:
        raise ValidationError(f"Invalid slip type: {validated['slip_type']}")
        
    validated['total_odds'] = validate_odds(data['total_odds'])
    validated['combined_probability'] = validate_probability(data['combined_probability'])
    validated['expected_value'] = float(data['expected_value'])
    
    validated['kelly_stake'] = validate_probability(data['kelly_stake'])
    validated['recommended_stake'] = validate_probability(data['recommended_stake'])
    
    validated['risk_level'] = str(data['risk_level'])
    if validated['risk_level'] not in ['Low', 'Medium', 'High']:
        raise ValidationError(f"Invalid risk level: {validated['risk_level']}")
        
    validated['confidence_score'] = validate_probability(data['confidence_score'])
    validated['diversification_score'] = validate_probability(data['diversification_score'])
    
    return validated


def validate_api_response(response: Dict[str, Any], endpoint: str) -> bool:
    """Validate API response structure"""
    if not isinstance(response, dict):
        logger.error(f"API response must be dict, got {type(response)}")
        return False
        
    # Check for errors
    if 'errors' in response and response['errors']:
        logger.error(f"API returned errors: {response['errors']}")
        return False
        
    # Check for response key
    if 'response' not in response:
        logger.error("API response missing 'response' key")
        return False
        
    # Endpoint-specific validation
    if endpoint == 'fixtures':
        if not isinstance(response['response'], list):
            logger.error("Fixtures response should be a list")
            return False
            
    elif endpoint == 'teams/statistics':
        if not isinstance(response['response'], dict):
            logger.error("Team statistics response should be a dict")
            return False
            
    return True


def sanitize_input(value: str, input_type: str = "text") -> str:
    """Sanitize user input"""
    if not isinstance(value, str):
        value = str(value)
        
    # Remove leading/trailing whitespace
    value = value.strip()
    
    # Type-specific sanitization
    if input_type == "text":
        # Remove potentially harmful characters
        value = re.sub(r'[<>\"\'&]', '', value)
        
    elif input_type == "numeric":
        # Keep only numbers and decimal points
        value = re.sub(r'[^0-9\.\-]', '', value)
        
    elif input_type == "alphanumeric":
        # Keep only letters, numbers, and basic punctuation
        value = re.sub(r'[^a-zA-Z0-9\s\-_]', '', value)
        
    return value


def validate_model_features(features: pd.DataFrame) -> Tuple[bool, List[str]]:
    """Comprehensive feature validation for model input"""
    errors = []
    
    # Check shape
    if features.empty:
        errors.append("Features DataFrame is empty")
        return False, errors
        
    # Check for required features
    required_features = MODEL_CONFIG['features']
    missing = set(required_features) - set(features.columns)
    if missing:
        errors.append(f"Missing features: {missing}")
        
    # Check data types
    numeric_features = features.select_dtypes(include=[np.number]).columns
    non_numeric = set(required_features) - set(numeric_features)
    if non_numeric:
        errors.append(f"Non-numeric features found: {non_numeric}")
        
    # Check for infinite values
    if np.isinf(features.values).any():
        inf_columns = features.columns[np.isinf(features).any()].tolist()
        errors.append(f"Infinite values in columns: {inf_columns}")
        
    # Check ranges
    for col in features.columns:
        if 'rate' in col or 'probability' in col:
            if (features[col] < 0).any() or (features[col] > 1).any():
                errors.append(f"Column '{col}' has values outside [0, 1] range")
                
    return len(errors) == 0, errors
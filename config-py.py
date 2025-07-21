"""
ProFootballAI - Configuration Module
Centralizes all configuration settings for the application
"""

import os
from typing import Dict, Any
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CACHE_DIR = DATA_DIR / "cache"
DB_PATH = DATA_DIR / "profootball.db"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

# API Configuration
API_CONFIG = {
    "base_url": "https://v3.football.api-sports.io",
    "api_key": os.getenv("API_FOOTBALL_KEY", ""),
    "rate_limit": {
        "calls_per_hour": 100,
        "calls_per_day": 1000,
        "retry_after": 60  # seconds
    },
    "timeout": 10,
    "max_retries": 3
}

# Model Configuration
MODEL_CONFIG = {
    "ensemble": {
        "random_forest": {
            "n_estimators": 200,
            "max_depth": 10,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "random_state": 42,
            "n_jobs": -1
        },
        "gradient_boosting": {
            "n_estimators": 150,
            "learning_rate": 0.05,
            "max_depth": 5,
            "random_state": 42
        },
        "xgboost": {
            "n_estimators": 200,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42
        }
    },
    "training": {
        "test_size": 0.2,
        "cv_folds": 5,
        "scoring": "roc_auc",
        "n_samples": 5000
    },
    "features": [
        'home_goals_avg', 'away_goals_avg', 'home_goals_conceded_avg', 
        'away_goals_conceded_avg', 'home_over25_rate', 'away_over25_rate',
        'h2h_over25_rate', 'home_form', 'away_form', 'total_matches',
        'league_over25_avg', 'combined_attack_strength', 'defensive_weakness',
        'recent_form_diff', 'home_advantage', 'fatigue_index'
    ]
}

# Betting Configuration
BETTING_CONFIG = {
    "min_probability": 0.55,
    "max_probability": 0.90,
    "kelly_fraction": 0.25,  # Conservative Kelly
    "max_stake_percent": 0.05,  # Max 5% of bankroll per bet
    "min_odds": 1.15,
    "max_odds": 4.0,
    "bet_types": {
        "single": {"size": 1, "min_prob": 0.65},
        "double": {"size": 2, "min_prob": 0.62},
        "triple": {"size": 3, "min_prob": 0.60},
        "quadruple": {"size": 4, "min_prob": 0.58},
        "quintuple": {"size": 5, "min_prob": 0.55}
    }
}

# Cache Configuration
CACHE_CONFIG = {
    "default_ttl": 3600,  # 1 hour
    "team_stats_ttl": 7200,  # 2 hours
    "fixtures_ttl": 1800,  # 30 minutes
    "max_cache_size": 100 * 1024 * 1024,  # 100MB
    "cleanup_interval": 3600  # 1 hour
}

# UI Configuration
UI_CONFIG = {
    "theme": "dark",
    "primary_color": "#00d4aa",
    "secondary_color": "#0066cc",
    "danger_color": "#ff4757",
    "warning_color": "#ffa502",
    "success_color": "#00d4aa",
    "page_icon": "⚽",
    "layout": "wide",
    "sidebar_state": "expanded"
}

# League Configuration
LEAGUES = {
    # Europe - Top 5
    "🏴󠁧󠁢󠁥󠁮󠁧󠁿 Premier League": {"id": 39, "country": "England", "region": "Europe", "tier": 1},
    "🇮🇹 Serie A": {"id": 135, "country": "Italy", "region": "Europe", "tier": 1},
    "🇪🇸 La Liga": {"id": 140, "country": "Spain", "region": "Europe", "tier": 1},
    "🇩🇪 Bundesliga": {"id": 78, "country": "Germany", "region": "Europe", "tier": 1},
    "🇫🇷 Ligue 1": {"id": 61, "country": "France", "region": "Europe", "tier": 1},
    
    # Europe - Others
    "🇳🇱 Eredivisie": {"id": 88, "country": "Netherlands", "region": "Europe", "tier": 2},
    "🇵🇹 Primeira Liga": {"id": 94, "country": "Portugal", "region": "Europe", "tier": 2},
    "🇧🇪 Pro League": {"id": 144, "country": "Belgium", "region": "Europe", "tier": 2},
    "🏴󠁧󠁢󠁥󠁮󠁧󠁿 Championship": {"id": 40, "country": "England", "region": "Europe", "tier": 2},
    "🇹🇷 Super Lig": {"id": 203, "country": "Turkey", "region": "Europe", "tier": 2},
    
    # Americas
    "🇧🇷 Serie A": {"id": 71, "country": "Brazil", "region": "Americas", "tier": 1},
    "🇦🇷 Primera División": {"id": 128, "country": "Argentina", "region": "Americas", "tier": 1},
    "🇺🇸 MLS": {"id": 253, "country": "USA", "region": "Americas", "tier": 2},
    "🇲🇽 Liga MX": {"id": 262, "country": "Mexico", "region": "Americas", "tier": 2},
    
    # Asia
    "🇯🇵 J1 League": {"id": 98, "country": "Japan", "region": "Asia", "tier": 1},
    
    # International
    "🏆 Champions League": {"id": 2, "country": "Europe", "region": "International", "tier": 1},
    "🏅 Europa League": {"id": 3, "country": "Europe", "region": "International", "tier": 1},
}

# Logging Configuration
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "[%(asctime)s] %(levelname)s in %(module)s: %(message)s",
        },
        "detailed": {
            "format": "[%(asctime)s] %(levelname)s in %(module)s.%(funcName)s:%(lineno)d: %(message)s",
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "default",
            "stream": "ext://sys.stdout"
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "DEBUG",
            "formatter": "detailed",
            "filename": str(DATA_DIR / "app.log"),
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5
        }
    },
    "loggers": {
        "profootball": {
            "level": "DEBUG",
            "handlers": ["console", "file"],
            "propagate": False
        }
    }
}

# Validation Rules
VALIDATION_RULES = {
    "probability": {"min": 0.0, "max": 1.0},
    "odds": {"min": 1.01, "max": 100.0},
    "stake": {"min": 1, "max": 10000},
    "matches": {"min": 1, "max": 50},
    "goals": {"min": 0, "max": 20}
}

# Feature Engineering Configuration
FEATURE_CONFIG = {
    "rolling_windows": [3, 5, 10],  # Last N matches
    "decay_factor": 0.95,  # Weight decay for older matches
    "home_advantage_factor": 1.1,
    "fatigue_threshold": 3,  # Days between matches
    "form_weights": {
        "win": 3,
        "draw": 1,
        "loss": 0
    }
}

def get_current_season() -> int:
    """Get current football season based on date"""
    now = datetime.now()
    if now.month >= 8:
        return now.year
    return now.year - 1

def get_league_season(league_name: str) -> int:
    """Get appropriate season for a league"""
    league_info = LEAGUES.get(league_name, {})
    if league_info.get("country") in ["Brazil", "USA", "Mexico", "Japan"]:
        return datetime.now().year
    return get_current_season()

# Export all configurations
__all__ = [
    'BASE_DIR', 'DATA_DIR', 'CACHE_DIR', 'DB_PATH',
    'API_CONFIG', 'MODEL_CONFIG', 'BETTING_CONFIG',
    'CACHE_CONFIG', 'UI_CONFIG', 'LEAGUES',
    'LOGGING_CONFIG', 'VALIDATION_RULES', 'FEATURE_CONFIG',
    'get_current_season', 'get_league_season'
]
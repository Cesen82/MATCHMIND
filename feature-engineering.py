"""
Advanced Feature Engineering for Over 2.5 Goals Prediction
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime, timedelta
import logging
from sklearn.preprocessing import PolynomialFeatures
from scipy import stats

from config import FEATURE_CONFIG

logger = logging.getLogger("profootball.features")


class FeatureEngineer:
    """Advanced feature engineering for football predictions"""
    
    def __init__(self):
        self.rolling_windows = FEATURE_CONFIG["rolling_windows"]
        self.decay_factor = FEATURE_CONFIG["decay_factor"]
        self.home_advantage = FEATURE_CONFIG["home_advantage_factor"]
        self.fatigue_threshold = FEATURE_CONFIG["fatigue_threshold"]
        self.form_weights = FEATURE_CONFIG["form_weights"]
        
        # Polynomial features for interactions
        self.poly = PolynomialFeatures(degree=2, include_bias=False)
        
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform raw features into engineered features"""
        logger.debug(f"Engineering features for {len(df)} samples")
        
        # Create a copy to avoid modifying original
        features = df.copy()
        
        # Basic engineered features
        features = self._create_basic_features(features)
        
        # Time-based features
        features = self._create_time_features(features)
        
        # Interaction features
        features = self._create_interaction_features(features)
        
        # Statistical features
        features = self._create_statistical_features(features)
        
        # Momentum features
        features = self._create_momentum_features(features)
        
        # Clean up
        features = self._clean_features(features)
        
        logger.debug(f"Created {len(features.columns)} features")
        
        return features
        
    def _create_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create basic engineered features"""
        
        # Total goals expectation
        df['total_goals_expected'] = df['home_goals_avg'] + df['away_goals_avg']
        
        # Defensive weakness
        df['defensive_weakness'] = (
            df['home_goals_conceded_avg'] + df['away_goals_conceded_avg']
        ) / 2
        
        # Attack vs Defense balance
        df['attack_defense_ratio'] = (
            df['total_goals_expected'] / 
            (df['defensive_weakness'] + 0.1)  # Avoid division by zero
        )
        
        # Combined Over 2.5 rate
        df['combined_over25_rate'] = (
            df['home_over25_rate'] + df['away_over25_rate']
        ) / 2
        
        # Form difference
        if 'home_form' in df.columns and 'away_form' in df.columns:
            df['form_difference'] = df['home_form'] - df['away_form']
            df['total_form'] = df['home_form'] + df['away_form']
        
        # Goals per match ratios
        df['home_goal_ratio'] = df['home_goals_avg'] / (df['home_goals_conceded_avg'] + 0.1)
        df['away_goal_ratio'] = df['away_goals_avg'] / (df['away_goals_conceded_avg'] + 0.1)
        
        # Scoring consistency (lower is better)
        if 'home_goals_std' in df.columns:
            df['home_scoring_consistency'] = df['home_goals_std'] / (df['home_goals_avg'] + 0.1)
            df['away_scoring_consistency'] = df['away_goals_std'] / (df['away_goals_avg'] + 0.1)
        
        return df
        
    def _create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features"""
        
        # Match timing features (if datetime available)
        if 'match_date' in df.columns:
            df['match_date'] = pd.to_datetime(df['match_date'])
            df['day_of_week'] = df['match_date'].dt.dayofweek
            df['month'] = df['match_date'].dt.month
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
            
            # Season progress (0-1)
            df['season_progress'] = (df['match_date'].dt.dayofyear - 213) / 365  # Aug 1 = day 213
            df['season_progress'] = df['season_progress'].clip(0, 1)
            
        # Fatigue index (if last match date available)
        if 'home_last_match_date' in df.columns:
            df['home_days_rest'] = (
                df['match_date'] - pd.to_datetime(df['home_last_match_date'])
            ).dt.days
            df['home_fatigue'] = (
                df['home_days_rest'] < self.fatigue_threshold
            ).astype(int)
            
        if 'away_last_match_date' in df.columns:
            df['away_days_rest'] = (
                df['match_date'] - pd.to_datetime(df['away_last_match_date'])
            ).dt.days
            df['away_fatigue'] = (
                df['away_days_rest'] < self.fatigue_threshold
            ).astype(int)
            
        # Match importance (if available)
        if 'league_position_home' in df.columns:
            # Top 6 clash
            df['top_clash'] = (
                (df['league_position_home'] <= 6) & 
                (df['league_position_away'] <= 6)
            ).astype(int)
            
            # Relegation battle
            df['relegation_battle'] = (
                (df['league_position_home'] >= 15) & 
                (df['league_position_away'] >= 15)
            ).astype(int)
            
        return df
        
    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between teams"""
        
        # Goal interactions
        df['goals_product'] = df['home_goals_avg'] * df['away_goals_avg']
        df['defense_product'] = df['home_goals_conceded_avg'] * df['away_goals_conceded_avg']
        
        # Style clash
        df['style_clash'] = np.abs(
            df['home_goals_avg'] - df['away_goals_avg']
        ) + np.abs(
            df['home_goals_conceded_avg'] - df['away_goals_conceded_avg']
        )
        
        # Momentum interaction
        if 'home_form' in df.columns:
            df['form_product'] = df['home_form'] * df['away_form']
            df['form_momentum'] = df['home_form'] + df['away_form'] - 10  # Baseline of 5+5
        
        # H2H adjustments
        if 'h2h_over25_rate' in df.columns:
            # Weight H2H based on sample size
            h2h_weight = np.minimum(df.get('h2h_matches', 5) / 10, 1.0)
            df['weighted_h2h_rate'] = (
                df['h2h_over25_rate'] * h2h_weight +
                df['combined_over25_rate'] * (1 - h2h_weight)
            )
        
        # Pressure index
        if 'total_matches' in df.columns:
            df['pressure_index'] = (
                (38 - df['total_matches']) / 38 *  # Late season
                df.get('top_clash', 0.5)  # Important match
            )
        
        return df
        
    def _create_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create statistical features"""
        
        # Poisson probability for Over 2.5
        df['poisson_over25_home'] = 1 - stats.poisson.cdf(
            2, df['home_goals_avg'] + df['home_goals_conceded_avg']
        )
        df['poisson_over25_away'] = 1 - stats.poisson.cdf(
            2, df['away_goals_avg'] + df['away_goals_conceded_avg']
        )
        df['poisson_over25_combined'] = (
            df['poisson_over25_home'] + df['poisson_over25_away']
        ) / 2
        
        # Z-scores for anomaly detection
        if len(df) > 1:
            for col in ['home_goals_avg', 'away_goals_avg']:
                if col in df.columns:
                    df[f'{col}_zscore'] = np.abs(stats.zscore(df[col]))
        
        # Entropy of scoring (randomness)
        if 'home_goals_distribution' in df.columns:
            # This would need actual distribution data
            pass
        
        return df
        
    def _create_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create momentum and trend features"""
        
        # Recent form momentum (if we have match-by-match data)
        if 'home_last_5_goals' in df.columns:
            # Weighted average giving more weight to recent matches
            weights = np.array([self.decay_factor ** i for i in range(5)])
            weights = weights / weights.sum()
            
            # This is simplified - in reality we'd have arrays
            df['home_momentum'] = df['home_form'] * 1.1  # Placeholder
            df['away_momentum'] = df['away_form'] * 1.1  # Placeholder
        
        # Streak features
        if 'home_win_streak' in df.columns:
            df['home_hot_streak'] = (df['home_win_streak'] >= 3).astype(int)
            df['away_hot_streak'] = (df['away_win_streak'] >= 3).astype(int)
        
        # Goal scoring trends
        if 'home_goals_last_3' in df.columns:
            df['home_scoring_trend'] = (
                df['home_goals_last_3'] - df['home_goals_avg'] * 3
            ) / 3
            df['away_scoring_trend'] = (
                df['away_goals_last_3'] - df['away_goals_avg'] * 3
            ) / 3
        
        return df
        
    def _create_polynomial_features(self, df: pd.DataFrame, key_features: List[str]) -> pd.DataFrame:
        """Create polynomial features for key variables"""
        
        # Select subset of features for polynomial expansion
        available_features = [f for f in key_features if f in df.columns]
        
        if len(available_features) >= 2:
            poly_features = self.poly.fit_transform(df[available_features])
            poly_names = self.poly.get_feature_names_out(available_features)
            
            # Add only interaction terms (not squares)
            for i, name in enumerate(poly_names):
                if ' ' in name:  # Interaction term
                    df[f'poly_{name}'] = poly_features[:, i]
        
        return df
        
    def _clean_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate features"""
        
        # Handle infinities
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Fill NaN values with sensible defaults
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            if df[col].isna().any():
                # Use median for most features
                if 'rate' in col or 'ratio' in col:
                    fill_value = 0.5
                elif 'avg' in col:
                    fill_value = df[col].median() if not df[col].isna().all() else 1.0
                else:
                    fill_value = 0
                    
                df[col] = df[col].fillna(fill_value)
        
        # Clip extreme values
        for col in numeric_columns:
            if 'rate' in col or 'probability' in col:
                df[col] = df[col].clip(0, 1)
            elif 'goals' in col and 'avg' in col:
                df[col] = df[col].clip(0, 5)
        
        return df
        
    def get_feature_importance_groups(self) -> Dict[str, List[str]]:
        """Get feature groups for importance analysis"""
        
        return {
            "Team Strength": [
                "home_goals_avg", "away_goals_avg", 
                "home_goals_conceded_avg", "away_goals_conceded_avg"
            ],
            "Over 2.5 History": [
                "home_over25_rate", "away_over25_rate", 
                "combined_over25_rate", "h2h_over25_rate"
            ],
            "Form & Momentum": [
                "home_form", "away_form", "form_difference", 
                "home_momentum", "away_momentum"
            ],
            "Statistical": [
                "poisson_over25_combined", "total_goals_expected",
                "defensive_weakness"
            ],
            "Match Context": [
                "is_weekend", "top_clash", "pressure_index",
                "home_fatigue", "away_fatigue"
            ]
        }
        
    def explain_features(self, features: pd.Series) -> str:
        """Generate human-readable explanation of features"""
        
        explanations = []
        
        # Goal expectations
        total_expected = features.get('total_goals_expected', 0)
        if total_expected > 3.0:
            explanations.append(f"High scoring match expected ({total_expected:.1f} goals)")
        elif total_expected < 2.0:
            explanations.append(f"Low scoring match expected ({total_expected:.1f} goals)")
            
        # Form
        form_diff = features.get('form_difference', 0)
        if abs(form_diff) > 3:
            better_team = "Home" if form_diff > 0 else "Away"
            explanations.append(f"{better_team} team in much better form")
            
        # H2H
        h2h_rate = features.get('h2h_over25_rate', 0.5)
        if h2h_rate > 0.7:
            explanations.append(f"H2H history favors Over 2.5 ({h2h_rate:.0%})")
        elif h2h_rate < 0.3:
            explanations.append(f"H2H history favors Under 2.5 ({h2h_rate:.0%})")
            
        return " | ".join(explanations) if explanations else "Standard match profile"
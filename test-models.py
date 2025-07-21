"""
Tests for models module
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
from datetime import datetime

from src.models.predictor import Over25Predictor, PredictionResult
from src.models.feature_engineering import FeatureEngineer
from src.models.bet_optimizer import BetOptimizer, Match, BettingSlip
from src.utils.exceptions import ModelNotTrainedError, InvalidFeaturesError


class TestOver25Predictor:
    """Test cases for Over25Predictor"""
    
    @pytest.fixture
    def predictor(self):
        """Create predictor instance"""
        return Over25Predictor(model_type="random_forest")
    
    @pytest.fixture
    def sample_features(self):
        """Create sample features"""
        return {
            'home_goals_avg': 1.5,
            'away_goals_avg': 1.3,
            'home_goals_conceded_avg': 1.2,
            'away_goals_conceded_avg': 1.4,
            'home_over25_rate': 0.6,
            'away_over25_rate': 0.55,
            'h2h_over25_rate': 0.58,
            'home_form': 7,
            'away_form': 6,
            'total_matches': 30,
            'league_over25_avg': 0.57,
            'combined_attack_strength': 1.4
        }
    
    @pytest.fixture
    def training_data(self):
        """Create sample training data"""
        n_samples = 100
        np.random.seed(42)
        
        data = {
            'home_goals_avg': np.random.uniform(0.5, 2.5, n_samples),
            'away_goals_avg': np.random.uniform(0.5, 2.5, n_samples),
            'home_goals_conceded_avg': np.random.uniform(0.5, 2.0, n_samples),
            'away_goals_conceded_avg': np.random.uniform(0.5, 2.0, n_samples),
            'home_over25_rate': np.random.uniform(0.2, 0.8, n_samples),
            'away_over25_rate': np.random.uniform(0.2, 0.8, n_samples),
            'h2h_over25_rate': np.random.uniform(0.2, 0.8, n_samples),
            'home_form': np.random.randint(0, 10, n_samples),
            'away_form': np.random.randint(0, 10, n_samples),
            'total_matches': np.random.randint(20, 38, n_samples),
            'league_over25_avg': np.random.uniform(0.4, 0.7, n_samples),
            'combined_attack_strength': np.random.uniform(0.8, 2.0, n_samples),
            'over25': np.random.randint(0, 2, n_samples)
        }
        
        return pd.DataFrame(data)
    
    def test_init(self, predictor):
        """Test predictor initialization"""
        assert predictor.model_type == "random_forest"
        assert predictor.model is None
        assert not predictor.is_trained
        assert len(predictor.feature_names) == 0
    
    def test_train_model(self, predictor, training_data):
        """Test model training"""
        X = training_data.drop('over25', axis=1)
        y = training_data['over25']
        
        metrics = predictor.train(X, y, validate=True)
        
        assert predictor.is_trained
        assert predictor.model is not None
        assert 'accuracy' in metrics
        assert 'auc_roc' in metrics
        assert metrics['accuracy'] > 0.5  # Better than random
    
    def test_predict_untrained(self, predictor, sample_features):
        """Test prediction with untrained model"""
        # Should train automatically or raise error
        with patch.object(predictor, '_load_model', side_effect=FileNotFoundError):
            with patch.object(predictor, 'train'):
                result = predictor.predict(sample_features)
                assert isinstance(result, PredictionResult)
    
    def test_predict_trained(self, predictor, training_data, sample_features):
        """Test prediction with trained model"""
        # Train model first
        X = training_data.drop('over25', axis=1)
        y = training_data['over25']
        predictor.train(X, y)
        
        # Make prediction
        result = predictor.predict(sample_features)
        
        assert isinstance(result, PredictionResult)
        assert result.prediction in [0, 1]
        assert 0 <= result.probability <= 1
        assert result.confidence in ["High", "Medium", "Low"]
        assert isinstance(result.feature_importance, dict)
    
    def test_batch_predict(self, predictor, training_data):
        """Test batch prediction"""
        # Train model
        X = training_data.drop('over25', axis=1)
        y = training_data['over25']
        predictor.train(X, y)
        
        # Create batch features
        batch_features = [
            {col: X.iloc[i][col] for col in X.columns}
            for i in range(5)
        ]
        
        results = predictor.batch_predict(batch_features)
        
        assert len(results) == 5
        assert all(isinstance(r, PredictionResult) for r in results)
    
    def test_feature_importance(self, predictor, training_data):
        """Test feature importance extraction"""
        X = training_data.drop('over25', axis=1)
        y = training_data['over25']
        predictor.train(X, y)
        
        # Create dummy features for importance
        dummy_features = pd.DataFrame([{col: 1.0 for col in X.columns}])
        importance = predictor._get_feature_importance(dummy_features)
        
        assert isinstance(importance, dict)
        assert len(importance) > 0
        assert all(0 <= v <= 1 for v in importance.values())
    
    def test_model_info(self, predictor, training_data):
        """Test getting model information"""
        # Before training
        info = predictor.get_model_info()
        assert not info['is_trained']
        
        # After training
        X = training_data.drop('over25', axis=1)
        y = training_data['over25']
        predictor.train(X, y)
        
        info = predictor.get_model_info()
        assert info['is_trained']
        assert info['model_type'] == "random_forest"
        assert 'performance_metrics' in info
        assert 'top_features' in info
    
    def test_explain_prediction(self, predictor, training_data, sample_features):
        """Test prediction explanation"""
        X = training_data.drop('over25', axis=1)
        y = training_data['over25']
        predictor.train(X, y)
        
        result = predictor.predict(sample_features)
        explanation = predictor.explain_prediction(sample_features, result)
        
        assert isinstance(explanation, str)
        assert "probability" in explanation.lower()
        assert "confidence" in explanation.lower()


class TestFeatureEngineer:
    """Test cases for FeatureEngineer"""
    
    @pytest.fixture
    def engineer(self):
        """Create feature engineer instance"""
        return FeatureEngineer()
    
    @pytest.fixture
    def raw_data(self):
        """Create raw data for feature engineering"""
        return pd.DataFrame({
            'home_goals_avg': [1.5, 2.0, 1.2],
            'away_goals_avg': [1.3, 1.8, 1.0],
            'home_goals_conceded_avg': [1.2, 1.0, 1.5],
            'away_goals_conceded_avg': [1.4, 1.2, 1.8],
            'home_over25_rate': [0.6, 0.7, 0.4],
            'away_over25_rate': [0.55, 0.65, 0.35],
            'h2h_over25_rate': [0.58, 0.68, 0.38],
            'home_form': [7, 8, 4],
            'away_form': [6, 7, 3],
            'total_matches': [30, 32, 28],
            'match_date': pd.to_datetime(['2024-01-15', '2024-02-20', '2024-03-25'])
        })
    
    def test_transform(self, engineer, raw_data):
        """Test feature transformation"""
        transformed = engineer.transform(raw_data)
        
        # Check new features are created
        assert len(transformed.columns) > len(raw_data.columns)
        assert 'total_goals_expected' in transformed.columns
        assert 'defensive_weakness' in transformed.columns
        assert 'attack_defense_ratio' in transformed.columns
    
    def test_basic_features(self, engineer, raw_data):
        """Test basic feature creation"""
        features = engineer._create_basic_features(raw_data)
        
        # Check calculations
        assert np.allclose(
            features['total_goals_expected'],
            raw_data['home_goals_avg'] + raw_data['away_goals_avg']
        )
        
        assert 'form_difference' in features.columns
        assert 'combined_over25_rate' in features.columns
    
    def test_time_features(self, engineer, raw_data):
        """Test time-based feature creation"""
        features = engineer._create_time_features(raw_data)
        
        assert 'day_of_week' in features.columns
        assert 'month' in features.columns
        assert 'is_weekend' in features.columns
        assert 'season_progress' in features.columns
    
    def test_interaction_features(self, engineer, raw_data):
        """Test interaction feature creation"""
        # First create basic features
        features = engineer._create_basic_features(raw_data)
        features = engineer._create_interaction_features(features)
        
        assert 'goals_product' in features.columns
        assert 'defense_product' in features.columns
        assert 'style_clash' in features.columns
    
    def test_statistical_features(self, engineer, raw_data):
        """Test statistical feature creation"""
        features = engineer._create_statistical_features(raw_data)
        
        assert 'poisson_over25_home' in features.columns
        assert 'poisson_over25_away' in features.columns
        assert 'poisson_over25_combined' in features.columns
        
        # Check Poisson probabilities are valid
        assert all(0 <= features['poisson_over25_home'] <= 1)
        assert all(0 <= features['poisson_over25_away'] <= 1)
    
    def test_clean_features(self, engineer):
        """Test feature cleaning"""
        # Create data with issues
        dirty_data = pd.DataFrame({
            'rate_feature': [0.5, 1.5, -0.1, np.inf],
            'goals_avg': [1.5, 10.0, -1.0, np.nan]
        })
        
        cleaned = engineer._clean_features(dirty_data)
        
        # Check cleaning
        assert all(0 <= cleaned['rate_feature'] <= 1)
        assert all(0 <= cleaned['goals_avg'] <= 5)
        assert not cleaned.isna().any().any()
        assert not np.isinf(cleaned.values).any()


class TestBetOptimizer:
    """Test cases for BetOptimizer"""
    
    @pytest.fixture
    def optimizer(self):
        """Create bet optimizer instance"""
        return BetOptimizer()
    
    @pytest.fixture
    def sample_matches(self):
        """Create sample matches"""
        matches = []
        for i in range(5):
            prediction = Mock(spec=PredictionResult)
            prediction.probability = 0.6 + i * 0.05
            prediction.confidence = "High" if i < 2 else "Medium"
            prediction.confidence_score = 0.7 + i * 0.05
            prediction.risk_factors = []
            
            match = Match(
                match_id=f"match_{i}",
                home_team=f"Home {i}",
                away_team=f"Away {i}",
                date="2024-01-20",
                prediction=prediction,
                odds=1.8 + i * 0.1
            )
            matches.append(match)
        
        return matches
    
    def test_optimize_portfolio(self, optimizer, sample_matches):
        """Test portfolio optimization"""
        portfolio = optimizer.optimize_portfolio(
            matches=sample_matches,
            bankroll=1000,
            max_slips=5,
            risk_tolerance="medium"
        )
        
        assert isinstance(portfolio, list)
        assert len(portfolio) <= 5
        assert all(isinstance(slip, BettingSlip) for slip in portfolio)
        
        # Check portfolio properties
        for slip in portfolio:
            assert slip.expected_value > 0  # Only profitable bets
            assert 0 <= slip.recommended_stake <= 0.05  # Max 5% per bet
    
    def test_kelly_calculation(self, optimizer, sample_matches):
        """Test Kelly criterion calculation"""
        match = sample_matches[0]
        kelly = match.kelly_stake
        
        assert 0 <= kelly <= 1
        
        # Manual calculation
        p = match.prediction.probability
        b = match.odds - 1
        q = 1 - p
        expected_kelly = max(0, (p * b - q) / b)
        
        assert np.isclose(kelly, expected_kelly)
    
    def test_risk_assessment(self, optimizer):
        """Test risk level assessment"""
        # High confidence, low variance
        risk1 = optimizer._assess_risk_level(0.8, 0.05)
        assert risk1 == "Low"
        
        # Medium confidence
        risk2 = optimizer._assess_risk_level(0.65, 0.1)
        assert risk2 == "Medium"
        
        # Low confidence or high variance
        risk3 = optimizer._assess_risk_level(0.55, 0.2)
        assert risk3 == "High"
    
    def test_diversification_score(self, optimizer, sample_matches):
        """Test diversification score calculation"""
        # Same date matches
        matches_same_date = sample_matches[:3]
        for m in matches_same_date:
            m.date = "2024-01-20"
        
        score1 = optimizer._calculate_diversification_score(matches_same_date)
        
        # Different date matches
        for i, m in enumerate(matches_same_date):
            m.date = f"2024-01-{20+i}"
        
        score2 = optimizer._calculate_diversification_score(matches_same_date)
        
        assert score2 > score1  # Better diversification
        assert 0 <= score1 <= 1
        assert 0 <= score2 <= 1
    
    def test_correlation_penalty(self, optimizer, sample_matches):
        """Test correlation penalty calculation"""
        # Same day penalty
        matches = sample_matches[:2]
        for m in matches:
            m.date = "2024-01-20"
        
        penalty1 = optimizer._estimate_correlation_penalty(matches)
        assert penalty1 > 0
        
        # Different days
        matches[0].date = "2024-01-20"
        matches[1].date = "2024-01-21"
        
        penalty2 = optimizer._estimate_correlation_penalty(matches)
        assert penalty2 < penalty1
    
    def test_portfolio_metrics(self, optimizer, sample_matches):
        """Test portfolio metrics calculation"""
        # Create sample portfolio
        portfolio = []
        for i in range(3):
            slip = BettingSlip(
                matches=[sample_matches[i]],
                slip_type="single",
                total_odds=sample_matches[i].odds,
                combined_probability=sample_matches[i].prediction.probability,
                expected_value=sample_matches[i].expected_value,
                kelly_stake=sample_matches[i].kelly_stake,
                risk_level="Medium",
                confidence_score=0.7,
                diversification_score=0.8,
                correlation_penalty=0.1,
                recommended_stake=0.02
            )
            portfolio.append(slip)
        
        metrics = optimizer.calculate_portfolio_metrics(portfolio, 1000)
        
        assert 'total_slips' in metrics
        assert 'total_stake' in metrics
        assert 'expected_return' in metrics
        assert 'expected_roi' in metrics
        assert metrics['total_slips'] == 3
        assert metrics['total_stake'] == 60  # 3 * 0.02 * 1000
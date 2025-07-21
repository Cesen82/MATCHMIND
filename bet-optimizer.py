"""
Advanced Betting Slip Optimizer with Kelly Criterion and Risk Management
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from itertools import combinations
import logging
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

from config import BETTING_CONFIG
from .predictor import PredictionResult

logger = logging.getLogger("profootball.betting")


@dataclass
class Match:
    """Match information for betting"""
    match_id: str
    home_team: str
    away_team: str
    date: str
    prediction: PredictionResult
    odds: float
    
    @property
    def expected_value(self) -> float:
        return (self.prediction.probability * self.odds) - 1
    
    @property
    def kelly_stake(self) -> float:
        """Calculate Kelly stake for single bet"""
        p = self.prediction.probability
        b = self.odds - 1
        q = 1 - p
        
        kelly = (p * b - q) / b
        return max(0, kelly)


@dataclass
class BettingSlip:
    """Optimized betting slip"""
    matches: List[Match]
    slip_type: str  # single, double, triple, etc.
    total_odds: float
    combined_probability: float
    expected_value: float
    kelly_stake: float
    risk_level: str
    confidence_score: float
    diversification_score: float
    correlation_penalty: float
    recommended_stake: float  # As percentage of bankroll
    
    @property
    def potential_return(self) -> float:
        return self.total_odds * self.recommended_stake
    
    @property
    def roi(self) -> float:
        return (self.potential_return - self.recommended_stake) / self.recommended_stake


class BetOptimizer:
    """Advanced betting slip optimizer with portfolio theory"""
    
    def __init__(self):
        self.kelly_fraction = BETTING_CONFIG["kelly_fraction"]
        self.max_stake_percent = BETTING_CONFIG["max_stake_percent"]
        self.correlation_estimator = LedoitWolf()
        
    def optimize_portfolio(
        self, 
        matches: List[Match], 
        bankroll: float = 1000,
        max_slips: int = 10,
        risk_tolerance: str = "medium"
    ) -> List[BettingSlip]:
        """Optimize betting portfolio using modern portfolio theory"""
        
        if len(matches) < 1:
            return []
            
        # Filter matches by expected value
        profitable_matches = [m for m in matches if m.expected_value > 0]
        
        if not profitable_matches:
            logger.warning("No profitable matches found")
            return []
            
        logger.info(f"Optimizing portfolio with {len(profitable_matches)} profitable matches")
        
        # Generate all possible betting slips
        all_slips = []
        
        # Singles
        for match in profitable_matches:
            if match.prediction.probability >= BETTING_CONFIG["bet_types"]["single"]["min_prob"]:
                slip = self._create_single_slip(match)
                all_slips.append(slip)
                
        # Multiples (doubles, triples, etc.)
        for bet_type, config in BETTING_CONFIG["bet_types"].items():
            if bet_type == "single":
                continue
                
            size = config["size"]
            min_prob = config["min_prob"]
            
            # Filter eligible matches
            eligible = [m for m in profitable_matches if m.prediction.probability >= min_prob]
            
            if len(eligible) >= size:
                # Generate combinations
                for combo in combinations(eligible, size):
                    slip = self._create_multi_slip(list(combo), bet_type)
                    if slip.expected_value > 0:
                        all_slips.append(slip)
                        
        # Sort by expected value
        all_slips.sort(key=lambda x: x.expected_value, reverse=True)
        
        # Apply portfolio optimization
        optimized_slips = self._apply_portfolio_optimization(
            all_slips[:50],  # Consider top 50 slips
            bankroll,
            risk_tolerance
        )
        
        # Select final portfolio
        final_portfolio = self._select_final_portfolio(
            optimized_slips,
            max_slips,
            risk_tolerance
        )
        
        return final_portfolio
        
    def _create_single_slip(self, match: Match) -> BettingSlip:
        """Create single bet slip"""
        kelly_stake = match.kelly_stake * self.kelly_fraction
        
        return BettingSlip(
            matches=[match],
            slip_type="single",
            total_odds=match.odds,
            combined_probability=match.prediction.probability,
            expected_value=match.expected_value,
            kelly_stake=kelly_stake,
            risk_level=self._assess_risk_level(match.prediction.probability, 0.0),
            confidence_score=match.prediction.confidence_score,
            diversification_score=1.0,  # Single bet has no diversification
            correlation_penalty=0.0,
            recommended_stake=min(kelly_stake, self.max_stake_percent)
        )
        
    def _create_multi_slip(self, matches: List[Match], bet_type: str) -> BettingSlip:
        """Create multiple bet slip with correlation adjustment"""
        # Calculate total odds
        total_odds = np.prod([m.odds for m in matches])
        
        # Calculate combined probability with correlation
        base_probabilities = [m.prediction.probability for m in matches]
        correlation_penalty = self._estimate_correlation_penalty(matches)
        
        # Adjust combined probability for correlation
        combined_prob = np.prod(base_probabilities)
        adjusted_prob = combined_prob * (1 - correlation_penalty)
        
        # Calculate expected value
        expected_value = (adjusted_prob * total_odds) - 1
        
        # Kelly stake for multiple
        kelly_stake = self._calculate_multi_kelly(
            adjusted_prob,
            total_odds,
            len(matches)
        )
        
        # Assess confidence and risk
        avg_confidence = np.mean([m.prediction.confidence_score for m in matches])
        prob_std = np.std(base_probabilities)
        risk_level = self._assess_risk_level(np.mean(base_probabilities), prob_std)
        
        # Diversification score
        diversification_score = self._calculate_diversification_score(matches)
        
        return BettingSlip(
            matches=matches,
            slip_type=bet_type,
            total_odds=round(total_odds, 2),
            combined_probability=adjusted_prob,
            expected_value=expected_value,
            kelly_stake=kelly_stake * self.kelly_fraction,
            risk_level=risk_level,
            confidence_score=avg_confidence,
            diversification_score=diversification_score,
            correlation_penalty=correlation_penalty,
            recommended_stake=min(kelly_stake * self.kelly_fraction, self.max_stake_percent)
        )
        
    def _estimate_correlation_penalty(self, matches: List[Match]) -> float:
        """Estimate correlation between matches"""
        penalty = 0.0
        
        # Same day matches have higher correlation
        dates = [m.date for m in matches]
        if len(set(dates)) == 1:
            penalty += 0.1
            
        # Same league matches
        leagues = []
        for m in matches:
            # Extract league from match info if available
            leagues.append("default")  # Simplified
            
        if len(set(leagues)) == 1:
            penalty += 0.05
            
        # Teams playing each other (avoid)
        all_teams = set()
        for m in matches:
            all_teams.add(m.home_team)
            all_teams.add(m.away_team)
            
        if len(all_teams) < len(matches) * 2:
            penalty += 0.15
            
        return min(penalty, 0.3)  # Cap at 30%
        
    def _calculate_multi_kelly(self, prob: float, odds: float, n_bets: int) -> float:
        """Calculate Kelly stake for multiple bets"""
        b = odds - 1
        p = prob
        q = 1 - p
        
        # Adjusted Kelly for multiples
        kelly = (p * b - q) / b
        
        # Reduce stake based on number of selections
        reduction_factor = 1 / np.sqrt(n_bets)
        
        return max(0, kelly * reduction_factor)
        
    def _assess_risk_level(self, avg_prob: float, std_prob: float) -> str:
        """Assess risk level of bet"""
        if avg_prob >= 0.75 and std_prob < 0.1:
            return "Low"
        elif avg_prob >= 0.65 or (avg_prob >= 0.60 and std_prob < 0.15):
            return "Medium"
        else:
            return "High"
            
    def _calculate_diversification_score(self, matches: List[Match]) -> float:
        """Calculate diversification score for multiple bets"""
        # Different dates
        dates = len(set([m.date for m in matches]))
        date_score = dates / len(matches)
        
        # Different teams
        all_teams = set()
        for m in matches:
            all_teams.add(m.home_team)
            all_teams.add(m.away_team)
        team_score = len(all_teams) / (len(matches) * 2)
        
        # Combined score
        return (date_score + team_score) / 2
        
    def _apply_portfolio_optimization(
        self,
        slips: List[BettingSlip],
        bankroll: float,
        risk_tolerance: str
    ) -> List[BettingSlip]:
        """Apply Markowitz portfolio optimization"""
        
        if not slips:
            return []
            
        # Create return matrix
        returns = np.array([s.expected_value for s in slips])
        
        # Estimate covariance matrix (simplified)
        n_slips = len(slips)
        cov_matrix = np.eye(n_slips) * 0.1  # Base variance
        
        # Add correlation based on shared matches
        for i in range(n_slips):
            for j in range(i+1, n_slips):
                shared = self._count_shared_matches(slips[i], slips[j])
                if shared > 0:
                    correlation = shared / max(len(slips[i].matches), len(slips[j].matches))
                    cov_matrix[i, j] = correlation * 0.05
                    cov_matrix[j, i] = correlation * 0.05
                    
        # Risk tolerance parameters
        risk_params = {
            "conservative": {"risk_aversion": 5.0, "max_position": 0.02},
            "medium": {"risk_aversion": 2.0, "max_position": 0.05},
            "aggressive": {"risk_aversion": 0.5, "max_position": 0.10}
        }
        
        params = risk_params.get(risk_tolerance, risk_params["medium"])
        
        # Optimization constraints
        constraints = [
            {"type": "eq", "fun": lambda x: np.sum(x) - 1.0},  # Weights sum to 1
        ]
        
        bounds = [(0, params["max_position"]) for _ in range(n_slips)]
        
        # Objective function (maximize Sharpe ratio)
        def objective(weights):
            portfolio_return = np.dot(weights, returns)
            portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
            sharpe = portfolio_return / np.sqrt(portfolio_variance)
            return -sharpe  # Minimize negative Sharpe
            
        # Initial guess (equal weights)
        x0 = np.ones(n_slips) / n_slips
        
        # Optimize
        try:
            result = minimize(
                objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000}
            )
            
            if result.success:
                weights = result.x
            else:
                logger.warning("Portfolio optimization failed, using equal weights")
                weights = x0
                
        except Exception as e:
            logger.error(f"Optimization error: {e}")
            weights = x0
            
        # Apply weights to slips
        for i, slip in enumerate(slips):
            slip.recommended_stake = weights[i] * params["max_position"]
            
        return slips
        
    def _count_shared_matches(self, slip1: BettingSlip, slip2: BettingSlip) -> int:
        """Count shared matches between two slips"""
        matches1 = {(m.home_team, m.away_team) for m in slip1.matches}
        matches2 = {(m.home_team, m.away_team) for m in slip2.matches}
        return len(matches1.intersection(matches2))
        
    def _select_final_portfolio(
        self,
        slips: List[BettingSlip],
        max_slips: int,
        risk_tolerance: str
    ) -> List[BettingSlip]:
        """Select final portfolio based on risk tolerance"""
        
        # Filter by risk level
        risk_filters = {
            "conservative": ["Low"],
            "medium": ["Low", "Medium"],
            "aggressive": ["Low", "Medium", "High"]
        }
        
        allowed_risks = risk_filters.get(risk_tolerance, risk_filters["medium"])
        filtered_slips = [s for s in slips if s.risk_level in allowed_risks]
        
        # Sort by expected value and diversification
        filtered_slips.sort(
            key=lambda x: x.expected_value * x.diversification_score,
            reverse=True
        )
        
        # Select top slips ensuring diversification
        final_portfolio = []
        used_matches = set()
        
        for slip in filtered_slips:
            # Check overlap with already selected slips
            slip_matches = {(m.home_team, m.away_team) for m in slip.matches}
            overlap = len(slip_matches.intersection(used_matches))
            
            # Allow some overlap but not complete duplication
            if overlap < len(slip_matches) * 0.5:
                final_portfolio.append(slip)
                used_matches.update(slip_matches)
                
                if len(final_portfolio) >= max_slips:
                    break
                    
        return final_portfolio
        
    def calculate_portfolio_metrics(self, portfolio: List[BettingSlip], bankroll: float) -> Dict[str, Any]:
        """Calculate portfolio-level metrics"""
        if not portfolio:
            return {}
            
        total_stake = sum(s.recommended_stake for s in portfolio) * bankroll
        expected_returns = sum(s.expected_value * s.recommended_stake * bankroll for s in portfolio)
        
        # Portfolio metrics
        metrics = {
            "total_slips": len(portfolio),
            "total_stake": round(total_stake, 2),
            "expected_return": round(expected_returns, 2),
            "expected_roi": round((expected_returns / total_stake - 1) * 100, 2) if total_stake > 0 else 0,
            "average_odds": round(np.mean([s.total_odds for s in portfolio]), 2),
            "risk_distribution": {
                "Low": sum(1 for s in portfolio if s.risk_level == "Low"),
                "Medium": sum(1 for s in portfolio if s.risk_level == "Medium"),
                "High": sum(1 for s in portfolio if s.risk_level == "High")
            },
            "diversification_score": round(np.mean([s.diversification_score for s in portfolio]), 2),
            "bet_types": {
                bet_type: sum(1 for s in portfolio if s.slip_type == bet_type)
                for bet_type in ["single", "double", "triple", "quadruple", "quintuple"]
            }
        }
        
        return metrics
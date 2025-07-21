"""
Backtesting Module
==================

Comprehensive backtesting framework for football betting strategies.
Supports multiple strategies, realistic simulation, and detailed analytics.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple, Optional, Callable
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from enum import Enum
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import json
import warnings
warnings.filterwarnings('ignore')

from config import Config
from database_manager import DatabaseManager
from predictor_model import FootballPredictor
from bet_optimizer import BetOptimizer
from logger_module import setup_logger

logger = setup_logger(__name__)


class BetType(Enum):
    """Types of bets supported in backtesting."""
    MATCH_RESULT = "match_result"
    OVER_UNDER = "over_under"
    BTTS = "btts"
    ASIAN_HANDICAP = "asian_handicap"
    CORRECT_SCORE = "correct_score"
    DOUBLE_CHANCE = "double_chance"
    DRAW_NO_BET = "draw_no_bet"


@dataclass
class BacktestConfig:
    """Configuration for backtesting."""
    start_date: datetime
    end_date: datetime
    initial_capital: float = 1000.0
    commission: float = 0.05  # 5% commission on winnings
    min_odds: float = 1.5
    max_odds: float = 10.0
    max_stake_pct: float = 0.1  # Max 10% of capital per bet
    min_stake: float = 1.0
    max_concurrent_bets: int = 10
    leagues: List[str] = field(default_factory=list)
    bet_types: List[BetType] = field(default_factory=lambda: [BetType.MATCH_RESULT])
    kelly_divisor: float = 4.0  # For Kelly criterion
    confidence_threshold: float = 0.55
    value_threshold: float = 0.05


@dataclass
class BacktestResult:
    """Results from a backtest run."""
    config: BacktestConfig
    trades: List[Dict[str, Any]]
    equity_curve: pd.Series
    metrics: Dict[str, float]
    analysis: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'config': {
                'start_date': self.config.start_date.isoformat(),
                'end_date': self.config.end_date.isoformat(),
                'initial_capital': self.config.initial_capital,
                'leagues': self.config.leagues
            },
            'metrics': self.metrics,
            'analysis': self.analysis,
            'trades_count': len(self.trades),
            'final_capital': self.equity_curve.iloc[-1] if len(self.equity_curve) > 0 else 0
        }


class Backtester:
    """Main backtesting engine for football betting strategies."""
    
    def __init__(self, config: Config):
        """
        Initialize backtester.
        
        Args:
            config: System configuration
        """
        self.config = config
        self.db = DatabaseManager(config)
        self.predictor = FootballPredictor(config)
        self.bet_optimizer = BetOptimizer(config)
        
    def backtest_strategy(self,
                         strategy_func: Callable,
                         backtest_config: BacktestConfig,
                         strategy_params: Dict[str, Any] = None) -> BacktestResult:
        """
        Run backtest for a given strategy.
        
        Args:
            strategy_func: Strategy function that generates bets
            backtest_config: Backtest configuration
            strategy_params: Additional strategy parameters
            
        Returns:
            BacktestResult object
        """
        logger.info(f"Starting backtest from {backtest_config.start_date} to {backtest_config.end_date}")
        
        # Initialize backtest state
        capital = backtest_config.initial_capital
        equity_curve = [capital]
        trades = []
        open_positions = {}
        
        # Get historical data
        matches = self._get_historical_matches(backtest_config)
        logger.info(f"Found {len(matches)} matches for backtesting")
        
        # Simulate day by day
        current_date = backtest_config.start_date
        while current_date <= backtest_config.end_date:
            # Update settled bets
            capital, settled = self._settle_bets(open_positions, current_date, capital, backtest_config)
            trades.extend(settled)
            
            # Get matches for current day
            day_matches = [m for m in matches if m['date'].date() == current_date.date()]
            
            if day_matches:
                # Generate predictions
                predictions = self._generate_predictions(day_matches)
                
                # Apply strategy
                new_bets = strategy_func(
                    predictions=predictions,
                    capital=capital,
                    open_positions=open_positions,
                    config=backtest_config,
                    params=strategy_params or {}
                )
                
                # Place bets
                for bet in new_bets:
                    if self._validate_bet(bet, capital, open_positions, backtest_config):
                        capital -= bet['stake']
                        open_positions[bet['id']] = bet
                        logger.debug(f"Placed bet: {bet['type']} on {bet['match_id']} for {bet['stake']:.2f}")
                        
            equity_curve.append(capital)
            current_date += timedelta(days=1)
            
        # Settle remaining bets
        final_capital, final_settled = self._settle_all_remaining(open_positions, capital, backtest_config)
        trades.extend(final_settled)
        equity_curve.append(final_capital)
        
        # Calculate metrics and analysis
        equity_series = pd.Series(equity_curve)
        metrics = self._calculate_metrics(trades, equity_series, backtest_config)
        analysis = self._analyze_results(trades, equity_series, backtest_config)
        
        return BacktestResult(
            config=backtest_config,
            trades=trades,
            equity_curve=equity_series,
            metrics=metrics,
            analysis=analysis
        )
        
    def _get_historical_matches(self, config: BacktestConfig) -> List[Dict[str, Any]]:
        """Get historical matches with odds for backtesting."""
        return self.db.get_matches_with_complete_data(
            start_date=config.start_date,
            end_date=config.end_date,
            leagues=config.leagues
        )
        
    def _generate_predictions(self, matches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate predictions for matches."""
        predictions = []
        
        for match in matches:
            try:
                # Generate comprehensive prediction
                pred = self.predictor.predict_match(match)
                pred['match_id'] = match['id']
                pred['match_date'] = match['date']
                pred['home_team'] = match['home_team']
                pred['away_team'] = match['away_team']
                pred['odds'] = match['odds']
                predictions.append(pred)
            except Exception as e:
                logger.warning(f"Failed to predict match {match['id']}: {e}")
                
        return predictions
        
    def _validate_bet(self,
                     bet: Dict[str, Any],
                     capital: float,
                     open_positions: Dict[str, Dict[str, Any]],
                     config: BacktestConfig) -> bool:
        """Validate if bet can be placed."""
        # Check stake limits
        if bet['stake'] < config.min_stake:
            return False
            
        if bet['stake'] > capital * config.max_stake_pct:
            return False
            
        if bet['stake'] > capital:
            return False
            
        # Check odds limits
        if bet['odds'] < config.min_odds or bet['odds'] > config.max_odds:
            return False
            
        # Check concurrent bets limit
        if len(open_positions) >= config.max_concurrent_bets:
            return False
            
        # Check confidence threshold
        if bet.get('confidence', 0) < config.confidence_threshold:
            return False
            
        return True
        
    def _settle_bets(self,
                    open_positions: Dict[str, Dict[str, Any]],
                    current_date: datetime,
                    capital: float,
                    config: BacktestConfig) -> Tuple[float, List[Dict[str, Any]]]:
        """Settle bets that have concluded."""
        settled = []
        to_remove = []
        
        for bet_id, bet in open_positions.items():
            if bet['match_date'] < current_date:
                # Get match result
                result = self.db.get_match_result(bet['match_id'])
                
                if result:
                    # Calculate outcome
                    won = self._is_bet_won(bet, result)
                    
                    if won:
                        winnings = bet['stake'] * bet['odds']
                        commission = winnings * config.commission
                        net_winnings = winnings - commission - bet['stake']
                        capital += bet['stake'] + net_winnings
                        bet['profit'] = net_winnings
                        bet['commission'] = commission
                    else:
                        bet['profit'] = -bet['stake']
                        bet['commission'] = 0
                        
                    bet['won'] = won
                    bet['settled_date'] = current_date
                    bet['result'] = result
                    settled.append(bet)
                    to_remove.append(bet_id)
                    
        # Remove settled bets
        for bet_id in to_remove:
            del open_positions[bet_id]
            
        return capital, settled
        
    def _settle_all_remaining(self,
                             open_positions: Dict[str, Dict[str, Any]],
                             capital: float,
                             config: BacktestConfig) -> Tuple[float, List[Dict[str, Any]]]:
        """Settle all remaining open positions."""
        settled = []
        
        for bet in open_positions.values():
            # Assume bet is void/cancelled
            capital += bet['stake']
            bet['profit'] = 0
            bet['won'] = None
            bet['void'] = True
            settled.append(bet)
            
        return capital, settled
        
    def _is_bet_won(self, bet: Dict[str, Any], result: Dict[str, Any]) -> bool:
        """Check if bet won based on result."""
        bet_type = BetType(bet['type'])
        
        if bet_type == BetType.MATCH_RESULT:
            return self._check_match_result(bet, result)
        elif bet_type == BetType.OVER_UNDER:
            return self._check_over_under(bet, result)
        elif bet_type == BetType.BTTS:
            return self._check_btts(bet, result)
        elif bet_type == BetType.ASIAN_HANDICAP:
            return self._check_asian_handicap(bet, result)
        elif bet_type == BetType.CORRECT_SCORE:
            return self._check_correct_score(bet, result)
        elif bet_type == BetType.DOUBLE_CHANCE:
            return self._check_double_chance(bet, result)
        elif bet_type == BetType.DRAW_NO_BET:
            return self._check_draw_no_bet(bet, result)
            
        return False
        
    def _check_match_result(self, bet: Dict[str, Any], result: Dict[str, Any]) -> bool:
        """Check match result bet."""
        if bet['selection'] == 'home' and result['home_score'] > result['away_score']:
            return True
        elif bet['selection'] == 'away' and result['away_score'] > result['home_score']:
            return True
        elif bet['selection'] == 'draw' and result['home_score'] == result['away_score']:
            return True
        return False
        
    def _check_over_under(self, bet: Dict[str, Any], result: Dict[str, Any]) -> bool:
        """Check over/under bet."""
        total_goals = result['home_score'] + result['away_score']
        line = bet['line']
        
        if bet['selection'] == 'over':
            return total_goals > line
        else:
            return total_goals < line
            
    def _check_btts(self, bet: Dict[str, Any], result: Dict[str, Any]) -> bool:
        """Check both teams to score bet."""
        btts = result['home_score'] > 0 and result['away_score'] > 0
        return btts if bet['selection'] == 'yes' else not btts
        
    def _check_asian_handicap(self, bet: Dict[str, Any], result: Dict[str, Any]) -> bool:
        """Check Asian handicap bet."""
        handicap = bet['handicap']
        
        if bet['selection'] == 'home':
            adjusted_home = result['home_score'] + handicap
            return adjusted_home > result['away_score']
        else:
            adjusted_away = result['away_score'] - handicap
            return adjusted_away > result['home_score']
            
    def _check_correct_score(self, bet: Dict[str, Any], result: Dict[str, Any]) -> bool:
        """Check correct score bet."""
        return (bet['home_score'] == result['home_score'] and 
                bet['away_score'] == result['away_score'])
                
    def _check_double_chance(self, bet: Dict[str, Any], result: Dict[str, Any]) -> bool:
        """Check double chance bet."""
        home_win = result['home_score'] > result['away_score']
        away_win = result['away_score'] > result['home_score']
        draw = result['home_score'] == result['away_score']
        
        if bet['selection'] == 'home_draw':
            return home_win or draw
        elif bet['selection'] == 'home_away':
            return home_win or away_win
        elif bet['selection'] == 'away_draw':
            return away_win or draw
            
        return False
        
    def _check_draw_no_bet(self, bet: Dict[str, Any], result: Dict[str, Any]) -> bool:
        """Check draw no bet."""
        if result['home_score'] == result['away_score']:
            # Draw - bet is void
            bet['void'] = True
            return None
            
        if bet['selection'] == 'home':
            return result['home_score'] > result['away_score']
        else:
            return result['away_score'] > result['home_score']
            
    def _calculate_metrics(self,
                          trades: List[Dict[str, Any]],
                          equity_curve: pd.Series,
                          config: BacktestConfig) -> Dict[str, float]:
        """Calculate comprehensive backtest metrics."""
        if not trades:
            return self._empty_metrics()
            
        # Filter out void bets
        valid_trades = [t for t in trades if not t.get('void', False)]
        
        if not valid_trades:
            return self._empty_metrics()
            
        # Basic metrics
        total_trades = len(valid_trades)
        winning_trades = sum(1 for t in valid_trades if t['won'])
        losing_trades = total_trades - winning_trades
        
        # Financial metrics
        total_stake = sum(t['stake'] for t in valid_trades)
        total_profit = sum(t['profit'] for t in valid_trades)
        total_commission = sum(t.get('commission', 0) for t in valid_trades)
        
        # Returns
        final_capital = equity_curve.iloc[-1]
        total_return = (final_capital - config.initial_capital)
        roi = (total_return / config.initial_capital) * 100
        
        # Win rate
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        # Average metrics
        avg_stake = total_stake / total_trades if total_trades > 0 else 0
        avg_odds = np.mean([t['odds'] for t in valid_trades]) if valid_trades else 0
        avg_win = np.mean([t['profit'] for t in valid_trades if t['won']]) if winning_trades > 0 else 0
        avg_loss = np.mean([t['profit'] for t in valid_trades if not t['won']]) if losing_trades > 0 else 0
        
        # Risk metrics
        returns = equity_curve.pct_change().dropna()
        sharpe_ratio = self._calculate_sharpe_ratio(returns)
        sortino_ratio = self._calculate_sortino_ratio(returns)
        max_drawdown, max_drawdown_duration = self._calculate_drawdown(equity_curve)
        
        # Kelly metrics
        kelly_trades = [t for t in valid_trades if 'kelly_fraction' in t]
        avg_kelly = np.mean([t['kelly_fraction'] for t in kelly_trades]) if kelly_trades else 0
        
        # Profit factor
        gross_profit = sum(t['profit'] for t in valid_trades if t['profit'] > 0)
        gross_loss = abs(sum(t['profit'] for t in valid_trades if t['profit'] < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Additional metrics
        expectancy = total_profit / total_trades if total_trades > 0 else 0
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_stake': total_stake,
            'total_profit': total_profit,
            'total_commission': total_commission,
            'roi': roi,
            'final_capital': final_capital,
            'avg_stake': avg_stake,
            'avg_odds': avg_odds,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'expectancy': expectancy,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'max_drawdown_duration': max_drawdown_duration,
            'avg_kelly_fraction': avg_kelly
        }
        
    def _calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) < 2 or returns.std() == 0:
            return 0.0
            
        excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
        return np.sqrt(252) * excess_returns.mean() / returns.std()
        
    def _calculate_sortino_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino ratio."""
        if len(returns) < 2:
            return 0.0
            
        excess_returns = returns - risk_free_rate / 252
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0:
            return float('inf')
            
        downside_std = downside_returns.std()
        if downside_std == 0:
            return float('inf')
            
        return np.sqrt(252) * excess_returns.mean() / downside_std
        
    def _calculate_drawdown(self, equity_curve: pd.Series) -> Tuple[float, int]:
        """Calculate maximum drawdown and duration."""
        rolling_max = equity_curve.expanding().max()
        drawdown = (equity_curve - rolling_max) / rolling_max
        
        max_drawdown = abs(drawdown.min()) * 100
        
        # Calculate max drawdown duration
        underwater = drawdown < 0
        duration = 0
        max_duration = 0
        
        for is_underwater in underwater:
            if is_underwater:
                duration += 1
                max_duration = max(max_duration, duration)
            else:
                duration = 0
                
        return max_drawdown, max_duration
        
    def _analyze_results(self,
                        trades: List[Dict[str, Any]],
                        equity_curve: pd.Series,
                        config: BacktestConfig) -> Dict[str, Any]:
        """Perform detailed analysis of backtest results."""
        if not trades:
            return {}
            
        valid_trades = [t for t in trades if not t.get('void', False)]
        
        analysis = {
            'by_league': self._analyze_by_league(valid_trades),
            'by_bet_type': self._analyze_by_bet_type(valid_trades),
            'by_odds_range': self._analyze_by_odds_range(valid_trades),
            'by_confidence': self._analyze_by_confidence(valid_trades),
            'by_time': self._analyze_by_time(valid_trades),
            'by_stake_size': self._analyze_by_stake_size(valid_trades),
            'monthly_returns': self._calculate_monthly_returns(equity_curve, config),
            'risk_analysis': self._analyze_risk(equity_curve, valid_trades)
        }
        
        return analysis
        
    def _analyze_by_league(self, trades: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        """Analyze performance by league."""
        leagues = {}
        
        for trade in trades:
            league = trade.get('league', 'Unknown')
            if league not in leagues:
                leagues[league] = {
                    'trades': 0,
                    'wins': 0,
                    'profit': 0,
                    'roi': 0
                }
                
            leagues[league]['trades'] += 1
            if trade['won']:
                leagues[league]['wins'] += 1
            leagues[league]['profit'] += trade['profit']
            
        # Calculate ROI for each league
        for league_data in leagues.values():
            if league_data['trades'] > 0:
                avg_stake = 10  # Assume average stake
                total_stake = league_data['trades'] * avg_stake
                league_data['roi'] = (league_data['profit'] / total_stake) * 100
                league_data['win_rate'] = (league_data['wins'] / league_data['trades']) * 100
                
        return leagues
        
    def _analyze_by_bet_type(self, trades: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        """Analyze performance by bet type."""
        bet_types = {}
        
        for trade in trades:
            bet_type = trade['type']
            if bet_type not in bet_types:
                bet_types[bet_type] = {
                    'trades': 0,
                    'wins': 0,
                    'profit': 0,
                    'avg_odds': []
                }
                
            bet_types[bet_type]['trades'] += 1
            if trade['won']:
                bet_types[bet_type]['wins'] += 1
            bet_types[bet_type]['profit'] += trade['profit']
            bet_types[bet_type]['avg_odds'].append(trade['odds'])
            
        # Calculate averages
        for bet_data in bet_types.values():
            if bet_data['trades'] > 0:
                bet_data['win_rate'] = (bet_data['wins'] / bet_data['trades']) * 100
                bet_data['avg_odds'] = np.mean(bet_data['avg_odds'])
                
        return bet_types
        
    def _analyze_by_odds_range(self, trades: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        """Analyze performance by odds ranges."""
        ranges = {
            '1.0-1.5': {'min': 1.0, 'max': 1.5},
            '1.5-2.0': {'min': 1.5, 'max': 2.0},
            '2.0-3.0': {'min': 2.0, 'max': 3.0},
            '3.0-5.0': {'min': 3.0, 'max': 5.0},
            '5.0+': {'min': 5.0, 'max': float('inf')}
        }
        
        results = {}
        for range_name, range_vals in ranges.items():
            range_trades = [
                t for t in trades 
                if range_vals['min'] <= t['odds'] < range_vals['max']
            ]
            
            if range_trades:
                wins = sum(1 for t in range_trades if t['won'])
                profit = sum(t['profit'] for t in range_trades)
                
                results[range_name] = {
                    'trades': len(range_trades),
                    'wins': wins,
                    'win_rate': (wins / len(range_trades)) * 100,
                    'profit': profit,
                    'avg_profit': profit / len(range_trades)
                }
                
        return results
        
    def _analyze_by_confidence(self, trades: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        """Analyze performance by confidence levels."""
        confidence_ranges = {
            'low': (0, 0.6),
            'medium': (0.6, 0.7),
            'high': (0.7, 0.85),
            'very_high': (0.85, 1.0)
        }
        
        results = {}
        for range_name, (min_conf, max_conf) in confidence_ranges.items():
            conf_trades = [
                t for t in trades 
                if min_conf <= t.get('confidence', 0) < max_conf
            ]
            
            if conf_trades:
                wins = sum(1 for t in conf_trades if t['won'])
                profit = sum(t['profit'] for t in conf_trades)
                
                results[range_name] = {
                    'trades': len(conf_trades),
                    'wins': wins,
                    'win_rate': (wins / len(conf_trades)) * 100,
                    'profit': profit,
                    'roi': (profit / sum(t['stake'] for t in conf_trades)) * 100
                }
                
        return results
        
    def _analyze_by_time(self, trades: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        """Analyze performance by time periods."""
        # Day of week analysis
        dow_results = {}
        for trade in trades:
            dow = trade['match_date'].strftime('%A')
            if dow not in dow_results:
                dow_results[dow] = {'trades': 0, 'wins': 0, 'profit': 0}
                
            dow_results[dow]['trades'] += 1
            if trade['won']:
                dow_results[dow]['wins'] += 1
            dow_results[dow]['profit'] += trade['profit']
            
        # Calculate win rates
        for dow_data in dow_results.values():
            if dow_data['trades'] > 0:
                dow_data['win_rate'] = (dow_data['wins'] / dow_data['trades']) * 100
                
        return {'day_of_week': dow_results}
        
    def _analyze_by_stake_size(self, trades: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        """Analyze performance by stake size."""
        # Quartile analysis
        stakes = [t['stake'] for t in trades]
        if not stakes:
            return {}
            
        quartiles = np.percentile(stakes, [25, 50, 75])
        
        ranges = {
            'small': (0, quartiles[0]),
            'medium_small': (quartiles[0], quartiles[1]),
            'medium_large': (quartiles[1], quartiles[2]),
            'large': (quartiles[2], float('inf'))
        }
        
        results = {}
        for range_name, (min_stake, max_stake) in ranges.items():
            stake_trades = [
                t for t in trades 
                if min_stake <= t['stake'] < max_stake
            ]
            
            if stake_trades:
                wins = sum(1 for t in stake_trades if t['won'])
                profit = sum(t['profit'] for t in stake_trades)
                
                results[range_name] = {
                    'trades': len(stake_trades),
                    'wins': wins,
                    'win_rate': (wins / len(stake_trades)) * 100,
                    'profit': profit,
                    'avg_stake': np.mean([t['stake'] for t in stake_trades])
                }
                
        return results
        
    def _calculate_monthly_returns(self,
                                  equity_curve: pd.Series,
                                  config: BacktestConfig) -> List[Dict[str, Any]]:
        """Calculate monthly returns."""
        if len(equity_curve) < 2:
            return []
            
        # Resample to monthly
        equity_df = pd.DataFrame({'equity': equity_curve})
        equity_df.index = pd.date_range(
            start=config.start_date,
            periods=len(equity_curve),
            freq='D'
        )
        
        monthly = equity_df.resample('M').last()
        monthly_returns = monthly.pct_change().dropna()
        
        results = []
        for date, row in monthly_returns.iterrows():
            results.append({
                'month': date.strftime('%Y-%m'),
                'return': row['equity'] * 100,
                'equity': monthly.loc[date, 'equity']
            })
            
        return results
        
    def _analyze_risk(self,
                     equity_curve: pd.Series,
                     trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze risk metrics."""
        returns = equity_curve.pct_change().dropna()
        
        # Value at Risk (VaR)
        var_95 = np.percentile(returns, 5) * 100
        var_99 = np.percentile(returns, 1) * 100
        
        # Conditional Value at Risk (CVaR)
        cvar_95 = returns[returns <= np.percentile(returns, 5)].mean() * 100
        cvar_99 = returns[returns <= np.percentile(returns, 1)].mean() * 100
        
        # Win/Loss streaks
        win_streak = 0
        loss_streak = 0
        max_win_streak = 0
        max_loss_streak = 0
        
        for trade in trades:
            if trade['won']:
                win_streak += 1
                loss_streak = 0
                max_win_streak = max(max_win_streak, win_streak)
            else:
                loss_streak += 1
                win_streak = 0
                max_loss_streak = max(max_loss_streak, loss_streak)
                
        return {
            'var_95': var_95,
            'var_99': var_99,
            'cvar_95': cvar_95,
            'cvar_99': cvar_99,
            'max_win_streak': max_win_streak,
            'max_loss_streak': max_loss_streak,
            'volatility': returns.std() * np.sqrt(252) * 100,
            'skewness': stats.skew(returns),
            'kurtosis': stats.kurtosis(returns)
        }
        
    def _empty_metrics(self) -> Dict[str, float]:
        """Return empty metrics dict."""
        return {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0,
            'total_stake': 0,
            'total_profit': 0,
            'total_commission': 0,
            'roi': 0,
            'final_capital': 0,
            'avg_stake': 0,
            'avg_odds': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'profit_factor': 0,
            'expectancy': 0,
            'sharpe_ratio': 0,
            'sortino_ratio': 0,
            'max_drawdown': 0,
            'max_drawdown_duration': 0,
            'avg_kelly_fraction': 0
        }
        
    def generate_backtest_report(self,
                                result: BacktestResult,
                                output_path: str = None) -> str:
        """Generate comprehensive backtest report."""
        if output_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = f"reports/backtest_{timestamp}.html"
            
        # Create visualizations
        self._create_backtest_plots(result)
        
        # Generate HTML report
        html = self._create_html_backtest_report(result)
        
        # Save report
        with open(output_path, 'w') as f:
            f.write(html)
            
        logger.info(f"Backtest report saved to {output_path}")
        return output_path
        
    def _create_backtest_plots(self, result: BacktestResult) -> None:
        """Create backtest visualization plots."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Equity curve plot
        plt.figure(figsize=(12, 6))
        result.equity_curve.plot(linewidth=2)
        plt.title('Equity Curve')
        plt.xlabel('Time')
        plt.ylabel('Capital')
        plt.grid(True, alpha=0.3)
        plt.savefig(f'reports/equity_curve_{timestamp}.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # Drawdown plot
        rolling_max = result.equity_curve.expanding().max()
        drawdown = (result.equity_curve - rolling_max) / rolling_max * 100
        
        plt.figure(figsize=(12, 4))
        drawdown.plot(kind='area', color='red', alpha=0.3)
        plt.title('Drawdown')
        plt.xlabel('Time')
        plt.ylabel('Drawdown %')
        plt.grid(True, alpha=0.3)
        plt.savefig(f'reports/drawdown_{timestamp}.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # Returns distribution
        returns = result.equity_curve.pct_change().dropna() * 100
        
        plt.figure(figsize=(10, 6))
        returns.hist(bins=50, alpha=0.7)
        plt.axvline(returns.mean(), color='red', linestyle='--', label=f'Mean: {returns.mean():.2f}%')
        plt.title('Returns Distribution')
        plt.xlabel('Daily Return %')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f'reports/returns_dist_{timestamp}.png', dpi=150, bbox_inches='tight')
        plt.close()
        
    def _create_html_backtest_report(self, result: BacktestResult) -> str:
        """Create HTML backtest report."""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Backtest Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
                .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 20px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
                h1, h2, h3 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #4CAF50; color: white; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .metric {{ font-size: 24px; font-weight: bold; }}
                .positive {{ color: #4CAF50; }}
                .negative {{ color: #f44336; }}
                .section {{ margin: 30px 0; padding: 20px; background-color: #f9f9f9; border-radius: 5px; }}
                .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; }}
                .metric-card {{ background: white; padding: 15px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Backtest Report</h1>
                <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p>Period: {result.config.start_date.strftime('%Y-%m-%d')} to {result.config.end_date.strftime('%Y-%m-%d')}</p>
                
                <div class="section">
                    <h2>Key Performance Metrics</h2>
                    <div class="metric-grid">
                        <div class="metric-card">
                            <h3>Total Return</h3>
                            <p class="metric {self._get_color_class(result.metrics['roi'])}">{result.metrics['roi']:.2f}%</p>
                        </div>
                        <div class="metric-card">
                            <h3>Win Rate</h3>
                            <p class="metric">{result.metrics['win_rate']:.2f}%</p>
                        </div>
                        <div class="metric-card">
                            <h3>Sharpe Ratio</h3>
                            <p class="metric">{result.metrics['sharpe_ratio']:.2f}</p>
                        </div>
                        <div class="metric-card">
                            <h3>Max Drawdown</h3>
                            <p class="metric negative">{result.metrics['max_drawdown']:.2f}%</p>
                        </div>
                    </div>
                </div>
                
                <div class="section">
                    <h2>Trading Statistics</h2>
                    <table>
                        <tr><th>Metric</th><th>Value</th></tr>
                        <tr><td>Total Trades</td><td>{result.metrics['total_trades']}</td></tr>
                        <tr><td>Winning Trades</td><td>{result.metrics['winning_trades']}</td></tr>
                        <tr><td>Losing Trades</td><td>{result.metrics['losing_trades']}</td></tr>
                        <tr><td>Average Stake</td><td>€{result.metrics['avg_stake']:.2f}</td></tr>
                        <tr><td>Average Odds</td><td>{result.metrics['avg_odds']:.2f}</td></tr>
                        <tr><td>Profit Factor</td><td>{result.metrics['profit_factor']:.2f}</td></tr>
                        <tr><td>Expectancy</td><td>€{result.metrics['expectancy']:.2f}</td></tr>
                        <tr><td>Total Commission</td><td>€{result.metrics['total_commission']:.2f}</td></tr>
                    </table>
                </div>
                
                <div class="section">
                    <h2>Performance by League</h2>
                    <table>
                        <tr>
                            <th>League</th>
                            <th>Trades</th>
                            <th>Win Rate</th>
                            <th>ROI</th>
                        </tr>
                        {self._format_league_analysis(result.analysis.get('by_league', {}))}
                    </table>
                </div>
                
                <div class="section">
                    <h2>Performance by Odds Range</h2>
                    <table>
                        <tr>
                            <th>Odds Range</th>
                            <th>Trades</th>
                            <th>Win Rate</th>
                            <th>Avg Profit</th>
                        </tr>
                        {self._format_odds_analysis(result.analysis.get('by_odds_range', {}))}
                    </table>
                </div>
            </div>
        </body>
        </html>
        """
        return html
        
    def _get_color_class(self, value: float) -> str:
        """Get CSS class based on value."""
        return 'positive' if value >= 0 else 'negative'
        
    def _format_league_analysis(self, league_data: Dict[str, Dict[str, float]]) -> str:
        """Format league analysis for HTML."""
        rows = []
        for league, data in league_data.items():
            row = f"""
            <tr>
                <td>{league}</td>
                <td>{data['trades']}</td>
                <td>{data.get('win_rate', 0):.1f}%</td>
                <td class="{self._get_color_class(data['roi'])}">{data['roi']:.1f}%</td>
            </tr>
            """
            rows.append(row)
        return ''.join(rows)
        
    def _format_odds_analysis(self, odds_data: Dict[str, Dict[str, float]]) -> str:
        """Format odds analysis for HTML."""
        rows = []
        for range_name, data in odds_data.items():
            row = f"""
            <tr>
                <td>{range_name}</td>
                <td>{data['trades']}</td>
                <td>{data['win_rate']:.1f}%</td>
                <td class="{self._get_color_class(data['avg_profit'])}">€{data['avg_profit']:.2f}</td>
            </tr>
            """
            rows.append(row)
        return ''.join(rows)


# Example strategy functions
def flat_betting_strategy(predictions: List[Dict[str, Any]],
                         capital: float,
                         open_positions: Dict[str, Dict[str, Any]],
                         config: BacktestConfig,
                         params: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Simple flat betting strategy."""
    bets = []
    stake = params.get('stake', 10.0)
    
    for pred in predictions:
        # Bet on favorites with high confidence
        if pred['confidence'] >= config.confidence_threshold:
            if pred['predicted_result'] == 'home':
                odds = pred['odds']['home_win']
                selection = 'home'
            elif pred['predicted_result'] == 'away':
                odds = pred['odds']['away_win']
                selection = 'away'
            else:
                continue
                
            if config.min_odds <= odds <= config.max_odds:
                bets.append({
                    'id': f"{pred['match_id']}_flat",
                    'match_id': pred['match_id'],
                    'match_date': pred['match_date'],
                    'type': 'match_result',
                    'selection': selection,
                    'stake': stake,
                    'odds': odds,
                    'confidence': pred['confidence'],
                    'home_team': pred['home_team'],
                    'away_team': pred['away_team']
                })
                
    return bets


def kelly_betting_strategy(predictions: List[Dict[str, Any]],
                          capital: float,
                          open_positions: Dict[str, Dict[str, Any]],
                          config: BacktestConfig,
                          params: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Kelly criterion betting strategy."""
    bets = []
    
    for pred in predictions:
        # Calculate Kelly fractions for each outcome
        outcomes = [
            ('home', pred['home_win_prob'], pred['odds']['home_win']),
            ('draw', pred['draw_prob'], pred['odds']['draw']),
            ('away', pred['away_win_prob'], pred['odds']['away_win'])
        ]
        
        best_kelly = 0
        best_outcome = None
        best_odds = None
        
        for outcome, prob, odds in outcomes:
            if prob > 0 and odds > 1:
                # Kelly formula
                q = 1 - prob
                b = odds - 1
                kelly = (prob * b - q) / b
                
                # Apply divisor for safety
                kelly = kelly / config.kelly_divisor
                
                if kelly > best_kelly and kelly > 0.01:  # Minimum 1% edge
                    best_kelly = kelly
                    best_outcome = outcome
                    best_odds = odds
                    
        if best_outcome:
            stake = min(capital * best_kelly, capital * config.max_stake_pct)
            
            if stake >= config.min_stake:
                bets.append({
                    'id': f"{pred['match_id']}_kelly",
                    'match_id': pred['match_id'],
                    'match_date': pred['match_date'],
                    'type': 'match_result',
                    'selection': best_outcome,
                    'stake': stake,
                    'odds': best_odds,
                    'kelly_fraction': best_kelly,
                    'confidence': pred[f'{best_outcome}_{"win" if best_outcome != "draw" else ""}_prob'],
                    'home_team': pred['home_team'],
                    'away_team': pred['away_team']
                })
                
    return bets


def value_betting_strategy(predictions: List[Dict[str, Any]],
                          capital: float,
                          open_positions: Dict[str, Dict[str, Any]],
                          config: BacktestConfig,
                          params: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Value betting strategy."""
    bets = []
    stake_pct = params.get('stake_pct', 0.02)  # 2% of capital
    
    for pred in predictions:
        # Check all markets for value
        markets = [
            ('match_result', 'home', pred['home_win_prob'], pred['odds']['home_win']),
            ('match_result', 'draw', pred['draw_prob'], pred['odds']['draw']),
            ('match_result', 'away', pred['away_win_prob'], pred['odds']['away_win']),
            ('over_under', 'over_2.5', pred.get('over_2.5_prob', 0), pred['odds'].get('over_2.5', 0)),
            ('btts', 'yes', pred.get('btts_yes_prob', 0), pred['odds'].get('btts_yes', 0))
        ]
        
        for market, selection, prob, odds in markets:
            if prob > 0 and odds > 1:
                # Calculate expected value
                ev = (prob * odds) - 1
                
                if ev >= config.value_threshold:
                    stake = capital * stake_pct
                    
                    if stake >= config.min_stake and config.min_odds <= odds <= config.max_odds:
                        bets.append({
                            'id': f"{pred['match_id']}_{market}_{selection}_value",
                            'match_id': pred['match_id'],
                            'match_date': pred['match_date'],
                            'type': market,
                            'selection': selection,
                            'stake': stake,
                            'odds': odds,
                            'expected_value': ev,
                            'confidence': prob,
                            'home_team': pred['home_team'],
                            'away_team': pred['away_team']
                        })
                        
    return bets


if __name__ == "__main__":
    # Example backtest
    config = Config()
    backtester = Backtester(config)
    
    # Configure backtest
    backtest_config = BacktestConfig(
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 12, 31),
        initial_capital=1000.0,
        leagues=['premier-league', 'la-liga', 'serie-a'],
        bet_types=[BetType.MATCH_RESULT, BetType.OVER_UNDER]
    )
    
    # Run backtest with Kelly strategy
    result = backtester.backtest_strategy(
        kelly_betting_strategy,
        backtest_config
    )
    
    # Generate report
    report_path = backtester.generate_backtest_report(result)
    print(f"Backtest completed. Report saved to: {report_path}")
    print(f"ROI: {result.metrics['roi']:.2f}%")
    print(f"Sharpe Ratio: {result.metrics['sharpe_ratio']:.2f}")
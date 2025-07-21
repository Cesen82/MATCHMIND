"""
Model Evaluator Module
=====================

Evaluates the performance of prediction models using various metrics.
Performs backtesting, cross-validation, and generates detailed reports.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    mean_absolute_error, mean_squared_error
)
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
import json
import joblib

from config import Config
from database_manager import DatabaseManager
from predictor_model import FootballPredictor
from logger_module import setup_logger

logger = setup_logger(__name__)


@dataclass
class EvaluationMetrics:
    """Container for model evaluation metrics."""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_roc: float
    confusion_matrix: np.ndarray
    classification_report: Dict[str, Any]
    
    # Betting-specific metrics
    roi: float
    win_rate: float
    average_odds: float
    kelly_performance: float
    max_drawdown: float
    sharpe_ratio: float
    
    # Additional metrics
    predictions_count: int
    correct_predictions: int
    profit_loss: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'auc_roc': self.auc_roc,
            'confusion_matrix': self.confusion_matrix.tolist(),
            'roi': self.roi,
            'win_rate': self.win_rate,
            'average_odds': self.average_odds,
            'kelly_performance': self.kelly_performance,
            'max_drawdown': self.max_drawdown,
            'sharpe_ratio': self.sharpe_ratio,
            'predictions_count': self.predictions_count,
            'correct_predictions': self.correct_predictions,
            'profit_loss': self.profit_loss
        }


class ModelEvaluator:
    """Evaluates prediction model performance."""
    
    def __init__(self, config: Config):
        """
        Initialize model evaluator.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.db = DatabaseManager(config)
        self.predictor = FootballPredictor(config)
        
    async def evaluate_all_models(self) -> Dict[str, Any]:
        """
        Evaluate all prediction models.
        
        Returns:
            Evaluation results for all models
        """
        results = {}
        
        # Evaluate match result predictor
        logger.info("Evaluating match result predictor...")
        results['match_result'] = await self.evaluate_match_predictor()
        
        # Evaluate goals predictor
        logger.info("Evaluating goals predictor...")
        results['goals'] = await self.evaluate_goals_predictor()
        
        # Evaluate betting strategy performance
        logger.info("Evaluating betting strategies...")
        results['betting_strategies'] = await self.evaluate_betting_strategies()
        
        # Evaluate feature importance
        logger.info("Analyzing feature importance...")
        results['feature_importance'] = self.analyze_feature_importance()
        
        # Generate evaluation report
        report_path = self.generate_evaluation_report(results)
        results['detailed_report_path'] = report_path
        
        return results
        
    async def evaluate_match_predictor(self, 
                                     start_date: Optional[datetime] = None,
                                     end_date: Optional[datetime] = None) -> EvaluationMetrics:
        """Evaluate match result prediction model."""
        if not start_date:
            start_date = datetime.now() - timedelta(days=90)
        if not end_date:
            end_date = datetime.now()
            
        # Get historical predictions and actual results
        predictions_df = self.db.get_predictions_with_results(start_date, end_date)
        
        if predictions_df.empty:
            logger.warning("No predictions found for evaluation period")
            return self._empty_metrics()
            
        # Calculate basic classification metrics
        y_true = predictions_df['actual_result']
        y_pred = predictions_df['predicted_result']
        y_proba = predictions_df[['home_win_prob', 'draw_prob', 'away_win_prob']].values
        
        # Multi-class metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
        
        # ROC AUC for multi-class
        try:
            # One-vs-rest approach
            from sklearn.preprocessing import label_binarize
            y_true_bin = label_binarize(y_true, classes=['home', 'draw', 'away'])
            auc_roc = roc_auc_score(y_true_bin, y_proba, average='weighted')
        except:
            auc_roc = 0.0
            
        conf_matrix = confusion_matrix(y_true, y_pred, labels=['home', 'draw', 'away'])
        class_report = classification_report(y_true, y_pred, output_dict=True)
        
        # Calculate betting performance metrics
        betting_metrics = self._calculate_betting_metrics(predictions_df)
        
        return EvaluationMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            auc_roc=auc_roc,
            confusion_matrix=conf_matrix,
            classification_report=class_report,
            roi=betting_metrics['roi'],
            win_rate=betting_metrics['win_rate'],
            average_odds=betting_metrics['average_odds'],
            kelly_performance=betting_metrics['kelly_performance'],
            max_drawdown=betting_metrics['max_drawdown'],
            sharpe_ratio=betting_metrics['sharpe_ratio'],
            predictions_count=len(predictions_df),
            correct_predictions=sum(y_true == y_pred),
            profit_loss=betting_metrics['total_profit']
        )
        
    async def evaluate_goals_predictor(self) -> Dict[str, float]:
        """Evaluate goals prediction model."""
        # Get predictions and actual goals
        predictions_df = self.db.get_goals_predictions_with_results()
        
        if predictions_df.empty:
            return {
                'mae_total': 0.0,
                'rmse_total': 0.0,
                'mae_home': 0.0,
                'mae_away': 0.0,
                'exact_score_accuracy': 0.0,
                'over_under_accuracy': 0.0
            }
            
        # Total goals metrics
        y_true_total = predictions_df['actual_total_goals']
        y_pred_total = predictions_df['predicted_total_goals']
        
        mae_total = mean_absolute_error(y_true_total, y_pred_total)
        rmse_total = np.sqrt(mean_squared_error(y_true_total, y_pred_total))
        
        # Individual team goals
        mae_home = mean_absolute_error(
            predictions_df['actual_home_goals'],
            predictions_df['predicted_home_goals']
        )
        mae_away = mean_absolute_error(
            predictions_df['actual_away_goals'],
            predictions_df['predicted_away_goals']
        )
        
        # Exact score accuracy
        exact_scores = (
            (predictions_df['predicted_home_goals'] == predictions_df['actual_home_goals']) &
            (predictions_df['predicted_away_goals'] == predictions_df['actual_away_goals'])
        )
        exact_score_accuracy = exact_scores.mean()
        
        # Over/Under 2.5 accuracy
        over_under_pred = predictions_df['predicted_total_goals'] > 2.5
        over_under_actual = predictions_df['actual_total_goals'] > 2.5
        over_under_accuracy = (over_under_pred == over_under_actual).mean()
        
        return {
            'mae_total': mae_total,
            'rmse_total': rmse_total,
            'mae_home': mae_home,
            'mae_away': mae_away,
            'exact_score_accuracy': exact_score_accuracy,
            'over_under_accuracy': over_under_accuracy
        }
        
    async def evaluate_betting_strategies(self) -> Dict[str, Dict[str, float]]:
        """Evaluate different betting strategies."""
        strategies = {
            'flat_betting': self._evaluate_flat_betting,
            'kelly_criterion': self._evaluate_kelly_betting,
            'value_betting': self._evaluate_value_betting,
            'arbitrage': self._evaluate_arbitrage_betting
        }
        
        results = {}
        for strategy_name, strategy_func in strategies.items():
            logger.info(f"Evaluating {strategy_name} strategy...")
            results[strategy_name] = strategy_func()
            
        return results
        
    def _calculate_betting_metrics(self, predictions_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate betting performance metrics."""
        # Initialize betting bankroll
        initial_bankroll = 1000.0
        bankroll = initial_bankroll
        bankroll_history = [bankroll]
        
        # Simulate betting
        for _, row in predictions_df.iterrows():
            if pd.notna(row.get('recommended_stake', None)):
                stake = row['recommended_stake']
                odds = row['bet_odds']
                
                if row['bet_won']:
                    bankroll += stake * (odds - 1)
                else:
                    bankroll -= stake
                    
                bankroll_history.append(bankroll)
                
        # Calculate metrics
        total_profit = bankroll - initial_bankroll
        roi = (total_profit / initial_bankroll) * 100
        
        # Win rate
        bets_placed = predictions_df['recommended_stake'].notna()
        win_rate = predictions_df[bets_placed]['bet_won'].mean() * 100
        
        # Average odds
        average_odds = predictions_df[bets_placed]['bet_odds'].mean()
        
        # Kelly performance
        kelly_growth = self._calculate_kelly_growth(predictions_df)
        
        # Maximum drawdown
        bankroll_series = pd.Series(bankroll_history)
        rolling_max = bankroll_series.expanding().max()
        drawdown = (bankroll_series - rolling_max) / rolling_max
        max_drawdown = abs(drawdown.min()) * 100
        
        # Sharpe ratio (simplified)
        returns = bankroll_series.pct_change().dropna()
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(365) if returns.std() > 0 else 0
        
        return {
            'roi': roi,
            'win_rate': win_rate,
            'average_odds': average_odds,
            'kelly_performance': kelly_growth,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'total_profit': total_profit
        }
        
    def _calculate_kelly_growth(self, predictions_df: pd.DataFrame) -> float:
        """Calculate Kelly criterion growth rate."""
        kelly_bets = predictions_df[predictions_df['bet_type'] == 'kelly']
        if kelly_bets.empty:
            return 0.0
            
        growth_rate = 1.0
        for _, bet in kelly_bets.iterrows():
            if bet['bet_won']:
                growth_rate *= (1 + bet['kelly_fraction'] * (bet['bet_odds'] - 1))
            else:
                growth_rate *= (1 - bet['kelly_fraction'])
                
        return (growth_rate - 1) * 100
        
    def _evaluate_flat_betting(self) -> Dict[str, float]:
        """Evaluate flat betting strategy."""
        # Get betting history with flat stakes
        bets = self.db.get_betting_history(strategy='flat')
        
        if not bets:
            return {'roi': 0, 'win_rate': 0, 'total_bets': 0}
            
        total_stake = len(bets) * 10  # Assuming €10 flat bets
        total_return = sum(bet['return'] for bet in bets)
        wins = sum(1 for bet in bets if bet['won'])
        
        return {
            'roi': ((total_return - total_stake) / total_stake) * 100,
            'win_rate': (wins / len(bets)) * 100,
            'total_bets': len(bets),
            'profit': total_return - total_stake
        }
        
    def _evaluate_kelly_betting(self) -> Dict[str, float]:
        """Evaluate Kelly criterion betting."""
        bets = self.db.get_betting_history(strategy='kelly')
        
        if not bets:
            return {'roi': 0, 'growth_rate': 0, 'total_bets': 0}
            
        bankroll = 1000.0
        initial_bankroll = bankroll
        
        for bet in bets:
            stake = bankroll * bet['kelly_fraction']
            if bet['won']:
                bankroll += stake * (bet['odds'] - 1)
            else:
                bankroll -= stake
                
        return {
            'roi': ((bankroll - initial_bankroll) / initial_bankroll) * 100,
            'growth_rate': (bankroll / initial_bankroll - 1) * 100,
            'total_bets': len(bets),
            'final_bankroll': bankroll
        }
        
    def _evaluate_value_betting(self) -> Dict[str, float]:
        """Evaluate value betting strategy."""
        bets = self.db.get_betting_history(strategy='value')
        
        if not bets:
            return {'roi': 0, 'average_value': 0, 'total_bets': 0}
            
        total_stake = sum(bet['stake'] for bet in bets)
        total_return = sum(bet['return'] for bet in bets)
        average_value = np.mean([bet['expected_value'] for bet in bets])
        
        return {
            'roi': ((total_return - total_stake) / total_stake) * 100 if total_stake > 0 else 0,
            'average_value': average_value,
            'total_bets': len(bets),
            'profit': total_return - total_stake
        }
        
    def _evaluate_arbitrage_betting(self) -> Dict[str, float]:
        """Evaluate arbitrage betting opportunities."""
        arb_bets = self.db.get_arbitrage_bets()
        
        if not arb_bets:
            return {'opportunities': 0, 'average_profit': 0, 'total_profit': 0}
            
        total_profit = sum(bet['guaranteed_profit'] for bet in arb_bets)
        
        return {
            'opportunities': len(arb_bets),
            'average_profit': total_profit / len(arb_bets),
            'total_profit': total_profit,
            'success_rate': 100.0  # Arbitrage should always profit
        }
        
    def analyze_feature_importance(self) -> Dict[str, List[Tuple[str, float]]]:
        """Analyze feature importance for all models."""
        importance_results = {}
        
        # Load models
        models = {
            'match_result': self.predictor.match_result_model,
            'goals': self.predictor.goals_model,
            'btts': self.predictor.btts_model
        }
        
        for model_name, model in models.items():
            if hasattr(model, 'feature_importances_'):
                # Get feature names
                feature_names = self.predictor.get_feature_names(model_name)
                
                # Get importances
                importances = model.feature_importances_
                
                # Sort by importance
                feature_importance = sorted(
                    zip(feature_names, importances),
                    key=lambda x: x[1],
                    reverse=True
                )
                
                importance_results[model_name] = feature_importance[:20]  # Top 20
                
        return importance_results
        
    def perform_cross_validation(self, model_name: str, n_splits: int = 5) -> Dict[str, Any]:
        """Perform time series cross-validation."""
        # Get training data
        X, y = self.db.get_model_training_data(model_name)
        
        # Time series split
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        # Get model
        model = getattr(self.predictor, f"{model_name}_model")
        
        # Perform cross-validation
        scores = cross_val_score(model, X, y, cv=tscv, scoring='accuracy')
        
        return {
            'mean_score': scores.mean(),
            'std_score': scores.std(),
            'scores': scores.tolist(),
            'n_splits': n_splits
        }
        
    def backtest_strategy(self,
                         strategy_name: str,
                         start_date: datetime,
                         end_date: datetime,
                         initial_capital: float = 1000.0) -> Dict[str, Any]:
        """Backtest a betting strategy on historical data."""
        logger.info(f"Backtesting {strategy_name} from {start_date} to {end_date}")
        
        # Get historical matches and odds
        matches = self.db.get_matches_with_odds(start_date, end_date)
        
        # Initialize backtest
        capital = initial_capital
        trades = []
        equity_curve = [capital]
        
        for match in matches:
            # Generate prediction
            prediction = self.predictor.predict_match(match)
            
            # Apply strategy
            bet = self._apply_strategy(strategy_name, prediction, capital, match['odds'])
            
            if bet:
                # Calculate outcome
                if self._is_bet_won(bet, match['result']):
                    profit = bet['stake'] * (bet['odds'] - 1)
                    capital += profit
                    bet['profit'] = profit
                else:
                    capital -= bet['stake']
                    bet['profit'] = -bet['stake']
                    
                trades.append(bet)
                equity_curve.append(capital)
                
        # Calculate backtest metrics
        return self._calculate_backtest_metrics(
            trades, equity_curve, initial_capital
        )
        
    def _apply_strategy(self,
                       strategy_name: str,
                       prediction: Dict[str, Any],
                       capital: float,
                       odds: Dict[str, float]) -> Optional[Dict[str, Any]]:
        """Apply betting strategy to generate bet."""
        if strategy_name == 'flat':
            return self._flat_bet(prediction, capital, odds)
        elif strategy_name == 'kelly':
            return self._kelly_bet(prediction, capital, odds)
        elif strategy_name == 'value':
            return self._value_bet(prediction, capital, odds)
        else:
            return None
            
    def _flat_bet(self, prediction: Dict[str, Any], capital: float, odds: Dict[str, float]) -> Optional[Dict[str, Any]]:
        """Generate flat bet."""
        stake = min(10.0, capital * 0.02)  # 2% of capital or €10
        
        # Bet on highest probability outcome
        if prediction['home_win_prob'] > 0.5:
            return {
                'type': 'home_win',
                'stake': stake,
                'odds': odds['home_win'],
                'probability': prediction['home_win_prob']
            }
        elif prediction['away_win_prob'] > 0.5:
            return {
                'type': 'away_win',
                'stake': stake,
                'odds': odds['away_win'],
                'probability': prediction['away_win_prob']
            }
            
        return None
        
    def _kelly_bet(self, prediction: Dict[str, Any], capital: float, odds: Dict[str, float]) -> Optional[Dict[str, Any]]:
        """Generate Kelly criterion bet."""
        # Calculate Kelly fractions
        kelly_fractions = {}
        
        for outcome in ['home_win', 'draw', 'away_win']:
            prob = prediction[f'{outcome}_prob']
            odd = odds[outcome]
            
            # Kelly formula: f = (p * b - q) / b
            # where p = probability of winning, q = 1-p, b = odds-1
            q = 1 - prob
            b = odd - 1
            f = (prob * b - q) / b if b > 0 else 0
            
            # Apply Kelly divisor for safety
            f = f / 4  # Quarter Kelly
            
            if f > 0.001:  # Minimum threshold
                kelly_fractions[outcome] = f
                
        if kelly_fractions:
            # Bet on outcome with highest Kelly fraction
            best_outcome = max(kelly_fractions, key=kelly_fractions.get)
            fraction = kelly_fractions[best_outcome]
            
            return {
                'type': best_outcome,
                'stake': capital * fraction,
                'odds': odds[best_outcome],
                'kelly_fraction': fraction,
                'probability': prediction[f'{best_outcome}_prob']
            }
            
        return None
        
    def _value_bet(self, prediction: Dict[str, Any], capital: float, odds: Dict[str, float]) -> Optional[Dict[str, Any]]:
        """Generate value bet."""
        min_value = 0.05  # 5% minimum edge
        
        for outcome in ['home_win', 'draw', 'away_win']:
            prob = prediction[f'{outcome}_prob']
            odd = odds[outcome]
            
            # Expected value: EV = (prob * odd) - 1
            ev = (prob * odd) - 1
            
            if ev > min_value:
                return {
                    'type': outcome,
                    'stake': capital * 0.02,  # 2% of capital
                    'odds': odd,
                    'expected_value': ev,
                    'probability': prob
                }
                
        return None
        
    def _is_bet_won(self, bet: Dict[str, Any], result: str) -> bool:
        """Check if bet won."""
        if bet['type'] == 'home_win' and result == 'home':
            return True
        elif bet['type'] == 'away_win' and result == 'away':
            return True
        elif bet['type'] == 'draw' and result == 'draw':
            return True
        return False
        
    def _calculate_backtest_metrics(self,
                                   trades: List[Dict[str, Any]],
                                   equity_curve: List[float],
                                   initial_capital: float) -> Dict[str, Any]:
        """Calculate comprehensive backtest metrics."""
        if not trades:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'total_return': 0,
                'roi': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0
            }
            
        # Basic metrics
        winning_trades = sum(1 for t in trades if t['profit'] > 0)
        losing_trades = sum(1 for t in trades if t['profit'] < 0)
        
        total_profit = sum(t['profit'] for t in trades)
        total_stake = sum(t['stake'] for t in trades)
        
        # Calculate returns
        equity_series = pd.Series(equity_curve)
        returns = equity_series.pct_change().dropna()
        
        # Sharpe ratio (annualized)
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        
        # Maximum drawdown
        rolling_max = equity_series.expanding().max()
        drawdown = (equity_series - rolling_max) / rolling_max
        max_drawdown = abs(drawdown.min())
        
        return {
            'total_trades': len(trades),
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': winning_trades / len(trades) * 100,
            'total_return': total_profit,
            'roi': (equity_curve[-1] - initial_capital) / initial_capital * 100,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown * 100,
            'final_capital': equity_curve[-1],
            'average_trade': total_profit / len(trades),
            'profit_factor': abs(sum(t['profit'] for t in trades if t['profit'] > 0) / 
                               sum(t['profit'] for t in trades if t['profit'] < 0))
                               if losing_trades > 0 else float('inf')
        }
        
    def generate_evaluation_report(self, results: Dict[str, Any]) -> str:
        """Generate detailed evaluation report."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = f"reports/model_evaluation_{timestamp}.html"
        
        # Create HTML report
        html_content = self._create_html_report(results)
        
        # Save report
        with open(report_path, 'w') as f:
            f.write(html_content)
            
        # Generate visualizations
        self._create_evaluation_plots(results, timestamp)
        
        logger.info(f"Evaluation report saved to {report_path}")
        return report_path
        
    def _create_html_report(self, results: Dict[str, Any]) -> str:
        """Create HTML evaluation report."""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Model Evaluation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2, h3 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .metric {{ font-size: 24px; font-weight: bold; color: #2196F3; }}
                .section {{ margin: 30px 0; }}
            </style>
        </head>
        <body>
            <h1>Model Evaluation Report</h1>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <div class="section">
                <h2>Match Result Predictor</h2>
                <table>
                    <tr><th>Metric</th><th>Value</th></tr>
                    <tr><td>Accuracy</td><td class="metric">{results['match_result'].accuracy:.2%}</td></tr>
                    <tr><td>Precision</td><td>{results['match_result'].precision:.2%}</td></tr>
                    <tr><td>Recall</td><td>{results['match_result'].recall:.2%}</td></tr>
                    <tr><td>F1 Score</td><td>{results['match_result'].f1_score:.2%}</td></tr>
                    <tr><td>ROI</td><td class="metric">{results['match_result'].roi:.2f}%</td></tr>
                    <tr><td>Win Rate</td><td>{results['match_result'].win_rate:.2f}%</td></tr>
                    <tr><td>Sharpe Ratio</td><td>{results['match_result'].sharpe_ratio:.2f}</td></tr>
                    <tr><td>Max Drawdown</td><td>{results['match_result'].max_drawdown:.2f}%</td></tr>
                </table>
            </div>
            
            <div class="section">
                <h2>Goals Predictor</h2>
                <table>
                    <tr><th>Metric</th><th>Value</th></tr>
                    <tr><td>MAE Total Goals</td><td>{results['goals']['mae_total']:.2f}</td></tr>
                    <tr><td>RMSE Total Goals</td><td>{results['goals']['rmse_total']:.2f}</td></tr>
                    <tr><td>Exact Score Accuracy</td><td>{results['goals']['exact_score_accuracy']:.2%}</td></tr>
                    <tr><td>Over/Under 2.5 Accuracy</td><td>{results['goals']['over_under_accuracy']:.2%}</td></tr>
                </table>
            </div>
            
            <div class="section">
                <h2>Betting Strategy Performance</h2>
                <table>
                    <tr>
                        <th>Strategy</th>
                        <th>ROI</th>
                        <th>Win Rate</th>
                        <th>Total Bets</th>
                    </tr>
                    {self._format_strategy_results(results['betting_strategies'])}
                </table>
            </div>
        </body>
        </html>
        """
        return html
        
    def _format_strategy_results(self, strategies: Dict[str, Dict[str, float]]) -> str:
        """Format strategy results for HTML table."""
        rows = []
        for strategy, metrics in strategies.items():
            row = f"""
            <tr>
                <td>{strategy.replace('_', ' ').title()}</td>
                <td>{metrics.get('roi', 0):.2f}%</td>
                <td>{metrics.get('win_rate', 0):.2f}%</td>
                <td>{metrics.get('total_bets', 0)}</td>
            </tr>
            """
            rows.append(row)
        return ''.join(rows)
        
    def _create_evaluation_plots(self, results: Dict[str, Any], timestamp: str) -> None:
        """Create evaluation visualizations."""
        # Confusion matrix plot
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            results['match_result'].confusion_matrix,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['Home', 'Draw', 'Away'],
            yticklabels=['Home', 'Draw', 'Away']
        )
        plt.title('Confusion Matrix - Match Results')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.savefig(f'reports/confusion_matrix_{timestamp}.png')
        plt.close()
        
        # Feature importance plot
        if 'feature_importance' in results and results['feature_importance']:
            for model_name, importances in results['feature_importance'].items():
                features, values = zip(*importances[:15])  # Top 15
                
                plt.figure(figsize=(10, 6))
                plt.barh(features, values)
                plt.xlabel('Importance')
                plt.title(f'Feature Importance - {model_name}')
                plt.tight_layout()
                plt.savefig(f'reports/feature_importance_{model_name}_{timestamp}.png')
                plt.close()
                
    def _empty_metrics(self) -> EvaluationMetrics:
        """Return empty metrics when no data available."""
        return EvaluationMetrics(
            accuracy=0.0,
            precision=0.0,
            recall=0.0,
            f1_score=0.0,
            auc_roc=0.0,
            confusion_matrix=np.zeros((3, 3)),
            classification_report={},
            roi=0.0,
            win_rate=0.0,
            average_odds=0.0,
            kelly_performance=0.0,
            max_drawdown=0.0,
            sharpe_ratio=0.0,
            predictions_count=0,
            correct_predictions=0,
            profit_loss=0.0
        )


if __name__ == "__main__":
    # Test evaluation
    import asyncio
    
    async def test_evaluation():
        config = Config()
        evaluator = ModelEvaluator(config)
        
        results = await evaluator.evaluate_all_models()
        print("Evaluation completed:", results)
        
    asyncio.run(test_evaluation())
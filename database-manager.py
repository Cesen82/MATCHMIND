"""
Database manager using SQLAlchemy with async support
"""

import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
import logging
from pathlib import Path

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, JSON, ForeignKey, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy import select, func, and_, or_, desc
import pandas as pd

from config import DB_PATH, DATA_DIR

logger = logging.getLogger("profootball.database")

Base = declarative_base()


# Database Models
class Prediction(Base):
    """Prediction model"""
    __tablename__ = 'predictions'
    
    id = Column(Integer, primary_key=True)
    fixture_id = Column(String, unique=True, index=True)
    home_team = Column(String, nullable=False)
    away_team = Column(String, nullable=False)
    league = Column(String, nullable=False)
    match_date = Column(DateTime, nullable=False)
    
    # Prediction details
    prediction = Column(Integer, nullable=False)  # 0 or 1
    probability = Column(Float, nullable=False)
    confidence = Column(String, nullable=False)
    confidence_score = Column(Float, nullable=False)
    
    # Betting info
    odds = Column(Float, nullable=False)
    expected_value = Column(Float, nullable=False)
    
    # Features used (JSON)
    features = Column(JSON)
    feature_importance = Column(JSON)
    
    # Result
    actual_result = Column(Integer, nullable=True)  # 0 or 1
    is_correct = Column(Boolean, nullable=True)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Indexes
    __table_args__ = (
        Index('idx_match_date', 'match_date'),
        Index('idx_league', 'league'),
        Index('idx_created_at', 'created_at'),
    )


class BettingSlip(Base):
    """Betting slip model"""
    __tablename__ = 'betting_slips'
    
    id = Column(Integer, primary_key=True)
    slip_type = Column(String, nullable=False)  # single, double, triple, etc
    
    # Slip details
    total_odds = Column(Float, nullable=False)
    combined_probability = Column(Float, nullable=False)
    expected_value = Column(Float, nullable=False)
    kelly_stake = Column(Float, nullable=False)
    recommended_stake = Column(Float, nullable=False)
    
    # Risk assessment
    risk_level = Column(String, nullable=False)
    confidence_score = Column(Float, nullable=False)
    diversification_score = Column(Float, nullable=False)
    
    # Result
    stake_amount = Column(Float, nullable=True)
    potential_return = Column(Float, nullable=True)
    actual_return = Column(Float, nullable=True)
    is_won = Column(Boolean, nullable=True)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    settled_at = Column(DateTime, nullable=True)
    
    # Relationships
    slip_predictions = relationship("SlipPrediction", back_populates="slip")


class SlipPrediction(Base):
    """Predictions in a betting slip"""
    __tablename__ = 'slip_predictions'
    
    id = Column(Integer, primary_key=True)
    slip_id = Column(Integer, ForeignKey('betting_slips.id'))
    prediction_id = Column(Integer, ForeignKey('predictions.id'))
    
    # Relationships
    slip = relationship("BettingSlip", back_populates="slip_predictions")
    prediction = relationship("Prediction")


class TeamStats(Base):
    """Team statistics cache"""
    __tablename__ = 'team_stats'
    
    id = Column(Integer, primary_key=True)
    team_id = Column(Integer, nullable=False)
    team_name = Column(String, nullable=False)
    league_id = Column(Integer, nullable=False)
    season = Column(Integer, nullable=False)
    
    # Statistics
    matches_played = Column(Integer, default=0)
    goals_for = Column(Integer, default=0)
    goals_against = Column(Integer, default=0)
    over25_count = Column(Integer, default=0)
    wins = Column(Integer, default=0)
    draws = Column(Integer, default=0)
    losses = Column(Integer, default=0)
    
    # Calculated fields
    over25_rate = Column(Float, default=0.0)
    goals_per_match = Column(Float, default=0.0)
    points = Column(Integer, default=0)
    
    # Metadata
    updated_at = Column(DateTime, default=datetime.utcnow)
    
    # Unique constraint
    __table_args__ = (
        Index('idx_team_league_season', 'team_id', 'league_id', 'season', unique=True),
    )


class ModelPerformance(Base):
    """Model performance tracking"""
    __tablename__ = 'model_performance'
    
    id = Column(Integer, primary_key=True)
    date = Column(DateTime, nullable=False)
    
    # Metrics
    predictions_count = Column(Integer, default=0)
    correct_predictions = Column(Integer, default=0)
    accuracy = Column(Float, default=0.0)
    precision = Column(Float, default=0.0)
    recall = Column(Float, default=0.0)
    f1_score = Column(Float, default=0.0)
    auc_roc = Column(Float, default=0.0)
    
    # Financial metrics
    total_staked = Column(Float, default=0.0)
    total_returned = Column(Float, default=0.0)
    roi = Column(Float, default=0.0)
    
    # By confidence level
    high_confidence_accuracy = Column(Float, default=0.0)
    medium_confidence_accuracy = Column(Float, default=0.0)
    low_confidence_accuracy = Column(Float, default=0.0)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)


class DatabaseManager:
    """Async database manager"""
    
    def __init__(self, database_url: str = None):
        if database_url is None:
            # Use SQLite with aiosqlite
            database_url = f"sqlite+aiosqlite:///{DB_PATH}"
            
        self.database_url = database_url
        self.engine = None
        self.async_session = None
        
    async def initialize(self):
        """Initialize database connection and create tables"""
        # Create async engine
        self.engine = create_async_engine(
            self.database_url,
            echo=False,
            future=True
        )
        
        # Create session factory
        self.async_session = async_sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
        
        # Create tables
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
            
        logger.info("Database initialized successfully")
        
    async def close(self):
        """Close database connection"""
        if self.engine:
            await self.engine.dispose()
            
    async def save_prediction(self, prediction_data: Dict[str, Any]) -> int:
        """Save a prediction to database"""
        async with self.async_session() as session:
            prediction = Prediction(**prediction_data)
            session.add(prediction)
            await session.commit()
            await session.refresh(prediction)
            return prediction.id
            
    async def save_betting_slip(self, slip_data: Dict[str, Any], prediction_ids: List[int]) -> int:
        """Save a betting slip with predictions"""
        async with self.async_session() as session:
            # Create slip
            slip = BettingSlip(**slip_data)
            session.add(slip)
            await session.flush()
            
            # Add predictions to slip
            for pred_id in prediction_ids:
                slip_pred = SlipPrediction(
                    slip_id=slip.id,
                    prediction_id=pred_id
                )
                session.add(slip_pred)
                
            await session.commit()
            return slip.id
            
    async def update_prediction_result(self, fixture_id: str, actual_result: int):
        """Update prediction with actual result"""
        async with self.async_session() as session:
            stmt = select(Prediction).where(Prediction.fixture_id == fixture_id)
            result = await session.execute(stmt)
            prediction = result.scalar_one_or_none()
            
            if prediction:
                prediction.actual_result = actual_result
                prediction.is_correct = (prediction.prediction == actual_result)
                await session.commit()
                
    async def get_predictions_count(self) -> int:
        """Get total predictions count"""
        async with self.async_session() as session:
            stmt = select(func.count(Prediction.id))
            result = await session.execute(stmt)
            return result.scalar() or 0
            
    async def get_win_rate(self) -> float:
        """Calculate overall win rate"""
        async with self.async_session() as session:
            stmt = select(
                func.count(Prediction.id).filter(Prediction.is_correct == True),
                func.count(Prediction.id).filter(Prediction.actual_result != None)
            )
            result = await session.execute(stmt)
            correct, total = result.first()
            
            if total > 0:
                return correct / total
            return 0.0
            
    async def get_recent_predictions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent predictions"""
        async with self.async_session() as session:
            stmt = select(Prediction).order_by(desc(Prediction.created_at)).limit(limit)
            result = await session.execute(stmt)
            predictions = result.scalars().all()
            
            return [
                {
                    'home_team': p.home_team,
                    'away_team': p.away_team,
                    'date': p.match_date.strftime('%Y-%m-%d'),
                    'time': p.match_date.strftime('%H:%M'),
                    'probability': p.probability,
                    'odds': p.odds,
                    'result': 'won' if p.is_correct else 'lost' if p.is_correct is False else None
                }
                for p in predictions
            ]
            
    async def get_performance_history(self, days: int = 30) -> pd.DataFrame:
        """Get model performance history"""
        async with self.async_session() as session:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            stmt = select(ModelPerformance).where(
                ModelPerformance.date >= cutoff_date
            ).order_by(ModelPerformance.date)
            
            result = await session.execute(stmt)
            performances = result.scalars().all()
            
            if performances:
                data = [
                    {
                        'date': p.date,
                        'accuracy': p.accuracy,
                        'predictions': p.predictions_count,
                        'roi': p.roi
                    }
                    for p in performances
                ]
                return pd.DataFrame(data)
            return pd.DataFrame()
            
    async def update_team_stats(self, stats_data: List[Dict[str, Any]]):
        """Update team statistics"""
        async with self.async_session() as session:
            for data in stats_data:
                # Check if exists
                stmt = select(TeamStats).where(
                    and_(
                        TeamStats.team_id == data['team_id'],
                        TeamStats.league_id == data['league_id'],
                        TeamStats.season == data['season']
                    )
                )
                result = await session.execute(stmt)
                team_stats = result.scalar_one_or_none()
                
                if team_stats:
                    # Update existing
                    for key, value in data.items():
                        setattr(team_stats, key, value)
                else:
                    # Create new
                    team_stats = TeamStats(**data)
                    session.add(team_stats)
                    
            await session.commit()
            
    async def get_team_stats(self, league_id: int, season: int) -> pd.DataFrame:
        """Get team statistics for a league"""
        async with self.async_session() as session:
            stmt = select(TeamStats).where(
                and_(
                    TeamStats.league_id == league_id,
                    TeamStats.season == season
                )
            )
            result = await session.execute(stmt)
            stats = result.scalars().all()
            
            if stats:
                data = [
                    {
                        'name': s.team_name,
                        'matches': s.matches_played,
                        'goals_for': s.goals_for,
                        'goals_against': s.goals_against,
                        'over25': s.over25_count,
                        'over25_rate': s.over25_rate,
                        'goals_per_match': s.goals_per_match
                    }
                    for s in stats
                ]
                return pd.DataFrame(data)
            return pd.DataFrame()
            
    async def calculate_daily_performance(self):
        """Calculate and store daily performance metrics"""
        async with self.async_session() as session:
            # Get today's predictions
            today = datetime.utcnow().date()
            tomorrow = today + timedelta(days=1)
            
            stmt = select(Prediction).where(
                and_(
                    Prediction.created_at >= today,
                    Prediction.created_at < tomorrow,
                    Prediction.actual_result != None
                )
            )
            result = await session.execute(stmt)
            predictions = result.scalars().all()
            
            if predictions:
                total = len(predictions)
                correct = sum(1 for p in predictions if p.is_correct)
                
                # Calculate metrics
                accuracy = correct / total
                
                # Financial metrics (simplified)
                total_staked = total * 10  # Assume €10 per bet
                total_returned = sum(p.odds * 10 for p in predictions if p.is_correct)
                roi = (total_returned - total_staked) / total_staked
                
                # Save performance
                performance = ModelPerformance(
                    date=today,
                    predictions_count=total,
                    correct_predictions=correct,
                    accuracy=accuracy,
                    total_staked=total_staked,
                    total_returned=total_returned,
                    roi=roi
                )
                session.add(performance)
                await session.commit()
                
    async def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        # Get file size
        db_size = 0
        if Path(DB_PATH).exists():
            db_size = Path(DB_PATH).stat().st_size
            
        async with self.async_session() as session:
            # Count records
            predictions_count = await session.execute(select(func.count(Prediction.id)))
            slips_count = await session.execute(select(func.count(BettingSlip.id)))
            teams_count = await session.execute(select(func.count(TeamStats.id)))
            
            return {
                'size_mb': db_size / (1024 * 1024),
                'records': predictions_count.scalar() + slips_count.scalar() + teams_count.scalar(),
                'predictions': predictions_count.scalar(),
                'betting_slips': slips_count.scalar(),
                'team_stats': teams_count.scalar()
            }
            
    async def backup(self):
        """Create database backup"""
        if Path(DB_PATH).exists():
            backup_path = DATA_DIR / f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
            import shutil
            shutil.copy2(DB_PATH, backup_path)
            logger.info(f"Database backed up to {backup_path}")
            
    async def get_training_data(self) -> Optional[pd.DataFrame]:
        """Get historical data for model training"""
        async with self.async_session() as session:
            stmt = select(Prediction).where(
                Prediction.actual_result != None
            ).limit(10000)
            
            result = await session.execute(stmt)
            predictions = result.scalars().all()
            
            if len(predictions) > 100:
                data = []
                for p in predictions:
                    if p.features:
                        features = p.features.copy()
                        features['over25'] = p.actual_result
                        data.append(features)
                        
                return pd.DataFrame(data)
            return None
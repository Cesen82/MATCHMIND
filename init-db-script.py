#!/usr/bin/env python
"""
Script to initialize the database and optionally populate with sample data
"""

import asyncio
import argparse
import logging
from pathlib import Path
import sys
from datetime import datetime, timedelta
import numpy as np

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.database import DatabaseManager, Prediction, TeamStats, BettingSlip
from src.utils.logger import setup_logging
from config import DB_PATH, LEAGUES

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)


async def init_database(populate_sample: bool = False):
    """Initialize database and optionally populate with sample data"""
    
    logger.info("Initializing ProFootballAI database...")
    
    # Check if database already exists
    if DB_PATH.exists() and not populate_sample:
        response = input(f"Database already exists at {DB_PATH}. Overwrite? (y/N): ")
        if response.lower() != 'y':
            logger.info("Database initialization cancelled")
            return False
    
    # Initialize database manager
    db_manager = DatabaseManager()
    
    try:
        # Initialize database (creates tables)
        await db_manager.initialize()
        logger.info("✅ Database tables created successfully")
        
        if populate_sample:
            logger.info("Populating with sample data...")
            await populate_sample_data(db_manager)
            logger.info("✅ Sample data populated successfully")
        
        # Get database stats
        stats = await db_manager.get_database_stats()
        logger.info(f"Database stats:")
        logger.info(f"  - Size: {stats['size_mb']:.2f} MB")
        logger.info(f"  - Total records: {stats['records']:,}")
        logger.info(f"  - Predictions: {stats['predictions']:,}")
        logger.info(f"  - Team stats: {stats['team_stats']:,}")
        logger.info(f"  - Betting slips: {stats['betting_slips']:,}")
        
        return True
        
    except Exception as e:
        logger.error(f"Database initialization failed: {str(e)}")
        return False
        
    finally:
        await db_manager.close()


async def populate_sample_data(db_manager: DatabaseManager):
    """Populate database with sample data"""
    
    # Sample teams by league
    teams_by_league = {
        39: ["Man City", "Arsenal", "Liverpool", "Chelsea", "Man United"],  # Premier League
        135: ["Inter", "Juventus", "AC Milan", "Napoli", "Roma"],  # Serie A
        140: ["Real Madrid", "Barcelona", "Atletico", "Sevilla", "Villarreal"],  # La Liga
        78: ["Bayern", "Dortmund", "Leipzig", "Leverkusen", "Frankfurt"],  # Bundesliga
        61: ["PSG", "Monaco", "Marseille", "Lyon", "Lille"],  # Ligue 1
    }
    
    # Add team statistics
    logger.info("Adding team statistics...")
    for league_id, teams in teams_by_league.items():
        for i, team in enumerate(teams):
            # Generate realistic stats based on team position
            if i < 2:  # Top teams
                goals_for = np.random.randint(60, 80)
                goals_against = np.random.randint(20, 35)
                over25_count = np.random.randint(20, 28)
                wins = np.random.randint(20, 28)
            elif i < 4:  # Mid-table
                goals_for = np.random.randint(40, 60)
                goals_against = np.random.randint(35, 50)
                over25_count = np.random.randint(15, 22)
                wins = np.random.randint(12, 20)
            else:  # Lower teams
                goals_for = np.random.randint(25, 40)
                goals_against = np.random.randint(45, 65)
                over25_count = np.random.randint(10, 18)
                wins = np.random.randint(5, 12)
            
            matches_played = 30
            draws = np.random.randint(5, 10)
            losses = matches_played - wins - draws
            
            stats_data = {
                'team_id': league_id * 100 + i,
                'team_name': team,
                'league_id': league_id,
                'season': datetime.now().year,
                'matches_played': matches_played,
                'goals_for': goals_for,
                'goals_against': goals_against,
                'over25_count': over25_count,
                'wins': wins,
                'draws': draws,
                'losses': losses,
                'over25_rate': over25_count / matches_played,
                'goals_per_match': (goals_for + goals_against) / matches_played,
                'points': wins * 3 + draws
            }
            
            await db_manager.update_team_stats([stats_data])
    
    # Add sample predictions
    logger.info("Adding sample predictions...")
    
    for days_ago in range(30):
        date = datetime.now() - timedelta(days=days_ago)
        
        # Generate 5-10 predictions per day
        for _ in range(np.random.randint(5, 11)):
            # Random teams
            league_id = np.random.choice(list(teams_by_league.keys()))
            teams = teams_by_league[league_id]
            home_team = np.random.choice(teams)
            away_team = np.random.choice([t for t in teams if t != home_team])
            
            # Generate prediction
            probability = np.random.uniform(0.5, 0.85)
            prediction = 1 if probability > 0.5 else 0
            confidence = "High" if probability > 0.75 else "Medium" if probability > 0.65 else "Low"
            odds = round(1.85 + np.random.uniform(-0.4, 0.4), 2)
            
            # Result (for past predictions)
            if days_ago > 1:
                actual_result = 1 if np.random.random() < probability else 0
                is_correct = (prediction == actual_result)
            else:
                actual_result = None
                is_correct = None
            
            prediction_data = {
                'fixture_id': f"FIX_{league_id}_{days_ago}_{_}",
                'home_team': home_team,
                'away_team': away_team,
                'league': f"League {league_id}",
                'match_date': date,
                'prediction': prediction,
                'probability': probability,
                'confidence': confidence,
                'confidence_score': probability,
                'odds': odds,
                'expected_value': (probability * odds) - 1,
                'features': {
                    'home_goals_avg': np.random.uniform(1.0, 2.5),
                    'away_goals_avg': np.random.uniform(1.0, 2.5),
                    'league_over25_avg': np.random.uniform(0.4, 0.7)
                },
                'actual_result': actual_result,
                'is_correct': is_correct
            }
            
            await db_manager.save_prediction(prediction_data)
    
    # Add sample betting slips
    logger.info("Adding sample betting slips...")
    
    for days_ago in range(10):
        date = datetime.now() - timedelta(days=days_ago)
        
        # Generate 1-3 slips per day
        for _ in range(np.random.randint(1, 4)):
            slip_type = np.random.choice(['single', 'double', 'triple'])
            num_matches = {'single': 1, 'double': 2, 'triple': 3}[slip_type]
            
            total_odds = round(np.random.uniform(1.5, 8.0), 2)
            combined_prob = np.random.uniform(0.3, 0.7)
            
            slip_data = {
                'slip_type': slip_type,
                'total_odds': total_odds,
                'combined_probability': combined_prob,
                'expected_value': (combined_prob * total_odds) - 1,
                'kelly_stake': min(0.05, (combined_prob * (total_odds - 1) - (1 - combined_prob)) / (total_odds - 1)),
                'recommended_stake': np.random.uniform(0.01, 0.05),
                'risk_level': np.random.choice(['Low', 'Medium', 'High']),
                'confidence_score': combined_prob,
                'diversification_score': np.random.uniform(0.6, 0.9),
                'stake_amount': np.random.choice([10, 20, 50, 100]),
                'potential_return': None,
                'actual_return': None,
                'is_won': None if days_ago <= 1 else np.random.random() < combined_prob,
                'created_at': date
            }
            
            # For settled slips
            if slip_data['is_won'] is not None:
                if slip_data['is_won']:
                    slip_data['actual_return'] = slip_data['stake_amount'] * total_odds
                else:
                    slip_data['actual_return'] = 0
                slip_data['settled_at'] = date + timedelta(hours=3)
            
            # Save slip (in real app would link to predictions)
            await db_manager.save_betting_slip(slip_data, [])
    
    # Calculate daily performance
    logger.info("Calculating performance metrics...")
    await db_manager.calculate_daily_performance()


def main():
    """Main entry point"""
    
    parser = argparse.ArgumentParser(description="Initialize ProFootballAI database")
    parser.add_argument(
        "--populate",
        action="store_true",
        help="Populate with sample data"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force initialization without confirmation"
    )
    
    args = parser.parse_args()
    
    # Run initialization
    success = asyncio.run(init_database(populate_sample=args.populate))
    
    if success:
        logger.info("✅ Database initialization completed successfully!")
        logger.info(f"Database location: {DB_PATH}")
    else:
        logger.error("❌ Database initialization failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
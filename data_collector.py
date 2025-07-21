"""
Data Collector Module
====================

Collects football data from various sources and stores it in the database.
Handles scheduled data collection, error recovery, and data validation.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import pandas as pd
from sqlalchemy.orm import Session

from database_manager import DatabaseManager
from football_api import FootballAPI
from cache_manager import CacheManager
from exceptions_module import DataCollectionError, APIError
from validators_module import DataValidator
from config import Config

logger = logging.getLogger(__name__)


class DataCollector:
    """Handles automated data collection from football APIs."""
    
    def __init__(self, 
                 db_manager: DatabaseManager,
                 football_api: FootballAPI,
                 cache_manager: CacheManager,
                 config: Config):
        """
        Initialize data collector.
        
        Args:
            db_manager: Database manager instance
            football_api: Football API client
            cache_manager: Cache manager instance
            config: Configuration object
        """
        self.db = db_manager
        self.api = football_api
        self.cache = cache_manager
        self.config = config
        self.validator = DataValidator()
        
    async def collect_all_data(self, 
                              leagues: Optional[List[str]] = None,
                              days_back: int = 7) -> Dict[str, Any]:
        """
        Collect all types of data for specified leagues.
        
        Args:
            leagues: List of league IDs to collect data for
            days_back: Number of days to look back for matches
            
        Returns:
            Summary of collected data
        """
        if leagues is None:
            leagues = self.config.SUPPORTED_LEAGUES
            
        start_date = datetime.now() - timedelta(days=days_back)
        end_date = datetime.now() + timedelta(days=7)  # Include upcoming matches
        
        summary = {
            'leagues': leagues,
            'start_date': start_date,
            'end_date': end_date,
            'collected': {
                'matches': 0,
                'teams': 0,
                'players': 0,
                'standings': 0,
                'odds': 0
            },
            'errors': []
        }
        
        try:
            # Collect data for each league
            for league_id in leagues:
                logger.info(f"Collecting data for league {league_id}")
                
                try:
                    # Collect league info and standings
                    await self._collect_league_data(league_id, summary)
                    
                    # Collect teams data
                    await self._collect_teams_data(league_id, summary)
                    
                    # Collect matches
                    await self._collect_matches_data(
                        league_id, start_date, end_date, summary
                    )
                    
                    # Collect odds data
                    await self._collect_odds_data(league_id, summary)
                    
                except Exception as e:
                    logger.error(f"Error collecting data for league {league_id}: {e}")
                    summary['errors'].append({
                        'league': league_id,
                        'error': str(e),
                        'timestamp': datetime.now()
                    })
                    
        except Exception as e:
            logger.error(f"Critical error in data collection: {e}")
            raise DataCollectionError(f"Data collection failed: {e}")
            
        return summary
        
    async def _collect_league_data(self, 
                                  league_id: str, 
                                  summary: Dict[str, Any]) -> None:
        """Collect league information and standings."""
        try:
            # Get league info
            league_info = await self.api.get_league_info(league_id)
            if self.validator.validate_league_data(league_info):
                self.db.upsert_league(league_info)
                
            # Get current standings
            standings = await self.api.get_league_standings(league_id)
            if standings and self.validator.validate_standings_data(standings):
                self.db.update_standings(league_id, standings)
                summary['collected']['standings'] += 1
                
        except APIError as e:
            logger.warning(f"Failed to collect league data: {e}")
            raise
            
    async def _collect_teams_data(self, 
                                 league_id: str, 
                                 summary: Dict[str, Any]) -> None:
        """Collect teams data for a league."""
        try:
            teams = await self.api.get_teams(league_id)
            
            for team in teams:
                if self.validator.validate_team_data(team):
                    # Get detailed team stats
                    team_stats = await self.api.get_team_statistics(
                        team['id'], league_id
                    )
                    team['statistics'] = team_stats
                    
                    # Get squad information
                    squad = await self.api.get_team_squad(team['id'])
                    team['squad'] = squad
                    
                    # Store in database
                    self.db.upsert_team(team)
                    summary['collected']['teams'] += 1
                    
                    # Collect player data
                    for player in squad:
                        if self.validator.validate_player_data(player):
                            self.db.upsert_player(player)
                            summary['collected']['players'] += 1
                            
        except APIError as e:
            logger.warning(f"Failed to collect teams data: {e}")
            raise
            
    async def _collect_matches_data(self,
                                   league_id: str,
                                   start_date: datetime,
                                   end_date: datetime,
                                   summary: Dict[str, Any]) -> None:
        """Collect matches data for a date range."""
        try:
            # Get matches
            matches = await self.api.get_matches(
                league_id=league_id,
                start_date=start_date,
                end_date=end_date
            )
            
            for match in matches:
                if self.validator.validate_match_data(match):
                    # Get detailed match statistics if match is finished
                    if match['status'] == 'finished':
                        match_stats = await self.api.get_match_statistics(
                            match['id']
                        )
                        match['statistics'] = match_stats
                        
                        # Get match events
                        events = await self.api.get_match_events(match['id'])
                        match['events'] = events
                        
                    # Store in database
                    self.db.upsert_match(match)
                    summary['collected']['matches'] += 1
                    
                    # Cache recent matches for quick access
                    if match['date'] >= datetime.now() - timedelta(days=1):
                        cache_key = f"match:{match['id']}"
                        self.cache.set(cache_key, match, ttl=3600)
                        
        except APIError as e:
            logger.warning(f"Failed to collect matches data: {e}")
            raise
            
    async def _collect_odds_data(self,
                                league_id: str,
                                summary: Dict[str, Any]) -> None:
        """Collect odds data for upcoming matches."""
        try:
            # Get upcoming matches
            upcoming_matches = await self.api.get_matches(
                league_id=league_id,
                start_date=datetime.now(),
                end_date=datetime.now() + timedelta(days=7),
                status='scheduled'
            )
            
            for match in upcoming_matches:
                try:
                    # Get odds from multiple bookmakers
                    odds = await self.api.get_match_odds(match['id'])
                    
                    if odds and self.validator.validate_odds_data(odds):
                        # Store odds history
                        self.db.insert_odds(match['id'], odds)
                        summary['collected']['odds'] += 1
                        
                        # Cache current odds
                        cache_key = f"odds:{match['id']}"
                        self.cache.set(cache_key, odds, ttl=300)  # 5 minutes
                        
                except Exception as e:
                    logger.warning(f"Failed to collect odds for match {match['id']}: {e}")
                    
        except APIError as e:
            logger.warning(f"Failed to collect odds data: {e}")
            raise
            
    async def collect_live_data(self) -> Dict[str, Any]:
        """Collect live match data for in-play betting."""
        summary = {
            'timestamp': datetime.now(),
            'live_matches': 0,
            'updates': 0
        }
        
        try:
            # Get all live matches
            live_matches = await self.api.get_live_matches()
            summary['live_matches'] = len(live_matches)
            
            for match in live_matches:
                try:
                    # Get live statistics
                    live_stats = await self.api.get_live_match_data(match['id'])
                    
                    if live_stats:
                        # Update database
                        self.db.update_live_match(match['id'], live_stats)
                        
                        # Update cache for real-time access
                        cache_key = f"live:{match['id']}"
                        self.cache.set(cache_key, live_stats, ttl=30)  # 30 seconds
                        
                        summary['updates'] += 1
                        
                except Exception as e:
                    logger.warning(f"Failed to update live match {match['id']}: {e}")
                    
        except Exception as e:
            logger.error(f"Error collecting live data: {e}")
            
        return summary
        
    async def update_historical_data(self,
                                   league_id: str,
                                   season: str) -> Dict[str, Any]:
        """Update historical data for a complete season."""
        summary = {
            'league': league_id,
            'season': season,
            'matches_updated': 0,
            'errors': []
        }
        
        try:
            # Get all matches for the season
            matches = await self.api.get_season_matches(league_id, season)
            
            for match in matches:
                try:
                    if match['status'] == 'finished':
                        # Get complete match data
                        full_data = await self.api.get_historical_match(match['id'])
                        
                        if full_data and self.validator.validate_match_data(full_data):
                            self.db.upsert_match(full_data)
                            summary['matches_updated'] += 1
                            
                except Exception as e:
                    logger.warning(f"Failed to update match {match['id']}: {e}")
                    summary['errors'].append({
                        'match_id': match['id'],
                        'error': str(e)
                    })
                    
        except Exception as e:
            logger.error(f"Error updating historical data: {e}")
            raise DataCollectionError(f"Historical data update failed: {e}")
            
        return summary
        
    def get_collection_status(self) -> Dict[str, Any]:
        """Get current data collection status and statistics."""
        with self.db.get_session() as session:
            status = {
                'last_update': self.db.get_last_update_time(),
                'data_counts': {
                    'leagues': self.db.count_leagues(),
                    'teams': self.db.count_teams(),
                    'players': self.db.count_players(),
                    'matches': self.db.count_matches(),
                    'odds_records': self.db.count_odds_records()
                },
                'coverage': self._calculate_data_coverage(),
                'health': self._check_data_health()
            }
            
        return status
        
    def _calculate_data_coverage(self) -> Dict[str, float]:
        """Calculate data coverage percentages."""
        coverage = {}
        
        # Check match statistics coverage
        total_matches = self.db.count_matches(status='finished')
        matches_with_stats = self.db.count_matches_with_statistics()
        coverage['match_statistics'] = (
            matches_with_stats / total_matches * 100 if total_matches > 0 else 0
        )
        
        # Check odds coverage
        upcoming_matches = self.db.count_matches(status='scheduled')
        matches_with_odds = self.db.count_matches_with_odds()
        coverage['odds_coverage'] = (
            matches_with_odds / upcoming_matches * 100 if upcoming_matches > 0 else 0
        )
        
        return coverage
        
    def _check_data_health(self) -> Dict[str, Any]:
        """Check data health and identify issues."""
        health = {
            'status': 'healthy',
            'issues': []
        }
        
        # Check for stale data
        last_update = self.db.get_last_update_time()
        if last_update and (datetime.now() - last_update).hours > 24:
            health['issues'].append({
                'type': 'stale_data',
                'message': 'Data has not been updated in over 24 hours',
                'severity': 'warning'
            })
            
        # Check for missing data
        missing_stats = self.db.find_matches_missing_statistics()
        if len(missing_stats) > 10:
            health['issues'].append({
                'type': 'missing_statistics',
                'message': f'{len(missing_stats)} matches missing statistics',
                'severity': 'warning'
            })
            
        if health['issues']:
            health['status'] = 'unhealthy'
            
        return health


# Standalone collection functions for scripts
async def collect_daily_data(config: Config) -> None:
    """Run daily data collection."""
    db_manager = DatabaseManager(config)
    football_api = FootballAPI(config)
    cache_manager = CacheManager(config)
    
    collector = DataCollector(db_manager, football_api, cache_manager, config)
    
    logger.info("Starting daily data collection...")
    summary = await collector.collect_all_data(days_back=2)
    logger.info(f"Data collection completed: {summary}")
    

async def collect_live_matches(config: Config) -> None:
    """Collect live match data."""
    db_manager = DatabaseManager(config)
    football_api = FootballAPI(config)
    cache_manager = CacheManager(config)
    
    collector = DataCollector(db_manager, football_api, cache_manager, config)
    
    logger.info("Collecting live match data...")
    summary = await collector.collect_live_data()
    logger.info(f"Live data collection completed: {summary}")


if __name__ == "__main__":
    # For testing
    import asyncio
    from config import Config
    
    config = Config()
    asyncio.run(collect_daily_data(config))
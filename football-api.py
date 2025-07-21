"""
Advanced Football API Client with rate limiting, caching, and error handling
"""

import asyncio
import aiohttp
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from functools import lru_cache
import backoff

from ..utils.exceptions import APIError, RateLimitError, InvalidDataError
from ..data.cache_manager import CacheManager
from .rate_limiter import RateLimiter
from config import API_CONFIG, LEAGUES, get_league_season

logger = logging.getLogger("profootball.api")


@dataclass
class TeamStats:
    """Team statistics data class"""
    team_id: int
    team_name: str
    matches_played: int
    goals_for: int
    goals_against: int
    over25_count: int
    wins: int
    draws: int
    losses: int
    
    @property
    def over25_rate(self) -> float:
        return self.over25_count / max(self.matches_played, 1)
    
    @property
    def goals_per_match(self) -> float:
        return (self.goals_for + self.goals_against) / max(self.matches_played, 1)
    
    @property
    def points(self) -> int:
        return self.wins * 3 + self.draws


@dataclass
class Fixture:
    """Match fixture data class"""
    fixture_id: int
    home_team: str
    home_team_id: int
    away_team: str
    away_team_id: int
    date: datetime
    venue: str
    status: str
    league: str
    referee: Optional[str] = None
    
    @property
    def is_upcoming(self) -> bool:
        return self.date > datetime.now()
    
    @property
    def match_string(self) -> str:
        return f"{self.home_team} vs {self.away_team}"


class FootballAPIClient:
    """Advanced Football API client with async support"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or API_CONFIG["api_key"]
        if not self.api_key:
            raise ValueError("API key is required")
            
        self.base_url = API_CONFIG["base_url"]
        self.headers = {"x-apisports-key": self.api_key}
        
        # Initialize components
        self.cache = CacheManager()
        self.rate_limiter = RateLimiter(
            calls_per_hour=API_CONFIG["rate_limit"]["calls_per_hour"],
            calls_per_day=API_CONFIG["rate_limit"]["calls_per_day"]
        )
        
        # Session management
        self._session: Optional[aiohttp.ClientSession] = None
        
    async def __aenter__(self):
        """Async context manager entry"""
        await self._init_session()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()
        
    async def _init_session(self):
        """Initialize aiohttp session"""
        if not self._session:
            timeout = aiohttp.ClientTimeout(total=API_CONFIG["timeout"])
            self._session = aiohttp.ClientSession(
                headers=self.headers,
                timeout=timeout
            )
            
    async def close(self):
        """Close aiohttp session"""
        if self._session:
            await self._session.close()
            self._session = None
            
    @backoff.on_exception(
        backoff.expo,
        (aiohttp.ClientError, asyncio.TimeoutError),
        max_tries=API_CONFIG["max_retries"],
        max_time=60
    )
    async def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """Make API request with retry logic"""
        # Check rate limit
        if not await self.rate_limiter.check_limit():
            raise RateLimitError("API rate limit exceeded")
            
        # Build cache key
        cache_key = self.cache.build_key(endpoint, params)
        
        # Check cache
        cached_data = await self.cache.get(cache_key)
        if cached_data:
            logger.debug(f"Cache hit for {endpoint}")
            return cached_data
            
        # Make request
        await self._init_session()
        url = f"{self.base_url}/{endpoint}"
        
        try:
            async with self._session.get(url, params=params) as response:
                await self.rate_limiter.record_call()
                
                if response.status == 429:
                    retry_after = int(response.headers.get("Retry-After", 60))
                    raise RateLimitError(f"Rate limit hit, retry after {retry_after}s")
                    
                response.raise_for_status()
                data = await response.json()
                
                # Validate response
                if "errors" in data and data["errors"]:
                    error_msg = data["errors"]
                    if isinstance(error_msg, dict):
                        error_msg = error_msg.get("requests", "Unknown error")
                    raise APIError(f"API Error: {error_msg}")
                    
                # Cache successful response
                await self.cache.set(cache_key, data)
                
                return data
                
        except aiohttp.ClientResponseError as e:
            logger.error(f"HTTP {e.status} error for {endpoint}: {e.message}")
            raise APIError(f"HTTP {e.status}: {e.message}")
        except Exception as e:
            logger.error(f"Request failed for {endpoint}: {str(e)}")
            raise
            
    async def get_team_stats(self, league_id: int, season: int) -> List[TeamStats]:
        """Get team statistics for a league"""
        # Get teams first
        teams_data = await self._make_request(
            "teams",
            {"league": league_id, "season": season}
        )
        
        if not teams_data.get("response"):
            logger.warning(f"No teams found for league {league_id}")
            return []
            
        stats_list = []
        
        # Get stats for each team (limit for performance)
        teams = teams_data["response"][:20]  # Top 20 teams
        
        # Batch requests for efficiency
        tasks = []
        for team_info in teams:
            team_id = team_info["team"]["id"]
            task = self._get_single_team_stats(team_id, league_id, season)
            tasks.append(task)
            
        # Execute in parallel with semaphore to avoid overwhelming API
        sem = asyncio.Semaphore(5)  # Max 5 concurrent requests
        
        async def bounded_task(task):
            async with sem:
                return await task
                
        results = await asyncio.gather(
            *[bounded_task(task) for task in tasks],
            return_exceptions=True
        )
        
        # Process results
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Failed to get stats for team {teams[i]['team']['name']}: {result}")
                continue
            if result:
                stats_list.append(result)
                
        return stats_list
        
    async def _get_single_team_stats(self, team_id: int, league_id: int, season: int) -> Optional[TeamStats]:
        """Get statistics for a single team"""
        try:
            data = await self._make_request(
                "teams/statistics",
                {"team": team_id, "league": league_id, "season": season}
            )
            
            if not data.get("response"):
                return None
                
            stats = data["response"]
            team_name = stats.get("team", {}).get("name", "Unknown")
            
            fixtures = stats.get("fixtures", {})
            goals = stats.get("goals", {})
            
            matches_played = fixtures.get("played", {}).get("total", 0)
            wins = fixtures.get("wins", {}).get("total", 0)
            draws = fixtures.get("draws", {}).get("total", 0)
            losses = fixtures.get("loses", {}).get("total", 0)
            
            goals_for = goals.get("for", {}).get("total", {}).get("total", 0)
            goals_against = goals.get("against", {}).get("total", {}).get("total", 0)
            
            # Calculate Over 2.5 (estimation based on average goals)
            if matches_played > 0:
                avg_goals_per_match = (goals_for + goals_against) / matches_played
                # Estimation formula
                if avg_goals_per_match > 3.5:
                    over25_rate = 0.80
                elif avg_goals_per_match > 3.0:
                    over25_rate = 0.65
                elif avg_goals_per_match > 2.5:
                    over25_rate = 0.50
                elif avg_goals_per_match > 2.0:
                    over25_rate = 0.35
                else:
                    over25_rate = 0.25
                    
                over25_count = int(matches_played * over25_rate)
            else:
                over25_count = 0
                
            return TeamStats(
                team_id=team_id,
                team_name=team_name,
                matches_played=matches_played,
                goals_for=goals_for,
                goals_against=goals_against,
                over25_count=over25_count,
                wins=wins,
                draws=draws,
                losses=losses
            )
            
        except Exception as e:
            logger.error(f"Error getting stats for team {team_id}: {e}")
            return None
            
    async def get_fixtures(
        self, 
        league_id: int, 
        season: int,
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None,
        limit: int = 50
    ) -> List[Fixture]:
        """Get fixtures for a league"""
        if not from_date:
            from_date = datetime.now()
        if not to_date:
            to_date = from_date + timedelta(days=7)
            
        params = {
            "league": league_id,
            "season": season,
            "from": from_date.strftime("%Y-%m-%d"),
            "to": to_date.strftime("%Y-%m-%d")
        }
        
        data = await self._make_request("fixtures", params)
        
        if not data.get("response"):
            logger.warning(f"No fixtures found for league {league_id}")
            return []
            
        fixtures = []
        for fixture_data in data["response"][:limit]:
            try:
                fixture_info = fixture_data["fixture"]
                teams = fixture_data["teams"]
                
                fixture = Fixture(
                    fixture_id=fixture_info["id"],
                    home_team=teams["home"]["name"],
                    home_team_id=teams["home"]["id"],
                    away_team=teams["away"]["name"],
                    away_team_id=teams["away"]["id"],
                    date=datetime.fromtimestamp(fixture_info["timestamp"]),
                    venue=fixture_info.get("venue", {}).get("name", "Unknown"),
                    status=fixture_info.get("status", {}).get("long", "Scheduled"),
                    league=self._get_league_name(league_id),
                    referee=fixture_info.get("referee")
                )
                fixtures.append(fixture)
                
            except Exception as e:
                logger.error(f"Error parsing fixture: {e}")
                continue
                
        return fixtures
        
    async def get_h2h_stats(self, team1_id: int, team2_id: int, limit: int = 10) -> Dict[str, Any]:
        """Get head-to-head statistics between two teams"""
        params = {"h2h": f"{team1_id}-{team2_id}", "last": limit}
        
        data = await self._make_request("fixtures/headtohead", params)
        
        if not data.get("response"):
            return {"matches": 0, "over25_count": 0, "avg_goals": 0}
            
        matches = data["response"]
        total_matches = len(matches)
        over25_count = 0
        total_goals = 0
        
        for match in matches:
            goals = match.get("goals", {})
            home_goals = goals.get("home", 0) or 0
            away_goals = goals.get("away", 0) or 0
            total = home_goals + away_goals
            
            if total > 2.5:
                over25_count += 1
            total_goals += total
            
        return {
            "matches": total_matches,
            "over25_count": over25_count,
            "over25_rate": over25_count / max(total_matches, 1),
            "avg_goals": total_goals / max(total_matches, 1)
        }
        
    async def get_team_form(self, team_id: int, last_n: int = 5) -> List[str]:
        """Get team's recent form (W/D/L)"""
        params = {"team": team_id, "last": last_n}
        
        data = await self._make_request("fixtures", params)
        
        if not data.get("response"):
            return []
            
        form = []
        for match in data["response"]:
            teams = match["teams"]
            goals = match["goals"]
            
            if teams["home"]["id"] == team_id:
                home_goals = goals.get("home", 0) or 0
                away_goals = goals.get("away", 0) or 0
                
                if home_goals > away_goals:
                    form.append("W")
                elif home_goals < away_goals:
                    form.append("L")
                else:
                    form.append("D")
            else:
                home_goals = goals.get("home", 0) or 0
                away_goals = goals.get("away", 0) or 0
                
                if away_goals > home_goals:
                    form.append("W")
                elif away_goals < home_goals:
                    form.append("L")
                else:
                    form.append("D")
                    
        return form
        
    @lru_cache(maxsize=128)
    def _get_league_name(self, league_id: int) -> str:
        """Get league name from ID"""
        for name, info in LEAGUES.items():
            if info["id"] == league_id:
                return name.split(' ', 1)[1] if ' ' in name else name
        return f"League {league_id}"
        
    async def get_league_standings(self, league_id: int, season: int) -> List[Dict[str, Any]]:
        """Get league standings"""
        params = {"league": league_id, "season": season}
        
        data = await self._make_request("standings", params)
        
        if not data.get("response") or not data["response"][0].get("league"):
            return []
            
        standings = data["response"][0]["league"]["standings"][0]
        
        return [
            {
                "position": team["rank"],
                "team": team["team"]["name"],
                "team_id": team["team"]["id"],
                "played": team["all"]["played"],
                "points": team["points"],
                "goals_for": team["all"]["goals"]["for"],
                "goals_against": team["all"]["goals"]["against"],
                "form": team.get("form", "")
            }
            for team in standings
        ]
        
    def get_status(self) -> Dict[str, Any]:
        """Get API client status"""
        return {
            "rate_limit": self.rate_limiter.get_status(),
            "cache": self.cache.get_status(),
            "api_key_valid": bool(self.api_key)
        }
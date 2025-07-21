"""
Tests for API module
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
import aiohttp

from src.api.football_api import FootballAPIClient, TeamStats, Fixture
from src.api.rate_limiter import RateLimiter
from src.api.cache_manager import CacheManager
from src.utils.exceptions import APIError, RateLimitError, AuthenticationError


class TestFootballAPIClient:
    """Test cases for FootballAPIClient"""
    
    @pytest.fixture
    async def api_client(self):
        """Create API client instance"""
        client = FootballAPIClient(api_key="test_key")
        await client._init_session()
        yield client
        await client.close()
    
    @pytest.fixture
    def mock_response(self):
        """Create mock API response"""
        return {
            "response": [{
                "team": {"id": 1, "name": "Test Team"},
                "venue": {"name": "Test Stadium"}
            }],
            "errors": []
        }
    
    @pytest.mark.asyncio
    async def test_init_session(self, api_client):
        """Test session initialization"""
        assert api_client._session is not None
        assert isinstance(api_client._session, aiohttp.ClientSession)
        assert api_client.headers["x-apisports-key"] == "test_key"
    
    @pytest.mark.asyncio
    async def test_make_request_success(self, api_client, mock_response):
        """Test successful API request"""
        with patch.object(api_client._session, 'get') as mock_get:
            mock_resp = AsyncMock()
            mock_resp.status = 200
            mock_resp.json = AsyncMock(return_value=mock_response)
            mock_get.return_value.__aenter__.return_value = mock_resp
            
            result = await api_client._make_request("test", {"param": "value"})
            
            assert result == mock_response
            mock_get.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_make_request_rate_limit(self, api_client):
        """Test rate limit handling"""
        with patch.object(api_client._session, 'get') as mock_get:
            mock_resp = AsyncMock()
            mock_resp.status = 429
            mock_resp.headers = {"Retry-After": "60"}
            mock_get.return_value.__aenter__.return_value = mock_resp
            
            with pytest.raises(RateLimitError) as exc_info:
                await api_client._make_request("test")
            
            assert "Rate limit hit" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_make_request_auth_error(self, api_client):
        """Test authentication error handling"""
        with patch.object(api_client._session, 'get') as mock_get:
            mock_resp = AsyncMock()
            mock_resp.status = 401
            mock_resp.raise_for_status.side_effect = aiohttp.ClientResponseError(
                request_info=None,
                history=None,
                status=401
            )
            mock_get.return_value.__aenter__.return_value = mock_resp
            
            with pytest.raises(APIError):
                await api_client._make_request("test")
    
    @pytest.mark.asyncio
    async def test_get_team_stats(self, api_client):
        """Test getting team statistics"""
        mock_teams = {
            "response": [
                {"team": {"id": 1, "name": "Team 1"}},
                {"team": {"id": 2, "name": "Team 2"}}
            ]
        }
        
        mock_stats = {
            "response": {
                "team": {"name": "Team 1"},
                "fixtures": {"played": {"total": 30}},
                "goals": {"for": {"total": {"total": 60}}},
                "goals": {"against": {"total": {"total": 40}}}
            }
        }
        
        with patch.object(api_client, '_make_request') as mock_request:
            mock_request.side_effect = [mock_teams, mock_stats, mock_stats]
            
            stats = await api_client.get_team_stats(39, 2024)
            
            assert len(stats) > 0
            assert isinstance(stats[0], TeamStats)
    
    @pytest.mark.asyncio
    async def test_get_fixtures(self, api_client):
        """Test getting fixtures"""
        mock_fixtures = {
            "response": [{
                "fixture": {
                    "id": 123,
                    "timestamp": int(datetime.now().timestamp()),
                    "venue": {"name": "Stadium"},
                    "status": {"long": "Not Started"}
                },
                "teams": {
                    "home": {"id": 1, "name": "Home Team"},
                    "away": {"id": 2, "name": "Away Team"}
                }
            }]
        }
        
        with patch.object(api_client, '_make_request', return_value=mock_fixtures):
            fixtures = await api_client.get_fixtures(39, 2024)
            
            assert len(fixtures) == 1
            assert isinstance(fixtures[0], Fixture)
            assert fixtures[0].home_team == "Home Team"
            assert fixtures[0].away_team == "Away Team"
    
    @pytest.mark.asyncio
    async def test_cache_hit(self, api_client):
        """Test cache functionality"""
        mock_data = {"response": [{"test": "data"}]}
        
        with patch.object(api_client.cache, 'get', return_value=mock_data):
            result = await api_client._make_request("test")
            
            assert result == mock_data


class TestRateLimiter:
    """Test cases for RateLimiter"""
    
    @pytest.fixture
    def rate_limiter(self):
        """Create rate limiter instance"""
        return RateLimiter(calls_per_hour=10, calls_per_day=100)
    
    @pytest.mark.asyncio
    async def test_check_limit_success(self, rate_limiter):
        """Test successful rate limit check"""
        assert await rate_limiter.check_limit() is True
    
    @pytest.mark.asyncio
    async def test_check_limit_hourly_exceeded(self, rate_limiter):
        """Test hourly limit exceeded"""
        # Fill up the hourly limit
        for _ in range(10):
            await rate_limiter.record_call()
        
        # Next call should fail
        assert await rate_limiter.check_limit() is False
    
    @pytest.mark.asyncio
    async def test_check_limit_burst_control(self, rate_limiter):
        """Test burst control with token bucket"""
        # Rapidly consume tokens
        for _ in range(rate_limiter.burst_size):
            assert await rate_limiter.check_limit() is True
            await rate_limiter.record_call()
        
        # Should be rate limited now
        assert await rate_limiter.check_limit() is False
    
    def test_get_status(self, rate_limiter):
        """Test rate limiter status"""
        status = rate_limiter.get_status()
        
        assert 'hourly_used' in status
        assert 'hourly_remaining' in status
        assert 'daily_used' in status
        assert 'daily_remaining' in status
        assert 'burst_tokens' in status
    
    def test_reset_limits(self, rate_limiter):
        """Test resetting rate limits"""
        # Use some calls
        asyncio.run(rate_limiter.record_call())
        asyncio.run(rate_limiter.record_call())
        
        # Reset
        rate_limiter.reset_limits()
        
        status = rate_limiter.get_status()
        assert status['hourly_used'] == 0
        assert status['daily_used'] == 0


class TestCacheManager:
    """Test cases for CacheManager"""
    
    @pytest.fixture
    async def cache_manager(self, tmp_path):
        """Create cache manager instance"""
        cache = CacheManager()
        cache.cache_dir = tmp_path / "cache"
        cache.cache_dir.mkdir()
        await cache.initialize()
        yield cache
        await cache.close()
    
    def test_build_key(self, cache_manager):
        """Test cache key building"""
        key1 = cache_manager.build_key("endpoint", {"param": "value"})
        key2 = cache_manager.build_key("endpoint", {"param": "value"})
        key3 = cache_manager.build_key("endpoint", {"param": "different"})
        
        assert key1 == key2  # Same params = same key
        assert key1 != key3  # Different params = different key
    
    @pytest.mark.asyncio
    async def test_set_and_get(self, cache_manager):
        """Test setting and getting cache values"""
        key = "test_key"
        value = {"data": "test"}
        
        await cache_manager.set(key, value, ttl=60)
        retrieved = await cache_manager.get(key)
        
        assert retrieved == value
    
    @pytest.mark.asyncio
    async def test_expiration(self, cache_manager):
        """Test cache expiration"""
        key = "test_key"
        value = {"data": "test"}
        
        # Set with 0 TTL (immediate expiration)
        await cache_manager.set(key, value, ttl=0)
        await asyncio.sleep(0.1)
        
        retrieved = await cache_manager.get(key)
        assert retrieved is None
    
    @pytest.mark.asyncio
    async def test_delete(self, cache_manager):
        """Test cache deletion"""
        key = "test_key"
        value = {"data": "test"}
        
        await cache_manager.set(key, value)
        await cache_manager.delete(key)
        
        retrieved = await cache_manager.get(key)
        assert retrieved is None
    
    @pytest.mark.asyncio
    async def test_clear(self, cache_manager):
        """Test clearing all cache"""
        # Add multiple entries
        for i in range(5):
            await cache_manager.set(f"key_{i}", f"value_{i}")
        
        # Clear all
        await cache_manager.clear()
        
        # Check all are gone
        for i in range(5):
            assert await cache_manager.get(f"key_{i}") is None
    
    def test_get_status(self, cache_manager):
        """Test cache status"""
        status = cache_manager.get_status()
        
        assert 'hit_rate' in status
        assert 'total_requests' in status
        assert 'entries' in status
        assert 'size_mb' in status
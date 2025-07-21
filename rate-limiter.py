"""
Advanced rate limiter with sliding window and token bucket algorithms
"""

import asyncio
import time
from typing import Dict, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging

from config import API_CONFIG

logger = logging.getLogger("profootball.ratelimit")


@dataclass
class RateLimitWindow:
    """Sliding window for rate limiting"""
    window_start: float
    requests: list = field(default_factory=list)
    
    def add_request(self, timestamp: float):
        """Add a request timestamp"""
        self.requests.append(timestamp)
        
    def cleanup(self, window_size: int):
        """Remove old requests outside the window"""
        cutoff = time.time() - window_size
        self.requests = [ts for ts in self.requests if ts > cutoff]
        
    def count_requests(self, window_size: int) -> int:
        """Count requests in the current window"""
        self.cleanup(window_size)
        return len(self.requests)


@dataclass
class TokenBucket:
    """Token bucket for rate limiting"""
    capacity: int
    tokens: float
    refill_rate: float
    last_refill: float
    
    def consume(self, tokens: int = 1) -> bool:
        """Try to consume tokens"""
        self.refill()
        
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False
        
    def refill(self):
        """Refill tokens based on time elapsed"""
        now = time.time()
        elapsed = now - self.last_refill
        
        # Add tokens based on refill rate
        tokens_to_add = elapsed * self.refill_rate
        self.tokens = min(self.capacity, self.tokens + tokens_to_add)
        self.last_refill = now
        
    def time_until_tokens(self, tokens: int) -> float:
        """Calculate time until enough tokens are available"""
        self.refill()
        
        if self.tokens >= tokens:
            return 0
            
        tokens_needed = tokens - self.tokens
        return tokens_needed / self.refill_rate


class RateLimiter:
    """Advanced rate limiter with multiple strategies"""
    
    def __init__(
        self,
        calls_per_hour: int = 100,
        calls_per_day: int = 1000,
        burst_size: int = 10
    ):
        self.calls_per_hour = calls_per_hour
        self.calls_per_day = calls_per_day
        self.burst_size = burst_size
        
        # Sliding windows for different time periods
        self.hourly_window = RateLimitWindow(time.time())
        self.daily_window = RateLimitWindow(time.time())
        
        # Token bucket for burst control
        self.token_bucket = TokenBucket(
            capacity=burst_size,
            tokens=burst_size,
            refill_rate=calls_per_hour / 3600,  # tokens per second
            last_refill=time.time()
        )
        
        # Lock for thread safety
        self._lock = asyncio.Lock()
        
        # Statistics
        self.stats = {
            'total_requests': 0,
            'blocked_requests': 0,
            'last_reset': datetime.now()
        }
        
    async def check_limit(self) -> bool:
        """Check if request can proceed"""
        async with self._lock:
            # Check hourly limit
            hourly_count = self.hourly_window.count_requests(3600)
            if hourly_count >= self.calls_per_hour:
                logger.warning(f"Hourly rate limit exceeded: {hourly_count}/{self.calls_per_hour}")
                self.stats['blocked_requests'] += 1
                return False
                
            # Check daily limit
            daily_count = self.daily_window.count_requests(86400)
            if daily_count >= self.calls_per_day:
                logger.warning(f"Daily rate limit exceeded: {daily_count}/{self.calls_per_day}")
                self.stats['blocked_requests'] += 1
                return False
                
            # Check burst limit
            if not self.token_bucket.consume(1):
                logger.warning("Burst limit exceeded")
                self.stats['blocked_requests'] += 1
                return False
                
            return True
            
    async def record_call(self):
        """Record a successful API call"""
        async with self._lock:
            timestamp = time.time()
            self.hourly_window.add_request(timestamp)
            self.daily_window.add_request(timestamp)
            self.stats['total_requests'] += 1
            
    async def wait_if_needed(self) -> float:
        """Wait if rate limit exceeded and return wait time"""
        async with self._lock:
            # Check which limit is exceeded
            hourly_count = self.hourly_window.count_requests(3600)
            daily_count = self.daily_window.count_requests(86400)
            
            wait_times = []
            
            # Calculate wait time for hourly limit
            if hourly_count >= self.calls_per_hour:
                oldest_request = min(self.hourly_window.requests)
                wait_time = oldest_request + 3600 - time.time()
                wait_times.append(wait_time)
                
            # Calculate wait time for daily limit
            if daily_count >= self.calls_per_day:
                oldest_request = min(self.daily_window.requests)
                wait_time = oldest_request + 86400 - time.time()
                wait_times.append(wait_time)
                
            # Calculate wait time for burst limit
            tokens_wait = self.token_bucket.time_until_tokens(1)
            if tokens_wait > 0:
                wait_times.append(tokens_wait)
                
            if wait_times:
                wait_time = max(0, min(wait_times))
                logger.info(f"Rate limited, waiting {wait_time:.1f} seconds")
                await asyncio.sleep(wait_time)
                return wait_time
                
            return 0
            
    def get_status(self) -> Dict[str, any]:
        """Get current rate limit status"""
        hourly_count = self.hourly_window.count_requests(3600)
        daily_count = self.daily_window.count_requests(86400)
        
        # Calculate reset times
        if self.hourly_window.requests:
            hourly_reset = datetime.fromtimestamp(
                min(self.hourly_window.requests) + 3600
            )
        else:
            hourly_reset = datetime.now() + timedelta(hours=1)
            
        if self.daily_window.requests:
            daily_reset = datetime.fromtimestamp(
                min(self.daily_window.requests) + 86400
            )
        else:
            daily_reset = datetime.now() + timedelta(days=1)
            
        return {
            'hourly_used': hourly_count,
            'hourly_limit': self.calls_per_hour,
            'hourly_remaining': max(0, self.calls_per_hour - hourly_count),
            'hourly_reset': hourly_reset.isoformat(),
            'daily_used': daily_count,
            'daily_limit': self.calls_per_day,
            'daily_remaining': max(0, self.calls_per_day - daily_count),
            'daily_reset': daily_reset.isoformat(),
            'burst_tokens': int(self.token_bucket.tokens),
            'burst_capacity': self.burst_size,
            'total_requests': self.stats['total_requests'],
            'blocked_requests': self.stats['blocked_requests'],
            'block_rate': self.stats['blocked_requests'] / max(self.stats['total_requests'], 1)
        }
        
    def reset_limits(self):
        """Reset all rate limits (for testing)"""
        self.hourly_window = RateLimitWindow(time.time())
        self.daily_window = RateLimitWindow(time.time())
        self.token_bucket.tokens = self.token_bucket.capacity
        self.token_bucket.last_refill = time.time()
        self.stats['last_reset'] = datetime.now()
        
    async def configure_adaptive_limits(self, api_response_headers: Dict[str, str]):
        """Adapt limits based on API response headers"""
        # Many APIs return rate limit info in headers
        if 'X-RateLimit-Limit' in api_response_headers:
            try:
                limit = int(api_response_headers['X-RateLimit-Limit'])
                if limit != self.calls_per_hour:
                    logger.info(f"Adapting hourly limit from {self.calls_per_hour} to {limit}")
                    self.calls_per_hour = limit
                    self.token_bucket.refill_rate = limit / 3600
            except ValueError:
                pass
                
        if 'X-RateLimit-Remaining' in api_response_headers:
            try:
                remaining = int(api_response_headers['X-RateLimit-Remaining'])
                # If remaining is very low, slow down
                if remaining < 10:
                    logger.warning(f"Only {remaining} API calls remaining")
                    # Reduce burst size temporarily
                    self.token_bucket.capacity = min(remaining, 5)
            except ValueError:
                pass


class DistributedRateLimiter(RateLimiter):
    """Rate limiter with distributed backend support (Redis)"""
    
    def __init__(self, redis_client=None, key_prefix="ratelimit:", **kwargs):
        super().__init__(**kwargs)
        self.redis = redis_client
        self.key_prefix = key_prefix
        
    async def check_limit(self) -> bool:
        """Check limit using Redis backend if available"""
        if self.redis:
            # Use Redis for distributed rate limiting
            try:
                hourly_key = f"{self.key_prefix}hourly:{int(time.time() // 3600)}"
                daily_key = f"{self.key_prefix}daily:{int(time.time() // 86400)}"
                
                pipe = self.redis.pipeline()
                pipe.incr(hourly_key)
                pipe.expire(hourly_key, 3600)
                pipe.incr(daily_key)
                pipe.expire(daily_key, 86400)
                
                hourly_count, _, daily_count, _ = await pipe.execute()
                
                if hourly_count > self.calls_per_hour or daily_count > self.calls_per_day:
                    self.stats['blocked_requests'] += 1
                    return False
                    
                return True
                
            except Exception as e:
                logger.error(f"Redis error, falling back to local rate limit: {e}")
                # Fall back to local rate limiting
                
        return await super().check_limit()
"""
Advanced caching system with TTL, size limits, and persistence
"""

import asyncio
import json
import pickle
import hashlib
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union
from datetime import datetime, timedelta
import aiofiles
import logging
from dataclasses import dataclass, asdict
import shutil

from config import CACHE_CONFIG, CACHE_DIR

logger = logging.getLogger("profootball.cache")


@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    key: str
    value: Any
    created_at: float
    expires_at: float
    access_count: int = 0
    last_accessed: float = None
    size_bytes: int = 0
    
    def is_expired(self) -> bool:
        """Check if cache entry has expired"""
        return time.time() > self.expires_at
    
    def update_access(self):
        """Update access statistics"""
        self.access_count += 1
        self.last_accessed = time.time()


class CacheManager:
    """Advanced cache manager with async support"""
    
    def __init__(self, max_size_mb: int = None, default_ttl: int = None):
        self.cache_dir = Path(CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_size_bytes = (max_size_mb or CACHE_CONFIG['max_cache_size'] // (1024 * 1024)) * 1024 * 1024
        self.default_ttl = default_ttl or CACHE_CONFIG['default_ttl']
        
        # In-memory cache for fast access
        self.memory_cache: Dict[str, CacheEntry] = {}
        self.cache_stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'total_size': 0
        }
        
        # Lock for thread safety
        self._lock = asyncio.Lock()
        
        # Start cleanup task
        self._cleanup_task = None
        
    async def initialize(self):
        """Initialize cache and load persistent data"""
        await self._load_persistent_cache()
        self._cleanup_task = asyncio.create_task(self._periodic_cleanup())
        
    async def close(self):
        """Close cache manager and cleanup"""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
                
        await self._save_persistent_cache()
        
    def build_key(self, endpoint: str, params: Optional[Dict] = None) -> str:
        """Build cache key from endpoint and parameters"""
        key_data = {
            'endpoint': endpoint,
            'params': params or {}
        }
        
        # Sort params for consistent hashing
        key_str = json.dumps(key_data, sort_keys=True)
        
        # Create hash for compact storage
        return hashlib.sha256(key_str.encode()).hexdigest()
        
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        async with self._lock:
            # Check memory cache first
            if key in self.memory_cache:
                entry = self.memory_cache[key]
                
                if entry.is_expired():
                    # Remove expired entry
                    await self._remove_entry(key)
                    self.cache_stats['misses'] += 1
                    return None
                    
                # Update access stats
                entry.update_access()
                self.cache_stats['hits'] += 1
                
                logger.debug(f"Cache hit for key: {key[:8]}...")
                return entry.value
                
            # Check disk cache
            entry = await self._load_from_disk(key)
            if entry:
                if not entry.is_expired():
                    # Load to memory cache
                    self.memory_cache[key] = entry
                    entry.update_access()
                    self.cache_stats['hits'] += 1
                    return entry.value
                else:
                    # Remove expired file
                    await self._remove_from_disk(key)
                    
            self.cache_stats['misses'] += 1
            return None
            
    async def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set value in cache"""
        async with self._lock:
            # Calculate size
            try:
                serialized = pickle.dumps(value)
                size_bytes = len(serialized)
            except Exception as e:
                logger.error(f"Failed to serialize value: {e}")
                return
                
            # Check size limit
            if await self._ensure_space(size_bytes):
                ttl = ttl or self.default_ttl
                expires_at = time.time() + ttl
                
                entry = CacheEntry(
                    key=key,
                    value=value,
                    created_at=time.time(),
                    expires_at=expires_at,
                    size_bytes=size_bytes
                )
                
                # Store in memory
                self.memory_cache[key] = entry
                self.cache_stats['total_size'] += size_bytes
                
                # Store on disk asynchronously
                asyncio.create_task(self._save_to_disk(key, entry))
                
                logger.debug(f"Cached key: {key[:8]}... (size: {size_bytes} bytes, ttl: {ttl}s)")
                
    async def delete(self, key: str):
        """Delete value from cache"""
        async with self._lock:
            await self._remove_entry(key)
            
    async def clear(self):
        """Clear all cache"""
        async with self._lock:
            self.memory_cache.clear()
            self.cache_stats = {
                'hits': 0,
                'misses': 0,
                'evictions': 0,
                'total_size': 0
            }
            
            # Clear disk cache
            try:
                shutil.rmtree(self.cache_dir)
                self.cache_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                logger.error(f"Failed to clear disk cache: {e}")
                
    async def _ensure_space(self, required_bytes: int) -> bool:
        """Ensure enough space for new entry using LRU eviction"""
        current_size = self.cache_stats['total_size']
        
        if current_size + required_bytes <= self.max_size_bytes:
            return True
            
        # Need to evict entries
        entries = list(self.memory_cache.values())
        # Sort by last accessed time (LRU)
        entries.sort(key=lambda e: e.last_accessed or e.created_at)
        
        freed_space = 0
        entries_to_remove = []
        
        for entry in entries:
            if current_size + required_bytes - freed_space <= self.max_size_bytes:
                break
                
            entries_to_remove.append(entry.key)
            freed_space += entry.size_bytes
            
        # Remove entries
        for key in entries_to_remove:
            await self._remove_entry(key)
            self.cache_stats['evictions'] += 1
            
        return True
        
    async def _remove_entry(self, key: str):
        """Remove entry from cache"""
        if key in self.memory_cache:
            entry = self.memory_cache[key]
            self.cache_stats['total_size'] -= entry.size_bytes
            del self.memory_cache[key]
            
        await self._remove_from_disk(key)
        
    async def _save_to_disk(self, key: str, entry: CacheEntry):
        """Save cache entry to disk"""
        try:
            file_path = self.cache_dir / f"{key}.pkl"
            
            async with aiofiles.open(file_path, 'wb') as f:
                data = pickle.dumps(entry)
                await f.write(data)
                
        except Exception as e:
            logger.error(f"Failed to save cache to disk: {e}")
            
    async def _load_from_disk(self, key: str) -> Optional[CacheEntry]:
        """Load cache entry from disk"""
        try:
            file_path = self.cache_dir / f"{key}.pkl"
            
            if not file_path.exists():
                return None
                
            async with aiofiles.open(file_path, 'rb') as f:
                data = await f.read()
                entry = pickle.loads(data)
                
            return entry
            
        except Exception as e:
            logger.error(f"Failed to load cache from disk: {e}")
            return None
            
    async def _remove_from_disk(self, key: str):
        """Remove cache file from disk"""
        try:
            file_path = self.cache_dir / f"{key}.pkl"
            if file_path.exists():
                file_path.unlink()
        except Exception as e:
            logger.error(f"Failed to remove cache file: {e}")
            
    async def _load_persistent_cache(self):
        """Load cache entries from disk on startup"""
        try:
            cache_files = list(self.cache_dir.glob("*.pkl"))
            
            for file_path in cache_files:
                try:
                    async with aiofiles.open(file_path, 'rb') as f:
                        data = await f.read()
                        entry = pickle.loads(data)
                        
                    if not entry.is_expired():
                        self.memory_cache[entry.key] = entry
                        self.cache_stats['total_size'] += entry.size_bytes
                    else:
                        # Remove expired file
                        file_path.unlink()
                        
                except Exception as e:
                    logger.error(f"Failed to load cache file {file_path}: {e}")
                    
            logger.info(f"Loaded {len(self.memory_cache)} cache entries from disk")
            
        except Exception as e:
            logger.error(f"Failed to load persistent cache: {e}")
            
    async def _save_persistent_cache(self):
        """Save important cache entries to disk"""
        try:
            # Save only non-expired entries
            for key, entry in self.memory_cache.items():
                if not entry.is_expired():
                    await self._save_to_disk(key, entry)
                    
            logger.info(f"Saved {len(self.memory_cache)} cache entries to disk")
            
        except Exception as e:
            logger.error(f"Failed to save persistent cache: {e}")
            
    async def _periodic_cleanup(self):
        """Periodic cleanup of expired entries"""
        while True:
            try:
                await asyncio.sleep(CACHE_CONFIG['cleanup_interval'])
                
                async with self._lock:
                    expired_keys = [
                        key for key, entry in self.memory_cache.items()
                        if entry.is_expired()
                    ]
                    
                    for key in expired_keys:
                        await self._remove_entry(key)
                        
                    if expired_keys:
                        logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")
                        
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cache cleanup: {e}")
                
    def get_status(self) -> Dict[str, Any]:
        """Get cache status and statistics"""
        total_requests = self.cache_stats['hits'] + self.cache_stats['misses']
        hit_rate = self.cache_stats['hits'] / max(total_requests, 1)
        
        return {
            'hit_rate': hit_rate,
            'total_requests': total_requests,
            'hits': self.cache_stats['hits'],
            'misses': self.cache_stats['misses'],
            'evictions': self.cache_stats['evictions'],
            'entries': len(self.memory_cache),
            'size_mb': self.cache_stats['total_size'] / (1024 * 1024),
            'max_size_mb': self.max_size_bytes / (1024 * 1024)
        }
        
    async def get_detailed_stats(self) -> Dict[str, Any]:
        """Get detailed cache statistics"""
        stats = self.get_status()
        
        # Add entry details
        entries_by_age = {}
        entries_by_access = {}
        
        for entry in self.memory_cache.values():
            # Age buckets
            age_hours = (time.time() - entry.created_at) / 3600
            age_bucket = f"{int(age_hours)}-{int(age_hours)+1}h"
            entries_by_age[age_bucket] = entries_by_age.get(age_bucket, 0) + 1
            
            # Access buckets
            access_bucket = f"{entry.access_count} accesses"
            entries_by_access[access_bucket] = entries_by_access.get(access_bucket, 0) + 1
            
        stats['entries_by_age'] = entries_by_age
        stats['entries_by_access'] = entries_by_access
        
        return stats
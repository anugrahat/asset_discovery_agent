"""
Caching layer with SQLite backend for API responses
"""
import sqlite3
import json
import hashlib
import time
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime, timedelta

class APICache:
    """SQLite-based cache for API responses with TTL support"""
    
    def __init__(self, cache_dir: Path = Path("cache")):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(exist_ok=True)
        self.db_path = self.cache_dir / "api_cache.db"
        self._init_db()
    
    def _init_db(self):
        """Initialize SQLite database with cache table"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS api_cache (
                    key TEXT PRIMARY KEY,
                    data TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP NOT NULL
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_expires ON api_cache(expires_at)")
    
    def _make_key(self, url: str, params: Dict = None) -> str:
        """Generate cache key from URL and parameters"""
        cache_input = f"{url}:{json.dumps(params or {}, sort_keys=True)}"
        return hashlib.md5(cache_input.encode()).hexdigest()
    
    def get(self, url: str, params: Dict = None) -> Optional[Dict[Any, Any]]:
        """Get cached response if not expired"""
        key = self._make_key(url, params)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT data FROM api_cache WHERE key = ? AND expires_at > datetime('now')",
                (key,)
            )
            result = cursor.fetchone()
            
            if result:
                return json.loads(result[0])
        
        return None
    
    def set(self, url: str, data: Dict[Any, Any], params: Dict = None, ttl_hours: int = 24):
        """Cache API response with TTL"""
        key = self._make_key(url, params)
        expires_at = datetime.now() + timedelta(hours=ttl_hours)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO api_cache (key, data, expires_at) VALUES (?, ?, ?)",
                (key, json.dumps(data), expires_at)
            )
    
    def cleanup_expired(self):
        """Remove expired cache entries"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM api_cache WHERE expires_at <= datetime('now')")
    
    def clear(self):
        """Clear all cache entries"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM api_cache")
    
    def stats(self) -> Dict[str, int]:
        """Get cache statistics"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM api_cache")
            total = cursor.fetchone()[0]
            
            cursor = conn.execute("SELECT COUNT(*) FROM api_cache WHERE expires_at <= datetime('now')")
            expired = cursor.fetchone()[0]
            
            return {"total_entries": total, "expired_entries": expired, "active_entries": total - expired}

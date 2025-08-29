"""
Async HTTP client with rate limiting and retry logic
"""
import asyncio
import time
from typing import Optional, Dict, Any, List
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from .cache import APICache
from .dns_resolver import DNSResolver

class RateLimitedClient:
    """HTTP client with per-domain rate limiting"""
    
    def __init__(self, cache: Optional[APICache] = None):
        self.cache = cache or APICache()
        self.rate_limits = {
            'eutils.ncbi.nlm.nih.gov': {'requests_per_second': 10, 'last_request': 0},
            'www.ebi.ac.uk': {'requests_per_second': 2, 'last_request': 0},  # Conservative for ChEMBL
            'search.rcsb.org': {'requests_per_second': 5, 'last_request': 0}
        }
        # Configure httpx with custom transport for DNS fallback
        self.client = httpx.AsyncClient(
            timeout=30.0,
            follow_redirects=True,
            verify=False  # Disable SSL verification for problematic domains
        )
    
    async def _enforce_rate_limit(self, domain: str):
        """Enforce rate limiting for domain"""
        if domain in self.rate_limits:
            limit_info = self.rate_limits[domain]
            min_interval = 1.0 / limit_info['requests_per_second']
            time_since_last = time.time() - limit_info['last_request']
            
            if time_since_last < min_interval:
                await asyncio.sleep(min_interval - time_since_last)
            
            limit_info['last_request'] = time.time()
    
    async def get(self, url: str, params: Optional[Dict] = None, headers: Optional[Dict] = None, 
                  cache_ttl_hours: int = 24, timeout: int = 30) -> Dict[Any, Any]:
        """GET request with caching and rate limiting"""
        
        # Check cache first
        cached_response = self.cache.get(url, params)
        if cached_response:
            return cached_response
        
        # Extract domain for rate limiting
        domain = httpx.URL(url).host
        await self._enforce_rate_limit(domain)
        
        # Try making request with retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Make request
                response = await self.client.get(url, params=params, headers=headers)
                response.raise_for_status()
                
                # Handle different response types
                if response.headers.get('content-type', '').startswith('application/json'):
                    data = response.json()
                else:
                    data = {"text": response.text, "status_code": response.status_code}
                
                # Cache successful response
                self.cache.set(url, data, params, cache_ttl_hours)
                
                return data
                
            except (httpx.ConnectError, httpx.TimeoutException) as e:
                if attempt < max_retries - 1:
                    # Exponential backoff
                    await asyncio.sleep(2 ** attempt)
                else:
                    # Log the error but return empty dict instead of raising
                    print(f"HTTP GET failed after {max_retries} attempts: {url} - {str(e)}")
                    return {}
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.ConnectError))
    )
    async def post(self, url: str, json_data: Optional[Dict] = None, 
                   cache_ttl_hours: int = 24) -> Dict[Any, Any]:
        """POST request with caching and rate limiting"""
        
        # Check cache (using POST data as part of cache key)
        cache_params = {"method": "POST", "json": json_data}
        cached_response = self.cache.get(url, cache_params)
        if cached_response:
            return cached_response
        
        # Extract domain for rate limiting
        domain = httpx.URL(url).host
        await self._enforce_rate_limit(domain)
        
        # Make request
        response = await self.client.post(url, json=json_data)
        response.raise_for_status()
        
        data = response.json()
        
        # Cache successful response
        self.cache.set(url, data, cache_params, cache_ttl_hours)
        
        return data
    
    async def close(self):
        """Close the HTTP client"""
        await self.client.aclose()
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

# Singleton instance
_client_instance = None

async def get_http_client() -> RateLimitedClient:
    """Get singleton HTTP client instance"""
    global _client_instance
    if _client_instance is None:
        _client_instance = RateLimitedClient()
    return _client_instance

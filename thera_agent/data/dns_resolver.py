"""DNS resolver with fallback mechanisms"""
import socket
from typing import Optional, Dict
import asyncio

class DNSResolver:
    """Handle DNS resolution with fallbacks"""
    
    # Known good IPs for services (as fallback)
    FALLBACK_IPS = {
        'eutils.ncbi.nlm.nih.gov': '130.14.29.110',  # NCBI's IP
        'html.duckduckgo.com': None,  # Skip DuckDuckGo for now
    }
    
    # Alternative URLs for services
    ALTERNATIVE_URLS = {
        'pubmed': 'https://api.ncbi.nlm.nih.gov/lit/ctxp/v1/pubmed/',  # Alternative PubMed API
        'duckduckgo': None,  # No alternative for now
    }
    
    @classmethod
    def resolve_host(cls, hostname: str) -> Optional[str]:
        """Resolve hostname to IP with fallback"""
        try:
            return socket.gethostbyname(hostname)
        except socket.gaierror:
            # Try fallback IP if available
            return cls.FALLBACK_IPS.get(hostname)
    
    @classmethod
    def get_alternative_url(cls, service: str) -> Optional[str]:
        """Get alternative URL for a service"""
        return cls.ALTERNATIVE_URLS.get(service)
    
    @classmethod
    async def test_connectivity(cls, url: str) -> bool:
        """Test if URL is accessible"""
        import httpx
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.head(url)
                return response.status_code < 500
        except:
            return False

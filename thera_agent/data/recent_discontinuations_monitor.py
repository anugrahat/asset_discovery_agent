"""
Monitor for recent drug discontinuations from FDA press releases and company announcements
"""
import asyncio
import logging
import re
import ssl
import json
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import aiohttp
from bs4 import BeautifulSoup
import random

logger = logging.getLogger(__name__)

class RecentDiscontinuationsMonitor:
    """Monitor recent drug discontinuations from multiple sources"""
    
    def __init__(self, session: Optional[aiohttp.ClientSession] = None):
        self._external_session = session
        self._session = None
        
        # Enhanced SSL and timeout settings
        self.timeout = aiohttp.ClientTimeout(total=20, connect=10)
        ssl_ctx = ssl.create_default_context()
        self.connector = aiohttp.TCPConnector(
            ssl=ssl_ctx,
            limit=10,
            ttl_dns_cache=300,
            use_dns_cache=True
        )
        
        # Browser-like headers to avoid bot detection
        self.headers = {
            "User-Agent": "Mozilla/5.0 (compatible; thera-agent/1.0)"
        }
        
        # Retry configuration
        self.max_retries = 3
        self.base_delay = 1.0
        self.max_delay = 30.0
        
        # Keyword pattern for discontinuation detection
        self.keywords_pattern = re.compile(
            r"\b(discontinu(at|ed|ion)|withdraw(n|al)|retir(ed|ement)|suspend|halt|scrap|axe)\b", 
            re.IGNORECASE
        )
        
        # Updated reliable data sources
        self.data_sources = {
            'fda_enforcement': 'https://api.fda.gov/drug/enforcement.json',
            'endpoints_rss': 'https://endpts.com/feed',
            'fiercepharma_rss': 'https://www.fiercepharma.com/rss/xml'
        }
        
        # Fallback: Use ClinicalTrials.gov API for terminated studies
        self.fallback_sources = {
            'clinicaltrials_terminated': 'https://clinicaltrials.gov/api/v2/studies',
            'pubmed_api': 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi'
        }
        
        # Company investor relations
        self.company_sources = {
            'merck': 'https://investors.merck.com/news-and-events/news-releases',
            'pfizer': 'https://investors.pfizer.com/news-and-events/press-releases',
            'jnj': 'https://www.investor.jnj.com/news-and-events/press-releases',
            'takeda': 'https://www.takeda.com/newsroom/newsreleases',
            'beigene': 'https://ir.beigene.com/news-releases'
        }
    
    async def __aenter__(self):
        if self._external_session:
            self._session = self._external_session
        else:
            self._session = aiohttp.ClientSession(
                timeout=self.timeout,
                connector=self.connector,
                headers=self.headers
            )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if not self._external_session and self._session:
            await self._session.close()
    
    async def get_recent_discontinuations(self, days_back: int = 365) -> List[Dict]:
        """Get recent drug discontinuations from the last N days"""
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        discontinuations = []
        try:
            # 1. FDA Enforcement API (drug recalls/withdrawals)
            fda_data = await self._fetch_fda_enforcement(days_back)
            discontinuations.extend(fda_data)
            
            # 2. RSS feeds from reliable sources
            rss_data = await self._fetch_rss_feeds(days_back)
            discontinuations.extend(rss_data)
            
        except Exception as e:
            logger.warning(f"Primary sources failed: {e}")
            
            # Fallback to ClinicalTrials.gov API
            try:
                fallback_results = await self._search_fallback_sources(cutoff_date)
                discontinuations.extend(fallback_results)
            except Exception as e:
                logger.warning(f"All network sources failed: {e}")
                pass
        
        # Deduplicate and sort by date
        discontinuations = self._deduplicate_discontinuations(discontinuations)
        discontinuations.sort(key=lambda x: x.get('date', ''), reverse=True)
        
        return discontinuations
    
    # Backward compatibility shim for old callers
    async def discover_recent_discontinuations(self, cutoff_date: datetime) -> List[Dict]:
        """Compatibility wrapper for old method name"""
        days_back = (datetime.now() - cutoff_date).days
        return await self.get_recent_discontinuations(days_back=days_back)
    
    async def _retry_request(self, url: str, **kwargs) -> Optional[aiohttp.ClientResponse]:
        """Make HTTP request with exponential backoff retry logic"""
        for attempt in range(self.max_retries):
            try:
                async with self._session.get(url, **kwargs) as response:
                    if response.status == 200:
                        return response
                    elif response.status in [429, 503, 504]:  # Rate limited or server error
                        if attempt < self.max_retries - 1:
                            delay = min(self.base_delay * (2 ** attempt) + random.uniform(0, 1), self.max_delay)
                            logger.warning(f"HTTP {response.status} for {url}, retrying in {delay:.1f}s (attempt {attempt + 1}/{self.max_retries})")
                            await asyncio.sleep(delay)
                            continue
                    else:
                        logger.warning(f"HTTP {response.status} for {url}")
                        return None
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                if attempt < self.max_retries - 1:
                    delay = min(self.base_delay * (2 ** attempt) + random.uniform(0, 1), self.max_delay)
                    logger.warning(f"Request failed for {url}: {e}, retrying in {delay:.1f}s (attempt {attempt + 1}/{self.max_retries})")
                    await asyncio.sleep(delay)
                else:
                    logger.warning(f"Request failed for {url} after {self.max_retries} attempts: {e}")
                    return None
        return None
    
    async def _fetch_fda_enforcement(self, days_back: int) -> List[Dict]:
        """Fetch discontinuations from FDA Enforcement API (recalls/withdrawals)"""
        discontinuations = []
        
        try:
            params = {
                'limit': 100,
                'search': 'status:"Ongoing" OR status:"Completed"'
            }
            
            async with self._session.get(self.data_sources['fda_enforcement'], params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    cutoff_date = datetime.now() - timedelta(days=days_back)
                    
                    results = data.get('results', [])
                    for record in results:
                        # Check recall date
                        recall_date = record.get('recall_initiation_date')
                        if recall_date:
                            try:
                                parsed_date = datetime.strptime(recall_date, "%Y%m%d")
                                if parsed_date < cutoff_date:
                                    continue
                            except:
                                pass
                        
                        # Extract product info
                        product_desc = record.get('product_description', '')
                        reason = record.get('reason_for_recall', '')
                        classification = record.get('classification', '')
                        
                        # Check if it's a drug discontinuation/withdrawal
                        text_blob = f"{product_desc} {reason}".lower()
                        if ('drug' in text_blob or 'tablet' in text_blob or 'capsule' in text_blob):
                            discontinuations.append({
                                "drug_name": product_desc[:100],
                                "reason": f"FDA Recall - {reason}",
                                "date": recall_date,
                                "source": "FDA Enforcement API",
                                "classification": classification,
                                "url": "https://api.fda.gov/drug/enforcement.json"
                            })
                            
        except Exception as e:
            logger.warning(f"FDA Enforcement API failed: {e}")
        
        return discontinuations
    
    async def _fetch_rss_feeds(self, days_back: int) -> List[Dict]:
        """Fetch discontinuations from RSS feeds"""
        discontinuations = []
        
        feeds = [
            ("Endpoints News", self.data_sources['endpoints_rss']),
            ("FiercePharma", self.data_sources['fiercepharma_rss']),
            ("FDA MedWatch", self.data_sources['fda_medwatch_rss'])
        ]
        
        for source_name, feed_url in feeds:
            try:
                items = await self._parse_rss_feed(feed_url)
                for item in items:
                    if (self._within_days(item.get("pubDate", ""), days_back) and 
                        self.keywords_pattern.search(item.get("title", ""))):
                        
                        discontinuations.append({
                            "drug_name": self._extract_drug_from_title(item["title"]),
                            "reason": "headline_match",
                            "date": item.get("pubDate"),
                            "source": source_name,
                            "url": item.get("link"),
                            "title": item.get("title")
                        })
            except Exception as e:
                logger.warning(f"{source_name} RSS feed failed: {e}")
                continue
        
        return discontinuations
    
    async def _parse_rss_feed(self, feed_url: str) -> List[Dict]:
        """Parse RSS feed and extract items"""
        try:
            async with self._session.get(feed_url) as response:
                if response.status == 200:
                    xml_content = await response.text()
                    root = ET.fromstring(xml_content)
                    
                    items = []
                    for item in root.findall(".//item"):
                        def get_text(tag):
                            el = item.find(tag)
                            return el.text.strip() if el is not None and el.text else ""
                        
                        items.append({
                            "title": get_text("title"),
                            "link": get_text("link"),
                            "pubDate": get_text("pubDate"),
                            "source": feed_url
                        })
                    return items
        except Exception as e:
            logger.warning(f"RSS parsing failed for {feed_url}: {e}")
        
        return []
    
    def _within_days(self, pubdate: str, days: int) -> bool:
        """Check if publication date is within specified days"""
        if not pubdate:
            return True  # If unknown, keep the item
        
        # Try several date formats commonly seen in RSS
        formats = [
            "%a, %d %b %Y %H:%M:%S %z",
            "%Y-%m-%dT%H:%M:%S%z", 
            "%a, %d %b %Y %H:%M:%S %Z",
            "%Y-%m-%d %H:%M:%S"
        ]
        
        for fmt in formats:
            try:
                d = datetime.strptime(pubdate, fmt)
                cutoff = datetime.now(d.tzinfo if d.tzinfo else None) - timedelta(days=days)
                return d >= cutoff
            except Exception:
                continue
        
        return True  # If parsing fails, keep the item
    
    def _extract_drug_from_title(self, title: str) -> str:
        """Extract drug name from news title - simplified version"""
        if not title:
            return "Unknown"
        
        # Simple extraction - look for capitalized words that might be drug names
        words = title.split()
        drug_candidates = []
        
        for word in words:
            # Look for capitalized words that aren't common words
            if (word[0].isupper() and len(word) > 3 and 
                word.lower() not in ['the', 'and', 'for', 'with', 'from', 'this', 'that']):
                drug_candidates.append(word)
        
        return " ".join(drug_candidates[:2]) if drug_candidates else title[:50]
    
    async def _search_fda_sources(self, cutoff_date: datetime) -> List[Dict]:
        """Search FDA press releases for discontinuations"""
        discontinuations = []
        
        try:
            # Search FDA press releases
            url = self.fda_sources['press_releases']
            response = await self._retry_request(url)
            if response:
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                
                # Look for recent press releases about withdrawals
                articles = soup.find_all('article', class_='node')
                for article in articles[:20]:  # Check last 20 articles
                    title_elem = article.find('h2')
                    if title_elem:
                        title = title_elem.get_text().strip()
                        
                        # Look for withdrawal/discontinuation keywords
                        if any(keyword in title.lower() for keyword in [
                            'withdraw', 'discontinu', 'suspend', 'recall', 'remove'
                        ]):
                            # Extract drug name and details
                            drug_info = self._extract_drug_from_fda_title(title)
                            if drug_info:
                                discontinuations.append({
                                    'drug_name': drug_info['drug'],
                                    'reason': 'fda_withdrawal',
                                    'source': 'FDA Press Release',
                                    'title': title,
                                    'url': url,
                                    'date': datetime.now().strftime('%Y-%m-%d')
                                })
        
        except Exception as e:
            logger.warning(f"Failed to search FDA sources: {e}")
        
        return discontinuations
    
    async def _search_news_sources(self, cutoff_date: datetime) -> List[Dict]:
        """Search pharma news sources for discontinuations"""
        discontinuations = []
        
        # Search BioPharma Dive for recent discontinuations
        try:
            biopharma_results = await self._search_biopharmadive(cutoff_date)
            discontinuations.extend(biopharma_results)
        except Exception as e:
            logger.warning(f"BioPharma Dive search failed: {e}")
        
        # Search FiercePharma for recent discontinuations  
        try:
            fierce_results = await self._search_fiercepharma(cutoff_date)
            discontinuations.extend(fierce_results)
        except Exception as e:
            logger.warning(f"FiercePharma search failed: {e}")
        
        # Search Endpoints News for recent discontinuations
        try:
            endpoints_results = await self._search_endpoints(cutoff_date)
            discontinuations.extend(endpoints_results)
        except Exception as e:
            logger.warning(f"Endpoints News search failed: {e}")
        
        return discontinuations
    
    async def _search_biopharmadive(self, cutoff_date: datetime) -> List[Dict]:
        """Search BioPharma Dive for discontinuation news"""
        discontinuations = []
        
        try:
            # Search BioPharma Dive articles
            search_url = f"{self.news_sources['biopharmadive']}/search"
            params = {
                'q': 'discontinue OR withdraw OR suspend OR terminate',
                'sort': 'date'
            }
            
            response = await self._retry_request(search_url, params=params)
            if response:
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                
                # Parse articles for drug discontinuations
                articles = soup.find_all('article', limit=20)
                for article in articles:
                    title_elem = article.find(['h1', 'h2', 'h3'])
                    if title_elem:
                        title = title_elem.get_text().strip()
                        
                        # Look for discontinuation keywords and drug names
                        if any(keyword in title.lower() for keyword in [
                            'discontinue', 'withdraw', 'suspend', 'halt', 'terminate'
                        ]):
                            drug_info = self._extract_drug_from_title(title)
                            if drug_info:
                                discontinuations.append({
                                    'drug_name': drug_info['drug'],
                                    'reason': 'news_reported_discontinuation',
                                    'source': 'BioPharma Dive',
                                    'title': title,
                                    'url': self.news_sources['biopharmadive'],
                                    'date': datetime.now().strftime('%Y-%m-%d')
                                })
        
        except Exception as e:
            logger.warning(f"BioPharma Dive search failed: {e}")
        
        return discontinuations
    
    async def _search_fiercepharma(self, cutoff_date: datetime) -> List[Dict]:
        """Search FiercePharma for discontinuation news"""
        discontinuations = []
        
        try:
            # Search FiercePharma articles
            search_url = f"{self.news_sources['fiercepharma']}/search"
            params = {
                'query': 'discontinue withdraw suspend',
                'sort': 'date'
            }
            
            response = await self._retry_request(search_url, params=params)
            if response:
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                
                # Parse articles for drug discontinuations
                articles = soup.find_all(['article', 'div'], class_=['article', 'story'], limit=20)
                for article in articles:
                    title_elem = article.find(['h1', 'h2', 'h3', 'a'])
                    if title_elem:
                        title = title_elem.get_text().strip()
                        
                        # Look for discontinuation keywords and drug names
                        if any(keyword in title.lower() for keyword in [
                            'discontinue', 'withdraw', 'suspend', 'halt', 'scrap'
                        ]):
                            drug_info = self._extract_drug_from_title(title)
                            if drug_info:
                                discontinuations.append({
                                    'drug_name': drug_info['drug'],
                                    'reason': 'news_reported_discontinuation',
                                    'source': 'FiercePharma',
                                    'title': title,
                                    'url': self.news_sources['fiercepharma'],
                                    'date': datetime.now().strftime('%Y-%m-%d')
                                })
        
        except Exception as e:
            logger.warning(f"FiercePharma search failed: {e}")
        
        return discontinuations
    
    async def _search_endpoints(self, cutoff_date: datetime) -> List[Dict]:
        """Search Endpoints News for discontinuation news"""
        discontinuations = []
        
        try:
            # Search Endpoints News articles
            search_url = f"{self.news_sources['endpoints']}/search"
            params = {
                'q': 'discontinue OR withdraw OR suspend',
                'sort': 'date'
            }
            
            response = await self._retry_request(search_url, params=params)
            if response:
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                
                # Parse articles for drug discontinuations
                articles = soup.find_all('article', limit=20)
                for article in articles:
                    title_elem = article.find(['h1', 'h2', 'h3'])
                    if title_elem:
                        title = title_elem.get_text().strip()
                        
                        # Look for discontinuation keywords and drug names
                        if any(keyword in title.lower() for keyword in [
                            'discontinue', 'withdraw', 'suspend', 'halt', 'axe'
                        ]):
                            drug_info = self._extract_drug_from_title(title)
                            if drug_info:
                                discontinuations.append({
                                    'drug_name': drug_info['drug'],
                                    'reason': 'news_reported_discontinuation',
                                    'source': 'Endpoints News',
                                    'title': title,
                                    'url': self.news_sources['endpoints'],
                                    'date': datetime.now().strftime('%Y-%m-%d')
                                })
        
        except Exception as e:
            logger.warning(f"Endpoints News search failed: {e}")
        
        return discontinuations
    
    async def _search_company_sources(self, cutoff_date: datetime) -> List[Dict]:
        """Search company investor relations for discontinuation announcements"""
        discontinuations = []
        
        # This would implement company-specific searches
        # For now, return empty as it requires company-specific parsing
        
        return discontinuations
    
    async def _search_fallback_sources(self, cutoff_date: datetime) -> List[Dict]:
        """Fallback: Search ClinicalTrials.gov API for terminated studies"""
        discontinuations = []
        
        try:
            # Search for recently terminated trials
            url = self.fallback_sources['clinicaltrials_terminated']
            params = {
                'filter.overallStatus': 'TERMINATED,WITHDRAWN,SUSPENDED',
                'filter.lastUpdatePostDate': cutoff_date.strftime('%Y-%m-%d'),
                'fields': 'NCTId,BriefTitle,OverallStatus,WhyStopped,InterventionName,Condition',
                'format': 'json',
                'countTotal': 'true',
                'pageSize': 50
            }
            
            async with self._session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    studies = data.get('studies', [])
                    
                    for study in studies:
                        protocol_section = study.get('protocolSection', {})
                        identification = protocol_section.get('identificationModule', {})
                        status = protocol_section.get('statusModule', {})
                        interventions = protocol_section.get('armsInterventionsModule', {}).get('interventions', [])
                        
                        # Extract drug names from interventions
                        for intervention in interventions:
                            if intervention.get('type') == 'Drug':
                                drug_name = intervention.get('name', '')
                                if drug_name:
                                    discontinuations.append({
                                        'drug_name': drug_name,
                                        'reason': 'clinical_trial_termination',
                                        'source': 'ClinicalTrials.gov API',
                                        'indication': identification.get('briefTitle', ''),
                                        'nct_id': identification.get('nctId', ''),
                                        'why_stopped': status.get('whyStopped', ''),
                                        'date': datetime.now().strftime('%Y-%m-%d')
                                    })
        
        except Exception as e:
            logger.warning(f"ClinicalTrials.gov fallback search failed: {e}")
        
        return discontinuations
    
    def _extract_drug_from_title(self, title: str) -> Optional[Dict]:
        """Extract drug name from news article title"""
        # Look for drug names in parentheses or capitalized words
        drug_patterns = [
            r'\b([A-Z][a-z]+(?:mab|nib|tinib|lizumab|zumab))\b',  # mAb/TKI patterns
            r'\(([^)]+)\)',  # Text in parentheses
            r'\b([A-Z]{2,}[a-z]*)\b'  # Capitalized drug names
        ]
        
        for pattern in drug_patterns:
            matches = re.findall(pattern, title)
            if matches:
                return {'drug': matches[0]}
        
        return None
    
    def _extract_drug_from_fda_title(self, title: str) -> Optional[Dict]:
        """Extract drug name from FDA press release title"""
        return self._extract_drug_from_title(title)
    
    def _deduplicate_discontinuations(self, discontinuations: List[Dict]) -> List[Dict]:
        """Remove duplicate discontinuations"""
        seen = set()
        unique = []
        
        for disc in discontinuations:
            drug_name = disc.get('drug_name', '').lower()
            if drug_name not in seen:
                seen.add(drug_name)
                unique.append(disc)
        
        return unique

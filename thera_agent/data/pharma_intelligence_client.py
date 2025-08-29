"""
Pharmaceutical Asset Intelligence Client
Tracks commercial availability, ownership, and licensing status of drug compounds
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
import json
from datetime import datetime
import httpx

logger = logging.getLogger(__name__)

class PharmaIntelligenceClient:
    """Client for tracking pharmaceutical asset availability and commercial status"""
    
    def __init__(self, http=None, cache=None, patent_crawler=None):
        self.http = http  # Shared RateLimitedClient
        self.cache = cache  # Shared APICache
        self.patent_crawler = patent_crawler  # Optional shared PatentCrawler
        
        # Data sources for asset intelligence
        self.data_sources = {
            "sec_filings": "https://www.sec.gov/edgar/searchedgar",
            "patent_db": "https://patents.google.com",
            "clinicaltrials": "https://clinicaltrials.gov",
            "pharma_intelligence": "https://www.evaluate.com/api",  # Requires subscription
            "press_releases": "https://www.biospace.com/api",
            "pipeline_tracker": "https://www.nature.com/api"  # Various pharma intelligence APIs
        }
    
    async def get_asset_availability_status(self, drug_name: str, chembl_id: str = None) -> Dict[str, Any]:
        """Get comprehensive asset availability and commercial status"""
        
        asset_status = {
            "drug_name": drug_name,
            "chembl_id": chembl_id,
            "availability_status": await self._check_availability_status(drug_name),
            "ownership_history": await self._get_ownership_history(drug_name),
            "patent_status": await self._get_patent_status(drug_name, chembl_id),
            "licensing_opportunities": await self._get_licensing_opportunities(drug_name),
            "recent_transactions": await self._get_recent_transactions(drug_name),
            "regulatory_status": await self._get_regulatory_status(drug_name),
            "last_updated": datetime.now().isoformat()
        }
        
        # Add real clinical trials data
        asset_status["clinical_trials_status"] = await self._check_clinical_trials_sponsors_real(drug_name)
        
        return asset_status
    
    async def _check_availability_status(self, drug_name: str) -> Dict[str, Any]:
        """Check if asset is available for licensing/acquisition"""
        
        # This would integrate with:
        # 1. SEC Edgar filings for asset sales/acquisitions
        # 2. Company press releases
        # 3. Pharmaceutical intelligence databases
        
        # For now, provide framework for real implementation
        status_indicators = [
            "Pipeline status (active/discontinued/sold)",
            "Recent M&A activity mentions",
            "Patent expiration dates",
            "Orphan drug designations",
            "Regulatory pathway changes"
        ]
        
        return {
            "availability": "unknown",  # available/sold/licensed/under_development
            "confidence": 0.0,
            "last_pipeline_mention": None,
            "indicators_checked": status_indicators,
            "data_sources": ["SEC filings", "Press releases", "Pipeline databases"]
        }
    
    async def _get_ownership_history(self, drug_name: str) -> List[Dict[str, Any]]:
        """Track ownership changes and transfers"""
        
        # Would search for:
        # - Original developer
        # - Asset sales/acquisitions
        # - Licensing agreements
        # - Spin-offs and divestitures
        
        return [
            {
                "date": "2020-01-01",
                "event_type": "original_development",
                "company": "Unknown",
                "details": "Would be extracted from clinical trial registrations and press releases"
            }
        ]
    
    async def _get_patent_status(self, drug_name: str, chembl_id: str = None) -> Dict[str, Any]:
        """Get patent landscape and expiration dates using PatentCrawler if available"""
        try:
            patents = []
            if getattr(self, "patent_crawler", None):
                async with self.patent_crawler as pc:
                    patents = await pc.search_patent_assets(
                        query=drug_name,
                        limit=10
                    )
            
            primary = [
                {
                    "patent_number": p.get("patent_number"),
                    "publication_date": p.get("publication_date"),
                    "status": p.get("status", "Unknown"),
                    "applicant": p.get("applicant")
                } for p in (patents or [])
            ]
            
            return {
                "primary_patents": primary,
                "patent_families": [],
                "expiration_dates": {},
                "freedom_to_operate": "unknown",
                "patent_cliff_risk": "unknown",
                "has_patents": len(primary) > 0
            }
        except Exception as e:
            logger.warning(f"Patent status lookup failed for {drug_name}: {e}")
            return {
                "primary_patents": [],
                "patent_families": [],
                "expiration_dates": {},
                "freedom_to_operate": "unknown",
                "patent_cliff_risk": "unknown"
            }
    
    async def _get_licensing_opportunities(self, drug_name: str) -> Dict[str, Any]:
        """Identify potential licensing opportunities"""
        
        # Would check:
        # - Company licensing pages
        # - Technology transfer offices
        # - Pharma partnership announcements
        # - Academic institution portfolios
        
        return {
            "licensing_available": "unknown",
            "contact_information": {},
            "licensing_terms": {},
            "exclusivity_status": "unknown"
        }
    
    async def _get_recent_transactions(self, drug_name: str) -> List[Dict[str, Any]]:
        """Find recent M&A or licensing transactions"""
        
        # Would search:
        # - BioPharma Dive transactions
        # - PharmaProjects database
        # - SEC filing mentions
        # - Press release databases
        
        return []
    
    async def _get_regulatory_status(self, drug_name: str) -> Dict[str, Any]:
        """Get current regulatory pathway status"""
        
        # Would check:
        # - FDA drug development databases
        # - EMA pipeline
        # - Orphan drug designations
        # - Fast track/breakthrough designations
        
        return {
            "fda_status": "unknown",
            "ema_status": "unknown",
            "orphan_designations": [],
            "special_designations": []
        }
    
    async def batch_asset_screening(self, drug_candidates: List[Dict]) -> Dict[str, Dict]:
        """Screen multiple drug candidates for availability"""
        
        results = {}
        
        for candidate in drug_candidates:
            drug_name = candidate.get("drug", "")
            chembl_id = candidate.get("chembl_id")
            
            if drug_name:
                try:
                    asset_status = await self.get_asset_availability_status(drug_name, chembl_id)
                    results[drug_name] = asset_status
                    
                    # Add brief pause to respect rate limits
                    await asyncio.sleep(0.1)
                    
                except Exception as e:
                    logger.error(f"Asset screening failed for {drug_name}: {e}")
                    results[drug_name] = {"error": str(e)}
        
        return results
    
    async def _check_clinical_trials_sponsors_real(self, drug_name: str) -> Dict[str, Any]:
        """Check ClinicalTrials.gov for recent activity and sponsors - REAL API"""
        
        try:
            url = "https://clinicaltrials.gov/api/v2/studies"
            params = {
                "query.intr": drug_name,
                "filter.overallStatus": "RECRUITING|ACTIVE_NOT_RECRUITING|COMPLETED",
                "pageSize": 10,
                "format": "json"
            }
            
            data = None
            if self.http:
                # Use shared RateLimitedClient with caching
                data = await self.http.get(url, params=params)
            else:
                # Fallback to direct httpx call
                async with httpx.AsyncClient(timeout=10) as client:
                    response = await client.get(url, params=params)
                    if response.status_code == 200:
                        data = response.json()
            
            if data:
                studies = data.get("studies", [])
                
                # Analyze study sponsors and status
                sponsors = []
                latest_date = None
                
                for study in studies:
                    sponsor_info = study.get("protocolSection", {}).get("sponsorCollaboratorsModule", {})
                    lead_sponsor = sponsor_info.get("leadSponsor", {}).get("name", "Unknown")
                    if lead_sponsor != "Unknown":
                        sponsors.append(lead_sponsor)
                    
                    # Get study dates
                    dates_module = study.get("protocolSection", {}).get("statusModule", {})
                    start_date = dates_module.get("startDateStruct", {}).get("date")
                    if start_date and (not latest_date or start_date > latest_date):
                        latest_date = start_date
                
                return {
                    "active_studies": len(studies),
                    "recent_sponsors": list(set(sponsors)),
                    "latest_activity": latest_date,
                    "status": "active" if studies else "inactive"
                }
            
        except Exception as e:
            logger.warning(f"ClinicalTrials.gov lookup failed for {drug_name}: {e}")
        
        return {
            "active_studies": 0,
            "recent_sponsors": [],
            "latest_activity": None,
            "status": "unknown"
        }

# Example usage for real implementation
"""
# Integration points for real data:

1. SEC Edgar API for M&A filings:
   - Search for drug name mentions in 8-K, 10-K filings
   - Track asset acquisition announcements

2. Google Patents API:
   - Patent family searches
   - Expiration date calculations
   - Freedom to operate analysis

3. ClinicalTrials.gov sponsor tracking:
   - Monitor sponsor changes over time
   - Identify asset transfers

4. Pharma intelligence databases:
   - Evaluate Pharma API (subscription)
   - Cortellis API (subscription)
   - BioCentury Intelligence Database

5. Press release monitoring:
   - Business Wire API
   - PR Newswire API
   - Company investor relations feeds

6. Patent databases:
   - USPTO API
   - EPO Open Patent Services
   - WIPO Global Brand Database
"""

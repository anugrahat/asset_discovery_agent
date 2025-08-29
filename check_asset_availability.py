#!/usr/bin/env python3
"""
Practical Asset Availability Checker
Uses publicly available sources to determine if pharmaceutical assets are available for licensing
"""

import asyncio
import json
import sys
from datetime import datetime
import httpx
from typing import Dict, List, Any

class AssetAvailabilityChecker:
    """Check pharmaceutical asset availability using public sources"""
    
    def __init__(self):
        self.timeout = 10
        
    async def check_asset(self, drug_name: str) -> Dict[str, Any]:
        """Check asset availability using multiple public sources"""
        
        print(f"🔍 Checking availability for: {drug_name}")
        
        results = {
            "drug_name": drug_name,
            "timestamp": datetime.now().isoformat(),
            "clinical_trials_status": await self._check_clinical_trials_status(drug_name),
            "patent_search": await self._search_patents(drug_name),
            "news_mentions": await self._search_recent_news(drug_name),
            "availability_assessment": "unknown"
        }
        
        # Simple heuristic assessment
        results["availability_assessment"] = self._assess_availability(results)
        
        return results
    
    async def _check_clinical_trials_status(self, drug_name: str) -> Dict[str, Any]:
        """Check ClinicalTrials.gov for recent activity"""
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                # Search for recent trials with this drug
                url = "https://clinicaltrials.gov/api/v2/studies"
                params = {
                    "query.intr": drug_name,
                    "filter.overallStatus": "RECRUITING|ACTIVE_NOT_RECRUITING|COMPLETED",
                    "pageSize": 10,
                    "format": "json"
                }
                
                response = await client.get(url, params=params)
                
                if response.status_code == 200:
                    data = response.json()
                    studies = data.get("studies", [])
                    
                    # Analyze study sponsors and status
                    sponsors = []
                    latest_date = None
                    
                    for study in studies:
                        sponsor_info = study.get("protocolSection", {}).get("sponsorCollaboratorsModule", {})
                        lead_sponsor = sponsor_info.get("leadSponsor", {}).get("name", "Unknown")
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
            print(f"❌ ClinicalTrials.gov search failed: {e}")
        
        return {"status": "unknown", "active_studies": 0}
    
    async def _search_patents(self, drug_name: str) -> Dict[str, Any]:
        """Search for recent patents (simplified approach)"""
        
        # Note: Google Patents API requires special access
        # This is a placeholder for patent search logic
        
        try:
            # Could integrate with:
            # - USPTO API
            # - Google Patents (with API key)
            # - Patent web scraping (with caution)
            
            return {
                "patent_families": "would_search_uspto",
                "recent_filings": "would_check_last_5_years",
                "expiration_estimates": "would_calculate_from_priority_dates",
                "status": "search_not_implemented"
            }
            
        except Exception as e:
            print(f"❌ Patent search failed: {e}")
        
        return {"status": "unknown"}
    
    async def _search_recent_news(self, drug_name: str) -> Dict[str, Any]:
        """Search for recent news about asset transactions"""
        
        # Note: This is a simplified example
        # Real implementation would use news APIs or web scraping
        
        try:
            # Could integrate with:
            # - News API
            # - BioPharma Dive API
            # - Google News API
            # - Company press releases
            
            search_terms = [
                f"{drug_name} acquisition",
                f"{drug_name} licensing",
                f"{drug_name} sold",
                f"{drug_name} partnership"
            ]
            
            return {
                "search_terms": search_terms,
                "recent_mentions": "would_search_news_apis",
                "transaction_signals": "would_analyze_for_m_and_a",
                "status": "search_not_implemented"
            }
            
        except Exception as e:
            print(f"❌ News search failed: {e}")
        
        return {"status": "unknown"}
    
    def _assess_availability(self, results: Dict[str, Any]) -> str:
        """Simple heuristic to assess likely availability"""
        
        clinical_status = results.get("clinical_trials_status", {})
        active_studies = clinical_status.get("active_studies", 0)
        
        if active_studies > 0:
            return "possibly_unavailable_active_development"
        elif active_studies == 0:
            return "possibly_available_no_recent_activity"
        else:
            return "unknown_insufficient_data"

async def main():
    """Test asset availability checker"""
    
    if len(sys.argv) < 2:
        print("Usage: python check_asset_availability.py <drug_name>")
        print("Example: python check_asset_availability.py 'BIA 5-453'")
        return
    
    drug_name = sys.argv[1]
    
    checker = AssetAvailabilityChecker()
    results = await checker.check_asset(drug_name)
    
    print("\\n" + "="*60)
    print(f"📊 ASSET AVAILABILITY ANALYSIS: {drug_name.upper()}")
    print("="*60)
    
    print(f"\\n🔬 Clinical Trials Status:")
    ct_status = results["clinical_trials_status"]
    print(f"  • Active Studies: {ct_status.get('active_studies', 0)}")
    print(f"  • Recent Sponsors: {ct_status.get('recent_sponsors', [])}")
    print(f"  • Latest Activity: {ct_status.get('latest_activity', 'None')}")
    
    print(f"\\n🎯 Availability Assessment:")
    assessment = results["availability_assessment"]
    if "possibly_available" in assessment:
        print(f"  ✅ {assessment}")
        print("  💡 Recommendation: Worth investigating for licensing opportunities")
    elif "possibly_unavailable" in assessment:
        print(f"  ⚠️ {assessment}")
        print("  💡 Recommendation: Check if development is in same indication")
    else:
        print(f"  ❓ {assessment}")
        print("  💡 Recommendation: Requires deeper investigation")
    
    print(f"\\n📄 Full Results:")
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    asyncio.run(main())

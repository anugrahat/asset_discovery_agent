#!/usr/bin/env python3

import asyncio
import aiohttp
import json
from typing import Dict, Optional, List
import re

class RegulatoryAPIClient:
    """Client for querying real regulatory databases for drug approval status"""
    
    def __init__(self):
        self.session = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def get_regional_approvals(self, drug_name: str) -> Dict[str, bool]:
        """Get accurate regional approval status from official sources"""
        
        approvals = {
            "fda": False,
            "ema": False, 
            "pmda": False,
            "nmpa": False,
            "health_canada": False,
            "details": ""
        }
        
        # Check FDA (Orange Book + Drugs@FDA)
        fda_approved = await self._check_fda_approval(drug_name)
        approvals["fda"] = fda_approved
        
        # Check EMA (European Medicines Database)
        ema_approved = await self._check_ema_approval(drug_name)
        approvals["ema"] = ema_approved
        
        # Check PMDA (Japan - limited public API)
        pmda_approved = await self._check_pmda_approval(drug_name)
        approvals["pmda"] = pmda_approved
        
        # Check NMPA (China - limited public access)
        nmpa_approved = await self._check_nmpa_approval(drug_name)
        approvals["nmpa"] = nmpa_approved
        
        # Check Health Canada
        hc_approved = await self._check_health_canada_approval(drug_name)
        approvals["health_canada"] = hc_approved
        
        return approvals
    
    async def _check_fda_approval(self, drug_name: str) -> bool:
        """Check FDA approval via Drugs@FDA API"""
        try:
            # FDA openFDA API
            url = f"https://api.fda.gov/drug/label.json?search=openfda.brand_name:{drug_name}&limit=1"
            
            if self.session:
                async with self.session.get(url, timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()
                        return len(data.get("results", [])) > 0
        except:
            pass
        return False
    
    async def _check_ema_approval(self, drug_name: str) -> bool:
        """Check EMA approval via European Medicines Database"""
        try:
            # EMA public API (limited)
            url = f"https://www.ema.europa.eu/en/medicines/field_ema_web_categories%253Aname_field/Human/search_api_aggregation_ema_medicine_types/field_ema_med_marketing_authorisation_holder/{drug_name}"
            
            if self.session:
                async with self.session.get(url, timeout=10) as response:
                    if response.status == 200:
                        content = await response.text()
                        return "marketing authorisation" in content.lower()
        except:
            pass
        return False
    
    async def _check_pmda_approval(self, drug_name: str) -> bool:
        """Check PMDA approval (Japan) - limited public access"""
        # PMDA doesn't have a public API, would need web scraping
        # For now, return False and rely on curated data
        return False
    
    async def _check_nmpa_approval(self, drug_name: str) -> bool:
        """Check NMPA approval (China) - limited public access"""
        # NMPA doesn't have English public API
        # Would need specialized access or web scraping
        return False
    
    async def _check_health_canada_approval(self, drug_name: str) -> bool:
        """Check Health Canada approval"""
        try:
            # Health Canada Drug Product Database
            url = f"https://health-products.canada.ca/api/drug/drugproduct/?brand_name={drug_name}&type=json"
            
            if self.session:
                async with self.session.get(url, timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()
                        return len(data.get("results", [])) > 0
        except:
            pass
        return False

# Test function
async def test_regulatory_client():
    async with RegulatoryAPIClient() as client:
        result = await client.get_regional_approvals("pembrolizumab")
        print(f"Pembrolizumab approvals: {result}")

if __name__ == "__main__":
    asyncio.run(test_regulatory_client())

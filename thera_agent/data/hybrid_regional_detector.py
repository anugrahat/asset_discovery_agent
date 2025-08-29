#!/usr/bin/env python3

import asyncio
from typing import Dict, Optional
from .regional_approvals_db import get_regional_approvals, REGIONAL_APPROVALS
from .drug_resolver import DrugResolver
from .asset_webcrawler import AssetWebCrawler
from .enhanced_llm_verifier import EnhancedLLMVerifier
import json

class HybridRegionalDetector:
    """
    Hybrid approach: Curated database first, EnhancedLLMVerifier for unknown drugs
    Provides 100% accuracy for known drugs, reasonable accuracy for unknown ones
    """
    
    def __init__(self):
        self.drug_resolver = DrugResolver()
        self.web_crawler = AssetWebCrawler()
        
        # Initialize EnhancedLLMVerifier for unknown drugs
        try:
            self.enhanced_verifier = EnhancedLLMVerifier()
        except Exception as e:
            print(f"Failed to initialize EnhancedLLMVerifier: {e}")
            self.enhanced_verifier = None
        
    async def get_regional_approvals(self, drug_name: str) -> Dict[str, any]:
        """Get regional approvals with hybrid approach"""
        
        # Step 1: Check curated database first (100% accurate)
        curated_result = get_regional_approvals(drug_name)
        
        if not curated_result["details"].startswith("Regional approval status unknown"):
            # Found in curated database - return with confidence
            curated_result["confidence"] = "high"
            curated_result["source"] = "curated_database"
            return curated_result
        
        # Step 2: Use EnhancedLLMVerifier for unknown drugs
        if self.enhanced_verifier:
            try:
                verifier_result = await self.enhanced_verifier.verify_regional_approvals(drug_name)
                if verifier_result and verifier_result.get("approvals"):
                    # Convert EnhancedLLMVerifier format to expected format
                    approvals = verifier_result["approvals"]
                    
                    # Extract indications for display
                    indications = {}
                    for region, data in approvals.items():
                        if isinstance(data, dict) and data.get("approved") and data.get("indications"):
                            indications[region] = data["indications"]
                    
                    result = {
                        "fda": approvals.get("fda", {}).get("approved", False),
                        "ema": approvals.get("ema", {}).get("approved", False),
                        "pmda": approvals.get("pmda", {}).get("approved", False),
                        "nmpa": approvals.get("nmpa", {}).get("approved", False),
                        "health_canada": approvals.get("health_canada", {}).get("approved", False),
                        "details": f"Enhanced LLM verification for {drug_name}",
                        "confidence": "medium",
                        "source": "enhanced_llm_verifier",
                        "indications": indications  # Include disease indications
                    }
                    return result
            except Exception as e:
                print(f"EnhancedLLMVerifier failed for {drug_name}: {e}")
        
        # Step 4: Conservative fallback - assume no approvals
        return {
            "fda": False,
            "ema": False,
            "pmda": False,
            "nmpa": False,
            "health_canada": False,
            "details": f"Unknown drug: {drug_name}",
            "confidence": "low",
            "source": "conservative_fallback"
        }
    
    async def _web_search_regional_approvals(self, drug_name: str) -> Dict[str, any]:
        """Use web search to find regional approval information"""
        try:
            # Use a simple web search approach since AssetWebCrawler doesn't have search_web
            # This is a placeholder - in production you'd use a proper search API
            import aiohttp
            
            search_query = f"{drug_name} drug approval FDA EMA PMDA NMPA Health Canada"
            
            # Simple Google search simulation (placeholder)
            # In production, use proper search APIs like SerpAPI, Bing API, etc.
            search_results = await self._simple_web_search(search_query)
            
            if not search_results:
                return None
            
            # Analyze search results for approval indicators
            approval_indicators = {
                "fda": ["fda approved", "fda approval", "approved by fda", "us approval"],
                "ema": ["ema approved", "ema approval", "european medicines agency", "eu approval", "europe approved"],
                "pmda": ["pmda approved", "japan approved", "japanese approval", "pmda approval"],
                "nmpa": ["nmpa approved", "china approved", "chinese approval", "cfda approved"],
                "health_canada": ["health canada approved", "canada approved", "canadian approval"]
            }
            
            approvals = {
                "fda": False,
                "ema": False, 
                "pmda": False,
                "nmpa": False,
                "health_canada": False
            }
            
            evidence = []
            
            # Check search results for approval indicators
            for result in search_results[:5]:  # Check top 5 results
                content = (result.get('title', '') + ' ' + result.get('snippet', '')).lower()
                
                for region, indicators in approval_indicators.items():
                    if any(indicator in content for indicator in indicators):
                        approvals[region] = True
                        evidence.append(f"{region.upper()}: {result.get('title', '')[:50]}...")
            
            # Determine confidence based on evidence
            total_approvals = sum(approvals.values())
            if total_approvals > 0:
                confidence = "medium" if len(evidence) >= 2 else "low"
                details = f"Web search found {total_approvals} regional approvals. Evidence: {'; '.join(evidence[:3])}"
            else:
                confidence = "low"
                details = f"No clear approval evidence found in web search for {drug_name}"
            
            return {
                **approvals,
                "details": details,
                "confidence": confidence,
                "source": "web_search",
                "evidence": evidence
            }
            
        except Exception as e:
            print(f"Web search error for {drug_name}: {e}")
            return None
    
    def add_to_curated_db(self, drug_name: str, approvals: Dict[str, any]):
        """Add verified drug approval data to curated database"""
        REGIONAL_APPROVALS[drug_name.lower()] = {
            "fda": approvals.get("fda", False),
            "ema": approvals.get("ema", False), 
            "pmda": approvals.get("pmda", False),
            "nmpa": approvals.get("nmpa", False),
            "health_canada": approvals.get("health_canada", False),
            "details": approvals.get("details", "")
        }
        print(f"✅ Added {drug_name} to curated database")

# Test the hybrid approach
async def test_hybrid_detector():
    detector = HybridRegionalDetector()
    
    test_drugs = [
        "sintilimab",      # In curated DB
        "pembrolizumab",   # In curated DB  
        "some_new_drug",   # Not in curated DB - will use LLM
    ]
    
    for drug in test_drugs:
        result = await detector.get_regional_approvals(drug)
        print(f"\n{drug}:")
        print(f"  Confidence: {result.get('confidence', 'unknown')}")
        print(f"  Source: {result.get('source', 'unknown')}")
        print(f"  Approvals: FDA={result.get('fda')}, China={result.get('nmpa')}")

if __name__ == "__main__":
    asyncio.run(test_hybrid_detector())

"""Drug Shelving Reason Investigator

Uses web search, patent databases, press releases, and LLMs to determine
why drugs were shelved or discontinued.
"""

import asyncio
import json
import os
from typing import Dict, List, Any, Optional
from datetime import datetime
import aiohttp
from bs4 import BeautifulSoup
import re
from openai import AsyncOpenAI
from .cache import APICache
from .http_client import RateLimitedClient


class ShelvingReasonInvestigator:
    """Investigates why drugs were shelved using multiple sources"""
    
    def __init__(self, cache_manager: Optional[APICache] = None, openai_api_key: Optional[str] = None):
        self.cache = cache_manager if cache_manager else APICache()
        self.http_client = RateLimitedClient()
        # Get OpenAI API key from environment if not provided
        if not openai_api_key:
            openai_api_key = os.getenv("OPENAI_API_KEY")
        self.openai_client = AsyncOpenAI(api_key=openai_api_key) if openai_api_key else None
        
    async def investigate_shelving_reasons(
        self, 
        drug_name: str,
        company: Optional[str] = None,
        phase: Optional[str] = None,
        disease: Optional[str] = None
    ) -> Dict[str, Any]:
        """Comprehensive investigation of why a drug was shelved"""
        # Check cache first
        cache_key = f"shelving_reason:{drug_name}:{company}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached
            
        results = {
            "drug_name": drug_name,
            "company": company,
            "phase": phase,
            "disease": disease,
            "shelving_reasons": [],
            "sources": [],
            "confidence": 0,
            "primary_reason": None,
            "detailed_analysis": None
        }
        
        # Search multiple sources in parallel
        tasks = [
            self._search_press_releases(drug_name, company),
            self._search_clinical_trials_registry(drug_name),
            self._search_web_news(drug_name, company),
        ]
        
        search_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Combine all findings
        all_text = []
        for result in search_results:
            if isinstance(result, dict) and not isinstance(result, Exception):
                if result.get("text"):
                    all_text.append(result["text"])
                if result.get("source"):
                    results["sources"].append(result["source"])
                    
        # Use LLM to analyze all findings
        if all_text and self.openai_client:
            analysis = await self._analyze_with_llm(
                drug_name, company, phase, disease, "\n\n".join(all_text)
            )
            results.update(analysis)
        else:
            # Fallback pattern matching
            results = self._extract_reasons_with_patterns(all_text, results)
            
        # Cache results
        self.cache.set(cache_key, results, ttl_hours=7*24)  # 7 days
        
        return results
    
    async def investigate_shelving_reason(
        self, 
        drug_name: str,
        company: Optional[str] = None,
        phase: Optional[str] = None,
        disease: Optional[str] = None
    ) -> Dict[str, Any]:
        """Wrapper for investigate_shelving_reasons for backward compatibility"""
        result = await self.investigate_shelving_reasons(drug_name, company, phase, disease)
        
        # Transform to expected format
        return {
            "reason": result.get("primary_reason") or "Unknown",
            "confidence": result.get("confidence", 0.0),
            "details": result.get("detailed_analysis", {}),
            "sources": result.get("sources", [])
        }
        
    async def _search_press_releases(self, drug_name: str, company: str) -> Dict[str, Any]:
        """Search company press releases for discontinuation announcements"""
        try:
            if not company:
                return {}
                
            # Search for press releases
            query = f'"{drug_name}" "{company}" "discontinued" OR "terminated" OR "stopped development"'
            
            # Use web search to find press releases
            search_results = await self._web_search(query + " site:prnewswire.com OR site:businesswire.com")
            
            return {
                "text": search_results,
                "source": f"Press Releases for {company}"
            }
            
        except Exception as e:
            print(f"Error searching press releases: {e}")
            return {}
            
    async def _search_clinical_trials_registry(self, drug_name: str) -> Dict[str, Any]:
        """Search ClinicalTrials.gov for termination reasons"""
        try:
            # Sanitize drug name for ClinicalTrials.gov API
            # Remove problematic characters that cause 400 errors
            sanitized_name = drug_name
            # Remove square brackets and their contents (e.g., [225Ac]-FPI-2068 -> FPI-2068)
            import re
            sanitized_name = re.sub(r'\[[^\]]*\]-?', '', sanitized_name)
            # Remove other problematic characters
            sanitized_name = re.sub(r'[<>{}|\\^`\[\]]', '', sanitized_name)
            sanitized_name = sanitized_name.strip()
            
            if not sanitized_name:
                return {}
                
            url = "https://clinicaltrials.gov/api/v2/studies"
            params = {
                "query.intr": sanitized_name,
                "filter.overallStatus": "TERMINATED,WITHDRAWN,SUSPENDED",
                "fields": "NCTId,BriefTitle,WhyStopped,DetailedDescription",
                "pageSize": 10
            }
            
            response = await self.http_client.get(url, params=params)
            if response and response.get("studies"):
                reasons = []
                for study in response["studies"]:
                    protocol = study.get("protocolSection", {})
                    status = protocol.get("statusModule", {})
                    why_stopped = status.get("whyStopped", "")
                    if why_stopped:
                        reasons.append(f"Trial stopped: {why_stopped}")
                        
                return {
                    "text": " | ".join(reasons),
                    "source": "ClinicalTrials.gov"
                }
                
        except Exception as e:
            print(f"Error searching clinical trials: {e}")
            
        return {}
        
    async def _search_web_news(self, drug_name: str, company: str) -> Dict[str, Any]:
        """Search web for news articles about drug discontinuation"""
        try:
            query = f'"{drug_name}"'
            if company:
                query += f' "{company}"'
            query += ' discontinued failed "clinical trial" development stopped'
            
            search_results = await self._web_search(query)
            
            return {
                "text": search_results,
                "source": "Web News Articles"
            }
            
        except Exception as e:
            print(f"Error searching web news: {e}")
            return {}
            
    async def _web_search(self, query: str) -> str:
        """Simple web search using search API - DISABLED to avoid errors"""
        # Web search disabled - drug resolver already handles web search for target identification
        return ""
        
    async def _analyze_with_llm(
        self, 
        drug_name: str, 
        company: str,
        phase: str,
        disease: str,
        search_text: str
    ) -> Dict[str, Any]:
        """Use LLM to analyze search results and extract shelving reasons"""
        if not self.openai_client:
            return {}
            
        try:
            prompt = f"""Analyze the following information about the drug {drug_name} developed by {company} for {disease} (reached {phase}) to determine why it was shelved/discontinued.

Search Results:
{search_text[:3000]}

Extract and categorize the reasons for discontinuation. Return a JSON with:
{{
    "primary_reason": "main reason (safety/efficacy/business/regulatory/other)",
    "detailed_analysis": "2-3 sentence explanation of what happened",
    "shelving_reasons": ["list", "of", "specific", "reasons"],
    "confidence": 0-100 (how confident you are in this analysis),
    "key_events": ["key events or dates mentioned"]
}}

Focus on facts from the search results. If unclear, indicate low confidence."""

            response = await self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            return result
            
        except Exception as e:
            print(f"LLM analysis error: {e}")
            return {}
            
    def _extract_reasons_with_patterns(self, text_list: List[str], results: Dict) -> Dict:
        """Fallback pattern matching for shelving reasons"""
        all_text = " ".join(text_list).lower()
        
        # Pattern matching for common reasons
        patterns = {
            "safety": [
                r"safety concern", r"adverse event", r"toxicity", r"side effect",
                r"serious adverse", r"death", r"hospitalization"
            ],
            "efficacy": [
                r"lack of efficacy", r"did not meet", r"failed to show", r"no significant",
                r"primary endpoint", r"insufficient efficacy", r"not effective"
            ],
            "business": [
                r"business decision", r"strategic", r"portfolio", r"commercial",
                r"market", r"competitive", r"resource", r"prioriti"
            ],
            "regulatory": [
                r"regulatory", r"fda", r"approval", r"clinical hold",
                r"compliance", r"requirement"
            ],
            "recruitment": [
                r"recruitment", r"enrollment", r"patient accrual", r"slow enrollment"
            ]
        }
        
        found_reasons = []
        primary_category = "unknown"
        max_matches = 0
        
        for category, patterns_list in patterns.items():
            matches = 0
            for pattern in patterns_list:
                if re.search(pattern, all_text):
                    matches += 1
                    
            if matches > max_matches:
                max_matches = matches
                primary_category = category
                
        # Extract specific phrases
        if "discontinued" in all_text:
            # Try to extract text around "discontinued"
            disc_matches = re.findall(r'(.{0,100}discontinued.{0,100})', all_text)
            for match in disc_matches[:2]:
                found_reasons.append(match.strip())
                
        results["primary_reason"] = primary_category
        results["shelving_reasons"] = found_reasons[:5]
        results["confidence"] = min(max_matches * 20, 60)  # Pattern matching = lower confidence
        results["detailed_analysis"] = f"Based on pattern matching, the drug appears to have been discontinued primarily due to {primary_category} reasons."
        
        return results

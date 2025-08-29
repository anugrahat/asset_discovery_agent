"""
Standard-of-Care Drug Identifier using LLM
Identifies currently approved standard treatments for diseases
"""
import json
import logging
from typing import List, Dict, Set
import asyncio
from datetime import datetime, timedelta
import hashlib

logger = logging.getLogger(__name__)

class StandardOfCareIdentifier:
    """Identify standard-of-care drugs for diseases using LLM intelligence"""
    
    def __init__(self, cache_client=None):
        self.cache = cache_client
        self._llm_available = None
        
    async def _llm_query(self, prompt: str) -> str:
        """Query LLM with fallback handling"""
        try:
            from openai import AsyncOpenAI
            import os
            
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                logger.warning("No OpenAI API key found")
                return None
                
            client = AsyncOpenAI(api_key=api_key)
            
            response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=2000
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.warning(f"LLM query failed: {e}")
            return None
    
    async def get_standard_of_care_drugs(self, disease: str) -> Set[str]:
        """Get comprehensive list of SOC drugs for a disease"""
        # Check cache first
        cache_key = f"soc_drugs:{disease.lower()}"
        if self.cache:
            cached = self.cache.get(cache_key)
            if cached:
                return set(cached)
        
        # Always use LLM for ANY disease
        soc_drugs = await self._identify_soc_drugs_llm(disease)
        
        # Cache results
        if self.cache and soc_drugs:
            self.cache.set(cache_key, list(soc_drugs), ttl_hours=168)  # 7 days
            
        return soc_drugs if soc_drugs else set()
    
    async def _identify_soc_drugs_llm(self, disease: str) -> Set[str]:
        """Use LLM to identify standard-of-care drugs"""
        prompt = f"""List ALL standard-of-care drugs currently used to treat {disease}.
        
Include:
- First-line treatments
- Second-line treatments  
- Common combination therapies
- Both generic AND brand names
- Drug classes commonly used

Format as JSON list of drug names. Be comprehensive - include all major drugs.

Example for hypertension:
["lisinopril", "enalapril", "ramipril", "amlodipine", "nifedipine", "metoprolol", 
"atenolol", "hydrochlorothiazide", "furosemide", "losartan", "valsartan", ...]

Now provide the comprehensive list for {disease}:"""

        response = await self._llm_query(prompt)
        if not response:
            return set()
            
        try:
            # Extract JSON from response
            import re
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                drugs = json.loads(json_match.group())
                # Normalize drug names
                normalized = set()
                for drug in drugs:
                    if isinstance(drug, str) and drug:
                        # Add original and lowercase
                        normalized.add(drug.lower().strip())
                        normalized.add(drug.strip())
                return normalized
        except Exception as e:
            logger.warning(f"Failed to parse LLM SOC response: {e}")
            
        return set()
    

    
    def is_standard_of_care(self, drug_name: str, disease: str, soc_drugs: Set[str] = None) -> bool:
        """Check if a drug is standard-of-care for a disease"""
        if not soc_drugs:
            # This should be async but keeping sync for compatibility
            # Caller should provide soc_drugs
            return False
            
        drug_lower = drug_name.lower().strip()
        
        # Direct match
        if drug_lower in soc_drugs:
            return True
            
        # Check if drug name contains any SOC drug
        for soc in soc_drugs:
            if soc in drug_lower or drug_lower in soc:
                return True
                
        return False

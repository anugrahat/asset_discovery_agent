#!/usr/bin/env python3

import asyncio
import aiohttp
from typing import Dict, List, Optional
import json

class MultiSourceVerifier:
    """
    Dynamic regional approval verification using multiple sources:
    1. Multiple LLM calls with different prompts
    2. Cross-reference with regulatory APIs where available
    3. Consensus-based decision making
    """
    
    def __init__(self, openai_client):
        self.openai_client = openai_client
        self.session = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def get_verified_regional_approvals(self, drug_name: str) -> Dict:
        """Get regional approvals using multi-source verification"""
        
        # Step 1: Multiple LLM queries with different approaches
        llm_results = await self._get_multiple_llm_opinions(drug_name)
        
        # Step 2: API verification where possible
        api_results = await self._verify_with_apis(drug_name)
        
        # Step 3: Consensus decision
        final_result = self._build_consensus(llm_results, api_results)
        
        return final_result
    
    async def _get_multiple_llm_opinions(self, drug_name: str) -> List[Dict]:
        """Get multiple LLM opinions with different prompting strategies"""
        
        prompts = [
            # Prompt 1: Fact-checking focus
            f"""You are a pharmaceutical regulatory expert. For the drug '{drug_name}', provide ONLY factual regulatory approval status.

CRITICAL: Only mark as approved if you are absolutely certain the drug has marketing authorization from that agency.

Respond with JSON:
{{
  "fda": true/false,
  "ema": true/false, 
  "pmda": true/false,
  "nmpa": true/false,
  "health_canada": true/false,
  "cdsco": true/false,
  "confidence": "high/medium/low"
}}""",

            # Prompt 2: Company-focused approach
            f"""Research the drug '{drug_name}' and its regulatory history. Focus on:
1. Which company developed/markets it
2. Where did they first get approval
3. Which regions have they expanded to

Be conservative - only mark as approved if certain.

JSON response:
{{
  "fda": true/false,
  "ema": true/false,
  "pmda": true/false, 
  "nmpa": true/false,
  "health_canada": true/false,
  "cdsco": true/false,
  "reasoning": "brief explanation"
}}""",

            # Prompt 3: Timeline-based approach  
            f"""For drug '{drug_name}', trace its approval timeline:
- When was it first approved and where?
- Has it expanded to other regions?
- Any rejections or withdrawals?

Conservative assessment only.

JSON:
{{
  "fda": true/false,
  "ema": true/false,
  "pmda": true/false,
  "nmpa": true/false,
  "health_canada": true/false,
  "cdsco": true/false,
  "timeline": "brief timeline if known"
}}"""
        ]
        
        results = []
        for i, prompt in enumerate(prompts):
            try:
                response = await self.openai_client.chat(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a pharmaceutical regulatory database expert. Respond with valid JSON only."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    response_format={"type": "json_object"}
                )
                
                content = response.choices[0].message.content.strip()
                result = json.loads(content)
                result["source"] = f"llm_prompt_{i+1}"
                results.append(result)
                
            except Exception as e:
                print(f"LLM query {i+1} failed: {e}")
        
        return results
    
    async def _verify_with_apis(self, drug_name: str) -> Dict:
        """Verify with available regulatory APIs"""
        
        api_results = {
            "fda": None,
            "ema": None,
            "health_canada": None,
            "cdsco": None,
            "source": "regulatory_apis"
        }
        
        if not self.session:
            return api_results
        
        # FDA verification
        try:
            fda_url = f"https://api.fda.gov/drug/label.json?search=openfda.brand_name:{drug_name}&limit=1"
            async with self.session.get(fda_url, timeout=5) as response:
                if response.status == 200:
                    data = await response.json()
                    api_results["fda"] = len(data.get("results", [])) > 0
        except:
            pass
        
        # Health Canada verification
        try:
            hc_url = f"https://health-products.canada.ca/api/drug/drugproduct/?brand_name={drug_name}&type=json"
            async with self.session.get(hc_url, timeout=5) as response:
                if response.status == 200:
                    data = await response.json()
                    api_results["health_canada"] = len(data.get("results", [])) > 0
        except:
            pass
        
        return api_results
    
    def _build_consensus(self, llm_results: List[Dict], api_results: Dict) -> Dict:
        """Build consensus from multiple sources"""
        
        regions = ["fda", "ema", "pmda", "nmpa", "health_canada"]
        consensus = {}
        
        for region in regions:
            votes = []
            confidence_scores = []
            
            # Collect LLM votes
            for result in llm_results:
                if region in result:
                    votes.append(result[region])
                    # Weight by confidence if available
                    conf = result.get("confidence", "medium")
                    weight = {"high": 1.0, "medium": 0.7, "low": 0.3}.get(conf, 0.5)
                    confidence_scores.append(weight)
            
            # Add API verification (higher weight)
            if api_results.get(region) is not None:
                votes.append(api_results[region])
                confidence_scores.append(1.5)  # Higher weight for API data
            
            # Consensus decision
            if votes:
                # Weighted voting
                weighted_sum = sum(vote * weight for vote, weight in zip(votes, confidence_scores))
                total_weight = sum(confidence_scores)
                consensus_score = weighted_sum / total_weight if total_weight > 0 else 0
                
                # Conservative threshold - need >0.6 confidence for approval
                consensus[region] = consensus_score > 0.6
            else:
                consensus[region] = False
        
        # Add metadata
        consensus["verification_method"] = "multi_source_consensus"
        consensus["llm_sources"] = len(llm_results)
        consensus["api_sources"] = sum(1 for v in api_results.values() if v is not None)
        
        return consensus

# Test function
async def test_multi_source_verifier():
    from thera_agent.data.drug_resolver import DrugResolver
    
    resolver = DrugResolver()
    if not resolver.openai_client:
        print("No OpenAI client available")
        return
    
    async with MultiSourceVerifier(resolver.openai_client) as verifier:
        result = await verifier.get_verified_regional_approvals("sintilimab")
        print(f"Multi-source verification for sintilimab:")
        print(json.dumps(result, indent=2))

if __name__ == "__main__":
    asyncio.run(test_multi_source_verifier())

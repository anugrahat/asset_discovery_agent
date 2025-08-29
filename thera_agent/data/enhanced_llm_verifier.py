#!/usr/bin/env python3

import asyncio
import json
from typing import Dict, Optional
from thera_agent.llm_client import LLMClient, extract_json

class EnhancedLLMVerifier:
    """
    Enhanced LLM-based regional approval detection with self-verification
    Uses chain-of-thought reasoning and self-correction
    """
    
    def __init__(self, openai_client=None, llm_client: Optional[LLMClient] = None):
        # Backward compatible signature; prefer unified LLMClient wrapper
        self._raw_openai_client = openai_client  # kept for compatibility, not used
        self.client: LLMClient = llm_client or LLMClient()
    
    async def get_verified_regional_approvals(self, drug_name: str) -> Dict:
        """Get regional approvals with enhanced LLM verification"""
        
        # Step 1: Initial research with chain-of-thought
        initial_result = await self._research_with_reasoning(drug_name)
        
        # Step 2: Self-verification step
        verified_result = await self._self_verify_result(drug_name, initial_result)
        
        return verified_result
    
    async def _research_with_reasoning(self, drug_name: str) -> Dict:
        """Research drug approvals with explicit reasoning"""
        
        prompt = f"""Research the drug '{drug_name}' step by step:

STEP 1: Identify the drug
- What type of drug is this?
- Which company developed/markets it?
- What is its mechanism of action?

STEP 2: Research regulatory history
- Where was it first approved?
- Which regulatory agencies have approved it?
- Any notable rejections or limitations?

STEP 3: Verify each region (be conservative):
- FDA (USA): Is it approved for marketing in the US?
- EMA (Europe): Is it approved for marketing in Europe?
- PMDA (Japan): Is it approved for marketing in Japan?
- NMPA (China): Is it approved for marketing in China?
- Health Canada: Is it approved for marketing in Canada?
- CDSCO (India): Is it approved for marketing in India?

IMPORTANT: Only mark as 'true' if you have strong evidence of marketing approval.
Clinical trials or investigational use does NOT count as approval.

Respond with JSON (be precise, list marketed indication disease names per region when approved; use an empty list when not approved or unknown):
{{
  "drug_info": {{
    "company": "developing company",
    "mechanism": "mechanism of action",
    "drug_class": "therapeutic class"
  }},
  "reasoning": {{
    "fda": "reasoning for FDA status",
    "ema": "reasoning for EMA status", 
    "pmda": "reasoning for PMDA status",
    "nmpa": "reasoning for NMPA status",
    "health_canada": "reasoning for Health Canada status",
    "cdsco": "reasoning for CDSCO status"
  }},
  "regional_approvals": {{
    "fda": true/false,
    "ema": true/false,
    "pmda": true/false,
    "nmpa": true/false,
    "health_canada": true/false,
    "cdsco": true/false
  }},
  "indications": {{
    "fda": ["Disease A", "Disease B"],
    "ema": ["Disease A"],
    "pmda": ["Disease X"],
    "nmpa": ["Disease Y"],
    "health_canada": ["Disease Z"],
    "cdsco": ["Disease W"]
  }}
}}"""
        
        try:
            response = await self.client.chat(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a pharmaceutical regulatory expert. Think step by step and provide detailed reasoning."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                response_format={"type": "json_object"}
            )

            # Robustly extract content for both OpenAI v1 and v0.28 responses
            content = None
            try:
                content = response.choices[0].message.content
            except Exception:
                try:
                    content = response["choices"][0]["message"]["content"]
                except Exception:
                    content = "{}"
            content = extract_json(content)
            return json.loads(content)

        except Exception as e:
            print(f"Initial research failed: {e}")
            return {}
    
    async def _self_verify_result(self, drug_name: str, initial_result: Dict) -> Dict:
        """Self-verify the initial result with critical questioning"""
        
        if not initial_result or "regional_approvals" not in initial_result:
            return self._conservative_fallback(drug_name)
        
        approvals = initial_result["regional_approvals"]
        reasoning = initial_result.get("reasoning", {})
        indications = initial_result.get("indications", {})
        
        verification_prompt = f"""You previously analyzed '{drug_name}' and concluded:

FDA: {approvals.get('fda', False)} - {reasoning.get('fda', 'No reasoning')}
EMA: {approvals.get('ema', False)} - {reasoning.get('ema', 'No reasoning')}
PMDA: {approvals.get('pmda', False)} - {reasoning.get('pmda', 'No reasoning')}
NMPA: {approvals.get('nmpa', False)} - {reasoning.get('nmpa', 'No reasoning')}
Health Canada: {approvals.get('health_canada', False)} - {reasoning.get('health_canada', 'No reasoning')}

Proposed indications per region (may be empty):
FDA: {indications.get('fda', [])}
EMA: {indications.get('ema', [])}
PMDA: {indications.get('pmda', [])}
NMPA: {indications.get('nmpa', [])}
Health Canada: {indications.get('health_canada', [])}

Now CRITICALLY REVIEW each conclusion:
1. Are you certain about each approval status?
2. Could you have confused clinical trials with marketing approval?
3. Are there any contradictions in your reasoning?
4. Should any 'true' values be changed to 'false' to be more conservative?
5. Correct the indications lists to include marketed indication disease names per region only if approved; otherwise return an empty list.

Respond with JSON as a single object. Provide corrected results with confidence levels in JSON format exactly with these keys:

{{
  "corrections_made": "description of any changes",
  "final_approvals": {{
    "fda": true/false,
    "ema": true/false,
    "pmda": true/false,
    "nmpa": true/false,
    "health_canada": true/false
  }},
  "final_indications": {{
    "fda": ["..."],
    "ema": ["..."],
    "pmda": ["..."],
    "nmpa": ["..."],
    "health_canada": ["..."]
  }},
  "confidence_levels": {{
    "fda": "high/medium/low",
    "ema": "high/medium/low",
    "pmda": "high/medium/low",
    "nmpa": "high/medium/low",
    "health_canada": "high/medium/low"
  }}
}}"""
        
        try:
            response = await self.client.chat(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are critically reviewing your previous analysis. Be conservative and correct any errors."},
                    {"role": "user", "content": verification_prompt}
                ],
                temperature=0.1,
                response_format={"type": "json_object"}
            )

            content = None
            try:
                content = response.choices[0].message.content
            except Exception:
                try:
                    content = response["choices"][0]["message"]["content"]
                except Exception:
                    content = "{}"
            content = extract_json(content)
            verification_result = json.loads(content)

            # Combine results (approvals + indications)
            final_result = verification_result.get("final_approvals", {})
            final_result["verification_method"] = "enhanced_llm_with_self_verification"
            final_result["confidence_levels"] = verification_result.get("confidence_levels", {})
            final_result["corrections_made"] = verification_result.get("corrections_made", "")
            # Attach verified indications (if provided); otherwise pass through initial indications
            final_result["indications"] = verification_result.get("final_indications", initial_result.get("indications", {}))

            return final_result

        except Exception as e:
            print(f"Self-verification failed: {e}")
            return self._conservative_fallback(drug_name)
    
    def _conservative_fallback(self, drug_name: str) -> Dict:
        """Conservative fallback when verification fails"""
        return {
            "fda": False,
            "ema": False,
            "pmda": False,
            "nmpa": False,
            "health_canada": False,
            "verification_method": "conservative_fallback",
            "details": f"Could not verify regional approvals for {drug_name}",
            "indications": {}
        }

# Test function
async def test_enhanced_verifier():
    from thera_agent.data.drug_resolver import DrugResolver
    
    resolver = DrugResolver()
    if not resolver.openai_client:
        print("No OpenAI client available")
        return
    
    verifier = EnhancedLLMVerifier(resolver.openai_client)
    
    test_drugs = ["sintilimab", "pembrolizumab"]
    
    for drug in test_drugs:
        print(f"\n🔍 Testing {drug}:")
        result = await verifier.get_verified_regional_approvals(drug)
        
        print(f"Approvals: FDA={result.get('fda')}, China={result.get('nmpa')}")
        print(f"Method: {result.get('verification_method')}")
        if result.get('corrections_made'):
            print(f"Corrections: {result.get('corrections_made')}")

if __name__ == "__main__":
    asyncio.run(test_enhanced_verifier())

"""
Drug name to ChEMBL ID resolver using hybrid approach with LLM normalization
"""
import re
import json
import logging
import os
from typing import Optional, Dict, List
from .http_client import RateLimitedClient
from .cache import APICache

# Try to import OpenAI (optional dependency)
try:
    from ..llm_client import LLMClient
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

logger = logging.getLogger(__name__)


class DrugResolver:
    """Resolves drug names to ChEMBL IDs using multiple strategies"""
    
    def __init__(self, http_client: RateLimitedClient = None, cache: APICache = None):
        self.http = http_client or RateLimitedClient()
        self.cache = cache or APICache()
        self.has_openai = HAS_OPENAI
        # Initialize OpenAI client if available
        if self.has_openai:
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if openai_api_key:
                self.openai_client = LLMClient()
            else:
                self.openai_client = None
        else:
            self.openai_client = None
        # No manual mappings - rely on LLM for dynamic resolution
        self.code_mappings = {}
        
    # Regex patterns for aggressive cleaning
    DOSE_ADMIN_RGX = re.compile(
        r"(\b\d+(\.\d+)?\s*(mg(?:/m2)?|mcg|µg|g|%|ml|units?|iu)\b)|"  # doses
        r"(\biv\b|\bsc\b|\bim\b|\bpo\b|intravenous|oral|subcutaneous|intramuscular|topical)|"  # admin routes
        r"(- iv|- sc|- oral|- po)",  # dash-separated admin routes
        re.I,
    )
    
    # Non-drugs to short-circuit immediately
    NON_DRUGS = {
        "placebo", "saline", "vehicle", "sham control", "saline solution",
        "normal saline", "control", "dummy", "inactive"
    }
    
    def _clean_drug_name(self, name: str) -> Optional[str]:
        """Clean drug name for better matching, return None for non-drugs"""
        if not name or not name.strip():
            return None
            
        cleaned = name.strip().lower()
        
        # Short-circuit obvious non-drugs
        for non_drug in self.NON_DRUGS:
            if non_drug in cleaned:
                return None
        
        # Handle imaging tracers separately (optional: could map via PubChem only)
        if re.search(r'\[[0-9]+[a-z]\]', cleaned):  # e.g., [123I], [18F]
            logger.debug(f"Skipping imaging tracer: {name}")
            return None
            
        # Apply aggressive dose/admin cleaning
        cleaned = self.DOSE_ADMIN_RGX.sub("", cleaned)
        
        # Remove brackets and parenthetical content
        cleaned = re.sub(r"[\(\[].*?[\)\]]", "", cleaned)
        
        # Remove common salt suffixes
        suffixes = ['hydrochloride', 'hcl', 'sodium', 'potassium', 'sulfate', 'acetate', 'tartrate', 'citrate']
        for suffix in suffixes:
            cleaned = re.sub(rf'\b{suffix}\b', '', cleaned, flags=re.I)
        
        # Clean up whitespace and normalize
        cleaned = ' '.join(cleaned.split())
        
        if not cleaned or len(cleaned) < 2:
            return None
            
        return cleaned
    
    async def _normalize_drug_name_with_llm(self, drug_name: str) -> Optional[Dict]:
        """Use LLM to normalize drug names and extract metadata"""
        
        cache_key = f"drug_resolve_{drug_name}"
        cached = self.cache.get(cache_key) if self.cache else None
        if cached:
            return cached
        cache_key = f"llm_normalize:{drug_name.lower()}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached
        
        prompt = f"""Research drug '{drug_name}' step-by-step:

STEP 1 - IDENTIFY:
- Drug type and mechanism
- Developing company
- Research codes: RO=Roche, CKD=Chong Kun Dang, JNJ=Johnson&Johnson, BMS=Bristol-Myers, MK=Merck, GS=Gilead

STEP 2 - REGULATORY RESEARCH:
Think carefully about each region. Only mark TRUE if you have strong evidence of MARKETING APPROVAL:
- FDA (USA): Approved for marketing? (not just clinical trials)
- EMA (Europe): Approved for marketing? (not just clinical trials)  
- PMDA (Japan): Approved for marketing? (not just clinical trials)
- NMPA (China): Approved for marketing? (Chinese companies often get approval here first)
- Health Canada: Approved for marketing? (not just clinical trials)

STEP 3 - SELF-CHECK:
Review each TRUE answer - are you absolutely certain? If unsure, mark FALSE.

CRITICAL: Clinical trials ≠ Marketing approval. Be conservative.

Respond with ONLY this JSON structure:
{{
  "normalized_name": "generic_name or combination or NOT_A_DRUG",
  "target": "target protein/receptor or null",
  "mechanism": "mechanism of action or null",
  "regional_approvals": {{
    "fda": true/false,
    "ema": true/false,
    "pmda": true/false,
    "nmpa": true/false,
    "health_canada": true/false,
    "details": "brief approval info or null"
  }}
}}"""
        
        try:
            if not self.openai_client:
                raise Exception("OpenAI client not available")
            
            response = await self.openai_client.chat(
                model="gpt-4o",  # Use cheaper model
                messages=[
                    {"role": "system", "content": "You are a pharmaceutical database expert. You MUST respond with valid JSON only. No additional text before or after the JSON object."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=300,
                response_format={"type": "json_object"}  # Force JSON response
            )
            
            content = response.choices[0].message.content.strip()
            
            # Remove markdown code block if present
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
            
            # Try to parse as JSON
            try:
                import json
                # Try to extract JSON from response if mixed with text
                json_start = content.find('{')
                json_end = content.rfind('}')
                if json_start >= 0 and json_end > json_start:
                    content = content[json_start:json_end+1]
                
                result = json.loads(content)
                normalized = result.get("normalized_name", "").strip()
                
                # Handle special responses or combinations with NOT_A_DRUG
                if normalized.upper() == "NOT_A_DRUG" or "NOT_A_DRUG" in normalized:
                    self.cache.set(cache_key, None, ttl_hours=24)
                    return None
                
                # Store the full result including target/mechanism and regional approvals
                full_result = {
                    "normalized_name": normalized,
                    "generic_name": normalized,  # Add generic_name for compatibility
                    "target": result.get("target"),
                    "mechanism": result.get("mechanism"),
                    "regional_approvals": result.get("regional_approvals")
                }
                self.cache.set(cache_key, full_result, ttl_hours=48)
                return full_result
                
            except json.JSONDecodeError as e:
                logger.error(f"JSON parse error for drug '{drug_name}'. LLM response: {content[:200]}... Error: {e}")
                # Don't use fallback - return None to try other resolution methods
                return None
            
        except Exception as e:
            logger.debug(f"LLM normalization failed for {drug_name}: {e}")
            return None
    
    async def _search_chembl_direct(self, query: str) -> Optional[Dict[str, any]]:
        """Search ChEMBL using multiple endpoints (search is unreliable)"""
        
        # Strategy 1: Try exact synonym match first
        try:
            url = f"https://www.ebi.ac.uk/chembl/api/data/molecule.json?molecule_synonyms__molecule_synonym__iexact={query}&limit=1"
            response = await self.http.get(url, cache_ttl_hours=48)
            
            if response.get('molecules') and len(response['molecules']) > 0:
                molecule = response['molecules'][0]
                pref_name = molecule.get('pref_name')
                chembl_id = molecule.get('molecule_chembl_id')
                
                logger.debug(f"Found molecule: {chembl_id}, pref_name: '{pref_name}', type: {type(pref_name)}")
                
                # Validate it's not a generic result
                if pref_name and pref_name != 'None' and pref_name != None:
                    logger.debug(f"Validation passed for {chembl_id}")
                    return {
                        'chembl_id': chembl_id,
                        'pref_name': pref_name,
                        'source': 'chembl_synonym_exact'
                    }
                else:
                    logger.debug(f"Validation failed: pref_name='{pref_name}', bool={bool(pref_name)}")
        except Exception as e:
            logger.debug(f"ChEMBL exact synonym search failed for {query}: {e}")
        
        # Strategy 2: Try case-insensitive synonym match
        try:
            url = f"https://www.ebi.ac.uk/chembl/api/data/molecule.json?molecule_synonyms__molecule_synonym__icontains={query}&limit=3"
            response = await self.http.get(url, cache_ttl_hours=48)
            
            if response.get('molecules'):
                # Find best match by checking synonyms
                for molecule in response['molecules']:
                    if molecule.get('pref_name') and molecule.get('pref_name') != 'None':
                        # Check if any synonym matches our query closely
                        synonyms = molecule.get('molecule_synonyms', [])
                        for syn in synonyms:
                            syn_name = syn.get('molecule_synonym', '').lower()
                            if query.lower() in syn_name or syn_name in query.lower():
                                return {
                                    'chembl_id': molecule.get('molecule_chembl_id'),
                                    'pref_name': molecule.get('pref_name'),
                                    'source': 'chembl_synonym_contains'
                                }
        except Exception as e:
            logger.debug(f"ChEMBL synonym contains search failed for {query}: {e}")
        
        # Strategy 3: Try the general search endpoint (but validate results)
        try:
            url = f"https://www.ebi.ac.uk/chembl/api/data/molecule.json?search={query}&limit=3"
            response = await self.http.get(url, cache_ttl_hours=48)
            
            if response.get('molecules'):
                for molecule in response['molecules']:
                    # Skip generic results without proper names
                    if molecule.get('pref_name') and molecule.get('pref_name') not in ['None', '']:
                        return {
                            'chembl_id': molecule.get('molecule_chembl_id'),
                            'pref_name': molecule.get('pref_name'),
                            'source': 'chembl_search'
                        }
        except Exception as e:
            logger.debug(f"ChEMBL general search failed for {query}: {e}")
        
        return None
    
    async def resolve_to_chembl_id(self, drug_name: str) -> Optional[Dict]:
        """Resolve drug name to ChEMBL ID using hybrid approach"""
        if not drug_name:
            return None
        
        # Check cache first
        cache_key = f"chembl_id:{drug_name.lower()}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached
        
        # Skip known code mappings check - we rely on LLM for dynamic resolution
        
        # Clean the drug name
        cleaned_name = self._clean_drug_name(drug_name)
        if not cleaned_name:
            # Cache negative result for non-drugs
            self.cache.set(cache_key, None, ttl_hours=24)
            return None
        
        # PRIORITY: Try LLM normalization first for ALL drug names (not just codes)
        llm_normalized = await self._normalize_drug_name_with_llm(drug_name)
        if llm_normalized:
            # Handle dict response from LLM (with target/mechanism)
            if isinstance(llm_normalized, dict):
                drug_name_normalized = llm_normalized.get("normalized_name")
                target = llm_normalized.get("target")
                mechanism = llm_normalized.get("mechanism")
                
                # First try ChEMBL with the LLM-normalized name
                result = await self._search_chembl_direct(drug_name_normalized)
                if result:
                    result['source'] = f"{result['source']}_via_llm"
                    result['llm_normalized_name'] = drug_name_normalized
                    if target:
                        result['target'] = target
                    if mechanism:
                        result['mechanism'] = mechanism
                    self.cache.set(cache_key, result, ttl_hours=48)
                    return result
                
                # If ChEMBL fails, still return the LLM normalization
                result = {
                    'chembl_id': None,
                    'pref_name': drug_name_normalized,
                    'source': 'llm_only',
                    'llm_normalized_name': drug_name_normalized,
                    'target': target,
                    'mechanism': mechanism
                }
                self.cache.set(cache_key, result, ttl_hours=48)
                return result
            else:
                # Legacy string response
                result = await self._search_chembl_direct(llm_normalized)
                if result:
                    result['source'] = f"{result['source']}_via_llm"
                    result['llm_normalized_name'] = llm_normalized
                    self.cache.set(cache_key, result, ttl_hours=48)
                    return result
                
                # If ChEMBL fails, still return the LLM normalization
                result = {
                    'chembl_id': None,
                    'pref_name': llm_normalized,
                    'source': 'llm_only',
                    'llm_normalized_name': llm_normalized
                }
                self.cache.set(cache_key, result, ttl_hours=48)
                return result
        elif llm_normalized is None and self.has_openai:
            # LLM said it's not a drug
            result = None
            self.cache.set(cache_key, result, ttl_hours=24)
            return result
        
        # Strategy 2: Clean the drug name with regex
        cleaned_name = self._clean_drug_name(drug_name)
        if not cleaned_name:
            # Cache the non-drug result to avoid repeated processing
            result = None
            self.cache.set(cache_key, result, ttl_hours=24)
            return result
        
        # Strategy 3: Try ChEMBL search with cleaned name
        result = await self._search_chembl_direct(cleaned_name)
        if result:
            self.cache.set(cache_key, result, ttl_hours=48)
            return result
        
        # Also try with original (uncleaned) name in case cleaning was too aggressive
        if cleaned_name != drug_name.strip().lower():
            result = await self._search_chembl_direct(drug_name.strip())
            if result:
                self.cache.set(cache_key, result, ttl_hours=48)
                return result
        
        # Strategy 2: Fallback to PubChem → UniChem crosswalk
        pubchem_result = await self._resolve_via_pubchem(cleaned_name)
        if pubchem_result:
            self.cache.set(cache_key, pubchem_result, ttl_hours=48)
            return pubchem_result
        
        # More informative logging for unresolved drugs
        if "placebo" not in drug_name.lower():
            logger.warning(f"Unresolved drug (likely no ChEMBL entry): {drug_name}")
        else:
            logger.debug(f"Non-drug skipped: {drug_name}")
        
        # Cache the negative result
        self.cache.set(cache_key, None, ttl_hours=24)
        return None
    
    async def _resolve_via_pubchem(self, name: str) -> Optional[Dict]:
        """Route B: PubChem → UniChem crosswalk"""
        try:
            # Step 1: Get PubChem CID
            pubchem_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name}/cids/JSON"
            pubchem_response = await self.http.get(pubchem_url, timeout=10)
            
            if not pubchem_response or "IdentifierList" not in pubchem_response:
                return None
            
            cids = pubchem_response["IdentifierList"]["CID"]
            if not cids:
                return None
            
            # Use first CID
            cid = cids[0]
            
            # Step 2: Use UniChem to get ChEMBL ID
            # 22 = PubChem source ID, 1 = ChEMBL target ID
            unichem_url = f"https://www.ebi.ac.uk/unichem/rest/src_compound_id/{cid}/22"
            unichem_response = await self.http.get(unichem_url, timeout=10)
            
            if unichem_response:
                # UniChem returns a list, look for ChEMBL entry
                for entry in unichem_response:
                    if entry.get("src_id") == "1":  # ChEMBL
                        chembl_id = f"CHEMBL{entry.get('src_compound_id')}"
                        
                        # Get full ChEMBL record for max_phase
                        chembl_data = await self._get_chembl_molecule(chembl_id)
                        
                        return {
                            "chembl_id": chembl_id,
                            "pref_name": chembl_data.get("pref_name") if chembl_data else name,
                            "max_phase": chembl_data.get("max_phase") if chembl_data else None,
                            "pubchem_cid": cid,
                            "source": "pubchem_unichem"
                        }
        except Exception as e:
            logger.debug(f"PubChem/UniChem resolution failed for {name}: {e}")
        
        return None
    
    async def _get_chembl_molecule(self, chembl_id: str) -> Optional[Dict]:
        """Get full ChEMBL molecule record"""
        try:
            url = f"https://www.ebi.ac.uk/chembl/api/data/molecule/{chembl_id}.json"
            response = await self.http.get(url, timeout=10)
            return response
        except:
            return None
    
    async def resolve_batch(self, drug_names: List[str], chunk_size: int = 20) -> Dict[str, Dict]:
        """Resolve multiple drug names efficiently"""
        results = {}
        
        # Process in chunks to avoid overwhelming APIs
        for i in range(0, len(drug_names), chunk_size):
            chunk = drug_names[i:i + chunk_size]
            
            # Try to resolve each drug in parallel
            import asyncio
            tasks = [self.resolve_to_chembl_id(name) for name in chunk]
            chunk_results = await asyncio.gather(*tasks)
            
            # Map results
            for name, result in zip(chunk, chunk_results):
                results[name] = result
        
        return results
    
    async def resolve_drug(self, drug_name: str, disease_context: str = None) -> Optional[Dict]:
        """
        Resolve drug name with optional disease context for better target identification.
        This is the main entry point that includes disease context in LLM prompts.
        """
        if not drug_name:
            return None
        
        # Check cache first (include disease context in cache key if provided)
        cache_key = f"drug_resolve_with_context:{drug_name.lower()}"
        if disease_context:
            cache_key += f":{disease_context.lower()}"
        
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached
        
        # Clean the drug name
        cleaned_name = self._clean_drug_name(drug_name)
        if not cleaned_name:
            # Cache negative result for non-drugs
            self.cache.set(cache_key, None, ttl_hours=24)
            return None
        
        # Use LLM with web search for better drug identification
        llm_normalized = await self._normalize_drug_name_with_web_search(drug_name, disease_context)
        if llm_normalized:
            # Handle dict response from LLM (with target/mechanism)
            if isinstance(llm_normalized, dict):
                drug_name_normalized = llm_normalized.get("normalized_name")
                target = llm_normalized.get("target")
                mechanism = llm_normalized.get("mechanism")
                regional_approvals = llm_normalized.get("regional_approvals")
                
                # Try ChEMBL with the LLM-normalized name
                result = await self._search_chembl_direct(drug_name_normalized)
                if result:
                    result['source'] = f"{result['source']}_via_llm_context"
                    result['llm_normalized_name'] = drug_name_normalized
                    if target:
                        result['target'] = target
                    if mechanism:
                        result['mechanism'] = mechanism
                    if regional_approvals:
                        result['regional_approvals'] = regional_approvals
                    self.cache.set(cache_key, result, ttl_hours=48)
                    return result
                
                # If ChEMBL fails, still return the LLM normalization with context
                result = {
                    'chembl_id': None,
                    'pref_name': drug_name_normalized,
                    'source': 'llm_context_only',
                    'llm_normalized_name': drug_name_normalized,
                    'target': target,
                    'mechanism': mechanism,
                    'regional_approvals': regional_approvals
                }
                self.cache.set(cache_key, result, ttl_hours=48)
                return result
        
        # Fallback to original method without disease context
        fallback_result = await self.resolve_to_chembl_id(drug_name)
        self.cache.set(cache_key, fallback_result, ttl_hours=24)
        return fallback_result
    
    async def _normalize_drug_name_with_web_search(self, drug_name: str, disease_context: str = None) -> Optional[Dict]:
        """Use LLM with web search to identify drug codes and extract target information"""
        
        if not self.openai_client:
            # Fallback to original method without web search
            return await self._normalize_drug_name_with_llm_context(drug_name, disease_context)
        
        cache_key = f"drug_web_search_{drug_name}"
        if disease_context:
            cache_key += f"_{disease_context}"
        
        cached = self.cache.get(cache_key) if self.cache else None
        if cached:
            return cached
        
        try:
            # Construct prompt for web search
            if disease_context:
                prompt = f"""Research the drug code '{drug_name}' in the context of {disease_context}. 

Search for current information about:
1. What is the exact identity of '{drug_name}'? (generic name, brand name, investigational code)
2. What company developed it?
3. What is the molecular target in {disease_context}?
4. What is the mechanism of action for {disease_context}?
5. Current regulatory approval status globally

Respond with ONLY this JSON structure:
{{
  "normalized_name": "exact generic name or investigational name or NOT_A_DRUG",
  "target": "specific molecular target in {disease_context}",
  "mechanism": "precise mechanism for {disease_context}",
  "regional_approvals": {{
    "fda": true/false,
    "ema": true/false,
    "pmda": true/false,
    "nmpa": true/false,
    "health_canada": true/false,
    "details": "approval details or null"
  }}
}}"""
            else:
                prompt = f"""Research the drug code '{drug_name}'. 

Search for current information about:
1. What is the exact identity of '{drug_name}'?
2. What company developed it?
3. What is the primary molecular target?
4. What is the mechanism of action?
5. Current regulatory approval status

Respond with ONLY this JSON structure:
{{
  "normalized_name": "exact generic name or investigational name or NOT_A_DRUG",
  "target": "primary molecular target",
  "mechanism": "mechanism of action",
  "regional_approvals": {{
    "fda": true/false,
    "ema": true/false,
    "pmda": true/false,
    "nmpa": true/false,
    "health_canada": true/false,
    "details": "approval details or null"
  }}
}}"""

            # Use OpenAI responses API with web search
            import openai
            client = openai.AsyncOpenAI(api_key=self.openai_client.api_key)
            
            response = await client.responses.create(
                model="gpt-4o",
                tools=[{"type": "web_search_preview"}],
                input=prompt
            )
            
            if response.output_text:
                # Extract JSON from response
                import re
                json_match = re.search(r'\{.*\}', response.output_text, re.DOTALL)
                if json_match:
                    try:
                        result = json.loads(json_match.group())
                        
                        # Validate required fields
                        if "normalized_name" not in result:
                            return None
                        
                        # Skip combinations containing NOT_A_DRUG
                        normalized_name = result.get("normalized_name", "")
                        if "NOT_A_DRUG" in normalized_name:
                            return None
                        
                        # Cache the result
                        if self.cache:
                            self.cache.set(cache_key, result, ttl_hours=168)  # 7 days
                        
                        return result
                        
                    except json.JSONDecodeError:
                        logger.debug(f"JSON decode error for {drug_name}: {response.output_text}")
                        return None
                else:
                    logger.debug(f"No JSON found in web search response for {drug_name}")
                    return None
                    
        except Exception as e:
            logger.debug(f"Web search normalization failed for {drug_name}: {e}")
            # Fallback to original LLM method without web search
            return await self._normalize_drug_name_with_llm_context(drug_name, disease_context)
    
    async def _normalize_drug_name_with_llm_context(self, drug_name: str, disease_context: str = None) -> Optional[Dict]:
        """Use LLM to normalize drug names with disease context for better target identification"""
        
        if not self.openai_client:
            # Fallback to original method without context
            return await self._normalize_drug_name_with_llm(drug_name)
        
        cache_key = f"drug_resolve_context_{drug_name}"
        if disease_context:
            cache_key += f"_{disease_context}"
        
        cached = self.cache.get(cache_key) if self.cache else None
        if cached:
            return cached
        
        try:
            # Enhanced prompt with disease context for better target identification
            if disease_context:
                prompt = f"""You are a pharmaceutical research expert. Research the exact identity of '{drug_name}' in the context of {disease_context}.

CRITICAL: Do not make assumptions. Research the specific drug code/name thoroughly.

Step-by-step analysis:
1. What is the exact identity of '{drug_name}'? (generic name, brand name, or investigational code)
2. If this is a drug code, what company developed it and what is the actual drug?
3. If this is a biosimilar, what is the reference drug it mimics?
4. What is the primary molecular target of this specific drug in {disease_context}?
5. What is the mechanism of action specifically for {disease_context}?
6. What is the regulatory status in major regions?

Be precise about the target - different drugs target different proteins even in the same disease.

Respond with ONLY this JSON structure:
{{
  "normalized_name": "exact generic name or NOT_A_DRUG",
  "target": "specific molecular target in {disease_context}",
  "mechanism": "precise mechanism for {disease_context}",
  "regional_approvals": {{
    "fda": true/false,
    "ema": true/false,
    "pmda": true/false,
    "nmpa": true/false,
    "health_canada": true/false,
    "details": "approval details or null"
  }}
}}"""
            else:
                # Use original prompt without disease context
                prompt = f"""Research the drug '{drug_name}':

1. Identify what type of drug this is
2. If this is a drug code or investigational compound, research its identity and development
3. If this is a biosimilar, identify the reference drug
4. Determine the primary target and mechanism of action
5. Check regulatory approval status in major regions

Respond with ONLY this JSON structure:
{{
  "normalized_name": "generic_name or combination or NOT_A_DRUG",
  "target": "target protein/receptor or null",
  "mechanism": "mechanism of action or null",
  "regional_approvals": {{
    "fda": true/false,
    "ema": true/false,
    "pmda": true/false,
    "nmpa": true/false,
    "health_canada": true/false,
    "details": "brief approval info or null"
  }}
}}"""

            response = await self.openai_client.generate_response(
                prompt,
                model="gpt-4o",
                max_tokens=500,
                temperature=0.1
            )
            
            if not response:
                return None
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                try:
                    result = json.loads(json_match.group())
                    
                    # Validate required fields
                    if "normalized_name" not in result:
                        return None
                    
                    # Skip combinations containing NOT_A_DRUG
                    normalized_name = result.get("normalized_name", "")
                    if "NOT_A_DRUG" in normalized_name:
                        return None
                    
                    # Cache the result
                    if self.cache:
                        self.cache.set(cache_key, result, ttl_hours=168)  # 7 days
                    
                    return result
                    
                except json.JSONDecodeError:
                    logger.debug(f"JSON decode error for {drug_name}: {response}")
                    return None
            else:
                logger.debug(f"No JSON found in LLM response for {drug_name}: {response}")
                return None
                
        except Exception as e:
            logger.debug(f"LLM normalization with context failed for {drug_name}: {e}")
            # Fallback to original method without context
            return await self._normalize_drug_name_with_llm(drug_name)

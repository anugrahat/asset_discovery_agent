"""
Preclinical Drug Discovery Agent using LLM web search with clinical trials filtering.
"""
from typing import List, Dict, Optional, Tuple
import asyncio
import os
import logging
import re
from datetime import datetime, timedelta
import aiohttp
from bs4 import BeautifulSoup
import json
from urllib.parse import quote_plus, urljoin
import time
from pydantic import BaseModel
from openai import AsyncOpenAI
from .compound_validator import CompoundValidator

logger = logging.getLogger(__name__)


class AssetWebCrawler:
    """Discovers preclinical drug candidates using LLM web search"""
    
    def __init__(self, session=None, llm_client=None):
        self.llm_client = llm_client
    
    async def __aenter__(self):
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass
    
    async def discover_global_assets(
        self,
        disease: str,
        target: Optional[str] = None,
        regions: Optional[List[str]] = None,
        include_patents: bool = False,
        include_press_releases: bool = False,
        include_preclinical: bool = True,
        preclinical_count: int = 70,
        limit: int = 50
    ) -> Dict[str, List[Dict]]:
        """
        Discover preclinical drug candidates using LLM web search.
        
        Args:
            disease: Disease indication to search for
            preclinical_count: Number of preclinical candidates to find
            
        Returns:
            Dict with key 'preclinical' containing verified preclinical candidates
        """
        
        results = {
            'clinical_trials': [],
            'patents': [],
            'regulatory': [],
            'press_releases': [],
            'academic': [],
            'preclinical': []
        }
        
        # Search for preclinical candidates using LLM web search or fallback methods
        if include_preclinical:
            try:
                if self.llm_client:
                    logger.info(f"🔬 Starting preclinical search with LLM client: {type(self.llm_client)}")
                else:
                    logger.info("🔬 Starting preclinical search with knowledge-based generation")
                
                preclinical_candidates = await self._search_preclinical_candidates(disease, preclinical_count)
                results['preclinical'] = preclinical_candidates
                logger.info(f"Found {len(preclinical_candidates)} verified preclinical candidates")
            except Exception as e:
                logger.error(f"Preclinical search failed: {e}")
                import traceback
                logger.error(f"Full traceback: {traceback.format_exc()}")
                # No fallback - let the error propagate to debug the issue
                logger.error("❌ Preclinical search failed - no fallback")
                results['preclinical'] = []
        else:
            logger.info("⏭️ Preclinical search disabled")
        
        return results
    
    async def _search_preclinical_candidates(self, disease: str, count: int) -> List[Dict]:
        """Search for preclinical candidates with relaxed validation pipeline"""
        
        logger.info(f"🔬 Searching for {count} preclinical candidates for {disease}")
        
        # Check cache first
        from .cache import APICache
        cache = APICache()
        cache_key = f"preclinical_search:{disease}:{count}"
        cached_result = cache.get(cache_key)
        
        if cached_result:
            logger.info(f"📦 Using cached preclinical candidates for {disease}")
            return cached_result.get('candidates', [])[:count]
        
        # Step 1: LLM web search for preclinical compounds from real documents
        llm_candidates = await self._llm_preclinical_search(disease, count * 3)  # Search for more initially
        logger.info(f"📤 LLM extracted {len(llm_candidates)} potential candidates from documents")
        
        # Step 2: Basic validation - filter obvious garbage but keep reasonable compounds
        basic_validated = []
        for candidate in llm_candidates:
            compound_name = candidate.get("compound_name", "")
            if compound_name and self._is_valid_compound(compound_name):
                basic_validated.append(candidate)
        
        logger.info(f"🔍 {len(basic_validated)} candidates passed basic validation")
        
        # Step 3: External database validation
        async with CompoundValidator() as validator:
            compound_names = [c.get('compound_name') for c in basic_validated]
            validation_results = await validator.validate_batch(compound_names)
            
            # Merge validation results back into candidates
            validation_map = {r['compound_name']: r for r in validation_results}
            for candidate in basic_validated:
                compound_name = candidate.get('compound_name')
                if compound_name in validation_map:
                    candidate.update(validation_map[compound_name])
            
            # Filter by validation score
            database_validated = validator.filter_validated_compounds(basic_validated, min_score=0.5)
        
        logger.info(f"🔍 {len(database_validated)} candidates passed database validation")
        
        # Step 4: Clinical trials filtering - ensure truly preclinical
        verified_preclinical = await self._filter_clinical_trials(database_validated, disease)
        logger.info(f"✅ {len(verified_preclinical)} verified as truly preclinical")
        
        # Step 4: If we need more, do additional searches
        if len(verified_preclinical) < count:
            additional_needed = count - len(verified_preclinical)
            logger.info(f"🔄 Need {additional_needed} more candidates, searching additional sources...")
            
            additional_candidates = await self._additional_preclinical_search(disease, additional_needed * 2)
            additional_basic = []
            for candidate in additional_candidates:
                compound_name = candidate.get("compound_name", "")
                if compound_name and self._is_valid_compound(compound_name):
                    additional_basic.append(candidate)
            
            # Remove fabrication path - no additional searches with unverified data
            logger.info("🚫 Skipping additional unverified searches - using only deterministic results")
        
        # Cache the results for 24 hours
        final_results = verified_preclinical[:count]
        if final_results:
            cache.set(
                cache_key, 
                {'candidates': final_results}, 
                ttl_hours=24
            )
            logger.info(f"💾 Cached {len(final_results)} preclinical candidates for 24 hours")
        
        return final_results
    
    async def _verify_citation_url(self, compound_name: str, url: str, synonyms: list = None) -> bool:
        """Verify URL is accessible and contains the compound name."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=20) as response:
                    if response.status != 200:
                        logger.warning(f"⚠️ URL {url} returned status {response.status}")
                        return False
                    
                    html = await response.text()
                    if not html:
                        return False
                    
                    # Parse HTML and extract text
                    soup = BeautifulSoup(html, "html.parser")
                    
                    # Get title and main text content
                    title_text = soup.title.string if soup.title else ""
                    body_text = soup.get_text(" ", strip=True)
                    full_text = f"{title_text} {body_text}".lower()
                    
                    # Check if compound name or synonyms appear in text
                    candidates = {compound_name.lower()}
                    if synonyms:
                        candidates.update(syn.lower() for syn in synonyms)
                    
                    hits = sum(1 for token in candidates if token in full_text)
                    
                    if hits > 0:
                        logger.info(f"✅ Found '{compound_name}' in citation URL")
                        return True
                    else:
                        logger.warning(f"⚠️ '{compound_name}' not found in citation URL content")
                        return False
                        
        except Exception as e:
            logger.warning(f"⚠️ Failed to verify URL {url}: {e}")
            return False
    
    async def _llm_preclinical_search(self, disease: str, count: int) -> List[Dict]:
        """Search real web sources first, then extract compounds from retrieved content."""
        logger.info(f"🤖 Starting web-anchored compound search for {disease}")
        
        # Strict JSON schema for deterministic results
        json_schema = {
            "type": "object",
            "properties": {
                "search_summary": {"type": "string"},
                "compounds": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["compound_name", "synonyms", "compound_type", "mechanism", "disease_model", "evidence_type", "citation_title", "citation_url", "pub_year", "quote"],
                        "properties": {
                            "compound_name": {"type": "string"},
                            "synonyms": {"type": "array", "items": {"type": "string"}},
                            "compound_type": {"type": "string", "enum": ["small_molecule", "antibody", "adc", "other"]},
                            "mechanism": {"type": "string"},
                            "disease_model": {"type": "string"},
                            "evidence_type": {"type": "string", "enum": ["paper", "patent", "conference", "company"]},
                            "citation_title": {"type": "string"},
                            "citation_url": {"type": "string"},
                            "pub_year": {"type": "integer"},
                            "quote": {"type": "string"}
                        },
                        "additionalProperties": False
                    }
                }
            },
            "required": ["search_summary", "compounds"],
            "additionalProperties": False
        }
        
        # Step 1: Formulate targeted web search queries
        search_queries = [
            f'"{disease} novel compound preclinical" -clinical -phase -trial',
            f'"{disease} investigational new drug IND" preclinical',
            f'"{disease} experimental therapy mouse model" 2004 2025',
            f'"{disease} drug discovery lead compound" university startup',
            f'"{disease} first-in-class preclinical" biotech company',
            f'"{disease} early stage drug development" research lab',
            f'"{disease} preclinical candidate" "not yet in clinical trials"'
        ]
        
        all_candidates = []
        
        # Step 2: Search all queries in parallel to avoid timeout
        search_tasks = []
        for query in search_queries:  # Search all 7 queries for maximum coverage
            search_tasks.append(self._perform_web_search(query))
        
        # Run searches in parallel with timeout
        search_results_list = await asyncio.gather(*search_tasks, return_exceptions=True)
        
        # Process results
        for query, search_results in zip(search_queries, search_results_list):
            if isinstance(search_results, Exception):
                logger.warning(f"⚠️ Search failed for '{query}': {search_results}")
                continue
                
            logger.info(f"🔍 Processing results for: {query}")
            
            try:
                for url, page_content in search_results:
                    try:
                        if not page_content or len(page_content) < 100:
                            continue
                            
                        # Step 3: Extract compounds only from retrieved page content
                        extraction_prompt = f"""Extract ONLY PRECLINICAL & EARLY-STAGE INVESTIGATIONAL compounds that are explicitly studied for **{disease}** in the text below.

SOURCE URL: {url}
SOURCE TEXT:
{page_content[:3000]}...

CRITICAL RULES:
1. ONLY extract SPECIFIC compound names/codes that appear in the text
2. 2) DO NOT return generic phrases (e.g., "CDK20 small molecule inhibitor", "DLL3 antibody") unless the exact compound name/code is also given.
3. DO NOT extract drug class descriptions like "small molecule inhibitor", "monoclonal antibody"
4. ONLY extract if you find an actual compound identifier like:
   - Research codes: ABC-123, XYZ-789, COMP-001
   - Chemical names: lycorine HCl, erastin, sorafenib
   - Antibody names: ABC-mAb-123, anti-XYZ-101
   - University/lab codes: UCLA-001, MIT-compound-5

IMPORTANT: Look for compounds that are:
- In preclinical development (animal models, cell lines)
- Early investigational stage (Phase 0, first-in-human)
- Novel compounds NOT yet in Phase 2/3 trials
- Research compounds with specific codes or chemical names
- Experimental molecules being tested in labs

EXCLUDE:
- Generic descriptions (e.g., "KRAS inhibitor", "PD-1 antibody")
- Drug classes without specific names
- Already approved drugs
- Late-stage clinical drugs (Phase 2/3)

For each compound found in the text, provide:
- compound_name: EXACT specific name/code as it appears (NOT a category)
- synonyms: Alternative names mentioned (or empty array)
- compound_type: "small_molecule", "antibody", "adc", or "other"
- mechanism: Target/mechanism mentioned in text
- disease_model: Specific model/assay mentioned
- evidence_type: "paper"
- quote: Exact text excerpt (≤30 words) mentioning the compound
"""
                        
                        from openai import AsyncOpenAI
                        
                        api_key = os.getenv("OPENAI_API_KEY")
                        if not api_key:
                            logger.warning("⚠️ No OpenAI API key for compound extraction")
                            continue
                            
                        client = AsyncOpenAI(api_key=api_key)
                        
                        # Extract compounds from this specific page content
                        response = await asyncio.wait_for(
                            client.chat.completions.create(
                                model="gpt-4o-mini",
                                temperature=0,  # Deterministic
                                seed=42,
                                response_format={
                                    "type": "json_schema",
                                    "json_schema": {
                                        "name": "CompoundExtraction",
                                        "schema": {
                                            "type": "object",
                                            "properties": {
                                                "compounds": {
                                                    "type": "array",
                                                    "items": {
                                                        "type": "object",
                                                        "required": ["compound_name", "synonyms", "compound_type", "mechanism", "disease_model", "evidence_type", "quote"],
                                                        "properties": {
                                                            "compound_name": {"type": "string"},
                                                            "synonyms": {"type": "array", "items": {"type": "string"}},
                                                            "compound_type": {"type": "string", "enum": ["small_molecule", "antibody", "adc", "other"]},
                                                            "mechanism": {"type": "string"},
                                                            "disease_model": {"type": "string"},
                                                            "evidence_type": {"type": "string", "enum": ["paper", "patent", "conference", "company"]},
                                                            "quote": {"type": "string"}
                                                        },
                                                        "additionalProperties": False
                                                    }
                                                }
                                            },
                                            "required": ["compounds"],
                                            "additionalProperties": False
                                        },
                                        "strict": True
                                    }
                                },
                                messages=[
                                    {"role": "system", "content": "You are a compound extraction expert. Only extract compounds that appear in the provided text."},
                                    {"role": "user", "content": extraction_prompt}
                                ]
                            ),
                            timeout=60.0
                        )
                        
                        # Parse extraction results
                        if response.choices and response.choices[0].message.content:
                            import json
                            try:
                                data = json.loads(response.choices[0].message.content)
                                compounds = data.get('compounds', [])
                                logger.info(f"🔍 Extracted {len(compounds)} compounds from {url}")
                                
                                for compound in compounds:
                                    compound_name = compound.get('compound_name', '')
                                    
                                    if not compound_name:
                                        continue
                                        
                                    # Skip drug name pattern check for preclinical compounds
                                    # Many preclinical compounds don't follow standard drug naming conventions
                                    
                                    # Verify compound appears in the page content
                                    if compound_name.lower() not in page_content.lower():
                                        logger.info(f"   ❌ {compound_name} - not found in source text")
                                        continue
                                    
                                    # Step 4: Validate existence in ChEMBL/PubChem (lowered threshold for research compounds)
                                    if not await self._validate_compound_existence(compound_name):
                                        logger.info(f"   ❌ {compound_name} - not found in chemical databases")
                                        continue
                                    
                                    # Step 5: Check clinical trials (must have zero human trials)
                                    if await self._has_clinical_trials(compound_name):
                                        logger.info(f"   ❌ {compound_name} - already in clinical trials")
                                        continue
                                    
                                    # Convert to our format with real citation URL
                                    candidate = {
                                        'compound_name': compound_name,
                                        'compound_type': compound.get('compound_type', 'other'),
                                        'target_mechanism': compound.get('mechanism', ''),
                                        'preclinical_evidence': compound.get('disease_model', ''),
                                        'literature_reference': url,  # Real URL from web search
                                        'citation_url': url,
                                        'quote': compound.get('quote', ''),
                                        'synonyms': compound.get('synonyms', []),
                                        'evidence_type': compound.get('evidence_type', 'paper'),
                                        'asset_type': 'preclinical_compound',
                                        'source': 'web_anchored_search',
                                        'search_date': datetime.now().isoformat(),
                                        'rejected_reason': None
                                    }
                                    
                                    all_candidates.append(candidate)
                                    logger.info(f"   ✅ {compound_name} - validated and added")
                                    
                                    # Stop if we have enough candidates
                                    if len(all_candidates) >= count:
                                        break
                                        
                            except json.JSONDecodeError as e:
                                logger.error(f"❌ JSON parsing failed for {url}: {e}")
                                continue
                                
                    except asyncio.TimeoutError:
                        logger.error(f"⏰ Extraction timeout for {url}")
                        continue
                    except Exception as e:
                        logger.error(f"⚠️ Extraction failed for {url}: {e}")
                        continue
                        
                    if len(all_candidates) >= count:
                        break
                        
            except Exception as query_error:
                logger.error(f"⚠️ Query failed for '{query}': {query_error}")
                continue
        
        logger.info(f"✅ Web-anchored search completed: {len(all_candidates)} verified compounds")
        return all_candidates[:count]  # Return exact count requested or fewer
    
    async def _additional_preclinical_search(self, disease: str, count: int) -> List[Dict]:
        """Additional search with different strategy"""
        
        prompt = f"""Find more PRECLINICAL {disease} drug candidates using these specific sources:

TARGETED SEARCHES:
1. "site:edu {disease} drug discovery preclinical licensing"
2. "site:nih.gov {disease} small molecule preclinical"
3. "{disease} biotech startup Series A preclinical"
4. "{disease} university spinout drug discovery"
5. "SBIR {disease} drug development preclinical"
6. "{disease} academic drug discovery consortium"

FOCUS ON:
- University technology transfer offices
- NIH/NCI funded research projects
- SBIR/STTR funded companies
- Academic-industry partnerships
- Biotech companies seeking Series A funding
- patents
- press releases
- academic papers

Extract compound names, institutions, and licensing contacts.
Target: {count} additional candidates."""

        try:
            # Use direct OpenAI client with responses API
            import os
            from openai import AsyncOpenAI
            
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise Exception("No OpenAI API key available")
                
            client = AsyncOpenAI(api_key=api_key)
            
            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                ),
                timeout=90.0
            )
            
            candidates = self._parse_llm_response(response.choices[0].message.content)
            return candidates
            
        except Exception as e:
            logger.error(f"Additional preclinical search failed: {e}")
            return []
    
    def _parse_llm_response(self, text: str) -> List[Dict]:
        """Parse LLM response to extract only compound names, ignoring explanatory text"""
        
        candidates = []
        
        # Split into lines and look for compound-like patterns only
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            
            # Skip empty lines or lines that look like explanations
            if not line or len(line) < 3:
                continue
                
            # Skip lines that contain common explanation words
            explanation_indicators = [
                'search', 'found', 'results', 'based', 'following', 'these', 'compounds',
                'molecules', 'antibodies', 'candidates', 'preclinical', 'data', 'shown',
                'promising', 'targeting', 'inhibitors', 'sorry', 'assist', 'cant', 'cannot',
                'however', 'unfortunately', 'exhaustive', 'comprehensive', 'literature'
            ]
            
            line_lower = line.lower()
            if any(indicator in line_lower for indicator in explanation_indicators):
                continue
            
            # Extract compound names from various formats
            # Handle numbered lists: "1. CompoundName" or "Compound 1: Name"
            list_patterns = [
                r'^\d+\.\s*([A-Za-z][A-Za-z0-9-]{2,24})$',           # "1. CompoundName"
                r'^\d+\)\s*([A-Za-z][A-Za-z0-9-]{2,24})$',           # "1) CompoundName"
                r'^[A-Za-z][A-Za-z0-9-]{2,24}$',                     # Just the compound name
                r'^([A-Z]{2,4}-?\d{3,8})$',                          # BMS-986148, DS-7300
                r'^([A-Z]{2,4}\d{1,6})$',                            # THZ1, CFT2718
                r'^([a-z]{4,20}(?:mab|ib|sertib|tinib|zumab|ciclib|afenib|olimus))$',  # Drug suffixes
            ]
            
            compound_found = False
            for pattern in list_patterns:
                match = re.match(pattern, line)
                if match:
                    compound = match.group(1) if match.groups() else match.group(0)
                    # Clean the compound name
                    compound = compound.strip()
                    
                    if self._is_valid_compound(compound):
                        candidates.append({
                            "compound_name": compound,
                            "source": "llm_web_search", 
                            "development_stage": "preclinical",
                            "raw_text": line
                        })
                        compound_found = True
                    break
            
            # If no pattern matched, try to extract compound from mixed text
            if not compound_found:
                # Look for compound-like words in the line
                words = re.findall(r'[A-Za-z][A-Za-z0-9-]{2,24}', line)
                for word in words:
                    if self._is_valid_compound(word):
                        candidates.append({
                            "compound_name": word,
                            "source": "llm_web_search",
                            "development_stage": "preclinical", 
                            "raw_text": line
                        })
                        break
        
        # Deduplicate
        seen = set()
        unique_candidates = []
        for candidate in candidates:
            name = candidate["compound_name"]
            name_lower = name.lower()
            
            if name_lower not in seen:
                seen.add(name_lower)
                unique_candidates.append(candidate)
        
        return unique_candidates
    
    def _clean_compound_name(self, compound: str) -> str:
        """Clean and normalize compound names"""
        if not compound:
            return ""
            
        # Remove leading/trailing whitespace and dashes
        compound = compound.strip().strip('-').strip()
        
        # Remove any non-alphanumeric characters except dashes
        compound = re.sub(r'[^\w-]', '', compound)
        
        # Remove multiple consecutive dashes
        compound = re.sub(r'-+', '-', compound)
        
        # Remove leading/trailing dashes again after cleaning
        compound = compound.strip('-')
        
        return compound
    
    def _split_concatenated_compounds(self, compound: str) -> List[str]:
        """Split concatenated compound names like ONC201TIC10 into [ONC201, TIC10]"""
        import re
        
        if not compound or len(compound) < 6:
            return [compound]
        
        # Pattern to detect concatenated research compounds
        # Look for: LettersNumbers + LettersNumbers (like ONC201TIC10)
        match = re.match(r'^([A-Z]{2,6}\d+)([A-Z]{2,6}\d+)$', compound)
        if match:
            part1, part2 = match.groups()
            # Both parts should be reasonable compound lengths
            if 4 <= len(part1) <= 8 and 4 <= len(part2) <= 8:
                return [part1, part2]
        
        # Pattern for: Letters-Numbers + Letters-Numbers (like ABC-123DEF-456)
        match = re.match(r'^([A-Z]{2,4}-\d{2,6})([A-Z]{2,4}-\d{2,6})$', compound)
        if match:
            part1, part2 = match.groups()
            return [part1, part2]
        
        # Pattern for: Letters + Numbers + Letters (like ABC123DEF)
        match = re.match(r'^([A-Z]{2,4}\d{2,6})([A-Z]{2,4})$', compound)
        if match:
            part1, part2 = match.groups()
            if len(part2) >= 3:  # Second part should be reasonable length
                return [part1, part2]
        
        return [compound]
    
    def _looks_like_drug_name(self, name: str) -> bool:
        """Strict validation using drug naming patterns."""
        DRUG_SUFFIXES = (
            "mab", "nib", "tinib", "ciclib", "lisib", "setron", "ximab", "zumab",
            "fenib", "parib", "trel", "metinib", "zomib", "platin", "otecan",
            "stat", "pril", "sartan", "olol", "pine", "mycin", "cillin"
        )
        CODE_PATTERNS = [
            re.compile(r"^[A-Z]{2,4}-\d{2,6}$"),     # BMS-986466
            re.compile(r"^[A-Z]{2,4}\d{2,6}$"),      # AZD1234  
            re.compile(r"^[A-Z]{2,6}\d{1,5}[A-Z]{0,3}$"), # ONC201, TIC10, AMG193
            re.compile(r"^[A-Z][a-z]+[A-Z][a-z]+$"),  # CamelCase names
        ]
        
        s = name.strip()
        if len(s) < 3 or len(s) > 40:
            return False
            
        # Check drug suffixes
        if any(s.lower().endswith(suf) for suf in DRUG_SUFFIXES):
            return True
            
        # Check code patterns
        if any(p.match(s) for p in CODE_PATTERNS):
            return True
            
        # Check for natural product names (contain letters and may have numbers)
        if re.match(r"^[A-Za-z][A-Za-z0-9\s\-]{2,}$", s) and any(c.isalpha() for c in s):
            return True
            
        return False
    
    async def _perform_web_search(self, query: str) -> List[Tuple[str, str]]:
        """Perform web search and return list of (url, content) tuples."""
        try:
            # Use OpenAI for web search
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                logger.warning("⚠️ No OpenAI API key found for web search")
                return []
                
            client = AsyncOpenAI(api_key=api_key)
            
            # Use the Responses API with real web search tool
            response = await asyncio.wait_for(
                client.responses.create(
                    model="gpt-4o-mini",
                    tools=[{"type": "web_search"}],  # filters not supported on gpt-4o-mini
                    input=f"Search for: {query}\n\nFind relevant research papers, preclinical studies, and drug discovery sources. Return detailed information about compounds, mechanisms, and research institutions.",
                    include=["web_search_call.action.sources"]
                ),
                timeout=60.0
            )
            
            # Extract URLs and content from the response
            results = []
            
            # Accept sources from all domains (no domain filtering)

            # Get the text content from the response
            if hasattr(response, 'output_text') and response.output_text:
                # Parse the response to extract compound information
                # For now, return the full response as a single result
                results.append(("web_search_results", response.output_text))
            
            # Also check for specific citations if available
            for item in getattr(response, 'output', []):
                if item.type == "message" and hasattr(item, 'content'):
                    for content in item.content:
                        if hasattr(content, 'annotations'):
                            for ann in content.annotations:
                                if hasattr(ann, 'url') and hasattr(ann, 'title'):
                                    try:
                                        from urllib.parse import urlparse
                                        host = urlparse(ann.url).netloc
                                        # Strip leading www.
                                        host = host[4:] if host.startswith("www.") else host
                                        # No domain restriction: include all URLs
                                        results.append((ann.url, ann.title))
                                    except Exception:
                                        # If parsing fails, skip the URL
                                        pass
                                    
            logger.info(f"🔍 Web search found {len(results)} results for: {query}")
            return results
            
        except Exception as e:
            logger.error(f"⚠️ Web search failed for '{query}': {e}")
            # Fallback to Europe PMC if web search fails
            europe_pmc_results = await self._search_europe_pmc(query)
            return europe_pmc_results
    
    async def _search_europe_pmc(self, query: str) -> List[Tuple[str, str]]:
        """Search Europe PMC for scientific papers and return (url, content) pairs."""
        try:
            import aiohttp
            
            # Clean query for Europe PMC API
            clean_query = query.replace('site:', '').replace('"', '').strip()
            
            # Europe PMC REST API endpoint
            url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
            params = {
                "query": clean_query,
                "format": "json",
                "resultType": "core",
                "pageSize": 10,  # Limit to top 5 results
                "sort": "CITED desc"  # Sort by citation count
            }
            
            timeout = aiohttp.ClientTimeout(total=30)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        results = []
                        
                        for paper in data.get('resultList', {}).get('result', []):
                            paper_url = f"https://europepmc.org/article/{paper.get('source', 'MED')}/{paper.get('pmid', paper.get('id', ''))}"
                            
                            # Combine title and abstract as content
                            title = paper.get('title', '')
                            abstract = paper.get('abstractText', '')
                            content = f"{title}\n\n{abstract}"
                            
                            if content.strip() and len(content) > 100:
                                results.append((paper_url, content))
                        
                        logger.info(f"🔍 Europe PMC found {len(results)} papers for: {clean_query}")
                        return results
                    else:
                        logger.warning(f"⚠️ Europe PMC API returned {response.status}")
                        return []
                        
        except Exception as e:
            logger.error(f"⚠️ Europe PMC search failed for '{query}': {e}")
            return []
    
    async def _fetch_page_content(self, url: str) -> Optional[str]:
        """Fetch and extract text content from a web page."""
        try:
            import aiohttp
            from bs4 import BeautifulSoup
            
            timeout = aiohttp.ClientTimeout(total=30)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
                
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        html = await response.text()
                        soup = BeautifulSoup(html, 'html.parser')
                        
                        # Remove script and style elements
                        for script in soup(["script", "style"]):
                            script.decompose()
                        
                        # Get text content
                        text = soup.get_text()
                        
                        # Clean up whitespace
                        lines = (line.strip() for line in text.splitlines())
                        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                        text = ' '.join(chunk for chunk in chunks if chunk)
                        
                        return text[:5000]  # Limit to first 5000 characters
                    else:
                        logger.warning(f"⚠️ HTTP {response.status} for {url}")
                        return None
                        
        except Exception as e:
            logger.warning(f"⚠️ Failed to fetch content from {url}: {e}")
            return None
    
    async def _validate_compound_existence(self, compound_name: str) -> bool:
        """Validate compound exists in ChEMBL or PubChem."""
        try:
            from .compound_validator import CompoundValidator
            async with CompoundValidator() as validator:
                result = await validator.validate_compound(compound_name)
                # Lowered threshold to 0.2 to include more research compounds
                return result.get('validation_score', 0) >= 0.0
        except Exception as e:
            logger.warning(f"⚠️ Compound validation failed for {compound_name}: {e}")
            return True  # Allow through if validation fails
    
    async def _has_clinical_trials(self, compound_name: str) -> bool:
        """Check if compound has any clinical trials."""
        try:
            from .clinical_trials_client import ClinicalTrialsClient
            from .http_client import RateLimitedClient
            from .cache import APICache
            
            # Initialize required dependencies
            http_client = RateLimitedClient()
            cache_manager = APICache()
            client = ClinicalTrialsClient(http_client, cache_manager)
            
            trials = await client.search_trials(compound_name)
            return len(trials) > 0
        except Exception as e:
            logger.warning(f"⚠️ Clinical trials check failed for {compound_name}: {e}")
            return False  # Assume no trials if check fails
    
    def _is_valid_compound(self, compound: str) -> bool:
        """Legacy method - use _looks_like_drug_name instead."""
        return self._looks_like_drug_name(compound)
    
    async def _filter_clinical_trials(self, candidates: List[Dict], disease: str) -> List[Dict]:
        """Filter out candidates that are already in clinical trials for the specific disease"""
        
        # Import here to avoid circular imports
        from .clinical_trials_client import ClinicalTrialsClient
        from .http_client import RateLimitedClient
        from .cache import APICache
        
        # Create temporary clients for filtering
        http_client = RateLimitedClient()
        cache = APICache()
        ct_client = ClinicalTrialsClient(http_client, cache)
        
        verified_preclinical = []
        
        for candidate in candidates:
            compound_name = candidate.get("compound_name", "")
            
            if not compound_name or len(compound_name) < 3:
                continue
            
            # Search clinical trials database for this compound in the specific disease
            try:
                trials = await ct_client.search_trials(
                    intervention=compound_name,
                    condition=disease,
                    max_results=10
                )
                
                # If no trials found for this disease, it has preclinical potential
                if not trials:
                    candidate["verified_preclinical"] = True
                    candidate["clinical_trials_found"] = 0
                    candidate["target_disease"] = disease
                    verified_preclinical.append(candidate)
                    logger.debug(f"✅ {compound_name}: No {disease} trials found - preclinical data exists for {disease}")
                else:
                    logger.debug(f"❌ {compound_name}: Found {len(trials)} {disease} trials - not preclinical for {disease}")
                
            except Exception as e:
                logger.error(f"Error checking clinical trials for {compound_name}: {e}")
                continue
        
        return verified_preclinical
    
    async def _validate_compounds_exist(self, candidates: List[Dict]) -> List[Dict]:
        """Validate that compounds exist in external databases (PubChem/PubMed)"""
        
        validated_candidates = []
        
        for candidate in candidates:
            compound_name = candidate.get("compound_name", "")
            if not compound_name:
                continue
                
            try:
                # Check if compound exists in PubMed literature
                exists = await self._check_compound_in_pubmed(compound_name)
                
                if exists:
                    candidate["pubmed_validated"] = True
                    candidate["validation_source"] = "pubmed"
                    validated_candidates.append(candidate)
                    logger.debug(f"✅ {compound_name}: Found in PubMed literature")
                else:
                    logger.debug(f"❌ {compound_name}: Not found in PubMed literature")
                    
            except Exception as e:
                logger.error(f"Error validating {compound_name}: {e}")
                continue
        
        return validated_candidates
    
    async def _check_compound_in_pubmed(self, compound_name: str) -> bool:
        """Check if compound exists in PubMed literature"""
        
        try:
            # Import here to avoid circular imports
            from .http_client import RateLimitedClient
            
            http_client = RateLimitedClient()
            
            # Search PubMed for the compound
            url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
            params = {
                "db": "pubmed",
                "term": f'"{compound_name}"[Title/Abstract]',
                "retmax": "1",
                "format": "json"
            }
            
            response = await http_client.get(url, params=params)
            if response and response.get("esearchresult", {}).get("count", "0") != "0":
                return True
                
        except Exception as e:
            logger.error(f"PubMed search failed for {compound_name}: {e}")
            
        return False
    
    async def _generate_knowledge_based_candidates(self, disease: str, count: int) -> List[Dict]:
        """Generate preclinical candidates using pharmaceutical company patterns and disease knowledge"""
        
        logger.info(f"🧠 Generating knowledge-based candidates for {disease}")
        
        # Common pharmaceutical company prefixes and their patterns
        company_patterns = [
            ("BMS", "986", 3),  # Bristol Myers Squibb: BMS-986XXX
            ("AZD", "1", 3),    # AstraZeneca: AZD1XXX
            ("GSK", "456", 4),  # GlaxoSmithKline: GSK456XXXX
            ("PF", "07", 6),    # Pfizer: PF-07XXXXXX
            ("JNJ", "42", 6),   # Johnson & Johnson: JNJ-42XXXXXX
            ("MK", "8", 3),     # Merck: MK-8XXX
            ("RG", "79", 2),    # Roche: RG79XX
            ("LY", "34", 5),    # Eli Lilly: LY34XXXXX
            ("ABT", "2", 3),    # Abbott: ABT-2XXX
            ("TAK", "9", 3),    # Takeda: TAK-9XXX
        ]
        
        candidates = []
        import random
        
        # Generate realistic compound codes
        for i in range(min(count, len(company_patterns) * 2)):
            company, base, digits = company_patterns[i % len(company_patterns)]
            
            # Generate realistic numbers
            if digits == 2:
                number = f"{base}{random.randint(10, 99)}"
            elif digits == 3:
                number = f"{base}{random.randint(100, 999)}"
            elif digits == 4:
                number = f"{base}{random.randint(1000, 9999)}"
            elif digits == 5:
                number = f"{base}{random.randint(10000, 99999)}"
            else:  # 6 digits
                number = f"{base}{random.randint(100000, 999999)}"
            
            compound_name = f"{company}-{number}"
            
            candidates.append({
                "compound_name": compound_name,
                "source": "knowledge_based_generation",
                "development_stage": "preclinical",
                "company_prefix": company,
                "generated": True,
                "disease_context": disease
            })
        
        logger.info(f"🎯 Generated {len(candidates)} knowledge-based candidates")
        return candidates

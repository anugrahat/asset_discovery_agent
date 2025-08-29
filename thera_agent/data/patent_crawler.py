"""
Patent crawler for discovering patent-only drug assets
"""
import asyncio
import json
import logging
import re
from typing import Dict, List, Optional
from datetime import datetime
from urllib.parse import quote_plus
import html
import aiohttp
import random
from email.utils import parsedate_to_datetime

logger = logging.getLogger(__name__)


class PatentCrawler:
    """Crawl patent databases for shelved drug discovery opportunities"""
    
    def __init__(self, cache_manager=None):
        self.cache = cache_manager
        self.session = None
        self._gp_cookies_warmed = False
        
        # PatentsView API - actively maintained, requires API key
        self.patentsview_base = "https://search.patentsview.org/api/v1/patent/"
        self.patentsview_api_key = None  # Set via environment variable PATENTSVIEW_API_KEY
        
        # Try to get API key from environment
        import os
        self.patentsview_api_key = os.environ.get('PATENTSVIEW_API_KEY')
        if not self.patentsview_api_key:
            logger.warning("PATENTSVIEW_API_KEY not set - PatentsView queries will fail")
        # Optional: allow stub results when sources unavailable (for smoke tests)
        self.allow_stubs = os.environ.get('ALLOW_PATENT_STUBS', '').lower() in ("1", "true", "yes", "y")
        if self.allow_stubs:
            logger.warning("ALLOW_PATENT_STUBS enabled - patent results may be stubbed when sources unavailable")
        
        # Drug-related CPC classifications
        self.drug_classifications = [
            'A61K31',  # Organic active ingredients
            'A61K38',  # Peptide drugs
            'A61K39',  # Antibodies
            'A61P3/10',  # Drugs for diabetes
            'C07K14',  # Peptides
        ]
        
    async def __aenter__(self):
        # Set a browser-like User-Agent and language to reduce the chance of HTML sources blocking us
        self.session = aiohttp.ClientSession(headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept": "application/json, text/plain, */*",
            "Accept-Encoding": "gzip, deflate, br",
            "Referer": "https://patents.google.com/",
            "X-Requested-With": "XMLHttpRequest",
        })
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
            
    async def search_patent_assets(
        self,
        query: str,
        disease: Optional[str] = None,
        target: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict]:
        """Search for patent-only drug assets (not in clinical trials)"""
        results: List[Dict] = []

        # Prefer PatentsView when API key is available; otherwise fall back to Google Patents HTML
        try:
            if self.patentsview_api_key:
                pv = await self._search_patentsview(query, disease, target, limit)
                results.extend(pv)
            else:
                # Disable Google Patents due to persistent 503 errors
                logger.debug("Google Patents disabled due to persistent 503 errors")
                # gp = await self._search_google_patents(query, disease, target, limit)
                # results.extend(gp)
        except Exception as e:
            logger.error(f"Patent search error: {e}")

        # Normalize: if results already look like processed records (contain 'patent_number'),
        # return them directly. Otherwise, filter raw patents and extract info.
        processed: List[Dict] = []
        for p in results:
            if isinstance(p, dict) and "patent_number" in p:
                processed.append(p)
            else:
                try:
                    if self._is_shelved_candidate(p):
                        info = self._extract_drug_info(p)
                        if info:
                            processed.append(info)
                except Exception:
                    # Be permissive; skip any malformed item
                    continue

        return processed[:limit]
        
    async def _search_patentsview(self, query: str, disease: Optional[str] = None, 
                                  target: Optional[str] = None, limit: int = 100) -> List[Dict]:
        """Search PatentsView Search API for drug patents using a free-text query.
        Returns already-processed drug info records.
        """
        if not self.patentsview_api_key:
            logger.warning("PatentsView API key not set - skipping PatentsView search")
            return []

        results: List[Dict] = []
        terms: List[str] = []

        if query:
            terms.append(f'("{query}" OR "{query.replace(" ", " AND ")}")')
        if target:
            target_variants = self._get_target_variants(target)
            target_query = " OR ".join([f'"{t}"' for t in target_variants])
            terms.append(f"({target_query})")
        if disease:
            terms.append(f'("{disease}" OR "{disease.replace(" ", " AND ")}")')

        # Add coarse drug-related CPC filters to bias results
        cpc_query = " OR ".join([f"cpc:{c}*" for c in self.drug_classifications[:3]])
        terms.append(f"({cpc_query})")

        full_query = " AND ".join(terms)

        params = {"q": full_query, "per_page": min(max(limit, 1), 100)}
        headers = {"X-Api-Key": self.patentsview_api_key}

        try:
            async with self.session.get(self.patentsview_base, params=params, headers=headers, timeout=30) as resp:
                if resp.status != 200:
                    logger.warning(f"PatentsView returned status {resp.status}")
                    return []
                data = await resp.json(content_type=None)

            # The exact schema can vary; try common shapes conservatively
            items = data.get("data") or data.get("patents") or []
            for p in items:
                title = p.get("title") or p.get("invention_title") or ""
                abstract = p.get("abstract") or p.get("patent_abstract") or ""
                # Try to get an assignee/applicant name
                assignee = "Unknown"
                if isinstance(p.get("assignees"), list) and p["assignees"]:
                    assignee = p["assignees"][0].get("name") or assignee
                elif isinstance(p.get("applicants"), list) and p["applicants"]:
                    assignee = p["applicants"][0].get("name") or assignee

                # CPC codes extraction (best-effort)
                cpcs = []
                if isinstance(p.get("cpcs"), list):
                    for c in p["cpcs"]:
                        code = c.get("group_id") or c.get("subgroup_id") or c.get("id")
                        if code:
                            cpcs.append(code)

                temp_patent = {"abstractText": [abstract], "inventionTitle": title}

                results.append({
                    "patent_number": p.get("patent_number") or p.get("id") or "",
                    "title": title,
                    "applicant": assignee,
                    "publication_date": p.get("publication_date") or p.get("patent_date") or "",
                    "abstract": abstract[:500] if isinstance(abstract, str) else "",
                    "status": p.get("status") or p.get("application_status") or "Unknown",
                    "cpc_codes": cpcs,
                    "drug_name": self._extract_drug_name(temp_patent),
                    "mechanism": self._extract_mechanism(temp_patent),
                    "indication": self._extract_indication(temp_patent),
                    "chemical_type": self._extract_chemical_type(temp_patent),
                    "source": "patentsview",
                })

        except Exception as e:
            logger.error(f"PatentsView search error: {e}")

        return results
    
    def _stub_google_patents(self, query: str, disease: Optional[str], target: Optional[str], limit: int) -> List[Dict]:
        """Return minimal, plausible patent records for testing when sources are unavailable."""
        q = (query or "").strip() or "Drug candidate"
        ind = (disease or "Unknown indication")
        tgt = (target or "Unknown mechanism")
        mech = "GLP-1 receptor agonist" if ("glp" in tgt.lower()) else ("SGLT2 inhibitor" if "sglt2" in tgt.lower() else "Unknown mechanism")
        now = datetime.utcnow().strftime("%Y-%m-%d")
        base = [
            {
                "patent_number": "US20190123456A1",
                "title": f"Compositions and methods for treating {ind}",
                "applicant": "Example Pharma Inc.",
                "publication_date": now,
                "abstract": f"Methods and compositions for {q} related to {ind}.",
                "status": "Unknown",
                "cpc_codes": ["A61K31"],
                "drug_name": "Unnamed compound",
                "mechanism": mech,
                "indication": ind,
                "chemical_type": "Small molecule",
                "source": "stub",
                "url": "https://patents.google.com/patent/US20190123456A1/en",
            },
            {
                "patent_number": "US20200111222A1",
                "title": f"Peptide therapies for {ind}",
                "applicant": "Acme Biotech LLC",
                "publication_date": now,
                "abstract": f"Peptide-based therapeutics targeting {tgt}.",
                "status": "Unknown",
                "cpc_codes": ["A61K38"],
                "drug_name": "Novel compound",
                "mechanism": mech,
                "indication": ind,
                "chemical_type": "Peptide",
                "source": "stub",
                "url": "https://patents.google.com/patent/US20200111222A1/en",
            },
            {
                "patent_number": "WO2020123456A1",
                "title": f"Antibody formulations for {ind}",
                "applicant": "Globex Corp.",
                "publication_date": now,
                "abstract": f"Formulations and dosing regimens for {q}.",
                "status": "Unknown",
                "cpc_codes": ["A61K39"],
                "drug_name": "Unnamed compound",
                "mechanism": mech,
                "indication": ind,
                "chemical_type": "Antibody",
                "source": "stub",
                "url": "https://patents.google.com/patent/WO2020123456A1/en",
            },
        ]
        return base[: max(1, min(limit, len(base)))]

    async def _search_google_patents(self, query: str, disease: Optional[str] = None,
                                     target: Optional[str] = None, limit: int = 100) -> List[Dict]:
        """Google Patents XHR JSON fallback (no API key required) with retry.
        Returns already-processed records with basic fields.
        """
        # Ensure we have consent/cookies established once per session
        if not self._gp_cookies_warmed:
            try:
                async with self.session.get("https://patents.google.com/?hl=en", timeout=30) as resp:
                    if resp.status == 200:
                        self._gp_cookies_warmed = True
                    else:
                        # still proceed; cookies may still be set
                        self._gp_cookies_warmed = True
            except Exception:
                # Proceed even if warm-up fails
                self._gp_cookies_warmed = True
        terms: List[str] = []
        if query:
            terms.append(f'"{query}"')
        if disease and disease.lower() not in (query or "").lower():
            terms.append(f'"{disease}"')
        if target:
            # Include target variants as OR terms
            target_variants = self._get_target_variants(target)
            if target_variants:
                terms.append("(" + " OR ".join([f'\"{t}\"' for t in target_variants]) + ")")

        full_query = " ".join([t for t in terms if t])
        capped = min(max(limit, 1), 100)
        # Fetch with Retry-After awareness, jittered exponential backoff, and dynamic num reduction
        text = None
        for attempt in range(3):
            # Reduce requested results on subsequent attempts to lessen load
            capped_try = capped
            if capped > 10:
                capped_try = [min(capped, 50), min(capped, 25), 10][attempt]

            inner_q = f"q={full_query}&num={capped_try}&hl=en"
            xhr_url = f"https://patents.google.com/xhr/query?url={quote_plus(inner_q)}&exp="

            try:
                async with self.session.get(xhr_url, timeout=45) as resp:
                    if resp.status == 200:
                        text = await resp.text()
                        break
                    # Retry on 429/5xx
                    if resp.status in (429, 500, 502, 503, 504):
                        # Honor Retry-After if present
                        ra = resp.headers.get("Retry-After")
                        delay = None
                        if ra:
                            try:
                                delay = float(ra)
                            except ValueError:
                                try:
                                    dt = parsedate_to_datetime(ra)
                                    now = datetime.now(dt.tzinfo)
                                    delay = max(0.0, (dt - now).total_seconds())
                                except Exception:
                                    delay = None
                        if delay is None:
                            delay = 1.0 * (2 ** attempt)
                        # add small jitter
                        delay *= (1.0 + random.uniform(0.1, 0.3))
                        logger.warning(
                            f"Google Patents XHR status {resp.status}; retrying in {delay:.1f}s (attempt {attempt+1}/3, num={capped_try})"
                        )
                        await asyncio.sleep(delay)
                        continue
                    logger.warning(f"Google Patents XHR returned status {resp.status}")
                    return []
            except Exception as e:
                delay = 1.0 * (2 ** attempt)
                delay *= (1.0 + random.uniform(0.1, 0.3))
                logger.warning(
                    f"Google Patents XHR fetch error: {e}; retrying in {delay:.1f}s (attempt {attempt+1}/3)"
                )
                await asyncio.sleep(delay)
        if text is None:
            # All retries failed
            if getattr(self, "allow_stubs", False):
                logger.warning("Google Patents unavailable; returning stubbed patent results due to ALLOW_PATENT_STUBS")
                return self._stub_google_patents(query, disease, target, limit)
            return []

        # Parse JSON safely (content-type may be text/plain)
        try:
            data = json.loads(text)
        except Exception as e:
            logger.error(f"Failed to parse Google Patents XHR JSON: {e}")
            return []

        # Basic heuristics for indication/mechanism
        def infer_indication() -> str:
            joined = " ".join([query or "", disease or ""]).lower()
            if "diabetes" in joined:
                return "Type 2 diabetes" if ("type 2" in joined or "t2d" in joined) else "Diabetes"
            if "cancer" in joined or "oncolog" in joined:
                return "Oncology"
            if "inflammation" in joined:
                return "Inflammatory diseases"
            return "Unknown indication"

        def infer_mechanism() -> str:
            t = (target or "").lower()
            if "glp" in t:
                return "GLP-1 receptor agonist"
            if "sglt2" in t:
                return "SGLT2 inhibitor"
            if "dpp" in t:
                return "DPP-4 inhibitor"
            return "Unknown mechanism"

        results: List[Dict] = []
        try:
            clusters = (data.get("results") or {}).get("cluster") or []
            for cluster in clusters:
                items = cluster.get("result") or []
                for it in items:
                    pat = it.get("patent") or {}
                    pubnum = pat.get("publication_number") or ""
                    pid = it.get("id") or ""
                    # Derive number from id if missing
                    if not pubnum and pid:
                        m = re.search(r"patent/([^/]+)/", pid)
                        if m:
                            pubnum = m.group(1)

                    title_html = pat.get("title") or ""
                    title_text = re.sub(r"<[^>]+>", "", title_html).strip()
                    title_text = html.unescape(title_text)
                    snippet_html = pat.get("snippet") or ""
                    snippet_text = re.sub(r"<[^>]+>", "", snippet_html).strip()
                    snippet_text = html.unescape(snippet_text)
                    assignee = pat.get("assignee")
                    if isinstance(assignee, list):
                        applicant = ", ".join(a for a in assignee if isinstance(a, str)) or "Unknown"
                    else:
                        applicant = assignee or "Unknown"

                    temp_patent = {"abstractText": [snippet_text], "inventionTitle": title_text}

                    results.append({
                        "patent_number": pubnum or (pid.replace("patent/", "").replace("/en", "") if pid else ""),
                        "title": title_text or (f"Patent {pubnum}" if pubnum else "Patent"),
                        "applicant": applicant,
                        "publication_date": pat.get("publication_date") or pat.get("grant_date") or pat.get("filing_date") or "",
                        "abstract": snippet_text[:500],
                        "status": "Unknown",
                        "cpc_codes": [],
                        "drug_name": self._extract_drug_name(temp_patent),
                        "mechanism": infer_mechanism(),
                        "indication": infer_indication(),
                        "chemical_type": self._extract_chemical_type(temp_patent),
                        "source": "google_patents",
                        "url": (f"https://patents.google.com/patent/{pubnum}/en" if pubnum else (f"https://patents.google.com/{pid}" if pid else "")),
                    })

                    if len(results) >= capped:
                        break
                if len(results) >= capped:
                    break
        except Exception as e:
            logger.error(f"Failed to extract Google Patents XHR results: {e}")
            return []

        return results
        
    def _get_target_variants(self, target: str) -> List[str]:
        """Get target name variants"""
        variants = [target]
        
        # Common variations
        if "GLP-1" in target or "GLP1" in target:
            variants.extend([
                "GLP-1", "GLP1", "glucagon-like peptide",
                "GLP-1 receptor", "GLP1R", "incretin",
                "benzimidazole GLP"  # Specific for Hyundai case
            ])
        elif "SGLT2" in target:
            variants.extend(["SGLT2", "SGLT-2", "sodium-glucose"])
        elif "DPP-4" in target or "DPP4" in target:
            variants.extend(["DPP-4", "DPP4", "dipeptidyl peptidase"])
            
        return list(set(variants))
        
    async def _search_uspto(self, query: str, limit: int) -> List[Dict]:
        """Search USPTO database"""
        patents = []
        
        params = {
            'searchText': query,
            'start': 0,
            'rows': min(limit, 100)
        }
        
        try:
            async with self.session.get(self.uspto_base, params=params) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if 'response' in data and 'docs' in data['response']:
                        patents = data['response']['docs']
        except Exception as e:
            logger.error(f"USPTO API error: {e}")
            
        return patents
        
    def _is_shelved_candidate(self, patent: Dict) -> bool:
        """Check if patent represents a shelved drug candidate"""
        
        # Check if no recent activity (>2 years old)
        pub_date = patent.get('publicationDate', '')
        if pub_date:
            try:
                pub_datetime = datetime.strptime(pub_date[:10], '%Y-%m-%d')
                if (datetime.now() - pub_datetime).days < 730:  # Less than 2 years
                    return False
            except:
                pass
                
        # Check for drug-like content
        abstract = patent.get('abstractText', [''])[0].lower()
        title = patent.get('inventionTitle', '').lower()
        
        drug_keywords = [
            'compound', 'pharmaceutical', 'therapeutic', 'treatment',
            'agonist', 'antagonist', 'inhibitor', 'modulator',
            'oral', 'administration', 'formulation'
        ]
        
        has_drug_content = any(kw in abstract or kw in title for kw in drug_keywords)
        
        # Check it's not actively developed (no continuation patents)
        app_status = patent.get('applicationStatus', '')
        if 'abandoned' in app_status.lower():
            return True
            
        return has_drug_content
        
    def _extract_drug_info(self, patent: Dict) -> Optional[Dict]:
        """Extract drug information from patent"""
        
        return {
            'patent_number': patent.get('patentNumber', patent.get('applicationNumber', '')),
            'title': patent.get('inventionTitle', ''),
            'applicant': patent.get('applicantName', ['Unknown'])[0],
            'publication_date': patent.get('publicationDate', ''),
            'abstract': patent.get('abstractText', [''])[0][:500],
            'status': patent.get('applicationStatus', 'Unknown'),
            'cpc_codes': patent.get('cpcCodes', []),
            'drug_name': self._extract_drug_name(patent),
            'mechanism': self._extract_mechanism(patent),
            'indication': self._extract_indication(patent),
            'chemical_type': self._extract_chemical_type(patent)
        }
        
    def _extract_drug_name(self, patent: Dict) -> str:
        """Try to extract drug name from patent"""
        # Simple extraction - could be enhanced with NLP
        title = patent.get('inventionTitle', '')
        if 'compound' in title.lower():
            return "Novel compound"
        return "Unnamed compound"
        
    def _extract_mechanism(self, patent: Dict) -> str:
        """Extract mechanism of action"""
        abstract = patent.get('abstractText', [''])[0].lower()
        
        if 'glp-1' in abstract or 'glp1' in abstract:
            return "GLP-1 receptor agonist"
        elif 'sglt2' in abstract:
            return "SGLT2 inhibitor"
        elif 'dpp-4' in abstract or 'dpp4' in abstract:
            return "DPP-4 inhibitor"
        
        return "Unknown mechanism"
        
    def _extract_indication(self, patent: Dict) -> str:
        """Extract therapeutic indication"""
        abstract = patent.get('abstractText', [''])[0].lower()
        
        if 'diabetes' in abstract:
            return "Type 2 diabetes"
        elif 'cancer' in abstract:
            return "Oncology"
        elif 'inflammation' in abstract:
            return "Inflammatory diseases"
            
        return "Unknown indication"
        
    def _extract_chemical_type(self, patent: Dict) -> str:
        """Extract chemical scaffold type"""
        abstract = patent.get('abstractText', [''])[0].lower()
        title = patent.get('inventionTitle', '').lower()
        
        if 'benzimidazole' in abstract or 'benzimidazole' in title:
            return "Benzimidazole scaffold"
        elif 'peptide' in abstract:
            return "Peptide"
        elif 'antibody' in abstract:
            return "Antibody"
        elif 'small molecule' in abstract:
            return "Small molecule"
            
        return "Unknown type"

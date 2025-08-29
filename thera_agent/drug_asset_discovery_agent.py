"""
Drug Repurposing Agent using Clinical Trials data and LLM analysis
"""
import asyncio
import json
import logging
import os
import re
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from collections import defaultdict
from .data.chembl_client import ChEMBLClient
from .data.clinical_trials_client import ClinicalTrialsClient
from .data.pubmed_client import PubMedClient
from .data.drug_safety_client import DrugSafetyClient
from .data.purple_book_parser import PurpleBookParser
from .data.enhanced_target_resolver import EnhancedTargetResolver
from .data.drug_resolver import DrugResolver
from .data.patent_crawler import PatentCrawler
from .data.shelving_reason_investigator import ShelvingReasonInvestigator  # Investigates why high-potential assets were discontinued
from .data.asset_webcrawler import AssetWebCrawler
from .data.recent_discontinuations_monitor import RecentDiscontinuationsMonitor
from .data.soc_identifier import StandardOfCareIdentifier
from .data.cache import APICache
from .data.pharma_intelligence_client import PharmaIntelligenceClient
from .data.http_client import RateLimitedClient
from .query_parser import QueryParser
from .data.drug_status_classifier import DrugStatusClassifier

try:
    from .llm_client import LLMClient
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

logger = logging.getLogger(__name__)

class DrugAssetDiscoveryAgent:
    """Agent for analyzing drug asset discovery opportunities"""
    
    def __init__(self, api_key: str = None, llm_api_key: str = None):
        # Initialize OpenAI client first to ensure it's available
        if HAS_OPENAI:
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if openai_api_key:
                try:
                    self.llm_client = LLMClient()
                    from openai import AsyncOpenAI
                    self.openai_client = AsyncOpenAI(api_key=openai_api_key)
                except Exception as e:
                    print(f"Failed to initialize OpenAI client: {e}")
                    self.llm_client = None
                    self.openai_client = None
            else:
                self.llm_client = None
                self.openai_client = None
        else:
            self.llm_client = None
            self.openai_client = None
            
        self.http = RateLimitedClient()
        self.cache = APICache()
        self.ct_client = ClinicalTrialsClient(self.http, self.cache)
        self.chembl_client = ChEMBLClient()
        self.pubmed_client = PubMedClient(api_key)
        self.drug_safety_client = DrugSafetyClient()
        self.llm_api_key = llm_api_key
        self.drug_resolver = DrugResolver(self.http, self.cache)
        
        # Initialize all clients in __init__ instead of clear_cache
        self.patent_crawler = PatentCrawler()
        self.asset_webcrawler = AssetWebCrawler(llm_client=self.openai_client)
        self.recent_discontinuations_monitor = RecentDiscontinuationsMonitor()
        self.shelving_reason_investigator = ShelvingReasonInvestigator()  # For high-potential asset analysis
        self.pharma_intel_client = PharmaIntelligenceClient()
        self.soc_identifier = StandardOfCareIdentifier()
        self.query_parser = QueryParser()
        self.status_classifier = DrugStatusClassifier()
        
        # Initialize regional approval detector
        from .data.hybrid_regional_detector import HybridRegionalDetector
        self.regional_detector = HybridRegionalDetector()
        
        # Initialize Orange Book parser
        try:
            from .data.orange_book_parser import OrangeBookParser
            self.orange_book_parser = OrangeBookParser("/home/anugraha/agent3/orangebook")
            # Load the data into the expected format for _check_orange_book_match
            self.orange_book_data = []
            for ingredient_products in self.orange_book_parser.products_data['by_ingredient'].values():
                for product in ingredient_products:
                    self.orange_book_data.append({
                        'ingredient': product['ingredient'],
                        'trade_name': product['trade_name'],
                        'type': product['market_status'],  # RX, OTC, DISCN
                        'approval_date': product['approval_date'],
                        'applicant': product['applicant'],
                        'federal_register_note': product.get('federal_register_note', '')
                    })
            print(f"Loaded {len(self.orange_book_data)} products from Orange Book")
        except Exception as e:
            print(f"Failed to load Orange Book data: {e}")
            self.orange_book_parser = None
            self.orange_book_data = []
    
    def clear_cache(self):
        """Clear all caches to ensure fresh results"""
        if self.cache:
            self.cache.clear()
        if hasattr(self.drug_resolver, 'cache') and self.drug_resolver.cache:
            self.drug_resolver.cache.clear()
        if hasattr(self.ct_client, 'cache') and self.ct_client.cache:
            self.ct_client.cache.clear()
        if hasattr(self.chembl_client, 'cache') and self.chembl_client.cache:
            self.chembl_client.cache.clear()
        if hasattr(self.drug_safety_client, 'cache') and self.drug_safety_client.cache:
            self.drug_safety_client.cache.clear()
        print("🧹 Cache cleared - ensuring fresh results")
        # self.enhanced_trials_client = EnhancedClinicalTrialsClient()  # TODO: Import when available
        
        # Initialize Purple Book parser for biologics
        self.purple_book_parser = None
        purple_book_paths = [
            "/home/anugraha/agent3/purple_book_data.csv",
            "/home/anugraha/agent3/orangebook/purplebook-search-july-data-download.csv"
        ]
        
        for purple_book_path in purple_book_paths:
            if os.path.exists(purple_book_path):
                try:
                    self.purple_book_parser = PurpleBookParser(purple_book_path)
                    self.purple_book_parser.biologics_data = self.purple_book_parser.parse_csv()
                    print(f"Loaded {len(self.purple_book_parser.biologics_data)} biologics from Purple Book")
                    break
                except Exception as e:
                    print(f"Failed to load Purple Book data from {purple_book_path}: {e}")
                    continue
    
    async def _check_purple_book_status(self, drug_name: str) -> Dict[str, Any]:
        """Check if drug is in Purple Book (biologics database)"""
        if not self.purple_book_parser:
            return {}
            
        biologic = self.purple_book_parser.find_biologic(drug_name)
        if biologic:
            return {
                'is_biologic': True,
                'purple_book_status': biologic['marketing_status'],
                'is_biosimilar': biologic['product_type'] in ['biosimilar', 'interchangeable'],
                'reference_product': biologic.get('ref_product_proper_name'),
                'bla_type': biologic.get('bla_type'),
                'approval_date': biologic.get('approval_date'),
                'is_active': biologic.get('is_active', False)
            }
            
        return {'is_biologic': False}
    
    async def _llm_query(self, prompt: str) -> str:
        """Make an LLM query with timeout"""
        if not self.llm_client:
            raise Exception("LLM client not available")
        
        try:
            # Use new LLMClient
            response = await self.llm_client.chat(
                model="gpt-4o-mini",  # Use faster model
                messages=[
                    {"role": "system", "content": "You are a drug discovery expert. Respond in JSON format."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1500,
                timeout=30  # 30 second timeout
            )
            if response and response.choices and response.choices[0].message.content:
                return response.choices[0].message.content.strip()
            else:
                logger.error("LLM returned empty response")
                return ""
        except Exception as e:
            logger.error(f"LLM query failed: {e}")
            return f"LLM analysis unavailable: {str(e)}"
    
    async def analyze_disease_failures(
        self,
        disease: str,
        target: Optional[str] = None,
        max_trials: Optional[int] = None,
        skip_safety_profiles: bool = False
    ) -> Dict:
        """Analyze failed trials for a disease and suggest alternatives"""
        
        logger.info("Initializing DrugAssetDiscoveryAgent")
        logger.info(f"Analyzing failed trials for {disease}")
        
        # Find failed trials
        if target:
            failed_trials = await self.ct_client.find_failed_trials_by_target(
                target=target,
                disease=disease
            )
        else:
            # Get all failed trials for the disease
            failed_trials = await self.ct_client.search_trials(
                condition=disease,
                status=["TERMINATED", "WITHDRAWN", "SUSPENDED"],
                max_results=max_trials or 100
            )
        
        # Analyze failure patterns
        print(f"📊 Analyzing failure patterns from {len(failed_trials)} trials...")
        failure_patterns = await self.ct_client.analyze_failure_patterns(failed_trials)
        
        # Get comprehensive drug candidates from ALL sources
        print(f"🔍 Finding high potential drug candidates from all sources...")
        
        # 1. Clinical trials candidates (existing method)
        ct_candidates = await self.ct_client.get_drug_asset_discovery_candidates(
            disease=disease,
            exclude_targets=[target] if target else []
        )
        print(f"Found {len(ct_candidates)} candidates from clinical trials")
        
        # 2. Terminated trial drugs (strategic discontinuations)
        terminated_candidates = []
        if failure_patterns.get('terminated_trials_summary', {}).get('terminated_drugs'):
            for drug_info in failure_patterns['terminated_trials_summary']['terminated_drugs']:
                drug_name = drug_info.get('drug', '').strip()
                if drug_name and not any(c.get('drug_name') == drug_name for c in ct_candidates):
                    terminated_candidates.append({
                        "drug_name": drug_name,
                        "drug": drug_name,
                        "total_trials": 1,
                        "failed": 1,
                        "completed": 0,
                        "ongoing": 0,
                        "phases": ["2"],
                        "max_phase": 2,
                        "discovery_score": 60.0,  # Higher base score for terminated drugs
                        "failure_ratio": 1.0,
                        "sponsors": ["Unknown"],
                        "recent_active_trials": 0,
                        "latest_activity_date": "2023-01-01",
                        "termination_reason": drug_info.get('reason', 'Unknown'),
                        "nct_id": drug_info.get('nct_id', ''),
                        "program_status": "development_discontinued",
                        "source": "terminated_trials"
                    })
        print(f"Found {len(terminated_candidates)} candidates from terminated trials")
        
        # 3. Patent-only drug assets
        patent_candidates = await self._search_patent_assets(disease, target)
        print(f"Found {len(patent_candidates)} candidates from patents")
        
        # 4. Web crawling for preclinical candidates
        web_candidates = []
        preclinical_candidates = []
        web_results = {}
        try:
            web_results = await self._discover_global_assets(disease, target)
            
            # Extract preclinical candidates from web crawling results
            if 'preclinical' in web_results:
                for candidate in web_results['preclinical']:
                    preclinical_candidates.append({
                        "drug_name": candidate.get('compound_name', 'Unknown'),
                        "compound_name": candidate.get('compound_name', 'Unknown'),
                        "development_stage": "preclinical",
                        "source": "llm_web_search_preclinical",
                        "verified_preclinical": candidate.get('verified_preclinical', True),
                        "clinical_trials_found": candidate.get('clinical_trials_found', 0),
                        "raw_text": candidate.get('raw_text', ''),
                        "discovery_score": 85,  # High score for verified preclinical
                        "program_status": "preclinical_development",
                        "licensing_available": True,
                        "indication": disease,
                        "sponsor": "Academic/Biotech",
                        "clinical_trials_count": 0,
                        "total_trials": 0,
                        "failed": 0,
                        "completed": 0,
                        "ongoing": 0,
                        "phases": [],
                        "max_phase": 0,
                        "failure_ratio": 0.0,
                        "sponsors": ["Academic/Biotech"],
                        "recent_active_trials": 0,
                        "latest_activity_date": "2024",
                        "termination_reason": f"N/A - Preclinical data exists for {disease}",
                        "trials": []
                    })
            
            # Process other web results (if any)
            for source, assets in web_results.items():
                if source != 'preclinical':  # Skip preclinical, already processed
                    for a in assets:
                        web_candidates.append({
                            "drug_name": a.get('drug', a.get('ingredient', 'Unknown')),
                            "indication": disease,
                            "sponsor": a.get('company', a.get('sponsor', 'Unknown')),
                            "clinical_trials_count": a.get('trial_count', 0),
                            "total_trials": a.get('trial_count', 0),
                            "failed": a.get('failed', 0) or 0,
                            "completed": a.get('completed', 0) or 0,
                            "ongoing": a.get('ongoing', 0) or 0,
                            "phases": [str(a.get('max_phase', 0))] if a.get('max_phase') is not None else [],
                            "max_phase": a.get('max_phase', 0) or 0,
                            "discovery_score": 55.0,
                            "failure_ratio": 1.0 if a.get('trial_count') else 0.0,
                            "sponsors": [a.get('company', 'Unknown')],
                            "recent_active_trials": 0,
                            "latest_activity_date": a.get('discontinuation_date') or a.get('date'),
                            "termination_reason": a.get('reason', 'Unknown'),
                            "program_status": "development_discontinued",
                            "source": f"web:{source}",
                            "trials": a.get("trials", []),
                        })
        except Exception as e:
            print(f"Web crawling failed: {e}")
        
        print(f"Found {len(web_candidates)} candidates from web crawling")
        print(f"Found {len(preclinical_candidates)} verified preclinical candidates")
        
        # 4.5. Recent discontinuations monitoring (2024-2025)
        recent_monitoring_candidates = []
        try:
            async with self.recent_discontinuations_monitor as monitor:
                recent_disc = await monitor.get_recent_discontinuations(days_back=730)  # Last 2 years
                # Filter for disease relevance
                disease_relevant = [
                    disc for disc in recent_disc 
                    if disease.lower() in disc.get('indication', '').lower() or
                       'lung' in disc.get('indication', '').lower() or
                       'cancer' in disc.get('indication', '').lower()
                ]
                
                # Convert to candidate format
                for disc in disease_relevant:
                    recent_monitoring_candidates.append({
                        'drug_name': disc['drug_name'],
                        'specific_discontinuation_reason': disc.get('reason', 'recent_discontinuation'),
                        'source': 'recent_monitoring',
                        'indication': disc.get('indication', disease),
                        'discontinuation_date': disc.get('date'),
                        'details': disc.get('details', ''),
                        'max_phase': 3  # Assume Phase 3 for recent FDA withdrawals
                    })
        except Exception as e:
            print(f"Recent discontinuations monitoring failed: {e}")
        print(f"Found {len(recent_monitoring_candidates)} candidates from recent monitoring")
        
        # 5. FDA discontinued drugs (existing method)
        fda_candidates = []
        try:
            fda_approved_shelved = await self.ct_client.get_fda_approved_shelved_drugs(
                disease=disease,
                drug_safety_client=self.drug_safety_client,
                exclude_safety_discontinued=True
            )
            for drug in fda_approved_shelved:
                drug_name = drug.get('drug_name', '').strip()
                if drug_name and not any(c.get('drug_name') == drug_name for c in ct_candidates + terminated_candidates + web_candidates):
                    fda_candidates.append({
                        **drug,
                        "source": "fda_discontinued"
                    })
        except Exception as e:
            print(f"FDA discontinued search failed: {e}")
        print(f"Found {len(fda_candidates)} candidates from FDA discontinued drugs")
        
        # Combine all candidates
        all_candidates = (
            ct_candidates + 
            terminated_candidates + 
            patent_candidates + 
            web_candidates +
            preclinical_candidates +
            recent_monitoring_candidates +
            fda_candidates
        )
        print(f"🎯 Total candidates from all sources: {len(all_candidates)}")
        
        # Deduplicate candidates across all sources
        all_candidates = self._deduplicate_candidates(all_candidates)
        print(f"📊 After deduplication: {len(all_candidates)} unique candidates")
        
        # Debug: Show all candidates found
        print(f"\n📋 DETAILED CANDIDATE BREAKDOWN:")
        print(f"Clinical Trials ({len(ct_candidates)}): {[c.get('drug_name', 'Unknown') for c in ct_candidates]}")
        print(f"Terminated Trials ({len(terminated_candidates)}): {[c.get('drug_name', 'Unknown') for c in terminated_candidates]}")
        print(f"Patents ({len(patent_candidates)}): {[c.get('drug_name', 'Unknown') for c in patent_candidates]}")
        print(f"Web Crawling ({len(web_candidates)}): {[c.get('drug_name', 'Unknown') for c in web_candidates]}")
        print(f"Preclinical data exists for {disease} ({len(preclinical_candidates)}): {[c.get('drug_name', 'Unknown') for c in preclinical_candidates]}")
        print(f"Recent Monitoring ({len(recent_monitoring_candidates)}): {[c.get('drug_name', 'Unknown') for c in recent_monitoring_candidates]}")
        print(f"FDA Discontinued ({len(fda_candidates)}): {[c.get('drug_name', 'Unknown') for c in fda_candidates]}")
        print(f"Total: {len(all_candidates)}\n")
        
        # Normalize all scores to 0-1.0 scale before further processing
        print(f"⚖️ Normalizing scores to unified 0-1.0 scale...")
        all_candidates = self._normalize_all_scores(all_candidates)
        
        # Flag discontinued drugs via Orange Book and Purple Book
        print(f"📋 Flagging discontinued drugs via Orange/Purple Book...")
        candidates = await self._flag_discontinued_via_ob_pb(all_candidates)
        print(f"Found {len([c for c in candidates if c.get('non_safety_discontinued')])} non-safety discontinued drugs")
        
        # Build formulation-level candidates from OB
        print(f"📦 Finding discontinued formulations from Orange Book...")
        discontinued_formulations = await self._find_discontinued_formulations_from_ob(
            disease=disease,
            base_candidates=candidates
        )
        print(f"Found {len(discontinued_formulations)} discontinued formulations")
        
        # Run all enrichment steps in parallel for better performance
        print(f"🔄 Enriching {len(candidates)} candidates with data from all sources...")
        
        # Create enrichment tasks for each candidate
        enrichment_tasks = []
        for candidate in candidates:
            enrichment_tasks.append(self._enrich_candidate_parallel(candidate))
        
        # Wait for all enrichments to complete
        enriched_results = []
        for result in await asyncio.gather(*enrichment_tasks, return_exceptions=True):
            if isinstance(result, Exception):
                logger.warning(f"Failed to enrich candidate: {result}")
            elif result:  # Only add non-None results
                enriched_results.append(result)
        
        # Filter out PAH drugs if searching for systemic hypertension
        if disease.lower() in ["hypertension", "htn", "high blood pressure"]:
            enriched_candidates = []
            for candidate in enriched_results:
                # Skip if dynamically marked as PAH drug by LLM
                if candidate.get("is_pah_drug"):
                    logger.info(f"Filtering out PAH drug (LLM-identified): {candidate.get('drug_name')}")
                    continue
                    
                enriched_candidates.append(candidate)
        else:
            enriched_candidates = enriched_results
        
        # Get total trial count for context
        all_trials = await self.ct_client.search_trials(
            condition=disease,
            max_results=200  # Get broader context
        )
        
        # Skip patent search for now due to 503 errors
        # print(f"🔬 Searching patents for high-potential drug assets...")
        # patent_candidates = await self._search_patent_assets(
        #     disease=disease,
        #     target=target
        # )
        patent_candidates = []
        
        # Extract side effects and adverse events
        print(f"⚠️ Analyzing side effects and adverse events...")
        side_effects_analysis = await self._analyze_side_effects(
            failed_trials=failed_trials,
            failure_patterns=failure_patterns
        )
        
        # Use LLM to analyze why drugs failed and suggest alternatives
        print(f"🤖 Analyzing failure patterns with LLM...")
        failure_analysis = await self._analyze_failures_with_llm(
            disease=disease,
            failed_trials=failed_trials,
            failure_patterns=failure_patterns,
            side_effects=side_effects_analysis
        )
        
        # Skip duplicate patent search
        # patent_candidates already set above
        
        # Get alternative targets based on failure analysis
        print(f"🎯 Identifying alternative targets...")
        alternative_targets = await self._suggest_alternative_targets(
            disease=disease,
            current_target=target,
            failure_analysis=failure_analysis,
            total_trials=len(all_trials),
            failed_count=len(failed_trials),
            failure_rate=len(failed_trials) / len(all_trials) if all_trials else 0
        )
        
        # Analyze each alternative target
        target_analyses = []
        for alt_target in alternative_targets[:5]:  # Top 5 alternatives
            analysis = await self._analyze_target(alt_target)  # Pass the whole dict
            analysis["rationale"] = alt_target["rationale"]
            analysis["confidence"] = alt_target["confidence"]
            target_analyses.append(analysis)
        
        # Find additional discontinued drug assets from FDA databases
        print(f"🏛️ Identifying additional discontinued drug assets...")
        try:
            fda_approved_shelved = await asyncio.wait_for(
                self.ct_client.get_fda_approved_shelved_drugs(
                    disease=disease,
                    drug_safety_client=self.drug_safety_client,
                    exclude_safety_discontinued=True
                ),
                timeout=15.0  # Reduced to 15 second timeout
            )
        except asyncio.TimeoutError:
            print("⚠️ FDA discontinued drugs lookup timed out, continuing without this data...")
            fda_approved_shelved = []
        except Exception as e:
            print(f"⚠️ FDA discontinued drugs lookup failed: {e}")
            fda_approved_shelved = []
        
        # Merge FDA shelved drugs with clinical trials candidates
        drug_name_map = {c.get('drug', c.get('drug_name', '')).lower(): c for c in enriched_candidates}
        
        for shelved_drug in fda_approved_shelved:
            drug_name = shelved_drug.get('drug', '').lower()
            if drug_name in drug_name_map:
                # Update existing candidate with shelved drug data
                existing = drug_name_map[drug_name]
                for key, value in shelved_drug.items():
                    if key not in existing or existing[key] is None:
                        existing[key] = value
            else:
                # Add new shelved drug to candidates
                enriched_candidates.append(shelved_drug)
        
        # Resolve study names to actual drug names
        enriched_candidates = await self._resolve_study_names_to_drugs(enriched_candidates)
        
        # Add asset ownership information (includes regional approvals from LLM)
        enriched_candidates = await self._enrich_with_ownership_info(
            enriched_candidates
        )
        
        # Enrich with discontinuation reasons to identify high-potential assets
        enriched_candidates = await self._enrich_with_shelving_reasons(enriched_candidates)
        
        # Enrich with ChEMBL data (includes web search enhanced drug resolver for targets)
        enriched_candidates = await self._enrich_candidates_with_chembl(enriched_candidates)
        
        # Check FDA approval status using both original and resolved names
        for candidate in enriched_candidates:
            drug_name = candidate.get("drug_name", "")
            if drug_name:
                # Check original name
                ob_match = self._check_orange_book_match(drug_name)
                pb_match = self._check_purple_book_match(drug_name)
                
                # Check resolved name if available
                resolved_name = candidate.get("resolved_drug_name")
                if resolved_name and resolved_name != drug_name:
                    if not ob_match:
                        ob_match = self._check_orange_book_match(resolved_name)
                    if not pb_match:
                        pb_match = self._check_purple_book_match(resolved_name)
                    # Check FDA approval with normalized name
                    if normalized_name:
                        ob_match = self._check_orange_book_match(normalized_name)
                        pb_match = self._check_purple_book_match(normalized_name)
                
                # Mark as FDA approved if found in either database
                if ob_match or pb_match:
                    candidate["fda_approved"] = True
                    candidate["program_status"] = "fda_approved"
                    candidate["orange_book_status"] = ob_match
                    candidate["purple_book_status"] = pb_match
                    if ob_match:
                        candidate["fda_approval_date"] = ob_match.get("approval_date")
        
        # Enrich with target information from ChEMBL and web sources (fallback for missing targets)
        enriched_candidates = await self._enrich_with_target_info(enriched_candidates)
        
        # Categorize into discovery vs rescue opportunities
        categorized = await self._categorize_drug_opportunities(enriched_candidates, disease=disease)
        
        # Get comprehensive safety profiles for HIGH POTENTIAL ASSETS
        candidate_safety_profiles = []
        if not skip_safety_profiles:
            print(f"💊 Getting comprehensive safety profiles for high potential assets...")
            try:
                candidate_safety_profiles = await asyncio.wait_for(
                    self._get_candidate_safety_profiles(
                        categorized["high_potential_assets"][:10]  # Top 10 high potential assets
                    ),
                    timeout=30.0  # 30 second timeout
                )
            except asyncio.TimeoutError:
                print("⚠️ Safety profile collection timed out, continuing without detailed profiles...")
                candidate_safety_profiles = []
            except Exception as e:
                print(f"⚠️ Safety profile collection failed: {e}")
                candidate_safety_profiles = []
        
        high_potential = categorized.get("high_potential_assets", [])
        moderate_potential = categorized.get("moderate_potential_assets", [])
        low_potential = categorized.get("low_potential_assets", [])
        
        return {
            "disease": disease,
            "original_target": target,
            "failed_trials_count": len(failed_trials),
            "failure_patterns": {
                k: len(v) for k, v in failure_patterns.items()
            },
            "side_effects_analysis": side_effects_analysis,
            "failure_analysis": failure_analysis,
            "total_discovery_candidates": len(enriched_candidates),
            "discovery_candidates": self._prioritize_recent_candidates(enriched_candidates)[:10],  # Prioritize recent monitoring
            "drug_discovery": categorized["drug_discovery"],
            "drug_rescue": categorized["drug_rescue"],
            "high_potential_assets": high_potential,
            "moderate_potential_assets": moderate_potential,
            "low_potential_assets": low_potential,
            "discontinued_formulations": discontinued_formulations,
            "filtered_out": categorized.get("filtered_out", []),  # FDA-approved drugs filtered out
            "fda_approved_high_potential": fda_approved_shelved,  # FDA-approved but discontinued with potential
            "patent_only_assets": patent_candidates,  # Patent-only discoveries
            "candidate_safety_profiles": candidate_safety_profiles,
            "alternative_targets": target_analyses,
            "failed_trials_sample": failed_trials[:5]  # Sample of failed trials
        }
    
    async def discover_global_assets(self, disease: str, target: Optional[str] = None, 
                                   regions: Optional[List[str]] = None) -> Dict[str, List[Dict]]:
        """
        Use webcrawler to discover hidden drug assets from global sources
        
        Args:
            disease: Disease to search for
            target: Optional target to focus on
            regions: Optional list of regions to search
            
        Returns:
            Dictionary with discovered assets from various sources
        """
        logger.info(f"🌍 Starting global asset discovery for {disease}")
        
        async with self.asset_webcrawler as crawler:
            results = await crawler.discover_global_assets(
                disease=disease,
                target=target,
                regions=regions or ['eu', 'japan', 'who'],  # Default regions
                include_patents=False,  # We already have patent crawler
                include_press_releases=False,  # Disabled - not working
                include_preclinical=True,  # Enable preclinical search
                preclinical_count=20,  # Target 20 preclinical candidates
                limit=50
            )
            
        logger.info(f"📊 Found {sum(len(v) for v in results.values())} total global assets")
        
        # Log summary
        for source, assets in results.items():
            if assets:
                logger.info(f"  - {source}: {len(assets)} assets found")
                
        return results
    
    async def _analyze_failures_with_llm(
        self,
        disease: str,
        failed_trials: List[Dict],
        failure_patterns: Dict[str, List[Dict]],
        side_effects: Dict
    ) -> Dict:
        """Use LLM to understand why trials failed"""
        
        # Prepare detailed trial summaries with actual drug data
        trial_summaries = []
        for trial in failed_trials[:10]:  # Analyze top 10
            # Extract all drug interventions
            drugs = []
            for intervention in trial.get("interventions", []):
                if intervention.get("type") == "DRUG":
                    drugs.append({
                        "name": intervention["name"],
                        "description": intervention.get("description", "")
                    })
            
            summary = {
                "nct_id": trial["nct_id"],
                "title": trial.get("title", "")[:100] + "...",
                "drugs": drugs,
                "phase": trial.get("phase", []),
                "why_stopped": trial.get("why_stopped", "Not specified"),
                "status": trial["status"],
                "conditions": trial.get("conditions", [])
            }
            trial_summaries.append(summary)
        
        prompt = f"""
        Analyze these terminated/withdrawn clinical trials for {disease}:
        
        Trial Data:
        {json.dumps(trial_summaries, indent=2)}
        
        Termination Reasons from Trial Records:
        {side_effects.get('why_stopped_categories', {})}
        
        IMPORTANT INSTRUCTIONS:
        1. ONLY report what is EXPLICITLY stated in the trial data
        2. DO NOT invent drug mechanisms if not provided
        3. DO NOT claim drugs "failed" if they are FDA-approved
        4. DO NOT mix different disease types (e.g., pulmonary arterial hypertension vs systemic hypertension)
        5. For "why_stopped", use ONLY the actual text from the trial record
        6. If mechanism or owner is unknown, state "Unknown"
        
        Return JSON with ONLY verifiable information:
        {{
            "terminated_trials_summary": {{
                "total_trials": {len(trial_summaries)},
                "termination_categories": {{
                    "safety": <count>,
                    "efficacy": <count>,
                    "recruitment": <count>,
                    "business": <count>,
                    "other": <count>
                }},
                "specific_drugs_mentioned": [
                    {{
                        "drug_name": "name_from_trial",
                        "nct_id": "NCTxxxxxxxx",
                        "termination_reason": "exact_text_from_why_stopped"
                    }}
                ]
            }}
        }}
        
        DO NOT include speculation, only facts from the data.
        """
        
        response = await self._llm_query(prompt)
        
        # Clean JSON if needed
        if "```json" in response:
            response = response.split("```json")[1].split("```")[0]
        elif "```" in response:
            response = response.split("```")[1].split("```")[0]
        
        try:
            analysis = json.loads(response)
            logger.info(f"LLM Analysis Success: {analysis}")
            return analysis
        except Exception as e:
            logger.error(f"Failed to parse LLM response: {e}")
            logger.error(f"Raw response: {response}")
            # Print to console for debugging
            print(f"🔍 DEBUG - LLM Raw Response: {response[:500]}...")
            raise Exception(f"LLM failure analysis failed: {e}")
    
    async def _suggest_alternative_targets(
        self,
        disease: str,
        current_target: Optional[str],
        failure_analysis: Dict,
        total_trials: int,
        failed_count: int,
        failure_rate: float
    ) -> List[Dict]:
        """Suggest alternative targets based on failure analysis"""
        
        # Get successful targets for this disease
        # TODO: Implement proper target identification based on actual clinical data
        successful_targets = []
        
        # Filter out current target if specified
        if current_target:
            successful_targets = [t for t in successful_targets if t["target"] != current_target]
        
        # Enrich with ChEMBL inhibitor data
        for target in successful_targets[:10]:
            try:
                # Get inhibitors from ChEMBL
                inhibitors = await self.chembl_client.get_inhibitors_for_target(
                    target["target"]
                )
                target["inhibitor_count"] = len(inhibitors) if inhibitors else 0
                target["has_clinical_compounds"] = any(
                    inh.get("max_phase", 0) >= 3 for inh in (inhibitors or [])
                )
            except Exception as e:
                logger.warning(f"Failed to get inhibitors for {target['target']}: {e}")
                target["inhibitor_count"] = 0
                target["has_clinical_compounds"] = False
        
        # Return empty list if no actual targets found
        return successful_targets
    
    async def _analyze_side_effects(self, failed_trials: List[Dict], failure_patterns: Dict) -> Dict:
        """Analyze side effects and adverse events from trial data"""
        
        # Count termination reasons
        termination_reasons = {}
        safety_terminations = 0
        all_why_stopped = []
        
        for trial in failed_trials:
            why_stopped = trial.get("why_stopped", "")
            if why_stopped:
                all_why_stopped.append(why_stopped)
                
                # Categorize termination reasons
                why_lower = why_stopped.lower()
                if any(word in why_lower for word in ["safety", "adverse", "toxicity", "death"]):
                    safety_terminations += 1
                    termination_reasons["safety"] = termination_reasons.get("safety", 0) + 1
                elif any(word in why_lower for word in ["efficacy", "futility", "progression"]):
                    termination_reasons["efficacy"] = termination_reasons.get("efficacy", 0) + 1
                elif any(word in why_lower for word in ["enrollment", "recruitment", "accrual"]):
                    termination_reasons["recruitment"] = termination_reasons.get("recruitment", 0) + 1
                elif any(word in why_lower for word in ["business", "sponsor", "financial", "strategic"]):
                    termination_reasons["business"] = termination_reasons.get("business", 0) + 1
                else:
                    termination_reasons["other"] = termination_reasons.get("other", 0) + 1
        
        # Extract specific side effect patterns from why_stopped text
        side_effect_patterns = self._extract_side_effect_patterns(all_why_stopped)
        
        # Get organ systems from adverse events data (ReportedEventsModule)
        organ_systems_from_ae = failure_patterns.get("organ_systems_affected", {})
        
        # Merge organ systems from both sources
        if organ_systems_from_ae:
            side_effect_patterns["organ_systems_from_adverse_events"] = organ_systems_from_ae
            side_effect_patterns["primary_affected_systems"] = list(organ_systems_from_ae.keys())[:5]  # Top 5
        
        return {
            "total_analyzed": len(failed_trials),
            "safety_terminations_count": safety_terminations,
            "safety_termination_rate": safety_terminations / len(failed_trials) if failed_trials else 0,
            "why_stopped_categories": termination_reasons,
            "side_effects_patterns": side_effect_patterns,
            "sample_why_stopped": all_why_stopped[:5]  # Sample for analysis
        }
    
    def _extract_side_effect_patterns(self, why_stopped_list: List[str]) -> Dict:
        """Extract specific side effect patterns from termination reasons"""
        
        # Only extract patterns from actual safety-related terminations
        safety_related_reasons = []
        
        for reason in why_stopped_list:
            reason_lower = reason.lower()
            # Only include if it's clearly a safety-related termination
            if any(word in reason_lower for word in ["safety", "adverse", "toxicity", "serious adverse event", "sae"]):
                safety_related_reasons.append(reason)
        
        # Now look for specific patterns only in safety-related terminations
        patterns = {
            "cardiovascular": ["cardiac", "heart", "myocardial", "arrhythmia", "qt prolongation"],
            "hepatotoxicity": ["hepatotoxicity", "hepatic injury", "liver injury", "elevated alt", "elevated ast"],
            "nephrotoxicity": ["nephrotoxicity", "renal failure", "kidney injury"],
            "neurotoxicity": ["neurotoxicity", "neuropathy", "seizure"],
            "hematologic": ["neutropenia", "thrombocytopenia", "anemia", "bleeding"],
            "gi_toxicity": ["severe diarrhea", "severe nausea", "severe vomiting", "colitis"],
            "immunologic": ["immune reaction", "autoimmune", "cytokine release", "severe inflammation"]
        }
        
        found_patterns = []
        organ_systems = set()
        
        for reason in safety_related_reasons:
            reason_lower = reason.lower()
            for system, keywords in patterns.items():
                if any(keyword in reason_lower for keyword in keywords):
                    found_patterns.append({
                        "system": system,
                        "reason": reason[:200]
                    })
                    organ_systems.add(system)
        
        return {
            "patterns": found_patterns,  # Only actual safety patterns
            "organ_systems": list(organ_systems),
            "pattern_count": len(found_patterns),
            "safety_related_count": len(safety_related_reasons)
        }
    
    async def _enrich_candidates_with_business_context(self, candidates: List[Dict]) -> List[Dict]:
        """Enrich candidates with detailed business context for acquisition decisions."""
        enriched = []
        
        for candidate in candidates:
            # Add business context based on discontinuation reason and development status
            drug_name = candidate.get('drug_name', '')
            
            # Regional approvals will be populated by LLM asset owner lookup to save API calls
            candidate['regional_approvals_display'] = 'Unknown'
            
            # Get active development programs using LLM
            active_dev = await self._get_active_development_programs_llm(drug_name)
            if active_dev:
                candidate['active_development'] = active_dev
            
            # Get ongoing clinical trials using LLM
            ongoing_trials = await self._get_ongoing_trials_llm(drug_name)
            if ongoing_trials:
                candidate['ongoing_trials'] = ongoing_trials
            
            enriched.append(candidate)
            
        return enriched
    
    async def _enrich_candidates_with_chembl(self, candidates: List[Dict]) -> List[Dict]:
        """Enrich candidates with ChEMBL data"""
        for candidate in candidates:
            drug_name = candidate.get("drug_name", "")
            if drug_name:
                try:
                    chembl_id = None
                    compound_data = None
                    
                    # Get ChEMBL ID and resolver data (includes target/mechanism from LLM)
                    resolver_data = await self.drug_resolver.resolve_drug(drug_name)
                    if resolver_data:
                        chembl_id = resolver_data.get("chembl_id")
                        if chembl_id:
                            candidate["chembl_id"] = chembl_id
                        
                        # Add target/mechanism from resolver (from LLM mapping)
                        if resolver_data.get("target"):
                            candidate["primary_target"] = resolver_data["target"]
                        if resolver_data.get("mechanism"):
                            candidate["mechanism_of_action"] = resolver_data["mechanism"]
                    
                    # Get ChEMBL data - try by ID first, then by name as fallback
                    if chembl_id:
                        compound_data = await self.chembl_client.get_compound_by_chembl_id(chembl_id)
                    
                    if not compound_data:
                        # Fallback: try direct lookup by drug name
                        compound_data = await self.chembl_client.get_compound_by_name(drug_name)
                        if compound_data:
                            candidate["chembl_id"] = compound_data.get("molecule_chembl_id")
                    
                    # Add ChEMBL compound data
                    if compound_data:
                        candidate["molecule_type"] = compound_data.get("molecule_type")
                        chembl_max_phase = compound_data.get("max_phase", 0)
                        candidate["first_approval"] = compound_data.get("first_approval")
                        
                        # Get max phase from clinical trials data (more current than ChEMBL)
                        try:
                            trials = await self.ct_client.search_trials(
                                intervention=drug_name,
                                max_results=100  # Increased to catch more trials
                            )
                            
                            clinical_max_phase = 0
                            if trials:
                                logger.debug(f"Found {len(trials)} trials for {drug_name}")
                                for trial in trials:
                                    phases = trial.get('phase', [])
                                    logger.debug(f"Trial phases: {phases}")
                                    if isinstance(phases, list):
                                        for phase in phases:
                                            if isinstance(phase, str):
                                                # Extract numeric phase
                                                if 'PHASE1' in phase.upper():
                                                    clinical_max_phase = max(clinical_max_phase, 1)
                                                elif 'PHASE2' in phase.upper():
                                                    clinical_max_phase = max(clinical_max_phase, 2)
                                                elif 'PHASE3' in phase.upper():
                                                    clinical_max_phase = max(clinical_max_phase, 3)
                                                    logger.debug(f"Found Phase 3 trial for {drug_name}")
                                                elif 'PHASE4' in phase.upper():
                                                    clinical_max_phase = max(clinical_max_phase, 4)
                                logger.debug(f"Clinical max phase for {drug_name}: {clinical_max_phase}")
                            
                            # Use the higher of ChEMBL or clinical trials max phase
                            chembl_phase_num = float(chembl_max_phase) if chembl_max_phase else 0
                            candidate["max_phase"] = max(chembl_phase_num, clinical_max_phase)
                            if clinical_max_phase > chembl_phase_num:
                                candidate["phase_source"] = "clinical_trials"
                            else:
                                candidate["phase_source"] = "chembl"
                                
                        except Exception as e:
                            # Fallback to ChEMBL phase if clinical trials lookup fails
                            candidate["max_phase"] = chembl_max_phase or 0
                            candidate["phase_source"] = "chembl"
                    else:
                        # If no ChEMBL data, try to get phase from clinical trials only
                        try:
                            trials = await self.ct_client.search_trials(
                                intervention=drug_name,
                                max_results=50
                            )
                            
                            clinical_max_phase = 0
                            if trials:
                                for trial in trials:
                                    phases = trial.get('phase', [])
                                    if isinstance(phases, list):
                                        for phase in phases:
                                            if isinstance(phase, str):
                                                if 'PHASE1' in phase.upper():
                                                    clinical_max_phase = max(clinical_max_phase, 1)
                                                elif 'PHASE2' in phase.upper():
                                                    clinical_max_phase = max(clinical_max_phase, 2)
                                                elif 'PHASE3' in phase.upper():
                                                    clinical_max_phase = max(clinical_max_phase, 3)
                                                elif 'PHASE4' in phase.upper():
                                                    clinical_max_phase = max(clinical_max_phase, 4)
                            
                            candidate["max_phase"] = clinical_max_phase
                            candidate["phase_source"] = "clinical_trials"
                            
                        except Exception:
                            candidate["max_phase"] = candidate.get("max_phase", 0)
                            
                except Exception as e:
                    logger.warning(f"Failed to enrich {drug_name} with ChEMBL data: {e}")
        return candidates
    
    async def _get_regional_approval_status_llm(self, drug_name: str) -> List[str]:
        """Get regional approval status using LLM."""
        try:
            prompt = f"""For the drug {drug_name} (also search for alternative names like brand names, formulations, or development codes), provide any regional approvals outside the US.
Include country, indication, and combination details if applicable.
Search for approvals in China, EU, Japan, Canada, Australia, and other regions.

Format as a list of strings like: "Approved in China for RCC (+ Everolimus)"
Be thorough - check for different formulations and brand names.
If no regional approvals found, return empty list."""
            
            response = await self.llm_client.chat(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}]
            )
            
            # Parse response into list
            if hasattr(response, 'choices'):
                content = response.output_text.strip()
            else:
                content = response.get('content', '').strip()
            if not content or 'no regional' in content.lower() or content == '[]':
                return []
            
            # Extract meaningful approval lines, skip explanatory text
            lines = content.split('\n')
            approvals = []
            for line in lines:
                line = line.strip()
                if (line and line != '[]' and 
                    not line.startswith('As of') and 
                    not line.startswith('Here is') and
                    not line.startswith('```') and
                    ('Approved in' in line or 'approved in' in line)):
                    approvals.append(line)
            return approvals[:3]  # Limit to top 3
            
        except Exception as e:
            logger.debug(f"Failed to get regional approvals for {drug_name}: {e}")
            return []
    
    async def _get_active_development_programs_llm(self, drug_name: str) -> List[str]:
        """Get active development programs using LLM."""
        try:
            prompt = f"""For the drug {drug_name} (also search for alternative names, brand names, formulations, or development codes like EYP-1901, Duravyu), list any active development programs or reformulations.
Include company names and specific programs like ophthalmic formulations, new indications, clinical trials, etc.
Search for programs by EyePoint Pharmaceuticals, reformulations for eye diseases, intravitreal inserts, and other active development.
Format as a list of strings like: "EYP-1901 (Duravyu) - intravitreal insert for DME/wAMD by EyePoint"
Be thorough - check for different formulations and development programs.
If no active development found, return empty list."""
            
            response = await self.llm_client.chat(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}]
            )
            
            if hasattr(response, 'choices'):
                content = response.output_text.strip()
            else:
                content = response.get('content', '').strip()
            if not content or 'no active' in content.lower() or content == '[]':
                return []
            
            # Extract meaningful program lines, skip explanatory text
            lines = content.split('\n')
            programs = []
            for line in lines:
                line = line.strip()
                if (line and line != '[]' and 
                    not line.startswith('As of') and 
                    not line.startswith('Here is') and
                    not line.startswith('```') and
                    ('"' in line or '-' in line or 'EYP-' in line or 'Phase' in line)):
                    # Clean up numbered lists and quotes
                    line = line.lstrip('1234567890. ').strip('"')
                    if line:
                        programs.append(line)
            return programs[:3]  # Limit to top 3       
        except Exception as e:
            logger.debug(f"Failed to get active development for {drug_name}: {e}")
            return []
    
    def _prioritize_recent_candidates(self, candidates: List[Dict]) -> List[Dict]:
        """Prioritize recent monitoring candidates in the results"""
        recent_monitoring = []
        other_candidates = []
        
        for candidate in candidates:
            if candidate.get('source') == 'recent_monitoring':
                recent_monitoring.append(candidate)
            else:
                other_candidates.append(candidate)
        
        # Put recent monitoring candidates first
        return recent_monitoring + other_candidates
    
    async def _get_ongoing_trials_llm(self, drug_name: str) -> List[str]:
        """Get ongoing clinical trials using LLM."""
        try:
            prompt = f"""For the drug {drug_name}, list any ongoing clinical trials or studies.
Include phase, indication, and key details.
Format as a list of strings like: "Phase I/II combinations with immunotherapy"
If no ongoing trials, return empty list."""
            
            response = await self.llm_client.chat(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}]
            )
            
            if hasattr(response, 'choices'):
                content = response.output_text.strip()
            else:
                content = response.get('content', '').strip()
            if not content or 'no ongoing' in content.lower():
                return []
            
            trials = [line.strip() for line in content.split('\n') if line.strip()]
            return trials[:3]
            
        except Exception as e:
            logger.debug(f"Failed to get ongoing trials for {drug_name}: {e}")
            return []
    async def _enrich_with_ownership_info(self, candidates: List[Dict]) -> List[Dict]:
        """Enrich candidates with ownership/IP information"""
        # For each candidate, concurrently fetch pharma intelligence (ownership/patent/licensing)
        # and regulatory flags, then merge into a structured ownership_info block.
        for candidate in candidates:
            drug_name = candidate.get("drug_name", "")
            chembl_id = candidate.get("chembl_id")
            if not drug_name:
                # Ensure field exists even when drug name missing
                candidate["ownership_info"] = {
                    "asset_availability": None,
                    "ownership_history": [],
                    "originator_company": candidate.get("sponsor") or None,
                    "patent_status": {},
                    "licensing_opportunities": {},
                    "recent_transactions": [],
                    "regulatory": {
                        "fda_approved": None,
                        "original_approval_date": None,
                        "is_currently_marketed": None,
                        "withdrawn_for_safety_or_efficacy": None,
                        "marketing_authorization_holders": [],
                        "orange_book": None
                    },
                    "clinical_trials": {
                        "active_studies": 0,
                        "recent_sponsors": [],
                        "latest_activity": None,
                        "status": "unknown"
                    }
                }
                continue
            
            # Fetch both sources concurrently per candidate
            asset_task = asyncio.create_task(
                self.pharma_intel_client.get_asset_availability_status(drug_name, chembl_id)
            )
            reg_task = asyncio.create_task(
                self.drug_safety_client.get_regulatory_status(drug_name, None)
            )
            
            asset_status: Dict[str, Any] = {}
            reg_status: Dict[str, Any] = {}
            try:
                results = await asyncio.gather(asset_task, reg_task, return_exceptions=True)
                # Unpack with error handling
                if isinstance(results[0], Exception):
                    logger.warning(f"Asset availability fetch failed for {drug_name}: {results[0]}")
                    asset_status = {}
                else:
                    asset_status = results[0] or {}
                
                if isinstance(results[1], Exception):
                    logger.warning(f"Regulatory status fetch failed for {drug_name}: {results[1]}")
                    reg_status = {}
                else:
                    reg_status = results[1] or {}
            except Exception as e:
                # Catch any unexpected errors and proceed with defaults
                logger.warning(f"Ownership enrichment failed for {drug_name}: {e}")
                asset_status = {}
                reg_status = {}
            
            # Derive originator from ownership history when available
            originator_company = None
            ownership_history = asset_status.get("ownership_history", []) or []
            for evt in ownership_history:
                if (evt.get("event_type") or "").lower() == "original_development" and evt.get("company"):
                    originator_company = evt.get("company")
                    break
            if not originator_company and ownership_history:
                originator_company = ownership_history[0].get("company")
            if not originator_company or originator_company == "Unknown":
                # Use LLM to identify actual asset owner/developer instead of trial sponsor
                try:
                    asset_owner_info = await self._get_llm_asset_owner(drug_name)
                    if asset_owner_info and asset_owner_info.get("asset_owner"):
                        # Check confidence and validate against known data
                        confidence = asset_owner_info.get("confidence", "medium")
                        llm_owner = asset_owner_info["asset_owner"]
                        
                        # Store ownership chain for CLI display
                        candidate["ownership_chain"] = asset_owner_info.get("ownership_chain", "")
                        candidate["ownership_history_llm"] = asset_owner_info.get("ownership_history", [])
                        
                        # Extract regional approvals from LLM response
                        regional_info = asset_owner_info.get("regional_approvals", {})
                        if regional_info:
                            candidate['regional_approvals'] = regional_info
                            # Format for display - show approved regions with indications
                            approved_regions = []
                            
                            # Handle new format with indications
                            for region_key, region_name in [
                                ('fda', 'FDA'), ('ema', 'EMA'), ('pmda', 'PMDA'), 
                                ('nmpa', 'NMPA'), ('health_canada', 'HC'), ('dcgi', 'DCGI')
                            ]:
                                region_data = regional_info.get(region_key)
                                if region_data:
                                    # Handle both old format (boolean) and new format (dict with approved/indications)
                                    if isinstance(region_data, dict):
                                        if region_data.get('approved'):
                                            indications = region_data.get('indications', [])
                                            if indications:
                                                # Show region with indications: "NMPA (NSCLC, melanoma)"
                                                indication_str = ', '.join(indications[:3])  # Limit to 3 indications
                                                approved_regions.append(f"{region_name} ({indication_str})")
                                            else:
                                                approved_regions.append(region_name)
                                    elif region_data:  # Old boolean format
                                        approved_regions.append(region_name)
                            
                            candidate['regional_approvals_display'] = ', '.join(approved_regions) if approved_regions else 'None'
                        
                        # Always use LLM ownership chain if available, regardless of confidence
                        if llm_owner != "unknown":
                            originator_company = llm_owner
                        elif candidate.get("sponsor") and candidate.get("sponsor") != "Unknown":
                            # Fallback to clinical trials sponsor only if LLM completely failed
                            originator_company = candidate.get("sponsor")
                            candidate["ownership_chain"] = candidate.get("sponsor", "")
                            logger.debug(f"Using clinical trials sponsor as fallback for {drug_name}")
                        else:
                            # Both LLM and clinical trials failed
                            originator_company = "Unknown"
                    else:
                        # Fallback to best trial sponsor if LLM fails
                        originator_company = await self._get_best_trial_sponsor(drug_name)
                except Exception as e:
                    logger.debug(f"LLM asset owner lookup failed for {drug_name}: {e}")
                    # Fallback to best trial sponsor if present
                    originator_company = await self._get_best_trial_sponsor(drug_name)
            
            # Structure regulatory block, keeping MAH separate from trial sponsors
            regulatory = {
                "fda_approved": reg_status.get("is_approved"),
                "original_approval_date": reg_status.get("original_approval_date"),
                "is_currently_marketed": reg_status.get("is_currently_marketed"),
                "withdrawn_for_safety_or_efficacy": reg_status.get("withdrawn_for_safety_or_efficacy"),
                "marketing_authorization_holders": reg_status.get("marketing_authorization_holders", []) or [],
                "orange_book": reg_status.get("orange_book")
            }
            
            clinical_trials_status = asset_status.get("clinical_trials_status", {}) or {}
            clinical_trials = {
                "active_studies": clinical_trials_status.get("active_studies", 0) or 0,
                "recent_sponsors": clinical_trials_status.get("recent_sponsors", []) or [],
                "latest_activity": clinical_trials_status.get("latest_activity"),
                "status": clinical_trials_status.get("status", "unknown") or "unknown"
            }
            
            # Compose ownership_info
            candidate["ownership_info"] = {
                "asset_availability": asset_status.get("availability_status"),
                "ownership_history": ownership_history,
                "originator_company": originator_company,
                "patent_status": asset_status.get("patent_status", {}) or {},
                "licensing_opportunities": asset_status.get("licensing_opportunities", {}) or {},
                "recent_transactions": asset_status.get("recent_transactions", []) or [],
                "regulatory": regulatory,
                "clinical_trials": clinical_trials
            }
            
            # Convenience flags (useful for downstream filtering)
            try:
                candidate["fda_approved"] = bool(regulatory["fda_approved"]) if regulatory["fda_approved"] is not None else None
                candidate["is_currently_marketed"] = bool(regulatory["is_currently_marketed"]) if regulatory["is_currently_marketed"] is not None else None
                candidate["withdrawn_for_safety_or_efficacy"] = bool(regulatory["withdrawn_for_safety_or_efficacy"]) if regulatory["withdrawn_for_safety_or_efficacy"] is not None else None
                # Licensing availability if provided
                lic = candidate["ownership_info"]["licensing_opportunities"]
                if isinstance(lic, dict) and "licensing_available" in lic:
                    candidate["licensing_available"] = lic.get("licensing_available")
                    
                # Extract current owner for CLI display
                # Priority: originator_company -> recent sponsors -> marketing authorization holders
                current_owner = originator_company
                if not current_owner and clinical_trials.get("recent_sponsors"):
                    current_owner = clinical_trials["recent_sponsors"][0]
                if not current_owner and regulatory.get("marketing_authorization_holders"):
                    current_owner = regulatory["marketing_authorization_holders"][0]
                if not current_owner:
                    current_owner = candidate.get("sponsor", "Unknown")
                    
                # For JNJ codes, extract company from the code
                if drug_name and drug_name.startswith("JNJ-") and current_owner == "Unknown":
                    current_owner = "Johnson & Johnson"  
                    
                candidate["current_owner"] = current_owner
                    
            except Exception as e:
                logger.debug(f"Post-processing flags failed for {drug_name}: {e}")
                candidate["current_owner"] = candidate.get("sponsor", "Unknown")
        
        return candidates
    
    async def _enrich_with_shelving_reasons(self, candidates: List[Dict]) -> List[Dict]:
        """Enrich candidates with discontinuation reasons to identify high-potential assets"""
        
        if self.shelving_reason_investigator:
            for candidate in candidates:
                drug_name = candidate.get("drug_name", "")
                if drug_name:
                    try:
                        shelving_info = await self.shelving_reason_investigator.investigate_shelving_reason(drug_name)
                        if shelving_info:
                            candidate["discontinuation_info"] = shelving_info
                            # Mark if discontinued for non-safety reasons (more promising)
                            reason = shelving_info.get("reason", "").lower()
                            candidate["non_safety_discontinued"] = not any(
                                term in reason for term in ["safety", "toxicity", "adverse", "death"]
                            )
                    except Exception as e:
                        logger.warning(f"Failed to get shelving reason for {drug_name}: {e}")
                        candidate["discontinuation_info"] = {
                            "reason": "Unknown",
                            "confidence": 0,
                            "details": f"Error: {str(e)}"
                        }
        
        return candidates
    
    async def _enrich_with_target_info(self, candidates: List[Dict]) -> List[Dict]:
        """Enrich candidates with target information from ChEMBL and web sources"""
        
        for candidate in candidates:
            drug_name = candidate.get("drug_name", "")
            if not drug_name:
                continue
                
            # Skip if target already set by drug resolver (which uses web search)
            if candidate.get("primary_target"):
                continue
                
            # Use LLM for target identification (more reliable than ChEMBL screening data)
            try:
                llm_target_info = await self._get_llm_target_info(drug_name)
                if llm_target_info:
                    candidate["primary_target"] = llm_target_info.get("target")
                    candidate["mechanism"] = llm_target_info.get("mechanism")
                    candidate["target_confidence"] = llm_target_info.get("confidence", "medium")
                        
            except Exception as e:
                logger.warning(f"LLM target lookup failed for {drug_name}: {e}")
                
            
            # Get ownership info from web sources if not already set
            if not candidate.get("current_owner") or candidate.get("current_owner") == "Unknown":
                try:
                    # Use shelving reason investigator to find company info
                    if self.shelving_reason_investigator:
                        shelving_info = await self.shelving_reason_investigator.investigate_shelving_reason(drug_name)
                        if shelving_info and shelving_info.get("company"):
                            candidate["current_owner"] = shelving_info["company"]
                        
                        # Extract discontinuation reason if not already set
                        if not candidate.get("discontinuation_info") and shelving_info:
                            candidate["discontinuation_info"] = {
                                "reason": shelving_info.get("primary_reason", "Unknown"),
                                "details": shelving_info.get("detailed_analysis", "")
                            }
                except Exception as e:
                    logger.warning(f"Web enrichment failed for {drug_name}: {e}")
        
        return candidates
    
    async def _discover_global_assets(self, disease: str, target: str = None) -> Dict[str, List[Dict]]:
        """Discover drug assets globally using web crawler"""
        try:
            if self.asset_webcrawler:
                async with self.asset_webcrawler as crawler:
                    return await crawler.discover_global_assets(
                        disease=disease,
                        target=target,
                        include_patents=True,
                        include_press_releases=True,
                        limit=50
                    )
        except Exception as e:
            print(f"Global asset discovery failed: {e}")
        
        return {}
    
    def _ingredient_key(self, name: str) -> str:
        """Normalize ingredient (strip common salt suffixes, lowercase)."""
        if not name:
            return ""
        n = name.lower().strip()
        n = re.sub(r"\s+(hydrochloride|disodium|sodium|mesylate|succinate|phosphate|tartrate|citrate|sulfate|acetate|maleate|fumarate|tosylate)\b", "", n)
        n = re.sub(r"\s+", " ", n)
        return n

    def _df_route_key(self, dosage_form: str, route: str) -> str:
        df = (dosage_form or "").strip().lower()
        rt = (route or "").strip().lower()
        return f"{df};{rt}" if df or rt else ""

    async def _find_discontinued_formulations_from_ob(self, disease: str, base_candidates: List[Dict]) -> List[Dict]:
        """
        Build 'discontinued formulation' candidates from Orange Book,
        even when the overall ingredient is still marketed.
        """
        # 1) limit to ingredients we care about (appear in your disease trials/candidates)
        interest_set = set()
        for c in base_candidates:
            nm = c.get("drug_name") or c.get("drug") or ""
            if nm:
                interest_set.add(self._ingredient_key(nm))

        results = []
        seen = set()

        for c in base_candidates:
            ingr_in = c.get("drug_name") or c.get("drug") or ""
            ingr = self._ingredient_key(ingr_in)
            if not ingr or ingr not in interest_set:
                continue

            # 2) pull Orange Book snapshot
            try:
                reg = await self.drug_safety_client.get_regulatory_status(ingr_in)
            except Exception:
                reg = {}
            ob = (reg or {}).get("orange_book") or {}

            active = ob.get("active_products") or []
            discontinued = ob.get("discontinued_products") or []

            # normalize to dict list
            if not isinstance(active, list): active = []
            if not isinstance(discontinued, list): discontinued = []

            # 3) index by DF;Route cluster
            clusters = defaultdict(lambda: {"active": [], "disc": []})

            def add_row(row, kind: str):
                if not isinstance(row, dict): return
                dfk = self._df_route_key(row.get("dosage_form"), row.get("route"))
                clusters[dfk][kind].append(row)

            for r in active: add_row(r, "active")
            for r in discontinued: add_row(r, "disc")

            withdrawn_for_safety = bool(reg.get("withdrawn_for_safety_or_efficacy"))

            # 4) build formulation candidates from each discontinued row
            for dfk, g in clusters.items():
                total = len(g["active"]) + len(g["disc"])
                if total == 0:
                    continue

                discontinued_share = len(g["disc"]) / total
                applicants_disc = { (r.get("applicant") or "").strip().lower() for r in g["disc"] if isinstance(r, dict) }
                multi_sponsor_disc = len({a for a in applicants_disc if a}) >= 2

                # find if any discontinued row looks like originator (NDA or RLD)
                originator_disc = any(
                    (str(r.get("application_type") or "").upper() == "NDA") or
                    (str(r.get("rld") or "").upper() == "YES")
                    for r in g["disc"] if isinstance(r, dict)
                )

                for row in g["disc"]:
                    if not isinstance(row, dict):
                        continue
                    # skip safety withdrawals
                    if withdrawn_for_safety:
                        continue

                    strength = (row.get("strength") or "").strip()
                    app_no = (row.get("application_number") or row.get("app_no") or "").strip()
                    applicant = (row.get("applicant") or "Unknown").strip()
                    dosage_form = (row.get("dosage_form") or "").strip()
                    route = (row.get("route") or "").strip()

                    key = (ingr, dosage_form, route, strength, applicant, app_no)
                    if key in seen:
                        continue
                    seen.add(key)

                    cand = {
                        "drug_name": ingr,                     # normalized ingredient
                        "display_name": ingr_in,               # original
                        "category": "discontinued_formulation",
                        "program_status": "formulation_discontinued",
                        "ob_application_number": app_no,
                        "ob_application_type": row.get("application_type"),
                        "ob_product_number": row.get("product_number"),
                        "ob_applicant": applicant,
                        "ob_strength": strength,
                        "ob_df_route": f"{dosage_form};{route}",
                        "ob_marketing_status": (row.get("marketing_status") or "Discontinued"),
                        "ob_te_code": row.get("te_code"),
                        "originator_discontinued": originator_disc,
                        "multi_sponsor_discontinued": multi_sponsor_disc,
                        "discontinued_share_in_df_route": round(discontinued_share, 3),
                        "sources": ["orange_book"],
                        # make it clear we're *not* declaring the whole drug shelved
                        "is_currently_marketed": bool(len(g["active"]) > 0),
                        "notes": "Orange Book: discontinued product-level row; ingredient still has active products."
                    }

                    # simple score for formulations
                    score = 0.0
                    if discontinued_share >= 0.5: score += 0.35
                    if originator_disc:            score += 0.30
                    if multi_sponsor_disc:         score += 0.20
                    # relevance boost if we saw this ingredient in disease trials already
                    if self._ingredient_key(c.get("drug_name")) in interest_set: score += 0.15
                    cand["formulation_rescue_score"] = min(1.0, score)

                    results.append(cand)

        return results

    async def _flag_discontinued_via_ob_pb(self, candidates: List[Dict]) -> List[Dict]:
        """
        Attach OB/PB discontinuation flags and mark likely non-safety shelves.
        Uses existing drug_safety_client + purple_book_parser.
        """
        out = []
        for c in candidates:
            name = c.get("drug_name") or c.get("drug") or ""
            if not name:
                out.append(c)
                continue

            # 1) Regulatory snapshot (Orange Book + safety withdrawal)
            try:
                reg = await self.drug_safety_client.get_regulatory_status(name)
            except Exception:
                reg = {}

            ob = (reg or {}).get("orange_book") or {}
            # normalize counts if your client doesn't return them
            ob_active_ct = ob.get("active_products_count") or len(ob.get("active_products", []) or [])
            ob_disc_ct = ob.get("discontinued_products_count") or len(ob.get("discontinued_products", []) or [])

            # More nuanced Orange Book discontinuation logic
            # Flag if there are significant discontinued products (potential high-value formulations)
            total_products = ob_active_ct + ob_disc_ct
            discontinuation_ratio = ob_disc_ct / total_products if total_products > 0 else 0
            
            # Flag as discontinued asset if:
            # 1. Completely discontinued (no active, some discontinued) OR
            # 2. High discontinuation ratio (>30%) with multiple discontinued products
            ob_discontinued_asset = (
                (ob_active_ct == 0 and ob_disc_ct > 0) or  # Completely discontinued
                (ob_disc_ct >= 3 and discontinuation_ratio > 0.3)  # Significant discontinuations
            )

            # 2) Purple Book (you already parsed it at init)
            pb = {}
            if self.purple_book_parser:
                pb_info = await self._check_purple_book_status(name)  # your helper
                pb = pb_info or {}
            pb_discontinued_asset = bool(pb and (pb.get("is_active") is False or
                                                str(pb.get("purple_book_status","")).lower() in {"withdrawn","discontinued"}))

            # 3) Reason sanity (prefer non-safety)
            withdrawn_for_safety = bool(reg.get("withdrawn_for_safety_or_efficacy"))

            # 4) Use LLM to understand context and classify discontinuation reason
            discontinuation_reason = "Unknown"
            if self.llm_client and (ob_discontinued_asset or pb_discontinued_asset):
                try:
                    llm_prompt = f"""Analyze the discontinuation context for drug "{name}":
                    
                    Orange Book Status: {"Discontinued" if ob_discontinued_asset else "Active"}
                    Purple Book Status: {"Discontinued" if pb_discontinued_asset else "Active"}
                    Safety Withdrawal: {"Yes" if withdrawn_for_safety else "No"}
                    
                    Provide a specific discontinuation reason from these categories:
                    - "sponsor_decision": Company strategic decision to discontinue
                    - "business_pivot": Commercial pivot to other indications/markets
                    - "development_pause": Temporary pause in development
                    - "portfolio_prioritization": Focus shifted to other assets
                    - "market_competition": Competitive landscape changes
                    - "regulatory_hurdles": Regulatory pathway challenges
                    - "manufacturing_issues": Production or supply problems
                    - "safety_concerns": Safety or efficacy issues
                    - "funding_constraints": Financial/resource limitations
                    - "partnership_ended": Collaboration or licensing ended
                    - "unknown": Insufficient information
                    
                    Return only the specific reason category."""
                    
                    messages = [{"role": "user", "content": llm_prompt}]
                    response = await self.llm_client.chat("gpt-4o-mini", messages, temperature=0.1, max_tokens=50)
                    llm_response = response.choices[0].message.content
                    discontinuation_reason = llm_response.strip().lower()
                    valid_reasons = [
                        "sponsor_decision", "business_pivot", "development_pause", 
                        "portfolio_prioritization", "market_competition", "regulatory_hurdles",
                        "manufacturing_issues", "safety_concerns", "funding_constraints",
                        "partnership_ended", "unknown"
                    ]
                    if discontinuation_reason not in valid_reasons:
                        discontinuation_reason = "unknown"
                        
                except Exception as e:
                    print(f"LLM discontinuation analysis failed for {name}: {e}")

            # 5) Decide + annotate
            is_discontinued = ob_discontinued_asset or pb_discontinued_asset
            if is_discontinued:
                c["program_status"] = c.get("program_status") or "discontinued"
                c.setdefault("discontinuation_info", {})
                c["discontinuation_info"]["ob_discontinued"] = ob_discontinued_asset
                c["discontinuation_info"]["pb_discontinued"] = pb_discontinued_asset
                c["discontinuation_info"]["reason"] = discontinuation_reason
                c["discontinuation_info"]["withdrawn_for_safety"] = withdrawn_for_safety
                
                # Mark as non-safety discontinued if not withdrawn for safety
                if not withdrawn_for_safety and discontinuation_reason not in ["safety_concerns", "unknown"]:
                    c["non_safety_discontinued"] = True
                    c["specific_discontinuation_reason"] = discontinuation_reason
                elif not withdrawn_for_safety and discontinuation_reason in ["business", "other"]:
                    # Map generic reasons to more specific ones
                    c["non_safety_discontinued"] = True
                    if discontinuation_reason == "business":
                        c["specific_discontinuation_reason"] = "sponsor_decision"
                    else:
                        c["specific_discontinuation_reason"] = "development_pause"
            
            # Always set discontinuation info for tracking
            c.setdefault("discontinuation_info", {})
            c["discontinuation_info"]["ob_discontinued"] = ob_discontinued_asset
            c["discontinuation_info"]["pb_discontinued"] = pb_discontinued_asset
            c["discontinuation_info"]["reason"] = discontinuation_reason
            
            # Set specific reason for all candidates
            if discontinuation_reason != "Unknown":
                # Map LLM classifications to specific reasons
                if discontinuation_reason == "business":
                    c["specific_discontinuation_reason"] = "sponsor_decision"
                elif discontinuation_reason == "other":
                    c["specific_discontinuation_reason"] = "development_pause"
                elif discontinuation_reason == "safety_concerns":
                    c["specific_discontinuation_reason"] = "safety_issues"
                else:
                    c["specific_discontinuation_reason"] = discontinuation_reason
            elif discontinuation_reason == "Unknown" and c.get("program_status") == "development_discontinued":
                # Try to infer reason from program status and other data
                c["specific_discontinuation_reason"] = "development_pause"
            else:
                # Default fallback
                c["specific_discontinuation_reason"] = "development_pause"
            
            out.append(c)
        
        return out
    
    async def _resolve_study_names_to_drugs(self, candidates: List[Dict]) -> List[Dict]:
        """Resolve study names/trial codes to actual drug names using Orange/Purple Book verification"""
        resolved_candidates = []
        
        for candidate in candidates:
            drug_name = candidate.get('drug_name', '')
            
            # First check if drug exists in Orange/Purple Book (FDA approved)
            ob_match = self._check_orange_book_match(drug_name)
            pb_match = self._check_purple_book_match(drug_name)
            
            if ob_match or pb_match:
                # Drug is FDA approved - update status
                candidate['program_status'] = 'fda_approved'
                candidate['specific_discontinuation_reason'] = 'fda_approved'
                candidate['fda_verified'] = True
                
                if ob_match:
                    candidate['orange_book_match'] = ob_match
                if pb_match:
                    candidate['purple_book_match'] = pb_match
            else:
                # Not in FDA databases - likely study name or investigational drug
                # Only use LLM for non-FDA approved candidates with missing data
                if (not candidate.get('primary_target') or 
                    candidate.get('current_owner') == 'Unknown') and self.llm_client:
                    
                    try:
                        prompt = f"""
                        "{drug_name}" is not FDA approved. Identify if this is:
                        1. A study name (like WU-KONG6 for sunvozertinib)
                        2. An investigational drug name
                        
                        If study name, provide the actual drug being tested.
                        
                        JSON response:
                        {{
                            "is_study_name": true/false,
                            "actual_drug_name": "name if different",
                            "target": "mechanism",
                            "developer": "company"
                        }}
                        """
                        
                        response = await self.llm_client.chat(
                            model="gpt-4o-mini",
                            messages=[{"role": "user", "content": prompt}]
                        )
                        
                        try:
                            import json
                            import re
                            # Handle new OpenAI API response format
                            if hasattr(response, 'choices'):
                                content = response.choices[0].message.content
                            else:
                                content = response.get('content', '')
                            
                            # Extract JSON from markdown code blocks if present
                            json_match = re.search(r'```json\s*(\{.*?\})\s*```', content, re.DOTALL)
                            if json_match:
                                json_content = json_match.group(1)
                            else:
                                # Try to find JSON object directly
                                json_match = re.search(r'\{.*?\}', content, re.DOTALL)
                                if json_match:
                                    json_content = json_match.group(0)
                                else:
                                    json_content = content
                            
                            drug_info = json.loads(json_content)
                            
                            if drug_info.get('is_study_name') and drug_info.get('actual_drug_name'):
                                resolved_drug_name = drug_info.get('actual_drug_name')
                                candidate['drug_name'] = resolved_drug_name
                                candidate['study_name'] = drug_name
                                
                                # Check FDA approval for resolved drug name
                                ob_match_resolved = self._check_orange_book_match(resolved_drug_name)
                                pb_match_resolved = self._check_purple_book_match(resolved_drug_name)
                                
                                # Also check FDA drugsfda endpoint for approval
                                fda_approved = False
                                try:
                                    approval_info = await self.drug_safety_client._get_fda_approval_status(resolved_drug_name)
                                    fda_approved = approval_info.get('is_approved', False)
                                except Exception as e:
                                    logger.debug(f"FDA approval check failed for {resolved_drug_name}: {e}")
                                
                                if ob_match_resolved or pb_match_resolved or fda_approved:
                                    candidate['program_status'] = 'fda_approved'
                                    candidate['specific_discontinuation_reason'] = 'fda_approved'
                                    candidate['fda_verified'] = True
                                    
                                    if ob_match_resolved:
                                        candidate['orange_book_match'] = ob_match_resolved
                                    if pb_match_resolved:
                                        candidate['purple_book_match'] = pb_match_resolved
                                
                            # Fill missing data
                            if drug_info.get('target') and not candidate.get('primary_target'):
                                candidate['primary_target'] = drug_info.get('target')
                            if drug_info.get('developer') and candidate.get('current_owner') == 'Unknown':
                                candidate['current_owner'] = drug_info.get('developer')
                        
                        except (json.JSONDecodeError, KeyError):
                            pass
                            
                    except Exception as e:
                        print(f"Study name resolution failed for {drug_name}: {e}")
            
            resolved_candidates.append(candidate)
        
        return candidates
    
    async def _get_llm_target_info(self, drug_name: str) -> Optional[Dict]:
        """Get target and mechanism information using hybrid LLM + clinical trials approach"""
        
        # Step 1: Get clinical trials data for context
        clinical_context = ""
        try:
            trials = await self.ct_client.search_trials(drug_name, max_results=5)
            if trials:
                for trial in trials[:3]:
                    title = trial.get('title', '')
                    clinical_context += f"Clinical trial: {title}. "
                    
                    interventions = trial.get('interventions', [])
                    for intervention in interventions:
                        if drug_name.lower() in intervention.get('name', '').lower():
                            desc = intervention.get('description', '')
                            clinical_context += f"Intervention: {desc}. "
        except Exception as e:
            print(f"Clinical trials lookup failed for {drug_name}: {e}")
        
        # Step 2: Enhanced LLM prompt with clinical context for multi-target detection
        prompt = f"""Analyze the pharmaceutical compound '{drug_name}' using both your knowledge and the following clinical trial context.

Clinical Trial Context:
{clinical_context or "No clinical trial data available."}

Identify ALL molecular targets this compound acts on:
1. Primary target(s) - direct molecular interactions
2. Secondary targets - if it's part of combination therapy or has multiple mechanisms
3. The precise mechanism for each target
4. Your confidence level based on available evidence

Be specific about molecular targets (e.g., "EGFR tyrosine kinase", "PD-1 receptor", "VEGFR-2").

If clinical trials mention multiple targets (like "dual TIGIT and PD-1 blockade"), identify which target this specific drug acts on vs combination partners.

Look for patterns:
- Combination therapies (drug A + drug B targeting different pathways)
- Multi-target drugs (single drug hitting multiple targets)
- Drug class effects (mAb suffix patterns)

Only respond with "Not available" if you cannot find reliable information from either source.

Respond with ONLY this JSON structure:
{{
  "primary_target": "main molecular target",
  "secondary_targets": ["additional targets if any"],
  "all_targets": ["complete list of all targets"],
  "mechanism": "precise mechanism of action for all targets", 
  "confidence": "high/medium/low",
  "sources": ["llm_knowledge", "clinical_trials", "both"],
  "combination_context": "if used in combination, describe partner drugs and their targets"
}}
"""

        try:
            if not hasattr(self, 'openai_client') or not self.openai_client:
                return None
                
            response = await self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a pharmaceutical expert with access to clinical trial data. Use both sources to provide accurate target information. Respond with valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=300
            )
            
            content = response.output_text.strip()
            
            # Parse JSON response
            import json
            target_info = json.loads(content)
            
            # Validate confidence - upgrade to high if both sources agree on any target
            if clinical_context and target_info.get('primary_target') != 'Not available':
                context_lower = clinical_context.lower()
                
                # Check all targets against clinical context
                all_targets = target_info.get('all_targets', [])
                if isinstance(all_targets, str):
                    all_targets = [all_targets]
                
                target_keywords = ['pd-1', 'pd-l1', 'ctla-4', 'tigit', 'egfr', 'vegfr', 'her2', 'cd20', 'cd19']
                matches = 0
                
                for target in all_targets:
                    target_lower = str(target).lower()
                    for keyword in target_keywords:
                        if keyword in target_lower and keyword in context_lower:
                            matches += 1
                            break
                
                if matches > 0:
                    target_info['confidence'] = 'high'
                    target_info['sources'] = ['both']
                    
                # For backward compatibility, set 'target' field to primary_target
                target_info['target'] = target_info.get('primary_target')
            
            return target_info
            
        except Exception as e:
            logger.debug(f"LLM target info failed for {drug_name}: {e}")
            return None

    async def _get_llm_asset_owner(self, drug_name: str) -> Optional[Dict]:
        """Get actual asset owner/developer using LLM (not trial sponsor)"""
        if not self.openai_client:
            return None
            
        try:
            prompt = f"""You are a pharmaceutical research analyst. Find comprehensive information for '{drug_name}' including ownership history and development timeline.

SEARCH STRATEGY:
1. Search for "{drug_name} ownership history"
2. Search for "{drug_name} developer company" 
3. Search for "{drug_name} generic name brand name"
4. Search for "{drug_name} licensing agreements acquisitions"
5. Search for "{drug_name} regulatory approval India DCGI"
6. Search for "{drug_name} remogliflozin etabonate" (if applicable)
7. Check all drug name variants and brand names

For '{drug_name}', provide:
- Current asset owner (company with IP/development rights)
- Original developer/discoverer
- Complete ownership transfer chain if available
- Regional drug approvals (FDA, EMA, PMDA, NMPA, Health Canada, DCGI India)
- SPECIFIC DISEASE INDICATIONS for each regional approval

IMPORTANT: For each regional approval, find the specific disease(s) or indication(s) the drug was approved for. Include cancer types, specific conditions, etc.

Respond with ONLY this JSON:
{{
  "asset_owner": "verified current owner OR unknown",
  "original_developer": "verified original developer OR unknown", 
  "ownership_chain": "verified chain OR just current owner",
  "confidence": "high/medium/low",
  "notes": "source of information or reason for uncertainty",
  "regional_approvals": {{
    "fda": {{
      "approved": true/false,
      "indications": ["specific disease 1", "specific disease 2"] OR []
    }},
    "ema": {{
      "approved": true/false,
      "indications": ["specific disease 1", "specific disease 2"] OR []
    }},
    "pmda": {{
      "approved": true/false,
      "indications": ["specific disease 1", "specific disease 2"] OR []
    }},
    "nmpa": {{
      "approved": true/false,
      "indications": ["specific disease 1", "specific disease 2"] OR []
    }},
    "health_canada": {{
      "approved": true/false,
      "indications": ["specific disease 1", "specific disease 2"] OR []
    }},
    "dcgi": {{
      "approved": true/false,
      "indications": ["specific disease 1", "specific disease 2"] OR []
    }},
    "details": "brief summary of approvals found"
  }}
}}"""

            response = await asyncio.wait_for(
                self.openai_client.responses.create(
                    model="gpt-4o",
                    tools=[{
                        "type": "web_search_preview",
                        "search_context_size": "medium"
                    }],
                    input=prompt
                ),
                timeout=60.0  # 60 second timeout for accurate web search results
            )
            
            content = response.output_text.strip()
            
            # Parse JSON response
            import json
            try:
                asset_owner_info = json.loads(content)
            except json.JSONDecodeError:
                # Try to extract JSON from mixed text response
                import re
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    try:
                        asset_owner_info = json.loads(json_match.group())
                    except json.JSONDecodeError:
                        # If JSON parsing fails completely, create fallback response
                        asset_owner_info = {
                            "asset_owner": "Unknown",
                            "confidence": "low",
                            "notes": f"LLM could not identify owner: {content[:100]}...",
                            "regional_partners": []
                        }
                else:
                    # No JSON found, try to use clinical trials data as fallback
                    try:
                        trials = await self.ct_client.search_trials(drug_name, max_results=5)
                        company_sponsors = []
                        individual_sponsors = []
                        
                        for trial in trials:
                            sponsor = trial.get('sponsor', {}).get('lead_sponsor', {}).get('agency', '')
                            if sponsor and 'hospital' not in sponsor.lower() and 'university' not in sponsor.lower():
                                # Prioritize companies over individuals
                                if any(keyword in sponsor.lower() for keyword in ['inc', 'ltd', 'corp', 'company', 'pharmaceuticals', 'pharma', 'bio']):
                                    company_sponsors.append(sponsor)
                                else:
                                    individual_sponsors.append(sponsor)
                        if company_sponsors:
                            asset_owner_info = {
                                'asset_owner': company_sponsors[0],
                                'confidence': 'low',
                                'notes': f'Using clinical trials sponsor - LLM could not identify global developer',
                                'regional_partners': []
                            }
                        elif individual_sponsors:
                            asset_owner_info = {
                                'asset_owner': individual_sponsors[0],
                                'confidence': 'low',
                                'notes': f'Using clinical trials sponsor - LLM could not identify global developer',
                                'regional_partners': []
                            }
                        else:
                            asset_owner_info = {
                                "asset_owner": "Unknown",
                                "confidence": "low",
                                "notes": f"LLM could not identify owner: {content[:100]}...",
                                "regional_partners": []
                            }
                    except:
                        asset_owner_info = {
                            "asset_owner": "Unknown",
                            "confidence": "low", 
                            "notes": f"LLM could not identify owner: {content[:100]}...",
                            "regional_partners": []
                        }
            
            # Use LLM to validate potential name conflicts
            if asset_owner_info.get("confidence") == "high":
                validation_result = await self._validate_drug_owner_with_llm(drug_name, asset_owner_info.get("asset_owner"))
                if validation_result and validation_result.get("potential_conflict"):
                    asset_owner_info["confidence"] = "low"
                    asset_owner_info["notes"] += f" (WARNING: {validation_result.get('conflict_reason', 'Potential name conflict')})"
            
            # When there are regional partners, combine global developer with regional partners
            if asset_owner_info.get("regional_partners"):
                global_dev = asset_owner_info.get("asset_owner", "")
                regional = asset_owner_info.get("regional_partners", [])
                if regional and global_dev and global_dev not in regional:
                    # Combine as "global_developer/regional_partner"
                    asset_owner_info["asset_owner"] = f"{global_dev}/{regional[0]}"
                    
            return asset_owner_info
            
        except asyncio.TimeoutError:
            logger.debug(f"LLM asset owner lookup timed out for {drug_name}")
            return None
        except Exception as e:
            logger.debug(f"LLM asset owner lookup failed for {drug_name}: {e}")
            return None           # Fallback to clinical trials sponsor if available
            try:
                sponsors = await self._get_clinical_trials_sponsors(drug_name)
                if sponsors:
                    # Filter out individual names, prefer companies
                    company_sponsors = [s for s in sponsors if not self._is_individual_name(s)]
                    sponsor_name = company_sponsors[0] if company_sponsors else sponsors[0]
                    
                    return {
                        'asset_owner': sponsor_name,
                        'confidence': 'low',
                        'notes': f'LLM failed, using clinical trials sponsor',
                        'regional_partners': []
                    }
            except:
                pass
                
            return None
    
    async def _get_best_trial_sponsor(self, drug_name: str) -> Optional[str]:
        """Get the best trial sponsor, prioritizing pharmaceutical companies over research groups"""
        try:
            trials = await self.ct_client.search_trials(intervention=drug_name, max_results=10)
            
            sponsors = []
            for trial in trials:
                sponsor_info = trial.get('sponsor', {})
                lead_sponsor = sponsor_info.get('lead_sponsor', {})
                collaborators = sponsor_info.get('collaborators', [])
                
                # Add lead sponsor
                lead_agency = lead_sponsor.get('agency', '')
                if lead_agency:
                    sponsors.append(lead_agency)
                
                # Add collaborators
                for collab in collaborators:
                    collab_agency = collab.get('agency', '')
                    if collab_agency:
                        sponsors.append(collab_agency)
            
            if not sponsors:
                return None
            
            # Score sponsors - prioritize pharmaceutical companies
            scored_sponsors = []
            for sponsor in set(sponsors):  # Remove duplicates
                score = self._score_sponsor_as_pharma_company(sponsor)
                scored_sponsors.append((sponsor, score))
            
            # Sort by score (highest first) and return best
            scored_sponsors.sort(key=lambda x: x[1], reverse=True)
            return scored_sponsors[0][0] if scored_sponsors else None
            
        except Exception as e:
            logger.debug(f"Failed to get best trial sponsor for {drug_name}: {e}")
            return None
    
    def _score_sponsor_as_pharma_company(self, sponsor: str) -> float:
        """Score a sponsor based on how likely it is to be a pharmaceutical company"""
        sponsor_lower = sponsor.lower()
        
        # High score for known pharma companies
        pharma_keywords = [
            'pharmaceutical', 'pharma', 'biotech', 'therapeutics', 'medicines',
            'biopharmaceutical', 'labs', 'laboratory', 'inc', 'ltd', 'corp',
            'corporation', 'company', 'ag', 'sa', 'gmbh', 'boehringer', 'roche',
            'novartis', 'pfizer', 'merck', 'bristol', 'astrazeneca', 'sanofi',
            'gsk', 'glaxo', 'eli lilly', 'abbvie', 'amgen', 'gilead', 'biogen'
        ]
        
        # Low score for academic/research institutions
        academic_keywords = [
            'university', 'hospital', 'medical center', 'research center',
            'institute', 'foundation', 'consortium', 'group', 'society',
            'organization', 'association', 'network', 'cooperative',
            'aktion', 'eortc', 'cancer center', 'clinic'
        ]
        
        score = 0.0
        
        # Add points for pharma keywords
        for keyword in pharma_keywords:
            if keyword in sponsor_lower:
                if keyword in ['boehringer', 'roche', 'novartis', 'pfizer']:
                    score += 10.0  # Major pharma companies
                elif keyword in ['pharmaceutical', 'pharma', 'biotech']:
                    score += 5.0   # Strong pharma indicators
                else:
                    score += 2.0   # General company indicators
        
        # Subtract points for academic keywords
        for keyword in academic_keywords:
            if keyword in sponsor_lower:
                score -= 3.0
        
        # Bonus for common pharma company patterns
        if any(pattern in sponsor_lower for pattern in [' inc', ' ltd', ' corp', ' ag', ' gmbh']):
            score += 1.0
        
        return max(score, 0.0)  # Don't go negative
    
    async def _validate_drug_owner_with_llm(self, drug_name: str, proposed_owner: str) -> Optional[Dict]:
        """Use LLM to validate if there are potential name conflicts for drug ownership"""
        if not self.openai_client:
            return None
            
        try:
            prompt = f"""Validate the drug ownership assignment: "{drug_name}" → "{proposed_owner}"

Check for potential conflicts:
1. Are there multiple drugs with the same or similar names from different companies?
2. Is this a common drug name that could refer to different compounds?
3. Are there known naming conflicts in pharmaceutical databases?

Examples of conflicts to detect:
- "Zimberelimab" could be GLS-010 (Guangzhou Gloria) OR AB928 (Arcus Biosciences)
- Generic names vs brand names from different companies
- Similar drug codes from different regions/companies

Respond with ONLY this JSON:
{{
  "potential_conflict": true/false,
  "conflict_reason": "brief explanation if conflict detected",
  "confidence": "high/medium/low"
}}"""

            response = await self.openai_client.chat.completions.create(
                model="gpt-4o-mini",  # Use faster model for validation
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=0.1
            )
            
            content = response.output_text.strip()
            
            # Parse JSON response
            import json
            validation_result = json.loads(content)
            
            return validation_result
            
        except Exception as e:
            logger.debug(f"LLM validation failed for {drug_name}: {e}")
            return None
    
    def _check_orange_book_match(self, drug_name: str) -> Optional[Dict]:
        """Check if drug exists in Orange Book"""
        if not hasattr(self, 'orange_book_data') or not self.orange_book_data:
            return None
            
        # Check generic and brand names
        for entry in self.orange_book_data:
            if (drug_name.lower() in entry.get('ingredient', '').lower() or
                drug_name.lower() in entry.get('trade_name', '').lower()):
                return entry
        return None
    
    def _check_purple_book_match(self, drug_name: str) -> Optional[Dict]:
        """Check if drug exists in Purple Book with exact matching only"""
        if not self.purple_book_parser or not self.purple_book_parser.biologics_data:
            return None
            
        drug_name_lower = drug_name.lower().strip()
        
        # Extract drug names from parenthetical formats like "Avastin (bevacizumab)"
        drug_variants = [drug_name_lower]
        if '(' in drug_name_lower and ')' in drug_name_lower:
            # Extract both the main name and the parenthetical name
            main_name = drug_name_lower.split('(')[0].strip()
            paren_name = drug_name_lower.split('(')[1].split(')')[0].strip()
            drug_variants.extend([main_name, paren_name])
        
        for entry in self.purple_book_parser.biologics_data:
            proper_name = entry.get('proper_name', '').lower().strip()
            proprietary_name = entry.get('proprietary_name', '').lower().strip()
            
            # Check all variants for exact matches
            for variant in drug_variants:
                if (proper_name == variant or proprietary_name == variant):
                    return entry
                
                # Also check if variant matches the base name of proper_name (before suffix)
                if proper_name and '-' in proper_name:
                    base_proper_name = proper_name.split('-')[0].strip()
                    if base_proper_name == variant:
                        return entry
                
        return None
    
    def _fuzzy_match(self, s1: str, s2: str, max_edits: int = 2) -> bool:
        """Check if two strings are similar within max edit distance"""
        if not s1 or not s2 or abs(len(s1) - len(s2)) > max_edits:
            return False
            
        # Simple Levenshtein distance calculation
        if len(s1) < len(s2):
            s1, s2 = s2, s1
            
        if len(s2) == 0:
            return len(s1) <= max_edits
            
        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
            
        return previous_row[-1] <= max_edits
    
    async def _search_patent_assets(self, disease: str, target: Optional[str] = None) -> List[Dict]:
        """Search for drug assets that only exist in patents"""
        try:
            if not self.patent_crawler:
                return []
            
            async with self.patent_crawler as pc:
                patents = await pc.search_patent_assets(
                    query=disease,
                    disease=disease,
                    target=target,
                    limit=100,
                )
                return patents
        except Exception as e:
            return []
    
    async def _enrich_candidate_parallel(self, candidate: Dict) -> Dict:
        """Enrich a single candidate with data from all sources in parallel"""
        drug_name = candidate.get("drug_name", "")
        
        # First, resolve drug code if needed (must complete before regulatory check)
        results = {}
        regional_approvals = None
        
        # 1. Drug resolution (code to generic name) - do this first
        if self._is_drug_code(drug_name):
            try:
                resolution = await self.drug_resolver.resolve_to_chembl_id(drug_name)
                results["drug_resolution"] = resolution
                # Extract regional approvals if available
                if resolution and isinstance(resolution, dict):
                    regional_approvals = resolution.get("regional_approvals")
            except Exception as e:
                logger.warning(f"Failed drug_resolution for {drug_name}: {e}")
                results["drug_resolution"] = None
        
        # Prepare remaining parallel tasks
        tasks = {}
        
        # 2. ChEMBL data if not already present
        if not candidate.get("chembl_id") and drug_name:
            tasks["chembl"] = self.chembl_client.get_compound_by_name(drug_name)
        
        # 3. Get ownership and patent info
        tasks["pharma_intel"] = self.pharma_intel_client.get_asset_availability_status(
            drug_name, candidate.get("chembl_id")
        )
        
        # 4. Get regulatory status with regional approvals
        tasks["regulatory"] = self.drug_safety_client.get_regulatory_status(drug_name, regional_approvals)
        
        # 5. Get shelving reasons
        if self.shelving_reason_investigator and drug_name:
            tasks["shelving"] = self.shelving_reason_investigator.investigate_shelving_reason(drug_name)
        
        # Execute remaining tasks in parallel
        for key, task in tasks.items():
            try:
                results[key] = await task
            except Exception as e:
                logger.warning(f"Failed {key} for {drug_name}: {e}")
                results[key] = None
        
        # Process drug resolution results
        if results.get("drug_resolution"):
            resolution = results["drug_resolution"]
            if resolution.get("pref_name"):
                candidate["resolved_name"] = resolution["pref_name"]
                candidate["chembl_id"] = resolution.get("chembl_id")
                # Update drug_name if it was a code
                if self._is_drug_code(drug_name):
                    candidate["original_code"] = drug_name
                    candidate["drug_name"] = resolution["pref_name"]
        
        # Process ChEMBL results
        if results.get("chembl"):
            chembl_data = results["chembl"]
            candidate["chembl_id"] = chembl_data.get("molecule_chembl_id")
            candidate["pref_name"] = chembl_data.get("pref_name")
            candidate["max_phase"] = chembl_data.get("max_phase", 0)
            candidate["first_approval"] = chembl_data.get("first_approval")
            
            # Get targets if available - skip for now as ChEMBL targets lookup is complex
            # TODO: Implement proper target lookup using bioactivities or mechanisms
        
        # Process ownership/pharma intel results
        asset_status = results.get("pharma_intel", {}) or {}
        reg_status = results.get("regulatory", {}) or {}
        
        # Extract ownership info
        ownership_history = asset_status.get("ownership_history", []) or []
        originator_company = None
        for evt in ownership_history:
            if evt.get("event_type", "").lower() == "original_development" and evt.get("company"):
                originator_company = evt.get("company")
                break
        
        if not originator_company and ownership_history:
            originator_company = ownership_history[0].get("company")
        if not originator_company:
            originator_company = candidate.get("sponsor")
        
        # Build ownership info
        candidate["ownership_info"] = {
            "asset_availability": asset_status.get("availability_status"),
            "ownership_history": ownership_history,
            "originator_company": originator_company,
            "patent_status": asset_status.get("patent_status", {}),
            "licensing_opportunities": asset_status.get("licensing_opportunities", {}),
            "recent_transactions": asset_status.get("recent_transactions", []),
            "regulatory": {
                "fda_approved": reg_status.get("is_approved"),
                "original_approval_date": reg_status.get("original_approval_date"),
                "is_currently_marketed": reg_status.get("is_currently_marketed"),
                "withdrawn_for_safety_or_efficacy": reg_status.get("withdrawn_for_safety_or_efficacy"),
                "marketing_authorization_holders": reg_status.get("marketing_authorization_holders", []),
                "orange_book": reg_status.get("orange_book")
            },
            "clinical_trials": asset_status.get("clinical_trials_status", {})
        }
        
        # Process shelving reasons
        if results.get("shelving"):
            shelving_info = results["shelving"]
            candidate["discontinuation_info"] = shelving_info
            reason = shelving_info.get("reason", "").lower()
            candidate["non_safety_discontinued"] = not any(
                term in reason for term in ["safety", "toxicity", "adverse", "death"]
            )
        
        # Classify drug status using all available data
        clinical_trials_data = {
            "trials": [candidate]  # Use the candidate as trial data
        }
        
        # Build proper regulatory data for classification
        regulatory_data_for_classifier = {
            "fda_approved": reg_status.get("is_approved", False),
            "ema_approved": reg_status.get("ema_approved"),
            "is_currently_marketed": reg_status.get("is_currently_marketed", False),
            "withdrawn_for_safety_or_efficacy": reg_status.get("withdrawn_for_safety_or_efficacy", False)
        }
        
        status_result = self.status_classifier.classify_drug_status(
            drug_name=candidate.get("drug_name", ""),
            clinical_trials_data=clinical_trials_data,
            regulatory_data=regulatory_data_for_classifier,
            shelving_reason_data=results.get("shelving"),
            last_activity_date=candidate.get("last_update_posted_date")
        )
        
        # Add status classification to candidate
        candidate["program_status"] = status_result["program_status"]
        
        # Override status if FDA approved and not withdrawn
        if reg_status.get("is_approved") and not reg_status.get("withdrawn_for_safety_or_efficacy"):
            if reg_status.get("is_currently_marketed"):
                candidate["program_status"] = "marketed"
            else:
                candidate["program_status"] = "approved"
        candidate["discontinuation_reason"] = status_result.get("reason")
        candidate["status_evidence"] = status_result["evidence"]
        candidate["regional_status"] = status_result["regions"]
        candidate["is_repurposing_candidate"] = self.status_classifier.is_repurposing_candidate(status_result)
        
        return candidate
    
    def _is_drug_code(self, name: str) -> bool:
        """Check if a drug name looks like a code (e.g., JNJ-67896049)"""
        # Pattern for drug codes: company prefix + numbers/letters
        code_patterns = [
            r'^[A-Z]{2,4}-\d+',  # GSK-123456, JNJ-123456
            r'^[A-Z]{2,4}\d+',   # GSK123456
            r'^[A-Z]+-[A-Z0-9]+$',  # TPI-287, TV-46017
            r'^[A-Z]{2}\d{4,}',  # LY2886721
        ]
        
        name_upper = name.strip().upper()
        for pattern in code_patterns:
            if re.match(pattern, name_upper):
                return True
        return False
    
    def _deduplicate_candidates(self, candidates: List[Dict]) -> List[Dict]:
        """Remove duplicate drug candidates based on drug name"""
        seen_drugs = {}
        deduplicated = []
        
        for candidate in candidates:
            drug_name = candidate.get('drug_name', '').strip()
            if not drug_name:
                continue
            
            # Extract base drug name (remove parenthetical variations like "(dose 1)", "(RP2D)")
            base_drug_name = drug_name.split('(')[0].strip().lower()
            if not base_drug_name:
                continue
                
            # If we haven't seen this drug, add it
            if base_drug_name not in seen_drugs:
                seen_drugs[base_drug_name] = candidate
                deduplicated.append(candidate)
            else:
                # If we have seen it, merge information if the new candidate has better data
                existing = seen_drugs[base_drug_name]
                
                # Prefer candidates with higher phase
                existing_phase = existing.get('max_phase', 0) or 0
                new_phase = candidate.get('max_phase', 0) or 0
                
                if new_phase > existing_phase:
                    # Replace with higher phase candidate
                    deduplicated[deduplicated.index(existing)] = candidate
                    seen_drugs[base_drug_name] = candidate
                elif new_phase == existing_phase:
                    # Merge sponsor information if different
                    existing_sponsor = existing.get('sponsor', '')
                    new_sponsor = candidate.get('sponsor', '')
                    if new_sponsor and new_sponsor != existing_sponsor:
                        # Keep both sponsors in a list
                        if isinstance(existing_sponsor, list):
                            if new_sponsor not in existing_sponsor:
                                existing_sponsor.append(new_sponsor)
                        else:
                            existing['sponsor'] = [existing_sponsor, new_sponsor] if existing_sponsor else [new_sponsor]
                    
                    # Merge scores by taking maximum
                    for score_field in ['score', 'failure_bonus', 'target_specificity_score']:
                        if score_field in candidate:
                            existing[score_field] = max(
                                existing.get(score_field, 0) or 0,
                                candidate.get(score_field, 0) or 0
                            )
        
        return deduplicated
    
    async def _analyze_target(self, target_info) -> Dict:
        """Analyze a specific target for drug discovery potential"""
        # Handle both string and dict inputs
        if isinstance(target_info, str):
            target_name = target_info
        else:
            target_name = target_info.get("target", "Unknown")
        
        analysis = {
            "target": target_name,
            "druggability": "Unknown",
            "existing_inhibitors": 0,
            "clinical_compounds": 0,
            "pdb_structures": [],
            "pathway": "Unknown"
        }
        
        try:
            # Get ChEMBL data for target
            inhibitors = await self.chembl_client.get_inhibitors_for_target(target_name)
            if inhibitors:
                analysis["existing_inhibitors"] = len(inhibitors)
                analysis["clinical_compounds"] = sum(
                    1 for inh in inhibitors 
                    if inh.get("max_phase") is not None and inh.get("max_phase", 0) >= 1
                )    
                
            # Get PDB structures for the target
            pdb_structures = await self.chembl_client.get_pdb_structures_for_target(target_name)
            analysis["pdb_structures"] = pdb_structures[:3]  # Keep top 3 structures
                
        except Exception as e:
            logger.warning(f"Failed to analyze target {target_name}: {e}")
        
        return analysis
    
    async def _get_candidate_safety_profiles(self, candidates: List[Dict]) -> List[Dict]:
        """Get comprehensive safety profiles for candidates"""
        safety_profiles = []
        
        for candidate in candidates:
            drug_name = candidate.get("drug_name", "")
            if drug_name:
                try:
                    # Get safety data from multiple sources with timeout
                    profile = await asyncio.wait_for(
                        self.drug_safety_client.get_comprehensive_safety_profile(drug_name),
                        timeout=10.0  # 10 second timeout per drug
                    )
                    profile["drug_name"] = drug_name
                    profile["chembl_id"] = candidate.get("chembl_id")
                    safety_profiles.append(profile)
                except asyncio.TimeoutError:
                    logger.warning(f"Safety profile lookup timed out for {drug_name}")
                    safety_profiles.append({
                        "drug_name": drug_name,
                        "error": "Timeout"
                    })
                except Exception as e:
                    logger.warning(f"Failed to get safety profile for {drug_name}: {e}")
                    safety_profiles.append({
                        "drug_name": drug_name,
                        "error": str(e)
                    })
        
        return safety_profiles
    
    async def _categorize_drug_opportunities(self, candidates: List[Dict], disease: str) -> Dict[str, List[Dict]]:
        """Categorize candidates into drug discovery vs rescue opportunities"""
        
        print(f"\n🔍 Categorizing {len(candidates)} candidates...")
        
        # Check if standard-of-care filter is available
        try:
            # Get standard-of-care drugs for the disease
            soc_drugs = set()
            if hasattr(self.ct_client, 'is_standard_of_care'):
                for candidate in candidates:
                    drug_name = candidate.get("drug_name", "")
                    if drug_name and await self.ct_client.is_standard_of_care(drug_name, disease):
                        soc_drugs.add(drug_name.lower())
        except Exception as e:
            logger.warning(f"Failed to check standard-of-care status: {e}")
            soc_drugs = set()
        
        drug_discovery = []
        drug_rescue = []
        shelved_assets = []  # For truly shelved/abandoned drugs
        filtered_out = []    # For FDA-approved or marketed drugs
        
        for i, candidate in enumerate(candidates):
            drug_name = candidate.get("drug_name", "")
            program_status = candidate.get("program_status", "").lower()
            
            # Removed debug output for cleaner logs
            
            # Filter out non-drug candidates (generic terms, procedures, targets/genes)
            non_drug_terms = [
                "chemotherapy", "adjuvant chemotherapy", "pre-intervention medication", 
                "post-intervention medication", "not specified", "investigational drug",
                "standard chemotherapy", "combination chemotherapy", "radiation therapy",
                "surgery", "placebo", "best supportive care"
            ]
            
            # Filter out common target/gene names that get picked up by web crawling
            target_gene_names = [
                "ros1", "alk", "egfr", "kras", "braf", "met", "ret", "ntrk", "fgfr",
                "vegfr", "pdgfr", "her2", "erbb2", "pik3ca", "akt", "mtor", "jak",
                "stat", "bcl2", "mcl1", "mdm2", "p53", "rb1", "cdkn2a", "pten",
                "tp53", "apc", "brca1", "brca2", "atm", "chek2", "palb2", "rad51",
                "pd-1", "pd-l1", "ctla-4", "lag-3", "tim-3", "tigit", "cd47",
                "phase", "trial", "study"  # Common trial terms
            ]
            
            drug_name_lower = drug_name.lower().strip()
            
            if (any(term in drug_name_lower for term in non_drug_terms) or
                drug_name_lower in target_gene_names or
                len(drug_name_lower) <= 2):  # Filter very short names
                candidate['filtered_reason'] = 'non_drug_candidate'
                filtered_out.append(candidate)
                continue
            
            # Filter out standard-of-care drugs first
            if drug_name.lower() in soc_drugs:
                candidate['filtered_reason'] = 'standard_of_care'
                filtered_out.append(candidate)
                continue
                
            # Special handling for preclinical compounds - bypass all filtering and scoring
            if (candidate.get("development_stage") == "preclinical" or 
                candidate.get("source") == "llm_web_search_preclinical"):
                
                # Calculate score for tracking but always put in drug_discovery
                high_potential_score = self._calculate_high_potential_score(candidate)
                candidate['high_potential_score'] = high_potential_score
                
                # All preclinical compounds go to drug_discovery to appear in preclinical assets
                drug_discovery.append(candidate)
                logger.info(f"🧪 Preclinical compound {drug_name} added to drug_discovery (score: {high_potential_score:.3f})")
                continue
                
            # Check Orange Book and Purple Book status using existing methods
            orange_book_status = self._check_orange_book_match(drug_name)
            purple_book_status = self._check_purple_book_match(drug_name)
            
            # If not found, try with normalized drug name to catch spelling variations
            if not orange_book_status and not purple_book_status:
                try:
                    from .data.drug_resolver import DrugResolver
                    resolver = DrugResolver()
                    normalized_result = await resolver.resolve_drug_name(drug_name)
                    if normalized_result and normalized_result.get('normalized_name'):
                        normalized_name = normalized_result['normalized_name']
                        if normalized_name.lower() != drug_name.lower():
                            orange_book_status = self._check_orange_book_match(normalized_name)
                            purple_book_status = self._check_purple_book_match(normalized_name)
                            logger.debug(f"Normalized '{drug_name}' -> '{normalized_name}': OB={bool(orange_book_status)}, PB={bool(purple_book_status)}")
                except Exception as e:
                    logger.debug(f"Drug name normalization failed for {drug_name}: {e}")
            
            # Check if drug is FDA approved for ANY indication (not just currently marketed)
            is_fda_approved = False
            program_status = candidate.get("program_status", "").lower()
            
            # Check ALL sources for FDA approval status (not just one)
            is_fda_approved = False
            
            # Check Orange Book
            if orange_book_status:
                ob_type = orange_book_status.get("type", "").upper()
                if ob_type in ["RX", "OTC", "DISCN"]:  # Active and discontinued FDA approvals
                    is_fda_approved = True
            
            # Check Purple Book - check actual marketing status
            if purple_book_status:
                pb_status = purple_book_status.get("marketing_status", "").upper()
                # Purple Book uses: Rx, Licensed, OTC for approved; Disc for discontinued
                if pb_status in ["RX", "LICENSED", "OTC"]:
                    is_fda_approved = True
            
            # Check program status and regulatory flags
            if (candidate.get("fda_approved") or 
                program_status == "fda_approved" or
                candidate.get("is_approved") or
                candidate.get("regulatory", {}).get("fda_approved")):
                is_fda_approved = True
            
            # Check LLM-derived regional approvals (if available from enrichment)
            regional_info = candidate.get('regional_approvals', {})
            if isinstance(regional_info, dict):
                fda_data = regional_info.get('fda')
                if fda_data:
                    # Handle new structured format
                    if isinstance(fda_data, dict):
                        if fda_data.get('approved'):
                            is_fda_approved = True
                    elif fda_data:  # Old boolean format
                        is_fda_approved = True
            
            # Filter out FDA-approved drugs (they're not discovery candidates)
            if is_fda_approved:
                candidate['filtered_reason'] = 'fda_approved_drug'
                filtered_out.append(candidate)
                continue
            
            # Only filter out if CURRENTLY marketed according to Orange/Purple Book
            is_currently_marketed = (
                (orange_book_status and orange_book_status.get("is_currently_marketed", False)) or
                (purple_book_status and purple_book_status.get("is_currently_marketed", False)) or
                candidate.get("is_currently_marketed", False)
            )
            
            if is_currently_marketed:
                candidate['filtered_reason'] = 'currently_marketed_orange_purple_book'
                filtered_out.append(candidate)
                continue
            
            # Store Orange/Purple Book status for scoring
            candidate['orange_book_status'] = orange_book_status
            candidate['purple_book_status'] = purple_book_status
            
            # Check if drug was discontinued according to Orange/Purple Book
            is_discontinued = False
            is_safety_discontinued = False
            
            # Check Orange Book for DISCN status
            if orange_book_status:
                ob_type = orange_book_status.get("type", "").upper()
                if ob_type == "DISCN":
                    is_discontinued = True
                    # Check if it has Federal Register note indicating NOT safety-related
                    federal_note = orange_book_status.get("federal_register_note", "")
                    if "not discontinued or withdrawn for safety or effectiveness reasons" in federal_note:
                        is_safety_discontinued = False
                    else:
                        # Assume safety-related if no note (conservative approach)
                        is_safety_discontinued = True
            
            # Check Purple Book for discontinued status
            if purple_book_status:
                pb_marketing_status = purple_book_status.get("marketing_status", "").upper()
                # Active statuses in Purple Book are RX, LICENSED, OTC
                if pb_marketing_status not in ["RX", "LICENSED", "OTC"]:
                    is_discontinued = True
                    # Purple Book doesn't have safety notes, assume non-safety
                    is_safety_discontinued = False
            
            # Check program status for discontinuation
            if program_status in ["discontinued", "shelved", "terminated", "withdrawn", "development_discontinued"]:
                is_discontinued = True
                # Check if safety-related discontinuation
                if any(term in program_status.lower() for term in ["safety", "toxicity", "adverse"]):
                    is_safety_discontinued = True
            
            # For investigational/experimental drugs, assume they could be discontinued (non-safety)
            if program_status in ["unknown", "investigational", "experimental"]:
                is_discontinued = True
                is_safety_discontinued = False
            
            # Store discontinuation info for scoring
            candidate['is_discontinued'] = is_discontinued
            candidate['is_safety_discontinued'] = is_safety_discontinued
            candidate['non_safety_discontinued'] = is_discontinued and not is_safety_discontinued
            
            # Only process discontinued drugs OR allow non-safety discontinued drugs
            if not is_discontinued:
                candidate['filtered_reason'] = 'not_discontinued'
                filtered_out.append(candidate)
                continue
            
            # Filter out safety-discontinued drugs
            if is_safety_discontinued:
                candidate['filtered_reason'] = 'safety_discontinued'
                filtered_out.append(candidate)
                continue
            
            # Calculate high potential score for discontinued drugs
            high_potential_score = self._calculate_high_potential_score(candidate)
            candidate['high_potential_score'] = high_potential_score
            
            # Categorize based on development stage and discontinuation reason
            max_phase = candidate.get("max_phase", 0)
            # Convert to float first, then int if it's a string or handle None
            if max_phase is None:
                max_phase = 0
            else:
                try:
                    max_phase = float(max_phase)
                    max_phase = int(max_phase)
                except (ValueError, TypeError):
                    max_phase = 0
            
            # New categorization based on score thresholds
            if high_potential_score >= 0.60:
                # Rescue-worthy: S >= 0.60
                # Prioritize if Phase II+ and non-safety discontinuation
                if max_phase >= 2 and candidate.get("non_safety_discontinued", False):
                    drug_rescue.append(candidate)
                else:
                    shelved_assets.append(candidate)
            elif high_potential_score >= 0.40:
                # Investigate / Watchlist: 0.40 <= S < 0.60
                drug_discovery.append(candidate)
            else:
                # De-prioritize: S < 0.40
                filtered_out.append(candidate)
                candidate['filtered_reason'] = f'low_score_{high_potential_score:.2f}'
        
        print(f"\n📊 CATEGORIZATION RESULTS:")
        print(f"  - Drug Discovery: {len(drug_discovery)}")
        print(f"  - Drug Rescue: {len(drug_rescue)}")
        print(f"  - Shelved Assets: {len(shelved_assets)}")
        print(f"  - Filtered Out: {len(filtered_out)}")
        
        # Debug: Show preclinical compounds in each category
        preclinical_debug = {
            'drug_discovery': [],
            'drug_rescue': [],
            'shelved_assets': [],
            'filtered_out': []
        }
        
        for category, candidates in [
            ('drug_discovery', drug_discovery),
            ('drug_rescue', drug_rescue), 
            ('shelved_assets', shelved_assets),
            ('filtered_out', filtered_out)
        ]:
            for candidate in candidates:
                if (candidate.get("development_stage") == "preclinical" or 
                    candidate.get("source") == "llm_web_search_preclinical"):
                    name = candidate.get('drug_name', candidate.get('compound_name', 'Unknown'))
                    preclinical_debug[category].append(name)
        
        # Show preclinical compound distribution
        total_preclinical = sum(len(compounds) for compounds in preclinical_debug.values())
        if total_preclinical > 0:
            print(f"\n🧪 PRECLINICAL COMPOUND DISTRIBUTION:")
            for category, compounds in preclinical_debug.items():
                if compounds:
                    print(f"  - {category}: {compounds}")
        
        # Show filtering reasons
        if filtered_out:
            reason_counts = {}
            for candidate in filtered_out:
                reason = candidate.get('filtered_reason', 'unknown')
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
            
            print(f"\n  Filtering reasons:")
            for reason, count in sorted(reason_counts.items(), key=lambda x: x[1], reverse=True):
                print(f"    - {reason}: {count}")
                
            # Debug: Show which preclinical compounds were filtered and why
            preclinical_filtered = []
            for candidate in filtered_out:
                if (candidate.get("development_stage") == "preclinical" or 
                    candidate.get("source") == "llm_web_search_preclinical"):
                    name = candidate.get('drug_name', candidate.get('compound_name', 'Unknown'))
                    reason = candidate.get('filtered_reason', 'unknown')
                    preclinical_filtered.append(f"{name} ({reason})")
            
            if preclinical_filtered:
                print(f"\n  🧪 Preclinical compounds filtered: {preclinical_filtered}")
        
        # Final deduplication within each category
        drug_discovery = self._deduplicate_candidates(drug_discovery)
        drug_rescue = self._deduplicate_candidates(drug_rescue)
        shelved_assets = self._deduplicate_candidates(shelved_assets)
        
        return {
            "drug_discovery": drug_discovery,
            "drug_rescue": drug_rescue,
            "high_potential_assets": shelved_assets,  # Renamed for consistency
            "filtered_out": filtered_out
        }
    
    def _calculate_high_potential_score(self, candidate: Dict) -> float:
        """Calculate a score for high potential drug assets based on multiple factors"""
        
        # Check if this is a preclinical compound
        if candidate.get("development_stage") == "preclinical" or candidate.get("source") == "llm_web_search_preclinical":
            return self._calculate_preclinical_score(candidate)
        
        # Weights for clinical compounds (without regional approvals)
        w_phase = 0.25
        w_disc = 0.25  
        w_rec = 0.20
        w_avail = 0.15
        w_mech = 0.15
        w_comp = 0.05
        w_safety = 0.02
        
        # 1. Phase P (late stage → higher)
        max_phase = candidate.get("max_phase", 0)
        # Convert to float first, then int if it's a string or handle None
        if max_phase is None:
            max_phase = 0
        else:
            try:
                max_phase = float(max_phase)
                max_phase = int(max_phase)
            except (ValueError, TypeError):
                max_phase = 0
        
        if max_phase >= 3:
            P = 1.0
        elif max_phase == 2:
            P = 0.67
        elif max_phase == 1:
            P = 0.33
        else:
            P = 0.0
            
        # 2. Discontinuation Reason D (non-safety tops)
        disc_info = candidate.get("discontinuation_info", {})
        if isinstance(disc_info, dict):
            disc_reason = disc_info.get("reason", "")
            disc_details = disc_info.get("details", "")
        else:
            disc_reason = ""
            disc_details = ""
        
        # Handle non-string types
        if isinstance(disc_reason, dict):
            disc_reason = str(disc_reason)
        if isinstance(disc_details, dict):
            disc_details = str(disc_details)
            
        disc_reason = disc_reason.lower() if disc_reason else ""
        disc_details = disc_details.lower() if disc_details else ""
        termination_reason = candidate.get("termination_reason", "")
        if isinstance(termination_reason, dict):
            termination_reason = str(termination_reason)
        termination_reason = termination_reason.lower() if termination_reason else ""
        all_reasons = f"{disc_reason} {disc_details} {termination_reason}"
        
        # Safety flag
        is_safety_issue = any(term in all_reasons for term in ["safety", "toxicity", "adverse", "death"])
        
        if any(term in all_reasons for term in ["strategic", "portfolio", "business", "pipeline", "commercial"]):
            D = 1.00
        elif any(term in all_reasons for term in ["pause", "suspend", "pivot", "recruitment", "operational", "manufacturing"]):
            D = 0.85
        elif any(term in all_reasons for term in ["efficacy", "endpoint", "futility"]):
            D = 0.50
        elif is_safety_issue:
            D = 0.00
        else:
            D = 0.35  # unknown/blank
        
        # Override if non_safety_discontinued flag is set
        if candidate.get("non_safety_discontinued", False):
            D = max(D, 0.90)
        
        # 3. Recency R (more recent → higher)
        ownership_info = candidate.get("ownership_info", {})
        ct_info = ownership_info.get("clinical_trials", {})
        latest_activity = ct_info.get("latest_activity")
        
        if latest_activity:
            try:
                from datetime import datetime
                activity_date = datetime.fromisoformat(latest_activity.replace('Z', '+00:00'))
                years_ago = (datetime.now() - activity_date.replace(tzinfo=None)).days / 365
                if years_ago < 2:
                    R = 1.00
                elif years_ago < 5:
                    R = 0.75
                elif years_ago < 10:
                    R = 0.50
                elif years_ago < 15:
                    R = 0.20
                else:
                    R = 0.00
            except:
                R = 0.00
        else:
            R = 0.00
        
        # 4. Availability A (licensable/available)
        availability = ownership_info.get("asset_availability", "")
        if isinstance(availability, dict):
            availability = str(availability)
        availability = availability.lower() if availability else ""
        
        if "available for licensing" in availability:
            A = 1.00
        elif any(term in availability for term in ["open to partnering", "exploring options"]):
            A = 0.70
        elif "not available" in availability:
            A = 0.00
        else:
            A = 0.30  # unknown
        
        # 5. Mechanistic Fit M (target validation in the disease)
        primary_target = candidate.get("primary_target", "")
        targets = candidate.get("targets", [])
        target_names = []
        
        # Handle primary_target
        if primary_target:
            if isinstance(primary_target, dict):
                primary_target = str(primary_target)
            target_names.append(primary_target)
        
        # Handle targets list
        if targets:
            for t in targets:
                if isinstance(t, dict):
                    target_name = t.get("target_name", "")
                    if target_name:
                        if isinstance(target_name, dict):
                            target_name = str(target_name)
                        target_names.append(target_name)
                elif isinstance(t, str):
                    target_names.append(t)
        
        all_targets = " ".join(str(name) for name in target_names).lower()
        
        # Tier A targets (1.00) - for lung cancer
        tier_a = ["pd-1", "pd-l1", "egfr", "alk", "ros1", "met", "ret", "kras g12c", "vegf", "vegfr", "parp"]
        # Tier B targets (0.65)
        tier_b = ["dll3", "trop2", "her3", "ctla-4", "tigit", "c-met"]
        # Tier C targets (0.35)
        tier_c = ["cediranib", "cytotoxic"]
        
        if any(target in all_targets for target in tier_a):
            M = 1.00
        elif any(target in all_targets for target in tier_b):
            M = 0.65
        elif any(target in all_targets for target in tier_c):
            M = 0.35
        else:
            M = 0.30  # unknown
        
        # 6. Competition/SoC crowding C (penalty)
        if "pd-1" in all_targets or "pd-l1" in all_targets:
            C = 1.00  # saturated
        elif "egfr" in all_targets:
            C = 0.50  # moderate
        else:
            C = 0.10  # sparse/niche
        
        # 7. Safety penalty T
        T = 1.0 if is_safety_issue else 0.0
        
        # Calculate final score
        S = (w_phase * P + w_disc * D + w_rec * R + w_avail * A + w_mech * M - w_comp * C - w_safety * T)
        
        # Clip to [0, 1]
        return max(0.0, min(1.0, S))
    
    def _calculate_preclinical_score(self, candidate: Dict) -> float:
        """Calculate score specifically for preclinical compounds"""
        
        # Preclinical scoring weights
        w_validation = 0.30  # Database validation (ChEMBL/PubChem)
        w_literature = 0.25  # Literature presence
        w_novelty = 0.20     # Novel mechanism/target
        w_source = 0.15      # Academic pedigree
        w_recent = 0.10      # Recent research activity
        
        # 1. Validation Score (V) - Database presence
        validation_score = candidate.get('validation_score', 0.0)
        if validation_score >= 1.0:  # Both PubChem + ChEMBL
            V = 1.0
        elif validation_score >= 0.5:  # ChEMBL only
            V = 0.7
        elif validation_score >= 0.2:  # Partial validation
            V = 0.4
        else:
            V = 0.0
        
        # 2. Literature Score (L) - Europe PMC presence
        # Assume high literature presence for compounds that passed web search
        if candidate.get('source') == 'llm_web_search_preclinical':
            L = 0.9  # High score for web-anchored compounds
        else:
            L = 0.5  # Default moderate score
        
        # 3. Novelty Score (N) - Novel mechanisms get higher scores
        mechanism = str(candidate.get('target_mechanism', '')).lower()
        target = str(candidate.get('compound_name', '')).lower()
        
        # Novel mechanisms for lung cancer
        novel_mechanisms = ['ferroptosis', 'autophagy', 'immunogenic cell death', 
                          'synthetic lethality', 'epigenetic', 'rna', 'protac']
        
        if any(mech in mechanism or mech in target for mech in novel_mechanisms):
            N = 0.9
        elif 'kinase' in mechanism or 'inhibitor' in mechanism:
            N = 0.6  # Established but validated approach
        else:
            N = 0.4  # Unknown mechanism
        
        # 4. Source Score (S) - Academic/research pedigree
        sponsor = str(candidate.get('sponsor', '')).lower()
        if 'academic' in sponsor or 'biotech' in sponsor or 'university' in sponsor:
            S = 0.8
        elif 'pharma' in sponsor:
            S = 0.6
        else:
            S = 0.5
        
        # 5. Recency Score (R) - Recent research activity
        latest_date = candidate.get('latest_activity_date', '2024')
        if '2024' in str(latest_date) or '2023' in str(latest_date):
            R = 1.0
        elif '2022' in str(latest_date) or '2021' in str(latest_date):
            R = 0.7
        else:
            R = 0.4
        
        # Calculate final preclinical score
        preclinical_score = (w_validation * V + w_literature * L + 
                           w_novelty * N + w_source * S + w_recent * R)
        
        # Boost for specific high-value compounds
        compound_name = candidate.get('compound_name', '').lower()
        if 'erastin' in compound_name:  # Known ferroptosis inducer
            preclinical_score = min(1.0, preclinical_score + 0.1)
        
        return max(0.0, min(1.0, preclinical_score))
    
    def _normalize_all_scores(self, candidates: List[Dict]) -> List[Dict]:
        """Apply unified scoring using _calculate_high_potential_score for all candidates"""
        
        # Use the same scoring method for ALL candidates regardless of source
        for candidate in candidates:
            # Calculate high potential score for every candidate
            candidate["normalized_score"] = self._calculate_high_potential_score(candidate)
            candidate["display_score"] = candidate["normalized_score"]
            
            # Keep original discovery_score for reference but don't use for ranking
            if candidate.get("discovery_score") is not None:
                candidate["original_discovery_score"] = candidate["discovery_score"]
        
        return candidates

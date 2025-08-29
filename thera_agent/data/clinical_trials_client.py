"""
ClinicalTrials.gov API client for drug repurposing analysis
"""
import aiohttp
import asyncio
import logging
from typing import List, Dict, Optional
import json
import re
import datetime
from datetime import timedelta
from bs4 import BeautifulSoup
from .cache import APICache
from .http_client import RateLimitedClient
from ..data.cache import APICache
from ..data.drug_resolver import DrugResolver

logger = logging.getLogger(__name__)

class ClinicalTrialsClient:
    """Client for ClinicalTrials.gov API v2"""
    
    BASE_URL = "https://clinicaltrials.gov/api/v2"
    
    def __init__(self, http_client: RateLimitedClient, cache_manager: APICache):
        self.http = http_client
        self.cache = cache_manager
        self.drug_resolver = DrugResolver()
    
    async def search_trials(
        self, 
        query: str = None,
        condition: str = None,
        intervention: str = None,
        status: List[str] = None,
        phase: List[str] = None,
        max_results: int = 2000
    ) -> List[Dict]:
        """Search clinical trials with filters using API v2"""
        
        # Build query parameters for v2 API
        params = {
            "format": "json",
            "pageSize": min(max_results, 1000)
        }
        
        # Use proper v2 API query parameters
        if condition:
            params["query.cond"] = condition
        if intervention:
            # Clean up intervention string and sanitize for API
            intervention = intervention.strip()
            
            # Remove problematic characters that cause 400 errors
            intervention = re.sub(r'[()*/\[\]{}]', '', intervention)
            intervention = re.sub(r'\*+', '', intervention)  # Remove asterisks
            intervention = re.sub(r'\s+', ' ', intervention).strip()  # Normalize whitespace
            
            # Skip if intervention becomes empty or too short after cleaning
            if len(intervention) < 2:
                pass  # Don't add query.intr parameter
            elif " OR " in intervention or " AND " in intervention:
                params["query.intr"] = intervention
            else:
                # Handle multiple interventions with proper formatting
                interventions = [i.strip() for i in intervention.split(",")]
                if len(interventions) > 1:
                    params["query.intr"] = " OR ".join(f'"{i}"' for i in interventions)
                else:
                    params["query.intr"] = intervention
        if query:
            params["query.term"] = query
        
        # Add filters using v2 format
        if status:
            params["filter.overallStatus"] = "|".join(status)
        if phase:
            params["filter.phase"] = "|".join(phase)
        
        # Check cache
        cached = self.cache.get(f"{self.BASE_URL}/studies", params)
        if cached:
            return cached
        
        # Make API request
        url = f"{self.BASE_URL}/studies"
        try:
            data = await self.http.get(url, params=params)
            studies = data.get("studies", [])
            
            # Extract relevant fields
            results = []
            for study in studies:
                protocol = study.get("protocolSection", {})
                id_module = protocol.get("identificationModule", {})
                status_module = protocol.get("statusModule", {})
                design_module = protocol.get("designModule", {})
                arms_module = protocol.get("armsInterventionsModule", {})
                outcomes_module = protocol.get("outcomesModule", {})
                
                # Extract sponsor information
                sponsor_module = protocol.get("sponsorCollaboratorsModule", {})
                lead_sponsor = sponsor_module.get("leadSponsor", {})
                
                result = {
                    "nct_id": id_module.get("nctId"),
                    "title": id_module.get("briefTitle"),
                    "status": status_module.get("overallStatus"),
                    "phase": design_module.get("phases", []),
                    "conditions": protocol.get("conditionsModule", {}).get("conditions", []),
                    "interventions": [],
                    "start_date": status_module.get("startDateStruct", {}).get("date"),
                    "completion_date": status_module.get("completionDateStruct", {}).get("date"),
                    "enrollment": design_module.get("enrollmentInfo", {}).get("count"),
                    "primary_outcomes": [],
                    "why_stopped": status_module.get("whyStopped", ""),
                    "results_available": study.get("hasResults", False),
                    "sponsor": {
                        "lead_sponsor": {
                            "agency": lead_sponsor.get("name", ""),
                            "class": lead_sponsor.get("class", "")
                        },
                        "collaborators": [
                            {"agency": collab.get("name", ""), "class": collab.get("class", "")}
                            for collab in sponsor_module.get("collaborators", [])
                        ]
                    }
                }
                
                # Extract interventions with details
                for intervention in arms_module.get("interventions", []):
                    result["interventions"].append({
                        "type": intervention.get("type"),
                        "name": intervention.get("name"),
                        "description": intervention.get("description", "")
                    })
                
                # Extract primary outcomes
                for outcome in outcomes_module.get("primaryOutcomes", []):
                    result["primary_outcomes"].append({
                        "measure": outcome.get("measure"),
                        "description": outcome.get("description", ""),
                        "time_frame": outcome.get("timeFrame", "")
                    })
                
                results.append(result)
            
            # Cache results
            self.cache.set(f"{self.BASE_URL}/studies", results, params=params, ttl_hours=24)
            return results
        except Exception as e:
            logger.error(f"Error searching trials: {e}")
            return []
    
    async def get_trial_results(self, nct_id: str) -> Optional[Dict]:
        """Get detailed results for a specific trial"""
        
        url = f"{self.BASE_URL}/studies/{nct_id}"
        params = {"format": "json", "fields": "ResultsSection"}
        
        cached = self.cache.get(url, params)
        if cached:
            return cached
        
        try:
            data = await self.http.get(url, params=params)
            study = data.get("studies", [{}])[0]
            results_section = study.get("resultsSection", {})
            
            if results_section:
                # Extract key results
                result = {
                    "nct_id": nct_id,
                    "participant_flow": results_section.get("participantFlowModule", {}),
                    "baseline": results_section.get("baselineCharacteristicsModule", {}),
                    "outcome_measures": results_section.get("outcomeMeasuresModule", {}),
                    "adverse_events": results_section.get("adverseEventsModule", {}),
                    "more_info": results_section.get("moreInfoModule", {})
                }
                
                self.cache.set(url, result, params=params, ttl_hours=48)
                return result
        except Exception as e:
            logger.error(f"Error getting trial results for {nct_id}: {e}")
        
        return None
    
    async def get_trial_adverse_events(self, nct_id: str) -> Optional[Dict]:
        """Get adverse events data for a specific trial including organ systems"""
        
        url = f"{self.BASE_URL}/studies/{nct_id}"
        params = {"format": "json", "fields": "ResultsSection"}
        
        cached = self.cache.get(f"{url}_adverse", params)
        if cached:
            return cached
        
        try:
            data = await self.http.get(url, params=params)
            study = data.get("studies", [{}])[0]
            results_section = study.get("resultsSection", {})
            adverse_module = results_section.get("adverseEventsModule", {})
            
            if adverse_module:
                # Extract organ systems from serious/other events
                organ_systems = set()
                
                # Process serious adverse events
                serious_events = adverse_module.get("seriousEvents", [])
                for event in serious_events:
                    organ_system = event.get("organSystem")
                    if organ_system:
                        organ_systems.add(organ_system)
                
                # Process other adverse events
                other_events = adverse_module.get("otherEvents", [])
                for event in other_events:
                    organ_system = event.get("organSystem")
                    if organ_system:
                        organ_systems.add(organ_system)
                
                result = {
                    "nct_id": nct_id,
                    "organ_systems": list(organ_systems),
                    "serious_events": serious_events,
                    "other_events": other_events,
                    "event_groups": adverse_module.get("eventGroups", []),
                    "frequency_threshold": adverse_module.get("frequencyThreshold")
                }
                
                self.cache.set(f"{url}_adverse", result, params=params, ttl_hours=48)
                return result
        except Exception as e:
            logger.debug(f"No adverse events data for {nct_id}: {e}")
        
        return None
    
    async def find_failed_trials_by_target(
        self,
        target: str,
        disease: str = None,
        include_withdrawn: bool = True,
        include_terminated: bool = True,
        include_suspended: bool = True
    ) -> List[Dict]:
        """Find failed/stopped trials for a specific target"""
        
        # Define failure statuses
        failure_statuses = []
        if include_withdrawn:
            failure_statuses.append("WITHDRAWN")
        if include_terminated:
            failure_statuses.append("TERMINATED")
        if include_suspended:
            failure_statuses.append("SUSPENDED")
        
        # Also include completed trials that might have failed
        failure_statuses.append("COMPLETED")
        
        # Search for trials
        trials = await self.search_trials(
            intervention=target,
            condition=disease,
            status=failure_statuses,
            max_results=2000
        )
        
        # Filter and enrich results
        failed_trials = []
        for trial in trials:
            # Skip if no clear failure reason and status is completed
            if trial["status"] == "COMPLETED" and not trial.get("why_stopped"):
                continue
            
            # Get detailed results if available
            if trial.get("results_available"):
                results = await self.get_trial_results(trial["nct_id"])
                if results:
                    trial["detailed_results"] = results
            
            failed_trials.append(trial)
        
        return failed_trials
    
    async def analyze_failure_patterns(
        self,
        trials: List[Dict]
    ) -> Dict[str, List[Dict]]:
        """Analyze patterns in trial failures"""
        
        patterns = {
            "safety_issues": [],
            "efficacy_issues": [],
            "recruitment_issues": [],
            "business_reasons": [],
            "other_reasons": [],
            "unknown": [],
            "organ_systems_affected": {}
        }
        
        # Keywords for categorization
        safety_keywords = ["safety", "adverse", "toxicity", "side effect", "SAE", "death"]
        efficacy_keywords = ["efficacy", "ineffective", "no benefit", "futility", "endpoint not met"]
        recruitment_keywords = ["enrollment", "recruitment", "accrual", "participants"]
        business_keywords = ["sponsor", "funding", "business", "strategic", "priorit"]
        
        # Collect NCT IDs for batch adverse events fetching
        safety_trial_ids = []
        
        for trial in trials:
            why_stopped = trial.get("why_stopped", "").lower()
            categorized = False
            
            # Check safety issues
            if any(keyword in why_stopped for keyword in safety_keywords):
                patterns["safety_issues"].append(trial)
                categorized = True
                if trial.get("nct_id"):
                    safety_trial_ids.append(trial["nct_id"])
            
            # Check efficacy issues
            elif any(keyword in why_stopped for keyword in efficacy_keywords):
                patterns["efficacy_issues"].append(trial)
                categorized = True
            
            # Check recruitment issues  
            elif any(keyword in why_stopped for keyword in recruitment_keywords):
                patterns["recruitment_issues"].append(trial)
                categorized = True
            
            # Check business reasons
            elif any(keyword in why_stopped for keyword in business_keywords):
                patterns["business_reasons"].append(trial)
                categorized = True
            
            # Other reasons
            elif why_stopped and len(why_stopped) > 10:
                patterns["other_reasons"].append(trial)
                categorized = True
            
            # Unknown
            if not categorized:
                patterns["unknown"].append(trial)
        
        # Fetch adverse events data for safety-related trials
        if safety_trial_ids:
            # Batch fetch with limited concurrency
            organ_systems_count = {}
            
            # Process in batches of 10
            for i in range(0, len(safety_trial_ids), 10):
                batch = safety_trial_ids[i:i+10]
                tasks = [self.get_trial_adverse_events(nct_id) for nct_id in batch]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for result in results:
                    if isinstance(result, dict) and result.get("organ_systems"):
                        for organ_system in result["organ_systems"]:
                            organ_systems_count[organ_system] = organ_systems_count.get(organ_system, 0) + 1
            
            # Sort by frequency
            patterns["organ_systems_affected"] = dict(sorted(
                organ_systems_count.items(),
                key=lambda x: x[1],
                reverse=True
            ))
        
        return patterns
    
    async def get_drug_asset_discovery_candidates(
        self,
        disease: str,
        exclude_targets: List[str] = None
    ) -> List[Dict]:
        """Find drug asset discovery candidates for a disease"""
        
        # Search for all trials in the disease
        all_trials = await self.search_trials(
            condition=disease,
            max_results=2000
        )
        
        # Group by intervention
        intervention_stats = {}
        now = datetime.datetime.utcnow()
        threshold_date_7y = now - timedelta(days=365*17)

        # Simple date parser
        def _parse_date_str(s: Optional[str]):
            if not s:
                return None
            for fmt in ("%Y-%m-%d", "%Y-%m", "%Y"):
                try:
                    return datetime.datetime.strptime(s, fmt)
                except Exception:
                    continue
            try:
                # Last resort for ISO strings
                return datetime.datetime.fromisoformat(s)
            except Exception:
                return None
        
        for trial in all_trials:
            for intervention in trial.get("interventions", []):
                if intervention["type"] != "DRUG":
                    continue
                
                name = intervention["name"]
                if exclude_targets and any(target in name for target in exclude_targets):
                    continue
                
                # Filter out placebo and non-drug interventions
                name_lower = name.lower()
                non_drug_terms = [
                    "placebo", "saline", "standard care", "usual care", 
                    "observation", "control", "sham", "vehicle",
                    "no treatment", "supportive care", "best supportive care"
                ]
                if any(term in name_lower for term in non_drug_terms):
                    continue
                
                if name not in intervention_stats:
                    intervention_stats[name] = {
                        "drug": name,
                        "total_trials": 0,
                        "completed": 0,
                        "failed": 0,
                        "ongoing": 0,
                        "phases": set(),
                        "trials": [],
                        "sponsors": set(),  # Track all sponsors
                        # Activity recency tracking
                        "active_trial_dates": [],  # ISO strings
                        "recent_active_trials": 0,  # last 7y
                        "latest_activity_date": None,
                        # Track failure reasons
                        "safety_failures": 0,
                        "efficacy_failures": 0,
                        "recruitment_failures": 0,
                        "business_failures": 0,
                        "other_failures": 0
                    }
                
                stats = intervention_stats[name]
                stats["total_trials"] += 1
                stats["trials"].append(trial["nct_id"])
                
                # Collect sponsors - corrected for API v2 structure
                sponsor_info = trial.get("sponsor", {})
                lead_sponsor_agency = sponsor_info.get("lead_sponsor", {}).get("agency")
                if lead_sponsor_agency:
                    stats["sponsors"].add(lead_sponsor_agency)
                    
                # Update phase info
                for phase in trial.get("phase", []):
                    stats["phases"].add(phase)
                
                # Categorize by status
                status = trial["status"]
                if status == "COMPLETED":
                    stats["completed"] += 1
                elif status in ["TERMINATED", "WITHDRAWN", "SUSPENDED"]:
                    # Track failure reason for scoring
                    why_stopped = trial.get("why_stopped", "").lower()
                    
                    # Categorize failure reason
                    safety_keywords = ["safety", "adverse", "toxicity", "dose limiting", "tolerability", "side effect"]
                    efficacy_keywords = ["efficacy", "ineffective", "futility", "lack of efficacy", "no improvement", "failed to meet"]
                    recruitment_keywords = ["recruitment", "enrollment", "accrual", "low enrollment", "slow recruitment"]
                    business_keywords = ["business", "funding", "sponsor", "strategic", "company decision", "commercial"]
                    
                    # Only count as "failed" if it's a drug-related failure (safety/efficacy)
                    if any(keyword in why_stopped for keyword in safety_keywords):
                        stats["failed"] += 1
                        stats["safety_failures"] += 1
                    elif any(keyword in why_stopped for keyword in efficacy_keywords):
                        stats["failed"] += 1
                        stats["efficacy_failures"] += 1
                    elif any(keyword in why_stopped for keyword in recruitment_keywords):
                        # Recruitment failures are operational, not drug failures
                        stats["recruitment_failures"] += 1
                    elif any(keyword in why_stopped for keyword in business_keywords):
                        # Business failures are operational, not drug failures
                        stats["business_failures"] += 1
                    else:
                        # Unknown reason - count as failed to be conservative
                        stats["failed"] += 1
                        stats["other_failures"] += 1
                        
                elif status in ["RECRUITING", "ACTIVE_NOT_RECRUITING", "NOT_YET_RECRUITING"]:
                    stats["ongoing"] += 1
                
                # Track activity dates and recent active trials (last 7 years)
                start_dt = _parse_date_str(trial.get("start_date"))
                completion_dt = _parse_date_str(trial.get("completion_date"))
                # Latest activity across start/completion
                latest_dt = None
                if start_dt and completion_dt:
                    latest_dt = max(start_dt, completion_dt)
                else:
                    latest_dt = start_dt or completion_dt
                if latest_dt:
                    current_latest = _parse_date_str(stats["latest_activity_date"]) if stats["latest_activity_date"] else None
                    if not current_latest or latest_dt > current_latest:
                        stats["latest_activity_date"] = latest_dt.strftime("%Y-%m-%d")
                
                # Capture active trial start dates for flexible filtering later
                if status in {"RECRUITING", "ACTIVE_NOT_RECRUITING", "NOT_YET_RECRUITING", "ENROLLING_BY_INVITATION"}:
                    if start_dt:
                        stats["active_trial_dates"].append(start_dt.strftime("%Y-%m-%d"))
                        if start_dt >= threshold_date_7y:
                            stats["recent_active_trials"] += 1
        
        # Deduplicate drug names (case/whitespace variations)
        deduplicated_stats = {}
        for name, stats in intervention_stats.items():
            # Normalize name for deduplication
            normalized = name.strip().lower()
            
            if normalized in deduplicated_stats:
                # Merge stats with existing entry
                existing = deduplicated_stats[normalized]
                for key, value in stats.items():
                    if key in ["total_trials", "ongoing", "completed", "failed", 
                              "recruitment_failures", "business_failures", "safety_failures",
                              "efficacy_failures", "other_failures"]:
                        existing[key] += value
                    elif key == "phases":
                        existing[key].update(value)
                    elif key == "sponsors":
                        existing[key].update(value)
                    elif key == "active_trial_dates":
                        existing[key].extend(value)
                    # Keep the better formatted name
                    if len(name.strip()) > len(existing.get("original_name", "")):
                        existing["original_name"] = name
            else:
                stats["original_name"] = name
                deduplicated_stats[normalized] = stats
        
        # Convert to list and calculate scores
        candidates = []
        for normalized_name, stats in deduplicated_stats.items():
            name = stats.get("original_name", normalized_name)
            
            # Skip invalid drug combinations
            if "NOT_A_DRUG" in name:
                continue
            # Calculate discovery score
            # Higher score for drugs that have been tested but not in late stage
            score = 0
            
            # Points for having trials
            score += min(stats["total_trials"] * 10, 50)
            
            # Points for early phase trials (more opportunity)
            if "PHASE1" in stats["phases"]:
                score += 20
            if "PHASE2" in stats["phases"]:
                score += 15
            if "PHASE3" not in stats["phases"]:  # Bonus if hasn't reached phase 3
                score += 10
            
            # Points for mixed results (some success)
            if stats["completed"] > 0:
                score += 15
            
            # Penalty for many failures
            failure_rate = stats["failed"] / max(stats["total_trials"], 1)
            score -= failure_rate * 20
            
            # BONUS for good failure reasons (drug wasn't the problem)
            # Get failure reason breakdown from trials
            reason_bonuses = {
                "recruitment": 50,    # BEST: Drug is fine, just couldn't find patients
                "business": 40,       # EXCELLENT: Drug works, just funding issues  
                "other": 10,          # GOOD: Non-drug issues
                "efficacy": -30,      # BAD: Drug doesn't work
                "safety": -50         # WORST: Drug is dangerous
            }
            
            # Apply failure reason bonuses
            for reason, bonus in reason_bonuses.items():
                reason_count = stats.get(f"{reason}_failures", 0)
                if reason_count > 0:
                    # Weight by proportion of trials with this reason
                    reason_weight = reason_count / max(stats["total_trials"], 1)
                    score += bonus * reason_weight
            
            stats["phases"] = list(stats["phases"])
            stats["discovery_score"] = score
            stats["drug_name"] = name  # Add drug_name field for compatibility
            stats["clinical_trials_count"] = stats["total_trials"]
            stats["max_phase"] = 3 if "PHASE3" in stats["phases"] else (2 if "PHASE2" in stats["phases"] else (1 if "PHASE1" in stats["phases"] else 0))
            
            # Extract primary sponsor from sponsors set
            sponsors_list = list(stats["sponsors"])
            stats["sponsor"] = sponsors_list[0] if sponsors_list else None
            stats["sponsors"] = sponsors_list  # Convert set to list for JSON serialization
            
            candidates.append(stats)
        
        # Sort by score
        candidates.sort(key=lambda x: x["discovery_score"], reverse=True)
        
        return candidates[:500]  # Top 100 candidates
    
    async def get_shelved_drug_candidates(
        self,
        disease: str,
        include_only_discontinued: bool = True,
        min_failure_ratio: float = 0.5,
        drug_safety_client=None,
        recent_years: int = 7,
        exclude_recent_activity: bool = True,
        exclude_active_marketing: bool = True,
        max_candidates: int = 50  # Limit candidates to prevent timeouts
    ) -> List[Dict]:
        """Find shelved/discontinued drugs for academic discovery
        
        Args:
            disease: Disease indication to search
            include_only_discontinued: Only return drugs with no active trials
            min_failure_ratio: Minimum ratio of failed/terminated trials
            drug_safety_client: DrugSafetyClient instance for FDA approval checks
            recent_years: Exclude drugs with any active trials in the last N years
            exclude_recent_activity: If True, exclude drugs with recent active trials
            exclude_active_marketing: If True, exclude drugs with active marketing presence
        
        Returns:
            List of shelved drug candidates with failure analysis
        """
        
        # Get all candidates first (with timeout protection)
        try:
            all_candidates = await asyncio.wait_for(
                self.get_drug_asset_discovery_candidates(disease),
                timeout=15.0  # 10 second timeout for candidate discovery
            )
        except asyncio.TimeoutError:
            logger.warning(f"Drug asset discovery candidates timed out for {disease}")
            return []
        
        shelved_candidates = []
        now = datetime.datetime.utcnow()
        cutoff = now - timedelta(days=365*max(1, recent_years))
        
        # Simple date parser
        def _parse_date_str(s: Optional[str]):
            if not s:
                return None
            for fmt in ("%Y-%m-%d", "%Y-%m", "%Y"):
                try:
                    return datetime.datetime.strptime(s, fmt)
                except Exception:
                    continue
            try:
                return datetime.datetime.fromisoformat(s)
            except Exception:
                return None
        
        # Limit processing to prevent timeouts
        all_candidates = all_candidates[:max_candidates]
        
        for candidate in all_candidates:
            # Calculate failure metrics
            total_trials = candidate.get("total_trials", 0)
            failed_trials = candidate.get("failed", 0)
            ongoing_trials = candidate.get("ongoing", 0)
            
            if total_trials == 0:
                continue
                
            failure_ratio = failed_trials / total_trials
            
            # Apply filters
            if include_only_discontinued and ongoing_trials > 0:
                continue  # Skip drugs with active trials
            
            # Exclude if any active trials in the last N years
            if exclude_recent_activity:
                active_dates = candidate.get("active_trial_dates", []) or []
                has_recent_active = any((lambda d: (d and _parse_date_str(d) and _parse_date_str(d) >= cutoff)) (d) for d in active_dates)
                candidate["has_recent_active_trials"] = bool(has_recent_active)
                if has_recent_active:
                    continue
            
            if failure_ratio < min_failure_ratio:
                continue  # Skip drugs without enough failures
            
            # Check FDA approval status if safety client provided
            if drug_safety_client:
                drug_name = candidate.get("drug", "")
                
                # Get regional approval data from drug resolver with disease context
                resolved_drug = await self.drug_resolver.resolve_drug(drug_name, disease)
                regional_approvals = None
                if resolved_drug and isinstance(resolved_drug, dict):
                    regional_approvals = resolved_drug.get("regional_approvals")
                
                reg_data = await drug_safety_client.get_regulatory_status(drug_name, regional_approvals)
                
                if reg_data:
                    # Exclude if actively marketed (Orange Book active products)
                    if exclude_active_marketing and reg_data.get("is_currently_marketed", False):
                        continue
                    # Carry key flags forward for downstream consumers
                    candidate["is_currently_marketed"] = reg_data.get("is_currently_marketed", False)
                    candidate["withdrawn_for_safety_or_efficacy"] = reg_data.get("withdrawn_for_safety_or_efficacy", False)
            
            # Calculate academic value score
            # Higher score for non-safety failures (recruitment, business)
            academic_score = 0
            
            # Recruitment failures are GOLD for academics (drug works, just logistics)
            recruitment_failures = candidate.get("recruitment_failures", 0)
            academic_score += recruitment_failures * 30
            
            # Business failures are great (drug abandoned for non-scientific reasons)
            business_failures = candidate.get("business_failures", 0)
            academic_score += business_failures * 25
            
            # Other failures might be interesting
            other_failures = candidate.get("other_failures", 0)
            academic_score += other_failures * 10
            
            # Safety/efficacy failures are less attractive
            safety_failures = candidate.get("safety_failures", 0)
            efficacy_failures = candidate.get("efficacy_failures", 0)
            academic_score -= (safety_failures + efficacy_failures) * 5
            
            # Penalize explicit FDA withdrawal for safety/efficacy
            if candidate.get("withdrawn_for_safety_or_efficacy"):
                academic_score -= 50
            
            # Bonus for early phase (more room for optimization)
            phases = candidate.get("phases", [])
            if "PHASE1" in phases and "PHASE3" not in phases:
                academic_score += 20
            
            # Add academic-specific fields
            candidate["is_shelved"] = True
            candidate["failure_ratio"] = round(failure_ratio, 2)
            candidate["academic_score"] = academic_score
            candidate["primary_failure_reason"] = self._get_primary_failure_reason(candidate)
            candidate["rescue_potential"] = self._assess_rescue_potential(candidate)
            
            shelved_candidates.append(candidate)
        
        # Sort by academic score
        shelved_candidates.sort(key=lambda x: x["academic_score"], reverse=True)
        
        return shelved_candidates
    
    def _get_primary_failure_reason(self, candidate: Dict) -> str:
        """Determine the primary reason for drug failure"""
        
        failure_types = {
            "recruitment": candidate.get("recruitment_failures", 0),
            "business": candidate.get("business_failures", 0),
            "safety": candidate.get("safety_failures", 0),
            "efficacy": candidate.get("efficacy_failures", 0),
            "other": candidate.get("other_failures", 0)
        }
        
        if sum(failure_types.values()) == 0:
            return "unknown"
            
        return max(failure_types.items(), key=lambda x: x[1])[0]
    
    def _assess_rescue_potential(self, candidate: Dict) -> str:
        """Assess the potential for academic rescue"""
        
        primary_reason = candidate.get("primary_failure_reason", "unknown")
        
        if primary_reason in ["recruitment", "business"]:
            return "HIGH - Failed for non-scientific reasons"
        elif primary_reason == "other":
            return "MEDIUM - Unclear failure, worth investigating"
        elif primary_reason == "efficacy":
            return "LOW - May need new formulation or combination"
        elif primary_reason == "safety":
            return "VERY LOW - Safety issues need resolution"
        else:
            return "UNKNOWN - Insufficient data"
    
    async def get_fda_approved_shelved_drugs(
        self,
        disease: str,
        drug_safety_client=None,
        exclude_safety_discontinued: bool = True
    ) -> List[Dict]:
        """Find FDA-approved drugs that were later shelved/discontinued
        
        Args:
            disease: Disease indication to search
            drug_safety_client: DrugSafetyClient instance for regulatory data
            exclude_safety_discontinued: If True, exclude drugs discontinued for safety
            
        Returns:
            List of FDA-approved but shelved drug candidates
        """
        
        # First get all shelved candidates
        shelved_candidates = await self.get_shelved_drug_candidates(
            disease=disease,
            include_only_discontinued=True
        )
        
        if not drug_safety_client:
            logger.warning("No drug safety client provided - cannot check FDA status")
            return []
        
        fda_approved_shelved = []
        
        for candidate in shelved_candidates:
            drug_name = candidate.get("drug", "")
            
            # Get regulatory status with timeout
            try:
                reg_data = await asyncio.wait_for(
                    drug_safety_client.get_regulatory_status(drug_name),
                    timeout=5.0  # 5 second timeout per drug
                )
            except asyncio.TimeoutError:
                logger.debug(f"Regulatory status lookup timed out for {drug_name}")
                continue
            except Exception as e:
                logger.debug(f"Regulatory status lookup failed for {drug_name}: {e}")
                continue
            
            if not reg_data:
                continue
                
            # Check if FDA approved
            is_fda_approved = reg_data.get("is_approved", False)
            
            if not is_fda_approved:
                continue
                
            # Check if drug is currently marketed
            is_currently_marketed = reg_data.get("is_currently_marketed", False)
            
            # A drug is only truly "shelved" if it's FDA approved but no longer marketed
            if is_currently_marketed:
                continue
                
            # Get discontinuation details from Orange Book if available
            discontinuation_reason = None
            approval_details = reg_data.get("approval_details", [])
            
            for detail in approval_details:
                if detail.get("source") == "orange_book":
                    ob_data = detail.get("data", {})
                    if ob_data.get("discontinued_flag") == "Y":
                        # Extract discontinuation reason if available
                        discontinuation_reason = ob_data.get("discontinuation_notation", "")
                
            # Check if discontinued for safety reasons
            is_safety_discontinued = False
            
            if discontinuation_reason:
                safety_keywords = ["safety", "adverse", "toxicity", "risk", "death", "serious"]
                for keyword in safety_keywords:
                    if keyword.lower() in discontinuation_reason.lower():
                        is_safety_discontinued = True
                        break
                        
            # Also check clinical trial failure reasons
            primary_failure = candidate.get("primary_failure_reason", "")
            if primary_failure == "safety":
                is_safety_discontinued = True
                
            if exclude_safety_discontinued and is_safety_discontinued:
                continue
                
            # Add regulatory details to candidate
            candidate["fda_approved"] = True
            candidate["orange_book_discontinued"] = True  # We only get here if not currently marketed
            candidate["discontinuation_reason"] = discontinuation_reason or "Unknown"
            candidate["safety_discontinued"] = is_safety_discontinued
            candidate["regulatory_data"] = reg_data
            candidate["is_currently_marketed"] = False  # By definition since we filtered these out
            
            # Adjust academic score for FDA-approved status (higher value)
            candidate["academic_score"] = candidate.get("academic_score", 0) + 50
            
            fda_approved_shelved.append(candidate)
            
        # Sort by academic score
        fda_approved_shelved.sort(key=lambda x: x["academic_score"], reverse=True)
        
        return fda_approved_shelved

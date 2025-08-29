"""
Drug Status Classifier with Controlled Vocabulary
Provides precise, evidence-based classification of drug development status
"""
from typing import Dict, List, Optional, Tuple
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ActivityStatus(Enum):
    """Controlled vocabulary for activity status"""
    ACTIVE = "active"  # Has ongoing trials
    COMPLETED = "completed"  # Trial finished as planned
    TERMINATED = "terminated"  # Trial stopped early
    WITHDRAWN = "withdrawn"  # Trial stopped before first participant
    SUSPENDED = "suspended"  # Temporarily stopped
    ON_HOLD_REGULATORY = "on_hold_regulatory"  # FDA clinical hold
    ON_HOLD_SPONSOR = "on_hold_sponsor"  # Company-initiated pause
    DEVELOPMENT_DISCONTINUED = "development_discontinued"  # Sponsor ended development
    DORMANT = "dormant"  # No activity ≥ N years
    COMMERCIALLY_DISCONTINUED = "commercially_discontinued"  # No longer sold
    MARKETED = "marketed"  # Currently marketed somewhere
    APPROVED = "approved"  # Approved but not yet marketed
    

class DiscontinuationReason(Enum):
    """Controlled vocabulary for discontinuation reasons"""
    SAFETY = "safety"
    LACK_OF_EFFICACY = "lack_of_efficacy"
    STRATEGIC_BUSINESS = "strategic_business"  # reprioritization, financing, portfolio
    CMC_MANUFACTURING = "cmc_manufacturing"
    RECRUITMENT_OPERATIONAL = "recruitment_operational"
    REGULATORY = "regulatory"  # RTF, CRL, hold
    UNKNOWN = "unknown"
    

class EvidenceStrength(Enum):
    """Evidence strength for status classification"""
    CONFIRMED = "confirmed"  # Explicit statement in filing/PR/registry
    INFERRED = "inferred"  # No activity ≥ N years + no active trials
    UNCLEAR = "unclear"


class DrugStatusClassifier:
    """Classify drug development status using controlled vocabulary"""
    
    # Deterministic synonym mappings
    DRUG_SYNONYMS = {
        # Tofogliflozin mappings
        "RO4998452": "tofogliflozin",
        "CSG-452": "tofogliflozin", 
        "RG-7201": "tofogliflozin",
        "RO-4998452": "tofogliflozin",
        
        # Other known mappings
        "CKD-501": "lobeglitazone",
        "ECC5004": "AZD5004",
        "AZD5004": "ECC5004",
    }
    
    # Japan-approved drugs that should never be marked as shelved
    JAPAN_MARKETED_DRUGS = {
        "tofogliflozin": {
            "brand_names": ["Apleway", "Deberza"],
            "companies": ["Sanofi", "Kowa"],
            "approval_year": 2014
        }
    }
    
    def __init__(self):
        self.inactivity_threshold_years = 3  # Years of inactivity before "dormant"
        
    def normalize_drug_name(self, drug_name: str) -> str:
        """Normalize drug name using synonym map"""
        # Check if it's a code that needs mapping
        drug_name = drug_name.strip()
        
        # Direct lookup
        if drug_name in self.DRUG_SYNONYMS:
            return self.DRUG_SYNONYMS[drug_name]
            
        # Case-insensitive lookup
        drug_name_upper = drug_name.upper()
        for code, name in self.DRUG_SYNONYMS.items():
            if code.upper() == drug_name_upper:
                return name
                
        return drug_name
        
    def classify_drug_status(
        self,
        drug_name: str,
        clinical_trials_data: Optional[Dict] = None,
        regulatory_data: Optional[Dict] = None,
        shelving_reason_data: Optional[Dict] = None,
        last_activity_date: Optional[str] = None
    ) -> Dict[str, any]:
        """
        Classify drug status using all available data sources
        
        Returns:
            Dict with:
            - program_status: ActivityStatus enum value
            - reason: DiscontinuationReason enum value (if discontinued)
            - evidence: EvidenceStrength enum value
            - regions: Dict of regional statuses
            - details: Additional context
        """
        
        # Normalize drug name first
        normalized_name = self.normalize_drug_name(drug_name)
        
        # Check for sentinel tokens - immediate fail
        if "NOT_A_DRUG" in drug_name or "NOT_A_DRUG" in normalized_name:
            return {
                "program_status": None,
                "reason": None,
                "evidence": EvidenceStrength.CONFIRMED.value,
                "regions": {},
                "details": "Invalid drug entity",
                "error": "Contains invalid sentinel token"
            }
        
        # Check regional approvals first - trumps everything
        regions = self._check_regional_status(normalized_name, regulatory_data)
        
        # If marketed anywhere, it's not shelved
        if any(status in ["marketed", "approved"] for status in regions.values()):
            return {
                "program_status": ActivityStatus.MARKETED.value,
                "reason": None,
                "evidence": EvidenceStrength.CONFIRMED.value,
                "regions": regions,
                "details": f"Marketed in: {', '.join([r for r, s in regions.items() if s == 'marketed'])}"
            }
        
        # Check clinical trials status
        trials_status = self._analyze_clinical_trials(clinical_trials_data)
        
        # Apply registry gate - must have completed/terminated trials
        if trials_status["has_active_trials"]:
            phase = trials_status.get("current_phase", "Unknown")
            return {
                "program_status": ActivityStatus.ACTIVE.value,
                "phase": phase,
                "reason": None,
                "evidence": EvidenceStrength.CONFIRMED.value,
                "regions": regions,
                "details": f"Active Phase {phase} trials"
            }
        
        # Check shelving reason if provided
        if shelving_reason_data:
            reason = self._map_shelving_reason(shelving_reason_data)
            evidence = (EvidenceStrength.CONFIRMED.value 
                       if shelving_reason_data.get("confidence", 0) > 0.7
                       else EvidenceStrength.INFERRED.value)
                       
            return {
                "program_status": ActivityStatus.DEVELOPMENT_DISCONTINUED.value,
                "reason": reason.value,
                "evidence": evidence,
                "regions": regions,
                "details": shelving_reason_data.get("details", "")
            }
        
        # Default to dormant if no recent activity
        return {
            "program_status": ActivityStatus.DORMANT.value,
            "reason": DiscontinuationReason.UNKNOWN.value,
            "evidence": EvidenceStrength.INFERRED.value,
            "regions": regions,
            "details": "No recent clinical activity"
        }
        
    def _check_regional_status(self, drug_name: str, regulatory_data: Optional[Dict]) -> Dict[str, str]:
        """Check regional approval/marketing status"""
        regions = {}
        
        # Check hardcoded Japan approvals
        if drug_name in self.JAPAN_MARKETED_DRUGS:
            regions["Japan"] = "marketed"
            
        # Check regulatory data
        if regulatory_data:
            if regulatory_data.get("fda_approved"):
                regions["US"] = "approved"
            if regulatory_data.get("ema_approved"):
                regions["EU"] = "approved"  
            if regulatory_data.get("pmda_approved"):
                regions["Japan"] = "approved"
            if regulatory_data.get("health_canada_approved"):
                regions["Canada"] = "approved"
                
            # Check if marketed vs just approved
            if regulatory_data.get("is_currently_marketed"):
                for region in regions:
                    if regions[region] == "approved":
                        regions[region] = "marketed"
                        
        return regions
        
    def _analyze_clinical_trials(self, trials_data: Optional[Dict]) -> Dict[str, any]:
        """Analyze clinical trials data to determine activity status"""
        if not trials_data:
            return {"has_active_trials": False}
            
        # Look for active/recruiting trials
        active_statuses = ["recruiting", "active", "not yet recruiting", "enrolling"]
        trials = trials_data.get("trials", [])
        
        active_trials = [t for t in trials 
                        if (t.get("status", "").lower() in active_statuses or
                            t.get("overall_status", "").lower() in active_statuses)]
        
        if active_trials:
            # Get highest phase from active trials
            phases = [t.get("phase", 0) for t in active_trials]
            max_phase = max(phases) if phases else 0
            
            return {
                "has_active_trials": True,
                "current_phase": max_phase,
                "active_trial_count": len(active_trials)
            }
            
        return {"has_active_trials": False}
        
    def _map_shelving_reason(self, shelving_data: Dict) -> DiscontinuationReason:
        """Map shelving reason to controlled vocabulary"""
        reason = shelving_data.get("reason", "").lower()
        
        if "safety" in reason:
            return DiscontinuationReason.SAFETY
        elif "efficacy" in reason:
            return DiscontinuationReason.LACK_OF_EFFICACY
        elif any(term in reason for term in ["business", "strategic", "portfolio", "financing"]):
            return DiscontinuationReason.STRATEGIC_BUSINESS
        elif any(term in reason for term in ["manufacturing", "cmc", "formulation"]):
            return DiscontinuationReason.CMC_MANUFACTURING
        elif any(term in reason for term in ["recruitment", "enrollment", "operational"]):
            return DiscontinuationReason.RECRUITMENT_OPERATIONAL
        elif any(term in reason for term in ["regulatory", "fda", "ema"]):
            return DiscontinuationReason.REGULATORY
        else:
            return DiscontinuationReason.UNKNOWN
            
    def is_repurposing_candidate(self, status_result: Dict) -> bool:
        """Determine if a drug is a good repurposing candidate"""
        
        # Not a candidate if marketed or active
        if status_result["program_status"] in [
            ActivityStatus.MARKETED.value,
            ActivityStatus.ACTIVE.value,
            ActivityStatus.APPROVED.value
        ]:
            return False
            
        # Good candidates: discontinued for non-efficacy/safety reasons
        if status_result["program_status"] == ActivityStatus.DEVELOPMENT_DISCONTINUED.value:
            reason = status_result.get("reason")
            if reason in [
                DiscontinuationReason.STRATEGIC_BUSINESS.value,
                DiscontinuationReason.RECRUITMENT_OPERATIONAL.value,
                DiscontinuationReason.CMC_MANUFACTURING.value
            ]:
                return True
                
        # Dormant drugs might be candidates
        if status_result["program_status"] == ActivityStatus.DORMANT.value:
            return True
            
        return False

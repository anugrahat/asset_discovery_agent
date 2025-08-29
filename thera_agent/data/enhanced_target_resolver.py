"""
Enhanced drug target resolver that handles biologics and uses multiple data sources
"""
import asyncio
from typing import List, Dict, Any, Optional
from .chembl_client import ChEMBLClient
from .http_client import get_http_client

class EnhancedTargetResolver:
    """Resolves drug targets using multiple strategies including biologics handling"""
    
    def __init__(self):
        self.chembl_client = ChEMBLClient()
        self.biologic_targets = {
            # PD-1/PD-L1 inhibitors
            "atezolizumab": {"target": "PD-L1", "gene": "CD274", "type": "antibody"},
            "pembrolizumab": {"target": "PD-1", "gene": "PDCD1", "type": "antibody"},
            "nivolumab": {"target": "PD-1", "gene": "PDCD1", "type": "antibody"},
            "durvalumab": {"target": "PD-L1", "gene": "CD274", "type": "antibody"},
            "avelumab": {"target": "PD-L1", "gene": "CD274", "type": "antibody"},
            
            # CTLA-4 inhibitors
            "ipilimumab": {"target": "CTLA-4", "gene": "CTLA4", "type": "antibody"},
            "tremelimumab": {"target": "CTLA-4", "gene": "CTLA4", "type": "antibody"},
            
            # HER2 inhibitors
            "trastuzumab": {"target": "HER2/ERBB2", "gene": "ERBB2", "type": "antibody"},
            "pertuzumab": {"target": "HER2/ERBB2", "gene": "ERBB2", "type": "antibody"},
            
            # VEGF inhibitors
            "bevacizumab": {"target": "VEGF-A", "gene": "VEGFA", "type": "antibody"},
            "ramucirumab": {"target": "VEGFR2", "gene": "KDR", "type": "antibody"},
            
            # CD20 inhibitors
            "rituximab": {"target": "CD20", "gene": "MS4A1", "type": "antibody"},
            "obinutuzumab": {"target": "CD20", "gene": "MS4A1", "type": "antibody"},
            
            # EGFR inhibitors
            "cetuximab": {"target": "EGFR", "gene": "EGFR", "type": "antibody"},
            "panitumumab": {"target": "EGFR", "gene": "EGFR", "type": "antibody"},
            
            # TNF inhibitors
            "adalimumab": {"target": "TNF-alpha", "gene": "TNF", "type": "antibody"},
            "infliximab": {"target": "TNF-alpha", "gene": "TNF", "type": "antibody"},
            "etanercept": {"target": "TNF-alpha", "gene": "TNF", "type": "fusion_protein"}
        }
    
    async def get_drug_targets(self, drug_name: str, chembl_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get drug targets using multiple strategies"""
        
        # Normalize drug name
        drug_name_lower = drug_name.lower().strip()
        
        # Strategy 1: Check if it's a known biologic
        if drug_name_lower in self.biologic_targets:
            biologic_info = self.biologic_targets[drug_name_lower]
            return [{
                "target_name": biologic_info["target"],
                "gene_symbol": biologic_info["gene"],
                "target_type": "SINGLE PROTEIN",
                "drug_type": biologic_info["type"],
                "confidence_score": 10,
                "source": "biologic_database"
            }]
        
        # Strategy 2: Use ChEMBL for small molecules
        if chembl_id:
            # Try bioactivities first
            targets = await self.chembl_client.get_compound_targets(chembl_id)
            if targets:
                return targets
            
            # Try mechanisms if no bioactivities
            mechanisms = await self.chembl_client.get_drug_mechanisms(chembl_id)
            if mechanisms:
                mechanism_targets = []
                for mech in mechanisms:
                    if mech.get("target_chembl_id"):
                        # Get target details
                        target_details = await self.chembl_client._get_target_details(
                            mech["target_chembl_id"]
                        )
                        mechanism_targets.append({
                            "target_chembl_id": mech["target_chembl_id"],
                            "target_name": target_details.get("pref_name", mech.get("mechanism_of_action")),
                            "mechanism": mech.get("mechanism_of_action"),
                            "action_type": mech.get("action_type"),
                            "source": "mechanism"
                        })
                return mechanism_targets
        
        # Strategy 3: Try to find ChEMBL ID if not provided
        if not chembl_id:
            compound = await self.chembl_client.get_compound_by_name(drug_name)
            if compound:
                chembl_id = compound.get("molecule_chembl_id")
                if chembl_id:
                    return await self.get_drug_targets(drug_name, chembl_id)
        
        # Strategy 4: Use LLM as fallback (would need OpenAI client)
        # For now, return empty if no targets found
        return []
    
    def is_biologic(self, drug_name: str) -> bool:
        """Check if a drug is a biologic"""
        drug_name_lower = drug_name.lower().strip()
        
        # Check known biologics
        if drug_name_lower in self.biologic_targets:
            return True
        
        # Check suffixes
        biologic_suffixes = ["mab", "umab", "ximab", "zumab", "cept", "kin", "tide"]
        return any(drug_name_lower.endswith(suffix) for suffix in biologic_suffixes)
    
    async def get_target_diseases(self, target_name: str) -> List[str]:
        """Get diseases associated with a target"""
        # This would query disease databases
        # For now, return common associations
        disease_associations = {
            "PD-L1": ["lung cancer", "melanoma", "bladder cancer", "kidney cancer"],
            "PD-1": ["lung cancer", "melanoma", "hodgkin lymphoma", "head and neck cancer"],
            "HER2": ["breast cancer", "gastric cancer"],
            "VEGF": ["colorectal cancer", "lung cancer", "glioblastoma", "kidney cancer"],
            "EGFR": ["lung cancer", "colorectal cancer", "head and neck cancer"],
            "CD20": ["lymphoma", "leukemia", "rheumatoid arthritis"],
            "TNF-alpha": ["rheumatoid arthritis", "crohn's disease", "psoriasis"]
        }
        
        return disease_associations.get(target_name, [])

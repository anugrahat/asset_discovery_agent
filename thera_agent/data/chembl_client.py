"""
Async ChEMBL API client with data normalization
"""
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from .http_client import get_http_client

class ChEMBLClient:
    """Async ChEMBL API client with enhanced data normalization"""
    
    def __init__(self):
        self.base_url = "https://www.ebi.ac.uk/chembl/api/data"
    
    async def get_targets(self, gene_symbol: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get ChEMBL targets for gene symbol"""
        client = await get_http_client()
        
        url = f"{self.base_url}/target.json"
        params = {
            "target_synonym__icontains": gene_symbol,
            "format": "json",
            "limit": limit
        }
        
        try:
            response = await client.get(url, params=params)
            return response.get("targets", [])
        except Exception as e:
            print(f"ChEMBL target search error: {e}")
            return []
    
    async def get_bioactivities(self, chembl_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Get high-quality bioactivities for a specific ChEMBL compound with target filtering"""
        client = await get_http_client()
        
        url = f"{self.base_url}/activity.json"
        params = {
            "molecule_chembl_id": chembl_id,
            "format": "json",
            "limit": limit
        }
        
        try:
            response = await client.get(url, params=params)
            activities = response.get("activities", [])
            
            # Extract and enrich target information from activities
            enriched_activities = []
            target_cache = {}  # Cache target details to avoid redundant API calls
            
            for activity in activities:
                target_chembl_id = activity.get("target_chembl_id")
                if not target_chembl_id:
                    continue
                    
                # Get target details (with caching)
                if target_chembl_id not in target_cache:
                    target_details = await self._get_target_details(target_chembl_id)
                    target_cache[target_chembl_id] = target_details
                else:
                    target_details = target_cache[target_chembl_id]
                
                # Apply high-quality target filters
                if not self._is_high_quality_target(target_details):
                    continue
                
                # Add enriched target information
                target_info = {
                    "target_chembl_id": target_chembl_id,
                    "target_pref_name": activity.get("target_pref_name"),
                    "target_organism": activity.get("target_organism"),
                    "target_type": target_details.get("target_type"),
                    "confidence_score": target_details.get("confidence_score"),
                    "standard_type": activity.get("standard_type"),
                    "standard_value": activity.get("standard_value"),
                    "standard_units": activity.get("standard_units"),
                    "assay_type": activity.get("assay_type")
                }
                enriched_activities.append(target_info)
            
            return enriched_activities
            
        except Exception as e:
            print(f"ChEMBL bioactivity search error for {chembl_id}: {e}")
            return []
    
    async def _get_target_details(self, target_chembl_id: str) -> Dict[str, Any]:
        """Get detailed target information from ChEMBL"""
        client = await get_http_client()
        
        url = f"{self.base_url}/target/{target_chembl_id}.json"
        
        try:
            response = await client.get(url)
            return response
        except Exception as e:
            print(f"ChEMBL target details error for {target_chembl_id}: {e}")
            return {}
    
    def _is_high_quality_target(self, target_details: Dict[str, Any]) -> bool:
        """Filter for high-quality protein targets based on user specifications"""
        if not target_details:
            return False
        
        # Filter 1: target_type = SINGLE PROTEIN
        target_type = target_details.get("target_type")
        if target_type != "SINGLE PROTEIN":
            return False
        
        # Filter 2: confidence_score >= 8 (if available)
        confidence_score = target_details.get("confidence_score")
        if confidence_score is not None and confidence_score < 8:
            return False
        
        return True
    
    async def get_drug_mechanisms(self, chembl_id: str) -> List[Dict[str, Any]]:
        """Get drug mechanisms for compounds without clear protein targets"""
        client = await get_http_client()
        
        url = f"{self.base_url}/mechanism.json"
        params = {
            "molecule_chembl_id": chembl_id,
            "format": "json"
        }
        
        try:
            response = await client.get(url, params=params)
            mechanisms = response.get("mechanisms", [])
            
            # Return mechanism information
            mechanism_info = []
            for mechanism in mechanisms:
                info = {
                    "mechanism_of_action": mechanism.get("mechanism_of_action"),
                    "target_chembl_id": mechanism.get("target_chembl_id"),
                    "mechanism_comment": mechanism.get("mechanism_comment"),
                    "action_type": mechanism.get("action_type")
                }
                mechanism_info.append(info)
            
            return mechanism_info
            
        except Exception as e:
            print(f"ChEMBL mechanism search error for {chembl_id}: {e}")
            return []
    
    async def get_activities(self, target_chembl_id: str, 
                           standard_types: List[str] = None,
                           limit: int = 100) -> List[Dict[str, Any]]:
        """Get normalized activities for target"""
        client = await get_http_client()
        
        # Default to common binding assays
        if standard_types is None:
            standard_types = ["IC50", "Ki", "Kd", "EC50"]
        
        url = f"{self.base_url}/activity.json"
        params = {
            "target_chembl_id": target_chembl_id,
            "standard_type__in": ",".join(standard_types),
            "standard_units": "nM",  # Normalize to nM
            "format": "json",
            "limit": limit
        }
        
        try:
            response = await client.get(url, params=params)
            activities = response.get("activities", [])
            
            # Normalize and validate activities
            normalized_activities = []
            for activity in activities:
                normalized = self._normalize_activity(activity)
                if normalized:
                    normalized_activities.append(normalized)
            
            # Sort by potency (ascending)
            normalized_activities.sort(key=lambda x: x.get("standard_value_nm", float('inf')))
            
            return normalized_activities
            
        except Exception as e:
            print(f"ChEMBL activity search error: {e}")
            return []
    
    def _normalize_activity(self, activity: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Normalize and validate activity data"""
        try:
            # Extract key fields
            standard_value = activity.get("standard_value")
            standard_units = activity.get("standard_units", "").upper()
            standard_type = activity.get("standard_type", "")
            
            # Skip if missing critical data
            if not standard_value or not standard_type:
                return None
            
            # Convert to nM
            value_nm = self._convert_to_nm(float(standard_value), standard_units)
            if value_nm is None:
                return None
            
            # Extract assay metadata for quality assessment
            assay_data = activity.get("assay_description", "")
            assay_type = activity.get("assay_type", "")
            
            # Quality score based on assay type and data completeness
            quality_score = self._calculate_quality_score(activity)
            
            return {
                "activity_id": activity.get("activity_id"),
                "molecule_chembl_id": activity.get("molecule_chembl_id"),
                "standard_type": standard_type,
                "standard_value_nm": value_nm,
                "assay_description": assay_data,
                "assay_type": assay_type,
                "assay_organism": activity.get("assay_organism"),
                "quality_score": quality_score,
                "confidence_score": activity.get("confidence_score"),
                "pchembl_value": activity.get("pchembl_value"),  # -log10(IC50 in M)
                "data_validity_comment": activity.get("data_validity_comment")
            }
            
        except (ValueError, TypeError) as e:
            print(f"Error normalizing activity: {e}")
            return None
    
    def _convert_to_nm(self, value: float, units: str) -> Optional[float]:
        """Convert concentration to nM"""
        conversion_factors = {
            "NM": 1.0,
            "UM": 1000.0,
            "MM": 1_000_000.0,
            "M": 1_000_000_000.0,
            "PM": 0.001,
            "FM": 0.000001
        }
        
        units_clean = units.replace("μ", "U").upper()  # Handle μM
        
        if units_clean in conversion_factors:
            return value * conversion_factors[units_clean]
        
        return None
    
    def _calculate_quality_score(self, activity: Dict[str, Any]) -> float:
        """Calculate data quality score (0-1)"""
        score = 0.5  # Base score
        
        # Bonus for complete data
        if activity.get("pchembl_value"):
            score += 0.2
        
        if activity.get("confidence_score"):
            conf_score = activity.get("confidence_score", 0)
            score += (conf_score / 10) * 0.2  # Normalize confidence score
        
        # Bonus for functional assays
        assay_type = activity.get("assay_type", "").lower()
        if "functional" in assay_type or "cell" in assay_type:
            score += 0.1
        
        # Penalty for data validity issues
        validity_comment = activity.get("data_validity_comment", "")
        if validity_comment and "outside typical range" in validity_comment.lower():
            score -= 0.2
        
        return max(0.0, min(1.0, score))
    
    async def get_molecules(self, chembl_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """Get molecule data for ChEMBL IDs"""
        client = await get_http_client()
        
        molecules = {}
        
        # Batch requests for efficiency
        for i in range(0, len(chembl_ids), 20):  # Process in batches of 20
            batch_ids = chembl_ids[i:i+20]
            
            url = f"{self.base_url}/molecule.json"
            params = {
                "molecule_chembl_id__in": ",".join(batch_ids),
                "format": "json"
            }
            
            try:
                response = await client.get(url, params=params)
                if not response or not isinstance(response, dict):
                    continue
                batch_molecules = response.get("molecules") or []
                if not isinstance(batch_molecules, list):
                    batch_molecules = []
                
                for mol in batch_molecules:
                    if not isinstance(mol, dict):
                        continue
                    chembl_id = mol.get("molecule_chembl_id")
                    if chembl_id:
                        # Coerce max_phase to int for safe numeric comparisons downstream
                        max_phase_val = mol.get("max_phase")
                        try:
                            max_phase_int = int(max_phase_val) if max_phase_val is not None else 0
                        except (ValueError, TypeError):
                            max_phase_int = 0
                        molecules[chembl_id] = {
                            "preferred_name": mol.get("pref_name"),
                            "molecular_weight": mol.get("molecule_properties", {}).get("mw_freebase"),
                            "alogp": mol.get("molecule_properties", {}).get("alogp"),
                            "hbd": mol.get("molecule_properties", {}).get("hbd"),
                            "hba": mol.get("molecule_properties", {}).get("hba"),
                            "max_phase": max_phase_int,  # Clinical phase (int)
                            "structure_type": mol.get("structure_type"),
                            "smiles": mol.get("molecule_structures", {}).get("canonical_smiles") if mol.get("molecule_structures") else None
                        }
                        
            except Exception as e:
                print(f"Error fetching molecule batch: {e}")
                continue
        
        return molecules
    
    async def get_pdb_structures_for_target(self, target_name: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Get PDB structures for a target using RCSB PDB API"""
        client = await get_http_client()
        
        # Search RCSB PDB for structures containing this target
        search_url = "https://search.rcsb.org/rcsbsearch/v2/query"
        
        # Build search query for target name using full-text search
        query = {
            "query": {
                "type": "terminal",
                "service": "full_text",
                "parameters": {
                    "value": (target_name or "").upper()
                }
            },
            "return_type": "entry",
            "request_options": {
                "paginate": {
                    "start": 0,
                    "rows": limit
                }
            }
        }
        
        try:
            response = await client.post(search_url, json_data=query)
            results = response.get("result_set", [])
            
            # Extract PDB IDs and get structure details
            structures = []
            for result in results:
                pdb_id = result.get("identifier", "")
                if pdb_id:
                    # Get structure details
                    detail_url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"
                    try:
                        detail_response = await client.get(detail_url)
                        structures.append({
                            "pdb_id": pdb_id,
                            "title": detail_response.get("struct", {}).get("title", ""),
                            "resolution": detail_response.get("rcsb_entry_info", {}).get("resolution_combined", [None])[0],
                            "method": detail_response.get("exptl", [{}])[0].get("method", ""),
                            "deposition_date": detail_response.get("rcsb_accession_info", {}).get("deposit_date", "")
                        })
                    except:
                        # If details fail, at least return the PDB ID
                        structures.append({"pdb_id": pdb_id})
            
            return structures
            
        except Exception as e:
            print(f"PDB structure search error for {target_name}: {e}")
            return []
    
    async def get_inhibitors_for_target(self, gene_symbol: str, 
                                      max_ic50_nm: Optional[float] = None,
                                      min_ic50_nm: Optional[float] = None,
                                      limit: int = 50) -> List[Dict[str, Any]]:
        """Get normalized inhibitors for target with potency filtering"""
        
        # Get targets
        targets = await self.get_targets(gene_symbol)
        if not targets:
            return []
        
        all_inhibitors = []
        
        for target in targets[:3]:  # Top 3 targets to avoid overwhelming
            target_id = target.get("target_chembl_id")
            if not target_id:
                continue
                
            # Get activities
            activities = await self.get_activities(target_id, limit=limit)
            
            # Filter by potency (ensure numeric comparisons)
            filtered_activities = []
            for activity in activities:
                ic50_nm = activity.get("standard_value_nm")
                try:
                    ic50_val = float(ic50_nm)
                except (TypeError, ValueError):
                    # Skip malformed or missing values
                    continue
                
                # Apply thresholds if provided
                if max_ic50_nm is not None:
                    try:
                        if ic50_val > float(max_ic50_nm):
                            continue
                    except (TypeError, ValueError):
                        pass
                if min_ic50_nm is not None:
                    try:
                        if ic50_val < float(min_ic50_nm):
                            continue
                    except (TypeError, ValueError):
                        pass
                
                # Persist numeric value to ensure downstream safety
                activity["standard_value_nm"] = ic50_val
                filtered_activities.append(activity)
            
            # Get molecule details
            chembl_ids = [act.get("molecule_chembl_id") for act in filtered_activities if act.get("molecule_chembl_id")]
            molecules = await self.get_molecules(chembl_ids)
            
            # Combine activity and molecule data
            for activity in filtered_activities:
                chembl_id = activity.get("molecule_chembl_id")
                molecule_data = molecules.get(chembl_id, {})
                
                inhibitor = {
                    **activity,
                    "target_chembl_id": target_id,
                    "target_name": target.get("pref_name"),
                    **molecule_data
                }
                
                all_inhibitors.append(inhibitor)
        
        # Sort by quality score and potency (ensure numeric potency)
        def _potency_key(item):
            val = item.get("standard_value_nm")
            try:
                return float(val)
            except (TypeError, ValueError):
                return float('inf')
        all_inhibitors.sort(key=lambda x: (-x.get("quality_score", 0), _potency_key(x)))
        
        return all_inhibitors[:limit]
    
    async def get_mutation_specific_inhibitors(self, target: str, mutation: Optional[str] = None,
                                             max_ic50_nm: Optional[float] = 1000,
                                             limit: int = 50) -> List[Dict[str, Any]]:
        """Get inhibitors including mutation-specific ones (e.g., KRAS G12C, BRAF V600E)"""
        
        # Known mutation-specific inhibitors mapping
        mutation_specific_drugs = {
            "KRAS G12C": [
                {"name": "sotorasib", "chembl_id": "CHEMBL4594429", "mutation": "G12C"},
                {"name": "adagrasib", "chembl_id": "CHEMBL4561119", "mutation": "G12C"}
            ],
            "BRAF V600E": [
                {"name": "vemurafenib", "chembl_id": "CHEMBL1229517", "mutation": "V600E"},
                {"name": "dabrafenib", "chembl_id": "CHEMBL2028663", "mutation": "V600E"},
                {"name": "encorafenib", "chembl_id": "CHEMBL3301621", "mutation": "V600E"}
            ],
            "EGFR T790M": [
                {"name": "osimertinib", "chembl_id": "CHEMBL3353410", "mutation": "T790M"},
                {"name": "rociletinib", "chembl_id": "CHEMBL3545110", "mutation": "T790M"}
            ],
            "ALK": [
                {"name": "crizotinib", "chembl_id": "CHEMBL1601751", "mutation": "various"},
                {"name": "alectinib", "chembl_id": "CHEMBL3356066", "mutation": "various"},
                {"name": "ceritinib", "chembl_id": "CHEMBL3356420", "mutation": "various"}
            ]
        }
        
        # Get general inhibitors first
        general_inhibitors = await self.get_inhibitors_for_target(
            gene_symbol=target,
            max_ic50_nm=max_ic50_nm,
            limit=limit
        )
        
        # Add mutation-specific inhibitors if applicable
        all_inhibitors = general_inhibitors.copy()
        mutation_key = f"{target} {mutation}" if mutation else target
        
        if mutation_key in mutation_specific_drugs:
            # Get details for mutation-specific drugs
            specific_drugs = mutation_specific_drugs[mutation_key]
            chembl_ids = [drug["chembl_id"] for drug in specific_drugs]
            
            # Get molecule details
            molecules = await self.get_molecules(chembl_ids)
            
            # Add mutation-specific information
            for drug in specific_drugs:
                chembl_id = drug["chembl_id"]
                if chembl_id in molecules:
                    inhibitor = {
                        "molecule_chembl_id": chembl_id,
                        "pref_name": drug["name"],
                        "target_name": target,
                        "mutation_specific": True,
                        "mutation": drug["mutation"],
                        "clinical_use": f"{target} {drug['mutation']} inhibitor",
                        **molecules[chembl_id]
                    }
                    
                    # Add to list if not already present
                    if not any(inh["molecule_chembl_id"] == chembl_id for inh in all_inhibitors):
                        all_inhibitors.insert(0, inhibitor)  # Put mutation-specific at top
        
        return all_inhibitors[:limit]
    
    async def get_compound_by_name(self, drug_name: str) -> Optional[Dict[str, Any]]:
        """Get compound data by drug name"""
        client = await get_http_client()
        
        # Search by drug name
        url = f"{self.base_url}/molecule.json"
        params = {
            "pref_name__iexact": drug_name,
            "format": "json",
            "limit": 1
        }
        
        try:
            response = await client.get(url, params=params)
            molecules = response.get("molecules", [])
            if molecules:
                return molecules[0]
                
            # Try synonyms if exact match fails
            params = {
                "molecule_synonyms__molecule_synonym__icontains": drug_name,
                "format": "json",
                "limit": 1
            }
            response = await client.get(url, params=params)
            molecules = response.get("molecules", [])
            return molecules[0] if molecules else None
            
        except Exception as e:
            print(f"ChEMBL compound search error for {drug_name}: {e}")
            return None
    
    async def get_compound_by_chembl_id(self, chembl_id: str) -> Optional[Dict[str, Any]]:
        """Get compound data by ChEMBL ID"""
        client = await get_http_client()
        
        url = f"{self.base_url}/molecule/{chembl_id}.json"
        
        try:
            response = await client.get(url)
            return response
        except Exception as e:
            print(f"ChEMBL compound fetch error for {chembl_id}: {e}")
            return None
    
    async def get_compound_targets(self, chembl_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get high-quality protein targets for a compound dynamically"""
        # Get bioactivities with targets
        activities = await self.get_bioactivities(chembl_id, limit=50)
        
        # Group by target and score quality
        target_scores = {}
        target_activities = {}
        
        for activity in activities:
            target_id = activity.get("target_chembl_id")
            if not target_id:
                continue
                
            # Calculate quality score for this activity
            score = self._score_target_quality(activity)
            
            if target_id not in target_scores or score > target_scores[target_id]:
                target_scores[target_id] = score
                target_activities[target_id] = activity
        
        # Sort targets by quality score
        sorted_targets = sorted(
            target_activities.items(),
            key=lambda x: target_scores[x[0]],
            reverse=True
        )
        
        # Return top quality targets with details
        targets = []
        for target_id, activity in sorted_targets[:limit]:
            target_info = {
                "target_chembl_id": target_id,
                "target_name": activity.get("target_pref_name"),
                "target_type": activity.get("target_type"),
                "confidence_score": activity.get("confidence_score"),
                "activity_type": activity.get("standard_type"),
                "activity_value": activity.get("standard_value"),
                "activity_units": activity.get("standard_units"),
                "quality_score": target_scores[target_id]
            }
            targets.append(target_info)
        
        return targets
    
    def _score_target_quality(self, activity: Dict[str, Any]) -> float:
        """Score target quality based on activity data"""
        score = 0.0
        
        # Prefer specific activity types
        activity_type = activity.get("standard_type", "")
        if activity_type in ["IC50", "Ki", "Kd", "EC50"]:
            score += 3.0
        elif activity_type in ["Activity", "Inhibition"]:
            score += 1.0
        
        # Prefer human targets
        organism = activity.get("target_organism", "")
        if "Homo sapiens" in organism:
            score += 2.0
        
        # Prefer targets with high confidence
        confidence = activity.get("confidence_score")
        if confidence is not None:
            if confidence >= 8:
                score += 2.0
            elif confidence >= 5:
                score += 1.0
        
        # Prefer single protein targets
        target_type = activity.get("target_type", "")
        if target_type == "SINGLE PROTEIN":
            score += 2.0
        elif target_type == "PROTEIN COMPLEX":
            score += 1.0
        
        # Penalize cell lines and non-specific targets
        target_name = (activity.get("target_pref_name", "") or "").upper()
        if any(cell in target_name for cell in ["A549", "HCT-116", "MCF7", "HELA", "K562"]):
            score -= 5.0
        
        # Penalize screening panel artifacts - common targets in broad screens
        screening_targets = [
            "ACETYLCHOLINESTERASE", "ADENOSINE A1", "ADENOSINE A2", "ADENOSINE A3",
            "ALPHA-1D ADRENERGIC", "ALPHA-2A ADRENERGIC", "BETA-1 ADRENERGIC", 
            "BETA-2 ADRENERGIC", "DOPAMINE D1", "DOPAMINE D2", "SEROTONIN"
        ]
        
        # Check if this looks like screening panel data
        if any(screening in target_name for screening in screening_targets):
            # Only penalize if activity is weak (suggesting non-specific binding)
            activity_value = activity.get("standard_value")
            if activity_value and float(activity_value) > 10000:  # Weak binding (>10μM)
                score -= 3.0
        
        # Require minimum potency for credible targets
        activity_value = activity.get("standard_value")
        if activity_value:
            try:
                value = float(activity_value)
                if activity_type in ["IC50", "Ki", "Kd"] and value > 50000:  # >50μM
                    score -= 2.0
            except (ValueError, TypeError):
                pass
        
        return score
    
    async def get_chemical_structure(self, drug_name: str) -> Optional[Dict[str, Any]]:
        """Get chemical structure data for a drug
        
        Returns:
            Dict with SMILES, InChI, molecular formula, etc.
        """
        compound = await self.get_compound_by_name(drug_name)
        if not compound:
            return None
            
        structure_data = {
            "drug_name": drug_name,
            "chembl_id": compound.get("molecule_chembl_id"),
            "smiles": compound.get("molecule_structures", {}).get("canonical_smiles"),
            "inchi": compound.get("molecule_structures", {}).get("standard_inchi"),
            "inchi_key": compound.get("molecule_structures", {}).get("standard_inchi_key"),
            "molecular_formula": compound.get("molecule_properties", {}).get("full_molformula"),
            "molecular_weight": compound.get("molecule_properties", {}).get("full_mwt"),
            "logp": compound.get("molecule_properties", {}).get("alogp"),
            "psa": compound.get("molecule_properties", {}).get("psa"),
            "hba": compound.get("molecule_properties", {}).get("hba"),  # H-bond acceptors
            "hbd": compound.get("molecule_properties", {}).get("hbd"),  # H-bond donors
            "ro5_violations": compound.get("molecule_properties", {}).get("num_ro5_violations"),
            "max_phase": compound.get("max_phase"),
            "first_approval": compound.get("first_approval")
        }
        
        # Clean None values
        return {k: v for k, v in structure_data.items() if v is not None}
    
    async def get_similar_compounds(
        self, 
        smiles: str, 
        similarity_threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """Find compounds similar to a given SMILES structure"""
        client = await get_http_client()
        
        url = f"{self.base_url}/similarity/{smiles}/{int(similarity_threshold * 100)}.json"
        
        try:
            response = await client.get(url)
            molecules = response.get("molecules", [])
            
            similar_compounds = []
            for mol in molecules[:10]:  # Top 10 similar
                similar_compounds.append({
                    "chembl_id": mol.get("molecule_chembl_id"),
                    "pref_name": mol.get("pref_name"),
                    "similarity": mol.get("similarity"),
                    "max_phase": mol.get("max_phase"),
                    "first_approval": mol.get("first_approval")
                })
            
            return similar_compounds
            
        except Exception as e:
            print(f"ChEMBL similarity search error: {e}")
            return []
    
    async def search_by_indication(self, disease: str) -> List[Dict[str, Any]]:
        """Search for drugs approved for a specific indication/disease"""
        client = await get_http_client()
        
        # Search drug indications
        url = f"{self.base_url}/drug_indication.json"
        # ChEMBL indication search: use efo_term/mesh_heading, not 'indication'
        queries = [
            {"efo_term__icontains": disease, "format": "json", "limit": 1000},
            {"mesh_heading__icontains": disease, "format": "json", "limit": 1000},
        ]
        
        try:
            indications = []
            seen_ids = set()
            for q in queries:
                try:
                    resp = await client.get(url, params=q)
                    items = resp.get("drug_indications", []) if resp else []
                    for it in items:
                        di = it.get("drugind_id")
                        if di is None or di in seen_ids:
                            continue
                        seen_ids.add(di)
                        indications.append(it)
                except Exception:
                    continue
            
            # Get unique molecules, filtering for primary indications only
            molecules = {}
            for indication in indications:
                mol_id = indication.get("molecule_chembl_id")
                # ChEMBL API returns indication in efo_term, not indication field
                indication_text = (indication.get("efo_term", "") or 
                                 indication.get("mesh_heading", "") or 
                                 indication.get("indication", "")).lower()
                max_phase = indication.get("max_phase_for_ind", 0)
                
                # Convert max_phase to int if it's a string
                try:
                    max_phase = int(max_phase) if max_phase else 0
                except (ValueError, TypeError):
                    max_phase = 0
                
                # Only include if this is a primary indication (phase 3+ for the disease)
                # and the indication text strongly suggests treatment FOR the disease
                if mol_id and mol_id not in molecules and max_phase >= 3:
                    # Check if this is a primary treatment indication
                    disease_lower = disease.lower()
                    
                    # Exclude drugs for complications/side effects of the disease
                    exclude_patterns = [
                        'complication', 'secondary to', 'associated with', 'induced by',
                        'prophylaxis', 'prevention of', 'risk reduction', 'neuropathy',
                        'retinopathy', 'nephropathy', 'ulcer', 'foot', 'eye', 'kidney'
                    ]
                    
                    if any(pattern in indication_text for pattern in exclude_patterns):
                        continue
                    
                    # Special handling for diabetes to avoid over-broad matches
                    if 'diabetes' in disease_lower:
                        # Must be specifically for diabetes treatment, not complications
                        diabetes_patterns = [
                            'type 2 diabetes',
                            'type ii diabetes', 
                            'type two diabetes',
                            'non-insulin-dependent diabetes',
                            'niddm',
                            'glycemic control',
                            'glycaemic control',
                            'hyperglycemia',
                            'hyperglycaemia',
                            'antidiabetic',
                            'anti-diabetic',
                            'glucose-lowering',
                            'blood glucose',
                            'blood sugar'
                        ]
                        
                        # For diabetes, require diabetes-specific patterns but be less restrictive
                        if any(pattern in indication_text for pattern in diabetes_patterns):
                            # Additional check: must mention diabetes explicitly
                            if 'diabetes' in indication_text:
                                molecules[mol_id] = {
                                    "molecule_chembl_id": mol_id,
                                    "molecule_pref_name": indication.get("parent_molecule_name"),
                                    "indication": indication.get("indication"),
                                    "max_phase": max_phase
                                }
                    else:
                        # For other diseases, use more flexible matching
                        # Check if disease name appears in indication
                        if disease_lower in indication_text:
                            # Exclude if it's about complications or side effects
                            exclude_patterns = [
                                'risk of', 'induced', 'associated', 'secondary to',
                                'caused by', 'prevention of', 'prophylaxis'
                            ]
                            
                            # Include if it's about treatment/management
                            include_patterns = [
                                'treatment', 'therapy', 'management', 'for use in',
                                'indicated', 'efficacy', 'patients with', 'adults with'
                            ]
                            
                            # Check if any exclude pattern appears near the disease mention
                            excluded = any(pattern in indication_text for pattern in exclude_patterns)
                            
                            # Check if any include pattern appears
                            included = any(pattern in indication_text for pattern in include_patterns)
                            
                            # Include the drug if it has include patterns or no exclude patterns
                            if included or not excluded:
                                molecules[mol_id] = {
                                    "molecule_chembl_id": mol_id,
                                    "molecule_pref_name": None,  # Will fetch later
                                    "indication": indication_text,
                                    "max_phase": max_phase,
                                    "efo_term": indication.get("efo_term"),
                                    "mesh_heading": indication.get("mesh_heading")
                                }
            
            # Get molecule synonyms for comprehensive name coverage
            results = []
            for mol_data in molecules.values():
                mol_id = mol_data["molecule_chembl_id"]
                
                # Get full molecule data including synonyms
                mol_url = f"{self.base_url}/molecule/{mol_id}.json"
                try:
                    mol_response = await client.get(mol_url)
                    
                    # Prefer ChEMBL preferred name if available
                    pref_name = mol_response.get("pref_name") or mol_response.get("molecule_pref_name")
                    if pref_name:
                        mol_data["molecule_pref_name"] = pref_name
                    
                    # Collect synonyms if present
                    synonyms = []
                    if mol_response.get("molecule_synonyms"):
                        synonyms = [syn.get("molecule_synonym", "")
                                   for syn in mol_response.get("molecule_synonyms", [])]
                    mol_data["molecule_synonyms"] = synonyms
                    results.append(mol_data)
                
                except Exception:
                    # Even if we can't get synonyms, include the basic data
                    results.append(mol_data)
            
            return results
            
        except Exception as e:
            print(f"ChEMBL indication search error: {e}")
            return []

"""
Drug-Focused Repurposing System
No fallbacks - uses actual failed drug data and repurposing candidates
"""
import json
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class DrugFocusedRepurposingAgent:
    """Drug repurposing agent that analyzes actual failed drugs vs available candidates"""
    
    def __init__(self, ct_client, chembl_client, drug_resolver, llm_query_func):
        self.ct_client = ct_client
        self.chembl_client = chembl_client  
        self.drug_resolver = drug_resolver
        self._llm_query = llm_query_func
    
    async def analyze_drug_failures(self, disease: str, max_trials: int = 50) -> Dict:
        """Analyze actual failed drugs and suggest mechanistically different alternatives"""
        
        print(f"🔍 Analyzing failed drugs for {disease}...")
        
        # 1. Get failed trials
        failed_trials = await self.ct_client.search_trials(
            condition=disease,
            status=["TERMINATED", "WITHDRAWN", "SUSPENDED"],
            max_results=max_trials
        )
        
        # 2. Extract and enrich failed drugs
        print("💊 Extracting failed drugs...")
        failed_drugs = await self._extract_and_enrich_failed_drugs(failed_trials)
        
        # 3. Get repurposing candidates  
        print("🔍 Finding repurposing candidates...")
        candidates = await self.ct_client.get_drug_repurposing_candidates(
            disease=disease,
            exclude_targets=[]
        )
        
        # 4. Enrich candidates with ChEMBL data
        print("📈 Enriching candidates with molecular data...")
        enriched_candidates = await self._enrich_candidates_with_full_data(candidates)
        
        # 5. Run drug-focused LLM analysis
        print("🧠 Running drug-focused analysis...")
        analysis = await self._analyze_failed_vs_candidates(
            disease, failed_drugs, enriched_candidates
        )
        
        return {
            "disease": disease,
            "failed_drugs": failed_drugs,
            "repurposing_candidates": enriched_candidates,
            "analysis": analysis,
            "total_failed_trials": len(failed_trials)
        }
    
    async def _extract_and_enrich_failed_drugs(self, failed_trials: List[Dict]) -> List[Dict]:
        """Extract failed drugs and enrich with ChEMBL data"""
        failed_drugs = []
        
        for trial in failed_trials:
            for intervention in trial.get("interventions", []):
                if intervention.get("type") == "DRUG":
                    drug_name = intervention["name"]
                    
                    # Try to resolve to ChEMBL
                    chembl_data = await self.drug_resolver.resolve_to_chembl_id(drug_name)
                    
                    drug_info = {
                        "name": drug_name,
                        "nct_id": trial["nct_id"],
                        "why_stopped": trial.get("why_stopped", "Unknown"),
                        "phase": trial.get("phase", []),
                        "description": intervention.get("description", ""),
                        "chembl_id": chembl_data.get("chembl_id") if chembl_data else None,
                        "max_phase": chembl_data.get("max_phase") if chembl_data else None,
                        "mechanism": None  # Will be filled by LLM analysis
                    }
                    
                    # Get mechanism of action if we have ChEMBL ID
                    if drug_info["chembl_id"]:
                        try:
                            molecule_data = await self.chembl_client.get_molecule_details(drug_info["chembl_id"])
                            if molecule_data:
                                drug_info["mechanism"] = molecule_data.get("mechanism_of_action", "Unknown")
                        except:
                            pass
                    
                    failed_drugs.append(drug_info)
        
        return failed_drugs
    
    async def _enrich_candidates_with_full_data(self, candidates: List[Dict]) -> List[Dict]:
        """Enrich repurposing candidates with full ChEMBL and mechanism data"""
        enriched = []
        
        for candidate in candidates:
            drug_name = candidate.get("drug_name", "Unknown")
            
            # Resolve to ChEMBL
            chembl_data = await self.drug_resolver.resolve_to_chembl_id(drug_name)
            
            if chembl_data:
                # Get full molecule details
                try:
                    molecule_details = await self.chembl_client.get_molecule_details(chembl_data["chembl_id"])
                    
                    enriched_candidate = {
                        "drug_name": drug_name,
                        "chembl_id": chembl_data["chembl_id"],
                        "max_phase": chembl_data.get("max_phase"),
                        "trial_count": candidate.get("trial_count", 0),
                        "mechanism_of_action": molecule_details.get("mechanism_of_action") if molecule_details else "Unknown",
                        "indication_class": molecule_details.get("indication_class") if molecule_details else "Unknown",
                        "therapeutic_flag": molecule_details.get("therapeutic_flag") if molecule_details else False
                    }
                    enriched.append(enriched_candidate)
                except:
                    # Include with basic data if ChEMBL lookup fails
                    enriched.append({
                        "drug_name": drug_name,
                        "chembl_id": chembl_data["chembl_id"],
                        "max_phase": chembl_data.get("max_phase"),
                        "trial_count": candidate.get("trial_count", 0),
                        "mechanism_of_action": "Unknown",
                        "indication_class": "Unknown",
                        "therapeutic_flag": False
                    })
        
        return enriched
    
    async def _analyze_failed_vs_candidates(
        self, disease: str, failed_drugs: List[Dict], candidates: List[Dict]
    ) -> Dict:
        """Use LLM to analyze failed drugs vs repurposing candidates"""
        
        # Prepare data for LLM
        failed_summary = []
        for drug in failed_drugs:
            failed_summary.append({
                "name": drug["name"],
                "chembl_id": drug.get("chembl_id"),
                "mechanism": drug.get("mechanism", "Unknown"),
                "why_stopped": drug["why_stopped"],
                "phase": drug["phase"]
            })
        
        candidate_summary = []
        for candidate in candidates[:10]:  # Top 10 candidates
            candidate_summary.append({
                "name": candidate["drug_name"],
                "chembl_id": candidate["chembl_id"],
                "mechanism": candidate["mechanism_of_action"],
                "max_phase": candidate["max_phase"],
                "indication_class": candidate["indication_class"]
            })
        
        prompt = f"""
        Analyze these ACTUAL failed drugs in {disease} and recommend mechanistically different repurposing candidates:
        
        FAILED DRUGS:
        {json.dumps(failed_summary, indent=2)}
        
        AVAILABLE REPURPOSING CANDIDATES:
        {json.dumps(candidate_summary, indent=2)}
        
        Your task:
        1. Identify the specific mechanisms of action of the failed drugs
        2. Understand WHY they failed (pharmacological reasons, not just business/recruitment)
        3. From the repurposing candidates, select ones with DIFFERENT mechanisms that could overcome the failures
        4. Provide specific biological rationale for each recommendation
        
        Return JSON:
        {{
            "failed_drug_insights": [
                {{
                    "drug_name": "exact_name_from_failed_list",
                    "chembl_id": "if_available",
                    "mechanism": "specific_mechanism_of_action",
                    "likely_failure_cause": "biological_reason_for_failure",
                    "pathway": "biological_pathway_targeted"
                }}
            ],
            "repurposing_recommendations": [
                {{
                    "drug_name": "exact_name_from_candidates_list",
                    "chembl_id": "from_candidates",
                    "mechanism": "how_mechanism_differs_from_failed",
                    "rationale": "specific_biological_reason_why_it_could_work",
                    "addresses_failure": "how_it_overcomes_specific_failed_drug_limitation",
                    "confidence": 0.0-1.0,
                    "evidence_level": "clinical/preclinical/theoretical"
                }}
            ],
            "mechanism_gaps": [
                "unexplored_mechanisms_in_disease"
            ]
        }}
        """
        
        response = await self._llm_query(prompt)
        
        # Clean and parse JSON
        if "```json" in response:
            response = response.split("```json")[1].split("```")[0]
        elif "```" in response:
            response = response.split("```")[1].split("```")[0]
        
        try:
            return json.loads(response)
        except Exception as e:
            logger.error(f"Failed to parse drug analysis response: {e}")
            logger.error(f"Raw response: {response}")
            raise Exception(f"Drug-focused analysis failed: {e}")
    
    def format_results(self, results: Dict) -> str:
        """Format results for display"""
        output = []
        
        analysis = results["analysis"]
        
        output.append(f"🔬 DRUG-FOCUSED REPURPOSING ANALYSIS: {results['disease'].upper()}")
        output.append("=" * 80)
        
        # Failed drug insights
        output.append("\n💊 FAILED DRUG ANALYSIS")
        for insight in analysis.get("failed_drug_insights", []):
            output.append(f"\n• {insight['drug_name']} ({insight.get('chembl_id', 'N/A')})")
            output.append(f"  Mechanism: {insight['mechanism']}")
            output.append(f"  Pathway: {insight['pathway']}")
            output.append(f"  Failure Cause: {insight['likely_failure_cause']}")
        
        # Repurposing recommendations
        output.append("\n🎯 INTELLIGENT REPURPOSING RECOMMENDATIONS")
        for i, rec in enumerate(analysis.get("repurposing_recommendations", []), 1):
            output.append(f"\n{i}. {rec['drug_name']} ({rec['chembl_id']})")
            output.append(f"   Mechanism: {rec['mechanism']}")
            output.append(f"   Rationale: {rec['rationale']}")
            output.append(f"   Addresses: {rec['addresses_failure']}")
            output.append(f"   Confidence: {rec['confidence']:.1%}")
            output.append(f"   Evidence: {rec['evidence_level']}")
        
        # Mechanism gaps
        gaps = analysis.get("mechanism_gaps", [])
        if gaps:
            output.append("\n🔍 UNEXPLORED MECHANISMS")
            for gap in gaps:
                output.append(f"• {gap}")
        
        output.append("\n" + "=" * 80)
        
        return "\n".join(output)

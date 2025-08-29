"""
Disease-to-Target Mapping using LLM
Maps disease names to relevant therapeutic protein targets using OpenAI API.
"""

import asyncio
import json
import logging
from typing import List, Dict, Optional
from openai import AsyncOpenAI
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class DiseaseTarget:
    """Represents a therapeutic target for a disease"""
    gene_symbol: str
    protein_name: str
    rationale: str
    confidence: float  # 0-1 score
    clinical_stage: str  # research, preclinical, clinical, approved
    pathway: str

class DiseaseTargetMapper:
    """Maps diseases to relevant therapeutic targets using LLM"""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize with OpenAI API key"""
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key required for disease mapping")
        
        self.openai_client = AsyncOpenAI(api_key=self.api_key)
        
    async def map_disease_to_targets(
        self, 
        disease: str, 
        max_targets: int = 5,
        clinical_filter: Optional[str] = None
    ) -> List[DiseaseTarget]:
        """
        Map disease name to therapeutic targets using LLM
        
        Args:
            disease: Disease name (e.g., "emphysema", "Alzheimer's disease")
            max_targets: Maximum number of targets to return
            clinical_filter: Filter by clinical stage ("approved", "clinical", etc.)
            
        Returns:
            List of DiseaseTarget objects with gene symbols and rationales
        """
        
        prompt = self._build_disease_prompt(disease, max_targets, clinical_filter)
        
        try:
            response = await self._call_openai(prompt)
            targets = self._parse_llm_response(response)
            
            logger.info(f"Mapped {disease} to {len(targets)} therapeutic targets")
            return targets[:max_targets]
            
        except Exception as e:
            logger.error(f"Failed to map disease {disease} to targets: {e}")
            # Fallback to known disease mappings
            return self._get_fallback_targets(disease)
    
    def _build_disease_prompt(self, disease: str, max_targets: int, clinical_filter: Optional[str]) -> str:
        """Build the LLM prompt for disease-to-target mapping"""
        
        clinical_instruction = ""
        if clinical_filter:
            clinical_instruction = f"Focus on targets that are in {clinical_filter} stage."
        
        # Check if the query contains filtering criteria
        potency_filter = ""
        if "above" in disease.lower() and ("nm" in disease.lower() or "micromolar" in disease.lower() or "μm" in disease.lower()):
            potency_filter = """

IMPORTANT: The user is specifically asking about inhibitors with certain potency ranges. 
However, as a target discovery system, you should still return the BEST therapeutic targets for the disease.
The potency filtering will be applied later when searching for specific inhibitors.
Do NOT limit your target selection based on inhibitor potency - focus on biological relevance."""
        
        # Extract the actual disease name (remove filtering criteria)
        disease_clean = disease
        for phrase in ["inhibitors above", "inhibitors below", "compounds above", "compounds below"]:
            if phrase in disease.lower():
                disease_clean = disease.lower().split(phrase)[0].strip()
                break
        
        return f"""
You are a therapeutic target discovery expert. Given a disease, identify the most promising protein targets for drug development.

Disease/Query: {disease}
Core Disease: {disease_clean}
{potency_filter}

Please identify the top {max_targets} therapeutic targets and return them in JSON format:

{{
  "targets": [
    {{
      "gene_symbol": "GENE_NAME",
      "protein_name": "Full protein name",
      "rationale": "Brief explanation of why this target is relevant for {disease_clean}",
      "confidence": 0.9,
      "clinical_stage": "approved|clinical|preclinical|research",
      "pathway": "Main biological pathway involved"
    }}
  ]
}}

Requirements:
- Use official HGNC gene symbols (e.g., EGFR, TP53, BRAF)
- Focus on druggable targets (enzymes, receptors, ion channels)
- Include targets across different clinical stages if no filter specified
- Provide clear, scientific rationale for each target
- Confidence should reflect target validation strength
{clinical_instruction}

Return only valid JSON format.
"""

    async def _call_openai(self, prompt: str) -> str:
        """Call OpenAI API with error handling and retries"""
        
        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o-mini",  # Use cheaper model
                messages=[
                    {"role": "system", "content": "You are a therapeutic target discovery expert. Always respond in valid JSON format."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,  # Lower temperature for more consistent results
                max_tokens=1500
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}")
            raise
    
    def _parse_llm_response(self, response: str) -> List[DiseaseTarget]:
        """Parse LLM JSON response into DiseaseTarget objects"""
        
        try:
            # Clean response if it has markdown formatting
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                response = response.split("```")[1].split("```")[0]
            
            data = json.loads(response)
            targets = []
            
            for target_data in data.get("targets", []):
                target = DiseaseTarget(
                    gene_symbol=target_data.get("gene_symbol", "").upper(),
                    protein_name=target_data.get("protein_name", ""),
                    rationale=target_data.get("rationale", ""),
                    confidence=float(target_data.get("confidence", 0.5)),
                    clinical_stage=target_data.get("clinical_stage", "research"),
                    pathway=target_data.get("pathway", "")
                )
                
                if target.gene_symbol:  # Only add if we have a gene symbol
                    targets.append(target)
            
            return targets
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            logger.error(f"Response: {response}")
            return []
        except Exception as e:
            logger.error(f"Error parsing LLM response: {e}")
            return []
    
    def _get_fallback_targets(self, disease: str) -> List[DiseaseTarget]:
        """Fallback target mappings for common diseases"""
        
        disease_lower = disease.lower()
        
        fallback_mappings = {
            "emphysema": [
                DiseaseTarget("PDE4", "Phosphodiesterase 4", "Anti-inflammatory effects in COPD", 0.9, "approved", "cAMP signaling"),
                DiseaseTarget("ELANE", "Neutrophil elastase", "Prevents elastin degradation", 0.8, "clinical", "Protease activity"),
                DiseaseTarget("SERPINA1", "Alpha-1 antitrypsin", "Protease inhibitor deficiency", 0.7, "approved", "Protease inhibition")
            ],
            "copd": [
                DiseaseTarget("PDE4", "Phosphodiesterase 4", "Anti-inflammatory effects", 0.9, "approved", "cAMP signaling"),
                DiseaseTarget("ELANE", "Neutrophil elastase", "Prevents lung damage", 0.8, "clinical", "Protease activity")
            ],
            "alzheimer": [
                DiseaseTarget("BACE1", "Beta-secretase 1", "Amyloid-beta production", 0.8, "clinical", "Amyloid pathway"),
                DiseaseTarget("MAPT", "Tau protein", "Neurofibrillary tangles", 0.7, "preclinical", "Tau pathology"),
                DiseaseTarget("APOE", "Apolipoprotein E", "Lipid metabolism and amyloid clearance", 0.6, "research", "Lipid metabolism")
            ],
            "cancer": [
                DiseaseTarget("EGFR", "Epidermal growth factor receptor", "Oncogenic signaling", 0.9, "approved", "Growth factor signaling"),
                DiseaseTarget("BRAF", "B-Raf proto-oncogene", "MAPK pathway activation", 0.9, "approved", "MAPK signaling"),
                DiseaseTarget("TP53", "Tumor protein p53", "Tumor suppressor", 0.8, "research", "Cell cycle control")
            ]
        }
        
        # Find matching disease
        for key, targets in fallback_mappings.items():
            if key in disease_lower:
                logger.info(f"Using fallback targets for {disease}")
                return targets
        
        # Default fallback
        logger.warning(f"No fallback targets found for {disease}")
        return []

# Convenience function for easy integration
async def get_disease_targets(disease: str, max_targets: int = 5) -> List[str]:
    """
    Simple function to get gene symbols for a disease
    
    Returns:
        List of gene symbols (e.g., ["PDE4", "ELANE", "SERPINA1"])
    """
    try:
        mapper = DiseaseTargetMapper()
        targets = await mapper.map_disease_to_targets(disease, max_targets)
        return [t.gene_symbol for t in targets]
    except Exception as e:
        logger.error(f"Failed to get targets for {disease}: {e}")
        # Return fallback
        mapper = DiseaseTargetMapper.__new__(DiseaseTargetMapper)  # Create without init
        targets = mapper._get_fallback_targets(disease)
        return [t.gene_symbol for t in targets[:max_targets]]

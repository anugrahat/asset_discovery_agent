"""
External database validation for preclinical compounds using PubChem and ChEMBL APIs.
"""
import asyncio
import logging
import aiohttp
from typing import Dict, List, Optional, Tuple
import json

logger = logging.getLogger(__name__)


class CompoundValidator:
    """Validates compounds against external databases (PubChem, ChEMBL)."""
    
    def __init__(self, session: Optional[aiohttp.ClientSession] = None):
        self.session = session
        self._own_session = session is None
        
    async def __aenter__(self):
        if self._own_session:
            self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._own_session and self.session:
            await self.session.close()
    
    async def validate_compound(self, compound_name: str) -> Dict:
        """
        Validate a compound against multiple databases.
        
        Returns:
            Dict with validation results from each database
        """
        logger.info(f"🔍 Validating compound: {compound_name}")
        
        results = {
            'compound_name': compound_name,
            'pubchem_valid': False,
            'chembl_valid': False,
            'pubchem_cid': None,
            'chembl_id': None,
            'molecular_formula': None,
            'molecular_weight': None,
            'validation_score': 0.0
        }
        
        # Run validations in parallel
        pubchem_task = self._validate_pubchem(compound_name)
        chembl_task = self._validate_chembl(compound_name)
        
        pubchem_result, chembl_result = await asyncio.gather(
            pubchem_task, chembl_task, return_exceptions=True
        )
        
        # Process PubChem results
        if not isinstance(pubchem_result, Exception) and pubchem_result:
            results.update(pubchem_result)
            results['pubchem_valid'] = True
            results['validation_score'] += 0.5
            
        # Process ChEMBL results  
        if not isinstance(chembl_result, Exception) and chembl_result:
            results.update(chembl_result)
            results['chembl_valid'] = True
            results['validation_score'] += 0.5
            
        logger.info(f"✅ {compound_name} validation score: {results['validation_score']:.2f}")
        return results
    
    async def _validate_pubchem(self, compound_name: str) -> Optional[Dict]:
        """Validate compound against PubChem database."""
        try:
            # Use alternative PubChem endpoint to avoid DNS issues
            search_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{compound_name}/JSON"
            
            # Add retry logic for DNS failures
            for attempt in range(2):
                try:
                    async with self.session.get(search_url, timeout=15) as response:
                        if response.status == 200:
                            data = await response.json()
                            
                            # Check if data is valid and has compounds
                            if data and isinstance(data, dict):
                                compounds = data.get('PC_Compounds', [])
                                
                                if compounds and len(compounds) > 0:
                                    compound = compounds[0]
                                    
                                    # Validate compound structure
                                    if compound and isinstance(compound, dict):
                                        props = compound.get('props', []) or []
                                        
                                        # Safely extract CID
                                        cid = None
                                        compound_id = compound.get('id')
                                        if compound_id and isinstance(compound_id, dict):
                                            id_data = compound_id.get('id')
                                            if id_data and isinstance(id_data, dict):
                                                cid = id_data.get('cid')
                                        
                                        result = {
                                            'pubchem_cid': cid
                                        }
                                
                                        # Extract molecular properties
                                        for prop in props:
                                            if prop and isinstance(prop, dict):
                                                urn = prop.get('urn', {}) or {}
                                                label = urn.get('label', '') if isinstance(urn, dict) else ''
                                                
                                                if label == 'Molecular Formula':
                                                    value = prop.get('value', {})
                                                    if value and isinstance(value, dict):
                                                        result['molecular_formula'] = value.get('sval')
                                                elif label == 'Molecular Weight':
                                                    value = prop.get('value', {})
                                                    if value and isinstance(value, dict):
                                                        result['molecular_weight'] = value.get('fval')
                                        
                                        logger.info(f"📊 PubChem found: {compound_name} (CID: {result.get('pubchem_cid')})")
                                        return result
                        break  # Exit retry loop if successful
                except Exception as dns_error:
                    if attempt == 0:
                        logger.warning(f"⚠️ PubChem attempt {attempt + 1} failed for {compound_name}: {dns_error}")
                        await asyncio.sleep(1)  # Brief delay before retry
                    else:
                        raise dns_error
                        
        except Exception as e:
            logger.warning(f"⚠️ PubChem validation failed for {compound_name}: {e}")
            
        return None
    
    async def _validate_chembl(self, compound_name: str) -> Optional[Dict]:
        """Validate compound against ChEMBL database."""
        try:
            # Search ChEMBL by name/synonym
            search_url = f"https://www.ebi.ac.uk/chembl/api/data/molecule/search?q={compound_name}&format=json"
            
            async with self.session.get(search_url, timeout=10) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Check if data is valid and has molecules
                    if data and isinstance(data, dict) and 'molecules' in data:
                        molecules = data.get('molecules', [])
                        
                        if molecules and len(molecules) > 0:
                            molecule = molecules[0]
                            
                            # Validate molecule has required fields
                            if molecule and isinstance(molecule, dict):
                                mol_props = molecule.get('molecule_properties')
                                if mol_props is None:
                                    mol_props = {}
                                elif not isinstance(mol_props, dict):
                                    mol_props = {}
                                
                                result = {
                                    'chembl_id': molecule.get('molecule_chembl_id'),
                                    'molecular_formula': mol_props.get('molecular_formula') if mol_props else None,
                                    'molecular_weight': mol_props.get('molecular_weight') if mol_props else None
                                }
                                
                                # Only return if we have a valid ChEMBL ID
                                if result.get('chembl_id'):
                                    logger.info(f"📊 ChEMBL found: {compound_name} (ID: {result.get('chembl_id')})")
                                    return result
                        
        except Exception as e:
            logger.warning(f"⚠️ ChEMBL validation failed for {compound_name}: {e}")
            
        return None
    
    async def validate_batch(self, compounds: List[str]) -> List[Dict]:
        """Validate multiple compounds in parallel."""
        logger.info(f"🔍 Batch validating {len(compounds)} compounds")
        
        tasks = [self.validate_compound(compound) for compound in compounds]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        valid_results = []
        for result in results:
            if not isinstance(result, Exception):
                valid_results.append(result)
            else:
                logger.error(f"❌ Batch validation error: {result}")
                
        return valid_results
    
    def filter_validated_compounds(self, compounds: List[Dict], min_score: float = 0.5) -> List[Dict]:
        """Filter compounds based on validation score."""
        filtered = []
        
        for compound in compounds:
            validation_score = compound.get('validation_score', 0.0)
            if validation_score >= min_score:
                filtered.append(compound)
                logger.info(f"✅ {compound.get('compound_name')} passed validation (score: {validation_score:.2f})")
            else:
                logger.info(f"❌ {compound.get('compound_name')} failed validation (score: {validation_score:.2f})")
                
        return filtered

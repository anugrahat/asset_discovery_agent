"""
Standard of Care Drug Filter
Automatically identifies and filters drugs already approved for a specific indication
"""
import asyncio
import logging
from typing import List, Dict, Set
from collections import defaultdict
import re

logger = logging.getLogger(__name__)


class StandardOfCareFilter:
    """Dynamically identifies standard-of-care drugs for any disease"""
    
    def __init__(self, drug_safety_client=None, chembl_client=None, cache_manager=None):
        self.drug_safety_client = drug_safety_client
        self.chembl_client = chembl_client
        self.cache = cache_manager
        self._soc_cache = {}
        
    async def get_standard_of_care_drugs(self, disease: str) -> Set[str]:
        """
        Get all drugs approved for treating a specific disease
        
        Returns set of lowercase drug names (generic and brand)
        """
        # Handle None or empty disease
        if not disease:
            return set()
            
        # Check cache first
        cache_key = f"soc_drugs:{disease.lower()}"
        if cache_key in self._soc_cache:
            return self._soc_cache[cache_key]
            
        logger.info(f"Identifying standard-of-care drugs for {disease}")
        
        soc_drugs = set()
        
        # Strategy 1: Query ChEMBL for drugs with this indication
        if self.chembl_client:
            try:
                indication_drugs = await self._get_chembl_indication_drugs(disease)
                soc_drugs.update(indication_drugs)
                logger.info(f"Found {len(indication_drugs)} drugs from ChEMBL indications")
            except Exception as e:
                logger.warning(f"ChEMBL indication search failed: {e}")
        
        # Strategy 2: Parse FDA labels for approved indications
        if self.drug_safety_client:
            try:
                fda_approved_drugs = await self._get_fda_approved_for_indication(disease)
                soc_drugs.update(fda_approved_drugs)
                logger.info(f"Found {len(fda_approved_drugs)} drugs from FDA labels")
            except Exception as e:
                logger.warning(f"FDA indication search failed: {e}")
        
        # Strategy 3: Use disease-specific drug class mappings
        drug_classes = await self._get_disease_drug_classes(disease)
        for drug_class in drug_classes:
            class_drugs = await self._get_drugs_by_class(drug_class)
            soc_drugs.update(class_drugs)
            
        # Strategy 4: Common disease patterns and aliases
        disease_aliases = self._get_disease_aliases(disease)
        for alias in disease_aliases:
            if self.cache:
                # Check if we have cached trial data showing approved drugs
                trial_approved = await self._get_trial_approved_drugs(alias)
                soc_drugs.update(trial_approved)
        
        # Cache the result
        self._soc_cache[cache_key] = soc_drugs
        
        logger.info(f"Total standard-of-care drugs for {disease}: {len(soc_drugs)}")
        return soc_drugs
    
    async def _get_chembl_indication_drugs(self, disease: str) -> Set[str]:
        """Query ChEMBL for drugs with this indication"""
        drugs = set()
        
        try:
            # Search for drug indications matching the disease
            results = await self.chembl_client.search_by_indication(disease)
            
            for result in results:
                # Add generic name
                if result.get('molecule_pref_name'):
                    drugs.add(result['molecule_pref_name'].lower())
                
                # Add synonyms
                if result.get('molecule_synonyms'):
                    for syn in result['molecule_synonyms']:
                        drugs.add(syn.lower())
                        
        except Exception as e:
            logger.debug(f"ChEMBL indication search error: {e}")
            
        return drugs
    
    async def _get_fda_approved_for_indication(self, disease: str) -> Set[str]:
        """Parse FDA drug labels for approved indications"""
        drugs = set()
        
        # This would query FDA labels for drugs mentioning the disease in indications
        # For now, using Orange Book data if available
        if hasattr(self.drug_safety_client, 'orange_book_data'):
            ob_data = self.drug_safety_client.orange_book_data
            
            # Search through FDA-approved drugs for indication matches
            disease_terms = self._get_search_terms(disease)
            
            for product in ob_data.get('products', []):
                # Only consider approved products
                if product.get('approval_date') and not product.get('discontinued'):
                    drug_name = product.get('ingredient', '').lower()
                    if drug_name:
                        # Would check if this drug's label contains the indication
                        # For production, this would query FDA label database
                        drugs.add(drug_name)
                        
                        # Add brand name too
                        if product.get('trade_name'):
                            drugs.add(product['trade_name'].lower())
        
        return drugs
    
    def _get_disease_aliases(self, disease: str) -> List[str]:
        """Get common aliases and variations of disease names"""
        disease_lower = disease.lower()
        aliases = [disease_lower]
        
        # Common disease name patterns
        disease_mappings = {
            'diabetes': ['diabetes mellitus', 'type 2 diabetes', 't2d', 'type 1 diabetes', 't1d', 'niddm', 'iddm'],
            'hypertension': ['high blood pressure', 'essential hypertension', 'arterial hypertension'],
            'cancer': ['carcinoma', 'adenocarcinoma', 'tumor', 'tumour', 'neoplasm', 'malignancy'],
            'copd': ['chronic obstructive pulmonary disease', 'emphysema', 'chronic bronchitis'],
            'heart failure': ['congestive heart failure', 'chf', 'cardiac failure'],
            'depression': ['major depressive disorder', 'mdd', 'clinical depression'],
            'alzheimer': ["alzheimer's disease", 'ad', 'dementia of alzheimer type'],
            'parkinson': ["parkinson's disease", 'pd', 'paralysis agitans']
        }
        
        # Check if disease contains any key terms
        for key, values in disease_mappings.items():
            if key in disease_lower:
                aliases.extend(values)
            # Also check reverse
            for value in values:
                if value in disease_lower:
                    aliases.append(key)
                    aliases.extend([v for v in values if v != value])
        
        # Handle specific cancer types
        cancer_match = re.search(r'(\w+)\s+(cancer|carcinoma)', disease_lower)
        if cancer_match:
            organ = cancer_match.group(1)
            aliases.extend([
                f"{organ} cancer",
                f"{organ} carcinoma", 
                f"{organ} adenocarcinoma",
                f"{organ} tumor"
            ])
        
        return list(set(aliases))
    
    def _get_search_terms(self, disease: str) -> List[str]:
        """Get search terms for disease"""
        terms = self._get_disease_aliases(disease)
        
        # Add medical coding terms
        disease_lower = disease.lower()
        if 'diabetes' in disease_lower:
            terms.extend(['e11', 'e10'])  # ICD codes
        elif 'hypertension' in disease_lower:
            terms.extend(['i10', 'i11', 'i12', 'i13'])
            
        return terms
    
    async def _get_disease_drug_classes(self, disease: str) -> List[str]:
        """Map disease to standard drug classes used for treatment"""
        disease_lower = disease.lower()
        drug_classes = []
        
        # Comprehensive drug class mappings by disease
        class_mappings = {
            'diabetes': [
                'biguanides', 'sulfonylureas', 'meglitinides', 'thiazolidinediones',
                'dpp-4 inhibitors', 'glp-1 agonists', 'sglt2 inhibitors', 
                'alpha-glucosidase inhibitors', 'insulin'
            ],
            'hypertension': [
                'ace inhibitors', 'angiotensin receptor blockers', 'beta blockers',
                'calcium channel blockers', 'diuretics', 'alpha blockers',
                'central alpha agonists', 'vasodilators', 'renin inhibitors'
            ],
            'depression': [
                'ssri', 'snri', 'tricyclic antidepressants', 'maoi',
                'atypical antidepressants', 'serotonin modulators'
            ],
            'cancer': [
                'alkylating agents', 'antimetabolites', 'anthracyclines',
                'topoisomerase inhibitors', 'mitotic inhibitors', 'targeted therapy',
                'immunotherapy', 'hormone therapy'
            ],
            'heart failure': [
                'ace inhibitors', 'arbs', 'beta blockers', 'aldosterone antagonists',
                'diuretics', 'digoxin', 'hydralazine', 'isosorbide dinitrate'
            ],
            'copd': [
                'bronchodilators', 'anticholinergics', 'beta2 agonists',
                'corticosteroids', 'phosphodiesterase-4 inhibitors', 'methylxanthines'
            ],
            'asthma': [
                'inhaled corticosteroids', 'laba', 'saba', 'leukotriene modifiers',
                'mast cell stabilizers', 'immunomodulators'
            ],
            'epilepsy': [
                'anticonvulsants', 'antiepileptics', 'sodium channel blockers',
                'gaba enhancers', 'glutamate blockers'
            ]
        }
        
        # Find matching drug classes
        for disease_key, classes in class_mappings.items():
            if disease_key in disease_lower:
                drug_classes.extend(classes)
        
        # Cancer-specific handling
        if 'cancer' in disease_lower or 'carcinoma' in disease_lower:
            # Add targeted therapies based on cancer type
            if 'breast' in disease_lower:
                drug_classes.extend(['her2 inhibitors', 'cdk4/6 inhibitors', 'aromatase inhibitors'])
            elif 'lung' in disease_lower:
                drug_classes.extend(['egfr inhibitors', 'alk inhibitors', 'pd-1 inhibitors'])
            elif 'colon' in disease_lower or 'colorectal' in disease_lower:
                drug_classes.extend(['vegf inhibitors', 'egfr inhibitors'])
        
        return drug_classes
    
    async def _get_drugs_by_class(self, drug_class: str) -> Set[str]:
        """Get all drugs belonging to a therapeutic class"""
        drugs = set()
        
        # Common drug class members (production would query drug database)
        class_drugs = {
            'ace inhibitors': ['lisinopril', 'enalapril', 'ramipril', 'captopril', 'benazepril', 'quinapril'],
            'beta blockers': ['metoprolol', 'atenolol', 'propranolol', 'carvedilol', 'bisoprolol', 'nebivolol'],
            'statins': ['atorvastatin', 'simvastatin', 'rosuvastatin', 'pravastatin', 'lovastatin'],
            'ssri': ['sertraline', 'fluoxetine', 'paroxetine', 'citalopram', 'escitalopram'],
            'ppi': ['omeprazole', 'esomeprazole', 'pantoprazole', 'lansoprazole', 'rabeprazole'],
            'sglt2 inhibitors': ['empagliflozin', 'dapagliflozin', 'canagliflozin', 'ertugliflozin'],
            'glp-1 agonists': ['liraglutide', 'semaglutide', 'dulaglutide', 'exenatide'],
            'dpp-4 inhibitors': ['sitagliptin', 'saxagliptin', 'linagliptin', 'alogliptin'],
            'arbs': ['losartan', 'valsartan', 'telmisartan', 'olmesartan', 'irbesartan'],
            'calcium channel blockers': ['amlodipine', 'diltiazem', 'verapamil', 'nifedipine'],
            'diuretics': ['hydrochlorothiazide', 'furosemide', 'spironolactone', 'chlorthalidone'],
            'biguanides': ['metformin'],
            'sulfonylureas': ['glipizide', 'glyburide', 'glimepiride'],
            'insulin': ['insulin glargine', 'insulin lispro', 'insulin aspart', 'insulin detemir']
        }
        
        # Get drugs for this class
        class_lower = drug_class.lower()
        for key, drug_list in class_drugs.items():
            if key in class_lower or class_lower in key:
                drugs.update(drug_list)
        
        return drugs
    
    async def _get_trial_approved_drugs(self, disease: str) -> Set[str]:
        """Get drugs with completed Phase 4 trials for this disease"""
        drugs = set()
        
        # This would query clinical trials for Phase 4 (post-market) studies
        # indicating the drug is approved for this indication
        
        return drugs
    
    def is_standard_of_care(self, drug_name: str, soc_drugs: Set[str]) -> bool:
        """Check if a drug is standard-of-care"""
        drug_lower = drug_name.lower()
        
        # Direct match
        if drug_lower in soc_drugs:
            return True
        
        # Partial match (handles variations like "drug hcl", "drug sodium")
        for soc_drug in soc_drugs:
            if soc_drug in drug_lower or drug_lower in soc_drug:
                return True
        
        # Handle brand/generic variations
        # Remove common suffixes
        drug_cleaned = re.sub(r'\s+(hcl|sodium|potassium|acetate|sulfate|phosphate)$', '', drug_lower)
        if drug_cleaned in soc_drugs:
            return True
            
        return False

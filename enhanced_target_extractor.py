"""
Enhanced target extraction from multiple sources when LLM fails
"""
import re
from typing import Dict, List, Optional

class EnhancedTargetExtractor:
    """Extract drug targets from clinical trials, literature, and patterns"""
    
    def __init__(self):
        # Common checkpoint inhibitor patterns
        self.target_patterns = {
            'PD-1': [
                r'PD-1 blockade', r'anti-PD-1', r'PD-1 inhibitor', 
                r'programmed death-1', r'programmed cell death protein-1'
            ],
            'PD-L1': [
                r'PD-L1 blockade', r'anti-PD-L1', r'PD-L1 inhibitor',
                r'programmed death-ligand 1'
            ],
            'CTLA-4': [
                r'CTLA-4 blockade', r'anti-CTLA-4', r'CTLA-4 inhibitor',
                r'cytotoxic T-lymphocyte antigen 4'
            ],
            'TIGIT': [
                r'TIGIT blockade', r'anti-TIGIT', r'TIGIT inhibitor'
            ]
        }
        
        # Drug name suffix patterns
        self.suffix_patterns = {
            '-limab': 'monoclonal antibody',
            '-mab': 'monoclonal antibody', 
            '-tinib': 'tyrosine kinase inhibitor',
            '-ciclib': 'CDK4/6 inhibitor'
        }
    
    def extract_from_clinical_trials(self, trials: List[Dict]) -> Optional[str]:
        """Extract target from clinical trial data"""
        all_text = ""
        
        for trial in trials:
            # Collect text from title, interventions, descriptions
            title = trial.get('title', '')
            all_text += f" {title}"
            
            interventions = trial.get('interventions', [])
            for intervention in interventions:
                desc = intervention.get('description', '')
                all_text += f" {desc}"
        
        return self._extract_target_from_text(all_text)
    
    def extract_from_literature(self, articles: List[Dict]) -> Optional[str]:
        """Extract target from literature abstracts and titles"""
        all_text = ""
        
        for article in articles:
            title = article.get('title', '')
            abstract = article.get('abstract', '')
            all_text += f" {title} {abstract}"
        
        return self._extract_target_from_text(all_text)
    
    def extract_from_drug_name(self, drug_name: str) -> Optional[Dict]:
        """Extract info from drug name patterns"""
        drug_lower = drug_name.lower()
        
        # Check suffix patterns
        for suffix, drug_type in self.suffix_patterns.items():
            if drug_lower.endswith(suffix):
                return {
                    'drug_type': drug_type,
                    'confidence': 'high' if suffix == '-limab' else 'medium'
                }
        
        return None
    
    def _extract_target_from_text(self, text: str) -> Optional[str]:
        """Extract target from text using pattern matching"""
        text_lower = text.lower()
        
        # Score each target based on pattern matches
        target_scores = {}
        
        for target, patterns in self.target_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, text_lower, re.IGNORECASE))
                score += matches
            
            if score > 0:
                target_scores[target] = score
        
        # Return highest scoring target
        if target_scores:
            best_target = max(target_scores.items(), key=lambda x: x[1])
            return best_target[0]
        
        return None
    
    def get_comprehensive_target_info(self, drug_name: str, trials: List[Dict] = None, 
                                    articles: List[Dict] = None) -> Dict:
        """Get target info from all available sources"""
        result = {
            'target': None,
            'mechanism': None,
            'confidence': 'low',
            'sources': []
        }
        
        # 1. Try clinical trials
        if trials:
            target = self.extract_from_clinical_trials(trials)
            if target:
                result['target'] = target
                result['confidence'] = 'high'
                result['sources'].append('clinical_trials')
        
        # 2. Try literature if no target from trials
        if not result['target'] and articles:
            target = self.extract_from_literature(articles)
            if target:
                result['target'] = target
                result['confidence'] = 'medium'
                result['sources'].append('literature')
        
        # 3. Try drug name patterns
        name_info = self.extract_from_drug_name(drug_name)
        if name_info:
            if not result['target'] and name_info['drug_type'] == 'monoclonal antibody':
                # For mAbs, try to infer target from context
                result['mechanism'] = name_info['drug_type']
                result['sources'].append('drug_name_pattern')
            elif result['target']:
                result['mechanism'] = f"{result['target']} {name_info['drug_type']}"
        
        return result

# Test the extractor
if __name__ == "__main__":
    extractor = EnhancedTargetExtractor()
    
    # Test with Zimberelimab data
    mock_trials = [
        {
            'title': 'Dual TIGIT and PD-1 blockade with domvanalimab plus zimberelimab',
            'interventions': [
                {'description': 'PD-1 inhibitor administered intravenously'}
            ]
        }
    ]
    
    result = extractor.get_comprehensive_target_info('Zimberelimab', trials=mock_trials)
    print(f"Zimberelimab target info: {result}")

"""
Purple Book CSV parser for FDA biologic drug data
"""
import csv
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path

class PurpleBookParser:
    """Parser for FDA Purple Book CSV data files"""
    
    def __init__(self, csv_path: str = None):
        self.csv_path = csv_path
        self.biologics_data = []
        self.biosimilar_map = {}
        self.reference_products = {}
        
    def parse_csv(self, csv_path: str = None) -> List[Dict[str, Any]]:
        """
        Parse Purple Book CSV file and extract biologic data
        
        Args:
            csv_path: Path to Purple Book CSV file
            
        Returns:
            List of parsed biologic products
        """
        if csv_path:
            self.csv_path = csv_path
            
        if not self.csv_path:
            raise ValueError("No CSV path provided")
            
        biologics = []
        with open(self.csv_path, 'r', encoding='utf-8-sig') as f:
            # Skip the first few header rows and read from row 5 (0-indexed row 4)
            reader = csv.reader(f)
            for i in range(4):  # Skip first 4 rows
                next(reader, None)
            
            # Row 5 should have the actual headers
            headers = next(reader, None)
            if not headers:
                return biologics
                
            # Create DictReader with the actual data
            reader = csv.DictReader(f, fieldnames=headers)
            
            for row in reader:
                # Skip empty rows - use any field that should have data
                if not any(row.values()) or not row:
                    continue
                    
                # Parse biologic data - use actual column positions from CSV
                # Based on the CSV structure we found: brand names in column 3, generic names in column 4
                row_values = list(row.values()) if hasattr(row, 'values') else []
                if len(row_values) < 5:
                    continue
                    
                biologic = {
                    'update_type': row_values[0] if len(row_values) > 0 else '',
                    'applicant': row_values[1] if len(row_values) > 1 else '',
                    'bla_number': row_values[2] if len(row_values) > 2 else '',
                    'proprietary_name': row_values[3] if len(row_values) > 3 else '',  # Brand names like "Tecentriq"
                    'proper_name': row_values[4] if len(row_values) > 4 else '',       # Generic names like "atezolizumab"
                    'bla_type': row_values[5] if len(row_values) > 5 else '',
                    'strength': row_values[6] if len(row_values) > 6 else '',
                    'dosage_form': row_values[7] if len(row_values) > 7 else '',
                    'route': row_values[8] if len(row_values) > 8 else '',
                    'product_presentation': row_values[9] if len(row_values) > 9 else '',  # "Single-Dose Vial"
                    'marketing_status': row_values[10] if len(row_values) > 10 else '',   # "Rx", "Disc", etc.
                    'licensure': '',
                    'approval_date': '',
                    'ref_product_proper_name': '',
                    'ref_product_proprietary_name': '',
                    'center': '',
                    'exclusivity_exp_date': '',
                    'orphan_exclusivity_exp': ''
                }
                
                # Determine product type
                if '351(k)' in biologic['bla_type']:
                    if 'Interchangeable' in biologic['bla_type']:
                        biologic['product_type'] = 'interchangeable'
                    else:
                        biologic['product_type'] = 'biosimilar'
                else:
                    biologic['product_type'] = 'reference'
                
                # Determine if product is active
                biologic['is_active'] = biologic['marketing_status'].upper() in ['RX', 'LICENSED', 'OTC']
                
                biologics.append(biologic)
                
                # Build biosimilar mapping
                if biologic['product_type'] in ['biosimilar', 'interchangeable'] and biologic['ref_product_proper_name']:
                    ref_name = biologic['ref_product_proper_name'].lower()
                    if ref_name not in self.biosimilar_map:
                        self.biosimilar_map[ref_name] = []
                    self.biosimilar_map[ref_name].append(biologic)
                    
                # Track reference products
                if biologic['product_type'] == 'reference':
                    self.reference_products[biologic['proper_name'].lower()] = biologic
        
        self.biologics_data = biologics
        return biologics
    
    def find_biologic(self, drug_name: str) -> Optional[Dict[str, Any]]:
        """
        Find a biologic by proprietary or proper name
        
        Args:
            drug_name: Drug name to search for
            
        Returns:
            Biologic data if found, None otherwise
        """
        drug_name_lower = drug_name.lower().strip()
        
        for biologic in self.biologics_data:
            if (drug_name_lower in biologic['proprietary_name'].lower() or 
                drug_name_lower in biologic['proper_name'].lower()):
                return biologic
                
        return None
    
    def get_biosimilars(self, reference_drug: str) -> List[Dict[str, Any]]:
        """
        Get all biosimilars/interchangeables for a reference product
        
        Args:
            reference_drug: Name of reference biologic
            
        Returns:
            List of biosimilar/interchangeable products
        """
        drug_name_lower = reference_drug.lower().strip()
        return self.biosimilar_map.get(drug_name_lower, [])
    
    def get_all_active_biologics(self) -> List[Dict[str, Any]]:
        """Get all currently active/marketed biologics"""
        return [b for b in self.biologics_data if b['is_active']]
    
    def get_discontinued_biologics(self) -> List[Dict[str, Any]]:
        """Get all discontinued/withdrawn biologics"""
        return [b for b in self.biologics_data 
                if b['marketing_status'].upper() in ['DISC', 'VOLUNTARILY REVOKED', 'WITHDRAWN']]
    
    def get_biologics_by_target_area(self, keywords: List[str]) -> List[Dict[str, Any]]:
        """
        Find biologics related to specific therapeutic areas
        
        Args:
            keywords: List of keywords to search in drug names
            
        Returns:
            List of matching biologics
        """
        matches = []
        for biologic in self.biologics_data:
            name_combined = f"{biologic['proprietary_name']} {biologic['proper_name']}".lower()
            if any(keyword.lower() in name_combined for keyword in keywords):
                matches.append(biologic)
        return matches
    
    def get_patent_expiry_candidates(self, months_ahead: int = 12) -> List[Dict[str, Any]]:
        """
        Find biologics with patents expiring soon
        
        Args:
            months_ahead: Number of months to look ahead
            
        Returns:
            List of biologics with upcoming patent expiry
        """
        from datetime import datetime, timedelta
        
        candidates = []
        cutoff_date = datetime.now() + timedelta(days=months_ahead * 30)
        
        for biologic in self.biologics_data:
            expiry_str = biologic.get('exclusivity_exp_date', '')
            if expiry_str:
                try:
                    # Parse various date formats
                    if ',' in expiry_str:
                        expiry_date = datetime.strptime(expiry_str, "%B %d, %Y")
                    else:
                        expiry_date = datetime.strptime(expiry_str, "%B %d %Y")
                        
                    if datetime.now() < expiry_date <= cutoff_date:
                        biologic['expiry_date'] = expiry_date
                        candidates.append(biologic)
                except:
                    pass
                    
        return candidates

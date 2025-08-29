#!/usr/bin/env python3

import os
import csv
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class OrangeBookParser:
    """Parser for FDA Orange Book data files to get accurate approval and discontinuation status"""
    
    def __init__(self, orange_book_dir: str = "/home/anugraha/agent3/orangebook"):
        self.orange_book_dir = orange_book_dir
        self.products_file = os.path.join(orange_book_dir, "products.txt")
        self.patent_file = os.path.join(orange_book_dir, "patent.txt")
        self.exclusivity_file = os.path.join(orange_book_dir, "exclusivity.txt")
        
        # Load data on initialization
        self.products_data = self._load_products()
        self.patent_data = self._load_patents()
        
    def _load_products(self) -> Dict[str, List[Dict]]:
        """Load products.txt into a searchable structure"""
        products_by_ingredient = {}
        products_by_trade = {}
        
        try:
            with open(self.products_file, 'r', encoding='utf-8', errors='ignore') as f:
                # Skip header line
                next(f)
                
                for line in f:
                    fields = line.strip().split('~')
                    if len(fields) >= 13:
                        # Extract Federal Register note if present in strength field
                        strength_field = fields[4]
                        federal_register_note = None
                        actual_strength = strength_field
                        
                        if '**Federal Register' in strength_field:
                            # Split the strength from the Federal Register note
                            parts = strength_field.split(' **')
                            actual_strength = parts[0].strip()
                            if len(parts) > 1:
                                federal_register_note = parts[1].rstrip('**').strip()
                        
                        product = {
                            'ingredient': fields[0],  # Keep original case for display
                            'ingredient_lower': fields[0].lower(),  # Lowercase for searching
                            'dosage_form_route': fields[1],
                            'trade_name': fields[2],
                            'trade_name_lower': fields[2].lower(),  # Lowercase for searching
                            'applicant': fields[3],
                            'strength': actual_strength,
                            'federal_register_note': federal_register_note,
                            'type': fields[5],  # N=NDA, A=ANDA
                            'nda_number': fields[6],
                            'product_number': fields[7],
                            'te_code': fields[8],
                            'approval_date': fields[9],
                            'rld': fields[10],  # Reference Listed Drug
                            'rs': fields[11],   # Reference Standard
                            'market_status': fields[12],  # RX, OTC, DISCN
                            'applicant_full_name': fields[13] if len(fields) > 13 else ''
                        }
                        
                        # Index by ingredient (lowercase for searching)
                        ingredient_key = product['ingredient_lower']
                        if ingredient_key not in products_by_ingredient:
                            products_by_ingredient[ingredient_key] = []
                        products_by_ingredient[ingredient_key].append(product)
                        
                        # Index by trade name (lowercase for searching)
                        trade_key = product['trade_name_lower']
                        if trade_key not in products_by_trade:
                            products_by_trade[trade_key] = []
                        products_by_trade[trade_key].append(product)
                        
        except Exception as e:
            logger.error(f"Error loading Orange Book products: {e}")
            
        return {'by_ingredient': products_by_ingredient, 'by_trade': products_by_trade}
    
    def _load_patents(self) -> Dict[str, List[Dict]]:
        """Load patent.txt data"""
        patents_by_nda = {}
        
        try:
            with open(self.patent_file, 'r', encoding='utf-8', errors='ignore') as f:
                # Skip header
                next(f)
                
                for line in f:
                    fields = line.strip().split('~')
                    if len(fields) >= 5:
                        patent = {
                            'type': fields[0],
                            'nda_number': fields[1],
                            'product_number': fields[2],
                            'patent_number': fields[3],
                            'patent_expire_date': fields[4]
                        }
                        
                        nda_key = patent['nda_number']
                        if nda_key not in patents_by_nda:
                            patents_by_nda[nda_key] = []
                        patents_by_nda[nda_key].append(patent)
                        
        except Exception as e:
            logger.error(f"Error loading Orange Book patents: {e}")
            
        return patents_by_nda
    
    def get_drug_status(self, drug_name: str) -> Dict[str, Any]:
        """Get comprehensive drug status from Orange Book"""
        
        drug_lower = drug_name.lower()
        results = {
            'found': False,
            'original_approval_date': None,
            'original_nda': None,
            'original_brand': None,
            'current_status': 'Unknown',
            'active_products': [],
            'discontinued_products': [],
            'generic_available': False,
            'all_products': []
        }
        
        # Generate drug name variations with salt form mappings
        drug_variations = self._generate_drug_name_variations(drug_lower)
        
        # Search by ingredient and trade name for all variations
        all_matches = []
        for variation in drug_variations:
            ingredient_matches = self.products_data['by_ingredient'].get(variation, [])
            all_matches.extend(ingredient_matches)
            
            # Also search trade names
            for trade_key, products in self.products_data['by_trade'].items():
                if variation in trade_key or trade_key in variation:
                    all_matches.extend(products)
        
        # Remove duplicates
        seen_products = set()
        unique_matches = []
        for product in all_matches:
            product_id = (product['nda_number'], product['product_number'])
            if product_id not in seen_products:
                seen_products.add(product_id)
                unique_matches.append(product)
        
        all_matches = unique_matches
        
        # No hardcoded mappings - rely on LLM for drug name resolution
        
        if all_matches:
            results['found'] = True
            
            # Find original NDA (type='N') and earliest approval
            nda_products = [p for p in all_matches if p['type'] == 'N']
            if nda_products:
                # Sort by approval date to find original
                sorted_ndas = sorted(nda_products, key=lambda x: self._parse_approval_date(x['approval_date']))
                original = sorted_ndas[0]
                
                results['original_nda'] = original['nda_number']
                results['original_brand'] = original['trade_name']
                results['original_approval_date'] = original['approval_date']
            
            # Check current market status
            active_count = 0
            discontinued_count = 0
            
            for product in all_matches:
                if product['market_status'] == 'DISCN':
                    discontinued_count += 1
                    results['discontinued_products'].append({
                        'name': product['trade_name'],
                        'type': 'Brand' if product['type'] == 'N' else 'Generic',
                        'applicant': product['applicant_full_name'] or product['applicant'],
                        'federal_register_note': product.get('federal_register_note')
                    })
                else:
                    active_count += 1
                    results['active_products'].append({
                        'name': product['trade_name'],
                        'type': 'Brand' if product['type'] == 'N' else 'Generic',
                        'applicant': product['applicant_full_name'] or product['applicant'],
                        'status': product['market_status']
                    })
                
                # Check for generics
                if product['type'] == 'A' and product['market_status'] != 'DISCN':
                    results['generic_available'] = True
            
            # Determine overall status
            if active_count > 0:
                if results['generic_available']:
                    results['current_status'] = 'Active (generics available)'
                else:
                    results['current_status'] = 'Active (brand only)'
            elif discontinued_count > 0 and active_count == 0:
                results['current_status'] = 'Discontinued'
            
            # Store all products for reference
            results['all_products'] = all_matches[:10]  # Limit to first 10
            
        return results
    
    def _generate_drug_name_variations(self, drug_name: str) -> List[str]:
        """Generate drug name variations to handle salt form mappings"""
        variations = [drug_name]  # Start with original
        
        # Salt form mappings
        salt_mappings = {
            'hcl': 'hydrochloride',
            'hydrochloride': 'hcl',
            'hbr': 'hydrobromide', 
            'hydrobromide': 'hbr',
            'sulfate': 'sulphate',
            'sulphate': 'sulfate',
            'phosphate': 'po4',
            'po4': 'phosphate',
            'acetate': 'ch3coo',
            'ch3coo': 'acetate',
            'tartrate': 'bitartrate',
            'bitartrate': 'tartrate',
            'maleate': 'mal',
            'mal': 'maleate',
            'succinate': 'succ',
            'succ': 'succinate',
            'fumarate': 'fum',
            'fum': 'fumarate'
        }
        
        # Apply salt form mappings
        for salt_short, salt_long in salt_mappings.items():
            if salt_short in drug_name:
                # Replace short form with long form
                variation = drug_name.replace(salt_short, salt_long)
                variations.append(variation)
                
                # Also try with spaces
                variation_spaced = drug_name.replace(salt_short, f' {salt_long}')
                variations.append(variation_spaced)
                
                # Try without the salt entirely
                base_name = drug_name.replace(salt_short, '').strip()
                if base_name and base_name != drug_name:
                    variations.append(base_name)
        
        # Remove duplicates and empty strings
        variations = list(set([v.strip() for v in variations if v.strip()]))
        
        return variations
    
    def _parse_approval_date(self, date_str: str) -> datetime:
        """Parse Orange Book date format"""
        if "prior to Jan 1, 1982" in date_str:
            return datetime(1982, 1, 1)
        
        try:
            # Format is "Mmm dd, yyyy" e.g., "Dec 29, 1992"
            return datetime.strptime(date_str.strip(), "%b %d, %Y")
        except:
            try:
                # Alternative format "Mmm d, yyyy"
                return datetime.strptime(date_str.strip(), "%b %d, %Y")
            except:
                return datetime(2000, 1, 1)  # Default fallback

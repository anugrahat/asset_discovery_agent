"""
FDA Approvals Database Validator

Cross-validates drug approval status against multiple FDA data sources
to ensure accuracy and consistency.
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import asyncio

logger = logging.getLogger(__name__)

class FDAApprovalsValidator:
    """Validates drug approval status across multiple FDA databases"""
    
    def __init__(self, drug_safety_client=None):
        """Initialize with optional drug safety client"""
        self.drug_safety_client = drug_safety_client
    
    async def validate_approval_status(self, drug_name: str) -> Dict[str, Any]:
        """
        Validate drug approval status across multiple sources:
        1. Orange Book (NMEs and generics)
        2. Purple Book (biologics)
        3. Drugs@FDA
        4. FDA drug labels
        
        Returns consolidated validation report
        """
        validation_report = {
            "drug_name": drug_name,
            "validation_timestamp": datetime.now().isoformat(),
            "sources_checked": [],
            "approval_status": "Not Approved",
            "confidence_score": 0.0,
            "discrepancies": [],
            "consolidated_data": {}
        }
        
        # Import here to avoid circular imports
        if not self.drug_safety_client:
            from .drug_safety_client import DrugSafetyClient
            self.drug_safety_client = DrugSafetyClient()
        
        # Check Orange Book
        try:
            orange_book_data = await self._check_orange_book(drug_name)
            validation_report["sources_checked"].append("Orange Book")
            if orange_book_data.get("is_approved"):
                validation_report["consolidated_data"]["orange_book"] = orange_book_data
        except Exception as e:
            logger.error(f"Orange Book check failed: {e}")
        
        # Check Purple Book for biologics
        try:
            purple_book_data = await self._check_purple_book(drug_name)
            validation_report["sources_checked"].append("Purple Book")
            if purple_book_data.get("is_biologic"):
                validation_report["consolidated_data"]["purple_book"] = purple_book_data
        except Exception as e:
            logger.error(f"Purple Book check failed: {e}")
        
        # Check Drugs@FDA
        try:
            drugsfda_data = await self._check_drugsfda(drug_name)
            validation_report["sources_checked"].append("Drugs@FDA")
            if drugsfda_data.get("has_applications"):
                validation_report["consolidated_data"]["drugsfda"] = drugsfda_data
        except Exception as e:
            logger.error(f"Drugs@FDA check failed: {e}")
        
        # Check FDA drug labels
        try:
            label_data = await self._check_fda_labels(drug_name)
            validation_report["sources_checked"].append("FDA Labels")
            if label_data.get("has_label"):
                validation_report["consolidated_data"]["fda_labels"] = label_data
        except Exception as e:
            logger.error(f"FDA Labels check failed: {e}")
        
        # Analyze results for consistency
        approval_signals = self._analyze_approval_signals(validation_report["consolidated_data"])
        
        # Set final approval status based on evidence
        if approval_signals["strong_approval_evidence"]:
            validation_report["approval_status"] = "FDA Approved"
            validation_report["confidence_score"] = approval_signals["confidence"]
        elif approval_signals["any_approval_evidence"]:
            # If Orange Book shows approved with date, that's definitive for small molecules
            if validation_report["consolidated_data"].get("orange_book", {}).get("is_approved") and \
               validation_report["consolidated_data"].get("orange_book", {}).get("approval_date"):
                validation_report["approval_status"] = "FDA Approved"
                validation_report["confidence_score"] = 1.0
            # If Purple Book shows approved biologic, that's definitive
            elif validation_report["consolidated_data"].get("purple_book", {}).get("is_approved"):
                validation_report["approval_status"] = "FDA Approved (Biologic)"
                validation_report["confidence_score"] = 1.0
            else:
                validation_report["approval_status"] = "Likely Approved"
                validation_report["confidence_score"] = approval_signals["confidence"]
        else:
            validation_report["approval_status"] = "Not FDA Approved"
            validation_report["confidence_score"] = 0.95  # High confidence in negative
        
        # Check for discrepancies
        validation_report["discrepancies"] = self._find_discrepancies(validation_report["consolidated_data"])
        
        return validation_report
    
    async def _check_orange_book(self, drug_name: str) -> Dict[str, Any]:
        """Check Orange Book for approval status"""
        # Use existing method from drug safety client
        fda_data = await self.drug_safety_client._get_fda_approval_status(drug_name)
        
        # Check if drug is approved (main field)
        if fda_data.get("is_approved"):
            result = {
                "is_approved": True,
                "approval_date": fda_data.get("original_approval_date")
            }
            
            # Add Orange Book specific data if available
            if fda_data.get("orange_book_data"):
                ob_data = fda_data["orange_book_data"]
                result.update({
                    "active_products": len(ob_data.get("active_products", [])),
                    "discontinued_products": len(ob_data.get("discontinued_products", [])),
                    "generic_available": ob_data.get("generic_available", False),
                    "original_brand": ob_data.get("original_brand_name")
                })
                
            return result
        
        return {"is_approved": False}
    
    async def _check_purple_book(self, drug_name: str) -> Dict[str, Any]:
        """Check Purple Book for biologic status"""
        import os
        from .purple_book_parser import PurpleBookParser
        
        # Initialize Purple Book parser
        purple_book_path = "/home/anugraha/agent3/orangebook/purplebook-search-july-data-download.csv"
        if not os.path.exists(purple_book_path):
            logger.debug(f"Purple Book CSV not found at {purple_book_path}")
            return {"is_biologic": False}
        
        try:
            parser = PurpleBookParser(purple_book_path)
            parser.parse_csv()
            
            # Find biologic by name
            biologic = parser.find_biologic(drug_name)
            
            if biologic:
                # Get all products for this biologic
                active_products = []
                biosimilars = []
                
                for prod in parser.biologics_data:
                    if (prod.get('proper_name', '').lower() == drug_name.lower() or
                        prod.get('proprietary_name', '').lower() == drug_name.lower()):
                        
                        if prod.get('is_active'):
                            active_products.append({
                                'proprietary_name': prod.get('proprietary_name'),
                                'license_number': prod.get('bla_number'),
                                'approval_date': prod.get('approval_date')
                            })
                        
                        if prod.get('product_type') in ['biosimilar', 'interchangeable']:
                            biosimilars.append(prod.get('proprietary_name'))
                
                return {
                    "is_biologic": True,
                    "is_approved": bool(active_products),
                    "license_number": biologic.get("bla_number"),
                    "approval_date": biologic.get("approval_date"),
                    "active_products": active_products,
                    "biosimilars": biosimilars,
                    "marketing_status": biologic.get("marketing_status")
                }
        
        except Exception as e:
            logger.error(f"Purple Book parsing failed: {e}")
        
        return {"is_biologic": False}
    
    async def _check_drugsfda(self, drug_name: str) -> Dict[str, Any]:
        """Check Drugs@FDA database"""
        fda_data = await self.drug_safety_client._get_fda_approval_status(drug_name)
        
        if fda_data.get("marketing_status") in ["Marketed", "Active"]:
            return {
                "has_applications": True,
                "marketing_status": fda_data.get("marketing_status"),
                "approval_date": fda_data.get("original_approval_date"),
                "active_ingredients": fda_data.get("active_ingredients", [])
            }
        
        return {"has_applications": False}
    
    async def _check_fda_labels(self, drug_name: str) -> Dict[str, Any]:
        """Check for FDA drug label existence"""
        label_data = await self.drug_safety_client._get_fda_drug_labels(drug_name)
        
        if label_data and label_data.get("brand_names"):
            return {
                "has_label": True,
                "brand_names": label_data.get("brand_names", []),
                "generic_names": label_data.get("generic_name", []),
                "has_warnings": bool(label_data.get("warnings")),
                "has_contraindications": bool(label_data.get("contraindications"))
            }
        
        return {"has_label": False}
    
    def _analyze_approval_signals(self, consolidated_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze approval signals from all sources"""
        approval_count = 0
        total_sources = 0
        
        # Orange Book approval
        if consolidated_data.get("orange_book", {}).get("is_approved"):
            approval_count += 1
        if "orange_book" in consolidated_data:
            total_sources += 1
        
        # Purple Book approval (biologics)
        if consolidated_data.get("purple_book", {}).get("is_approved"):
            approval_count += 1
        if "purple_book" in consolidated_data:
            total_sources += 1
        
        # Drugs@FDA active status
        if consolidated_data.get("drugsfda", {}).get("has_applications"):
            approval_count += 1
        if "drugsfda" in consolidated_data:
            total_sources += 1
        
        # FDA label existence
        if consolidated_data.get("fda_labels", {}).get("has_label"):
            approval_count += 0.5  # Weaker signal
        if "fda_labels" in consolidated_data:
            total_sources += 0.5
        
        confidence = approval_count / max(total_sources, 1)
        
        return {
            "strong_approval_evidence": approval_count >= 2,
            "any_approval_evidence": approval_count >= 1,
            "confidence": min(confidence, 1.0)
        }
    
    def _find_discrepancies(self, consolidated_data: Dict[str, Any]) -> List[str]:
        """Find discrepancies between data sources"""
        discrepancies = []
        
        # Check if Orange Book and Purple Book disagree
        ob_approved = consolidated_data.get("orange_book", {}).get("is_approved", False)
        pb_approved = consolidated_data.get("purple_book", {}).get("is_approved", False)
        drugsfda_active = consolidated_data.get("drugsfda", {}).get("has_applications", False)
        
        # Check for approval date discrepancies
        dates = []
        if consolidated_data.get("orange_book", {}).get("approval_date"):
            dates.append(("Orange Book", consolidated_data["orange_book"]["approval_date"]))
        if consolidated_data.get("purple_book", {}).get("approval_date"):
            dates.append(("Purple Book", consolidated_data["purple_book"]["approval_date"]))
        if consolidated_data.get("drugsfda", {}).get("approval_date"):
            dates.append(("Drugs@FDA", consolidated_data["drugsfda"]["approval_date"]))
        
        if len(dates) > 1:
            # Check if dates differ significantly
            date_values = [d[1] for d in dates if d[1]]
            if len(set(date_values)) > 1:
                discrepancies.append(f"Approval dates differ: {dates}")
        
        # Check marketing status consistency
        if ob_approved and not drugsfda_active:
            discrepancies.append("Orange Book shows approved but Drugs@FDA shows inactive")
        
        return discrepancies
    
    async def batch_validate(self, drug_names: List[str]) -> Dict[str, Dict[str, Any]]:
        """Validate multiple drugs in batch"""
        tasks = [self.validate_approval_status(drug) for drug in drug_names]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        validation_results = {}
        for drug_name, result in zip(drug_names, results):
            if isinstance(result, Exception):
                validation_results[drug_name] = {
                    "error": str(result),
                    "approval_status": "Unknown"
                }
            else:
                validation_results[drug_name] = result
        
        return validation_results

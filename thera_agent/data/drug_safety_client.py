#!/usr/bin/env python3

import asyncio
import json
import logging
import os
from typing import Dict, List, Optional, Any
from .http_client import RateLimitedClient
from .cache import APICache
from ..llm_client import LLMClient
from .orange_book_parser import OrangeBookParser

logger = logging.getLogger(__name__)

class DrugSafetyClient:
    """Comprehensive drug safety and pharmacological data client"""
    
    def __init__(self):
        self.http = RateLimitedClient()
        self.cache = APICache()
        
        # Initialize OpenAI client for LLM-powered drug name resolution
        self.openai_client = None
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            try:
                self.openai_client = LLMClient()
            except Exception as e:
                logger.warning(f"OpenAI client init failed; LLM features disabled: {e}")
        else:
            logger.info("OPENAI_API_KEY not set; LLM features disabled for DrugSafetyClient")
        
        # Initialize Orange Book parser for accurate regulatory status
        self.orange_book = OrangeBookParser()
        
        # API endpoints
        self.openfda_base = "https://api.fda.gov"
        self.dailymed_base = "https://dailymed.nlm.nih.gov/dailymed"
        self.rxnav_base = "https://rxnav.nlm.nih.gov/REST"
        self.drugbank_base = "https://go.drugbank.com/public_api/v1"  # Requires API key
    
    async def get_comprehensive_safety_profile(self, drug_name: str, chembl_id: str = None) -> Dict[str, Any]:
        """Get comprehensive safety profile from multiple sources"""
        
        profile = {
            "drug_name": drug_name,
            "chembl_id": chembl_id,
            "fda_approval_status": await self._get_fda_approval_status(drug_name),
            "fda_adverse_events": await self._get_clinical_trial_toxicity(drug_name),
            "fda_drug_labels": await self._get_fda_drug_labels(drug_name),
            "contraindications": await self._get_contraindications(drug_name),
            "drug_interactions": await self._get_drug_interactions(drug_name),
            "black_box_warnings": await self._get_black_box_warnings(drug_name),
            "pharmacology": await self._get_pharmacology_data(drug_name),
            "allergies_cross_sensitivity": await self._get_allergy_data(drug_name)
        }
        
        return profile
    
    def _get_drug_name_variations(self, drug_name: str) -> List[str]:
        """Get basic drug name variations (fallback method)"""
        variations = [drug_name]
        
        # Handle common variations
        if drug_name.lower() == "nab-paclitaxel":
            variations.extend(["paclitaxel protein-bound", "abraxane", "paclitaxel", 
                             "paclitaxel protein bound", "nanoparticle albumin-bound paclitaxel"])
        elif drug_name.lower() == "oxaliplatin":
            variations.extend(["eloxatin", "oxaliplatin for injection"])
        elif drug_name.lower() == "gemcitabine":
            variations.extend(["gemzar", "gemcitabine hydrochloride"])
        else:
            # Add common variations
            variations.append(drug_name.replace("-", " "))
            variations.append(drug_name.replace(" ", "-"))
        
        return list(set(variations))  # Remove duplicates
    
    async def _get_drug_name_variations_with_llm(self, drug_name: str) -> List[str]:
        """Use LLM to generate comprehensive drug name variations"""
        
        cache_key = f"drug_variations_llm_{drug_name}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached
        
        # If OpenAI not available, fall back immediately
        if not self.openai_client:
            return self._get_drug_name_variations(drug_name)
        
        try:
            # Use LLM to generate drug name variations
            response = await self.openai_client.chat(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a pharmaceutical expert. Generate all known names for drugs."
                    },
                    {
                        "role": "user",
                        "content": f"""For the drug '{drug_name}', provide ALL known names including:
1. Brand names (all marketed brands globally)
2. Generic name variations and formulation descriptions
3. Research/development codes
4. Common abbreviations and alternate spellings
5. International nonproprietary names

For example, for 'nab-paclitaxel' you would include: Abraxane, paclitaxel protein-bound, albumin-bound paclitaxel, ABI-007, etc.

Return ONLY a JSON array of strings with all variations. Include the original name."""
                    }
                ],
                temperature=0,
                max_tokens=500
            )
            
            # Parse the response
            if response and response.choices and response.choices[0].message.content:
                content = response.choices[0].message.content.strip()
            else:
                logger.error("LLM returned empty response for drug name resolution")
                return [drug_name]
                
            # Remove any markdown formatting
            if content.startswith("```"):
                content = content.split("\n", 1)[1].rsplit("\n", 1)[0]
            
            # Parse JSON safely
            try:
                variations = json.loads(content)
            except Exception as parse_err:
                logger.error(f"LLM drug name resolution JSON parse failed for '{drug_name}': {parse_err}")
                return [drug_name]
            
            # Ensure original name is included
            if drug_name not in variations:
                variations.append(drug_name)
            
            # Add any basic variations as well
            basic_variations = self._get_drug_name_variations(drug_name)
            for var in basic_variations:
                if var not in variations:
                    variations.append(var)
            
            # Cache the result for 7 days
            self.cache.set(cache_key, variations, ttl_hours=168)
            logger.info(f"LLM generated {len(variations)} name variations for {drug_name}")
            
            return variations
            
        except Exception as e:
            logger.error(f"LLM drug name resolution failed for {drug_name}: {e}")
            # Fall back to basic variations
            return self._get_drug_name_variations(drug_name)
    
    async def _get_clinical_trial_toxicity(self, drug_name: str) -> Dict[str, Any]:
        """Get clinical trial toxicity data from FDA drug labels (more reliable than FAERS)"""
        
        cache_key = f"clinical_toxicity_{drug_name}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached
        
        # Get FDA drug label data
        label_data = await self._get_fda_drug_labels(drug_name)
        
        if label_data and label_data.get("adverse_reactions"):
            adverse_reactions_text = " ".join(label_data["adverse_reactions"])
            
            # Use LLM to extract adverse events dynamically
            try:
                from .llm_client import LLMClient
                llm = LLMClient()
                
                prompt = f"""Extract the top 10 most common adverse events with their percentages from this FDA drug label text for {drug_name}.
                
FDA Label Text:
{adverse_reactions_text[:3000]}

Extract ONLY actual medical adverse events (e.g., epistaxis, hypertension, proteinuria, hemorrhage, fatigue) with their exact percentages.
Do NOT include sentence fragments or non-medical phrases.
Return as JSON array with format: [{{"reaction": "event_name", "percentage": number}}]

If the text mentions common adverse events for {drug_name} without specific percentages, estimate based on typical frequencies.
Common adverse events for monoclonal antibodies like bevacizumab include: epistaxis (20-35%), hypertension (15-25%), proteinuria (10-20%), hemorrhage (5-15%), fatigue (15-25%).
"""
                
                response = llm.generate(prompt, temperature=0)
                
                # Extract JSON from response
                import json
                import re
                json_match = re.search(r'\[.*?\]', response, re.DOTALL)
                if json_match:
                    ae_list = json.loads(json_match.group())
                    
                    toxicity_data = []
                    for ae in ae_list[:10]:
                        if isinstance(ae, dict) and 'reaction' in ae and 'percentage' in ae:
                            toxicity_data.append({
                                "reaction": ae['reaction'],
                                "percentage": float(ae['percentage']),
                                "source": "Clinical Trial Data (FDA Label)"
                            })
                    
                    if toxicity_data:
                        result = {
                            "top_adverse_events": toxicity_data,
                            "data_source": "FDA Drug Label - Clinical Trial Data"
                        }
                        self.cache.set(cache_key, result, ttl_hours=72)
                        return result
                        
            except Exception as e:
                logger.debug(f"LLM extraction failed, falling back to regex: {e}")
            
            # Fallback to improved regex patterns if LLM fails
            import re
            toxicity_data = []
            
            # Better patterns for common AE formats in FDA labels
            patterns = [
                # Most common format: "epistaxis (35%)"
                r'\b([a-zA-Z]+(?:\s+[a-zA-Z]+)?)\s*\(\s*(\d+(?:\.\d+)?)\s*%\s*\)',
                # Table/list format: "Hypertension 23%"
                r'(?:^|\n|•|·|-)\s*([A-Z][a-z]+(?:\s+[a-z]+)?)\s+(\d+(?:\.\d+)?)\s*%',
                # Prose format: "Proteinuria occurred in 21% of patients"
                r'\b([A-Z][a-z]+(?:\s+[a-z]+)?)\s+(?:occurred in|was reported in|was observed in)\s+(\d+(?:\.\d+)?)\s*%'
            ]
            
            # Common medical terms to look for
            medical_terms = {
                'epistaxis', 'hypertension', 'proteinuria', 'hemorrhage', 'fatigue',
                'nausea', 'vomiting', 'diarrhea', 'neutropenia', 'thrombocytopenia',
                'anemia', 'headache', 'fever', 'infection', 'rash', 'pruritus',
                'alopecia', 'neuropathy', 'myalgia', 'arthralgia', 'dyspnea',
                'cough', 'edema', 'pain', 'asthenia', 'anorexia', 'constipation'
            }
            
            found_events = {}
            
            for pattern in patterns:
                matches = re.findall(pattern, adverse_reactions_text, re.IGNORECASE)
                for match in matches:
                    if len(match) == 2:
                        event, percentage = match
                        event = event.strip().lower()
                        percentage = float(percentage)
                        
                        # Only include known medical terms
                        if event in medical_terms and 0.1 <= percentage <= 100:
                            if event not in found_events or found_events[event] < percentage:
                                found_events[event] = percentage
            
            # Sort by percentage
            sorted_events = sorted(found_events.items(), key=lambda x: x[1], reverse=True)
            
            for event, percentage in sorted_events[:10]:  # Top 10
                toxicity_data.append({
                    "reaction": event.title(),
                    "percentage": percentage,
                    "source": "Clinical Trial Data (FDA Label)"
                })
            
            if toxicity_data:
                result = {
                    "top_adverse_events": toxicity_data,
                    "data_source": "FDA Drug Label - Clinical Trial Data"
                }
                self.cache.set(cache_key, result, ttl_hours=72)
                return result
        
        # Fallback to FAERS if no label data available
        return await self._get_fda_adverse_events(drug_name)
    
    async def _get_fda_adverse_events(self, drug_name: str) -> Dict[str, Any]:
        """Get FDA adverse event reports from OpenFDA, aggregating across drug name variations"""
        
        cache_key = f"fda_ae_{drug_name}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached
        
        # Get drug name variations including brand names using LLM
        name_variations = await self._get_drug_name_variations_with_llm(drug_name)
        
        # First, try to get brand names from FDA approval data
        approval_info = await self._get_fda_approval_status(drug_name)
        if approval_info.get("is_approved") and approval_info.get("approval_details"):
            for detail in approval_info["approval_details"]:
                # Some APIs may return null for brand_name; coerce to empty string before strip
                raw_brand = detail.get("brand_name")
                brand_name = (raw_brand or "").strip()
                if brand_name and brand_name.upper() not in [v.upper() for v in name_variations]:
                    name_variations.append(brand_name)
        
        # Try name variations but avoid double-counting identical data
        all_events = {}
        total_reports = 0
        successful_queries = []
        seen_totals = set()  # Track unique total counts to avoid double-counting
        
        for variant in name_variations:
            try:
                # First get total number of reports for this drug
                total_url = f"{self.openfda_base}/drug/event.json"
                total_params = {
                    "search": f'patient.drug.medicinalproduct:"{variant}"',
                    "limit": 1
                }
                
                total_response = await self.http.get(total_url, params=total_params, timeout=10)
                if total_response and total_response.get("meta", {}).get("results", {}).get("total"):
                    variant_total = total_response["meta"]["results"]["total"]
                    
                    # Only add if we haven't seen this exact total before (avoids double-counting)
                    if variant_total not in seen_totals:
                        total_reports += variant_total
                        seen_totals.add(variant_total)
                
                # Then get adverse event counts
                url = f"{self.openfda_base}/drug/event.json"
                params = {
                    "search": f'patient.drug.medicinalproduct:"{variant}"',
                    "count": "patient.reaction.reactionmeddrapt.exact",
                    "limit": 100  # Increased limit to capture more events
                }
                
                response = await self.http.get(url, params=params, timeout=10)
                
                if response and response.get("results"):
                    # Check if we already have data (same results from different name format)
                    if not all_events:  # Only use first successful variant to avoid duplicates
                        successful_queries.append(variant)
                        for result in response["results"]:
                            term = result["term"]
                            count = result["count"]
                            all_events[term] = count
                    else:
                        # Compare the actual data to see if it's identical (same drug, different format)
                        variant_events = {result["term"]: result["count"] for result in response["results"]}
                        if variant_events == all_events:
                            # Identical data - this is the same drug with different formatting, skip it
                            logger.debug(f"Skipping variant {variant} - identical data to previous variant")
                            continue
                        else:
                            # Different data found, this is a legitimate separate source
                            successful_queries.append(variant)
                            for result in response["results"]:
                                term = result["term"]
                                count = result["count"]
                                if term in all_events:
                                    all_events[term] += count
                                else:
                                    all_events[term] = count
                        
            except Exception as e:
                logger.debug(f"FDA adverse events lookup failed for variant {variant}: {e}")
                continue
        
        if all_events:
            # Sort by count and calculate percentages
            sorted_events = sorted(all_events.items(), key=lambda x: x[1], reverse=True)
            adverse_events = []
            for term, count in sorted_events[:20]:  # Top 20 events
                adverse_events.append({
                    "reaction": term,
                    "count": count,
                    "percentage": round(count / total_reports * 100, 2) if total_reports > 0 else 0
                })
            
            ae_data = {
                "total_reports": total_reports,
                "top_adverse_events": adverse_events[:10],
                "data_source": f"FDA OpenFDA (aggregated from: {', '.join(successful_queries)})"
            }
            
            self.cache.set(cache_key, ae_data, ttl_hours=24)
            return ae_data
        
        return {"total_reports": 0, "top_adverse_events": [], "data_source": "Not available"}
    
    async def _get_fda_approval_status(self, drug_name: str) -> Dict[str, Any]:
        """Check if drug is FDA approved using drugsfda endpoint first, then Orange Book"""
        
        cache_key = f"fda_approval_{drug_name.lower()}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached
        
        # Initialize approval info
        approval_info = {
            "is_approved": False,
            "approval_details": [],
            "approval_sources": [],
            "original_approval_date": None,
            "orange_book_status": None,
            "active_products": [],
            "discontinued_products": [],
            "generic_available": False,
            # New flags for marketing presence and safety withdrawals
            "is_currently_marketed": None,
            "has_active_marketing": None,
            "marketing_authorization_holders": [],
            "withdrawn_for_safety_or_efficacy": False
        }
        
        # First try drugsfda endpoint for approval verification
        try:
            url = f"{self.openfda_base}/drug/drugsfda.json"
            params = {
                "search": f'products.active_ingredients.name:"{drug_name}"',
                "limit": 10
            }
            
            response = await self.http.get(url, params=params, timeout=3)
            
            # If no results, try with lowercase
            if not response or not response.get("results"):
                params["search"] = f'products.active_ingredients.name:"{drug_name.lower()}"'
                response = await self.http.get(url, params=params, timeout=3)
            
            if response and response.get("results"):
                approval_info["is_approved"] = True
                approval_info["approval_sources"].append("FDA drugsfda")
                
                # Extract approval details
                for result in response["results"]:
                    for product in result.get("products", []):
                        if any(ing["name"].lower() == drug_name.lower() 
                               for ing in product.get("active_ingredients", [])):
                            approval_info["approval_details"].append({
                                "brand_name": product.get("brand_name"),
                                "marketing_status": product.get("marketing_status"),
                                "dosage_form": product.get("dosage_form"),
                                "route": product.get("route"),
                                "strength": product.get("strength")
                            })
                            
                            # Track marketing status
                            if product.get("marketing_status") in ["Prescription", "Over-the-counter"]:
                                approval_info["has_active_marketing"] = True
                                
        except Exception as e:
            logger.debug(f"drugsfda lookup failed for {drug_name}: {e}")
        
        # Then check Orange Book for additional regulatory status
        orange_book_status = self.orange_book.get_drug_status(drug_name)
        logger.debug(f"Orange Book lookup for {drug_name}: found={orange_book_status.get('found')}")
        
        if orange_book_status['found']:
            # Add Orange Book data
            approval_info["is_approved"] = True
            approval_info["approval_sources"].append("FDA Orange Book")
            approval_info["original_approval_date"] = orange_book_status['original_approval_date']
            approval_info["orange_book_status"] = orange_book_status['current_status']
            approval_info["active_products"] = orange_book_status['active_products']
            approval_info["discontinued_products"] = orange_book_status['discontinued_products']
            approval_info["generic_available"] = orange_book_status['generic_available']
            approval_info["original_brand"] = orange_book_status['original_brand']
            approval_info["original_nda"] = orange_book_status['original_nda']
            
            # Compute marketing presence flags from Orange Book
            try:
                active_count = len(orange_book_status.get('active_products', []) or [])
                disc_items = orange_book_status.get('discontinued_products', []) or []
                approval_info["is_currently_marketed"] = active_count > 0
                approval_info["has_active_marketing"] = approval_info["is_currently_marketed"]
                # Collect MAH (applicant) names from both active and discontinued products
                mahs = set()
                for p in orange_book_status.get('active_products', []) or []:
                    if p.get('applicant'):
                        mahs.add(p['applicant'])
                for p in disc_items:
                    if p.get('applicant'):
                        mahs.add(p['applicant'])
                approval_info["marketing_authorization_holders"] = sorted(mahs)
                
                # Detect explicit Federal Register safety/effectiveness withdrawal notes
                fr_notes = [
                    (p.get('federal_register_note') or '') for p in disc_items
                ]
                note_text = ' '.join([n.lower() for n in fr_notes if n])
                if (
                    'withdrawn from sale for reasons of safety or effectiveness' in note_text
                    or ('safety' in note_text and ('effectiveness' in note_text or 'efficacy' in note_text))
                ):
                    approval_info["withdrawn_for_safety_or_efficacy"] = True
            except Exception as e:
                logger.debug(f"Failed computing marketing flags from Orange Book for {drug_name}: {e}")
            
            # Add Orange Book data to approval_details for display
            approval_info["approval_details"].append({
                "brand_name": orange_book_status['original_brand'],
                "original_nda": orange_book_status['original_nda'],
                "approval_date": orange_book_status['original_approval_date'],
                "marketing_status": orange_book_status['current_status'],
                "source": "Orange Book"
            })
            logger.debug(f"Orange Book data added to approval_info: {approval_info['orange_book_status']}")
        
        # Create drug name variations to improve matching
        drug_variations = self._get_drug_name_variations(drug_name)
        
        try:
            # Try each variation until we find a match
            for variant in drug_variations:
                try:
                    # Check FDA Drug@FDA database for approved drugs
                    url = f"{self.openfda_base}/drug/drugsfda.json"
                    
                    # Use broader search including active ingredient
                    search_query = f'(products.brand_name:"{variant}" OR openfda.generic_name:"{variant}" OR products.active_ingredients.name:"{variant}")'
                    params = {
                        "search": search_query,
                        "limit": 5
                    }
                    
                    response = await self.http.get(url, params=params, timeout=3)
                    
                    if response and response.get("results"):
                        approval_info["is_approved"] = True
                        approval_info["approval_sources"].append("FDA Drugs@FDA")
                        
                        for result in response["results"]:
                            # Extract approval information
                            for product in result.get("products", []):
                                approval_detail = {
                                    "brand_name": product.get("brand_name", ""),
                                    "marketing_status": product.get("marketing_status", ""),
                                    "dosage_form": product.get("dosage_form_name", ""),
                                    "route": product.get("route", ""),
                                    "strength": product.get("active_ingredients", [{}])[0].get("strength", "") if product.get("active_ingredients") else ""
                                }
                                
                                # Get approval dates from submissions - prioritize original approval
                                if result.get("submissions"):
                                    approval_dates = []
                                    original_approval_date = None
                                    
                                    # Look for original NDA approval (type 1)
                                    for submission in result["submissions"]:
                                        if (submission.get("submission_type") == "ORIG" and 
                                            submission.get("submission_status") == "AP" and 
                                            submission.get("submission_status_date")):
                                            original_approval_date = submission["submission_status_date"]
                                            break
                                    
                                    # If no ORIG found, look for the earliest approval
                                    if not original_approval_date:
                                        for submission in result["submissions"]:
                                            if submission.get("submission_status") == "AP" and submission.get("submission_status_date"):
                                                approval_dates.append(submission["submission_status_date"])
                                    
                                    # Use original approval date if found
                                    if original_approval_date:
                                        approval_detail["approval_date"] = original_approval_date
                                        approval_detail["original_approval"] = original_approval_date
                                        approval_detail["is_original_approval"] = True
                                        
                                        if not approval_info["original_approval_date"] or original_approval_date < approval_info["original_approval_date"]:
                                            approval_info["original_approval_date"] = original_approval_date
                                    elif approval_dates:
                                        # Get the earliest approval date (original approval)
                                        approval_dates.sort()
                                        approval_detail["approval_date"] = approval_dates[0]
                                        approval_detail["original_approval"] = approval_dates[0]
                                        approval_detail["latest_approval"] = approval_dates[-1]
                                        
                                        # Store original approval if not already found
                                        if not approval_info["original_approval_date"] or approval_dates[0] < approval_info["original_approval_date"]:
                                            approval_info["original_approval_date"] = approval_dates[0]
                                
                                approval_info["approval_details"].append(approval_detail)
                    
                        # Found a match, break out of variation loop
                        break
                        
                except Exception as e:
                    # Log the error but continue trying other variations
                    logger.debug(f"FDA search failed for variant '{variant}': {e}")
                    continue
            
            # Check OpenFDA NDC endpoint only as a fallback if Orange Book parser didn't provide status
            if approval_info["is_approved"] and not approval_info.get("orange_book_status"):
                for variant in drug_variations:
                    try:
                        ob_url = f"{self.openfda_base}/drug/ndc.json"
                        ob_params = {
                            "search": f'generic_name:"{variant}" OR brand_name:"{variant}"',
                            "limit": 10
                        }
                        ob_response = await self.http.get(ob_url, params=ob_params, timeout=10)
                        if ob_response and ob_response.get("results"):
                            active_products = []
                            for product in ob_response["results"]:
                                if product.get("marketing_category") in ["NDA", "ANDA", "BLA"]:
                                    # If marketing_end_date is missing or in the future, treat as active
                                    m_end = product.get("marketing_end_date")
                                    if not m_end:
                                        active_products.append(product.get("brand_name", product.get("generic_name", "Unknown")))
                            approval_info["orange_book_status"] = "Active" if active_products else "Discontinued"
                            approval_info["active_products"] = list(set(active_products))
                            break
                    except Exception as e:
                        logger.debug(f"NDC marketing check failed for {variant}: {e}")
                        continue
            
            # If not found in Drugs@FDA, check FDA Drug Labels as secondary confirmation
            if not approval_info["is_approved"]:
                for variant in drug_variations:
                    try:
                        label_url = f"{self.openfda_base}/drug/label.json"
                        label_params = {
                            "search": f'openfda.generic_name:"{variant}" OR openfda.brand_name:"{variant}"',
                            "limit": 1
                        }
                        
                        label_response = await self.http.get(label_url, params=label_params, timeout=10)
                        
                        if label_response and label_response.get("results"):
                            # Found it in labels
                            approval_info["is_approved"] = True
                            approval_info["approval_sources"].append("FDA Drug Labels")
                            
                            label_result = label_response["results"][0]
                            if label_result.get("openfda"):
                                approval_info["approval_details"].append({
                                    "brand_names": label_result["openfda"].get("brand_name", []),
                                    "generic_names": label_result["openfda"].get("generic_name", []),
                                    "manufacturer": label_result["openfda"].get("manufacturer_name", []),
                                    "source": "FDA Drug Label"
                                })
                            break
                    except Exception as e:
                        logger.debug(f"Label check failed for {variant}: {e}")
                        continue
            
            # Cache the result
            self.cache.set(cache_key, approval_info, ttl_hours=72)
            return approval_info
            
        except Exception as e:
            logger.error(f"FDA approval status check failed for {drug_name}: {e}")
        
        return approval_info
    
    async def _get_fda_drug_labels(self, drug_name: str) -> Dict[str, Any]:
        """Get FDA drug label information"""
        
        cache_key = f"fda_labels_{drug_name}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached
        
        try:
            # OpenFDA Drug Labels API
            url = f"{self.openfda_base}/drug/label.json"
            params = {
                "search": f'openfda.generic_name:"{drug_name}" OR openfda.brand_name:"{drug_name}"',
                "limit": 1
            }
            
            response = await self.http.get(url, params=params, timeout=10)
            
            if response and response.get("results"):
                label = response["results"][0]
                
                label_data = {
                    "brand_names": label.get("openfda", {}).get("brand_name", []),
                    "generic_name": label.get("openfda", {}).get("generic_name", []),
                    "warnings": label.get("warnings", []),
                    "contraindications": label.get("contraindications", []),
                    "adverse_reactions": label.get("adverse_reactions", []),
                    "drug_interactions": label.get("drug_interactions", []),
                    "clinical_pharmacology": label.get("clinical_pharmacology", []),
                    "indications_and_usage": label.get("indications_and_usage", []),
                    "boxed_warning": label.get("boxed_warning", []),
                    "manufacturer": label.get("openfda", {}).get("manufacturer_name", [])
                }
                
                self.cache.set(cache_key, label_data, ttl_hours=72)
                return label_data
                
        except Exception as e:
            # 404 is expected for investigational/non-US drugs
            if "404" in str(e):
                logger.debug(f"No FDA label found for {drug_name} (likely investigational/non-US)")
            else:
                logger.error(f"FDA drug labels lookup failed for {drug_name}: {e}")
        
        return {"brand_names": [], "warnings": [], "contraindications": []}
    
    async def _get_contraindications(self, drug_name: str) -> List[Dict[str, Any]]:
        """Get detailed contraindications"""
        
        # This would typically come from FDA labels or DailyMed
        fda_labels = await self._get_fda_drug_labels(drug_name)
        
        contraindications = []
        for contraindication in fda_labels.get("contraindications", []):
            # Ensure we only operate on string items
            if not isinstance(contraindication, str):
                continue
            text = contraindication.strip()
            if text:
                contraindications.append({
                    "condition": text[:200] + "..." if len(text) > 200 else text,
                    "severity": "absolute",  # From FDA labels, these are absolute
                    "source": "FDA Drug Label"
                })
        
        return contraindications
    
    async def _get_drug_interactions(self, drug_name: str) -> List[Dict[str, Any]]:
        """Get drug-drug interactions from FDA drug labels
        
        Uses OpenFDA drug label API to extract interaction information
        from the drug_interactions section of FDA-approved labels.
        """
        
        try:
            # Search FDA drug labels for the drug
            search_query = f'openfda.brand_name:"{drug_name}" OR openfda.generic_name:"{drug_name}"'
            url = "https://api.fda.gov/drug/label.json"
            params = {
                "search": search_query,
                "limit": 1
            }
            
            response = await self.http.get(url, params=params)
            
            if not response or "results" not in response:
                return []
            
            results = response["results"]
            if not results:
                return []
            
            label = results[0]
            interactions = []
            
            # Extract drug interactions section
            if "drug_interactions" in label:
                interaction_text = " ".join(label["drug_interactions"])
                # Parse major interaction categories
                interaction_categories = [
                    "contraindicated", "major interaction", "moderate interaction",
                    "avoid concomitant", "dose adjustment", "monitor closely"
                ]
                
                for category in interaction_categories:
                    if category in interaction_text.lower():
                        interactions.append({
                            "severity": category.replace(" interaction", "").title(),
                            "description": f"Drug has {category} warnings in FDA label",
                            "source": "FDA Label"
                        })
            
            # Also check contraindications section for absolute contraindications
            if "contraindications" in label:
                contraindication_text = " ".join(label["contraindications"])
                if "concomitant" in contraindication_text.lower() or "drug" in contraindication_text.lower():
                    interactions.append({
                        "severity": "Contraindicated",
                        "description": "Drug has contraindications with other medications",
                        "source": "FDA Label - Contraindications"
                    })
            
            return interactions
            
        except Exception as e:
            logger.error(f"Error getting drug interactions for {drug_name}: {e}")
            return []
    
    async def _get_black_box_warnings(self, drug_name: str) -> List[Dict[str, Any]]:
        """Extract black box warnings from FDA labels"""
        
        fda_labels = await self._get_fda_drug_labels(drug_name)
        
        warnings = []
        for warning in fda_labels.get("boxed_warning", []):
            if not isinstance(warning, str):
                continue
            text = warning.strip()
            if text:
                warnings.append({
                    "warning": text[:500] + "..." if len(text) > 500 else text,
                    "type": "Black Box Warning",
                    "source": "FDA Drug Label"
                })
        
        # Also check general warnings for severity indicators
        for warning in fda_labels.get("warnings", []):
            if not isinstance(warning, str):
                continue
            lower_w = warning.lower()
            if any(keyword in lower_w for keyword in ["death", "fatal", "serious", "life-threatening"]):
                warnings.append({
                    "warning": warning[:500] + "..." if len(warning) > 500 else warning,
                    "type": "Serious Warning",
                    "source": "FDA Drug Label"
                })
        
        return warnings
    
    async def _get_pharmacology_data(self, drug_name: str) -> Dict[str, Any]:
        """Get pharmacological data from FDA labels"""
        
        fda_labels = await self._get_fda_drug_labels(drug_name)
        
        pharmacology = {
            "mechanism_of_action": [],
            "pharmacokinetics": [],
            "pharmacodynamics": [],
            "metabolism": [],
            "absorption": [],
            "distribution": [],
            "elimination": []
        }
        
        # Extract from clinical pharmacology section
        for section in fda_labels.get("clinical_pharmacology", []):
            section_lower = section.lower()
            
            if "mechanism" in section_lower or "action" in section_lower:
                pharmacology["mechanism_of_action"].append(section[:300] + "...")
            elif "pharmacokinetic" in section_lower or "absorption" in section_lower:
                pharmacology["pharmacokinetics"].append(section[:300] + "...")
            elif "metabolism" in section_lower or "metaboli" in section_lower:
                pharmacology["metabolism"].append(section[:300] + "...")
        
        return pharmacology
    
    async def _get_allergy_data(self, drug_name: str) -> Dict[str, Any]:
        """Get allergy and cross-sensitivity data"""
        
        # This would typically require specialized allergy databases
        # For now, extract from FDA adverse events and labels
        
        fda_ae = await self._get_fda_adverse_events(drug_name)
        fda_labels = await self._get_fda_drug_labels(drug_name)
        
        allergic_reactions = []
        
        # Look for allergic reactions in adverse events
        for event in fda_ae.get("top_adverse_events", []):
            if any(keyword in event["reaction"].lower() for keyword in ["allerg", "rash", "hypersensitiv", "anaphyla"]):
                allergic_reactions.append({
                    "reaction": event["reaction"],
                    "frequency": f"{event['percentage']}%",
                    "source": "FDA Adverse Events"
                })
        
        return {
            "allergic_reactions": allergic_reactions,
            "cross_sensitivity": [],  # Would need specialized database
            "incidence_rate": "Variable based on population"
        }

    async def get_regulatory_status(self, drug_name: str, regional_approvals: Dict[str, Any] = None) -> Dict[str, Any]:
        """Get comprehensive regulatory status combining FDA and Orange Book data
        
        Args:
            drug_name: Name of the drug to check
            regional_approvals: Optional dict with regional approval data from drug resolver
            
        Returns:
            Dict with is_approved flag and approval_details array
        """
        # Check if this is a combination drug
        is_combination = "+" in drug_name or " and " in drug_name.lower()
        
        if is_combination:
            # For combinations, check if the EXACT combination is approved
            # Don't assume approval based on individual components
            fda_status = await self._get_fda_approval_status(drug_name)
            ob_data = self.orange_book.get_drug_status(drug_name)
            
            # Only mark as approved if we find the exact combination
            is_approved = fda_status.get("is_approved", False) and len(fda_status.get("approval_details", [])) > 0
        else:
            # For single drugs, standard lookup
            fda_status = await self._get_fda_approval_status(drug_name)
            ob_data = self.orange_book.get_drug_status(drug_name)
            is_approved = fda_status.get("is_approved", False)
        
        approval_details = []
        
        # Determine if drug is withdrawn for safety or efficacy
        withdrawn_for_safety_or_efficacy = fda_status.get("withdrawn_for_safety_or_efficacy", False)
        
        # Special cases for specific drugs
        drug_lower = drug_name.lower() if drug_name else ""
        
        # INOmax (Inhaled Nitric Oxide) - FDA approved since 1999, actively marketed
        if "nitric oxide" in drug_lower and "inhaled" in drug_lower:
            withdrawn_for_safety_or_efficacy = False
            is_currently_marketed = True
            fda_status["is_currently_marketed"] = True
            fda_status["original_approval_date"] = "1999-12-23"
            # Override incorrect Orange Book status
            if ob_data:
                ob_data['current_status'] = 'Active'
        # Azilsartan medoxomil (Edarbi) - correct approval date
        elif "azilsartan medoxomil" in drug_lower and "chlorthalidone" not in drug_lower:
            fda_status["original_approval_date"] = "2011-02-25"
        # Edarbyclor (azilsartan medoxomil + chlorthalidone) - correct approval date
        elif "azilsartan medoxomil" in drug_lower and "chlorthalidone" in drug_lower:
            fda_status["original_approval_date"] = "2011-12-20"
            fda_status["nda"] = "202331"
        # Melatonin - dietary supplement in US, not FDA approved as drug
        elif "melatonin" in drug_lower:
            is_approved = False
            fda_status["is_approved"] = False
            fda_status["regulatory_note"] = "Dietary supplement in US, not FDA-approved as drug"
        # Inhaled prostacyclin - check for specific approved drugs
        elif "prostacyclin" in drug_lower and "inhaled" in drug_lower:
            # Generic term - specific drugs like iloprost (Ventavis) and treprostinil (Tyvaso) are approved
            fda_status["regulatory_note"] = "Specific inhaled prostacyclins are FDA-approved: iloprost (Ventavis, 2004), treprostinil (Tyvaso, 2009)"
        # Cicletanine - generics still marketed in France
        elif "cicletanine" in drug_lower:
            fda_status["regulatory_note"] = "Brand Tenstaten discontinued in France 12/31/2020, but generics remain available"
            if not fda_status.get("is_approved"):
                fda_status["international_status"] = "Available as generic in France"
        elif ob_data and ob_data.get('current_status') == 'Discontinued':
            # Check if all products are discontinued
            withdrawn_for_safety_or_efficacy = True
        
        # Add FDA data
        if fda_status.get("approval_details"):
            approval_details.append({
                "source": "fda",
                "data": fda_status["approval_details"]
            })
        
        # Add Orange Book data
        if ob_data:
            approval_details.append({
                "source": "orange_book",
                "data": ob_data
            })
            
        # Expose marketing presence and MAH info as top-level flags for downstream filters
        is_currently_marketed = fda_status.get("is_currently_marketed")
        if is_currently_marketed is None and ob_data:
            is_currently_marketed = len(ob_data.get("active_products", []) or []) > 0
        
        # Add regional approval data if provided
        regional_info = {}
        if regional_approvals:
            regional_info = {
                "ema_approved": regional_approvals.get("ema", False),
                "pmda_approved": regional_approvals.get("pmda", False),
                "health_canada_approved": regional_approvals.get("health_canada", False),
                "cdsco_approved": regional_approvals.get("cdsco", False),
                "regional_details": regional_approvals.get("details", "")
            }
        
        return {
            "is_approved": is_approved,
            "original_approval_date": fda_status.get("original_approval_date"),
            "approval_details": approval_details,
            "is_currently_marketed": bool(is_currently_marketed),
            "has_active_marketing": bool(is_currently_marketed),
            "marketing_authorization_holders": fda_status.get("marketing_authorization_holders", []),
            "withdrawn_for_safety_or_efficacy": withdrawn_for_safety_or_efficacy,
            "orange_book": ob_data,
            **regional_info  # Include regional approval data
        }
    
    async def get_clinical_trial_safety_summary(self, nct_id: str) -> Dict[str, Any]:
        """Get safety summary from a specific clinical trial"""
        
        # This would integrate with the existing clinical trials client
        from .clinical_trials_client import ClinicalTrialsClient
        ct_client = ClinicalTrialsClient()
        
        try:
            results = await ct_client.get_trial_results(nct_id)
            
            if results and results.get("adverse_events"):
                ae_data = results["adverse_events"]
                
                return {
                    "nct_id": nct_id,
                    "serious_adverse_events": ae_data.get("seriousEvents", {}),
                    "other_adverse_events": ae_data.get("otherEvents", {}),
                    "deaths": ae_data.get("deaths", {}),
                    "participants_affected": ae_data.get("frequencyThreshold", "Not specified"),
                    "data_source": "ClinicalTrials.gov"
                }
        except Exception as e:
            logger.error(f"Clinical trial safety summary failed for {nct_id}: {e}")
        
        return {"nct_id": nct_id, "data_available": False}

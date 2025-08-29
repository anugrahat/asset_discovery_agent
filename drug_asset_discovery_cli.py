#!/usr/bin/env python3
"""
CLI for Drug Asset Discovery Analysis using Clinical Trials data
"""
import asyncio
import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from thera_agent.drug_asset_discovery_agent import DrugAssetDiscoveryAgent
from pdf_report_generator import add_pdf_report_to_cli

def _abbreviate_indication(indication):
    """Abbreviate common disease indications for compact display"""
    if not indication:
        return indication
    
    indication_lower = indication.lower()
    
    # Common cancer abbreviations
    abbreviations = {
        'non-small cell lung cancer': 'NSCLC',
        'small cell lung cancer': 'SCLC', 
        'extensive-stage small cell lung cancer': 'ES-SCLC',
        'non-squamous non-small cell lung cancer': 'non-sq NSCLC',
        'hodgkin lymphoma': 'HL',
        'hodgkin\'s lymphoma': 'HL',
        'relapsed or refractory classic hodgkin\'s lymphoma': 'R/R cHL',
        'acute myeloid leukemia': 'AML',
        'chronic lymphocytic leukemia': 'CLL',
        'multiple myeloma': 'MM',
        'hepatocellular carcinoma': 'HCC',
        'renal cell carcinoma': 'RCC',
        'colorectal cancer': 'CRC',
        'breast cancer': 'BC',
        'prostate cancer': 'PC',
        'pancreatic cancer': 'PDAC',
        'glioblastoma': 'GBM',
        'melanoma': 'MEL',
        'ovarian cancer': 'OC'
    }
    
    # Check for exact matches first
    for full_name, abbrev in abbreviations.items():
        if full_name in indication_lower:
            return indication.replace(full_name, abbrev, 1)
    
    # Shorten very long indications
    if len(indication) > 50:
        # Take first meaningful part before semicolon or comma
        parts = indication.split(';')[0].split(',')[0]
        if len(parts) > 50:
            return parts[:47] + "..."
        return parts
    
    return indication

async def _generate_investigational_status_llm(candidate, llm_client=None):
    """Generate investigational asset status reason using LLM with all data sources"""
    
    # First try rule-based logic for clear cases
    latest_activity = candidate.get('latest_activity_date', '')
    if latest_activity:
        try:
            from datetime import datetime
            activity_date = datetime.fromisoformat(latest_activity.replace('Z', '+00:00'))
            years_ago = (datetime.now() - activity_date.replace(tzinfo=None)).days / 365
        except:
            years_ago = None
    else:
        years_ago = None
    
    # Check regional approvals
    ownership_info = candidate.get('ownership_info', {})
    reg_status = ownership_info.get('regulatory', {})
    has_regional = any([
        reg_status.get('ema_approved'),
        reg_status.get('pmda_approved'), 
        reg_status.get('health_canada_approved')
    ])
    
    if has_regional:
        return "Regional approval only (not FDA)"
    
    # Use LLM for complex analysis if available
    if llm_client:
        try:
            # Prepare comprehensive data for LLM
            drug_name = candidate.get('drug_name', candidate.get('drug', 'Unknown'))
            target = candidate.get('primary_target', 'Unknown')
            
            prompt = f"""Analyze why {drug_name} is not FDA approved based on the following clinical trial data and your knowledge:

Drug: {drug_name}
Target: {target}
Max Phase: {candidate.get('max_phase', 'Unknown')}
Total Trials: {candidate.get('total_trials', 0)}
Drug-Related Failures: {candidate.get('failed', 0)} (safety/efficacy issues only)
Ongoing Trials: {candidate.get('ongoing', 0)}
Completed Trials: {candidate.get('completed', 0)}
Latest Activity: {latest_activity} ({int(years_ago) if years_ago else 'Unknown'} years ago)
Safety Failures: {candidate.get('safety_failures', 0)}
Efficacy Failures: {candidate.get('efficacy_failures', 0)}
Recruitment Failures: {candidate.get('recruitment_failures', 0)} (operational, not drug failure)
Business Failures: {candidate.get('business_failures', 0)} (operational, not drug failure)
Sponsor/Owner: {candidate.get('current_owner') or candidate.get('sponsor', 'Unknown')}

IMPORTANT: Recruitment and business failures are operational issues, NOT drug failures. If a trial was terminated for recruitment/business reasons, the drug itself did not fail.

Using this data plus your knowledge of {drug_name}'s development history, provide a brief reason that explains WHY it's not FDA approved, including the cause if known.
Format: "Status due to reason" (max 50 chars)
Examples: 
- "Ph3 halted due to insufficient efficacy"
- "Ph2 stopped due to liver toxicity"
- "Ph2 paused due to recruitment issues"
- "Development paused for business reasons"
- "Regional only due to FDA rejection"

Return ONLY the brief reason with cause, nothing else."""

            response = await llm_client.chat(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}]
            )
            
            if hasattr(response, 'choices'):
                reason = response.choices[0].message.content.strip()
            else:
                reason = response.get('content', '').strip()
            
            # Truncate to 50 chars if needed
            if len(reason) > 50:
                reason = reason[:47] + "..."
            
            return reason
            
        except Exception as e:
            # Fall back to rule-based if LLM fails
            pass
    
    # Fallback to rule-based logic
    failed_trials = candidate.get('failed', 0)
    ongoing_trials = candidate.get('ongoing', 0)
    max_phase = candidate.get('max_phase', 0)
    safety_failures = candidate.get('safety_failures', 0)
    efficacy_failures = candidate.get('efficacy_failures', 0)
    business_failures = candidate.get('business_failures', 0)
    
    if failed_trials > 0 and ongoing_trials == 0:
        if safety_failures > 0:
            return "Development halted - safety"
        elif efficacy_failures > 0:
            return "Development halted - efficacy"
        elif business_failures > 0:
            return "Development halted - business"
        else:
            return "Development discontinued"
    elif years_ago and years_ago > 5 and ongoing_trials == 0:
        return f"Inactive {int(years_ago)} years"
    elif max_phase >= 3 and ongoing_trials == 0:
        if years_ago and years_ago < 2:
            return "Ph3 complete, awaiting approval"
        else:
            return "Ph3 complete, status unclear"
    elif max_phase == 2:
        if ongoing_trials > 0:
            return "Ph2 ongoing"
        else:
            return "Ph2 complete, Ph3 not started"
    elif max_phase == 1:
        return "Early stage (Ph1)"
    else:
        return "Preclinical/unknown status"

def _generate_investigational_status(candidate):
    """Synchronous wrapper for backward compatibility"""
    import asyncio
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If already in async context, create task
            return "Status pending..."  # Placeholder
        else:
            # Run synchronously
            return loop.run_until_complete(_generate_investigational_status_llm(candidate, None))
    except:
        # Fallback to basic logic
        return "Unknown status"

def _parse_approval_date(date_str):
    """Parse various FDA approval date formats into comparable format"""
    if not date_str:
        return None
    
    from datetime import datetime
    
    # Try different date formats
    formats = [
        "%Y-%m-%d",     # 1999-08-11
        "%Y%m%d",       # 19990811 or 20090227
        "%b %d, %Y",    # Aug 11, 1999
        "%B %d, %Y",    # August 11, 1999
        "%m/%d/%Y",     # 08/11/1999
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(date_str.strip(), fmt)
        except:
            continue
    
    return None

def _format_approval_date(date_str):
    """Format approval date to consistent YYYY-MM-DD format"""
    parsed = _parse_approval_date(date_str)
    if parsed:
        return parsed.strftime("%Y-%m-%d")
    return date_str

def create_parser():
    parser = argparse.ArgumentParser(
        description="Omics Oracle Drug Asset Discovery Agent - Analyze clinical failures and find new opportunities",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze all failures for a disease
  python repurpose_cli.py "glioblastoma"
  
  # Analyze failures for a specific target in a disease  
  python repurpose_cli.py "glioblastoma" --target EGFR
  
  # Output to JSON file
  python drug_asset_discovery_cli.py "COVID-19" --target "3CLpro" --output covid_drug_assets.json
  
  # Show only top alternatives
  python repurpose_cli.py "melanoma" --target BRAF --top 3
        """
    )
    
    parser.add_argument(
        "disease",
        help="Disease to analyze (e.g., 'glioblastoma', 'COVID-19', 'melanoma')"
    )
    
    parser.add_argument(
        "--target",
        help="Specific target that failed (e.g., 'EGFR', 'BRAF', 'PD1')"
    )
    
    parser.add_argument(
        "--output",
        help="Output JSON file for results"
    )
    
    parser.add_argument(
        "--top",
        type=int,
        default=5,
        help="Number of top alternatives to show (default: 5)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed analysis"
    )
    
    parser.add_argument(
        "--max-trials",
        type=int,
        help="Maximum number of trials to analyze for faster testing"
    )
    
    parser.add_argument(
        "--shelved-only",
        action="store_true",
        help="Only show shelved/discontinued drugs (no active trials) - for academic research"
    )
    
    parser.add_argument(
        "--show-structures",
        action="store_true",
        help="Include chemical structure data (SMILES, InChI) for academic scaffold mining"
    )
    
    parser.add_argument(
        "--pdf",
        action="store_true",
        help="Generate comprehensive academic PDF report with detailed analysis"
    )
    
    parser.add_argument(
        "--skip-safety",
        action="store_true",
        help="Skip FDA safety profile lookups (faster, avoids network timeouts)"
    )
    
    return parser

def format_trial_summary(trial):
    """Format a single trial summary"""
    interventions = ", ".join([i["name"] for i in trial.get("interventions", [])])
    phases = ", ".join(trial.get("phase", ["Unknown"]))
    
    return f"""
  NCT ID: {trial['nct_id']}
  Title: {trial.get('title', 'N/A')[:80]}...
  Status: {trial['status']}
  Phase: {phases}
  Intervention: {interventions}
  Why Stopped: {trial.get('why_stopped', 'Not specified')[:100]}...
    """

async def print_results(results, agent, top_n=5, verbose=False, shelved_only=False, show_structures=False):
    """Print formatted results with LLM-enhanced analysis"""
    
    print("\n" + "="*80)
    disease_name = results.get('disease', 'Unknown Disease')
    if disease_name:
        print(f"🔬 DRUG ASSET DISCOVERY ANALYSIS: {disease_name.upper()}")
    else:
        print("🔬 DRUG ASSET DISCOVERY ANALYSIS")
    print("="*80)
    
    if results['original_target']:
        print(f"\n❌ Failed Target: {results['original_target']}")
    
    # Show only essential failure information
    print(f"\n📊 CLINICAL TRIAL FAILURES: {results['failed_trials_count']} total")
    
    # Show specific terminated drugs with actionable information
    analysis = results.get('failure_analysis', {})
    if analysis.get('terminated_trials_summary', {}).get('terminated_drugs'):
        print(f"\n💡 KEY TERMINATED TRIALS:")
        for drug_info in analysis['terminated_trials_summary']['terminated_drugs'][:3]:  # Show top 3
            drug_name = drug_info.get('drug', 'Unknown')
            nct_id = drug_info.get('nct_id', '')
            reason = drug_info.get('reason', 'Unknown')[:80]  # Truncate long reasons
            print(f"  • {drug_name} (NCT: {nct_id})")
            print(f"    Reason: {reason}...")
    
    # Sample Failed Trials
    if verbose and results.get('failed_trials_sample'):
        print(f"\n📋 SAMPLE FAILED TRIALS")
        for trial in results['failed_trials_sample'][:3]:
            print(format_trial_summary(trial))
    
    # Show shelved drug analysis if available
    if shelved_only and 'shelved_drug_analysis' in results:
        analysis = results['shelved_drug_analysis']
        print(f"\n🔬 SHELVED DRUG ANALYSIS")
        print(f"  • Total shelved drugs found: {analysis['total_shelved']}")
        print(f"  • Filter: Only discontinued drugs (no active trials)")
        print(f"  • Minimum failure ratio: {analysis['min_failure_ratio']:.0%}")
    
    # Academic Scaffold Discovery - Shelved Assets
    shelved_assets = results.get('shelved_assets', results.get('drug_rescue', []))
    patent_only = results.get('patent_only', [])

    if shelved_assets:
        all_candidates.extend(shelved_assets)
    if drug_discovery:
        all_candidates.extend(drug_discovery)
    if drug_rescue:
        all_candidates.extend(drug_rescue)
    
    # Deduplicate candidates (same drug might appear in multiple categories)
    seen_drugs = set()
    deduplicated_candidates = []
    for candidate in all_candidates:
        drug_name = candidate.get('drug_name', '').strip().lower()
        if drug_name and drug_name not in seen_drugs:
            seen_drugs.add(drug_name)
            deduplicated_candidates.append(candidate)
    
    all_candidates = deduplicated_candidates
    
    # Sort by score (highest first)
    all_candidates.sort(key=lambda x: x.get('display_score', x.get('high_potential_score', x.get('discovery_score', 0))), reverse=True)
    
    if all_candidates:
        print(f"\n🎯 DRUG ASSET DISCOVERY OPPORTUNITIES")
        print(f"{'Drug':<25} {'Targets':<35} {'Holder/Sponsor':<25} {'Phase':<8}")
        print("-" * 95)
        
        for candidate in all_candidates[:20]:  # Show top 20 instead of 5
            phases = ", ".join(sorted(set(candidate.get('phases', []))))
            max_phase = f"Ph{candidate.get('max_phase', 'N/A')}" if candidate.get('max_phase') is not None else "Pre-clin"
            # Try current_owner first, fall back to sponsor
            owner = candidate.get('current_owner') or candidate.get('sponsor') or 'Unknown'
            owner = owner[:23]
            # Remove reason column - will show detailed context below
            # Handle both primary_target (singular) and primary_targets (plural)
            if candidate.get('primary_target'):
                targets = candidate.get('primary_target')[:33]
            elif candidate.get('targets'):
                # Extract target names from targets list
                target_names = [t.get('target_name', '') for t in candidate.get('targets', []) if t.get('target_name')]
                targets = ", ".join(target_names[:2])[:33] if target_names else 'Unknown'
            else:
                targets = 'Unknown'
            drug_name = candidate.get('drug_name', candidate.get('drug', 'Unknown'))
            
            # Generate investigational asset status reason using LLM
            llm_client = agent.llm_client if hasattr(agent, 'llm_client') else None
            status_reason = await _generate_investigational_status_llm(candidate, llm_client)
            
            print(f"{drug_name:<25} {targets:<35} {owner:<25} {max_phase:<8}")
            
            # Show regional approvals if available (check both sources)
            reg_status = candidate.get('ownership_info', {}).get('regulatory', {})
            regional_approvals = []
            
            # Check structured regional data from drug resolver
            if reg_status.get('ema_approved'):
                regional_approvals.append('EMA')
            if reg_status.get('pmda_approved'):
                regional_approvals.append('PMDA')
            if reg_status.get('health_canada_approved'):
                regional_approvals.append('HC')
            
            # Check LLM-based regional approvals
            llm_regional = candidate.get('regional_approvals', [])
            if llm_regional and isinstance(llm_regional, list):
                for approval in llm_regional:
                    if approval and approval.strip():
                        print(f"    🌍 Regional Status: {approval}")
            elif regional_approvals:
                print(f"    🌍 Regional Approvals: {', '.join(regional_approvals)}")
                if reg_status.get('regional_details'):
                    print(f"       {reg_status['regional_details']}")
            
            # Show shelving reason if available
            shelving_reason = candidate.get('shelving_reason', 'Unknown')
            if shelving_reason != 'Unknown':
                confidence = candidate.get('shelving_confidence', 0.0)
                print(f"    📋 Shelving Reason: {shelving_reason} (Confidence: {confidence:.0%})")
                
                # Show shelving details if available
                details = candidate.get('shelving_details', {})
                if details:
                    if 'safety_issues' in details:
                        print(f"       - Safety: {details['safety_issues']}")
                    if 'efficacy_issues' in details:
                        print(f"       - Efficacy: {details['efficacy_issues']}")
                    if 'commercial_reasons' in details:
                        print(f"       - Commercial: {details['commercial_reasons']}")
                
                # Show sources
                sources = candidate.get('shelving_sources', [])
                if sources:
                    print(f"       - Sources: {', '.join(sources[:2])}")
            
            # Show chemical structure if available
            if show_structures and 'chemical_structure' in candidate:
                structure = candidate['chemical_structure']
                if structure.get('smiles'):
                    print(f"    💊 SMILES: {structure['smiles'][:80]}...")
                if structure.get('inchi_key'):
                    print(f"    🔑 InChI Key: {structure['inchi_key']}")
    
    # Show detailed business context for top assets
    if shelved_assets:
        print(f"\n💼 BUSINESS OPPORTUNITY ANALYSIS")
        print("=" * 80)
        for i, candidate in enumerate(shelved_assets[:3], 1):  # Top 3 detailed analysis
            drug_name = candidate.get('drug_name', candidate.get('drug', 'Unknown'))
            print(f"\n{i}. {drug_name.upper()}")
            print("-" * 50)
            
            # Show specific business context like Vorolanib example
            specific_reason = candidate.get('specific_discontinuation_reason', 'Unknown')
            
            # Get dynamic business context from drug data (show for all drugs)
            regional_status = candidate.get('regional_approvals', [])
            active_development = candidate.get('active_development', [])
            ongoing_trials = candidate.get('ongoing_trials', [])
            
            if specific_reason == 'development_pause':
                print("📋 Development Status: Temporarily Paused")
                print("💡 Acquisition Opportunity: Development can be resumed with proper resources")
                print("🎯 Risk Level: Low - Non-safety discontinuation")
            elif specific_reason == 'sponsor_decision':
                print("📋 Development Status: Strategic Sponsor Decision")
                print("💡 Acquisition Opportunity: Available due to portfolio prioritization")
                print("🎯 Risk Level: Low - Business decision, not efficacy/safety")
            elif specific_reason == 'fda_approved':
                print("📋 Development Status: FDA Approved")
                print("💡 Opportunity: New indication development or lifecycle management")
                print("🎯 Risk Level: Very Low - Proven safety/efficacy profile")
            
            # Show business context for all drugs
            if regional_status:
                for approval in regional_status:
                    print(f"🌍 Regional Status: {approval}")
            
            if active_development:
                for dev in active_development:
                    print(f"👁️  Active Development: {dev}")
            
            if ongoing_trials:
                for trial in ongoing_trials:
                    print(f"🔬 Ongoing Trials: {trial}")
            
            # Show phase and de-risking
            max_phase = candidate.get('max_phase')
            if max_phase is not None:
                try:
                    phase_num = float(max_phase)
                    if phase_num >= 2:
                        print(f"⚡ De-risking: Phase {max_phase} data available - reduced development risk")
                except (ValueError, TypeError):
                    pass
            
            # Show target validation
            target = candidate.get('primary_target', 'Unknown')
            if target != 'Unknown':
                print(f"🎯 Target Validation: {target} - established mechanism")
    
    # Drug Asset Discovery vs Rescue Opportunities
    drug_discovery = results.get('drug_discovery', [])
    drug_rescue = results.get('drug_rescue', [])
    
    # Display FDA-Approved but Shelved Drugs
    all_shelved = results.get('all_shelved_candidates', [])
    if all_shelved:
        print(f"\n🔬 SHELVED/ABANDONED DRUGS (All Clinical Phases)")
        print(f"{'Drug':<25} {'Discontinuation Reason':<35} {'Phase History':<20} {'Academic Score':<15}")
        print("-" * 95)
        
        for candidate in all_shelved[:top_n]:
            drug_name = candidate.get('drug', 'Unknown')[:23]
            disc_reason = candidate.get('discontinuation_reason', 'Business/Marketing')[:33]
            phases = ", ".join(sorted(set(candidate.get('phases', []))))[:18]
            academic_score = candidate.get('academic_score', 0)
            
            print(f"{drug_name:<25} {disc_reason:<35} {phases:<20} {academic_score:<15.1f}")
            
            # Show regulatory details
            if verbose:
                reg_data = candidate.get('regulatory_data', {})
                approval_date = reg_data.get('original_approval_date', 'Unknown')
                print(f"    📅 FDA Approval: {approval_date}")
                print(f"    ❌ Primary Failure: {candidate.get('primary_failure_reason', 'Unknown')}")
                print(f"    🎯 Rescue Potential: {candidate.get('rescue_potential', 'Unknown')}")
    
    # Display Patent-Only Assets
    patent_assets = results.get('patent_only_assets', [])
    if patent_assets:
        print(f"\n🔬 PATENT FILINGS (May include non-drug patents)")
        print(f"{'Applicant':<30} {'Title/Abstract':<70} {'Date':<12}")
        print("-" * 112)
        
        for asset in patent_assets[:5]:
            company = asset.get('company', 'Unknown')[:29]
            date = asset.get('date', 'Unknown')[:11]
            
            # Get patent title or abstract excerpt
            title = ''
            if 'title' in asset:
                title = asset['title'][:69]
            elif 'abstract' in asset:
                title = asset['abstract'][:69] + '...'
            else:
                title = 'No description available'
            
            print(f"{company:<30} {title:<70} {date:<12}")
    
    # Display General Drug Rescue Opportunities (lower scoring candidates)
    if drug_rescue and not shelved_assets:  # Only show if no high potential assets shown
        print(f"\n🚑 OTHER DISCONTINUED DRUG CANDIDATES")
        print(f"{'Drug':<25} {'Status':<35} {'Holder/Sponsor':<25} {'Trials':<8}")
        print("-" * 93)
        
        for candidate in drug_rescue[:top_n]:
            drug_name = candidate.get('drug_name', 'Unknown')[:24]
            owner = candidate.get('current_owner', 'Unknown')[:24]
            trial_count = candidate.get('trial_count', 0)
            
            # Determine status
            status = 'Status unknown'
            if 'ownership_info' in candidate and 'regulatory' in candidate['ownership_info']:
                reg_info = candidate['ownership_info']['regulatory']
                if reg_info.get('withdrawn_for_safety_or_efficacy'):
                    status = 'Withdrawn for safety/efficacy'
                elif not reg_info.get('is_currently_marketed', True):
                    status = 'Not currently marketed'
                else:
                    status = 'Discontinued'
            
            print(f"{drug_name:<25} {status[:35]:<35} {owner:<25} {trial_count:<8}")
    
    # Safety profiles for top candidates
    safety_profiles = results.get('candidate_safety_profiles', [])
    if safety_profiles:
        print(f"\n🛡️ COMPREHENSIVE SAFETY PROFILES")
        print("=" * 80)
        
        # Deduplicate safety profiles by drug name, keeping the one with the earliest approval date
        seen_drugs = {}
        deduplicated_profiles = []
        
        for profile in safety_profiles:
            drug_name = profile['drug_name'].upper()
            
            # If we haven't seen this drug yet, or if this entry has an earlier approval date
            if drug_name not in seen_drugs:
                seen_drugs[drug_name] = profile
                deduplicated_profiles.append(profile)
            else:
                # Compare approval dates and keep the earlier one
                existing_date = seen_drugs[drug_name].get('fda_approval_status', {}).get('original_approval_date', '')
                new_date = profile.get('fda_approval_status', {}).get('original_approval_date', '')
                
                # Convert dates for comparison if needed
                if new_date and existing_date:
                    # Handle different date formats
                    existing_parsed = _parse_approval_date(existing_date)
                    new_parsed = _parse_approval_date(new_date)
                    
                    if new_parsed and existing_parsed and new_parsed < existing_parsed:
                        # Replace with the earlier date entry
                        seen_drugs[drug_name] = profile
                        deduplicated_profiles = [p if p['drug_name'].upper() != drug_name else profile 
                                                for p in deduplicated_profiles]
        
        for i, profile in enumerate(deduplicated_profiles[:3], 1):
            drug_name = profile['drug_name']
            print(f"\n{i}. {drug_name.upper()}")
            print("-" * 50)
            
            # FDA Approval Status
            fda_status = profile.get("fda_approval_status", {})
            if fda_status.get("is_approved"):
                print(f"    💊 FDA Status: Approved")
                
                # Show Orange Book data if available (authoritative source)
                if fda_status.get("orange_book_status"):
                    print(f"       📘 Orange Book Data:")
                    if fda_status.get("original_approval_date"):
                        formatted_date = _format_approval_date(fda_status['original_approval_date'])
                        print(f"          - Original Approval: {formatted_date}")
                    if fda_status.get("original_brand"):
                        print(f"          - Original Brand: {fda_status['original_brand']}")
                    if fda_status.get("original_nda"):
                        print(f"          - Original NDA: {fda_status['original_nda']}")
                    print(f"          - Current Status: {fda_status['orange_book_status']}")
                    
                    # Show active/discontinued products
                    active = fda_status.get("active_products", [])
                    discontinued = fda_status.get("discontinued_products", [])
                    
                    if active:
                        print(f"          - Active Products: {len(active)}")
                        # Show first few active products
                        for i, prod in enumerate(active[:2]):
                            # Check if prod is a dict before accessing properties
                            if isinstance(prod, dict):
                                print(f"            • {prod.get('name', 'Unknown')} ({prod.get('type', 'N/A')}), {prod.get('applicant', 'N/A')}")
                            else:
                                print(f"            • {prod}")
                        if len(active) > 2:
                            print(f"            • ... and {len(active) - 2} more")
                    
                    if discontinued:
                        print(f"          - Discontinued Products: {len(discontinued)}")
                        # Check for Federal Register notes
                        fed_notes = [p for p in discontinued if isinstance(p, dict) and p.get('federal_register_note')]
                        if fed_notes:
                            for prod in fed_notes:
                                print(f"            ⚠️  {prod.get('name', 'Unknown')}: {prod.get('federal_register_note', '')}")
                
                # Show FDA API data if available (current formulations)
                if fda_status.get("approval_details") and "FDA Drugs@FDA" in fda_status.get("approval_sources", []):
                    print(f"       💊 FDA Drugs@FDA Current Formulations:")
                    # Group by marketing status
                    active_forms = [d for d in fda_status["approval_details"] if d.get("marketing_status") == "Prescription"]
                    discontinued_forms = [d for d in fda_status["approval_details"] if d.get("marketing_status") == "Discontinued"]
                    
                    if active_forms:
                        print(f"          - Active: {len(active_forms)} formulations")
                        for form in active_forms[:2]:
                            print(f"            • {form.get('brand_name', 'Generic')} - {form.get('dosage_form', 'N/A')}, {form.get('strength', 'N/A')}")
                            approval_date = _format_approval_date(form.get('approval_date', 'N/A'))
                            print(f"              Approval: {approval_date}, NDA: {form.get('application_number', 'N/A')}")
                    
                    if discontinued_forms:
                        print(f"          - Discontinued Formulations: {len(discontinued_forms)}")
                    
            else:
                print(f"    💊 FDA Status: Not FDA Approved (Experimental/Investigational)")
            
            # Adverse events
            ae_data = profile.get('fda_adverse_events', {})
            ae_list = ae_data.get('top_adverse_events', [])
            if ae_list and ae_list[0] != "Would be retrieved from FDA OpenFDA API":
                print(f"⚠️ Top Adverse Events:")
                for event in ae_list[:5]:  # Show top 5
                    if isinstance(event, dict):
                        # Format: reaction name (percentage%)
                        reaction = event.get('reaction', 'Unknown')
                        percentage = event.get('percentage', 0)
                        print(f"  • {reaction} ({percentage}%)")
                    else:
                        # Fallback for string format
                        print(f"  • {event}")
            
            # Drug interactions
            interactions = profile.get('drug_interactions', [])
            if interactions and interactions[0] != "Would be retrieved from RxNav API":
                print(f"💊 Major Drug Interactions:")
                for interaction in interactions[:2]:
                    print(f"  • {interaction}")
            
            # Contraindications
            contras = profile.get('contraindications', [])
            if contras and contras[0] != "Would be retrieved from FDA drug labels":
                print(f"🚫 Key Contraindications:")
                for contra in contras[:2]:
                    print(f"  • {contra}")
            
            # Mechanism
            mechanism = profile.get('mechanism_summary', '')
            if mechanism and not mechanism.startswith("Would be retrieved"):
                print(f"🧬 Mechanism: {mechanism}")
            
            # ChEMBL ID if available
            if profile.get('chembl_id'):
                print(f"🔗 ChEMBL ID: {profile['chembl_id']}")
    
    # Alternative Targets - only show if we have real data
    alternatives = results.get('alternative_targets', [])[:top_n]
    if alternatives:
        print(f"\n🎯 ALTERNATIVE THERAPEUTIC TARGETS")
        for i, target in enumerate(alternatives, 1):
            print(f"\n{i}. {target['target']} (Confidence: {target.get('confidence', 0):.0%})")
            print(f"   Rationale: {target.get('rationale', 'No rationale provided')}")
            
            # Target metrics
            if 'inhibitor_count' in target:
                print(f"   • Inhibitors: {target['inhibitor_count']}")
                if target.get('most_potent_ic50'):
                    print(f"   • Most Potent IC50: {target['most_potent_ic50']:.1f} nM")
                
                # Show most advanced compound
                compound = alt.get('most_advanced_compound', {})
                if compound.get('chembl_id'):
                    phase = compound.get('phase', 'N/A')
                    phase_str = f"Phase {phase}" if phase else "Preclinical"
                    print(f"   • Most Advanced: {compound['chembl_id']} ({phase_str})")
                
                trials = alt.get('clinical_trials', {})
                if trials:
                    print(f"   • Clinical Trials: {trials['total']} total "
                          f"({trials['recruiting']} recruiting, "
                          f"{trials['completed']} completed)")
                
                print(f"   • Development Score: {alt.get('development_score', 0):.2f}/1.00")
                
                # Show PDB structures if available
                pdb_structures = alt.get('pdb_structures', [])
                if pdb_structures:
                    print(f"   • PDB Structures: {len(pdb_structures)} available")
                    for pdb in pdb_structures[:2]:  # Show first 2
                        print(f"     - {pdb.get('pdb_id', 'N/A')}: {pdb.get('title', 'N/A')[:60]}...")
                        if pdb.get('resolution') != 'N/A':
                            print(f"       Resolution: {pdb.get('resolution')} Å, Method: {pdb.get('method', 'N/A')}")
    
    print("\n" + "="*80)

async def main():
    parser = create_parser()
    args = parser.parse_args()
    
    print("🚀 Initializing Omics Oracle Drug Asset Discovery Agent...")
    
    try:
        # Get API key from environment
        api_key = os.environ.get('OPENAI_API_KEY')
        if not api_key:
            print("⚠️ Warning: OPENAI_API_KEY not set. Ownership enrichment will be limited.")
        
        agent = DrugAssetDiscoveryAgent(llm_api_key=api_key)
        
        # Clear cache to ensure fresh results every run
        agent.clear_cache()
        
        print(f"🔍 Analyzing {args.disease} failures" + 
              (f" for target {args.target}" if args.target else "") + "...")
        
        # Run analysis
        results = await agent.analyze_disease_failures(
            disease=args.disease,
            target=args.target,
            max_trials=args.max_trials,
            skip_safety_profiles=args.skip_safety
        )
        
        # If shelved-only flag is set, filter to only discontinued drugs
        if args.shelved_only:
            print("🔬 Filtering to show only shelved/discontinued drugs...")
            
            # Use the new shelved drug method
            shelved_candidates = await agent.ct_client.get_shelved_drug_candidates(
                disease=args.disease,
                include_only_discontinued=True,
                min_failure_ratio=0.5
            )
            
            # Replace candidates with shelved ones
            results['discovery_candidates'] = shelved_candidates[:10]
            results['total_discovery_candidates'] = len(shelved_candidates)
        
        # Display results
        print("\n" + "="*100)
        print(f"🎯 DRUG ASSET DISCOVERY RESULTS FOR {args.disease.upper()}")
        print("="*100)
        
        # Separate clinical and preclinical candidates
        clinical_candidates = []
        preclinical_candidates = []
        
        # Collect all candidates and separate by type
        all_sources = ['drug_rescue', 'drug_discovery', 'shelved_assets', 'high_potential_assets']
        for source in all_sources:
            if results.get(source):
                for candidate in results[source]:
                    if (candidate.get("development_stage") == "preclinical" or 
                        candidate.get("source") == "llm_web_search_preclinical"):
                        preclinical_candidates.append(candidate)
                    else:
                        clinical_candidates.append(candidate)
        
        # Display clinical candidates (existing format)
        if clinical_candidates:
            # Precompute which drugs are approved for the query disease using ChEMBL
            approved_for_disease_names = set()
            try:
                chembl_matches = await agent.chembl_client.search_by_indication(args.disease)
                for m in chembl_matches or []:
                    # Add preferred name
                    name = (m.get('molecule_pref_name') or m.get('parent_molecule_name') or '').strip()
                    if name:
                        approved_for_disease_names.add(name.lower())
                    # Add synonyms if any
                    for syn in m.get('molecule_synonyms', []) or []:
                        if syn:
                            approved_for_disease_names.add(str(syn).lower())
            except Exception:
                approved_for_disease_names = set()
            
            print(f"\n📊 HIGH POTENTIAL CANDIDATES ({len(clinical_candidates)} total)")
            print("─" * 100)
            print(f"{'Drug':<25} {'Target':<30} {'Approved For':<30} {'Ownership History':<70}")
            print("─" * 155)
            
            # Show top candidates (limited by args.top, default 5)
            displayed_count = min(len(clinical_candidates), args.top)
            for drug in clinical_candidates[:args.top]:
                drug_name = drug.get('drug_name', 'Unknown')[:24]
                # Show multiple targets if available
                targets = []
                
                # Get primary target
                primary_target = drug.get('primary_target', '')
                if primary_target and primary_target != 'Unknown':
                    targets.append(primary_target)
                
                # Get secondary targets
                secondary_targets = drug.get('secondary_targets', []) or []
                if isinstance(secondary_targets, list):
                    targets.extend([t for t in secondary_targets if t and t != primary_target])
                
                # Get all_targets as fallback
                all_targets = drug.get('all_targets', []) or []
                if isinstance(all_targets, list) and not targets:
                    targets.extend([t for t in all_targets if t])
                
                # Format target display
                if targets:
                    target = ', '.join(targets)[:39]
                else:
                    target = 'Unknown'
                
                # Build detailed regional approvals with indication context
                regional_approvals = 'Not approved'
                try:
                    detailed_regions = []
                    regional = drug.get('regional_approvals') or {}
                    # Fallback to simple display if dict not present
                    if isinstance(regional, dict) and regional:
                        # Determine if this drug matches the query disease by name
                        dn_lower = (drug.get('drug_name') or '').lower()
                        also_names = [
                            drug.get('generic_name'),
                            drug.get('molecule_pref_name'),
                            drug.get('compound_name')
                        ]
                        match_for_disease = (
                            dn_lower in approved_for_disease_names or
                            any((n or '').lower() in approved_for_disease_names for n in also_names)
                        )
                        region_labels = [
                            ('fda', 'FDA'),
                            ('ema', 'EMA'),
                            ('pmda', 'PMDA'),
                            ('nmpa', 'NMPA'),
                            ('health_canada', 'HC'),
                            ('cdsco', 'CDSCO'),
                            ('dcgi', 'DCGI'),
                        ]
                        # Try to find indications dicts either inside regional or at top-level
                        indications_map = {}
                        if isinstance(regional.get('indications'), dict):
                            indications_map = regional.get('indications') or {}
                        elif isinstance(drug.get('regional_indications'), dict):
                            indications_map = drug.get('regional_indications') or {}
                        
                        # Also check if indications are stored directly in each region's data
                        for key, _ in [('fda', 'FDA'), ('ema', 'EMA'), ('pmda', 'PMDA'), ('nmpa', 'NMPA'), ('health_canada', 'HC'), ('cdsco', 'CDSCO'), ('dcgi', 'DCGI')]:
                            region_data = regional.get(key)
                            if isinstance(region_data, dict) and region_data.get('indications'):
                                if key not in indications_map:
                                    indications_map[key] = region_data.get('indications', [])
                        
                        for key, label in region_labels:
                            region_data = regional.get(key)
                            if region_data:
                                # Check if actually approved (handle both dict and boolean formats)
                                is_approved = False
                                if isinstance(region_data, dict):
                                    is_approved = region_data.get('approved', False)
                                elif region_data is True:
                                    is_approved = True
                                
                                if is_approved:
                                    # 1) Prefer explicit indications if provided (any disease)
                                    inds = []
                                    if isinstance(indications_map.get(key), list):
                                        inds = [str(i) for i in indications_map.get(key) if i]
                                    if inds:
                                        # Abbreviate common disease terms
                                        abbreviated_inds = []
                                        for ind in inds[:2]:
                                            abbrev = _abbreviate_indication(ind)
                                            abbreviated_inds.append(abbrev)
                                        shown = "; ".join(abbreviated_inds)
                                        more = f"; +{len(inds)-2}" if len(inds) > 2 else ""
                                        detailed_regions.append(f"{label}({shown}{more})")
                                        continue
                                    # 2) If no indications available but the drug matches the query disease, show that
                                    if match_for_disease:
                                        abbrev_disease = _abbreviate_indication(args.disease)
                                        detailed_regions.append(f"{label}({abbrev_disease})")
                                    else:
                                        # 3) Fallback when nothing else known
                                        detailed_regions.append(f"{label}(other)")
                        regional_approvals = ', '.join(detailed_regions) if detailed_regions else 'Not approved'
                    else:
                        # Fallback to precomputed display string
                        regional_approvals = (drug.get('regional_approvals_display') or 'Unknown')
                except Exception:
                    regional_approvals = (drug.get('regional_approvals_display') or 'Unknown')
                
                # Display ownership chain if available, otherwise show current owner
                ownership_chain = drug.get('ownership_chain', '')
                # Handle both string and list types for ownership_chain
                if isinstance(ownership_chain, list):
                    # Extract string values from list items (handle dicts in the list)
                    chain_strings = []
                    for item in ownership_chain:
                        if isinstance(item, dict):
                            # Try to get a name or company field from the dict
                            chain_strings.append(item.get('name', item.get('company', str(item))))
                        else:
                            chain_strings.append(str(item))
                    ownership_chain = ' → '.join(chain_strings) if chain_strings else ''
                elif not isinstance(ownership_chain, str):
                    ownership_chain = str(ownership_chain) if ownership_chain else ''
                
                if ownership_chain and ownership_chain.strip() and ownership_chain.lower() != "unknown":
                    owner = ownership_chain[:80]  # Show full chain, allow more space for history
                else:
                    # Handle multiple owners for conflicted drugs
                    owner = drug.get('current_owner', drug.get('sponsor', 'Unknown'))
                    alternative_owners = drug.get('alternative_owners', [])
                    if alternative_owners and len(alternative_owners) > 0:
                        owner = f"{owner[:12]}, {alternative_owners[0][:12]}"
                    else:
                        owner = owner[:24]
                
                print(f"{drug_name:<25} {target:<20} {regional_approvals:<30} {owner:<80}")
            
            # Explain why only limited number shown
            if len(clinical_candidates) > args.top:
                print(f"\n💡 Showing top {displayed_count} candidates (use --top {len(clinical_candidates)} to see all {len(clinical_candidates)} candidates)")
        
        # Display preclinical candidates (new section)
        if preclinical_candidates:
            print(f"\n🧪 POTENTIAL PRECLINICAL ASSETS ({len(preclinical_candidates)} total)")
            print("─" * 100)
            
            # Show top preclinical candidates with full details
            displayed_preclinical = min(len(preclinical_candidates), args.top)
            for drug in preclinical_candidates[:args.top]:
                drug_name = drug.get('drug_name', drug.get('compound_name', 'Unknown'))
                target = drug.get('primary_target', drug.get('target', 'Unknown'))
                mechanism = drug.get('mechanism_of_action', drug.get('mechanism', 'Unknown'))
                
                print(f"\n💊 {drug_name}")
                print(f"   Target: {target}")
                print(f"   Mechanism: {mechanism}")
            
            if len(preclinical_candidates) > args.top:
                print(f"\n💡 Showing top {displayed_preclinical} preclinical assets (use --top {len(preclinical_candidates)} to see all)")
        
        # Save results to JSON if requested
        if args.output:
            output_path = Path(args.output)
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\n💾 Results saved to: {output_path}")
        
        # Generate PDF report if requested
        if args.pdf:
            print(f"\n📄 Generating comprehensive academic PDF report...")
            try:
                from pdf_report_generator import generate_comprehensive_disease_pdf
                from pathlib import Path
                
                # Create results directory
                results_dir = Path("results")
                results_dir.mkdir(exist_ok=True)
                
                # Collect all drug candidates for comprehensive report
                all_drugs = []
                if results.get('drug_rescue'):
                    all_drugs.extend(results['drug_rescue'])
                if results.get('drug_discovery'):
                    all_drugs.extend(results['drug_discovery'])
                
                if all_drugs:
                    # Generate comprehensive PDF with all drugs
                    openai_api_key = os.environ.get('OPENAI_API_KEY')
                    from pdf_report_generator import _generate_comprehensive_disease_pdf_async
                    from pathlib import Path
                    
                    # Create results directory
                    results_dir = Path("results")
                    results_dir.mkdir(exist_ok=True)
                    
                    # Generate filename if not provided
                    if args.pdf == True:
                        safe_disease_name = "".join(c for c in args.disease if c.isalnum() or c in (' ', '-', '_')).rstrip()
                        safe_disease_name = safe_disease_name.replace(' ', '_').lower()
                        output_pdf = f"{safe_disease_name}_comprehensive_report.pdf"
                    else:
                        output_pdf = args.pdf
                    
                    # Ensure PDF is saved in results folder
                    if not str(output_pdf).startswith('results/'):
                        output_pdf = results_dir / output_pdf
                    
                    pdf_path = await _generate_comprehensive_disease_pdf_async(
                        disease_name=args.disease,
                        drug_candidates=all_drugs,
                        output_pdf=str(output_pdf),
                        openai_api_key=openai_api_key
                    )
                    print(f"✅ Comprehensive academic PDF report generated: {pdf_path}")
                    print(f"📊 Report includes {len(all_drugs)} drug candidates with detailed analysis")
                else:
                    print("❌ No drug candidates found to generate PDF report")
                    
            except Exception as e:
                print(f"❌ PDF generation failed: {e}")
                import traceback
                traceback.print_exc()
    
    except Exception as e:
        print(f"\n Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    asyncio.run(main())

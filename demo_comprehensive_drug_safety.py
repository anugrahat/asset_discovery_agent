#!/usr/bin/env python3

import asyncio
import json
from thera_agent.data.drug_safety_client import DrugSafetyClient

async def demo_comprehensive_drug_safety():
    """Demonstrate comprehensive drug safety data retrieval"""
    
    # Initialize client
    safety_client = DrugSafetyClient()
    
    # Test drugs with known safety issues
    test_drugs = [
        "warfarin",      # Known for interactions
        "acetaminophen", # Black box warning
        "aspirin"        # Well-documented
    ]
    
    for drug_name in test_drugs:
        print(f"\n{'='*80}")
        print(f"🔍 COMPREHENSIVE SAFETY PROFILE: {drug_name.upper()}")
        print('='*80)
        
        # Get comprehensive profile
        profile = await safety_client.get_comprehensive_safety_profile(drug_name)
        
        # 1. FDA Adverse Events
        print(f"\n⚠️ FDA ADVERSE EVENTS REPORTS")
        ae_data = profile["fda_adverse_events"]
        print(f"Total Reports: {ae_data['total_reports']}")
        
        if ae_data["top_adverse_events"]:
            print("\nTop Adverse Events:")
            for i, event in enumerate(ae_data["top_adverse_events"][:5], 1):
                print(f"  {i}. {event['reaction']} ({event['percentage']}% of reports)")
        
        # 2. Black Box Warnings
        print(f"\n🚨 BLACK BOX & SERIOUS WARNINGS")
        warnings = profile["black_box_warnings"]
        if warnings:
            for warning in warnings:
                print(f"• {warning['type']}: {warning['warning'][:100]}...")
        else:
            print("• No black box warnings found")
        
        # 3. Drug Interactions
        print(f"\n💊 DRUG INTERACTIONS")
        interactions = profile["drug_interactions"]
        if interactions:
            for i, interaction in enumerate(interactions[:3], 1):
                print(f"  {i}. {interaction['interacting_drug']} - {interaction['severity']}")
                print(f"     {interaction['description'][:80]}...")
        else:
            print("• No interactions found in database")
        
        # 4. Contraindications
        print(f"\n🚫 CONTRAINDICATIONS")
        contras = profile["contraindications"]
        if contras:
            for contra in contras[:3]:
                print(f"• {contra['condition'][:100]}...")
        else:
            print("• No contraindications found")
        
        # 5. Pharmacology
        print(f"\n🧬 PHARMACOLOGY")
        pharm = profile["pharmacology"]
        if pharm["mechanism_of_action"]:
            print(f"Mechanism: {pharm['mechanism_of_action'][0][:100]}...")
        if pharm["pharmacokinetics"]:
            print(f"Pharmacokinetics: {pharm['pharmacokinetics'][0][:100]}...")
        
        # 6. Allergies & Cross-Sensitivity
        print(f"\n🤧 ALLERGIES & CROSS-SENSITIVITY")
        allergy_data = profile["allergies_cross_sensitivity"]
        if allergy_data["allergic_reactions"]:
            for reaction in allergy_data["allergic_reactions"][:3]:
                print(f"• {reaction['reaction']} ({reaction['frequency']})")
        else:
            print("• No specific allergic reactions identified")
        
        # 7. FDA Drug Labels Summary
        print(f"\n📋 FDA DRUG LABEL INFO")
        labels = profile["fda_drug_labels"]
        if labels["brand_names"]:
            print(f"Brand Names: {', '.join(labels['brand_names'][:3])}")
        if labels["manufacturer"]:
            print(f"Manufacturers: {', '.join(labels['manufacturer'][:2])}")
        
        print(f"\n" + "-"*50)
        print(f"Data Sources: FDA OpenFDA, RxNav/NLM, FDA Drug Labels")

async def demo_clinical_trial_safety():
    """Demonstrate clinical trial-specific safety data"""
    
    safety_client = DrugSafetyClient()
    
    # Example NCT IDs (you can replace with actual IDs from your analysis)
    test_nct_ids = ["NCT02576665", "NCT05788926"]
    
    print(f"\n{'='*80}")
    print(f"🏥 CLINICAL TRIAL SAFETY DATA")
    print('='*80)
    
    for nct_id in test_nct_ids:
        print(f"\n📊 Trial: {nct_id}")
        
        safety_summary = await safety_client.get_clinical_trial_safety_summary(nct_id)
        
        if safety_summary.get("data_available", True):
            print(f"Serious Adverse Events: {len(safety_summary.get('serious_adverse_events', {}))}")
            print(f"Other Adverse Events: {len(safety_summary.get('other_adverse_events', {}))}")
            print(f"Deaths Reported: {len(safety_summary.get('deaths', {}))}")
        else:
            print("• No detailed adverse event data available")

if __name__ == "__main__":
    print("🚀 Starting Comprehensive Drug Safety Analysis...")
    
    print("\n🔍 PART 1: Multi-API Drug Safety Profiles")
    asyncio.run(demo_comprehensive_drug_safety())
    
    print("\n🏥 PART 2: Clinical Trial Safety Data")
    asyncio.run(demo_clinical_trial_safety())

#!/usr/bin/env python3
"""Check if FDA-approved MS drugs are being filtered correctly"""

import asyncio
import sys
sys.path.append('/home/anugraha/agent3')

from thera_agent.drug_asset_discovery_agent import DrugAssetDiscoveryAgent

async def check_ms_drugs():
    print("Checking FDA Filtering for MS Drugs")
    print("=" * 60)
    
    # Initialize agent
    agent = DrugAssetDiscoveryAgent()
    
    # Test known MS drugs
    test_drugs = [
        ("Fampridine-SR", "FDA approved 2010 as Ampyra"),
        ("Daclizumab", "FDA approved 2016, withdrawn 2018"),
        ("Fingolimod", "FDA approved 2010 as Gilenya"),
        ("Natalizumab", "FDA approved 2004 as Tysabri"),
        ("Ocrelizumab", "FDA approved 2017 as Ocrevus"),
        ("Idebenone", "Not FDA approved for MS"),
        ("ABT-555", "Not FDA approved")
    ]
    
    # Create mock candidates
    candidates = []
    for drug, status in test_drugs:
        candidates.append({
            "drug": drug,
            "score": 50,
            "total_trials": 10,
            "failed": 5,
            "phases": ["PHASE2", "PHASE3"],
            "max_phase": 3,
            "trial_status": "TERMINATED",
            "_status": status  # For display
        })
    
    print("\nTesting drugs:")
    for c in candidates:
        print(f"  {c['drug']}: {c['_status']}")
    
    # Enrich candidates
    enriched = await agent._enrich_candidates_with_chembl(candidates)
    
    print("\n\nFDA STATUS AFTER ENRICHMENT:")
    print("-" * 60)
    for c in enriched:
        print(f"{c['drug']:20} FDA Approved: {c.get('fda_approved', False):5} | Marketed: {c.get('currently_marketed', False)}")
    
    # Test categorization
    categorized = await agent._categorize_drug_opportunities(enriched, "multiple sclerosis")
    
    print("\n\nCATEGORIZATION RESULTS:")
    print("-" * 60)
    print(f"✅ Shelved Assets ({len(categorized['shelved_assets'])}):")
    for asset in categorized['shelved_assets']:
        print(f"   - {asset['drug']}")
    
    print(f"\n❌ Filtered Out ({len(categorized.get('filtered_out', []))}):")
    for filtered in categorized.get('filtered_out', []):
        print(f"   - {filtered['drug']:20} Reason: {filtered.get('filtered_reason', 'unknown')}")

if __name__ == "__main__":
    asyncio.run(check_ms_drugs())

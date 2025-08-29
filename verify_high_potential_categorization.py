#!/usr/bin/env python3
"""
Verify that high potential assets are correctly categorized and receive safety profiles
"""
import asyncio
import json
from thera_agent.drug_asset_discovery_agent import DrugAssetDiscoveryAgent

async def verify_categorization():
    """Verify high potential asset categorization and safety profile assignment"""
    
    agent = DrugAssetDiscoveryAgent()
    
    print("🔬 Verifying High Potential Asset Categorization")
    print("=" * 80)
    
    # Run the discovery
    results = await agent.analyze_disease_failures(
        disease="lung cancer",
        max_trials=20
    )
    
    # Check high potential assets
    high_potential = results.get("high_potential_assets", [])
    safety_profiles = results.get("candidate_safety_profiles", [])
    
    print(f"\n📊 Found {len(high_potential)} high potential assets:")
    for drug in high_potential:
        print(f"  - {drug.get('drug_name')} (Score: {drug.get('high_potential_score', 0):.2f})")
        
    print(f"\n💊 Safety profiles generated for {len(safety_profiles)} drugs:")
    for profile in safety_profiles:
        print(f"  - {profile.get('drug_name')}")
        
    # Verify specific drugs
    print("\n🎯 Verification Results:")
    print("-" * 40)
    
    # Check if Vorolanib and PF-07284892 are in high potential
    vorolanib_found = any(d.get('drug_name', '').lower() == 'vorolanib' for d in high_potential)
    pf_found = any('pf-07284892' in d.get('drug_name', '').lower() for d in high_potential)
    
    print(f"Vorolanib in high potential assets: {'✅ YES' if vorolanib_found else '❌ NO'}")
    print(f"PF-07284892 in high potential assets: {'✅ YES' if pf_found else '❌ NO'}")
    
    # Check if they have safety profiles
    vorolanib_profile = any(p.get('drug_name', '').lower() == 'vorolanib' for p in safety_profiles)
    pf_profile = any('pf-07284892' in p.get('drug_name', '').lower() for p in safety_profiles)
    
    print(f"Vorolanib has safety profile: {'✅ YES' if vorolanib_profile else '❌ NO'}")
    print(f"PF-07284892 has safety profile: {'✅ YES' if pf_profile else '❌ NO'}")
    
    # Show filtered out drugs
    filtered_out = results.get("filtered_out", [])
    print(f"\n🚫 Correctly filtered out {len(filtered_out)} active/approved drugs")
    
    return results

if __name__ == "__main__":
    asyncio.run(verify_categorization())

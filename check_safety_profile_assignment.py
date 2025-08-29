#!/usr/bin/env python3
"""
Check that safety profiles are correctly assigned to high potential assets
"""
import json

# Run the discovery and save results
print("Running lung cancer discovery...")
import subprocess
result = subprocess.run(
    ["python", "-c", """
import asyncio
import json
from thera_agent.drug_asset_discovery_agent import DrugAssetDiscoveryAgent

async def run():
    agent = DrugAssetDiscoveryAgent()
    # Mock the LLM calls to avoid API key requirement
    agent._llm_query = lambda x: {'failure_patterns': {}, 'key_insights': 'mocked'}
    agent._analyze_failures_with_llm = lambda x, y: {'failure_patterns': {}, 'key_insights': 'mocked'} 
    agent._analyze_side_effects = lambda x: {'common_aes': [], 'serious_aes': []}
    
    results = await agent.analyze_disease_failures('lung cancer', max_trials=10)
    return results

results = asyncio.run(run())
print(json.dumps(results, indent=2))
"""],
    capture_output=True,
    text=True,
    cwd="/home/anugraha/agent3"
)

if result.returncode == 0:
    results = json.loads(result.stdout.split('\n')[-1])
    
    print("\n" + "="*80)
    print("SAFETY PROFILE ASSIGNMENT CHECK")
    print("="*80)
    
    # Check high potential assets
    high_potential = results.get("high_potential_assets", [])
    print(f"\n🎯 High Potential Assets: {len(high_potential)}")
    for drug in high_potential[:5]:
        print(f"  - {drug.get('drug_name')} (Score: {drug.get('high_potential_score', 0):.2f})")
    
    # Check safety profiles
    safety_profiles = results.get("candidate_safety_profiles", [])
    print(f"\n💊 Safety Profiles Generated For: {len(safety_profiles)}")
    for profile in safety_profiles:
        drug_name = profile.get('drug_name', 'Unknown')
        print(f"  - {drug_name}")
        
        # Check if this drug is in high potential assets
        is_high_potential = any(
            drug.get('drug_name', '').lower() == drug_name.lower() 
            for drug in high_potential
        )
        print(f"    {'✅' if is_high_potential else '❌'} {'In high potential assets' if is_high_potential else 'NOT in high potential assets'}")
    
    # Summary
    print("\n" + "-"*40)
    all_correct = all(
        any(drug.get('drug_name', '').lower() == profile.get('drug_name', '').lower() 
            for drug in high_potential)
        for profile in safety_profiles
    )
    
    if all_correct and len(safety_profiles) > 0:
        print("✅ ALL safety profiles correctly assigned to high potential assets!")
    elif len(safety_profiles) == 0:
        print("⚠️  No safety profiles generated")
    else:
        print("❌ Some safety profiles assigned to wrong drugs!")
        
else:
    print(f"Error running discovery: {result.stderr}")

#!/usr/bin/env python3
"""
Simple test to verify safety profiles are assigned to high potential assets
"""
from thera_agent.drug_asset_discovery_agent import DrugAssetDiscoveryAgent

# Test the categorization and scoring
agent = DrugAssetDiscoveryAgent()

# Test drugs with their expected categorization
test_drugs = [
    {
        "drug_name": "Vorolanib",
        "max_phase": 1,
        "discontinuation_info": {
            "reason": "Strategic decision",
            "details": "Development paused, pivoted to ophthalmic formulation"
        },
        "ownership_info": {
            "clinical_trials": {"latest_activity": "2023-01-01"},
            "ownership_history": [{"company": "Xcovery"}]
        },
        "program_status": "development_discontinued",
        "is_fda_approved": False,
        "is_currently_marketed": False
    },
    {
        "drug_name": "PF-07284892",
        "max_phase": 1, 
        "discontinuation_info": {
            "reason": "Strategic reasons",
            "details": "Trial terminated due to strategic portfolio decisions"
        },
        "ownership_info": {
            "clinical_trials": {"latest_activity": "2024-01-01"},
            "ownership_history": [{"company": "Pfizer"}]
        },
        "program_status": "development_discontinued",
        "is_fda_approved": False,
        "is_currently_marketed": False
    },
    {
        "drug_name": "Pembrolizumab",
        "max_phase": 4,
        "discontinuation_info": {},
        "ownership_info": {},
        "program_status": "approved",
        "is_fda_approved": True,
        "is_currently_marketed": True
    }
]

print("🎯 Testing Drug Categorization and Safety Profile Assignment")
print("=" * 80)

# Categorize drugs
high_potential_drugs = []
filtered_out_drugs = []

for drug in test_drugs:
    # Check if should be filtered out
    if drug["is_fda_approved"] or drug["is_currently_marketed"] or drug["program_status"] in ["approved", "marketed"]:
        filtered_out_drugs.append(drug)
        print(f"\n❌ {drug['drug_name']}: FILTERED OUT (approved/marketed)")
    else:
        # Calculate high potential score
        score = agent._calculate_high_potential_score(drug)
        drug['high_potential_score'] = score
        
        if score >= 0.5:  # Threshold for high potential
            high_potential_drugs.append(drug)
            print(f"\n✅ {drug['drug_name']}: HIGH POTENTIAL ASSET")
            print(f"   Score: {score:.2f}")
            print(f"   Reason: {drug['discontinuation_info'].get('reason', 'Unknown')}")
        else:
            print(f"\n❌ {drug['drug_name']}: LOW POTENTIAL")
            print(f"   Score: {score:.2f}")

print("\n" + "-" * 80)
print(f"High Potential Assets: {len(high_potential_drugs)}")
for drug in high_potential_drugs:
    print(f"  - {drug['drug_name']} (Score: {drug['high_potential_score']:.2f})")

print(f"\nFiltered Out (Active/Approved): {len(filtered_out_drugs)}")
for drug in filtered_out_drugs:
    print(f"  - {drug['drug_name']}")

print("\n💊 Safety profiles should be generated ONLY for high potential assets:")
print("  ✅ Vorolanib - strategic pause/pivot")
print("  ✅ PF-07284892 - strategic decision")
print("  ❌ Pembrolizumab - active/approved (no safety profile needed)")

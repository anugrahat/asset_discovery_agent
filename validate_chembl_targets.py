#!/usr/bin/env python3
"""
Validate ChEMBL target fetching accuracy for known drug-target pairs
"""
import asyncio
from thera_agent.data.chembl_client import ChEMBLClient

async def validate_drug_targets():
    """Validate that known drugs return their expected targets"""
    client = ChEMBLClient()
    
    # Known drug-target relationships
    test_cases = [
        {
            "drug": "atezolizumab", 
            "chembl_id": "CHEMBL1201088",
            "expected_target": "PD-L1",
            "target_keywords": ["PD-L1", "CD274", "B7-H1", "Programmed cell death 1 ligand 1"]
        },
        {
            "drug": "sotorasib",
            "chembl_id": "CHEMBL4594429",  # Correct ChEMBL ID for sotorasib
            "expected_target": "KRAS G12C",
            "target_keywords": ["KRAS", "GTPase KRas", "V-Ki-ras2"]
        },
        {
            "drug": "irinotecan",
            "chembl_id": "CHEMBL1022",  # Correct ChEMBL ID for irinotecan
            "expected_target": "Topoisomerase I",
            "target_keywords": ["Topoisomerase", "TOP1", "DNA topoisomerase I"]
        },
        {
            "drug": "nintedanib",
            "chembl_id": "CHEMBL502835",
            "expected_target": "VEGFR/FGFR/PDGFR",
            "target_keywords": ["VEGFR", "FGFR", "PDGFR", "Vascular endothelial growth factor receptor"]
        }
    ]
    
    print("Validating Known Drug-Target Relationships")
    print("=" * 80)
    
    for test in test_cases:
        print(f"\n🧪 Testing {test['drug']} ({test['chembl_id']})")
        print(f"   Expected: {test['expected_target']}")
        print("-" * 60)
        
        # Get compound info first
        compound = await client.get_compound_by_chembl_id(test['chembl_id'])
        if compound:
            print(f"   ✓ Found compound: {compound.get('pref_name', 'N/A')}")
            print(f"   Max phase: {compound.get('max_phase', 'N/A')}")
        else:
            print(f"   ✗ Compound not found!")
            continue
        
        # Get targets
        targets = await client.get_compound_targets(test['chembl_id'], limit=10)
        
        if targets:
            print(f"\n   Found {len(targets)} targets:")
            
            # Check if expected target is found
            found_expected = False
            for i, target in enumerate(targets, 1):
                target_name = target.get('target_name', '').upper()
                
                # Check if any expected keyword is in the target name
                is_expected = any(keyword.upper() in target_name for keyword in test['target_keywords'])
                
                marker = "✓" if is_expected else " "
                print(f"   {marker} {i}. {target['target_name']}")
                print(f"        Activity: {target['activity_type']} = {target['activity_value']} {target['activity_units']}")
                
                if is_expected:
                    found_expected = True
            
            if found_expected:
                print(f"\n   ✅ SUCCESS: Found expected target!")
            else:
                print(f"\n   ❌ FAILURE: Expected target not found!")
        else:
            print("   ❌ No targets found!")
    
    # Test drug mechanisms as alternative
    print("\n" + "=" * 80)
    print("Testing Drug Mechanisms (Alternative to Targets)")
    print("=" * 80)
    
    for test in test_cases[:2]:  # Test first two drugs
        print(f"\n🔧 Mechanisms for {test['drug']} ({test['chembl_id']})")
        mechanisms = await client.get_drug_mechanisms(test['chembl_id'])
        
        if mechanisms:
            for mech in mechanisms:
                print(f"   - {mech['mechanism_of_action']}")
                print(f"     Target: {mech['target_chembl_id']}")
                print(f"     Action: {mech['action_type']}")
        else:
            print("   No mechanisms found")

if __name__ == "__main__":
    asyncio.run(validate_drug_targets())

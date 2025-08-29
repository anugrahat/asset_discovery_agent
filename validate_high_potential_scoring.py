#!/usr/bin/env python3
"""
Validate that the high potential asset scoring system correctly identifies
drugs with strategic/business discontinuations as high potential opportunities
"""
import asyncio
import os
import sys
from thera_agent.drug_asset_discovery_agent import DrugAssetDiscoveryAgent

async def validate_scoring():
    """Test the scoring system with known high-potential drugs"""
    
    # Note: API key not needed for scoring validation
    agent = DrugAssetDiscoveryAgent()
    
    # Test individual drug scoring
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
            }
        },
        {
            "drug_name": "PF-07284892", 
            "max_phase": 1,
            "discontinuation_info": {
                "reason": "Strategic reasons",
                "details": "Trial terminated due to strategic portfolio decisions, still in Pfizer pipeline"
            },
            "ownership_info": {
                "clinical_trials": {"latest_activity": "2024-01-01"},
                "ownership_history": [{"company": "Pfizer"}]
            }
        },
        {
            "drug_name": "Generic Safety Failure",
            "max_phase": 2,
            "discontinuation_info": {
                "reason": "Safety concerns",
                "details": "Discontinued due to serious adverse events"
            },
            "ownership_info": {}
        }
    ]
    
    print("\n🎯 Testing High Potential Asset Scoring System")
    print("=" * 80)
    
    for drug in test_drugs:
        score = agent._calculate_high_potential_score(drug)
        print(f"\n{drug['drug_name']}:")
        print(f"  Score: {score:.2f}")
        print(f"  Phase: {drug['max_phase']}")
        print(f"  Reason: {drug['discontinuation_info']['reason']}")
        print(f"  Expected: {'HIGH' if 'strategic' in drug['discontinuation_info']['reason'].lower() else 'LOW'}")
        print(f"  Result: {'✅ PASS' if (score >= 0.5) == ('strategic' in drug['discontinuation_info']['reason'].lower()) else '❌ FAIL'}")
    
    print("\n" + "=" * 80)
    print("Scoring System Validation Complete")
    
if __name__ == "__main__":
    asyncio.run(validate_scoring())

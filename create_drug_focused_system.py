"""Create a proper drug-focused repurposing system without fallbacks"""
import asyncio
import os
import json
from thera_agent.repurposing_agent import DrugRepurposingAgent

async def test_drug_focused_analysis():
    """Test the new drug-focused analysis system"""
    
    # Ensure OPENAI_API_KEY is set in environment
    if "OPENAI_API_KEY" not in os.environ:
        raise ValueError("Please set OPENAI_API_KEY environment variable")
    
    agent = DrugRepurposingAgent()
    
    # Get failed trials
    print("🔍 Getting small cell lung cancer failed trials...")
    failed_trials = await agent.ct_client.search_trials(
        condition="small cell lung cancer",
        status=["TERMINATED", "WITHDRAWN", "SUSPENDED"],  
        max_results=5
    )
    
    # Extract actual failed drugs and their mechanisms
    failed_drugs = []
    for trial in failed_trials:
        for intervention in trial.get("interventions", []):
            if intervention.get("type") == "DRUG":
                failed_drugs.append({
                    "name": intervention["name"],
                    "description": intervention.get("description", ""),
                    "why_stopped": trial.get("why_stopped", "Unknown"),
                    "phase": trial.get("phase", []),
                    "nct_id": trial["nct_id"]
                })
    
    print(f"Found {len(failed_drugs)} failed drugs:")
    for drug in failed_drugs:
        print(f"  - {drug['name']}: {drug['why_stopped']}")
    
    # Get repurposing candidates  
    print("\n🔍 Getting repurposing candidates...")
    candidates = await agent.ct_client.get_drug_repurposing_candidates(
        disease="small cell lung cancer",
        exclude_targets=[]
    )
    
    # Enrich with ChEMBL data
    print(f"📈 Enriching {len(candidates)} candidates with ChEMBL data...")
    enriched_candidates = await agent._enrich_candidates_with_chembl(candidates)
    
    # Show enriched candidates
    print("\n💊 ENRICHED REPURPOSING CANDIDATES:")
    for i, candidate in enumerate(enriched_candidates[:5], 1):
        chembl_id = candidate.get("chembl_id", "N/A")
        phase = candidate.get("max_phase", "N/A") 
        name = candidate.get("drug_name", "Unknown")
        print(f"{i}. {name}")
        print(f"   ChEMBL ID: {chembl_id}")
        print(f"   Max Phase: {phase}")
        print(f"   Trial Count: {candidate.get('trial_count', 0)}")
        
    # Now create LLM prompt that focuses on actual failed drugs vs successful candidates
    print("\n🤖 Analyzing ACTUAL failed drugs vs repurposing candidates...")
    
    # Custom drug-focused analysis prompt
    prompt = f"""
    Analyze these ACTUAL failed drugs in small cell lung cancer and suggest mechanistically different alternatives:
    
    FAILED DRUGS:
    {json.dumps(failed_drugs, indent=2)}
    
    AVAILABLE REPURPOSING CANDIDATES (with ChEMBL data):
    {json.dumps([{
        'name': c.get('drug_name'),
        'chembl_id': c.get('chembl_id'), 
        'max_phase': c.get('max_phase'),
        'trial_count': c.get('trial_count')
    } for c in enriched_candidates[:10]], indent=2)}
    
    Based on this REAL data:
    1. Identify the specific mechanisms of the failed drugs
    2. Understand WHY they failed (not just generic reasons)
    3. From the available repurposing candidates, suggest ones with DIFFERENT mechanisms
    4. Explain how the repurposing candidates overcome the specific failures
    
    Return JSON:
    {{
        "failed_drug_analysis": [
            {{
                "drug_name": "actual_failed_drug",
                "mechanism": "specific_mechanism",
                "failure_reason": "why_it_failed_specifically",
                "chembl_id": "if_available"
            }}
        ],
        "repurposing_recommendations": [
            {{
                "drug_name": "repurposing_candidate",
                "chembl_id": "from_candidates_list",
                "mechanism": "how_different_from_failed",
                "rationale": "why_it_overcomes_failures",
                "confidence": 0.0-1.0
            }}
        ]
    }}
    """
    
    # Test the drug-focused LLM analysis
    print("\n🧠 Running drug-focused LLM analysis...")
    try:
        response = await agent._llm_query(prompt)
        print("✅ LLM Response received!")
        
        # Parse response
        if "```json" in response:
            response = response.split("```json")[1].split("```")[0]
        elif "```" in response:
            response = response.split("```")[1].split("```")[0]
            
        analysis = json.loads(response)
        
        print("\n📊 DRUG-FOCUSED ANALYSIS RESULTS:")
        print("\nFAILED DRUG ANALYSIS:")
        for drug in analysis.get("failed_drug_analysis", []):
            print(f"• {drug['drug_name']}: {drug['mechanism']} -> {drug['failure_reason']}")
            
        print("\nREPURPOSING RECOMMENDATIONS:")
        for rec in analysis.get("repurposing_recommendations", []):
            print(f"• {rec['drug_name']} ({rec['chembl_id']})")
            print(f"  Mechanism: {rec['mechanism']}")
            print(f"  Rationale: {rec['rationale']}")
            print(f"  Confidence: {rec['confidence']}")
            print()
            
    except Exception as e:
        print(f"❌ Error: {e}")
        print(f"Raw response: {response}")

if __name__ == "__main__":
    asyncio.run(test_drug_focused_analysis())

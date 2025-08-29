#!/usr/bin/env python3
"""
Check VOSAROXIN's actual indication - was it studied for lung cancer?
"""

import asyncio
import sys
import os
import json

# Add the project root to the path
sys.path.append('/home/anugraha/agent3')

from thera_agent.drug_asset_discovery_agent import DrugAssetDiscoveryAgent

async def check_vosaroxin_indication():
    """Check what VOSAROXIN was actually studied for"""
    
    print("🔍 Checking VOSAROXIN's Actual Indication")
    print("=" * 60)
    
    # Initialize the agent
    agent = DrugAssetDiscoveryAgent()
    
    drug_name = "VOSAROXIN"
    
    print(f"🧪 Analyzing: {drug_name}")
    print("-" * 40)
    
    # Check clinical trials for actual conditions studied
    try:
        print("📊 Checking Clinical Trials for Conditions...")
        trials = await agent.ct_client.search_trials(
            intervention=drug_name,
            max_results=20
        )
        
        if trials:
            print(f"✅ Found {len(trials)} trials")
            
            # Extract conditions from trials
            conditions = set()
            for trial in trials:
                trial_conditions = trial.get('conditions', [])
                if isinstance(trial_conditions, list):
                    conditions.update([c.lower() for c in trial_conditions])
                elif isinstance(trial_conditions, str):
                    conditions.add(trial_conditions.lower())
            
            print(f"\n🎯 CONDITIONS STUDIED:")
            for condition in sorted(conditions):
                print(f"   - {condition}")
            
            # Check for lung cancer specifically
            lung_cancer_terms = ['lung', 'nsclc', 'sclc', 'pulmonary', 'bronchial']
            lung_cancer_found = any(term in ' '.join(conditions) for term in lung_cancer_terms)
            
            print(f"\n🫁 LUNG CANCER INDICATION:")
            if lung_cancer_found:
                print("   ✅ VOSAROXIN was studied for lung cancer")
            else:
                print("   ❌ VOSAROXIN was NOT studied for lung cancer")
                
        else:
            print("❌ No trials found")
            
    except Exception as e:
        print(f"❌ Clinical trials check failed: {e}")
    
    # Use LLM to get definitive answer
    if agent.openai_client:
        print(f"\n🤖 LLM VERIFICATION:")
        print("-" * 40)
        
        try:
            prompt = f"""What was {drug_name} (vosaroxin) primarily studied for? 

Please provide:
1. Primary indication(s) studied in clinical trials
2. Was it ever studied for lung cancer specifically?
3. What type of cancer was the main focus?
4. Key trial names and phases

Be specific about the actual indications tested."""

            response = await asyncio.wait_for(
                agent.openai_client.responses.create(
                    model="gpt-4o",
                    tools=[{
                        "type": "web_search_preview",
                        "search_context_size": "medium"
                    }],
                    input=prompt
                ),
                timeout=30.0
            )
            
            print("📋 LLM Analysis:")
            print(response.output_text)
            
        except Exception as e:
            print(f"❌ LLM verification failed: {e}")
    
    print("\n" + "=" * 60)
    print("🏁 Indication Check Complete")

if __name__ == "__main__":
    asyncio.run(check_vosaroxin_indication())

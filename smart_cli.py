#!/usr/bin/env python3
"""Smart CLI that understands context like 'inhibitors above X nM'"""

import asyncio
import json
import re
from typing import Optional, Tuple
from thera_agent import TherapeuticTargetAgent

def parse_potency_filter(query: str) -> Tuple[str, Optional[float], Optional[float]]:
    """Extract disease and potency filters from query"""
    
    # Patterns to match potency filters
    patterns = [
        (r"(.+?)\s+inhibitors?\s+above\s+([\d.]+)\s*(nm|um|μm|micromolar)", "above"),
        (r"(.+?)\s+inhibitors?\s+below\s+([\d.]+)\s*(nm|um|μm|micromolar)", "below"),
        (r"(.+?)\s+compounds?\s+above\s+([\d.]+)\s*(nm|um|μm|micromolar)", "above"),
        (r"(.+?)\s+compounds?\s+below\s+([\d.]+)\s*(nm|um|μm|micromolar)", "below"),
    ]
    
    query_lower = query.lower()
    
    for pattern, filter_type in patterns:
        match = re.search(pattern, query_lower, re.IGNORECASE)
        if match:
            disease = match.group(1).strip()
            value = float(match.group(2))
            unit = match.group(3).lower()
            
            # Convert to nM
            if unit in ["um", "μm", "micromolar"]:
                value *= 1000
            
            if filter_type == "above":
                return disease, value, None  # min_ic50
            else:
                return disease, None, value  # max_ic50
    
    return query, None, None

def filter_inhibitors_by_potency(inhibitors, min_ic50=None, max_ic50=None):
    """Filter inhibitors by IC50 range"""
    filtered = []
    
    for inh in inhibitors:
        ic50 = inh.get('ic50_nm') or inh.get('standard_value_nm')
        if ic50 is None:
            continue
            
        if min_ic50 and ic50 < min_ic50:
            continue
        if max_ic50 and ic50 > max_ic50:
            continue
            
        filtered.append(inh)
    
    return filtered

async def smart_analyze(query: str):
    """Analyze with smart context understanding"""
    
    # Parse the query
    disease, min_ic50, max_ic50 = parse_potency_filter(query)
    
    print(f"🚀 Smart Therapeutic Target Analysis")
    print(f"Query: {query}")
    
    if min_ic50 or max_ic50:
        print(f"Detected filters:")
        if min_ic50:
            print(f"  - Minimum IC50: {min_ic50} nM ({min_ic50/1000:.1f} μM)")
        if max_ic50:
            print(f"  - Maximum IC50: {max_ic50} nM ({max_ic50/1000:.1f} μM)")
        print(f"  - Core disease: {disease}")
    
    print("=" * 60)
    
    # Initialize agent
    agent = TherapeuticTargetAgent()
    
    # Analyze the disease (use full query to get targets)
    results = await agent.analyze_disease(disease, is_disease=True)
    
    if not results:
        print("No results found")
        return
    
    # Apply potency filters if specified
    if min_ic50 or max_ic50:
        print(f"\n🔍 Applying potency filters...")
        
        for target, data in results.items():
            if 'inhibitors' in data:
                original_count = len(data['inhibitors'])
                data['inhibitors'] = filter_inhibitors_by_potency(
                    data['inhibitors'], 
                    min_ic50=min_ic50, 
                    max_ic50=max_ic50
                )
                filtered_count = len(data['inhibitors'])
                
                if original_count != filtered_count:
                    print(f"  {target}: {original_count} → {filtered_count} inhibitors")
    
    # Display results
    print("\n📋 ANALYSIS RESULTS")
    print("=" * 60)
    
    for target, data in results.items():
        inhibitors = data.get('inhibitors', [])
        structures = data.get('structures', [])
        score = data.get('target_score', 0)
        
        print(f"\n🎯 Target: {target} (Score: {score:.1f}/10)")
        print(f"  Structures: {len(structures)}")
        print(f"  Inhibitors: {len(inhibitors)}")
        
        if inhibitors:
            # Sort by IC50
            inhibitors.sort(key=lambda x: x.get('ic50_nm', float('inf')))
            
            # Show top 5
            print(f"\n  Top inhibitors:")
            for i, inh in enumerate(inhibitors[:5]):
                ic50_nm = inh.get('ic50_nm', 0)
                ic50_um = ic50_nm / 1000
                chembl_id = inh.get('molecule_chembl_id', 'Unknown')
                name = inh.get('pref_name', 'No name')
                
                print(f"    {i+1}. {chembl_id}: {name}")
                print(f"       IC50: {ic50_nm:.1f} nM ({ic50_um:.2f} μM)")
            
            if len(inhibitors) > 5:
                print(f"    ... and {len(inhibitors) - 5} more")
    
    # Save results
    output_file = f"smart_analysis_{disease.replace(' ', '_')}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Results saved to {output_file}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python smart_cli.py \"<query>\"")
        print("Examples:")
        print("  python smart_cli.py \"type 2 diabetes\"")
        print("  python smart_cli.py \"type 2 diabetes inhibitors above 10 micromolar\"")
        print("  python smart_cli.py \"cancer inhibitors below 100 nM\"")
        sys.exit(1)
    
    query = " ".join(sys.argv[1:])
    asyncio.run(smart_analyze(query))

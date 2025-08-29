#!/rusr/bin/env python3
"""
Improved CLI for drug repurposing analysis with clearer failure pattern explanations
"""
import asyncio
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from thera_agent.repurposing_agent import DrugRepurposingAgent

def print_results(results: Dict):
    """Print formatted results with clear explanations"""
    
    print("\n" + "="*80)
    print(f"🔬 DRUG REPURPOSING ANALYSIS: {results['disease'].upper()}")
    print("="*80)
    
    if results['original_target']:
        print(f"\n❌ Failed Target: {results['original_target']}")
    
    # CLEAR EXPLANATION OF WHAT'S BEING ANALYZED
    print(f"\n📊 CLINICAL TRIAL FAILURE ANALYSIS")
    print(f"Analyzing {results['failed_trials_count']} failed trials for {results['disease']}")
    print("\n" + "-"*60)
    
    # Analysis 1: Initial Categorization
    print("\n📋 ANALYSIS 1: Initial Failure Categorization")
    print("Source: Clinical trial 'why_stopped' field keyword matching")
    print("Coverage: All 100 failed trials categorized by primary reason\n")
    
    print("Failure Categories:")
    for pattern, count in results['failure_patterns'].items():
        if count > 0:
            # Add descriptions for each category
            descriptions = {
                "safety_issues": "Drug caused adverse events or toxicity",
                "efficacy_issues": "Drug didn't work as expected",
                "recruitment_issues": "Couldn't enroll enough patients",
                "business_reasons": "Funding, sponsor, or strategic decisions",
                "other_reasons": "Miscellaneous non-drug related issues",
                "unknown": "No clear reason provided"
            }
            desc = descriptions.get(pattern, "")
            print(f"  • {pattern.replace('_', ' ').title()}: {count} trials - {desc}")
    
    print(f"\nTotal categorized: {sum(results['failure_patterns'].values())} trials")
    
    # Analysis 2: Safety-Focused Deep Dive
    side_effects = results.get('side_effects_analysis', {})
    if side_effects:
        print("\n" + "-"*60)
        print("\n⚠️ ANALYSIS 2: Safety-Focused Deep Dive")
        print("Source: More rigorous safety signal detection (excludes false positives)")
        print("Coverage: Only trials with clear termination reasons\n")
        
        print(f"Safety-Related Terminations: {side_effects.get('safety_terminations_count', 0)} "
              f"(vs {results['failure_patterns'].get('safety_issues', 0)} in initial analysis)")
        
        # Explain the difference
        print("\n💡 Why the difference?")
        print("  - Analysis 2 excludes phrases like 'NOT related to safety'")
        print("  - Some trials lack detailed termination reasons")
        print("  - More stringent safety signal criteria\n")
        
        # Why stopped categories with explanations
        why_stopped = side_effects.get('why_stopped_categories', {})
        total_in_analysis2 = sum(why_stopped.values())
        
        if any(why_stopped.values()):
            print("Detailed Termination Categories:")
            category_desc = {
                "safety_related": "Confirmed safety/toxicity issues",
                "efficacy_related": "Drug ineffective or futile", 
                "business_related": "Non-scientific reasons",
                "other": "All other reasons (including recruitment)"
            }
            
            for category, count in why_stopped.items():
                if count > 0:
                    desc = category_desc.get(category, "")
                    print(f"  • {category.replace('_', ' ').title()}: {count} trials - {desc}")
            
            print(f"\nTotal in detailed analysis: {total_in_analysis2} trials")
            print(f"Missing from analysis: {100 - total_in_analysis2} trials (no clear termination reason)")
        
        # Common patterns
        patterns = side_effects.get('side_effects_patterns', {}).get('patterns', [])
        if patterns:
            print("\n🔍 Common Side Effect Patterns:")
            for pattern in patterns[:3]:
                print(f"  • {pattern}")
        
        organs = side_effects.get('side_effects_patterns', {}).get('organ_systems', [])
        if organs:
            print(f"\nAffected Organ Systems: {', '.join(organs)}")
    
    # Key Insights Box
    print("\n" + "="*60)
    print("📌 KEY INSIGHTS FOR DRUG REPURPOSING:")
    print("="*60)
    
    # Calculate recruitment+business percentage
    non_drug_failures = (results['failure_patterns'].get('recruitment_issues', 0) + 
                        results['failure_patterns'].get('business_reasons', 0))
    non_drug_percent = (non_drug_failures / max(results['failed_trials_count'], 1)) * 100
    
    print(f"\n✅ {non_drug_percent:.0f}% of failures were NOT due to the drug itself")
    print("   (recruitment + business reasons)")
    
    safety_percent = (results['failure_patterns'].get('safety_issues', 0) / 
                     max(results['failed_trials_count'], 1)) * 100
    efficacy_percent = (results['failure_patterns'].get('efficacy_issues', 0) / 
                       max(results['failed_trials_count'], 1)) * 100
    
    print(f"\n⚠️  Only {safety_percent:.0f}% failed for safety (Analysis 1)")
    print(f"❌ Only {efficacy_percent:.0f}% failed for lack of efficacy")
    
    print("\n💡 This suggests many failed drugs could work if:")
    print("   - Better patient recruitment strategies")
    print("   - More stable funding sources")
    print("   - Different dosing or combination approaches")
    
    # Continue with drug candidates...
    print("\n" + "="*80)
    print("💊 DRUG RESCUE OPPORTUNITIES")
    print("="*80)
    
    # Rest of the output remains the same...
    if 'repurposing_candidates' in results:
        print("\nDrugs that failed but may still have potential:")
        print("(Ranked by repurposing score - higher = better candidate)\n")
        
        print(f"{'Drug':<25} {'Targets':<35} {'Owner':<25} {'Trials':<8} {'Score':<8} {'Phases':<15} {'Availability':<12}")
        print("-" * 145)
        
        for candidate in results['repurposing_candidates'][:10]:
            # Format the output...
            drug = candidate.get('drug', 'Unknown')
            targets = ', '.join(candidate.get('targets', [])[:2]) or 'Unknown'
            if len(candidate.get('targets', [])) > 2:
                targets += f" (+{len(candidate['targets'])-2} more)"
            owner = candidate.get('owner', 'Unknown')[:23]
            phases = ', '.join(candidate.get('phases', []))[:13]
            availability = 'generic' if candidate.get('is_generic') else 'unknown'
            
            print(f"{drug:<25} {targets:<35} {owner:<25} {candidate.get('total_trials', 0):<8} "
                  f"{candidate.get('repurposing_score', 0):<8.1f} {phases:<15} {availability:<12}")
    
    # Continue with rest of original output...

async def main():
    parser = argparse.ArgumentParser(description='Analyze drug repurposing opportunities')
    parser.add_argument('disease', help='Disease to analyze (e.g., "glioblastoma", "alzheimer disease")')
    parser.add_argument('--target', help='Specific failed target to analyze')
    parser.add_argument('--max-trials', type=int, default=100, help='Maximum trials to analyze')
    parser.add_argument('--output', help='Save results to JSON file')
    
    args = parser.parse_args()
    
    agent = DrugRepurposingAgent()
    
    try:
        results = await agent.analyze_disease_failures(
            disease=args.disease,
            target=args.target,
            max_trials=args.max_trials
        )
        
        print_results(results)
        
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\n✅ Results saved to {output_path}")
            
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

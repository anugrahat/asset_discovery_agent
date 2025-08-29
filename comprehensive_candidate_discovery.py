#!/usr/bin/env python3
"""
Comprehensive Drug Asset Discovery - All Sources
Shows ALL high potential candidates found across ALL data sources
"""

import asyncio
import os
import sys
import json
from datetime import datetime
from typing import Dict, List, Any

# Add the project root to the path
sys.path.append('/home/anugraha/agent3')

from thera_agent.drug_asset_discovery_agent import DrugAssetDiscoveryAgent

async def discover_all_candidates(disease: str = "lung cancer") -> Dict[str, Any]:
    """
    Discover ALL high potential drug candidates from ALL sources
    Focus on non-safety discontinuations (business, funding, strategic)
    """
    
    # Set up API key from environment
    if "OPENAI_API_KEY" not in os.environ:
        raise ValueError("Please set OPENAI_API_KEY environment variable")
    
    print(f"🎯 COMPREHENSIVE DRUG ASSET DISCOVERY")
    print(f"Disease: {disease}")
    print(f"Focus: Non-safety discontinuations (business, funding, strategic)")
    print("=" * 80)
    
    # Initialize the agent
    agent = DrugAssetDiscoveryAgent()
    
    try:
        # Run the analysis
        results = await agent.analyze_disease_failures(disease=disease, max_trials=100)
        
        # Collect all candidates
        all_candidates = []
        
        # High potential assets
        high_potential = results.get('high_potential_assets', [])
        for candidate in high_potential:
            candidate['category'] = 'high_potential'
            all_candidates.append(candidate)
        
        # Moderate potential assets  
        moderate_potential = results.get('moderate_potential_assets', [])
        for candidate in moderate_potential:
            candidate['category'] = 'moderate_potential'
            all_candidates.append(candidate)
        
        # Low potential assets
        low_potential = results.get('low_potential_assets', [])
        for candidate in low_potential:
            candidate['category'] = 'low_potential'
            all_candidates.append(candidate)
        
        print(f"\n📊 DISCOVERY SUMMARY")
        print(f"High Potential Assets: {len(high_potential)}")
        print(f"Moderate Potential Assets: {len(moderate_potential)}")
        print(f"Low Potential Assets: {len(low_potential)}")
        print(f"Total Candidates: {len(all_candidates)}")
        
        # Analyze discontinuation reasons
        non_safety_candidates = []
        safety_candidates = []
        unknown_reason_candidates = []
        
        for candidate in all_candidates:
            # Get discontinuation reasons
            termination_reason = str(candidate.get('termination_reason', '')).lower()
            shelving_reason = str(candidate.get('shelving_reason', {}).get('reason', '')).lower()
            combined_reason = termination_reason + ' ' + shelving_reason
            
            # Classify discontinuation type
            is_safety = any(keyword in combined_reason for keyword in [
                'safety', 'adverse', 'toxic', 'side effect', 'harm', 'risk', 'tolerability'
            ])
            
            is_non_safety = any(keyword in combined_reason for keyword in [
                'business', 'funding', 'strategic', 'commercial', 'financial', 'portfolio',
                'priority', 'resource', 'pivot', 'restructur', 'acquisition', 'merger',
                'licensing', 'partnership', 'development halt', 'program discontinu',
                'competitive', 'market', 'economic'
            ])
            
            if is_safety:
                safety_candidates.append(candidate)
            elif is_non_safety or candidate.get('high_potential_score', 0) > 0.6:
                non_safety_candidates.append(candidate)
            else:
                unknown_reason_candidates.append(candidate)
        
        print(f"\n📈 DISCONTINUATION ANALYSIS")
        print(f"Non-Safety Discontinuations: {len(non_safety_candidates)}")
        print(f"Safety-Related Discontinuations: {len(safety_candidates)}")
        print(f"Unknown/Other Reasons: {len(unknown_reason_candidates)}")
        
        # Display detailed results for high potential non-safety candidates
        print(f"\n🎯 HIGH POTENTIAL NON-SAFETY DISCONTINUATIONS")
        print("=" * 80)
        
        # Sort by score
        sorted_non_safety = sorted(non_safety_candidates, 
                                 key=lambda x: x.get('high_potential_score', 0), 
                                 reverse=True)
        
        for i, candidate in enumerate(sorted_non_safety[:15], 1):
            drug_name = candidate.get('drug_name', 'Unknown')
            score = candidate.get('high_potential_score', 0)
            category = candidate.get('category', 'unknown')
            max_phase = candidate.get('max_phase', 0)
            
            # Get company/sponsor info
            sponsors = candidate.get('sponsors', [])
            company = sponsors[0] if sponsors else 'Unknown'
            
            # Get ownership info
            ownership = candidate.get('ownership_info', {})
            current_owner = 'Unknown'
            if ownership.get('ownership_history'):
                current_owner = ownership['ownership_history'][-1].get('company', 'Unknown')
            elif ownership.get('asset_availability', {}).get('current_owner'):
                current_owner = ownership['asset_availability']['current_owner']
            
            # Get target info
            target_info = candidate.get('target_info', {})
            targets = target_info.get('targets', [])
            target_str = ', '.join(targets[:2]) if targets else 'Unknown'
            
            # Get discontinuation reason
            reason = candidate.get('termination_reason', '')
            if not reason:
                shelving_info = candidate.get('shelving_reason', {})
                reason = shelving_info.get('reason', 'Unknown')
            
            # Get trial info
            total_trials = candidate.get('total_trials', 0)
            failed_trials = candidate.get('failed', 0)
            
            print(f"\n{i:2d}. {drug_name}")
            print(f"    📊 Score: {score:.3f} | Category: {category} | Max Phase: {max_phase}")
            print(f"    🏢 Company: {company} | Current Owner: {current_owner}")
            print(f"    🎯 Targets: {target_str}")
            print(f"    📋 Trials: {total_trials} total, {failed_trials} failed")
            print(f"    ❌ Discontinuation: {reason}")
            
            # Show asset availability if available
            asset_availability = ownership.get('asset_availability', {})
            if asset_availability.get('availability_status'):
                availability = asset_availability['availability_status']
                print(f"    💼 Availability: {availability}")
        
        # Show some safety-discontinued drugs for comparison
        if safety_candidates:
            print(f"\n⚠️  SAFETY-RELATED DISCONTINUATIONS (for comparison)")
            print("=" * 60)
            
            sorted_safety = sorted(safety_candidates, 
                                 key=lambda x: x.get('high_potential_score', 0), 
                                 reverse=True)
            
            for i, candidate in enumerate(sorted_safety[:5], 1):
                drug_name = candidate.get('drug_name', 'Unknown')
                score = candidate.get('high_potential_score', 0)
                reason = candidate.get('termination_reason', '')
                if not reason:
                    shelving_info = candidate.get('shelving_reason', {})
                    reason = shelving_info.get('reason', 'Unknown')
                
                print(f"{i}. {drug_name} (Score: {score:.3f}) - {reason}")
        
        # Save comprehensive results
        output_file = f"comprehensive_candidates_{disease.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        output_data = {
            'disease': disease,
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_candidates': len(all_candidates),
                'high_potential': len(high_potential),
                'moderate_potential': len(moderate_potential),
                'low_potential': len(low_potential),
                'non_safety_discontinuations': len(non_safety_candidates),
                'safety_discontinuations': len(safety_candidates),
                'unknown_reasons': len(unknown_reason_candidates)
            },
            'all_candidates': all_candidates,
            'non_safety_candidates': sorted_non_safety,
            'safety_candidates': sorted_safety,
            'analysis_results': results
        }
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2, default=str)
        
        print(f"\n💾 Comprehensive results saved to: {output_file}")
        
        return output_data
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return {}

async def main():
    """Run comprehensive discovery for multiple diseases"""
    
    diseases = ["lung cancer", "pancreatic cancer"]
    
    for disease in diseases:
        print(f"\n{'='*100}")
        print(f"ANALYZING: {disease.upper()}")
        print(f"{'='*100}")
        
        results = await discover_all_candidates(disease)
        
        if results:
            summary = results['summary']
            print(f"\n✅ ANALYSIS COMPLETE FOR {disease.upper()}")
            print(f"   Total Candidates: {summary['total_candidates']}")
            print(f"   High Potential: {summary['high_potential']}")
            print(f"   Non-Safety Discontinuations: {summary['non_safety_discontinuations']}")
        else:
            print(f"\n❌ Analysis failed for {disease}")

if __name__ == "__main__":
    asyncio.run(main())

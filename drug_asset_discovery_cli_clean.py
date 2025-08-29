#!/usr/bin/env python3
"""
High-Potential Drug Asset Discovery CLI
Integrates data from PubMed, ChEMBL, ClinicalTrials.gov, press releases, and LLMs
"""
import argparse
import json
import logging
import sys
import asyncio
from typing import Dict, List, Optional

from thera_agent.drug_asset_discovery_agent import DrugAssetDiscoveryAgent

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)

def create_parser():
    """Create command line argument parser"""
    parser = argparse.ArgumentParser(
        description="Discover high-potential drug assets from all available sources"
    )
    
    parser.add_argument(
        "disease",
        help="Disease or condition to analyze (e.g., 'type 2 diabetes', 'hypertension')"
    )
    
    parser.add_argument(
        "--max-trials",
        type=int,
        default=100,
        help="Maximum clinical trials to analyze (default: 100)"
    )
    
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Number of top candidates to show (default: 10)"
    )
    
    parser.add_argument(
        "--output",
        help="Save results to JSON file"
    )
    
    return parser

def print_clean_results(results: Dict, top_n: int = 10):
    """Print simplified, focused results"""
    
    disease = results.get('disease', 'Unknown').upper()
    print(f"\n{'='*80}")
    print(f"💊 HIGH-POTENTIAL DRUG ASSETS FOR {disease}")
    print(f"{'='*80}")
    
    # Get all analyzed drug assets (shelved, active, or otherwise)
    all_assets = results.get('shelved_assets', results.get('drug_rescue', []))
    
    # Filter for high-potential assets based on score (regardless of status)
    high_potential = [
        asset for asset in all_assets 
        if asset.get('discovery_score', 0) > 50  # Only show high-scoring assets
    ]
    high_potential.sort(key=lambda x: x.get('discovery_score', 0), reverse=True)
    
    if not high_potential:
        print("\n❌ No high-potential drug assets found for this disease.")
        print("   Try a different disease or broader search terms.")
        return
    
    print(f"\n🎯 Found {len(high_potential)} HIGH-POTENTIAL DRUG ASSETS")
    print("   (Based on data from PubMed, ChEMBL, ClinicalTrials.gov, press releases)")
    print()
    
    # Show top candidates
    for i, asset in enumerate(high_potential[:top_n], 1):
        drug_name = asset.get('drug', 'Unknown')
        score = asset.get('discovery_score', 0)
        phase = f"Phase {asset.get('max_phase', 'N/A')}" if asset.get('max_phase') else "Pre-clinical"
        
        # Get target info
        target = 'Unknown target'
        if asset.get('primary_target'):
            target = asset.get('primary_target')
        elif asset.get('targets'):
            target_names = [t.get('target_name', '') for t in asset.get('targets', []) if t.get('target_name')]
            target = target_names[0] if target_names else 'Unknown target'
        
        # Get shelving reason
        shelving_info = asset.get('shelving_reason', {})
        if isinstance(shelving_info, dict):
            reason = shelving_info.get('reason', 'unknown')
            confidence = shelving_info.get('confidence', 0)
        else:
            reason = str(shelving_info) if shelving_info != 'Unknown' else 'unknown'
            confidence = 0
        
        # Determine repurposing potential
        potential = "⭐⭐⭐" if score > 80 else "⭐⭐" if score > 60 else "⭐"
        
        # Print asset info
        print(f"{i}. {drug_name} {potential}")
        print(f"   • Target: {target}")
        print(f"   • Max Phase: {phase}")
        print(f"   • Potential Score: {score:.0f}/100")
        
        # Show program status and data sources
        program_status = asset.get('program_status', 'Unknown')
        discontinuation_reason = asset.get('discontinuation_reason', 'Unknown')
        print(f"   • Status: {program_status}")
        if discontinuation_reason and discontinuation_reason != 'Unknown':
            print(f"   • Reason: {discontinuation_reason}")
        
        # Show data sources that contributed to this asset
        data_sources = []
        if asset.get('chembl_id'):
            data_sources.append('ChEMBL')
        if asset.get('pubmed_refs'):
            data_sources.append('PubMed')
        if asset.get('clinical_trials'):
            data_sources.append('ClinicalTrials.gov')
        if asset.get('press_releases'):
            data_sources.append('Press releases')
        if asset.get('indication_from_llm'):
            data_sources.append('LLM analysis')
        
        if data_sources:
            print(f"   • Data sources: {', '.join(data_sources)}")
        
        # Show regional status if available
        regional_status = asset.get('regional_status', [])
        if regional_status:
            print(f"   🌍 Regional status: {', '.join(regional_status)}")
        
        
        print()
    
    # Summary recommendations
    print("\n" + "="*80)
    print("📊 ASSET DISCOVERY INSIGHTS")
    print("="*80)
    
    # Count by program status
    status_counts = {}
    for asset in high_potential:
        status = asset.get('program_status', 'Unknown')
        status_counts[status] = status_counts.get(status, 0) + 1
    
    print("\nProgram status distribution:")
    for status, count in sorted(status_counts.items(), key=lambda x: x[1], reverse=True):
        if status is None:
            emoji = "❓"
            status_display = "Unknown"
        else:
            emoji = "🔄" if "Active" in status else "⏸️" if "Discontinued" in status else "✅" if "Marketed" in status else "❓"
            status_display = status
        print(f"  {emoji} {status_display}: {count} drugs")
    
    # Best candidates - high scoring with good evidence
    best_candidates = [
        asset for asset in high_potential 
        if asset.get('discovery_score', 0) > 70 and 
        asset.get('status_evidence', '') != 'Unclear'
    ]
    
    if best_candidates:
        print(f"\n🏆 TOP OPPORTUNITIES (highest potential): {len(best_candidates)} drugs")
        for asset in best_candidates[:3]:
            print(f"   • {asset.get('drug', 'Unknown')} - Score: {asset.get('discovery_score', 0):.0f}")
    
    print("\n💡 High-potential assets identified through comprehensive analysis")
    print("   of clinical trials, literature, regulatory data, and press releases")

async def main():
    """Main entry point"""
    parser = create_parser()
    args = parser.parse_args()
    
    # Disable verbose logging
    logging.getLogger("thera_agent").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("aiohttp").setLevel(logging.WARNING)
    
    try:
        # Create agent
        llm_api_key = None  # Will use environment variable if set
        
        agent = DrugAssetDiscoveryAgent(api_key=None, llm_api_key=llm_api_key)
        
        print(f"🔍 Discovering high-potential drug assets for {args.disease}...")
        print(f"   (Searching PubMed, ChEMBL, ClinicalTrials.gov, and other sources)")
        
        # Run comprehensive drug asset discovery
        try:
            results = await agent.analyze_disease_failures(
                disease=args.disease,
                max_trials=args.max_trials
            )
        except Exception as e:
            print(f"\n⚠️  Warning: Some API connections failed, results may be incomplete")
            print(f"   Error details: {str(e)}")
            results = {}
        
        # Print clean results
        print_clean_results(results, top_n=args.top)
        
        # Save to file if requested
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\n📁 Full results saved to: {args.output}")
    
    except KeyboardInterrupt:
        print("\n\nAnalysis cancelled by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"\n❌ Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())

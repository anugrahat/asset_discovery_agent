#!/usr/bin/env python3
"""
Command-line interface for Therapeutic Target Agent
"""
import argparse
import asyncio
import os
import json
from pathlib import Path
import re
from dotenv import load_dotenv
from thera_agent import TherapeuticTargetAgent
from thera_agent.disease_mapper import DiseaseTargetMapper
from thera_agent.query_parser import QueryParser
from thera_agent.result_summarizer import result_summarizer


async def main():
    """Main CLI entry point"""
    load_dotenv()
    
    parser = argparse.ArgumentParser(
        description="Production Therapeutic Target Discovery Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "EGFR inhibitors under 10 nM"
  %(prog)s "EGFR, JAK2, CDK9 under 100 nM" --multi-target
  %(prog)s "emphysema" --disease --output emphysema_targets.json
  %(prog)s "Alzheimer's disease" --disease
  %(prog)s "BRAF between 5 and 50 nM" --output results.json
        """
    )
    
    parser.add_argument("query", help="Target discovery query or disease name")
    parser.add_argument("--multi-target", action="store_true", 
                       help="Treat query as comma-separated list of targets")
    parser.add_argument("--disease", action="store_true",
                       help="Treat query as disease name (uses LLM to find targets)")
    parser.add_argument("--output", "-o", type=Path, default="target_analysis.json",
                       help="Output JSON file (default: target_analysis.json)")
    parser.add_argument("--cache-dir", type=Path, default="cache",
                       help="Cache directory for API responses")
    parser.add_argument("--max-ic50", type=float, 
                       help="Maximum IC50 in nM (overrides query parsing)")
    parser.add_argument("--min-ic50", type=float,
                       help="Minimum IC50 in nM (overrides query parsing)")
    
    args = parser.parse_args()
    
    # Initialize agent
    ncbi_api_key = os.getenv("NCBI_API_KEY")
    agent = TherapeuticTargetAgent(ncbi_api_key=ncbi_api_key, cache_dir=args.cache_dir)
    
    try:
        print(f"🚀 Therapeutic Target Agent v2.0")
        print(f"Query: {args.query}")
        print("=" * 60)
        
        if args.disease:
            # Disease-to-targets mapping using LLM
            print(f"🧠 Using LLM to identify therapeutic targets for: {args.query}")
            try:
                mapper = DiseaseTargetMapper()
                disease_targets = await mapper.map_disease_to_targets(args.query, max_targets=5)
                
                if not disease_targets:
                    print("❌ No therapeutic targets found for this disease")
                    return
                
                print(f"🎯 Found {len(disease_targets)} potential targets:")
                for i, target in enumerate(disease_targets, 1):
                    print(f"  {i}. {target.gene_symbol} - {target.rationale} (Confidence: {target.confidence:.1f})")
                
                # Use multi-target analysis on the discovered targets
                gene_symbols = [t.gene_symbol for t in disease_targets]
                results = await agent.multi_target_analysis(
                    gene_symbols,
                    max_ic50_nm=args.max_ic50,
                    min_ic50_nm=args.min_ic50
                )
                
                # Add disease mapping info to results
                results["disease_query"] = args.query
                results["discovered_targets"] = [
                    {
                        "gene_symbol": t.gene_symbol,
                        "protein_name": t.protein_name,
                        "rationale": t.rationale,
                        "confidence": t.confidence,
                        "clinical_stage": t.clinical_stage,
                        "pathway": t.pathway
                    }
                    for t in disease_targets
                ]
                
            except Exception as e:
                print(f"⚠️  LLM mapping failed, using fallback: {e}")
                # Fallback to basic parsing
                gene_symbol = extract_gene_symbol(args.query)
                results = await agent.analyze_target(
                    gene_symbol,
                    max_ic50_nm=args.max_ic50,
                    min_ic50_nm=args.min_ic50
                )
                
        elif args.multi_target:
            # Parse multiple targets
            gene_symbols = [g.strip().upper() for g in args.query.split(",")]
            results = await agent.multi_target_analysis(
                gene_symbols, 
                max_ic50_nm=args.max_ic50,
                min_ic50_nm=args.min_ic50
            )
        else:
            # Single target analysis with LLM query parsing
            print(f"🤖 Parsing query with LLM...")
            query_parser = QueryParser()
            parsed_query = await query_parser.parse_query(args.query)
            
            if parsed_query.is_disease_query:
                # Redirect to disease analysis
                print(f"🧠 Detected disease query, switching to disease mode...")
                mapper = DiseaseTargetMapper()
                disease_targets = await mapper.map_disease_to_targets(parsed_query.disease_context or args.query, max_targets=5)
                
                if disease_targets:
                    gene_symbols = [t.gene_symbol for t in disease_targets]
                    results = await agent.multi_target_analysis(
                        gene_symbols,
                        max_ic50_nm=parsed_query.max_ic50_nm or args.max_ic50,
                        min_ic50_nm=parsed_query.min_ic50_nm or args.min_ic50
                    )
                else:
                    print("❌ No targets found for this disease")
                    return
            elif len(parsed_query.gene_symbols) > 1:
                # Multi-target analysis detected by LLM
                print(f"🎯 Multi-target query detected: {parsed_query.gene_symbols}")
                results = await agent.multi_target_analysis(
                    parsed_query.gene_symbols,
                    max_ic50_nm=parsed_query.max_ic50_nm or args.max_ic50,
                    min_ic50_nm=parsed_query.min_ic50_nm or args.min_ic50
                )
            elif len(parsed_query.gene_symbols) == 1:
                # Single target analysis
                gene_symbol = parsed_query.gene_symbols[0]
                print(f"🎯 Single target detected: {gene_symbol}")
                if parsed_query.min_ic50_nm or parsed_query.max_ic50_nm:
                    print(f"🔬 IC50 range: {parsed_query.min_ic50_nm or 'N/A'} - {parsed_query.max_ic50_nm or 'N/A'} nM")
                
                results = await agent.analyze_target(
                    gene_symbol,
                    max_ic50_nm=parsed_query.max_ic50_nm or args.max_ic50,
                    min_ic50_nm=parsed_query.min_ic50_nm or args.min_ic50
                )
            else:
                # Fallback to basic parsing
                print(f"⚠️  LLM parsing incomplete, using fallback...")
                gene_symbol = extract_gene_symbol(args.query)
                results = await agent.analyze_target(
                    gene_symbol,
                    max_ic50_nm=args.max_ic50,
                    min_ic50_nm=args.min_ic50
                )
        
        # Display summary
        print("\n" + "=" * 60)
        print("📋 ANALYSIS SUMMARY")
        print("=" * 60)
        
        if "summary" in results:
            print(results["summary"])
        
        # Display key findings
        if "targets" in results:  # Multi-target
            print(f"\n🎯 TOP TARGETS:")
            for i, target in enumerate(results["targets"][:5], 1):
                score = target.get("target_score", 0)
                inhibitor_count = len(target.get("inhibitors", []))
                structure_count = len(target.get("structures", []))
                print(f"{i}. {target['gene_symbol']} (Score: {score:.1f}) - "
                      f"{inhibitor_count} inhibitors, {structure_count} structures")
        
        else:  # Single target
            inhibitors = results.get("inhibitors", [])
            if inhibitors:
                print(f"\n💊 TOP INHIBITORS:")
                for i, inh in enumerate(inhibitors[:5], 1):
                    # Handle None preferred_name explicitly
                    preferred_name = inh.get("preferred_name")
                    chembl_id = inh.get("molecule_chembl_id", "Unknown")
                    name = preferred_name if preferred_name else chembl_id
                    
                    ic50 = inh.get("standard_value_nm", "N/A")
                    quality = inh.get("quality_score", 0)
                    print(f"{i}. {name} - IC50: {ic50} nM (Quality: {quality:.2f})")
            
            # Display IC50 table if available
            ic50_table = results.get("ic50_table")
            if ic50_table:
                print(f"\n📊 IC50 TABLE (Top {len(ic50_table)} compounds):")
                print("-" * 100)
                print(f"{'ChEMBL ID':<15} {'IC50':<12} {'Assay Type':<15} {'Phase':<12} {'Quality':<8} {'Description':<40}")
                print("-" * 100)
                
                for row in ic50_table:
                    chembl_id = row['chembl_id'][:14]
                    ic50_display = row['ic50_display'][:11]
                    assay_type = row['assay_type'][:14]
                    max_phase = str(row.get('max_phase', 'Pre'))[:11]
                    quality = f"{row['quality_score']:.2f}"
                    description = row['assay_description'][:39]
                    
                    print(f"{chembl_id:<15} {ic50_display:<12} {assay_type:<15} {max_phase:<12} {quality:<8} {description:<40}")
        
        # Save results
        output_file = args.output
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"💾 Results saved to {output_file}")
        
        # Generate LLM-powered beautiful summary
        print("\n🤖 Generating intelligent summary...")
        try:
            summary = await result_summarizer.summarize_results(output_file, args.query)
            print("\n" + "="*80)
            print("📋 INTELLIGENT ANALYSIS SUMMARY")
            print("="*80)
            print(summary)
            print("="*80)
        except Exception as e:
            print(f"⚠️ Summary generation failed: {e}")
        
        print(f"\n✅ Analysis complete! Results saved to {output_file}")
        
    except KeyboardInterrupt:
        print("\n⚠️  Analysis interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await agent.cleanup()


def extract_gene_symbol(query: str) -> str:
    """Extract gene symbol from query string (basic implementation)"""
    # Simple patterns for common gene names
    import re
    
    # Look for common gene patterns
    gene_patterns = [
        r'\b([A-Z]{2,6}\d*)\b',  # Standard gene names like EGFR, JAK2, CDK9
        r'\b(p53|RAS|MYC|BCL2)\b',  # Common oncogenes
    ]
    
    for pattern in gene_patterns:
        matches = re.findall(pattern, query.upper())
        if matches:
            return matches[0]
    
    # Fallback: return first word that looks gene-like
    words = query.upper().split()
    for word in words:
        if len(word) >= 3 and word.isalpha():
            return word
    
    return "UNKNOWN"


if __name__ == "__main__":
    asyncio.run(main())

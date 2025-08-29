#!/usr/bin/env python3
"""
Example analyses demonstrating drug repurposing capabilities
"""
import asyncio
import json
from thera_agent.repurposing_agent import DrugRepurposingAgent

async def analyze_glioblastoma_egfr_failures():
    """Analyze why EGFR inhibitors fail in glioblastoma"""
    agent = DrugRepurposingAgent()
    
    print("🧠 CASE STUDY: Glioblastoma EGFR Failures")
    print("="*60)
    print("Context: EGFR is amplified in ~60% of glioblastomas, but")
    print("EGFR inhibitors that work in lung cancer fail in brain tumors.")
    print("Let's understand why and find alternatives...\n")
    
    results = await agent.analyze_disease_failures(
        disease="glioblastoma", 
        target="EGFR"
    )
    
    # Key insights
    print("🔬 KEY FINDINGS:")
    print(f"Failed trials analyzed: {results['failed_trials_count']}")
    
    analysis = results['failure_analysis']
    print("\n📊 Why EGFR inhibitors fail in glioblastoma:")
    for reason in analysis['main_failure_reasons']:
        print(f"  • {reason}")
    
    print(f"\n🧬 Biological insight: {analysis['biological_insights']}")
    
    print("\n🎯 Alternative targets suggested:")
    for i, target in enumerate(results['alternative_targets'][:3], 1):
        print(f"\n{i}. {target['target']} (Confidence: {target['confidence']:.0%})")
        print(f"   Rationale: {target['rationale']}")
        print(f"   Development score: {target.get('development_score', 0):.2f}")
    
    return results

async def analyze_covid_antivirals():
    """Analyze COVID-19 antiviral failures and repurposing"""
    agent = DrugRepurposingAgent()
    
    print("\n\n🦠 CASE STUDY: COVID-19 Antiviral Repurposing")
    print("="*60)
    print("Context: Many antivirals were tested for COVID-19.")
    print("Let's find which failed and which could be repurposed...\n")
    
    results = await agent.analyze_disease_failures(
        disease="COVID-19",
        target="viral replication"
    )
    
    # Repurposing candidates
    print("💊 TOP REPURPOSING CANDIDATES:")
    candidates = results['repurposing_candidates'][:5]
    for cand in candidates:
        print(f"\n• {cand['drug']}")
        print(f"  Clinical trials: {cand['total_trials']} "
              f"(Completed: {cand['completed']}, Failed: {cand['failed']})")
        print(f"  Repurposing score: {cand['repurposing_score']:.1f}")
    
    return results

async def analyze_melanoma_braf_resistance():
    """Analyze BRAF inhibitor resistance in melanoma"""
    agent = DrugRepurposingAgent()
    
    print("\n\n🎗️ CASE STUDY: Melanoma BRAF Resistance")
    print("="*60) 
    print("Context: BRAF V600E inhibitors work initially but resistance develops.")
    print("Let's find combination targets and next-gen approaches...\n")
    
    results = await agent.analyze_disease_failures(
        disease="melanoma",
        target="BRAF"
    )
    
    # Focus on resistance mechanisms
    print("🧬 Understanding BRAF inhibitor failures:")
    failures = results['failure_analysis']
    print(f"Common failure mechanisms: {', '.join(failures['common_mechanisms_failed'])}")
    
    print("\n🎯 Combination/Alternative targets for resistance:")
    for target in results['alternative_targets'][:3]:
        print(f"\n• {target['target']}")
        print(f"  Why it helps: {target['rationale'][:100]}...")
        if target.get('inhibitor_count', 0) > 0:
            print(f"  Available inhibitors: {target['inhibitor_count']}")
    
    return results

async def generate_repurposing_report():
    """Generate a comprehensive repurposing report"""
    
    diseases = [
        ("glioblastoma", "EGFR"),
        ("COVID-19", None),
        ("melanoma", "BRAF"),
        ("Alzheimer disease", "BACE1"),
        ("pancreatic cancer", "KRAS")
    ]
    
    agent = DrugRepurposingAgent()
    report = {
        "title": "Drug Repurposing Opportunities Report",
        "generated_date": "2024-07-29",
        "analyses": []
    }
    
    print("\n📊 GENERATING COMPREHENSIVE REPURPOSING REPORT")
    print("="*60)
    
    for disease, target in diseases:
        print(f"\nAnalyzing {disease}" + (f" ({target})" if target else "") + "...")
        
        try:
            results = await agent.analyze_disease_failures(disease, target)
            
            summary = {
                "disease": disease,
                "target": target,
                "failed_trials": results['failed_trials_count'],
                "top_repurposing_candidates": [
                    {
                        "drug": c['drug'],
                        "score": c['repurposing_score'],
                        "trials": c['total_trials']
                    }
                    for c in results['repurposing_candidates'][:3]
                ],
                "alternative_targets": [
                    {
                        "target": t['target'],
                        "confidence": t['confidence'],
                        "rationale": t['rationale'][:100] + "..."
                    }
                    for t in results['alternative_targets'][:3]
                ]
            }
            
            report["analyses"].append(summary)
            
        except Exception as e:
            print(f"  ⚠️  Error: {str(e)}")
            continue
    
    # Save report
    with open("results/repurposing_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print("\n✅ Report saved to results/repurposing_report.json")
    
    # Print summary
    print("\n📋 REPORT SUMMARY:")
    print(f"Diseases analyzed: {len(report['analyses'])}")
    
    total_candidates = sum(
        len(a['top_repurposing_candidates']) 
        for a in report['analyses']
    )
    print(f"Total repurposing candidates identified: {total_candidates}")
    
    total_targets = sum(
        len(a['alternative_targets'])
        for a in report['analyses']  
    )
    print(f"Total alternative targets suggested: {total_targets}")

async def main():
    """Run all examples"""
    
    # Example 1: Glioblastoma EGFR
    await analyze_glioblastoma_egfr_failures()
    
    # Example 2: COVID-19 antivirals
    await analyze_covid_antivirals()
    
    # Example 3: Melanoma BRAF resistance
    await analyze_melanoma_braf_resistance()
    
    # Generate comprehensive report
    await generate_repurposing_report()

if __name__ == "__main__":
    asyncio.run(main())

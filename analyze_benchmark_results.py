#!/usr/bin/env python3
"""Analyze benchmark results for all targets"""
import json
import os
from collections import defaultdict

def analyze_results():
    """Analyze all benchmark results"""
    
    # Known FDA drugs and their typical IC50 ranges
    fda_drugs = {
        "EGFR": {
            "gefitinib": {"chembl_ids": ["CHEMBL939"], "typical_ic50_nm": 33},
            "erlotinib": {"chembl_ids": ["CHEMBL553"], "typical_ic50_nm": 2},
            "afatinib": {"chembl_ids": ["CHEMBL1173655"], "typical_ic50_nm": 0.5},
            "osimertinib": {"chembl_ids": ["CHEMBL3353410"], "typical_ic50_nm": 18}
        },
        "BRAF": {
            "vemurafenib": {"chembl_ids": ["CHEMBL1336"], "typical_ic50_nm": 31},
            "dabrafenib": {"chembl_ids": ["CHEMBL2028663"], "typical_ic50_nm": 0.65},
            "encorafenib": {"chembl_ids": ["CHEMBL3545110"], "typical_ic50_nm": 0.35}
        },
        "JAK2": {
            "ruxolitinib": {"chembl_ids": ["CHEMBL1789941"], "typical_ic50_nm": 3.3},
            "fedratinib": {"chembl_ids": ["CHEMBL1287853"], "typical_ic50_nm": 3}
        },
        "BCL2": {
            "venetoclax": {"chembl_ids": ["CHEMBL3137309"], "typical_ic50_nm": 0.01}
        }
    }
    
    results_files = {
        "EGFR": "results/egfr_bench.json",
        "JAK2": "results/jak2_bench.json", 
        "CDK9": "results/cdk9_bench.json",
        "BRAF": "results/braf_bench.json",
        "BCL2": "results/bcl2_bench.json"
    }
    
    print("🔬 OMICS ORACLE BENCHMARK ANALYSIS")
    print("=" * 60)
    
    for target, file_path in results_files.items():
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            print(f"\n📊 {target} Analysis:")
            print(f"  Total compounds found: {len(data.get('inhibitors', []))}")
            
            # Get IC50 table data
            ic50_table = data.get('ic50_table', [])
            if ic50_table:
                print(f"  Top compounds in IC50 table: {len(ic50_table)}")
                print(f"  Most potent: {ic50_table[0]['chembl_id']} ({ic50_table[0]['ic50_display']})")
                
                # IC50 distribution
                ic50_ranges = defaultdict(int)
                for compound in ic50_table:
                    ic50 = compound.get('ic50_nm', 0)
                    if ic50 < 1:
                        ic50_ranges["<1 nM"] += 1
                    elif ic50 < 10:
                        ic50_ranges["1-10 nM"] += 1
                    elif ic50 < 100:
                        ic50_ranges["10-100 nM"] += 1
                    else:
                        ic50_ranges[">100 nM"] += 1
                
                print("  IC50 distribution:")
                for range_name, count in sorted(ic50_ranges.items()):
                    print(f"    {range_name}: {count} compounds")
            
            # Check for FDA drugs if target has them
            if target in fda_drugs:
                print(f"\n  FDA-approved drugs for {target}:")
                found_chembl_ids = {comp['chembl_id'] for comp in ic50_table}
                
                for drug_name, drug_info in fda_drugs[target].items():
                    found = False
                    for chembl_id in drug_info["chembl_ids"]:
                        if chembl_id in found_chembl_ids:
                            found = True
                            # Find the compound in results
                            for comp in ic50_table:
                                if comp['chembl_id'] == chembl_id:
                                    print(f"    ✓ {drug_name} ({chembl_id}) - FOUND at {comp['ic50_display']}")
                                    break
                            break
                    
                    if not found:
                        typical_ic50 = drug_info["typical_ic50_nm"]
                        print(f"    ✗ {drug_name} ({drug_info['chembl_ids'][0]}) - NOT FOUND (typical IC50: {typical_ic50} nM)")
            
            # Paper analysis
            papers = data.get('papers', [])
            print(f"\n  Literature analysis:")
            print(f"    Total papers: {len(papers)}")
            if papers:
                # Check relevance
                relevant = sum(1 for p in papers if target.lower() in p.get('title', '').lower())
                print(f"    Papers mentioning {target}: {relevant} ({relevant/len(papers)*100:.1f}%)")

if __name__ == "__main__":
    analyze_results()

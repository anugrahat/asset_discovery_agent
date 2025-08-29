#!/usr/bin/env python3
"""Run a simplified benchmark to test Omics Oracle"""
import subprocess
import json
import time
import os

def run_query(query, output_file):
    """Run a single CLI query"""
    print(f"Running: {query}")
    cmd = f'python cli.py "{query}" --output {output_file}'
    
    start = time.time()
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    elapsed = time.time() - start
    
    if result.returncode == 0:
        print(f"  ✓ Success in {elapsed:.1f}s")
        # Load and summarize results
        if os.path.exists(output_file):
            with open(output_file, 'r') as f:
                data = json.load(f)
            n_compounds = len(data.get('inhibitors', []))
            print(f"  Found {n_compounds} compounds")
            if 'ic50_table' in data and data['ic50_table']:
                top = data['ic50_table'][0]
                print(f"  Most potent: {top['chembl_id']} ({top['ic50_display']})")
    else:
        print(f"  ✗ Failed: {result.stderr}")
    
    return result.returncode == 0

def main():
    print("🚀 OMICS ORACLE SIMPLE BENCHMARK")
    print("=" * 60)
    
    # Ensure results directory exists
    os.makedirs("results", exist_ok=True)
    
    # Test queries
    queries = [
        # Remove IC50 filter to find FDA drugs
        ("EGFR inhibitors", "results/egfr_all.json"),
        ("BRAF inhibitors", "results/braf_all.json"),
        ("JAK2 inhibitors", "results/jak2_all.json"),
        
        # With IC50 filter
        ("CDK9 inhibitors IC50 < 10 nM", "results/cdk9_potent.json"),
        
        # Disease queries
        ("lung cancer", "results/lung_cancer.json", "--disease"),
        ("melanoma", "results/melanoma.json", "--disease"),
        
        # Novel targets
        ("SARS-CoV-2 main protease inhibitors", "results/covid_mpro.json"),
        ("KRAS G12C inhibitors", "results/kras_g12c.json")
    ]
    
    success_count = 0
    for query_parts in queries:
        if len(query_parts) == 3:
            query, output, flag = query_parts
            full_query = f"{flag} {query}"
        else:
            query, output = query_parts
            full_query = query
        
        if run_query(full_query, output):
            success_count += 1
        print()
    
    print(f"\nCompleted {success_count}/{len(queries)} queries successfully")
    
    # Analyze FDA drug coverage
    print("\n📊 FDA DRUG COVERAGE ANALYSIS")
    print("=" * 60)
    
    fda_analysis = {
        "EGFR": {
            "file": "results/egfr_all.json",
            "drugs": {
                "CHEMBL939": "gefitinib",
                "CHEMBL553": "erlotinib",
                "CHEMBL1173655": "afatinib",
                "CHEMBL3353410": "osimertinib"
            }
        },
        "BRAF": {
            "file": "results/braf_all.json", 
            "drugs": {
                "CHEMBL1336": "vemurafenib",
                "CHEMBL2028663": "dabrafenib"
            }
        },
        "JAK2": {
            "file": "results/jak2_all.json",
            "drugs": {
                "CHEMBL1789941": "ruxolitinib"
            }
        }
    }
    
    for target, info in fda_analysis.items():
        if os.path.exists(info["file"]):
            with open(info["file"], 'r') as f:
                data = json.load(f)
            
            found_ids = {inh['molecule_chembl_id'] for inh in data.get('inhibitors', [])}
            print(f"\n{target}:")
            for chembl_id, drug_name in info["drugs"].items():
                if chembl_id in found_ids:
                    print(f"  ✓ {drug_name} ({chembl_id}) - FOUND")
                else:
                    print(f"  ✗ {drug_name} ({chembl_id}) - NOT FOUND")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
CLI-based benchmark script for Omics Oracle
Uses natural language queries through the CLI interface
"""
import subprocess
import json
import time
import os
import statistics
from typing import List, Dict, Set, Tuple
from pathlib import Path

# Benchmark queries with natural language and ground truth
BENCHMARK_QUERIES = [
    {
        "nl_query": "EGFR inhibitors IC50 < 10 nM for lung cancer",
        "ground_truth": ["CHEMBL553", "CHEMBL554"],  # Erlotinib, Lapatinib
        "output_file": "egfr_bench.json"
    },
    {
        "nl_query": "JAK2 inhibitors under 5 nM for myelofibrosis", 
        "ground_truth": ["CHEMBL1287853", "CHEMBL1789941"],  # Fedratinib, Ruxolitinib
        "output_file": "jak2_bench.json"
    },
    {
        "nl_query": "CDK9 inhibitors IC50 below 5 nM",
        "ground_truth": ["CHEMBL2103840", "CHEMBL4756595"],  # Dinaciclib, Enitociclib
        "output_file": "cdk9_bench.json"
    },
    {
        "nl_query": "BRAF V600E mutation inhibitors IC50 < 1 nM",
        "ground_truth": ["CHEMBL2028663", "CHEMBL3301612"],  # Dabrafenib, Encorafenib
        "output_file": "braf_bench.json"
    },
    {
        "nl_query": "BCL2 inhibitors under 10 nM for leukemia",
        "ground_truth": ["CHEMBL3137309"],  # Venetoclax
        "output_file": "bcl2_bench.json"
    }
]

def run_cli_query(query: str, output_file: str) -> Tuple[float, Set[str], Dict]:
    """Run a CLI query and return latency, discovered ChEMBL IDs, and full results"""
    
    # Ensure output directory exists
    output_path = Path("results") / output_file
    output_path.parent.mkdir(exist_ok=True)
    
    # Build command
    cmd = [
        "python", "cli.py",
        query,
        "--output", str(output_path)
    ]
    
    print(f"  🔍 Running: {' '.join(cmd)}")
    
    # Measure time
    start_time = time.time()
    
    try:
        # Run CLI command
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd="/home/anugraha/agent2",
            env={**os.environ, "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY", "")}
        )
        
        latency = time.time() - start_time
        
        if result.returncode != 0:
            print(f"  ❌ CLI error: {result.stderr}")
            return latency, set(), {}
        
        # Load the output JSON
        with open(output_path, 'r') as f:
            data = json.load(f)
        
        # Extract ChEMBL IDs
        chembl_ids = set()
        
        # Handle single or multi-target results
        if 'results' in data:  # Multi-target
            for target_name, target_data in data['results'].items():
                ic50_table = target_data.get('ic50_table', [])
                for row in ic50_table:
                    if row.get('chembl_id'):
                        chembl_ids.add(row['chembl_id'])
        else:  # Single target
            ic50_table = data.get('ic50_table', [])
            for row in ic50_table:
                if row.get('chembl_id'):
                    chembl_ids.add(row['chembl_id'])
        
        return latency, chembl_ids, data
        
    except Exception as e:
        print(f"  ❌ Exception: {e}")
        return time.time() - start_time, set(), {}

def run_cli_benchmark():
    """Run the CLI-based benchmark"""
    print("🧬 Omics Oracle CLI Benchmark Suite")
    print("=" * 60)
    print("Using natural language queries through CLI...")
    
    results = []
    
    for i, query_item in enumerate(BENCHMARK_QUERIES):
        query = query_item['nl_query']
        ground_truth = set(query_item['ground_truth'])
        output_file = query_item['output_file']
        
        print(f"\n[{i+1}/{len(BENCHMARK_QUERIES)}] Query: {query}")
        print(f"  📋 Ground truth: {', '.join(ground_truth)}")
        
        # Run CLI query
        latency, discovered_ids, full_results = run_cli_query(query, output_file)
        
        # Calculate metrics
        hits = discovered_ids & ground_truth
        hit_rate = len(hits) / len(ground_truth) if ground_truth else 0
        
        # Display results
        print(f"  ⏱️  Latency: {latency:.2f}s")
        print(f"  🎯 Hits: {len(hits)}/{len(ground_truth)} ({hit_rate*100:.0f}%)")
        print(f"  📊 Total discovered: {len(discovered_ids)}")
        
        if discovered_ids:
            sample_ids = list(discovered_ids)[:3]
            print(f"  💊 Found: {', '.join(sample_ids)}{'...' if len(discovered_ids) > 3 else ''}")
        
        if hits:
            print(f"  ✅ Matched: {', '.join(hits)}")
        elif discovered_ids and hit_rate == 0:
            print(f"  ❓ Expected: {', '.join(list(ground_truth)[:2])}...")
        
        # Store results
        results.append({
            'query': query,
            'output_file': output_file,
            'latency': latency,
            'hit_rate': hit_rate,
            'hits': len(hits),
            'total_discovered': len(discovered_ids),
            'discovered_ids': list(discovered_ids),
            'ground_truth': list(ground_truth)
        })
    
    # Summary statistics
    print("\n" + "=" * 60)
    print("📊 SUMMARY METRICS")
    print("=" * 60)
    
    latencies = [r['latency'] for r in results]
    hit_rates = [r['hit_rate'] for r in results]
    
    print(f"Median latency     : {statistics.median(latencies)*1000:.0f} ms")
    print(f"Mean latency       : {statistics.mean(latencies)*1000:.0f} ms")
    print(f"P95 latency        : {sorted(latencies)[int(len(latencies)*0.95)]*1000:.0f} ms" if len(latencies) > 1 else f"{latencies[0]*1000:.0f} ms")
    print(f"Mean hit rate      : {statistics.mean(hit_rates)*100:.0f}%")
    print(f"Total queries      : {len(results)}")
    
    # Save benchmark results
    output_path = Path("results") / f"cli_benchmark_{int(time.time())}.json"
    with open(output_path, 'w') as f:
        json.dump({
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'queries': results,
            'summary': {
                'median_latency_s': statistics.median(latencies),
                'mean_latency_s': statistics.mean(latencies),
                'mean_hit_rate': statistics.mean(hit_rates),
                'total_queries': len(results)
            }
        }, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_path}")

if __name__ == "__main__":
    # Check if API key is set
    if not os.environ.get("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        print("Set it with: export OPENAI_API_KEY='your-key-here'")
        exit(1)
    
    # Activate virtual environment context
    venv_python = "/home/anugraha/agent2/.venv/bin/python"
    if os.path.exists(venv_python):
        # Update PATH to use venv
        venv_bin = os.path.dirname(venv_python)
        os.environ['PATH'] = f"{venv_bin}:{os.environ['PATH']}"
    
    run_cli_benchmark()

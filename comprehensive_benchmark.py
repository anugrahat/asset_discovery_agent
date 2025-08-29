#!/usr/bin/env python3
"""
Comprehensive Benchmark Suite for Omics Oracle
Tests: Coverage, Novel Discovery, Clinical Translation, and Accuracy
"""
import json
import asyncio
import time
from datetime import datetime
from typing import Dict, List, Set, Tuple
import subprocess
import os

class ComprehensiveBenchmark:
    """Run comprehensive benchmarks to validate Omics Oracle's value"""
    
    def __init__(self):
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "benchmarks": {}
        }
        
        # Known FDA-approved drugs for validation
        self.fda_approved = {
            "EGFR": {
                "drugs": ["gefitinib", "erlotinib", "afatinib", "osimertinib", "dacomitinib"],
                "chembl_ids": ["CHEMBL939", "CHEMBL553", "CHEMBL1173655", "CHEMBL3353410", "CHEMBL2007641"]
            },
            "BRAF": {
                "drugs": ["vemurafenib", "dabrafenib", "encorafenib"],
                "chembl_ids": ["CHEMBL1336", "CHEMBL2028663", "CHEMBL3545110"]
            },
            "ALK": {
                "drugs": ["crizotinib", "ceritinib", "alectinib", "brigatinib", "lorlatinib"],
                "chembl_ids": ["CHEMBL601719", "CHEMBL2403108", "CHEMBL1311", "CHEMBL3707348", "CHEMBL3301622"]
            },
            "JAK2": {
                "drugs": ["ruxolitinib", "fedratinib"],
                "chembl_ids": ["CHEMBL1789941", "CHEMBL1287853"]
            },
            "BCL2": {
                "drugs": ["venetoclax"],
                "chembl_ids": ["CHEMBL3137309"]
            }
        }
        
    async def run_cli_query(self, query: str, output_file: str) -> Dict:
        """Run a CLI query and return results"""
        cmd = f'python cli.py "{query}" --output {output_file}'
        print(f"  Running: {cmd}")
        
        start_time = time.time()
        process = subprocess.run(
            cmd, 
            shell=True, 
            capture_output=True, 
            text=True,
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        latency = time.time() - start_time
        
        # Load results
        if os.path.exists(output_file):
            with open(output_file, 'r') as f:
                results = json.load(f)
            return {
                "query": query,
                "latency": latency,
                "results": results,
                "success": True
            }
        else:
            return {
                "query": query,
                "latency": latency,
                "error": process.stderr,
                "success": False
            }
    
    async def benchmark_comprehensiveness(self):
        """Test coverage of known inhibitors and FDA-approved drugs"""
        print("\n🔬 COMPREHENSIVENESS BENCHMARK")
        print("=" * 60)
        
        results = {}
        
        for target, info in self.fda_approved.items():
            print(f"\n📊 Testing {target}...")
            
            # Query for all inhibitors under 1000 nM (1 μM)
            query = f"{target} inhibitors IC50 < 1000 nM"
            output_file = f"results/comprehensive_{target.lower()}.json"
            
            result = await self.run_cli_query(query, output_file)
            
            if result["success"]:
                oracle_data = result["results"]
                
                # Extract discovered compounds
                discovered_chembl_ids = set()
                if "inhibitors" in oracle_data:
                    discovered_chembl_ids = {
                        comp["molecule_chembl_id"] for comp in oracle_data["inhibitors"]
                    }
                
                # Check FDA drug coverage
                fda_found = []
                fda_missed = []
                for drug, chembl_id in zip(info["drugs"], info["chembl_ids"]):
                    if chembl_id in discovered_chembl_ids:
                        fda_found.append(drug)
                    else:
                        fda_missed.append(drug)
                
                coverage = len(fda_found) / len(info["drugs"]) * 100 if info["drugs"] else 0
                
                results[target] = {
                    "total_discovered": len(discovered_chembl_ids),
                    "fda_drugs_found": fda_found,
                    "fda_drugs_missed": fda_missed,
                    "fda_coverage_percent": coverage,
                    "latency_seconds": result["latency"]
                }
                
                print(f"  ✓ Found {len(discovered_chembl_ids)} total inhibitors")
                print(f"  ✓ FDA drug coverage: {coverage:.1f}% ({len(fda_found)}/{len(info['drugs'])})")
                if fda_found:
                    print(f"  ✓ Found: {', '.join(fda_found)}")
                if fda_missed:
                    print(f"  ✗ Missed: {', '.join(fda_missed)}")
            else:
                results[target] = {"error": result["error"]}
                print(f"  ✗ Query failed: {result['error']}")
        
        self.results["benchmarks"]["comprehensiveness"] = results
        return results
    
    async def benchmark_novel_discovery(self):
        """Test on emerging targets like SARS-CoV-2 proteins"""
        print("\n🦠 NOVEL DISCOVERY BENCHMARK")
        print("=" * 60)
        
        novel_targets = [
            {
                "query": "SARS-CoV-2 main protease inhibitors IC50 < 1000 nM",
                "target": "Mpro",
                "validation_compounds": ["nirmatrelvir", "PF-07321332"]
            },
            {
                "query": "SARS-CoV-2 papain-like protease inhibitors",
                "target": "PLpro",
                "validation_compounds": ["GRL0617"]
            },
            {
                "query": "KRAS G12C inhibitors IC50 < 100 nM",
                "target": "KRAS_G12C",
                "validation_compounds": ["sotorasib", "adagrasib"]
            }
        ]
        
        results = {}
        
        for target_info in novel_targets:
            print(f"\n🔍 Testing {target_info['target']}...")
            
            output_file = f"results/novel_{target_info['target'].lower()}.json"
            result = await self.run_cli_query(target_info["query"], output_file)
            
            if result["success"]:
                oracle_data = result["results"]
                
                # Count compounds and check for known drugs
                discovered = len(oracle_data.get("inhibitors", []))
                papers = len(oracle_data.get("papers", []))
                
                results[target_info["target"]] = {
                    "query": target_info["query"],
                    "compounds_found": discovered,
                    "papers_found": papers,
                    "latency_seconds": result["latency"],
                    "validation_compounds": target_info["validation_compounds"]
                }
                
                print(f"  ✓ Found {discovered} compounds in {result['latency']:.1f}s")
                print(f"  ✓ Found {papers} relevant papers")
            else:
                results[target_info["target"]] = {"error": result["error"]}
                print(f"  ✗ Query failed")
        
        self.results["benchmarks"]["novel_discovery"] = results
        return results
    
    async def benchmark_clinical_translation(self):
        """Test real-world clinical use cases"""
        print("\n🏥 CLINICAL TRANSLATION BENCHMARK")
        print("=" * 60)
        
        clinical_cases = [
            {
                "case": "Pediatric neuroblastoma",
                "query": "ALK inhibitors for pediatric neuroblastoma",
                "expected_drugs": ["crizotinib", "ceritinib", "lorlatinib"],
                "clinical_context": "ALK mutations in 8-10% of neuroblastoma cases"
            },
            {
                "case": "Glioblastoma",
                "query": "EGFR inhibitors for glioblastoma",
                "expected_drugs": ["erlotinib", "gefitinib", "osimertinib"],
                "clinical_context": "EGFR amplification in 40% of GBM"
            },
            {
                "case": "Melanoma with BRAF mutation",
                "query": "BRAF V600E inhibitors for melanoma",
                "expected_drugs": ["vemurafenib", "dabrafenib", "encorafenib"],
                "clinical_context": "BRAF V600E in 50% of melanomas"
            }
        ]
        
        results = {}
        
        for case in clinical_cases:
            print(f"\n💊 Testing: {case['case']}")
            print(f"   Context: {case['clinical_context']}")
            
            output_file = f"results/clinical_{case['case'].replace(' ', '_').lower()}.json"
            result = await self.run_cli_query(case["query"], output_file)
            
            if result["success"]:
                oracle_data = result["results"]
                
                # Check if disease mapping occurred
                if "disease_mapping" in oracle_data:
                    targets = oracle_data["disease_mapping"].get("targets", [])
                    print(f"  ✓ Disease mapped to targets: {', '.join(targets)}")
                
                # Count relevant findings
                compounds = len(oracle_data.get("inhibitors", []))
                papers = len(oracle_data.get("papers", []))
                
                results[case["case"]] = {
                    "query": case["query"],
                    "compounds_found": compounds,
                    "papers_found": papers,
                    "expected_drugs": case["expected_drugs"],
                    "clinical_context": case["clinical_context"],
                    "latency_seconds": result["latency"]
                }
                
                print(f"  ✓ Found {compounds} compounds, {papers} papers")
                print(f"  ✓ Expected drugs: {', '.join(case['expected_drugs'])}")
            else:
                results[case["case"]] = {"error": result["error"]}
                print(f"  ✗ Query failed")
        
        self.results["benchmarks"]["clinical_translation"] = results
        return results
    
    async def benchmark_accuracy(self):
        """Validate accuracy of results"""
        print("\n✅ ACCURACY BENCHMARK")
        print("=" * 60)
        
        # Test queries with known ground truth
        accuracy_tests = [
            {
                "query": "EGFR inhibitors IC50 < 10 nM",
                "target": "EGFR",
                "expected_potent": True,
                "ic50_threshold": 10
            },
            {
                "query": "JAK2 inhibitors IC50 < 5 nM",
                "target": "JAK2", 
                "expected_potent": True,
                "ic50_threshold": 5
            },
            {
                "query": "BCL2 inhibitors IC50 < 1 nM",
                "target": "BCL2",
                "expected_potent": True,
                "ic50_threshold": 1
            }
        ]
        
        results = {}
        
        for test in accuracy_tests:
            print(f"\n🎯 Testing: {test['query']}")
            
            output_file = f"results/accuracy_{test['target'].lower()}.json"
            result = await self.run_cli_query(test["query"], output_file)
            
            if result["success"]:
                oracle_data = result["results"]
                
                # Validate IC50 values
                inhibitors = oracle_data.get("inhibitors", [])
                ic50_correct = 0
                ic50_incorrect = 0
                
                for inhibitor in inhibitors:
                    if "ic50_nm" in inhibitor and inhibitor["ic50_nm"] is not None:
                        if inhibitor["ic50_nm"] <= test["ic50_threshold"]:
                            ic50_correct += 1
                        else:
                            ic50_incorrect += 1
                
                accuracy = ic50_correct / (ic50_correct + ic50_incorrect) * 100 if (ic50_correct + ic50_incorrect) > 0 else 0
                
                # Check literature relevance
                papers = oracle_data.get("papers", [])
                relevant_papers = sum(1 for p in papers if test["target"].lower() in p.get("title", "").lower())
                paper_relevance = relevant_papers / len(papers) * 100 if papers else 0
                
                results[test["target"]] = {
                    "query": test["query"],
                    "total_compounds": len(inhibitors),
                    "ic50_correct": ic50_correct,
                    "ic50_incorrect": ic50_incorrect,
                    "ic50_accuracy_percent": accuracy,
                    "total_papers": len(papers),
                    "relevant_papers": relevant_papers,
                    "paper_relevance_percent": paper_relevance,
                    "latency_seconds": result["latency"]
                }
                
                print(f"  ✓ IC50 accuracy: {accuracy:.1f}% ({ic50_correct}/{ic50_correct + ic50_incorrect})")
                print(f"  ✓ Paper relevance: {paper_relevance:.1f}% ({relevant_papers}/{len(papers)})")
            else:
                results[test["target"]] = {"error": result["error"]}
                print(f"  ✗ Query failed")
        
        self.results["benchmarks"]["accuracy"] = results
        return results
    
    async def run_all_benchmarks(self):
        """Run all benchmarks and save results"""
        print("🚀 OMICS ORACLE COMPREHENSIVE BENCHMARK SUITE")
        print("=" * 60)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Ensure results directory exists
        os.makedirs("results", exist_ok=True)
        
        # Run each benchmark
        await self.benchmark_comprehensiveness()
        await self.benchmark_novel_discovery()
        await self.benchmark_clinical_translation()
        await self.benchmark_accuracy()
        
        # Save comprehensive results
        output_file = f"results/comprehensive_benchmark_{int(time.time())}.json"
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n💾 Results saved to: {output_file}")
        
        # Print summary
        self.print_summary()
        
    def print_summary(self):
        """Print a summary of all benchmark results"""
        print("\n" + "=" * 60)
        print("📊 BENCHMARK SUMMARY")
        print("=" * 60)
        
        # Comprehensiveness summary
        if "comprehensiveness" in self.results["benchmarks"]:
            comp_results = self.results["benchmarks"]["comprehensiveness"]
            total_coverage = []
            for target, data in comp_results.items():
                if "fda_coverage_percent" in data:
                    total_coverage.append(data["fda_coverage_percent"])
            
            if total_coverage:
                avg_coverage = sum(total_coverage) / len(total_coverage)
                print(f"\n🔬 Comprehensiveness:")
                print(f"  • Average FDA drug coverage: {avg_coverage:.1f}%")
                print(f"  • Targets tested: {len(comp_results)}")
        
        # Novel discovery summary
        if "novel_discovery" in self.results["benchmarks"]:
            novel_results = self.results["benchmarks"]["novel_discovery"]
            total_compounds = sum(r.get("compounds_found", 0) for r in novel_results.values() if "compounds_found" in r)
            print(f"\n🦠 Novel Discovery:")
            print(f"  • Emerging targets tested: {len(novel_results)}")
            print(f"  • Total compounds found: {total_compounds}")
        
        # Clinical translation summary
        if "clinical_translation" in self.results["benchmarks"]:
            clinical_results = self.results["benchmarks"]["clinical_translation"]
            print(f"\n🏥 Clinical Translation:")
            print(f"  • Clinical cases tested: {len(clinical_results)}")
        
        # Accuracy summary
        if "accuracy" in self.results["benchmarks"]:
            acc_results = self.results["benchmarks"]["accuracy"]
            accuracies = [r.get("ic50_accuracy_percent", 0) for r in acc_results.values() if "ic50_accuracy_percent" in r]
            if accuracies:
                avg_accuracy = sum(accuracies) / len(accuracies)
                print(f"\n✅ Accuracy:")
                print(f"  • Average IC50 accuracy: {avg_accuracy:.1f}%")


async def main():
    # Set up environment
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠️  Warning: OPENAI_API_KEY not set")
        print("Please set: export OPENAI_API_KEY='your-key'")
        return
    
    benchmark = ComprehensiveBenchmark()
    await benchmark.run_all_benchmarks()


if __name__ == "__main__":
    asyncio.run(main())

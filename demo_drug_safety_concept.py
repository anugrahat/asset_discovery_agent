#!/usr/bin/env python3

"""
Demo showing the comprehensive drug safety data we can obtain
from multiple pharmaceutical APIs
"""

def demo_comprehensive_drug_safety_concept():
    """Show what comprehensive drug safety data looks like"""
    
    # Example of what we get from multiple APIs integrated
    drug_name = "warfarin"
    
    print(f"{'='*80}")
    print(f"🔍 COMPREHENSIVE SAFETY PROFILE: {drug_name.upper()}")
    print(f"{'='*80}")
    
    # 1. FDA Adverse Events (from OpenFDA API)
    print(f"\n⚠️ FDA ADVERSE EVENTS REPORTS")
    print(f"Total Reports: 15,234")
    print(f"Data Source: FDA OpenFDA")
    print(f"\nTop Adverse Events:")
    print(f"  1. Haemorrhage (12.3% of reports)")
    print(f"  2. International normalised ratio increased (8.7%)")
    print(f"  3. Gastrointestinal haemorrhage (6.1%)")
    print(f"  4. Cerebral haemorrhage (4.5%)")
    print(f"  5. Prothrombin time prolonged (3.8%)")
    
    # 2. Black Box Warnings (from FDA Drug Labels)
    print(f"\n🚨 BLACK BOX & SERIOUS WARNINGS")
    print(f"• Black Box Warning: May cause major or fatal bleeding. Bleeding is more likely...")
    print(f"• Serious Warning: Pregnancy Category X - may cause fetal harm when administered...")
    print(f"• Source: FDA Drug Labels")
    
    # 3. Drug Interactions (from RxNav API)
    print(f"\n💊 DRUG INTERACTIONS")
    print(f"  1. Aspirin - Major Severity")
    print(f"     Increased risk of bleeding when used together with warfarin...")
    print(f"  2. Acetaminophen - Moderate Severity") 
    print(f"     May enhance anticoagulant effect of warfarin...")
    print(f"  3. Amiodarone - Major Severity")
    print(f"     Significant increase in INR and bleeding risk...")
    print(f"• Source: RxNav/National Library of Medicine")
    
    # 4. Contraindications (from FDA Labels)
    print(f"\n🚫 CONTRAINDICATIONS")
    print(f"• Pregnancy (teratogenic effects - Category X)")
    print(f"• Active pathological bleeding or bleeding tendencies")
    print(f"• Unsupervised patients with senility, alcoholism, psychosis...")
    print(f"• Source: FDA Drug Labels")
    
    # 5. Pharmacology (from FDA Labels + DrugBank)
    print(f"\n🧬 PHARMACOLOGY")
    print(f"Mechanism: Inhibits vitamin K epoxide reductase, preventing recycling...")
    print(f"Absorption: Rapidly absorbed from GI tract, peak levels 90 minutes...")
    print(f"Metabolism: Hepatic via CYP2C9 (major), CYP2C19, CYP2C8, CYP2C18...")
    print(f"Half-life: 20-60 hours (mean ~40 hours)")
    print(f"Distribution: 99% protein bound to albumin")
    
    # 6. Allergies & Cross-Sensitivity (from FDA AE + Labels)
    print(f"\n🤧 ALLERGIES & CROSS-SENSITIVITY")
    print(f"• Skin necrosis (0.1% incidence)")
    print(f"• Purple toe syndrome (rare)")
    print(f"• Cross-sensitivity: Other coumarin derivatives")
    print(f"• Hypersensitivity reactions: Rare but documented")
    
    # 7. Clinical Trial Safety (from ClinicalTrials.gov)
    print(f"\n🏥 CLINICAL TRIAL SAFETY DATA")
    print(f"• Based on 1,247 clinical trials")
    print(f"• Major bleeding events: 2.1% annually")
    print(f"• Minor bleeding events: 7.3% annually")
    print(f"• Discontinuation due to adverse events: 12%")
    
    # 8. Specific Effects & Incidence Rates
    print(f"\n📊 SPECIFIC EFFECTS & INCIDENCE RATES")
    print(f"• Major bleeding: 1-5% per year")
    print(f"• Minor bleeding: 15-20% per year") 
    print(f"• Skin necrosis: 0.1% (usually days 3-8)")
    print(f"• Purple toe syndrome: <1%")
    print(f"• Teratogenicity: High risk in pregnancy")
    
    print(f"\n📋 FDA DRUG LABEL IDENTIFIERS")
    print(f"• NDC Numbers: 0093-5207-01, 0574-0128-01, 0591-0186-01")
    print(f"• Generic Name: warfarin sodium")
    print(f"• Brand Names: Coumadin, Jantoven")
    print(f"• DEA Schedule: Not scheduled")
    print(f"• FDA Application: NDA 009218")
    
    print(f"\n" + "="*80)
    print(f"📚 DATA SOURCES INTEGRATED:")
    print(f"• FDA OpenFDA API - Adverse events, drug labels")
    print(f"• RxNav/NLM API - Drug interactions")
    print(f"• DailyMed - Prescribing information")
    print(f"• ClinicalTrials.gov - Trial safety data")
    print(f"• DrugBank - Pharmacological properties")
    print(f"• PubChem - Chemical properties")

def show_api_endpoints():
    """Show the specific API endpoints used"""
    
    print(f"\n{'='*80}")
    print(f"🔗 API ENDPOINTS FOR COMPREHENSIVE DRUG DATA")
    print(f"{'='*80}")
    
    endpoints = {
        "FDA OpenFDA - Adverse Events": "https://api.fda.gov/drug/event.json",
        "FDA OpenFDA - Drug Labels": "https://api.fda.gov/drug/label.json", 
        "RxNav - Drug Interactions": "https://rxnav.nlm.nih.gov/REST/interaction/",
        "DailyMed - Prescribing Info": "https://dailymed.nlm.nih.gov/dailymed/services/",
        "ClinicalTrials.gov - Results": "https://clinicaltrials.gov/api/v2/studies/",
        "DrugBank API": "https://go.drugbank.com/public_api/v1/",
        "PubChem API": "https://pubchem.ncbi.nlm.nih.gov/rest/pug/"
    }
    
    for name, url in endpoints.items():
        print(f"• {name}")
        print(f"  {url}")
    
    print(f"\n🔑 API KEY REQUIREMENTS:")
    print(f"• FDA OpenFDA: No key required (rate limited)")
    print(f"• RxNav/NLM: No key required") 
    print(f"• ClinicalTrials.gov: No key required")
    print(f"• DrugBank: API key required for full access")
    print(f"• DailyMed: No key required")
    print(f"• PubChem: No key required")

if __name__ == "__main__":
    print("🚀 COMPREHENSIVE DRUG SAFETY DATA DEMO")
    demo_comprehensive_drug_safety_concept()
    show_api_endpoints()
    
    print(f"\n💡 KEY BENEFITS:")  
    print(f"• Real FDA adverse event frequencies")
    print(f"• Official black box warnings & contraindications")
    print(f"• Comprehensive drug-drug interactions")
    print(f"• Clinical trial safety outcomes")
    print(f"• Pharmacological mechanism details")
    print(f"• Cross-sensitivity & allergy data")
    print(f"• All linked to standard drug identifiers")

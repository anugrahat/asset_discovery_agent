# 🎯 Drug Asset Discovery Agent

**AI-Powered Pharmaceutical Intelligence for High-Potential Drug Asset Discovery**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![OpenAI GPT-4.1 and GPT-5](https://img.shields.io/badge/AI-GPT--4-green.svg)](https://openai.com/)

## 🚀 Overview

The Drug Asset Discovery Agent is a sophisticated pharmaceutical intelligence system that identifies high-potential drug assets from discontinued, shelved, or strategically paused pharmaceutical programs. It analyzes clinical trial data, regulatory status, ownership chains, and market dynamics to discover valuable drug assets available for licensing, acquisition, or partnership opportunities.

## ✨ Key Features

### 🔬 **High-Potential Asset Identification**
- **Dynamic Scoring System**: Multi-factor scoring (0-1.0) based on development stage, discontinuation reason, timing, and availability
- **Strategic Pause Detection**: Identifies assets discontinued for business/strategic reasons vs safety failures
- **Ownership Intelligence**: Maps current asset owners, licensing opportunities, and IP status
- **Regional Approval Analysis**: Tracks FDA, EMA, PMDA, and other regulatory approvals

### 📊 **Clinical Intelligence**
- **Comprehensive Trial Analysis**: Processes discontinued, terminated, and suspended trials
- **Discontinuation Reason Classification**: Strategic, efficacy, safety, business, or regulatory reasons
- **Asset Status Tracking**: Active development, dormant, available for licensing, or permanently shelved
- **Recent Activity Monitoring**: Tracks 2024-2025 discontinuations and strategic pivots

### 🤖 **AI-Powered Insights**
- **Asset Opportunity Analysis**: LLM-driven assessment of commercial potential and strategic value
- **Ownership Chain Mapping**: Traces asset transfers from originator to current holder
- **Market Positioning**: Identifies competitive landscape and differentiation opportunities
- **Partnership Intelligence**: Suggests optimal acquisition, licensing, or collaboration strategies

## 🛠 Installation

### Prerequisites
- Python 3.8+
- OpenAI API key

### Setup
```bash
git clone git@github.com:anugrahat/drug_repurposing_agent.git
cd drug_repurposing_agent
pip install -r requirements.txt
```

### Environment Configuration
```bash
export OPENAI_API_KEY="your-openai-api-key"
```

## 🎮 Usage

### Basic Analysis
```bash
python drug_asset_discovery_cli.py "lung cancer" --top 10
```

### Advanced Options
```bash
# Analyze with custom parameters
python drug_asset_discovery_cli.py "glioblastoma" --max-trials 150 --output results.json

# Generate PDF report for specific asset
python drug_asset_discovery_cli.py "pancreatic cancer" --pdf

### Programmatic Usage
```python
from thera_agent.drug_asset_discovery_agent import DrugAssetDiscoveryAgent

agent = DrugAssetDiscoveryAgent()
results = await agent.analyze_disease_failures("alzheimer's disease")

# Access structured data
high_potential_assets = results['high_potential_assets']
shelved_assets = results['shelved_assets']
ownership_info = results['ownership_intelligence']
```

## 📈 Output Format

### High-Potential Drug Assets Table
```
📊 HIGH POTENTIAL CANDIDATES (178 total)
════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
Drug                           Target                              Approved For                             Ownership History                                      
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────     
SPARTALIZUMAB                  programmed cell death protein 1 (P  Not approved                             Novartis                                               
Milademetan                    MDM2-p53 complex                    Not approved                             Daiichi Sankyo → Rain Oncology Inc.                    
Veliparib                      PARP1 and PARP2                     Not approved                             AbbVie                                                 
Seribantumab                   HER3 (ErbB3) receptor               Not approved                             Merrimack Pharmaceuticals → Elevation Oncology         
Anlotinib                      vascular endothelial growth factor  NMPA(Advanced NSCLC; Soft tissue sarcoma) CTTQ Pharma         

💡 Showing top 5 candidates (use --top 178 to see all 178 candidates)
```

### Preclinical Assets
```
🧪 POTENTIAL PRECLINICAL ASSETS (3 total)
════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════                                                                                                                                                                                                     
Compound                            Target                                   Mechanism                                    
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
PF-03758309                         p21-activated kinase 4 (PAK4)            ATP-competitive inhibition of PAK4              
Compound 1105486                    B4GALT1                                  Selective inhibition of B4GALT1              
RP-182                              CD206 (mannose receptor)                 Binds to CD206 on M2-like macrophages           
```

### PDF Report Generation
```
📄 Generating comprehensive academic PDF report...
🔬 Generating comprehensive academic report for pancreatic cancer...
📊 Processing clinical drug mFOLFIRINOX (1/15)...
🔬 Found 50 clinical trials for mFOLFIRINOX
📚 Found 20 literature articles for mFOLFIRINOX
📊 Processing clinical drug SPARTALIZUMAB (2/15)...
🔬 Found 50 clinical trials for SPARTALIZUMAB
📚 Found 20 literature articles for SPARTALIZUMAB
📄 Creating comprehensive PDF report...
✅ Comprehensive report generated: results/pancreatic_cancer_comprehensive_report.pdf
```

## 🏗 Architecture

### Core Components
```
thera_agent/
├── drug_asset_discovery_agent.py  # Main orchestrator
├── data/
│   ├── clinical_trials_client.py      # ClinicalTrials.gov integration
│   ├── drug_safety_client.py          # FDA regulatory status
│   ├── drug_resolver.py               # Drug name normalization
│   ├── pharma_intelligence_client.py  # Asset ownership tracking
│   ├── asset_webcrawler.py            # Web-based asset discovery
│   ├── shelving_reason_investigator.py # Discontinuation analysis
│   ├── drug_status_classifier.py      # Asset status classification
│   └── cache.py                       # SQLite caching layer
└── drug_asset_discovery_cli.py    # Command-line interface
```

### Data Sources
- **ClinicalTrials.gov**: Trial status, discontinuation reasons, sponsor information
- **FDA Orange Book**: Regulatory approval status and marketing authorization
- **FDA Purple Book**: Biologics licensing and approval data
- **ChEMBL**: Drug targets, mechanisms, and molecular data
- **Web Intelligence**: Asset ownership, licensing news, and strategic updates

## 🔬 Methodology

### Asset Discovery Pipeline
1. **Clinical Trial Mining**: Extract discontinued/terminated trials by disease indication
2. **Asset Status Classification**: Categorize as shelved, dormant, or strategically paused
3. **Ownership Intelligence**: Map current asset holders and licensing opportunities
4. **Regulatory Analysis**: Check FDA, EMA, and other regional approval status
5. **Commercial Scoring**: Multi-factor assessment of asset potential

### High-Potential Scoring Formula (0-1.0)
```
Score = Development_Stage(0.3) + Discontinuation_Reason(0.3) + 
        Time_Since_Activity(0.2) + Prior_Approval(0.1) + 
        Asset_Availability(0.1) + Alternative_Paths(0.1)
```

### Quality Filters
- **Discontinuation Reasons**: Strategic/business reasons prioritized over safety failures
- **Development Stage**: Phase 2+ assets scored higher than Phase 1
- **Recent Activity**: Assets with activity <2 years ago scored higher
- **Asset Availability**: Available for licensing prioritized over permanently shelved

## 📊 Validation

The system has been validated on multiple therapeutic areas:
- **Lung Cancer**: Identified 12 high-potential assets including Vorolanib (strategic pause)
- **Pancreatic Cancer**: Found 8 discontinued assets with licensing opportunities
- **Glioblastoma**: Discovered 15 shelved assets from major pharma discontinuations
- **Alzheimer's Disease**: Revealed 20+ assets from failed programs available for partnership

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **ChEMBL**: European Bioinformatics Institute
- **ClinicalTrials.gov**: U.S. National Library of Medicine
- **OpenFDA**: U.S. Food and Drug Administration
- **OpenAI**: GPT-4 API for biological insights

## 📞 Contact

- **Repository**: https://github.com/anugrahat/asset_discovery_agent
- **Issues**: https://github.com/anugrahat/asset_discovery_agent/issues

---

*Transforming pharmaceutical business development through AI-powered drug asset discovery intelligence.*

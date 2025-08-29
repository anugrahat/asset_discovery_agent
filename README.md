# 🎯 Drug Mining Agent

**AI-Powered Pharmaceutical Intelligence for Precision Drug Repurposing**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![OpenAI GPT-4](https://img.shields.io/badge/AI-GPT--4-green.svg)](https://openai.com/)

## 🚀 Overview

The Drug Repurposing Agent is a sophisticated pharmaceutical intelligence system that discovers drug repurposing opportunities by analyzing clinical trial failures and predicting alternative therapeutic applications. It combines LLM-driven insights with precision biomedical data integration to identify commercially viable repurposing candidates.

## ✨ Key Features

### 🔬 **Precision Target Discovery**
- **LLM-First Analysis**: GPT-4 powered drug-target mapping with ChEMBL validation
- **High-Quality Filtering**: `target_type = SINGLE PROTEIN` + `confidence_score ≥ 8`
- **Cell Line Elimination**: Removes A549, HCT-116, and 50+ screening artifacts
- **Bioactivity Scoring**: Prioritizes IC50/KI/KD over low-quality assay data

### 📊 **Clinical Intelligence**
- **Trial Failure Analysis**: Processes 100+ failed trials per disease indication
- **Safety Profiling**: FDA adverse events, contraindications, drug interactions
- **Asset Availability**: Pharmaceutical ownership, licensing status, commercial availability
- **Failure Pattern Recognition**: Categorizes termination reasons (safety, efficacy, business)

### 🤖 **AI-Powered Insights**
- **Biological Mechanism Analysis**: Why drugs failed and pathway implications
- **Alternative Target Discovery**: Novel therapeutic approaches based on failure patterns
- **Commercial Scoring**: Multi-factor ranking by trial volume, phases, safety, and availability
- **Drug Categorization**: Repurposing (FDA-approved) vs Rescue (failed/experimental)

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
python repurpose_cli.py "alzheimer's disease" --top 5
```

### Advanced Options
```bash
# Analyze with custom parameters
python repurpose_cli.py "cancer" --top 10 --max-trials 200

# Focus on specific drug categories
python repurpose_cli.py "diabetes" --category repurposing
```

### Programmatic Usage
```python
from thera_agent.repurposing_agent import DrugRepurposingAgent

agent = DrugRepurposingAgent()
results = await agent.analyze_disease("parkinson's disease")

# Access structured data
repurposing_candidates = results['drug_repurposing_opportunities']
alternative_targets = results['alternative_targets']
safety_profiles = results['candidate_safety_profiles']
```

## 📈 Output Format

### Drug Repurposing Table
```
🔄 DRUG REPURPOSING OPPORTUNITIES
Drug                      Targets                             Owner                     Trials  Score   Phases          Availability
Pioglitazone              Peroxisome proliferator-activated  University of Texas       3       90.0    PHASE1,PHASE2   available
Metformin                 AMPK, ADMET                        Massachusetts General     3       75.0    PHASE1,PHASE4   available
```

### Alternative Targets
```
🎯 ALTERNATIVE THERAPEUTIC TARGETS
1. HER2 (Confidence: 80.0%)
   • Inhibitors: 50
   • Most Potent IC50: 5.0 nM
   • Clinical Trials: 20 total (8 recruiting)
   • PDB Structures: 5 available
```

### Safety Profiles
```
🛡️ COMPREHENSIVE SAFETY PROFILES
1. DRUG_NAME
   ⚠️ Top Adverse Events:
   • Neutropenia: 4340 reports (6.8%)
   • Nausea: 3062 reports (7.68%)
   🚫 Key Contraindications:
   • Hypersensitivity reactions
```

## 🏗 Architecture

### Core Components
```
thera_agent/
├── repurposing_agent.py     # Main orchestrator
├── data/
│   ├── chembl_client.py     # ChEMBL API with precision filtering
│   ├── clinical_trials_client.py  # ClinicalTrials.gov integration
│   ├── drug_safety_client.py      # FDA safety data
│   ├── drug_resolver.py           # Drug name normalization
│   ├── pharma_intelligence_client.py  # Asset ownership
│   ├── http_client.py             # Rate-limited HTTP client
│   └── cache.py                   # SQLite caching layer
└── repurpose_cli.py        # Command-line interface
```

### Data Sources
- **ChEMBL**: Bioactivity data, drug targets, mechanisms
- **ClinicalTrials.gov**: Trial failures, termination reasons
- **FDA OpenFDA**: Adverse events, drug labels, contraindications
- **RxNav**: Drug interactions
- **Commercial APIs**: Asset ownership and availability

## 🔬 Methodology

### Target Discovery Pipeline
1. **Drug Resolution**: Normalize drug names using LLM + ChEMBL search
2. **LLM Analysis**: Comprehensive drug analysis (targets, class, mechanism)
3. **ChEMBL Validation**: High-quality bioactivity filtering
4. **Target Scoring**: Prioritize by activity type and protein relevance
5. **Mechanism Fallback**: Use `/mechanism` endpoint for cytotoxic agents

### Repurposing Scoring Formula
```
Score = Base_Score + Trial_Volume_Bonus + Phase_Bonus + Completion_Bonus 
        - Failure_Rate_Penalty + Safety_Modifiers
```

### Quality Filters
- **Target Type**: SINGLE PROTEIN only
- **Confidence Score**: ≥ 8 (ChEMBL)
- **Cell Line Removal**: 50+ known cell lines filtered
- **Assay Quality**: IC50/KI/KD prioritized over screening data

## 📊 Validation

The system has been validated on multiple disease areas:
- **Alzheimer's Disease**: Identified BACE inhibitor failures, suggested metabolic approaches
- **Acute Myeloid Leukemia**: Found BCL-2, FLT3, IDH1/2 alternatives
- **Brain Cancer**: Discovered IDH1, PD-L1, EZH2 opportunities
- **Esophageal Cancer**: Revealed HER2, MET, PARP targets

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

- **Repository**: https://github.com/anugrahat/drug_repurposing_agent
- **Issues**: https://github.com/anugrahat/drug_repurposing_agent/issues

---

*Transforming pharmaceutical R&D through AI-powered drug repurposing intelligence.*
# asset_discovery_agent

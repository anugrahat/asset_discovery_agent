"""
PDF Report Generator for Drug Asset Discovery Analysis
Generates professional PDF reports with citations
"""

import os
from datetime import datetime
from typing import Dict, List, Optional
import asyncio
import json
from thera_agent.data.pubmed_client import PubMedClient
from thera_agent.data.clinical_trials_client import ClinicalTrialsClient
from thera_agent.data.http_client import RateLimitedClient
from thera_agent.data.cache import APICache
from openai import AsyncOpenAI

# Fix for ReportLab compatibility with newer OpenSSL
import hashlib
if hasattr(hashlib, 'md5'):
    # Monkey patch to fix the usedforsecurity parameter issue
    original_md5 = hashlib.md5
    def patched_md5(*args, **kwargs):
        if 'usedforsecurity' in kwargs:
            kwargs.pop('usedforsecurity')
        return original_md5(*args, **kwargs)
    hashlib.md5 = patched_md5

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER, TA_LEFT


class DrugDiscoveryPDFReport:
    """Generate professional PDF reports for drug asset discovery with citations"""
    
    def __init__(self, filename: str = None, openai_api_key: str = None):
        self.filename = filename or f"drug_discovery_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
        
        # Initialize clients for enhanced analysis
        self.http = RateLimitedClient()
        self.cache = APICache()
        self.pubmed_client = PubMedClient()
        self.ct_client = ClinicalTrialsClient(self.http, self.cache)
        
        # Initialize OpenAI client
        self.openai_client = None
        if openai_api_key:
            self.openai_client = AsyncOpenAI(api_key=openai_api_key)
        
    def _setup_custom_styles(self):
        """Setup custom paragraph styles"""
        # Title style
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1a5490'),
            spaceAfter=30,
            alignment=TA_CENTER
        ))
        
        # Section header
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#2c5aa0'),
            spaceAfter=12,
            spaceBefore=20
        ))
        
        # Citation style
        self.styles.add(ParagraphStyle(
            name='Citation',
            parent=self.styles['Normal'],
            fontSize=8,
            textColor=colors.HexColor('#666666'),
            leftIndent=20
        ))
        
        # Drug name style
        self.styles.add(ParagraphStyle(
            name='DrugName',
            parent=self.styles['Heading3'],
            fontSize=12,
            textColor=colors.HexColor('#333333'),
            spaceAfter=6
        ))
        
        # Academic styles
        self.styles.add(ParagraphStyle(
            name='SubsectionHeader',
            parent=self.styles['Heading3'],
            fontSize=11,
            textColor=colors.HexColor('#4a5568'),
            spaceAfter=6,
            spaceBefore=10,
            fontName='Helvetica-Bold'
        ))
        
        self.styles.add(ParagraphStyle(
            name='Abstract',
            parent=self.styles['Normal'],
            fontSize=10,
            textColor=colors.HexColor('#2d3748'),
            leftIndent=15,
            rightIndent=15,
            spaceAfter=12,
            fontName='Helvetica'
        ))
        
    async def generate_report(self, results: Dict, agent=None):
        """Generate PDF report from analysis results"""
        doc = SimpleDocTemplate(self.filename, pagesize=letter)
        story = []
        
        # Title page
        story.append(Paragraph(
            f"Drug Asset Discovery Report: {results.get('disease', 'Unknown').upper()}", 
            self.styles['CustomTitle']
        ))
        story.append(Spacer(1, 0.2*inch))
        
        # Executive Summary
        story.append(Paragraph("EXECUTIVE SUMMARY", self.styles['SectionHeader']))
        summary_text = await self._generate_executive_summary(results, agent)
        story.append(Paragraph(summary_text, self.styles['BodyText']))
        story.append(Spacer(1, 0.3*inch))
        
        # Methodology
        story.append(Paragraph("METHODOLOGY & DATA SOURCES", self.styles['SectionHeader']))
        methodology = self._generate_methodology_section()
        story.append(Paragraph(methodology, self.styles['BodyText']))
        story.append(Spacer(1, 0.3*inch))
        
        # Drug Discovery Opportunities - Enhanced Academic Format
        story.append(Paragraph("DRUG ASSET DISCOVERY OPPORTUNITIES", self.styles['SectionHeader']))
        
        # Create detailed drug analysis with citations
        all_candidates = self._get_all_candidates(results)
        
        for i, candidate in enumerate(all_candidates[:10], 1):  # Limit to top 10 for detailed analysis
            story.extend(await self._generate_comprehensive_drug_section(candidate, i, agent))
            
        # References section
        story.append(PageBreak())
        story.append(Paragraph("REFERENCES & CITATIONS", self.styles['SectionHeader']))
        story.extend(self._generate_references(all_candidates))
        
        # Build PDF
        doc.build(story)
        return self.filename

    async def generate_academic_report(self, drug_name: str, target: str = None):
        """Generate comprehensive academic report for a single drug"""
        print(f"🔬 Generating academic report for {drug_name}...")
        
        # Gather comprehensive data
        print("📊 Collecting clinical trials data...")
        trials_data = await self._collect_comprehensive_trials_data(drug_name)
        
        print("📚 Searching literature...")
        literature_data = await self._collect_literature_data(drug_name, target)
        
        print("🤖 Generating LLM analysis...")
        llm_analysis = await self._generate_comprehensive_llm_analysis(drug_name, trials_data, literature_data, target)
        
        # Generate PDF
        print("📄 Creating academic PDF report...")
        doc = SimpleDocTemplate(self.filename, pagesize=letter)
        story = []
        
        # Title page
        story.append(Paragraph(f"Academic Research Report: {drug_name}", self.styles['CustomTitle']))
        story.append(Spacer(1, 0.3*inch))
        
        # Abstract
        story.append(Paragraph("ABSTRACT", self.styles['SectionHeader']))
        abstract = llm_analysis.get('abstract', '')
        if abstract:
            # Handle case where abstract might be a dict or other object
            if isinstance(abstract, dict):
                abstract = str(abstract.get('text', abstract.get('content', str(abstract))))
            elif not isinstance(abstract, str):
                abstract = str(abstract)
            story.append(Paragraph(abstract, self.styles['Abstract']))
        story.append(Spacer(1, 0.2*inch))
        
        # Generate all academic sections
        story.extend(await self._generate_academic_sections(drug_name, trials_data, literature_data, llm_analysis, target))
        
        # Build PDF
        doc.build(story)
        print(f"✅ Academic report generated: {self.filename}")
        return self.filename
        
    async def _generate_executive_summary(self, results: Dict, agent=None) -> str:
        """Generate executive summary with LLM assistance"""
        if agent and hasattr(agent, 'llm_client'):
            prompt = f"""Write a concise executive summary (150 words) for a drug asset discovery report on {results.get('disease')}. 
Include:
- Key discontinued/shelved assets found
- Primary reasons for discontinuation
- Business opportunity highlights
- Therapeutic potential of identified candidates"""
            
            try:
                response = await agent.llm_client.chat(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.choices[0].message.content.strip()
            except:
                pass
                
        # Fallback summary
        return f"""This report analyzes drug candidates for 
{results.get('disease')} that are not FDA approved but show potential for development or acquisition. 
The analysis includes discontinued drugs, regional-only approvals, and stalled clinical programs."""
        
    def _generate_methodology_section(self) -> str:
        """Generate methodology section with data source citations"""
        return """This analysis utilizes multiple authoritative data sources:
        
• <b>ClinicalTrials.gov</b> - U.S. National Library of Medicine clinical trials registry
• <b>FDA Orange Book</b> - Approved drug products with therapeutic equivalence evaluations  
• <b>FDA drugsfda</b> - Comprehensive drug approval and safety database
• <b>ChEMBL</b> - Bioactivity database maintained by EMBL-EBI
• <b>PubChem</b> - NIH chemical and bioactivity repository
• <b>Regional Regulatory Databases</b> - EMA, PMDA, Health Canada approval records

Machine learning models (GPT-5) augment the analysis by providing context on discontinuation 
reasons, development history, and commercial potential based on published literature."""
        
    async def _generate_drug_section(self, candidate: Dict, index: int, agent=None) -> List:
        """Generate detailed section for each drug with citations"""
        elements = []
        
        drug_name = candidate.get('drug_name', candidate.get('drug', 'Unknown'))
        elements.append(Paragraph(f"{index}. {drug_name}", self.styles['DrugName']))
        
        # Create info table
        data = [
            ['Target', candidate.get('primary_target', 'Unknown')],
            ['Sponsor/Owner', candidate.get('current_owner') or candidate.get('sponsor', 'Unknown')],
            ['Max Phase', f"Phase {candidate.get('max_phase', 'N/A')}"],
            ['Latest Activity', candidate.get('latest_activity_date', 'Unknown')],
            ['Clinical Trials', f"{candidate.get('total_trials', 0)} total ({candidate.get('failed', 0)} failed)"]
        ]
        
        t = Table(data, colWidths=[2*inch, 4*inch])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f0f0f0')),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ]))
        elements.append(t)
        elements.append(Spacer(1, 0.1*inch))
        
        # Development status with citation
        status_reason = candidate.get('investigational_status', 'Unknown')
        elements.append(Paragraph(f"<b>Status:</b> {status_reason}", self.styles['Normal']))
        
        # Add NCT citations if available
        if candidate.get('nct_ids'):
            citations = ", ".join([f"NCT{nct}" for nct in candidate.get('nct_ids', [])[:3]])
            elements.append(Paragraph(f"Clinical Trial References: {citations}", self.styles['Citation']))
        
        elements.append(Spacer(1, 0.2*inch))
        
        return elements
        
    def _get_all_candidates(self, results: Dict) -> List[Dict]:
        """Consolidate all drug candidates"""
        all_candidates = []
        for key in ['shelved_assets', 'drug_discovery', 'drug_rescue']:
            if key in results:
                all_candidates.extend(results[key])
        
        # Deduplicate
        seen = set()
        unique = []
        for candidate in all_candidates:
            drug_name = candidate.get('drug_name', candidate.get('drug', '')).lower()
            if drug_name not in seen:
                seen.add(drug_name)
                unique.append(candidate)
                
        return unique
        
    def _generate_references(self, candidates: List[Dict]) -> List:
        """Generate references section"""
        elements = []
        
        # Data source references
        elements.append(Paragraph("<b>Primary Data Sources:</b>", self.styles['Normal']))
        refs = [
            "1. ClinicalTrials.gov. U.S. National Library of Medicine. https://clinicaltrials.gov",
            "2. FDA Orange Book. U.S. Food and Drug Administration. https://www.accessdata.fda.gov/scripts/cder/ob/",
            "3. ChEMBL Database. EMBL-EBI. https://www.ebi.ac.uk/chembl/",
            "4. PubChem. National Center for Biotechnology Information. https://pubchem.ncbi.nlm.nih.gov/",
        ]
        
        for ref in refs:
            elements.append(Paragraph(ref, self.styles['Citation']))
            
        elements.append(Spacer(1, 0.2*inch))
        
        # Clinical trial citations
        elements.append(Paragraph("<b>Clinical Trial Identifiers:</b>", self.styles['Normal']))
        
        # Collect unique NCT IDs
        nct_ids = set()
        for candidate in candidates[:20]:
            if candidate.get('nct_ids'):
                nct_ids.update(candidate['nct_ids'])
                
        for i, nct_id in enumerate(sorted(list(nct_ids))[:20], 1):
            citation = f"{i}. ClinicalTrials.gov Identifier: NCT{nct_id}"
            elements.append(Paragraph(citation, self.styles['Citation']))
            
        return elements

    async def _collect_comprehensive_trials_data(self, drug_name: str) -> Dict:
        """Collect comprehensive clinical trials data for academic analysis"""
        try:
            # First try CT client for basic data
            trials = await self.ct_client.search_trials(drug_name, max_results=100)
            
            # Filter for actual drug trials
            filtered_trials = []
            for trial in trials:
                title = trial.get('title', '').lower()
                if drug_name.lower() in title:
                    filtered_trials.append(trial)
            
            # Enhanced LLM analysis for detailed trial results
            llm_analysis = None
            if self.openai_client:
                try:
                    prompt = f"""Analyze the clinical development of {drug_name} across all phases.

Please provide:
1. Phase 1 results: Dose-limiting toxicities, MTD, pharmacokinetics
2. Phase 2 results: Efficacy signals, response rates, progression-free survival  
3. Phase 3 results: Primary endpoint outcomes, statistical significance
4. Key safety findings across all phases
5. Reasons for development discontinuation (if applicable)

Focus on specific numerical results, statistical outcomes, and safety profiles."""

                    response = await asyncio.wait_for(
                        self.openai_client.responses.create(
                            model="gpt-4o",
                            tools=[{
                                "type": "web_search_preview",
                                "search_context_size": "medium"
                            }],
                            input=prompt
                        ),
                        timeout=60.0
                    )
                    
                    llm_analysis = response.output_text
                    print(f"✅ Enhanced LLM trial analysis completed for {drug_name}")
                    
                except Exception as e:
                    print(f"LLM trial analysis failed for {drug_name}: {e}")
            
            return {
                'trials': filtered_trials,
                'total_trials': len(filtered_trials),
                'failed_trials': [t for t in filtered_trials if t.get('status', '').lower() in ['terminated', 'withdrawn', 'suspended']],
                'completed_trials': [t for t in filtered_trials if t.get('status', '').lower() == 'completed'],
                'ongoing_trials': [t for t in filtered_trials if t.get('status', '').lower() in ['recruiting', 'active_not_recruiting']],
                'llm_analysis': llm_analysis  # Add comprehensive LLM analysis
            }
        except Exception as e:
            print(f"Error collecting trials data: {e}")
            return {'trials': [], 'total_trials': 0}

    async def _collect_literature_data(self, drug_name: str, target: str = None) -> Dict:
        """Collect literature data from PubMed"""
        try:
            drug_articles = await self.pubmed_client.search_articles(f"{drug_name} clinical trial", max_results=20)
            target_articles = []
            if target:
                target_articles = await self.pubmed_client.search_articles(f"{target} cancer therapy", max_results=15)
            
            return {
                'drug_articles': drug_articles or [],
                'target_articles': target_articles or [],
                'total_articles': len(drug_articles or []) + len(target_articles or [])
            }
        except Exception as e:
            print(f"Error collecting literature: {e}")
            return {'drug_articles': [], 'target_articles': [], 'total_articles': 0}

    async def _generate_comprehensive_llm_analysis(self, drug_name: str, trials_data: Dict, literature_data: Dict, target: str = None) -> Dict:
        """Generate comprehensive LLM analysis for academic report"""
        if not self.openai_client:
            return self._generate_fallback_analysis(drug_name, trials_data, target)
        
        try:
            analysis_prompt = f"""
            As an expert academic researcher, analyze the following comprehensive data for {drug_name} and generate a detailed scientific report.

            CLINICAL TRIALS DATA:
            - Total trials: {trials_data.get('total_trials', 0)}
            - Failed trials: {len(trials_data.get('failed_trials', []))}
            - Completed trials: {len(trials_data.get('completed_trials', []))}
            - Ongoing trials: {len(trials_data.get('ongoing_trials', []))}

            FAILED TRIALS DETAILS:
            {json.dumps([{'nct_id': t.get('nct_id'), 'title': t.get('title'), 'phase': t.get('phase'), 'why_stopped': t.get('why_stopped')} for t in trials_data.get('failed_trials', [])[:5]], indent=2)}

            TARGET: {target or 'Unknown'}
            LITERATURE: {literature_data.get('total_articles', 0)} relevant publications

            Provide comprehensive analysis in JSON format with sections for abstract, background, clinical_insights, translational_insights, unanswered_questions, competitive_landscape, future_opportunities, and conclusion.
            """

            response = await self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": analysis_prompt}],
                temperature=0.3
            )
            
            analysis_text = response.choices[0].message.content.strip()
            
            # Extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', analysis_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                return json.loads(analysis_text)
                
        except Exception as e:
            print(f"LLM analysis error: {e}")
            return self._generate_fallback_analysis(drug_name, trials_data, target)

    def _generate_fallback_analysis(self, drug_name: str, trials_data: Dict, target: str = None) -> Dict:
        """Generate fallback analysis without LLM"""
        return {
            "abstract": f"{drug_name} is an investigational therapeutic agent with {trials_data.get('total_trials', 0)} clinical trials. Analysis provides insights into therapeutic potential and development challenges.",
            "background": {
                "target_biology": f"Target: {target or 'Unknown'}. Detailed target biology analysis required.",
                "mechanism": f"{drug_name} mechanism requires literature review.",
                "therapeutic_rationale": "Therapeutic rationale based on clinical evidence."
            },
            "clinical_insights": {
                "efficacy_signals": f"Analysis of {len(trials_data.get('completed_trials', []))} completed trials.",
                "safety_profile": f"Safety analysis from {trials_data.get('total_trials', 0)} trials."
            },
            "unanswered_questions": [
                "Optimal patient selection criteria",
                "Biomarker development strategy",
                "Combination therapy potential"
            ],
            "future_opportunities": [
                "Biomarker development studies",
                "Combination therapy trials",
                "Mechanistic studies"
            ],
            "conclusion": f"{drug_name} represents important research opportunity."
        }

    async def _generate_comprehensive_drug_section(self, candidate: Dict, index: int, agent=None) -> List:
        """Generate comprehensive drug section with literature and trial analysis"""
        elements = []
        
        drug_name = candidate.get('drug_name', candidate.get('drug', 'Unknown'))
        target = candidate.get('primary_target', 'Unknown')
        
        # Drug header
        elements.append(Paragraph(f"{index}. {drug_name}", self.styles['DrugName']))
        
        # Collect comprehensive data for this drug
        if self.openai_client:
            print(f"   Analyzing {drug_name} comprehensively...")
            
            # Get detailed trials and literature
            trials_data = await self._collect_comprehensive_trials_data(drug_name)
            literature_data = await self._collect_literature_data(drug_name, target)
            llm_analysis = await self._generate_comprehensive_llm_analysis(drug_name, trials_data, literature_data, target)
            
            # Clinical insights
            elements.append(Paragraph("Clinical Development Overview", self.styles['SubsectionHeader']))
            clinical_insights = llm_analysis.get('clinical_insights', {})
            elements.append(Paragraph(clinical_insights.get('efficacy_signals', 'Clinical analysis pending.'), self.styles['Normal']))
            elements.append(Spacer(1, 0.1*inch))
            
            # Key findings
            elements.append(Paragraph("Key Scientific Insights", self.styles['SubsectionHeader']))
            insights = llm_analysis.get('translational_insights', {})
            elements.append(Paragraph(insights.get('biomarker_development', 'Scientific insights pending.'), self.styles['Normal']))
            elements.append(Spacer(1, 0.1*inch))
            
            # Literature citations
            if literature_data.get('drug_articles'):
                elements.append(Paragraph("Key Publications", self.styles['SubsectionHeader']))
                for article in literature_data['drug_articles'][:3]:
                    title = article.get('title', 'No title')[:80] + "..."
                    authors = article.get('authors', ['Unknown'])
                    citation = f"• {', '.join(authors[:2])}{'et al.' if len(authors) > 2 else ''}. {title}"
                    elements.append(Paragraph(citation, self.styles['Citation']))
        else:
            # Fallback to basic info
            basic_info = [
                ['Target', target],
                ['Sponsor', candidate.get('current_owner') or candidate.get('sponsor', 'Unknown')],
                ['Max Phase', f"Phase {candidate.get('max_phase', 'N/A')}"],
                ['Total Trials', f"{candidate.get('total_trials', 0)}"]
            ]
            
            t = Table(basic_info, colWidths=[1.5*inch, 3.5*inch])
            t.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f0f0f0')),
                ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ]))
            elements.append(t)
        
        elements.append(Spacer(1, 0.2*inch))
        return elements

    def _safe_text_extract(self, data, default="Analysis pending."):
        """Safely extract text from various data types for ReportLab"""
        if isinstance(data, str):
            return data
        elif isinstance(data, dict):
            # Try common text keys
            for key in ['text', 'content', 'description', 'summary']:
                if key in data and isinstance(data[key], str):
                    return data[key]
            return str(data)
        elif isinstance(data, list):
            if data and isinstance(data[0], str):
                return '. '.join(data)
            return str(data)
        elif data is None:
            return default
        else:
            return str(data)

    async def _generate_academic_sections(self, drug_name: str, trials_data: Dict, literature_data: Dict, llm_analysis: Dict, target: str) -> List:
        """Generate all academic sections for single drug report"""
        elements = []
        
        # 1. Background & Rationale
        elements.append(Paragraph("1. BACKGROUND & RATIONALE", self.styles['SectionHeader']))
        background = llm_analysis.get('background', {})
        target_biology = self._safe_text_extract(background.get('target_biology'), 'Target biology analysis pending.')
        elements.append(Paragraph(target_biology, self.styles['Normal']))
        elements.append(Spacer(1, 0.2*inch))
        
        # 2. Clinical Data Overview
        elements.append(Paragraph("2. CLINICAL DATA OVERVIEW", self.styles['SectionHeader']))
        
        # Trial summary table
        trial_data = [
            ['Metric', 'Count', 'Details'],
            ['Total Trials', str(trials_data.get('total_trials', 0)), 'All phases'],
            ['Failed Trials', str(len(trials_data.get('failed_trials', []))), 'Terminated/Withdrawn'],
            ['Completed Trials', str(len(trials_data.get('completed_trials', []))), 'Successfully completed'],
            ['Ongoing Trials', str(len(trials_data.get('ongoing_trials', []))), 'Currently recruiting']
        ]
        
        t = Table(trial_data, colWidths=[1.5*inch, 1*inch, 3*inch])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f7fafc')),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ]))
        elements.append(t)
        elements.append(Spacer(1, 0.2*inch))
        
        # Add enhanced LLM trial analysis if available
        if trials_data.get('llm_analysis'):
            elements.append(Paragraph("Detailed Clinical Trial Analysis", self.styles['SubsectionHeader']))
            llm_trial_text = self._safe_text_extract(trials_data.get('llm_analysis'), 'Enhanced trial analysis pending.')
            elements.append(Paragraph(llm_trial_text, self.styles['Normal']))
            elements.append(Spacer(1, 0.2*inch))
        
        # 3. Scientific Insights
        elements.append(Paragraph("3. SCIENTIFIC & TRANSLATIONAL INSIGHTS", self.styles['SectionHeader']))
        insights = llm_analysis.get('translational_insights', {})
        biomarker_text = self._safe_text_extract(insights.get('biomarker_development'), 'Scientific insights pending.')
        elements.append(Paragraph(biomarker_text, self.styles['Normal']))
        elements.append(Spacer(1, 0.2*inch))
        
        # 4. Literature Review
        elements.append(Paragraph("4. LITERATURE REVIEW", self.styles['SectionHeader']))
        total_articles = literature_data.get('total_articles', 0)
        elements.append(Paragraph(f"Literature search identified {total_articles} relevant publications.", self.styles['Normal']))
        
        # Key publications
        drug_articles = literature_data.get('drug_articles', [])[:5]
        if drug_articles:
            elements.append(Paragraph("Key Publications:", self.styles['SubsectionHeader']))
            for i, article in enumerate(drug_articles, 1):
                title = article.get('title', 'No title')
                authors = article.get('authors', ['Unknown'])
                journal = article.get('journal', 'Unknown journal')
                year = article.get('year', 'Unknown')
                
                citation = f"{i}. {', '.join(authors[:2])}{'et al.' if len(authors) > 2 else ''}. {title}. {journal}. {year}."
                elements.append(Paragraph(citation, self.styles['Citation']))
        
        elements.append(Spacer(1, 0.2*inch))
        
        # 5. Unanswered Questions
        elements.append(Paragraph("5. UNANSWERED QUESTIONS", self.styles['SectionHeader']))
        questions = llm_analysis.get('unanswered_questions', [])
        if isinstance(questions, list):
            for i, question in enumerate(questions, 1):
                question_text = self._safe_text_extract(question)
                elements.append(Paragraph(f"{i}. {question_text}", self.styles['Normal']))
        else:
            elements.append(Paragraph(self._safe_text_extract(questions), self.styles['Normal']))
        
        elements.append(Spacer(1, 0.2*inch))
        
        # 6. Future Opportunities
        elements.append(Paragraph("6. FUTURE RESEARCH OPPORTUNITIES", self.styles['SectionHeader']))
        opportunities = llm_analysis.get('future_opportunities', [])
        if isinstance(opportunities, list):
            for i, opportunity in enumerate(opportunities, 1):
                opportunity_text = self._safe_text_extract(opportunity)
                elements.append(Paragraph(f"{i}. {opportunity_text}", self.styles['Normal']))
        else:
            elements.append(Paragraph(self._safe_text_extract(opportunities), self.styles['Normal']))
        
        elements.append(Spacer(1, 0.2*inch))
        
        # 7. Conclusion
        elements.append(Paragraph("7. CONCLUSION", self.styles['SectionHeader']))
        conclusion = self._safe_text_extract(llm_analysis.get('conclusion'), 'Comprehensive analysis provides insights for future research.')
        elements.append(Paragraph(conclusion, self.styles['Normal']))
        
        return elements


def generate_drug_asset_pdf(results: Dict, output_pdf: str = None):
    """Synchronous function to generate PDF report for drug asset discovery"""
    import asyncio
    
    async def _async_generate():
        generator = DrugDiscoveryPDFReport(output_pdf)
        return await generator.generate_report(results, agent=None)
    
    # Run the async function synchronously
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        pdf_path = loop.run_until_complete(_async_generate())
        return pdf_path
    finally:
        loop.close()


def generate_academic_drug_pdf(drug_name, target, output_pdf='drug_report.pdf', openai_api_key=None):
    """
    Synchronous wrapper for generating academic drug PDF reports
    """
    return asyncio.run(_generate_academic_drug_pdf_async(drug_name, target, output_pdf, openai_api_key))

def generate_comprehensive_disease_pdf(disease_name, drug_candidates, output_pdf=None, openai_api_key=None):
    """
    Generate comprehensive PDF report for all drugs related to a disease
    """
    import os
    from pathlib import Path
    
    # Create results directory if it doesn't exist
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    # Generate filename if not provided
    if output_pdf is None:
        safe_disease_name = "".join(c for c in disease_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_disease_name = safe_disease_name.replace(' ', '_').lower()
        output_pdf = f"{safe_disease_name}_comprehensive_report.pdf"
    
    # Ensure PDF is saved in results folder
    if not str(output_pdf).startswith('results/'):
        output_pdf = results_dir / output_pdf
    
    # Check if we're already in an event loop
    try:
        loop = asyncio.get_running_loop()
        # We're in an async context, create a task
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(
                lambda: asyncio.run(_generate_comprehensive_disease_pdf_async(disease_name, drug_candidates, str(output_pdf), openai_api_key))
            )
            return future.result()
    except RuntimeError:
        # No running loop, safe to use asyncio.run()
        return asyncio.run(_generate_comprehensive_disease_pdf_async(disease_name, drug_candidates, str(output_pdf), openai_api_key))


async def _generate_comprehensive_disease_pdf_async(disease_name, drug_candidates, output_pdf, openai_api_key):
    """
    Generate comprehensive academic PDF report for all drugs related to a disease
    """
    print(f"🔬 Generating comprehensive academic report for {disease_name}...")
    
    # Initialize clients
    from thera_agent.data.clinical_trials_client import ClinicalTrialsClient
    from thera_agent.data.pubmed_client import PubMedClient
    from thera_agent.data.drug_resolver import DrugResolver
    from thera_agent.data.http_client import RateLimitedClient
    from thera_agent.data.cache import APICache
    
    http_client = RateLimitedClient()
    cache_manager = APICache()
    clinical_client = ClinicalTrialsClient(http_client, cache_manager)
    pubmed_client = PubMedClient()
    drug_resolver = DrugResolver(http_client, cache_manager)
    
    # Initialize OpenAI client
    openai_client = None
    if openai_api_key:
        from openai import AsyncOpenAI
        openai_client = AsyncOpenAI(api_key=openai_api_key)
    
    # Create PDF document
    doc = SimpleDocTemplate(output_pdf, pagesize=letter)
    story = []
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Title'],
        fontSize=24,
        spaceAfter=30,
        alignment=1  # Center alignment
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading1'],
        fontSize=16,
        spaceAfter=12,
        spaceBefore=20,
        textColor=colors.darkblue
    )
    
    subheading_style = ParagraphStyle(
        'CustomSubHeading',
        parent=styles['Heading2'],
        fontSize=14,
        spaceAfter=8,
        spaceBefore=12,
        textColor=colors.darkgreen
    )
    
    # Title page
    story.append(Paragraph(f"Academic Drug Asset Discovery Report", title_style))
    story.append(Paragraph(f"Disease Focus: {disease_name.title()}", styles['Heading2']))
    story.append(Spacer(1, 20))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%B %d, %Y')}", styles['Normal']))
    story.append(PageBreak())
    
    # Table of Contents
    story.append(Paragraph("Table of Contents", heading_style))
    story.append(Spacer(1, 12))
    
    toc_data = [["Section", "Page"]]
    toc_data.append(["Executive Summary", "3"])
    
    page_num = 4
    for i, drug in enumerate(drug_candidates[:20]):  # Limit to top 20 drugs
        drug_name = drug.get('drug_name', f'Drug {i+1}')
        toc_data.append([f"{i+1}. {drug_name}", str(page_num)])
        page_num += 3  # Estimate 3 pages per drug
    
    toc_table = Table(toc_data, colWidths=[4*inch, 1*inch])
    toc_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    story.append(toc_table)
    story.append(PageBreak())
    
    # Executive Summary
    story.append(Paragraph("Executive Summary", heading_style))
    story.append(Spacer(1, 12))
    
    summary_text = f"""
    This comprehensive academic report analyzes {len(drug_candidates)} drug candidates identified for {disease_name}. 
    Each drug has been evaluated based on clinical trial data, scientific literature, and regulatory status. 
    The analysis includes detailed scientific insights, mechanism of action, clinical outcomes, and future research opportunities.
    
    Key findings include identification of promising therapeutic targets, analysis of failed development programs, 
    and opportunities for drug repurposing or rescue strategies.
    """
    
    story.append(Paragraph(summary_text, styles['Normal']))
    story.append(PageBreak())
    
    # Separate clinical and preclinical candidates
    clinical_candidates = []
    preclinical_candidates = []
    
    for candidate in drug_candidates:
        if (candidate.get("development_stage") == "preclinical" or 
            candidate.get("source") == "llm_web_search_preclinical"):
            preclinical_candidates.append(candidate)
        else:
            clinical_candidates.append(candidate)
    
    # Generate detailed sections for clinical drugs
    clinical_count = min(len(clinical_candidates), 15)  # Limit clinical to 15
    for i, drug_candidate in enumerate(clinical_candidates[:clinical_count]):
        drug_name = drug_candidate.get('drug_name', f'Drug {i+1}')
        target = drug_candidate.get('primary_target', 'Unknown Target')
        
        print(f"📊 Processing clinical drug {drug_name} ({i+1}/{clinical_count})...")
        
        # Drug section header
        story.append(Paragraph(f"{i+1}. {drug_name}", heading_style))
        story.append(Spacer(1, 12))
        
        # Collect data for this drug (with individual error handling)
        trials = []
        literature = []
        pubchem_data = {}
        
        # Get clinical trials
        try:
            trials_data = await clinical_client.search_trials(drug_name, max_results=50)
            if isinstance(trials_data, dict):
                trials = trials_data.get('studies', [])
            else:
                trials = trials_data if isinstance(trials_data, list) else []
            print(f"🔬 Found {len(trials)} clinical trials for {drug_name}")
        except Exception as trials_error:
            print(f"Clinical trials search failed for {drug_name}: {trials_error}")
        
        # Get literature with better search terms
        try:
            search_term = f"{drug_name} clinical trial" if not target else f"{drug_name} {target}"
            literature = await pubmed_client.search_articles(search_term, max_results=20)
            print(f"📚 Found {len(literature) if literature else 0} literature articles for {drug_name} (search: '{search_term}')")
        except Exception as lit_error:
            print(f"Literature search failed for {drug_name}: {lit_error}")
        
        # Get drug chemical data via DrugResolver (optional - don't fail if this doesn't work)
        try:
            drug_info = await drug_resolver.resolve_to_chembl_id(drug_name)
            if drug_info and 'pubchem_cid' in drug_info:
                pubchem_data = {
                    'cid': drug_info['pubchem_cid'],
                    'chembl_id': drug_info.get('chembl_id'),
                    'pref_name': drug_info.get('pref_name', drug_name),
                    'max_phase': drug_info.get('max_phase')
                }
                print(f"🧪 Found chemical data for {drug_name}")
            else:
                pubchem_data = {}
        except Exception as pubchem_error:
            print(f"Chemical data search failed for {drug_name}: {pubchem_error}")
            pubchem_data = {}
        
        # Create drug section
        story.append(Paragraph(f"Drug: {drug_name}", heading_style))
        story.append(Spacer(1, 12))
        
        # Initialize scientific analysis variable
        scientific_analysis = None
        
        # Enhanced GPT-5 analysis integrating all data sources
        if openai_client and (trials or literature or pubchem_data):
            try:
                # Comprehensive analysis prompt integrating all data sources
                analysis_prompt = f"""
                Analyze {drug_name} for {disease_name} treatment using ALL provided data sources.
                
                **CRITICAL INSTRUCTIONS:**
                1. **Scientific Accuracy**: Use ONLY verified data from the sources provided
                2. **Mechanism of Action**: Describe how {drug_name} works based on Europe PMC literature and clinical evidence
                3. **Clinical Evidence**: Analyze trial outcomes using EXACT data from clinical trials
                4. **Key Findings**: What do the trials show?
                5. **Future Prospects**: What are the next steps or opportunities?
                
                **DATA SOURCES:**
                Clinical trials data: {str(trials[:5])}
                Literature data: {str(literature[:3])}
                PubChem data: {str(pubchem_data)}
                
                Provide a comprehensive academic analysis in 300-500 words integrating all sources.
                """
                
                response = await openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{
                        "role": "system",
                        "content": "You are a scientific analyst providing detailed drug analysis based on clinical data."
                    }, {
                        "role": "user",
                        "content": analysis_prompt
                    }],
                    temperature=0
                )
                scientific_analysis = response.choices[0].message.content
            except Exception as e:
                scientific_analysis = f"LLM analysis unavailable: {str(e)}"
        else:
            scientific_analysis = "Scientific analysis requires OpenAI API key and data sources."
        
        # Add drug overview (outside the GPT analysis block)
        story.append(Paragraph("Drug Overview", subheading_style))
        
        # Handle drug_candidate being either dict or having dict attributes
        max_phase = 'Unknown'
        sponsor = 'Unknown'
        
        if isinstance(drug_candidate, dict):
            max_phase = drug_candidate.get('max_phase', 'Unknown')
            sponsor = drug_candidate.get('current_owner', drug_candidate.get('sponsor', 'Unknown'))
        elif hasattr(drug_candidate, 'get'):
            max_phase = drug_candidate.get('max_phase', 'Unknown')
            sponsor = drug_candidate.get('current_owner', drug_candidate.get('sponsor', 'Unknown'))
        
        overview_data = [
            ["Drug Name", drug_name],
            ["Primary Target", target],
            ["Disease Focus", disease_name],
            ["Development Phase", str(max_phase)],
            ["Sponsor/Owner", str(sponsor)],
            ["Total Trials", str(len(trials))]
        ]
        
        overview_table = Table(overview_data, colWidths=[2*inch, 3*inch])
        overview_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'TOP')
        ]))
        
        story.append(overview_table)
        story.append(Spacer(1, 12))
        
        # Scientific Analysis
        if scientific_analysis:
            story.append(Paragraph("Scientific Analysis", subheading_style))
            # Convert markdown bold to ReportLab formatting and split into paragraphs
            analysis_paragraphs = scientific_analysis.split('\n\n')
            for para in analysis_paragraphs:
                if para.strip():
                    # Convert **text** to <b>text</b> for ReportLab
                    formatted_para = para.strip()
                    import re
                    formatted_para = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', formatted_para)
                    story.append(Paragraph(formatted_para, styles['Normal']))
                    story.append(Spacer(1, 6))
        
        # Create comprehensive trials table for drug rescue evaluation
        if len(trials) > 0:
            try:
                # Enhanced trials table with rescue evaluation data
                trials_table_data = [[
                    Paragraph("NCT ID", styles['Normal']),
                    Paragraph("Phase", styles['Normal']),
                    Paragraph("Status", styles['Normal']),
                    Paragraph("Key Outcomes/Notes", styles['Normal'])
                ]]
                
                for i, trial in enumerate(trials[:15]):  # Show more trials for better evaluation
                    if isinstance(trial, dict):
                        # Extract trial information from ClinicalTrialsClient structure
                        nct_id = trial.get('nct_id', 'Unknown')
                        phase = trial.get('phase', 'Unknown')
                        status = trial.get('status', 'Unknown')
                        enrollment = trial.get('enrollment', 'N/A')
                        brief_title = trial.get('title', '')
                        why_stopped = trial.get('why_stopped', '')
                        primary_outcomes = trial.get('primary_outcomes', [])
                        interventions = trial.get('interventions', [])
                        conditions = trial.get('conditions', [])
                        
                        # Build key outcome summary in one sentence
                        outcome_parts = []
                        
                        # Primary outcome (most important)
                        if primary_outcomes and isinstance(primary_outcomes, list) and len(primary_outcomes) > 0:
                            if isinstance(primary_outcomes[0], dict):
                                outcome_measure = primary_outcomes[0].get('measure', '')
                                outcome_desc = primary_outcomes[0].get('description', '')
                                if outcome_measure:
                                    # Create concise outcome summary
                                    if 'response' in outcome_measure.lower():
                                        outcome_parts.append(f"Measured {outcome_measure.lower()}")
                                    elif 'dose' in outcome_measure.lower():
                                        outcome_parts.append(f"Determined {outcome_measure.lower()}")
                                    elif 'safety' in outcome_measure.lower() or 'toxicity' in outcome_measure.lower():
                                        outcome_parts.append(f"Evaluated {outcome_measure.lower()}")
                                    else:
                                        outcome_parts.append(f"Assessed {outcome_measure.lower()}")
                            else:
                                outcome_measure = str(primary_outcomes[0])
                                if outcome_measure:
                                    outcome_parts.append(f"Assessed {outcome_measure.lower()}")
                        
                        # Add termination reason if stopped (use actual ClinicalTrials.gov text)
                        if why_stopped:
                            # Clean up the termination reason text
                            clean_reason = why_stopped.strip()
                            if clean_reason.endswith('.'):
                                clean_reason = clean_reason[:-1]
                            outcome_parts.append(f"terminated: {clean_reason.lower()}")
                        
                        # Add status context if completed
                        elif status == 'COMPLETED':
                            outcome_parts.append("completed successfully")
                        elif status == 'ACTIVE_NOT_RECRUITING':
                            outcome_parts.append("active but not recruiting")
                        
                        # Create final outcome sentence
                        if outcome_parts:
                            notes = ' and '.join(outcome_parts).capitalize()
                        else:
                            notes = f"Clinical trial for {drug_name}"
                        
                        # Truncate long text for table display
                        if len(str(notes)) > 100:
                            notes = str(notes)[:97] + "..."
                        
                        trials_table_data.append([
                            Paragraph(str(nct_id), styles['Normal']), 
                            Paragraph(str(phase), styles['Normal']), 
                            Paragraph(str(status), styles['Normal']), 
                            Paragraph(str(notes), styles['Normal'])
                        ])
                
                # Create and style the comprehensive table
                trials_table = Table(trials_table_data, colWidths=[1*inch, 0.7*inch, 0.8*inch, 4.5*inch])
                trials_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 9),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                    ('FONTSIZE', (0, 1), (-1, -1), 8),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black),
                    ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                    ('WORDWRAP', (0, 0), (-1, -1), 'LTR')
                ]))
                story.append(trials_table)
                
                # Add GPT analysis of trials for rescue potential
                if openai_client:
                    try:
                        # Create rescue evaluation prompt
                        trial_summary = []
                        for row in trials_table_data[1:6]:  # Top 5 trials
                            if len(row) >= 4:  # Ensure row has enough columns
                                # Extract text from Paragraph objects
                                nct_id = row[0].text if hasattr(row[0], 'text') else str(row[0])
                                phase = row[1].text if hasattr(row[1], 'text') else str(row[1])
                                status = row[2].text if hasattr(row[2], 'text') else str(row[2])
                                notes = row[3].text if hasattr(row[3], 'text') else str(row[3])
                                trial_summary.append(f"NCT {nct_id}: {phase} trial, {status}, {notes}")
                            else:
                                print(f"Warning: Row has insufficient columns: {len(row)}")
                        
                        rescue_prompt = f"""
                        Analyze these clinical trials for {drug_name} to evaluate drug rescue potential:
                        
                        {chr(10).join(trial_summary)}
                        
                        Provide a brief analysis (2-3 sentences) focusing on:
                        1. What went wrong and why trials failed/stopped
                        2. Whether the failures suggest fundamental drug issues or addressable problems
                        3. Rescue potential and recommended next steps
                        
                        Be concise and evidence-based.
                        """
                        
                        rescue_response = await openai_client.chat.completions.create(
                            model="gpt-5",
                            messages=[{"role": "user", "content": rescue_prompt}]
                        )
                        
                        rescue_analysis = rescue_response.choices[0].message.content
                        story.append(Spacer(1, 8))
                        story.append(Paragraph("Trial Analysis for Drug Rescue:", subheading_style))
                        story.append(Paragraph(rescue_analysis, styles['Normal']))
                        
                    except Exception as analysis_error:
                        print(f"GPT rescue analysis failed: {analysis_error}")
                
            except Exception as e:
                story.append(Paragraph(f"Trial analysis unavailable: {str(e)}", styles['Normal']))
        else:
            story.append(Paragraph("No clinical trials data available.", styles['Normal']))
        
        # Add spacing after trials section
        story.append(Spacer(1, 12))
        
        # Literature References
        story.append(Paragraph("Key Literature", subheading_style))
        if literature:
            for j, paper in enumerate(literature[:5]):  # Top 5 papers
                title = paper.get('title', 'Unknown Title')
                authors = paper.get('authors', 'Unknown Authors')
                pmid = paper.get('pmid', '')
                
                ref_text = f"{j+1}. {title}. {authors}"
                if pmid:
                    ref_text += f" PMID: {pmid}"
                
                story.append(Paragraph(ref_text, styles['Normal']))
                story.append(Spacer(1, 4))
        else:
            story.append(Paragraph("No literature found in database.", styles['Normal']))
        
        story.append(Spacer(1, 12))
        
        # Add page break between drugs
        if i < clinical_count - 1:
            story.append(PageBreak())
    
    # Add Preclinical Assets Section
    if preclinical_candidates:
        story.append(PageBreak())
        story.append(Paragraph("Preclinical Assets", heading_style))
        story.append(Spacer(1, 12))
        
        # Preclinical section introduction
        preclinical_intro = f"""
        This section analyzes {len(preclinical_candidates)} preclinical compounds identified through literature mining and 
        compound validation. These assets represent early-stage therapeutic opportunities with validated targets and 
        mechanisms of action documented in scientific literature.
        """
        story.append(Paragraph(preclinical_intro, styles['Normal']))
        story.append(Spacer(1, 12))
        
        # Create preclinical compounds table
        preclinical_table_data = [
            ["Compound", "Target", "Mechanism", "Validation Score", "Literature Source"]
        ]
        
        for compound in preclinical_candidates[:10]:  # Limit to top 10
            compound_name = compound.get('drug_name', compound.get('compound_name', 'Unknown'))
            target = compound.get('primary_target', compound.get('target', 'Unknown'))
            mechanism = compound.get('mechanism_of_action', compound.get('mechanism', 'Unknown'))
            
            # Get validation score
            validation_score = compound.get('validation_score', 0)
            if isinstance(validation_score, (int, float)):
                validation_display = f"{validation_score:.2f}"
            else:
                validation_display = "N/A"
            
            # Get literature source and URL
            source_info = compound.get('source_paper', compound.get('literature_source', {}))
            citation_url = compound.get('citation_url', '')
            
            # Create source display with URL if available
            if citation_url and citation_url != 'web_search_results':
                # Create a clickable link
                source_display = f'<link href="{citation_url}" color="blue">{citation_url[:50]}...</link>'
            elif isinstance(source_info, dict):
                source_display = source_info.get('title', 'Unknown')[:50] + "..."
            else:
                source_display = str(source_info)[:50] + "..."
            
            preclinical_table_data.append([
                Paragraph(compound_name, styles['Normal']),
                Paragraph(target, styles['Normal']),
                Paragraph(mechanism, styles['Normal']),
                Paragraph(validation_display, styles['Normal']),
                Paragraph(source_display, styles['Normal'])
            ])
        
        preclinical_table = Table(preclinical_table_data, colWidths=[1.5*inch, 1.5*inch, 2*inch, 1*inch, 2*inch])
        preclinical_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'TOP')
        ]))
        
        story.append(preclinical_table)
        story.append(Spacer(1, 12))
        
        # Add detailed analysis for top preclinical compounds
        if openai_client and preclinical_candidates:
            story.append(Paragraph("Preclinical Compound Analysis", subheading_style))
            story.append(Spacer(1, 8))
            
            try:
                # Generate LLM analysis for preclinical compounds
                compounds_summary = []
                for compound in preclinical_candidates[:5]:
                    name = compound.get('drug_name', compound.get('compound_name', 'Unknown'))
                    target = compound.get('primary_target', compound.get('target', 'Unknown'))
                    mechanism = compound.get('mechanism_of_action', compound.get('mechanism', 'Unknown'))
                    compounds_summary.append(f"- {name}: targets {target} via {mechanism}")
                
                preclinical_prompt = f"""
                Analyze these preclinical compounds for {disease_name}:
                
                {chr(10).join(compounds_summary)}
                
                Provide a comprehensive analysis covering:
                1. Therapeutic potential and novelty of mechanisms
                2. Validation status and database presence
                3. Research opportunities and next steps
                4. Comparison to existing clinical approaches
                
                Write in academic style, 400-500 words.
                """
                
                response = await openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": preclinical_prompt}],
                    temperature=0.3
                )
                
                preclinical_analysis = response.choices[0].message.content
                story.append(Paragraph(preclinical_analysis, styles['Normal']))
                
            except Exception as e:
                story.append(Paragraph(f"Preclinical analysis unavailable: {str(e)}", styles['Normal']))
    
    # Build PDF
    print("📄 Creating comprehensive PDF report...")
    doc.build(story)
    print(f"✅ Comprehensive report generated: {output_pdf}")
    
    return output_pdf

async def add_pdf_report_to_cli(results: Dict, agent, output_pdf: str = None):
    """Helper function to add PDF generation to CLI"""
    openai_api_key = os.environ.get('OPENAI_API_KEY')
    generator = DrugDiscoveryPDFReport(output_pdf, openai_api_key)
    pdf_path = await generator.generate_report(results, agent)
    return pdf_path

"""
Intelligent Query Parser using LLM
Parses natural language queries for therapeutic target discovery.
"""

import json
import logging
import re
from typing import Dict, List, Optional, Any, Tuple
from openai import AsyncOpenAI
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ParsedQuery:
    """Represents a parsed therapeutic query"""
    gene_symbols: List[str]
    min_ic50_nm: Optional[float] = None
    max_ic50_nm: Optional[float] = None
    disease_context: Optional[str] = None
    query_type: str = "single_target"  # single_target, multi_target, disease
    confidence: float = 0.8
    is_disease_query: bool = False

class QueryParser:
    """Parses natural language queries using LLM"""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize with OpenAI API key"""
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if self.api_key:
            self.openai_client = AsyncOpenAI(api_key=self.api_key)
        else:
            self.openai_client = None
    
    async def parse_query(self, query: str) -> ParsedQuery:
        """
        Parse natural language query into structured format
        
        Args:
            query: Natural language query (e.g., "EGFR inhibitors between 10 and 100 nM")
            
        Returns:
            ParsedQuery object with extracted parameters
        """
        
        # Try LLM parsing first
        if self.api_key:
            try:
                return await self._llm_parse_query(query)
            except Exception as e:
                logger.warning(f"LLM parsing failed: {e}, falling back to regex")
        
        # Fallback to regex parsing
        return self._regex_parse_query(query)
    
    async def _llm_parse_query(self, query: str) -> ParsedQuery:
        """Parse query using OpenAI LLM"""
        
        prompt = f"""
Parse this therapeutic target discovery query and extract key information:

Query: "{query}"

Return a JSON object with:
- gene_symbols: List of gene/protein symbols
- min_ic50_nm: Minimum IC50 in nM (if mentioned)
- max_ic50_nm: Maximum IC50 in nM (if mentioned)
- disease_context: Disease name (if mentioned)
- query_type: "single_target", "multi_target", or "disease"
- is_disease_query: true if disease-focused, false if protein-focused
- confidence: Your confidence (0.0-1.0)

Notes:
- Convert all units to nM (1 μM = 1000 nM)
- Understand the query intent: researchers might want potent OR weak inhibitors
- "IC50 above X" means min_ic50_nm = X (weaker compounds)
- "IC50 below X" means max_ic50_nm = X (more potent compounds)
- Use your understanding of drug discovery context to interpret ambiguous queries
- Consider: Why would someone want compounds with specific potency ranges?

Return only valid JSON.
"""

        try:
            if not self.openai_client:
                raise Exception("OpenAI client not available")
        
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o-mini",  # Use cheaper model
                messages=[
                    {"role": "system", "content": "You are a query parsing expert. Always respond in valid JSON format."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,  # Low temperature for consistent parsing
                max_tokens=500
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # Clean response if it has markdown formatting
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0]
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0]
            
            data = json.loads(response_text)
            
            return ParsedQuery(
                gene_symbols=data.get("gene_symbols", []),
                min_ic50_nm=data.get("min_ic50_nm"),
                max_ic50_nm=data.get("max_ic50_nm"),
                disease_context=data.get("disease_context"),
                query_type=data.get("query_type", "single_target"),
                confidence=float(data.get("confidence", 0.8)),
                is_disease_query=bool(data.get("is_disease_query", False))
            )
            
        except Exception as e:
            logger.error(f"LLM query parsing failed: {e}")
            raise
    
    def _regex_parse_query(self, query: str) -> ParsedQuery:
        """Fallback regex-based query parsing"""
        
        query_upper = query.upper()
        
        # Check if it's a disease query (common disease terms)
        disease_terms = [
            "CANCER", "DISEASE", "SYNDROME", "DISORDER", "SCLEROSIS", 
            "ALZHEIMER", "PARKINSON", "DIABETES", "ARTHRITIS", "EMPHYSEMA"
        ]
        
        is_disease_query = any(term in query_upper for term in disease_terms)
        
        if is_disease_query:
            return ParsedQuery(
                gene_symbols=[],
                disease_context=query,
                query_type="disease",
                is_disease_query=True,
                confidence=0.6
            )
        
        # Extract gene symbols
        gene_patterns = [
            r'\\b([A-Z]{2,6}\\d*)\\b',  # Standard gene names like EGFR, JAK2
            r'\\b(TP53|BRCA1|BRCA2|PIK3CA)\\b',  # Special cases
        ]
        
        gene_symbols = []
        for pattern in gene_patterns:
            matches = re.findall(pattern, query_upper)
            gene_symbols.extend(matches)
        
        # Remove duplicates while preserving order
        gene_symbols = list(dict.fromkeys(gene_symbols))
        
        # Extract IC50 ranges
        min_ic50, max_ic50 = self._extract_ic50_ranges(query)
        
        # Determine query type
        query_type = "multi_target" if len(gene_symbols) > 1 else "single_target"
        
        return ParsedQuery(
            gene_symbols=gene_symbols,
            min_ic50_nm=min_ic50,
            max_ic50_nm=max_ic50,
            query_type=query_type,
            is_disease_query=False,
            confidence=0.7 if gene_symbols else 0.3
        )
    
    def _extract_ic50_ranges(self, query: str) -> Tuple[Optional[float], Optional[float]]:
        """Extract IC50 ranges from query using regex"""
        
        query_lower = query.lower()
        
        # Patterns for IC50 ranges
        patterns = [
            # "between X and Y nM"
            r'between\\s+(\\d+(?:\\.\\d+)?)\\s+(?:and|to)\\s+(\\d+(?:\\.\\d+)?)\\s*nm',
            # "under/below X nM"
            r'(?:under|below|less than|<)\\s+(\\d+(?:\\.\\d+)?)\\s*nm',
            # "above/over X nM"  
            r'(?:above|over|greater than|>)\\s+(\\d+(?:\\.\\d+)?)\\s*nm',
            # "X to Y nM"
            r'(\\d+(?:\\.\\d+)?)\\s*(?:to|-)\\s*(\\d+(?:\\.\\d+)?)\\s*nm',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query_lower)
            if match:
                if "between" in pattern or "to" in pattern:
                    # Range: min and max
                    return float(match.group(1)), float(match.group(2))
                elif "under" in pattern or "below" in pattern or "<" in pattern:
                    # Max only
                    return None, float(match.group(1))
                elif "above" in pattern or "over" in pattern or ">" in pattern:
                    # Min only
                    return float(match.group(1)), None
        
        # Special case: "potent" implies very low IC50
        if "potent" in query_lower:
            return None, 10.0  # Under 10 nM is considered potent
            
        return None, None

# Convenience function
async def parse_query(query: str) -> ParsedQuery:
    """Parse a query string into structured format"""
    parser = QueryParser()
    return await parser.parse_query(query)

"""
Async literature client with Europe PMC primary, PubMed fallback
"""
import asyncio
from typing import List, Dict, Any, Optional
import xmltodict
from .http_client import get_http_client

class PubMedClient:
    """Async literature client with Europe PMC primary, PubMed fallback and enhanced parsing"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.ncbi_base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        self.europepmc_base_url = "https://www.ebi.ac.uk/europepmc/webservices/rest"
        self.api_key = api_key
    
    async def search_articles(self, query: str, max_results: int = 20) -> List[Dict[str, Any]]:
        """Search for articles with Europe PMC primary, PubMed fallback"""
        
        # Try Europe PMC first
        try:
            articles = await self._search_europepmc(query, max_results)
            if articles:
                return articles
        except Exception as e:
            print(f"Europe PMC search error: {e}")
        
        # Fallback to PubMed
        print("Trying PubMed fallback...")
        try:
            return await self._search_pubmed(query, max_results)
        except Exception as e:
            print(f"PubMed search error: {e}")
            return []
    
    async def _search_pubmed(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Search PubMed via E-utilities API"""
        client = await get_http_client()
        
        # Step 1: Search for PMIDs
        search_params = {
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "retmode": "json"
        }
        
        if self.api_key:
            search_params["api_key"] = self.api_key
        
        search_url = f"{self.ncbi_base_url}/esearch.fcgi"
        search_response = await client.get(search_url, params=search_params, cache_ttl_hours=6)
        
        pmids = search_response.get("esearchresult", {}).get("idlist", [])
        if not pmids:
            return []
        
        # Step 2: Fetch article details
        fetch_params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "xml"
        }
        
        if self.api_key:
            fetch_params["api_key"] = self.api_key
        
        fetch_url = f"{self.ncbi_base_url}/efetch.fcgi"
        fetch_response = await client.get(fetch_url, params=fetch_params, cache_ttl_hours=24)
        
        # Parse XML response
        return self._parse_pubmed_xml(fetch_response.get("text", ""))
    
    async def _search_europepmc(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Search Europe PMC API"""
        client = await get_http_client()
        
        params = {
            "query": query,
            "format": "json",
            "pageSize": max_results,
            "resultType": "core"
        }
        
        url = f"{self.europepmc_base_url}/search"
        response = await client.get(url, params=params, cache_ttl_hours=6)
        
        results = response.get("resultList", {}).get("result", [])
        
        articles = []
        for result in results:
            # Normalize Europe PMC format to PubMed format
            article = {
                "pmid": result.get("pmid", result.get("id", "")),
                "title": result.get("title", ""),
                "abstract": result.get("abstractText", ""),
                "authors": self._parse_europepmc_authors(result.get("authorList", {})),
                "journal": result.get("journalTitle", ""),
                "pub_date": result.get("firstPublicationDate", ""),
                "citation_count": result.get("citedByCount", 0),
                "doi": result.get("doi", ""),
                "source": "Europe PMC"
            }
            articles.append(article)
        
        return articles
    
    def _parse_pubmed_xml(self, xml_text: str) -> List[Dict[str, Any]]:
        """Parse PubMed XML response"""
        try:
            xml_data = xmltodict.parse(xml_text)
        except Exception as e:
            print(f"XML parsing error: {e}")
            return []
        
        articles = []
        pubmed_articles = xml_data.get("PubmedArticleSet", {}).get("PubmedArticle", [])
        
        if not isinstance(pubmed_articles, list):
            pubmed_articles = [pubmed_articles]
        
        for article_data in pubmed_articles:
            try:
                article = self._extract_pubmed_article(article_data)
                if article:
                    articles.append(article)
            except Exception as e:
                print(f"Error parsing article: {e}")
                continue
        
        return articles
    
    def _extract_pubmed_article(self, article_data: Dict) -> Optional[Dict[str, Any]]:
        """Extract structured data from PubMed article"""
        try:
            medline = article_data.get("MedlineCitation", {})
            pubmed_data = article_data.get("PubmedData", {})
            
            # Basic info
            pmid = medline.get("PMID", {})
            if isinstance(pmid, dict):
                pmid = pmid.get("#text", "")
            
            article_info = medline.get("Article", {})
            title = article_info.get("ArticleTitle", "")
            
            # Abstract
            abstract = self._extract_abstract(article_info.get("Abstract", {}))
            
            # Authors
            authors = self._extract_authors(article_info.get("AuthorList", {}))
            
            # Journal info
            journal_info = article_info.get("Journal", {})
            journal = journal_info.get("Title", "")
            
            # Publication date
            pub_date = self._extract_pub_date(article_info.get("ArticleDate", []))
            
            # Keywords and MeSH terms for relevance scoring
            keywords = self._extract_keywords(medline.get("KeywordList", {}))
            mesh_terms = self._extract_mesh_terms(medline.get("MeshHeadingList", {}))
            
            return {
                "pmid": pmid,
                "title": title,
                "abstract": abstract,
                "authors": authors,
                "journal": journal,
                "pub_date": pub_date,
                "keywords": keywords,
                "mesh_terms": mesh_terms,
                "source": "PubMed"
            }
            
        except Exception as e:
            print(f"Error extracting article data: {e}")
            return None
    
    def _extract_abstract(self, abstract_data: Dict) -> str:
        """Extract abstract text from various formats"""
        if not abstract_data:
            return ""
        
        abstract_text = abstract_data.get("AbstractText", "")
        
        if isinstance(abstract_text, list):
            # Handle structured abstracts with sections
            sections = []
            for section in abstract_text:
                if isinstance(section, dict):
                    label = section.get("@Label", "")
                    text = section.get("#text", section)
                    if isinstance(text, dict):
                        text = text.get("#text", "")
                    sections.append(f"{label}: {text}".strip(": "))
                else:
                    sections.append(str(section))
            return " ".join(sections)
        
        elif isinstance(abstract_text, dict):
            return abstract_text.get("#text", "")
        
        return str(abstract_text)
    
    def _extract_authors(self, author_list: Dict) -> List[str]:
        """Extract author names"""
        authors = []
        
        if not author_list:
            return authors
        
        author_data = author_list.get("Author", [])
        if not isinstance(author_data, list):
            author_data = [author_data]
        
        for author in author_data:
            if isinstance(author, dict):
                last_name = author.get("LastName", "")
                first_name = author.get("ForeName", "")
                if last_name:
                    full_name = f"{first_name} {last_name}".strip()
                    authors.append(full_name)
        
        return authors
    
    def _extract_pub_date(self, article_dates: List) -> str:
        """Extract publication date"""
        if not article_dates:
            return ""
        
        if not isinstance(article_dates, list):
            article_dates = [article_dates]
        
        for date_info in article_dates:
            if isinstance(date_info, dict):
                year = date_info.get("Year", {})
                month = date_info.get("Month", {})
                day = date_info.get("Day", {})
                
                # Handle nested dict format
                if isinstance(year, dict):
                    year = year.get("#text", "")
                if isinstance(month, dict):
                    month = month.get("#text", "")
                if isinstance(day, dict):
                    day = day.get("#text", "")
                
                return f"{year}-{month}-{day}".strip("-")
        
        return ""
    
    def _extract_keywords(self, keyword_list: Dict) -> List[str]:
        """Extract keywords"""
        keywords = []
        
        if not keyword_list:
            return keywords
        
        keyword_data = keyword_list.get("Keyword", [])
        if not isinstance(keyword_data, list):
            keyword_data = [keyword_data]
        
        for keyword in keyword_data:
            if isinstance(keyword, dict):
                keyword = keyword.get("#text", "")
            keywords.append(str(keyword))
        
        return keywords
    
    def _extract_mesh_terms(self, mesh_list: Dict) -> List[str]:
        """Extract MeSH terms for topic classification"""
        mesh_terms = []
        
        if not mesh_list:
            return mesh_terms
        
        mesh_headings = mesh_list.get("MeshHeading", [])
        if not isinstance(mesh_headings, list):
            mesh_headings = [mesh_headings]
        
        for heading in mesh_headings:
            if isinstance(heading, dict):
                descriptor = heading.get("DescriptorName", {})
                if isinstance(descriptor, dict):
                    term = descriptor.get("#text", "")
                    mesh_terms.append(term)
        
        return mesh_terms
    
    def _parse_europepmc_authors(self, author_list: Dict) -> List[str]:
        """Parse Europe PMC author format"""
        authors = []
        
        if not author_list:
            return authors
        
        author_data = author_list.get("author", [])
        if not isinstance(author_data, list):
            author_data = [author_data]
        
        for author in author_data:
            if isinstance(author, dict):
                full_name = author.get("fullName", "")
                if full_name:
                    authors.append(full_name)
        
        return authors

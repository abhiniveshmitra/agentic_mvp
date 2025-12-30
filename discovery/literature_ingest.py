"""
PubMed Literature Ingestion.

Fetches papers from NCBI PubMed E-utilities API.
Returns structured paper metadata for compound extraction.
"""

import time
from typing import List, Dict, Optional
from dataclasses import dataclass
import requests
import xml.etree.ElementTree as ET

from utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class Paper:
    """Structured paper data."""
    paper_id: str
    title: str
    abstract: str
    authors: List[str]
    journal: str
    pub_date: str
    pubmed_url: str
    query_metadata: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            "paper_id": self.paper_id,
            "title": self.title,
            "abstract": self.abstract,
            "authors": self.authors,
            "journal": self.journal,
            "pub_date": self.pub_date,
            "pubmed_url": self.pubmed_url,
            "query_metadata": self.query_metadata,
        }


def fetch_pubmed_papers(
    query: str,
    max_results: int = 100,
    retry_count: int = 3,
    retry_delay: float = 1.0,
) -> List[Dict]:
    """
    Fetch papers from PubMed using E-utilities API.
    
    Args:
        query: PubMed search query
        max_results: Maximum number of papers to fetch
        retry_count: Number of retries on failure
        retry_delay: Delay between retries in seconds
    
    Returns:
        List of paper dictionaries with metadata
    """
    from config.settings import PUBMED_BASE_URL, NCBI_EMAIL
    
    logger.info(f"Searching PubMed for: {query}")
    
    # Step 1: ESearch to get PMIDs
    search_url = f"{PUBMED_BASE_URL}/esearch.fcgi"
    search_params = {
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "retmode": "json",
        "sort": "relevance",
    }
    
    if NCBI_EMAIL:
        search_params["email"] = NCBI_EMAIL
    
    pmids = []
    for attempt in range(retry_count):
        try:
            response = requests.get(search_url, params=search_params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            pmids = data.get("esearchresult", {}).get("idlist", [])
            logger.info(f"Found {len(pmids)} papers")
            break
            
        except requests.RequestException as e:
            logger.warning(f"Search attempt {attempt + 1} failed: {e}")
            if attempt < retry_count - 1:
                time.sleep(retry_delay)
            else:
                logger.error("All search attempts failed")
                return []
    
    if not pmids:
        logger.warning("No papers found for query")
        return []
    
    # Step 2: EFetch to get paper details
    papers = _fetch_paper_details(pmids, query, retry_count, retry_delay)
    
    return papers


def _fetch_paper_details(
    pmids: List[str],
    query: str,
    retry_count: int = 3,
    retry_delay: float = 1.0,
) -> List[Dict]:
    """
    Fetch detailed paper information for a list of PMIDs.
    
    Args:
        pmids: List of PubMed IDs
        query: Original search query (for metadata)
        retry_count: Number of retries
        retry_delay: Delay between retries
    
    Returns:
        List of Paper dictionaries
    """
    from config.settings import PUBMED_BASE_URL, NCBI_EMAIL
    
    fetch_url = f"{PUBMED_BASE_URL}/efetch.fcgi"
    
    # Batch PMIDs (max 200 per request)
    batch_size = 200
    all_papers = []
    
    for i in range(0, len(pmids), batch_size):
        batch = pmids[i:i + batch_size]
        
        fetch_params = {
            "db": "pubmed",
            "id": ",".join(batch),
            "retmode": "xml",
        }
        
        if NCBI_EMAIL:
            fetch_params["email"] = NCBI_EMAIL
        
        for attempt in range(retry_count):
            try:
                response = requests.get(fetch_url, params=fetch_params, timeout=60)
                response.raise_for_status()
                
                papers = _parse_pubmed_xml(response.text, query)
                all_papers.extend(papers)
                break
                
            except requests.RequestException as e:
                logger.warning(f"Fetch attempt {attempt + 1} failed: {e}")
                if attempt < retry_count - 1:
                    time.sleep(retry_delay)
        
        # Rate limiting
        time.sleep(0.34)  # ~3 requests per second
    
    return all_papers


def _parse_pubmed_xml(xml_content: str, query: str) -> List[Dict]:
    """
    Parse PubMed XML response into Paper objects.
    
    Args:
        xml_content: XML response from EFetch
        query: Original search query
    
    Returns:
        List of Paper dictionaries
    """
    papers = []
    
    try:
        root = ET.fromstring(xml_content)
        
        for article in root.findall(".//PubmedArticle"):
            try:
                # Extract PMID
                pmid_elem = article.find(".//PMID")
                pmid = pmid_elem.text if pmid_elem is not None else ""
                
                # Extract title
                title_elem = article.find(".//ArticleTitle")
                title = title_elem.text if title_elem is not None else ""
                
                # Extract abstract
                abstract_parts = []
                for abstract_text in article.findall(".//AbstractText"):
                    if abstract_text.text:
                        label = abstract_text.get("Label", "")
                        if label:
                            abstract_parts.append(f"{label}: {abstract_text.text}")
                        else:
                            abstract_parts.append(abstract_text.text)
                abstract = " ".join(abstract_parts)
                
                # Extract authors
                authors = []
                for author in article.findall(".//Author"):
                    last_name = author.find("LastName")
                    fore_name = author.find("ForeName")
                    if last_name is not None and fore_name is not None:
                        authors.append(f"{fore_name.text} {last_name.text}")
                    elif last_name is not None:
                        authors.append(last_name.text)
                
                # Extract journal
                journal_elem = article.find(".//Journal/Title")
                journal = journal_elem.text if journal_elem is not None else ""
                
                # Extract publication date
                pub_date_elem = article.find(".//PubDate")
                year = pub_date_elem.find("Year") if pub_date_elem is not None else None
                month = pub_date_elem.find("Month") if pub_date_elem is not None else None
                pub_date = ""
                if year is not None:
                    pub_date = year.text
                    if month is not None:
                        pub_date = f"{month.text} {year.text}"
                
                # Create Paper object
                paper = Paper(
                    paper_id=pmid,
                    title=title,
                    abstract=abstract,
                    authors=authors,
                    journal=journal,
                    pub_date=pub_date,
                    pubmed_url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                    query_metadata=query,
                )
                papers.append(paper.to_dict())
                
            except Exception as e:
                logger.warning(f"Error parsing article: {e}")
                continue
        
    except ET.ParseError as e:
        logger.error(f"XML parse error: {e}")
    
    return papers


def search_papers_by_compound(compound_name: str, max_results: int = 20) -> List[Dict]:
    """
    Search PubMed for papers mentioning a specific compound.
    
    Args:
        compound_name: Name of the compound to search for
        max_results: Maximum number of papers
    
    Returns:
        List of Paper dictionaries
    """
    query = f"{compound_name}[Title/Abstract] AND drug[Title/Abstract]"
    return fetch_pubmed_papers(query, max_results)


def search_papers_by_target(target_name: str, max_results: int = 50) -> List[Dict]:
    """
    Search PubMed for papers about a specific drug target.
    
    Args:
        target_name: Name of the target protein
        max_results: Maximum number of papers
    
    Returns:
        List of Paper dictionaries
    """
    query = f"{target_name}[Title/Abstract] AND (inhibitor OR modulator OR drug)"
    return fetch_pubmed_papers(query, max_results)

"""
Discovery Query Templates.

Provides compound-focused query templates for better extraction recall.
Abstract/conceptual queries don't yield named compounds.
"""

from typing import List, Dict


# Target-specific query templates
QUERY_TEMPLATES = {
    "EGFR": {
        "compound_focused": [
            "EGFR inhibitor IC50 compound",
            "EGFR inhibitor small molecule synthesis",
            "EGFR kinase inhibitor structure activity",
            "novel EGFR inhibitor identified",
        ],
        "benchmark_anchors": [
            "gefitinib EGFR binding affinity",
            "erlotinib EGFR IC50 inhibition",
            "lapatinib EGFR HER2 inhibitor",
            "afatinib EGFR irreversible inhibitor",
            "osimertinib EGFR T790M inhibitor",
        ],
        "known_drugs": [
            "gefitinib", "erlotinib", "lapatinib", "afatinib", 
            "osimertinib", "dacomitinib", "neratinib"
        ],
    },
    "BCR-ABL": {
        "compound_focused": [
            "BCR-ABL inhibitor IC50 compound",
            "BCR-ABL kinase inhibitor small molecule",
            "BCR-ABL inhibitor synthesis",
        ],
        "benchmark_anchors": [
            "imatinib BCR-ABL binding",
            "dasatinib BCR-ABL inhibition",
            "nilotinib BCR-ABL IC50",
        ],
        "known_drugs": [
            "imatinib", "dasatinib", "nilotinib", "bosutinib", "ponatinib"
        ],
    },
    "BRAF": {
        "compound_focused": [
            "BRAF inhibitor IC50 compound",
            "BRAF V600E inhibitor small molecule",
            "BRAF kinase inhibitor synthesis",
        ],
        "benchmark_anchors": [
            "vemurafenib BRAF V600E",
            "dabrafenib BRAF inhibition",
            "encorafenib BRAF binding",
        ],
        "known_drugs": [
            "vemurafenib", "dabrafenib", "encorafenib"
        ],
    },
    "CDK4": {
        "compound_focused": [
            "CDK4 inhibitor IC50 compound",
            "CDK4/6 inhibitor small molecule",
            "CDK4 kinase inhibitor synthesis",
        ],
        "benchmark_anchors": [
            "palbociclib CDK4 binding",
            "ribociclib CDK4/6 inhibition",
            "abemaciclib CDK4 IC50",
        ],
        "known_drugs": [
            "palbociclib", "ribociclib", "abemaciclib"
        ],
    },
}

# Default template for unknown targets
DEFAULT_TEMPLATE = {
    "compound_focused": [
        "{target} inhibitor IC50 compound",
        "{target} inhibitor small molecule synthesis",
        "{target} inhibitor structure activity relationship",
        "novel {target} inhibitor identified",
    ],
    "benchmark_anchors": [],
    "known_drugs": [],
}


def get_query_templates(target: str) -> Dict[str, List[str]]:
    """
    Get query templates for a specific target.
    
    Args:
        target: Target protein name
    
    Returns:
        Dictionary with compound_focused, benchmark_anchors, known_drugs
    """
    target_upper = target.upper()
    
    # Check for known targets
    for key in QUERY_TEMPLATES:
        if key in target_upper:
            return QUERY_TEMPLATES[key]
    
    # Return default with placeholder filled
    return {
        "compound_focused": [
            q.format(target=target) for q in DEFAULT_TEMPLATE["compound_focused"]
        ],
        "benchmark_anchors": [],
        "known_drugs": [],
    }


def get_recommended_queries(target: str, max_queries: int = 3) -> List[str]:
    """
    Get recommended PubMed queries for a target.
    
    Args:
        target: Target protein name
        max_queries: Maximum number of queries to return
    
    Returns:
        List of recommended query strings
    """
    templates = get_query_templates(target)
    
    queries = []
    
    # Prioritize benchmark anchors (most likely to have named compounds)
    for q in templates.get("benchmark_anchors", [])[:max_queries]:
        queries.append(q)
    
    # Fill with compound-focused queries
    remaining = max_queries - len(queries)
    for q in templates.get("compound_focused", [])[:remaining]:
        queries.append(q)
    
    return queries


def get_combined_query(target: str) -> str:
    """
    Get a single combined query string that's more likely to find compounds.
    
    Args:
        target: Target protein name
    
    Returns:
        Combined query string
    """
    templates = get_query_templates(target)
    drugs = templates.get("known_drugs", [])
    
    if drugs:
        # Include known drug names to find papers discussing them
        drug_str = " OR ".join(drugs[:3])
        return f"({target} inhibitor) AND ({drug_str}) AND (IC50 OR binding OR affinity)"
    else:
        return f"{target} inhibitor IC50 small molecule compound"


def get_seeding_compounds(target: str) -> List[Dict]:
    """
    Get known compounds for discovery seeding.
    
    These are NOT cheating - they verify the pipeline works
    and prime the extractor for the linguistic pattern.
    
    Args:
        target: Target protein name
    
    Returns:
        List of seeding compound dictionaries
    """
    templates = get_query_templates(target)
    known_drugs = templates.get("known_drugs", [])
    
    seeding_compounds = []
    for drug in known_drugs[:5]:  # Max 5 seeds
        seeding_compounds.append({
            "compound_name": drug,
            "context": f"Known {target} inhibitor for discovery seeding",
            "paper_id": "DISCOVERY_SEED",
            "confidence": 1.0,
            "mention_type": "drug",
            "is_seed": True,
        })
    
    return seeding_compounds

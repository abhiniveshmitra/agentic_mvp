"""
Candidate Builder - Convert text mentions to molecules.

CRITICAL CONSTRAINTS (from agent knowledge base):
- Path A (Preferred): PubChem exact match → PUBCHEM source
- Path B (Flagged): LLM-inferred → LLM_INFERRED source, low confidence
- NEVER improve, clean, or guess SMILES
"""

from typing import List, Dict, Optional, Tuple
import time
import requests

from utils.logging import get_logger
from utils.provenance import (
    ProvenanceTracker,
    ExtractionSource,
    ExtractionInfo,
    PaperSource,
)

logger = get_logger(__name__)


def build_candidates(
    compound_mentions: List[Dict],
    provenance: ProvenanceTracker,
) -> List[Dict]:
    """
    Convert compound mentions to candidate molecules with SMILES.
    
    Uses two paths:
    - Path A: PubChem exact match (preferred, trusted)
    - Path B: LLM-inferred (flagged, low confidence)
    
    Args:
        compound_mentions: List of compound mention dicts from text mining
        provenance: Provenance tracker for metadata
    
    Returns:
        List of candidate dictionaries with SMILES
    """
    candidates = []
    seen_smiles = set()  # Deduplicate
    
    for i, mention in enumerate(compound_mentions):
        compound_name = mention.get("compound_name", "")
        
        if not compound_name:
            continue
        
        # Try Path A: PubChem exact match
        pubchem_result = query_pubchem(compound_name)
        
        if pubchem_result:
            smiles = pubchem_result["smiles"]
            
            # Skip duplicates
            if smiles in seen_smiles:
                continue
            seen_smiles.add(smiles)
            
            # Create candidate (Path A - trusted)
            compound_id = f"cand_{i:04d}"
            candidate = {
                "compound_id": compound_id,
                "compound_name": compound_name,
                "smiles": smiles,
                "source": ExtractionSource.PUBCHEM.value,
                "pubchem_cid": pubchem_result["cid"],
                "paper_id": mention.get("paper_id", ""),
                "context": mention.get("context", ""),
                "confidence": 0.9,  # High confidence for PubChem match
                "is_control": False,
            }
            candidates.append(candidate)
            
            # Track provenance
            provenance.add_compound(
                compound_id=compound_id,
                smiles=smiles,
                paper_source=PaperSource(
                    paper_id=mention.get("paper_id", ""),
                    title="",
                    abstract=None,
                    query_used=None,
                ),
                extraction_info=ExtractionInfo(
                    source=ExtractionSource.PUBCHEM,
                    compound_name=compound_name,
                    pubchem_cid=pubchem_result["cid"],
                    confidence=0.9,
                ),
            )
            
            logger.debug(f"Path A match: {compound_name} -> CID {pubchem_result['cid']}")
            
        else:
            # Path B: No PubChem match - flag as LLM_INFERRED
            # NOTE: We do NOT generate SMILES here - that is forbidden
            # We only record that this compound could not be verified
            
            compound_id = f"cand_{i:04d}_unverified"
            candidate = {
                "compound_id": compound_id,
                "compound_name": compound_name,
                "smiles": None,  # NO SMILES - cannot be verified
                "source": ExtractionSource.LLM_INFERRED.value,
                "pubchem_cid": None,
                "paper_id": mention.get("paper_id", ""),
                "context": mention.get("context", ""),
                "confidence": 0.3,  # Low confidence - unverified
                "is_control": False,
                "flag": "UNVERIFIED_STRUCTURE",
            }
            # Don't add to main candidates - no SMILES means can't process
            # But track for provenance
            provenance.add_compound(
                compound_id=compound_id,
                smiles="UNVERIFIED",
                extraction_info=ExtractionInfo(
                    source=ExtractionSource.LLM_INFERRED,
                    compound_name=compound_name,
                    pubchem_cid=None,
                    confidence=0.3,
                ),
            )
            
            logger.debug(f"Path B (unverified): {compound_name} - no PubChem match")
    
    logger.info(f"Built {len(candidates)} verified candidates from {len(compound_mentions)} mentions")
    return candidates


def query_pubchem(compound_name: str) -> Optional[Dict]:
    """
    Query PubChem for exact compound match.
    
    Args:
        compound_name: Name of the compound to search
    
    Returns:
        Dictionary with CID and canonical SMILES, or None if not found
    """
    from config.settings import PUBCHEM_BASE_URL
    
    try:
        # Search by name
        search_url = f"{PUBCHEM_BASE_URL}/compound/name/{requests.utils.quote(compound_name)}/property/CanonicalSMILES/JSON"
        
        response = requests.get(search_url, timeout=10)
        
        if response.status_code == 404:
            # Not found - try synonyms
            return _search_pubchem_synonyms(compound_name)
        
        response.raise_for_status()
        data = response.json()
        
        properties = data.get("PropertyTable", {}).get("Properties", [])
        if properties:
            prop = properties[0]
            return {
                "cid": prop.get("CID"),
                "smiles": prop.get("CanonicalSMILES"),
            }
        
        return None
        
    except requests.RequestException as e:
        logger.warning(f"PubChem query failed for {compound_name}: {e}")
        return None
    except Exception as e:
        logger.warning(f"PubChem parse error for {compound_name}: {e}")
        return None
    finally:
        # Rate limiting
        time.sleep(0.2)


def _search_pubchem_synonyms(compound_name: str) -> Optional[Dict]:
    """
    Search PubChem using compound name as synonym.
    
    Fallback when exact name match fails.
    """
    from config.settings import PUBCHEM_BASE_URL
    
    try:
        # Try autocomplete API for fuzzy matching
        autocomplete_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/autocomplete/compound/{requests.utils.quote(compound_name)}/json"
        
        response = requests.get(autocomplete_url, timeout=10)
        
        if response.status_code != 200:
            return None
        
        data = response.json()
        options = data.get("dictionary_terms", {}).get("compound", [])
        
        if not options:
            return None
        
        # Try first suggestion
        first_match = options[0]
        return query_pubchem(first_match)
        
    except Exception as e:
        logger.debug(f"Synonym search failed: {e}")
        return None


def get_pubchem_properties(cid: int) -> Optional[Dict]:
    """
    Get additional properties for a PubChem compound.
    
    Args:
        cid: PubChem Compound ID
    
    Returns:
        Dictionary with molecular properties
    """
    from config.settings import PUBCHEM_BASE_URL
    
    try:
        properties = [
            "MolecularFormula",
            "MolecularWeight",
            "XLogP",
            "TPSA",
            "HBondDonorCount",
            "HBondAcceptorCount",
            "RotatableBondCount",
        ]
        
        url = f"{PUBCHEM_BASE_URL}/compound/cid/{cid}/property/{','.join(properties)}/JSON"
        
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        props = data.get("PropertyTable", {}).get("Properties", [])
        
        if props:
            return props[0]
        
        return None
        
    except Exception as e:
        logger.warning(f"Property fetch failed for CID {cid}: {e}")
        return None

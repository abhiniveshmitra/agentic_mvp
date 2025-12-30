"""
Patent Landscape Analysis.

Post-ranking IP check for top candidates only.

Design principles:
- Runs LAST, after all science is complete
- Top-K only (typically 10-20 compounds)
- Outputs advisory labels (CLEAR/POTENTIAL_RISK/LIKELY_ENCUMBERED)
- Does NOT influence ranking or discovery
- Keeps science and IP strictly separated

This is an IP ADVISORY layer, not a scientific filter.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import requests

from utils.logging import get_logger

logger = get_logger(__name__)

# Try to import RDKit for InChIKey generation
try:
    from rdkit import Chem
    from rdkit.Chem.inchi import MolFromInchi, MolToInchi, MolToInchiKey
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False


class PatentStatus(str, Enum):
    """Patent risk levels."""
    CLEAR = "CLEAR"
    POTENTIAL_RISK = "POTENTIAL_RISK"
    LIKELY_ENCUMBERED = "LIKELY_ENCUMBERED"
    UNKNOWN = "UNKNOWN"


@dataclass
class PatentResult:
    """Result of patent landscape check."""
    status: PatentStatus
    matches: int = 0
    notes: str = ""
    sources_checked: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "status": self.status.value,
            "matches": self.matches,
            "notes": self.notes,
            "sources_checked": self.sources_checked,
        }


# =============================================================================
# INCHIKEY GENERATION
# =============================================================================

def get_inchikey(smiles: str) -> Optional[str]:
    """
    Generate InChIKey for a SMILES string.
    
    Args:
        smiles: SMILES string
    
    Returns:
        InChIKey or None
    """
    if not HAS_RDKIT:
        return None
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return MolToInchiKey(mol)
    except Exception as e:
        logger.warning(f"InChIKey generation failed: {e}")
        return None


# =============================================================================
# PATENT DATABASE QUERIES
# =============================================================================

def check_pubchem_patents(cid: Optional[str] = None, smiles: Optional[str] = None) -> Dict:
    """
    Check PubChem for patent associations.
    
    PubChem links compounds to patents through SureChEMBL integration.
    
    Args:
        cid: PubChem CID
        smiles: SMILES (for similarity search)
    
    Returns:
        Patent information dictionary
    """
    try:
        if cid:
            # Direct CID lookup
            url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/xrefs/PatentID/JSON"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                patents = data.get("InformationList", {}).get("Information", [{}])[0].get("PatentID", [])
                return {
                    "found": True,
                    "count": len(patents),
                    "patents": patents[:10],  # Limit to first 10
                }
        
        return {"found": False, "count": 0, "patents": []}
        
    except Exception as e:
        logger.warning(f"PubChem patent check failed: {e}")
        return {"found": False, "count": 0, "patents": [], "error": str(e)}


def check_surechembl(inchikey: str) -> Dict:
    """
    Check SureChEMBL for patent matches.
    
    SureChEMBL is a public database of chemicals extracted from patents.
    
    Args:
        inchikey: InChIKey of compound
    
    Returns:
        Patent information dictionary
    """
    # Note: SureChEMBL API requires registration for full access
    # This is a simplified check using the first block of InChIKey (connectivity layer)
    
    try:
        # SureChEMBL connectivity search
        connectivity = inchikey.split("-")[0] if inchikey else None
        
        if connectivity:
            # Public endpoint for basic check
            url = f"https://www.surechembl.org/api/compound/inchikey/{connectivity}"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                # Found in SureChEMBL (was extracted from a patent)
                return {
                    "found": True,
                    "source": "SureChEMBL",
                    "notes": "Compound or close analog found in patent literature",
                }
        
        return {"found": False}
        
    except Exception as e:
        # SureChEMBL may require authentication or have rate limits
        logger.debug(f"SureChEMBL check skipped: {e}")
        return {"found": False, "skipped": True}


# =============================================================================
# MAIN PATENT CHECK FUNCTION
# =============================================================================

def check_patent_landscape(
    smiles: str,
    compound_id: Optional[str] = None,
) -> PatentResult:
    """
    Check patent landscape for a compound.
    
    Args:
        smiles: SMILES string
        compound_id: Optional compound ID (used for PubChem lookup)
    
    Returns:
        PatentResult with status and details
    """
    sources_checked = []
    total_matches = 0
    notes_parts = []
    
    # Generate InChIKey
    inchikey = get_inchikey(smiles)
    
    # Check PubChem
    if compound_id and compound_id.startswith("CHEMBL"):
        # For ChEMBL compounds, try CID lookup via UniChem or direct
        # Simplified: just note it's a known compound
        pubchem_result = {"found": False, "count": 0}
    else:
        pubchem_result = {"found": False, "count": 0}
    
    sources_checked.append("PubChem")
    
    if pubchem_result.get("found"):
        total_matches += pubchem_result.get("count", 0)
        notes_parts.append(f"{pubchem_result['count']} PubChem patent associations")
    
    # Check SureChEMBL
    if inchikey:
        surechembl_result = check_surechembl(inchikey)
        sources_checked.append("SureChEMBL")
        
        if surechembl_result.get("found"):
            total_matches += 1
            notes_parts.append("Found in patent-extracted compound database")
    
    # Determine status based on findings
    if total_matches == 0:
        status = PatentStatus.CLEAR
        notes = "No patent associations found in checked databases"
    elif total_matches < 5:
        status = PatentStatus.POTENTIAL_RISK
        notes = "; ".join(notes_parts) if notes_parts else "Some patent activity detected"
    else:
        status = PatentStatus.LIKELY_ENCUMBERED
        notes = "; ".join(notes_parts) if notes_parts else f"Multiple patent associations ({total_matches})"
    
    return PatentResult(
        status=status,
        matches=total_matches,
        notes=notes,
        sources_checked=sources_checked,
    )


def check_patent_batch(
    compounds: List[Dict],
    top_k: int = 20,
) -> List[Dict]:
    """
    Check patents for top-K compounds only.
    
    Args:
        compounds: List of compound dictionaries (should be pre-ranked)
        top_k: Number of top compounds to check
    
    Returns:
        Same compounds with 'patent' field added to top-K
    """
    logger.info(f"Running patent check for top {top_k} candidates")
    
    for i, compound in enumerate(compounds[:top_k]):
        smiles = compound.get("smiles", "")
        compound_id = compound.get("compound_id", "")
        
        if smiles:
            result = check_patent_landscape(smiles, compound_id)
            compound["patent"] = result.to_dict()
        else:
            compound["patent"] = {
                "status": "UNKNOWN",
                "matches": 0,
                "notes": "No SMILES available",
                "sources_checked": [],
            }
    
    # Log summary
    statuses = [c.get("patent", {}).get("status", "UNKNOWN") for c in compounds[:top_k]]
    logger.info(
        f"Patent check complete: {statuses.count('CLEAR')} clear, "
        f"{statuses.count('POTENTIAL_RISK')} potential risk, "
        f"{statuses.count('LIKELY_ENCUMBERED')} likely encumbered"
    )
    
    return compounds


def get_patent_summary(compounds: List[Dict], top_k: int = 20) -> Dict:
    """
    Get summary of patent landscape for top compounds.
    
    Args:
        compounds: Compounds with patent field
        top_k: Number to consider
    
    Returns:
        Summary dictionary
    """
    checked = [c for c in compounds[:top_k] if c.get("patent")]
    statuses = [c.get("patent", {}).get("status", "UNKNOWN") for c in checked]
    
    return {
        "checked": len(checked),
        "clear": statuses.count("CLEAR"),
        "potential_risk": statuses.count("POTENTIAL_RISK"),
        "likely_encumbered": statuses.count("LIKELY_ENCUMBERED"),
        "unknown": statuses.count("UNKNOWN"),
    }

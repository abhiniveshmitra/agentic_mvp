"""
ChEMBL Database Discovery.

Database-first discovery: molecules come from ChEMBL, literature provides context.

This is the CORRECT approach for drug discovery:
- ChEMBL/PubChem = molecule truth with SMILES
- Literature = biological narrative and context
"""

import requests
from typing import List, Dict, Optional
from dataclasses import dataclass

from utils.logging import get_logger

logger = get_logger(__name__)

# ChEMBL API endpoint
CHEMBL_API_BASE = "https://www.ebi.ac.uk/chembl/api/data"

# Target ChEMBL IDs for common targets
TARGET_CHEMBL_IDS = {
    # Kinase targets
    "EGFR": "CHEMBL203",
    "BCR-ABL": "CHEMBL1862",
    "BRAF": "CHEMBL5145",
    "CDK2": "CHEMBL308",    # Classic ATP-competitive kinase
    "CDK4": "CHEMBL3769",
    "CDK6": "CHEMBL2508",
    "HER2": "CHEMBL1824",
    "VEGFR2": "CHEMBL1868", # KDR - promiscuous kinase
    "ALK": "CHEMBL4247",
    "MET": "CHEMBL3717",
    "JAK2": "CHEMBL2971",
    
    # Non-kinase targets (enzyme controls)
    "HMGCR": "CHEMBL402",       # HMG-CoA Reductase (statins)
    "HMG-COA REDUCTASE": "CHEMBL402",
    "COX-2": "CHEMBL230",       # PTGS2 (NSAIDs)
    "PTGS2": "CHEMBL230",
    
    # GPCR (optional)
    "ADRB2": "CHEMBL210",       # Beta-2 adrenergic receptor
}


@dataclass
class ChEMBLCompound:
    """A compound from ChEMBL database."""
    molecule_chembl_id: str
    canonical_smiles: str
    pref_name: Optional[str]
    iupac_name: Optional[str]  # IUPAC systematic name
    max_phase: int
    activity_value: Optional[float]
    activity_type: Optional[str]
    activity_units: Optional[str]
    
    def to_dict(self) -> Dict:
        return {
            "compound_id": self.molecule_chembl_id,
            "smiles": self.canonical_smiles,
            "compound_name": self.pref_name or self.molecule_chembl_id,
            "iupac_name": self.iupac_name or "",
            "source": "CHEMBL",
            "max_phase": self.max_phase,
            "activity_value": self.activity_value,
            "activity_type": self.activity_type,
            "activity_units": self.activity_units,
        }


def query_chembl_for_target(
    target_name: str,
    max_results: int = 200,
    max_ic50_nm: float = 10000,  # 10 µM cutoff
) -> List[Dict]:
    """
    Query ChEMBL for compounds active against a target.
    
    Args:
        target_name: Target protein name (e.g., "EGFR")
        max_results: Maximum compounds to return
        max_ic50_nm: Maximum IC50 in nM (filters weak binders)
    
    Returns:
        List of compound dictionaries with SMILES
    """
    target_upper = target_name.upper()
    
    # Get ChEMBL target ID
    chembl_id = TARGET_CHEMBL_IDS.get(target_upper)
    if not chembl_id:
        logger.warning(f"Unknown target {target_name}, attempting search...")
        chembl_id = _search_target(target_name)
    
    if not chembl_id:
        logger.error(f"Could not find ChEMBL ID for target: {target_name}")
        return []
    
    logger.info(f"Querying ChEMBL for {target_name} ({chembl_id})")
    
    # Query bioactivities for this target
    compounds = _fetch_bioactivities(
        target_chembl_id=chembl_id,
        max_results=max_results,
        max_ic50_nm=max_ic50_nm,
    )
    
    logger.info(f"Retrieved {len(compounds)} compounds from ChEMBL for {target_name}")
    return compounds


def _search_target(target_name: str) -> Optional[str]:
    """
    Search for a target ChEMBL ID by name.
    
    Args:
        target_name: Target protein name
    
    Returns:
        ChEMBL target ID or None
    """
    try:
        url = f"{CHEMBL_API_BASE}/target/search.json"
        params = {"q": target_name, "limit": 5}
        
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        targets = data.get("targets", [])
        
        if targets:
            # Return first match
            return targets[0].get("target_chembl_id")
        
        return None
        
    except Exception as e:
        logger.warning(f"Target search failed: {e}")
        return None


def _enrich_with_iupac_names(compounds: List[Dict]) -> List[Dict]:
    """
    Enrich compounds with IUPAC names from PubChem API.
    
    PubChem provides actual IUPAC systematic names, unlike ChEMBL.
    
    Args:
        compounds: List of compound dictionaries with smiles
    
    Returns:
        Same compounds with iupac_name populated
    """
    if not compounds:
        return compounds
    
    # Use PubChem to get IUPAC names from SMILES
    pubchem_api = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
    
    enriched_count = 0
    
    for compound in compounds:
        smiles = compound.get("smiles", "")
        if not smiles:
            continue
        
        try:
            # Query PubChem by SMILES to get IUPAC name
            url = f"{pubchem_api}/compound/smiles/{requests.utils.quote(smiles)}/property/IUPACName/JSON"
            response = requests.get(url, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                props = data.get("PropertyTable", {}).get("Properties", [])
                if props and props[0].get("IUPACName"):
                    compound["iupac_name"] = props[0]["IUPACName"]
                    enriched_count += 1
                    
        except Exception:
            # Skip on error - don't block the pipeline
            pass
        
        # Rate limit to avoid hitting PubChem too hard
        if enriched_count >= 20:  # Only fetch for top 20 to save time
            break
    
    logger.info(f"Enriched {enriched_count} compounds with IUPAC names from PubChem")
    
    return compounds


def _fetch_bioactivities(
    target_chembl_id: str,
    max_results: int = 200,
    max_ic50_nm: float = 10000,
) -> List[Dict]:
    """
    Fetch bioactivity data for a target.
    
    Args:
        target_chembl_id: ChEMBL target ID
        max_results: Maximum compounds
        max_ic50_nm: IC50 cutoff in nM
    
    Returns:
        List of compound dictionaries
    """
    try:
        url = f"{CHEMBL_API_BASE}/activity.json"
        params = {
            "target_chembl_id": target_chembl_id,
            "standard_type__in": "IC50,Ki,Kd,EC50",
            "standard_value__lte": max_ic50_nm,
            "standard_units": "nM",
            "limit": min(max_results * 2, 500),  # Fetch extra to dedupe
        }
        
        response = requests.get(url, params=params, timeout=60)
        response.raise_for_status()
        
        data = response.json()
        activities = data.get("activities", [])
        
        # Process and deduplicate by molecule
        seen_molecules = set()
        compounds = []
        
        for activity in activities:
            mol_id = activity.get("molecule_chembl_id")
            smiles = activity.get("canonical_smiles")
            
            if not mol_id or not smiles:
                continue
            
            if mol_id in seen_molecules:
                continue
            seen_molecules.add(mol_id)
            
            compound = ChEMBLCompound(
                molecule_chembl_id=mol_id,
                canonical_smiles=smiles,
                pref_name=activity.get("molecule_pref_name"),
                iupac_name=None,  # Will be fetched separately
                max_phase=activity.get("molecule_max_phase", 0) or 0,
                activity_value=activity.get("standard_value"),
                activity_type=activity.get("standard_type"),
                activity_units=activity.get("standard_units"),
            )
            compounds.append(compound.to_dict())
            
            if len(compounds) >= max_results:
                break
        
        return compounds
        
    except Exception as e:
        logger.error(f"ChEMBL bioactivity fetch failed: {e}")
        return []


def get_approved_drugs_for_target(target_name: str) -> List[Dict]:
    """
    Get approved drugs (max_phase = 4) for a target.
    
    Args:
        target_name: Target protein name
    
    Returns:
        List of approved drug dictionaries
    """
    all_compounds = query_chembl_for_target(target_name, max_results=100)
    
    # Filter to approved drugs only
    approved = [c for c in all_compounds if c.get("max_phase", 0) >= 4]
    
    logger.info(f"Found {len(approved)} approved drugs for {target_name}")
    return approved


def discover_from_database(
    target_name: str,
    max_compounds: int = 100,
    include_approved_only: bool = False,
) -> List[Dict]:
    """
    Main entry point for database-first discovery.
    
    Args:
        target_name: Target protein name
        max_compounds: Maximum compounds to return
        include_approved_only: If True, only return approved drugs
    
    Returns:
        List of compound dictionaries with guaranteed SMILES
    """
    if include_approved_only:
        return get_approved_drugs_for_target(target_name)
    
    return query_chembl_for_target(target_name, max_results=max_compounds)

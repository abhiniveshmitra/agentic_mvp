"""
Scaffold Diversity Analysis.

Uses Murcko decomposition to:
1. Cluster compounds by scaffold
2. Identify top compound per scaffold
3. Prevent over-selection of analogs
"""

from typing import List, Dict, Optional, Tuple
from collections import defaultdict

from utils.logging import get_logger

logger = get_logger(__name__)

# Try to import RDKit
try:
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False
    logger.warning("RDKit not available - scaffold analysis disabled")


def get_murcko_scaffold(smiles: str) -> Optional[str]:
    """
    Get the Murcko generic scaffold for a molecule.
    
    Args:
        smiles: SMILES string
    
    Returns:
        Scaffold SMILES or None
    """
    if not HAS_RDKIT:
        return None
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        # Get generic scaffold (atoms replaced with carbons)
        core = MurckoScaffold.GetScaffoldForMol(mol)
        generic = MurckoScaffold.MakeScaffoldGeneric(core)
        
        return Chem.MolToSmiles(generic, canonical=True)
        
    except Exception as e:
        logger.warning(f"Scaffold extraction failed: {e}")
        return None


def cluster_by_scaffold(compounds: List[Dict]) -> Dict[str, List[Dict]]:
    """
    Cluster compounds by their Murcko scaffold.
    
    Args:
        compounds: List of compound dictionaries with 'smiles'
    
    Returns:
        Dictionary mapping scaffold SMILES -> list of compounds
    """
    clusters = defaultdict(list)
    
    for compound in compounds:
        smiles = compound.get("smiles", "")
        scaffold = get_murcko_scaffold(smiles)
        
        if scaffold:
            compound["scaffold"] = scaffold
            clusters[scaffold].append(compound)
        else:
            # No scaffold - put in "unknown" cluster
            compound["scaffold"] = "unknown"
            clusters["unknown"].append(compound)
    
    logger.info(f"Found {len(clusters)} unique scaffolds from {len(compounds)} compounds")
    return dict(clusters)


def get_top_per_scaffold(
    compounds: List[Dict],
    n_per_scaffold: int = 1,
) -> List[Dict]:
    """
    Get top N compounds per scaffold.
    
    Compounds should already be ranked (sorted by score descending).
    
    Args:
        compounds: Ranked list of compounds
        n_per_scaffold: Max compounds to keep per scaffold
    
    Returns:
        Diverse subset of compounds
    """
    clusters = cluster_by_scaffold(compounds)
    
    diverse_compounds = []
    scaffold_counts = defaultdict(int)
    
    for compound in compounds:
        scaffold = compound.get("scaffold", "unknown")
        
        if scaffold_counts[scaffold] < n_per_scaffold:
            compound["is_scaffold_representative"] = True
            diverse_compounds.append(compound)
            scaffold_counts[scaffold] += 1
    
    logger.info(
        f"Diversity filter: {len(diverse_compounds)} compounds "
        f"from {len(scaffold_counts)} scaffolds (max {n_per_scaffold} each)"
    )
    
    return diverse_compounds


def add_scaffold_info(compounds: List[Dict]) -> List[Dict]:
    """
    Add scaffold information to compounds.
    
    Args:
        compounds: List of compound dictionaries
    
    Returns:
        Compounds with scaffold info added
    """
    clusters = cluster_by_scaffold(compounds)
    
    # Add cluster size info
    scaffold_sizes = {s: len(c) for s, c in clusters.items()}
    
    for compound in compounds:
        scaffold = compound.get("scaffold", "unknown")
        compound["scaffold_cluster_size"] = scaffold_sizes.get(scaffold, 0)
    
    return compounds


def calculate_diversity_metrics(compounds: List[Dict]) -> Dict:
    """
    Calculate diversity metrics for a compound set.
    
    Args:
        compounds: List of compounds with scaffolds
    
    Returns:
        Diversity metrics dictionary
    """
    clusters = cluster_by_scaffold(compounds)
    
    n_compounds = len(compounds)
    n_scaffolds = len(clusters)
    
    # Scaffold evenness (1 = perfectly even, 0 = one dominates)
    if n_scaffolds > 0:
        cluster_sizes = [len(c) for c in clusters.values()]
        max_size = max(cluster_sizes)
        evenness = 1 - (max_size / n_compounds) if n_compounds > 0 else 0
    else:
        evenness = 0
    
    # Top-3 scaffolds
    sorted_scaffolds = sorted(
        clusters.items(),
        key=lambda x: len(x[1]),
        reverse=True
    )[:3]
    
    return {
        "n_compounds": n_compounds,
        "n_unique_scaffolds": n_scaffolds,
        "diversity_ratio": n_scaffolds / n_compounds if n_compounds > 0 else 0,
        "scaffold_evenness": round(evenness, 3),
        "top_scaffolds": [
            {"scaffold": s[:30] + "..." if len(s) > 30 else s, "count": len(c)}
            for s, c in sorted_scaffolds
        ],
    }

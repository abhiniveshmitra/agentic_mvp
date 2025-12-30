"""
Positive Controls - Known binders for validation.

These compounds are known to bind to specific targets
and should rank HIGH in the output.
"""

from typing import Dict, List

# Known EGFR inhibitors (positive controls for EGFR target)
EGFR_POSITIVE_CONTROLS = [
    {
        "name": "Erlotinib",
        "smiles": "COc1cc2ncnc(Nc3ccc(OCCOc4ccccc4)cc3)c2cc1OC",
        "target": "EGFR",
        "source": "FDA Approved",
        "expected_affinity": "high",
    },
    {
        "name": "Gefitinib",
        "smiles": "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN4CCOCC4",
        "target": "EGFR",
        "source": "FDA Approved",
        "expected_affinity": "high",
    },
    {
        "name": "Lapatinib",
        "smiles": "CS(=O)(=O)CCNCc1ccc(-c2ccc3ncnc(Nc4ccc(F)c(Cl)c4)c3c2)o1",
        "target": "EGFR",
        "source": "FDA Approved",
        "expected_affinity": "high",
    },
]

# Known Kinase inhibitors
KINASE_POSITIVE_CONTROLS = [
    {
        "name": "Imatinib",
        "smiles": "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc4nccc(-c5cccnc5)n4",
        "target": "BCR-ABL",
        "source": "FDA Approved",
        "expected_affinity": "high",
    },
    {
        "name": "Sunitinib",
        "smiles": "CCN(CC)CCNC(=O)c1c(C)[nH]c(/C=C2\\C(=O)Nc3ccc(F)cc32)c1C",
        "target": "Multi-kinase",
        "source": "FDA Approved",
        "expected_affinity": "high",
    },
]


def get_positive_controls(target: str) -> List[Dict]:
    """
    Get positive controls for a specific target.
    
    Args:
        target: Target protein name (e.g., "EGFR", "BCR-ABL")
    
    Returns:
        List of positive control dictionaries
    """
    target_upper = target.upper()
    
    if "EGFR" in target_upper:
        return EGFR_POSITIVE_CONTROLS
    elif "ABL" in target_upper or "BCR" in target_upper:
        return KINASE_POSITIVE_CONTROLS
    else:
        # Return generic kinase inhibitors for unknown targets
        return KINASE_POSITIVE_CONTROLS[:1]


def get_positive_smiles(target: str) -> List[str]:
    """Get just the SMILES strings for positive controls."""
    controls = get_positive_controls(target)
    return [c["smiles"] for c in controls]

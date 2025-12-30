"""
Negative Controls - Known non-binders for validation.

These compounds are known NOT to bind and should rank LOW.
"""

from typing import List

# Simple molecules that should not bind kinases
NEGATIVE_CONTROLS = [
    {
        "name": "Caffeine",
        "smiles": "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
        "reason": "Common metabolite, weak kinase binding",
    },
    {
        "name": "Acetaminophen",
        "smiles": "CC(=O)Nc1ccc(O)cc1",
        "reason": "COX inhibitor, not kinase related",
    },
    {
        "name": "Aspirin",
        "smiles": "CC(=O)Oc1ccccc1C(=O)O",
        "reason": "COX inhibitor, not kinase related",
    },
    {
        "name": "Glucose",
        "smiles": "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O",
        "reason": "Sugar, no drug-like properties for kinases",
    },
]


def get_negative_controls() -> List[dict]:
    """Get list of negative control compounds."""
    return NEGATIVE_CONTROLS


def get_negative_smiles() -> List[str]:
    """Get just the SMILES strings for negative controls."""
    return [c["smiles"] for c in NEGATIVE_CONTROLS]

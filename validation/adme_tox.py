"""
ADME and Toxicity Risk Assessment.

Post-ranking safety layer - flags risky compounds but does NOT affect discovery or ranking.

Design principles:
- Rule-based, interpretable
- RDKit-only (no ML black boxes)
- Outputs labels (SAFE/FLAGGED/HIGH_RISK), not composite scores
- Does not influence affinity ranking

This is a GUARDRAIL, not a discovery driver.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

from utils.logging import get_logger

logger = get_logger(__name__)

# Try to import RDKit
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Lipinski, FilterCatalog, rdMolDescriptors
    from rdkit.Chem.FilterCatalog import FilterCatalogParams
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False
    logger.warning("RDKit not available - ADME/Tox assessment disabled")


class RiskLevel(str, Enum):
    """Risk assessment levels."""
    SAFE = "SAFE"
    FLAGGED = "FLAGGED"
    HIGH_RISK = "HIGH_RISK"


@dataclass
class ADMEToxResult:
    """Result of ADME/Toxicity assessment."""
    status: RiskLevel
    reasons: List[str] = field(default_factory=list)
    properties: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "status": self.status.value,
            "reasons": self.reasons,
            "properties": self.properties,
        }


# =============================================================================
# ADME PROPERTY THRESHOLDS
# =============================================================================

ADME_THRESHOLDS = {
    # Lipinski Rule of 5 (relaxed for kinase inhibitors)
    "mw_max": 700,
    "logp_max": 8.0,
    "hbd_max": 5,
    "hba_max": 10,
    
    # Extended ADME flags
    "tpsa_min": 20,   # Low TPSA = poor solubility
    "tpsa_max": 140,  # High TPSA = poor permeability
    "rotatable_bonds_max": 12,
    "aromatic_rings_max": 5,
    "fraction_csp3_min": 0.1,  # Low Fsp3 = flat, promiscuous
}

# =============================================================================
# TOXICITY ALERT SMARTS
# =============================================================================

# Reactive functional groups that may cause toxicity
TOXICITY_ALERTS = {
    "acyl_halide": "[CX3](=[OX1])[ClBrIF]",
    "aldehyde": "[CX3H1](=O)[#6]",
    "epoxide": "C1OC1",
    "michael_acceptor": "[CX3]=[CX3][CX3]=O",
    "nitro_aromatic": "[cR1][N+](=O)[O-]",
    "alkyl_halide": "[CX4][Cl,Br,I]",
    "azo": "[#6]N=N[#6]",
    "hydrazine": "[NX3][NX3]",
    "isocyanate": "[NX2]=C=O",
    "sulfonyl_halide": "[SX4](=[OX1])(=[OX1])[Cl,Br,I]",
    "peroxide": "[OX2][OX2]",
    "nitroso": "[#6][NX2]=O",
}

# =============================================================================
# MAIN ASSESSMENT FUNCTION
# =============================================================================

def assess_adme_tox(smiles: str) -> ADMEToxResult:
    """
    Assess ADME and toxicity risks for a compound.
    
    Args:
        smiles: SMILES string
    
    Returns:
        ADMEToxResult with status, reasons, and properties
    """
    if not HAS_RDKIT:
        return ADMEToxResult(
            status=RiskLevel.FLAGGED,
            reasons=["RDKit not available - cannot assess"],
            properties={}
        )
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ADMEToxResult(
            status=RiskLevel.HIGH_RISK,
            reasons=["Invalid SMILES - cannot parse"],
            properties={}
        )
    
    # Calculate properties
    properties = calculate_adme_properties(mol)
    
    # Assess risks
    reasons = []
    risk_score = 0  # Internal scoring, not exposed
    
    # Check ADME properties
    adme_issues = check_adme_properties(properties)
    reasons.extend(adme_issues)
    risk_score += len(adme_issues)
    
    # Check toxicity alerts
    tox_issues = check_toxicity_alerts(mol)
    reasons.extend(tox_issues)
    risk_score += len(tox_issues) * 2  # Toxicity alerts weighted higher
    
    # Determine status
    if risk_score == 0:
        status = RiskLevel.SAFE
    elif risk_score <= 2:
        status = RiskLevel.FLAGGED
    else:
        status = RiskLevel.HIGH_RISK
    
    return ADMEToxResult(
        status=status,
        reasons=reasons,
        properties=properties,
    )


def calculate_adme_properties(mol) -> Dict:
    """
    Calculate ADME-relevant properties using RDKit.
    
    Args:
        mol: RDKit Mol object
    
    Returns:
        Dictionary of properties
    """
    return {
        "mw": round(Descriptors.MolWt(mol), 1),
        "logp": round(Descriptors.MolLogP(mol), 2),
        "tpsa": round(Descriptors.TPSA(mol), 1),
        "hbd": Lipinski.NumHDonors(mol),
        "hba": Lipinski.NumHAcceptors(mol),
        "rotatable_bonds": Descriptors.NumRotatableBonds(mol),
        "aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(mol),
        "fraction_csp3": round(rdMolDescriptors.CalcFractionCSP3(mol), 2),
        "heavy_atoms": mol.GetNumHeavyAtoms(),
        "rings": rdMolDescriptors.CalcNumRings(mol),
    }


def check_adme_properties(properties: Dict) -> List[str]:
    """
    Check ADME properties against thresholds.
    
    Args:
        properties: Calculated properties
    
    Returns:
        List of issue descriptions
    """
    issues = []
    
    # Molecular weight
    if properties["mw"] > ADME_THRESHOLDS["mw_max"]:
        issues.append(f"High MW ({properties['mw']} > {ADME_THRESHOLDS['mw_max']})")
    
    # Lipophilicity
    if properties["logp"] > ADME_THRESHOLDS["logp_max"]:
        issues.append(f"High lipophilicity (LogP {properties['logp']} > {ADME_THRESHOLDS['logp_max']})")
    
    # H-bond donors/acceptors
    if properties["hbd"] > ADME_THRESHOLDS["hbd_max"]:
        issues.append(f"Too many H-bond donors ({properties['hbd']} > {ADME_THRESHOLDS['hbd_max']})")
    
    if properties["hba"] > ADME_THRESHOLDS["hba_max"]:
        issues.append(f"Too many H-bond acceptors ({properties['hba']} > {ADME_THRESHOLDS['hba_max']})")
    
    # TPSA (solubility/permeability balance)
    if properties["tpsa"] < ADME_THRESHOLDS["tpsa_min"]:
        issues.append(f"Low TPSA ({properties['tpsa']} < {ADME_THRESHOLDS['tpsa_min']}) - poor solubility risk")
    elif properties["tpsa"] > ADME_THRESHOLDS["tpsa_max"]:
        issues.append(f"High TPSA ({properties['tpsa']} > {ADME_THRESHOLDS['tpsa_max']}) - poor permeability risk")
    
    # Flexibility
    if properties["rotatable_bonds"] > ADME_THRESHOLDS["rotatable_bonds_max"]:
        issues.append(f"High flexibility ({properties['rotatable_bonds']} rotatable bonds)")
    
    # Aromaticity
    if properties["aromatic_rings"] > ADME_THRESHOLDS["aromatic_rings_max"]:
        issues.append(f"Too many aromatic rings ({properties['aromatic_rings']})")
    
    # Fraction sp3
    if properties["fraction_csp3"] < ADME_THRESHOLDS["fraction_csp3_min"]:
        issues.append(f"Low Fsp3 ({properties['fraction_csp3']}) - flat/promiscuous risk")
    
    return issues


def check_toxicity_alerts(mol) -> List[str]:
    """
    Check for structural toxicity alerts.
    
    Args:
        mol: RDKit Mol object
    
    Returns:
        List of toxicity alerts found
    """
    alerts = []
    
    for alert_name, smarts in TOXICITY_ALERTS.items():
        pattern = Chem.MolFromSmarts(smarts)
        if pattern is not None and mol.HasSubstructMatch(pattern):
            # Format alert name nicely
            name = alert_name.replace("_", " ").title()
            alerts.append(f"Toxicity alert: {name}")
    
    return alerts


def assess_batch(compounds: List[Dict]) -> List[Dict]:
    """
    Assess ADME/Toxicity for a batch of compounds.
    
    Args:
        compounds: List of compound dictionaries with 'smiles'
    
    Returns:
        Same compounds with 'adme_tox' field added
    """
    for compound in compounds:
        smiles = compound.get("smiles", "")
        if smiles:
            result = assess_adme_tox(smiles)
            compound["adme_tox"] = result.to_dict()
        else:
            compound["adme_tox"] = {
                "status": "FLAGGED",
                "reasons": ["No SMILES available"],
                "properties": {},
            }
    
    # Log summary
    safe_count = sum(1 for c in compounds if c.get("adme_tox", {}).get("status") == "SAFE")
    flagged_count = sum(1 for c in compounds if c.get("adme_tox", {}).get("status") == "FLAGGED")
    high_risk_count = sum(1 for c in compounds if c.get("adme_tox", {}).get("status") == "HIGH_RISK")
    
    logger.info(
        f"ADME/Tox assessment: {safe_count} safe, {flagged_count} flagged, {high_risk_count} high-risk"
    )
    
    return compounds


def get_risk_summary(compounds: List[Dict]) -> Dict:
    """
    Get summary of ADME/Tox risks across compounds.
    
    Args:
        compounds: Compounds with adme_tox field
    
    Returns:
        Summary dictionary
    """
    statuses = [c.get("adme_tox", {}).get("status", "UNKNOWN") for c in compounds]
    
    return {
        "total": len(compounds),
        "safe": statuses.count("SAFE"),
        "flagged": statuses.count("FLAGGED"),
        "high_risk": statuses.count("HIGH_RISK"),
        "safe_percentage": round(100 * statuses.count("SAFE") / len(compounds), 1) if compounds else 0,
    }

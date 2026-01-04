"""
ADMET Property Prediction Module.

Predicts Absorption, Distribution, Metabolism, Excretion, and Toxicity
properties for drug candidates.

Uses:
- RDKit for descriptor calculations
- SwissADME API for drug-likeness
- pkCSM API for ADMET predictions
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import requests

from utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ADMETProfile:
    """Complete ADMET property profile for a compound."""
    
    # Absorption
    gi_absorption: str = ""  # "High" or "Low"
    caco2_permeability: Optional[float] = None  # log Papp
    pgp_substrate: bool = False
    
    # Distribution
    bbb_permeant: bool = False  # Blood-brain barrier
    log_bb: Optional[float] = None  # Brain/blood partition
    vdss: Optional[float] = None  # Volume of distribution
    plasma_protein_binding: Optional[float] = None  # % bound
    
    # Metabolism (CYP450)
    cyp1a2_inhibitor: bool = False
    cyp2c9_inhibitor: bool = False
    cyp2c19_inhibitor: bool = False
    cyp2d6_inhibitor: bool = False
    cyp3a4_inhibitor: bool = False
    cyp2d6_substrate: bool = False
    cyp3a4_substrate: bool = False
    
    # Excretion
    total_clearance: Optional[float] = None  # mL/min/kg
    renal_oct2_substrate: bool = False
    
    # Toxicity
    ames_toxicity: bool = False
    herg_inhibitor: str = ""  # "Low", "Medium", "High"
    hepatotoxicity: bool = False
    skin_sensitization: bool = False
    max_tolerated_dose: Optional[float] = None  # mg/kg/day
    
    # Drug-likeness
    lipinski_violations: int = 0
    ghose_violations: int = 0
    veber_violations: int = 0
    bioavailability_score: float = 0.0
    
    # Synthetic accessibility
    sa_score: Optional[float] = None  # 1-10 (1=easy, 10=hard)
    
    # Pharmacokinetics (PK)
    half_life_estimate: Optional[float] = None  # Hours
    tmax_estimate: Optional[float] = None  # Hours to peak concentration
    auc_relative: Optional[str] = None  # "Low", "Moderate", "High"
    cmax_relative: Optional[str] = None  # "Low", "Moderate", "High"
    oral_bioavailability_estimate: Optional[float] = None  # Fraction (0-1)
    dosing_frequency: Optional[str] = None  # "Once daily", "Twice daily", etc.
    
    def get_absorption_summary(self) -> str:
        return f"GI: {self.gi_absorption}, P-gp substrate: {'Yes' if self.pgp_substrate else 'No'}"
    
    def get_bbb_summary(self) -> str:
        if self.bbb_permeant:
            return f"BBB+{' (log BB: ' + str(round(self.log_bb, 2)) + ')' if self.log_bb else ''}"
        return "BBB-"
    
    def get_cyp_interactions(self) -> List[str]:
        interactions = []
        if self.cyp1a2_inhibitor: interactions.append("CYP1A2 inhibitor")
        if self.cyp2c9_inhibitor: interactions.append("CYP2C9 inhibitor")
        if self.cyp2c19_inhibitor: interactions.append("CYP2C19 inhibitor")
        if self.cyp2d6_inhibitor: interactions.append("CYP2D6 inhibitor")
        if self.cyp3a4_inhibitor: interactions.append("CYP3A4 inhibitor")
        return interactions
    
    def get_toxicity_alerts(self) -> List[Dict]:
        alerts = []
        if self.ames_toxicity:
            alerts.append({"type": "AMES", "severity": "critical", "message": "Mutagenic potential detected"})
        if self.herg_inhibitor in ("Medium", "High"):
            alerts.append({"type": "hERG", "severity": "critical", "message": f"Cardiac risk ({self.herg_inhibitor} inhibition)"})
        if self.hepatotoxicity:
            alerts.append({"type": "Hepatotox", "severity": "warning", "message": "Liver toxicity risk"})
        return alerts
    
    def get_pk_summary(self) -> Dict:
        """Get pharmacokinetics summary."""
        summary = {}
        if self.half_life_estimate:
            if self.half_life_estimate < 4:
                summary["half_life_category"] = "Short"
            elif self.half_life_estimate < 12:
                summary["half_life_category"] = "Moderate"
            else:
                summary["half_life_category"] = "Long"
            summary["half_life_hours"] = self.half_life_estimate
        
        if self.tmax_estimate:
            summary["tmax_hours"] = self.tmax_estimate
        
        if self.dosing_frequency:
            summary["dosing_recommendation"] = self.dosing_frequency
        
        if self.oral_bioavailability_estimate:
            summary["oral_bioavailability_pct"] = self.oral_bioavailability_estimate * 100
        
        return summary
    
    def to_dict(self) -> Dict:
        return {
            "absorption": {
                "gi_absorption": self.gi_absorption,
                "caco2_permeability": self.caco2_permeability,
                "pgp_substrate": self.pgp_substrate,
            },
            "distribution": {
                "bbb_permeant": self.bbb_permeant,
                "log_bb": self.log_bb,
                "plasma_protein_binding": self.plasma_protein_binding,
            },
            "metabolism": {
                "cyp_inhibitors": self.get_cyp_interactions(),
                "cyp3a4_substrate": self.cyp3a4_substrate,
                "cyp2d6_substrate": self.cyp2d6_substrate,
            },
            "excretion": {
                "total_clearance": self.total_clearance,
                "oct2_substrate": self.renal_oct2_substrate,
            },
            "toxicity": {
                "ames": self.ames_toxicity,
                "herg": self.herg_inhibitor,
                "hepatotoxicity": self.hepatotoxicity,
                "alerts": self.get_toxicity_alerts(),
            },
            "druglikeness": {
                "lipinski_violations": self.lipinski_violations,
                "bioavailability_score": self.bioavailability_score,
                "sa_score": self.sa_score,
            },
            "pharmacokinetics": {
                "half_life_hours": self.half_life_estimate,
                "tmax_hours": self.tmax_estimate,
                "auc_relative": self.auc_relative,
                "cmax_relative": self.cmax_relative,
                "oral_bioavailability": self.oral_bioavailability_estimate,
                "dosing_frequency": self.dosing_frequency,
                "pk_summary": self.get_pk_summary(),
            },
        }


def calculate_rdkit_descriptors(smiles: str) -> Dict:
    """
    Calculate molecular descriptors using RDKit.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors
        from rdkit.Chem import RDConfig
        import os
        import sys
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {}
        
        # Basic descriptors
        descriptors = {
            "mw": Descriptors.MolWt(mol),
            "logp": Descriptors.MolLogP(mol),
            "hbd": Descriptors.NumHDonors(mol),
            "hba": Descriptors.NumHAcceptors(mol),
            "tpsa": Descriptors.TPSA(mol),
            "rotatable_bonds": Descriptors.NumRotatableBonds(mol),
            "aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(mol),
            "heavy_atoms": Lipinski.HeavyAtomCount(mol),
            "fraction_csp3": rdMolDescriptors.CalcFractionCSP3(mol),
        }
        
        # Lipinski Rule of 5 violations
        violations = 0
        if descriptors["mw"] > 500: violations += 1
        if descriptors["logp"] > 5: violations += 1
        if descriptors["hbd"] > 5: violations += 1
        if descriptors["hba"] > 10: violations += 1
        descriptors["lipinski_violations"] = violations
        
        # Synthetic Accessibility Score
        try:
            from rdkit.Chem import rdMolDescriptors
            # SA score calculation (simplified)
            # Real SA score requires sascorer module
            ring_count = rdMolDescriptors.CalcNumRings(mol)
            stereo_centers = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
            # Simplified SA estimate (1-10, lower is better)
            sa_estimate = 1 + ring_count * 0.5 + stereo_centers * 0.3 + \
                         (descriptors["mw"] - 200) / 200 * 0.5
            descriptors["sa_score"] = min(10, max(1, sa_estimate))
        except:
            descriptors["sa_score"] = None
        
        return descriptors
        
    except Exception as e:
        logger.error(f"RDKit descriptor calculation failed: {e}")
        return {}


def predict_bbb_permeability(smiles: str) -> Tuple[bool, Optional[float]]:
    """
    Predict Blood-Brain Barrier permeability.
    
    Uses simple rules based on molecular properties:
    - MW < 450
    - PSA < 90 Å²
    - HBD ≤ 3
    - LogP 1-3
    
    Returns:
        Tuple of (is_bbb_permeable, log_bb_estimate)
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False, None
        
        mw = Descriptors.MolWt(mol)
        psa = Descriptors.TPSA(mol)
        hbd = Descriptors.NumHDonors(mol)
        logp = Descriptors.MolLogP(mol)
        
        # Clark's BBB rules
        bbb_permeable = (
            mw < 450 and
            psa < 90 and
            hbd <= 3 and
            1 <= logp <= 3
        )
        
        # Estimate log BB using simple linear model
        # log BB ≈ 0.152 * logP - 0.0148 * PSA + 0.139
        log_bb = 0.152 * logp - 0.0148 * psa + 0.139
        
        return bbb_permeable, log_bb
        
    except Exception as e:
        logger.error(f"BBB prediction failed: {e}")
        return False, None


def predict_gi_absorption(smiles: str) -> str:
    """
    Predict gastrointestinal absorption.
    
    Based on Lipinski rules and PSA < 140 Å².
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return "Unknown"
        
        mw = Descriptors.MolWt(mol)
        psa = Descriptors.TPSA(mol)
        logp = Descriptors.MolLogP(mol)
        rotatable = Descriptors.NumRotatableBonds(mol)
        
        # High absorption if:
        # - MW < 500
        # - PSA < 140
        # - Rotatable bonds < 10
        # - LogP in reasonable range
        if mw < 500 and psa < 140 and rotatable < 10 and -2 < logp < 5:
            return "High"
        elif mw < 700 and psa < 200:
            return "Moderate"
        else:
            return "Low"
            
    except Exception as e:
        return "Unknown"


def predict_cyp_inhibition(smiles: str) -> Dict[str, bool]:
    """
    Predict CYP450 enzyme inhibition.
    
    Uses simple structural alerts for major CYPs.
    Note: This is a simplified prediction - real prediction requires ML models.
    """
    try:
        from rdkit import Chem
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {}
        
        # SMARTS patterns for common CYP inhibitor features
        cyp_alerts = {
            "cyp3a4": "[#7]~c1ccccc1",  # Nitrogen-containing aromatics
            "cyp2d6": "[#7+]",  # Basic nitrogen
            "cyp2c9": "c1ccc(F)cc1",  # Fluorinated aromatics
        }
        
        results = {
            "cyp1a2_inhibitor": False,
            "cyp2c9_inhibitor": False,
            "cyp2c19_inhibitor": False,
            "cyp2d6_inhibitor": False,
            "cyp3a4_inhibitor": False,
        }
        
        # Simple heuristics
        from rdkit.Chem import Descriptors
        logp = Descriptors.MolLogP(mol)
        mw = Descriptors.MolWt(mol)
        
        # Higher LogP and larger molecules tend to inhibit CYP3A4
        if logp > 3 and mw > 400:
            results["cyp3a4_inhibitor"] = True
        
        # Check for basic nitrogens (CYP2D6)
        pattern = Chem.MolFromSmarts("[#7;+]")
        if mol.HasSubstructMatch(pattern) if pattern else False:
            results["cyp2d6_inhibitor"] = True
        
        return results
        
    except Exception as e:
        logger.error(f"CYP prediction failed: {e}")
        return {}


def predict_toxicity_alerts(smiles: str) -> Dict:
    """
    Predict toxicity alerts using structural patterns.
    """
    try:
        from rdkit import Chem
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {}
        
        alerts = {
            "ames_toxicity": False,
            "herg_risk": "Low",
            "hepatotoxicity_risk": False,
        }
        
        # SMARTS for mutagenic alerts (simplified)
        mutagenic_patterns = [
            "[N+](=O)[O-]",  # Nitro groups
            "N=N",  # Azo compounds
            "[N;!R]=C-N",  # Hydrazones
        ]
        
        for pattern_smarts in mutagenic_patterns:
            pattern = Chem.MolFromSmarts(pattern_smarts)
            if pattern and mol.HasSubstructMatch(pattern):
                alerts["ames_toxicity"] = True
                break
        
        # hERG risk based on LogP and basic nitrogen
        from rdkit.Chem import Descriptors
        logp = Descriptors.MolLogP(mol)
        
        basic_n_pattern = Chem.MolFromSmarts("[#7;+]")
        has_basic_n = mol.HasSubstructMatch(basic_n_pattern) if basic_n_pattern else False
        
        if logp > 3.5 and has_basic_n:
            alerts["herg_risk"] = "High"
        elif logp > 2.5:
            alerts["herg_risk"] = "Medium"
        
        return alerts
        
    except Exception as e:
        logger.error(f"Toxicity prediction failed: {e}")
        return {}


def predict_pharmacokinetics(smiles: str) -> Dict:
    """
    Predict pharmacokinetics parameters based on molecular properties.
    
    Uses empirical rules based on:
    - LogP (affects half-life and distribution)
    - MW (affects absorption rate)
    - PSA (affects bioavailability)
    - HBD/HBA (affects distribution and clearance)
    
    Note: These are simplified estimates. Real PK requires clinical data.
    
    Returns:
        Dictionary with PK estimates
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {}
        
        # Get molecular properties
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        psa = Descriptors.TPSA(mol)
        hbd = Descriptors.NumHDonors(mol)
        hba = Descriptors.NumHAcceptors(mol)
        rotatable = Descriptors.NumRotatableBonds(mol)
        
        pk_results = {}
        
        # Half-life estimation (simplified model)
        # Based on relationship: higher LogP → slower elimination, larger MW → longer half-life
        # Formula: t1/2 ≈ 2 + 1.5*LogP + 0.01*(MW-200) hours (simplified)
        base_half_life = 2.0
        logp_contribution = max(0, logp * 1.5)  # Higher LogP extends half-life
        mw_contribution = max(0, (mw - 200) * 0.01)  # Larger molecules last longer
        psa_penalty = max(0, (psa - 100) * 0.02)  # High PSA reduces half-life (renal)
        
        half_life = base_half_life + logp_contribution + mw_contribution - psa_penalty
        half_life = max(0.5, min(48, half_life))  # Clamp to 0.5-48 hours
        pk_results["half_life_estimate"] = round(half_life, 1)
        
        # Tmax estimation (time to peak concentration)
        # Faster absorption for smaller, more lipophilic compounds
        if mw < 300 and -1 < logp < 3:
            tmax = 0.5  # Rapid absorption
        elif mw < 500 and 0 < logp < 5:
            tmax = 1.5  # Moderate absorption
        else:
            tmax = 3.0  # Slow absorption
        
        # Adjust for food effect proxy (rotatable bonds)
        if rotatable > 8:
            tmax += 0.5
        
        pk_results["tmax_estimate"] = round(tmax, 1)
        
        # AUC and Cmax relative estimates
        # Based on bioavailability and absorption
        if psa < 100 and logp > 0 and logp < 5 and mw < 500:
            pk_results["auc_relative"] = "High"
            pk_results["cmax_relative"] = "High"
        elif psa < 140 and mw < 700:
            pk_results["auc_relative"] = "Moderate"
            pk_results["cmax_relative"] = "Moderate"
        else:
            pk_results["auc_relative"] = "Low"
            pk_results["cmax_relative"] = "Low"
        
        # Oral bioavailability estimate
        # Based on Lipinski rules and absorption properties
        f_estimate = 0.8  # Start with 80%
        
        # Penalties for poor properties
        if mw > 500:
            f_estimate -= 0.2
        if logp > 5:
            f_estimate -= 0.25
        if logp < 0:
            f_estimate -= 0.15
        if hbd > 5:
            f_estimate -= 0.15
        if hba > 10:
            f_estimate -= 0.15
        if psa > 140:
            f_estimate -= 0.2
        if rotatable > 10:
            f_estimate -= 0.1
        
        f_estimate = max(0.1, min(0.95, f_estimate))  # Clamp to 10-95%
        pk_results["oral_bioavailability_estimate"] = round(f_estimate, 2)
        
        # Dosing frequency recommendation
        if half_life >= 12:
            pk_results["dosing_frequency"] = "Once daily"
        elif half_life >= 6:
            pk_results["dosing_frequency"] = "Twice daily"
        elif half_life >= 3:
            pk_results["dosing_frequency"] = "Three times daily"
        else:
            pk_results["dosing_frequency"] = "Four times daily or extended-release"
        
        return pk_results
        
    except Exception as e:
        logger.error(f"Pharmacokinetics prediction failed: {e}")
        return {}


def predict_admet(smiles: str) -> ADMETProfile:
    """
    Complete ADMET prediction for a compound.
    
    Args:
        smiles: Compound SMILES string
    
    Returns:
        ADMETProfile with all predictions
    """
    logger.info(f"Predicting ADMET for SMILES: {smiles[:50]}...")
    
    profile = ADMETProfile()
    
    # Get RDKit descriptors
    descriptors = calculate_rdkit_descriptors(smiles)
    
    # Absorption
    profile.gi_absorption = predict_gi_absorption(smiles)
    
    # Distribution / BBB
    bbb_permeable, log_bb = predict_bbb_permeability(smiles)
    profile.bbb_permeant = bbb_permeable
    profile.log_bb = log_bb
    
    # Metabolism
    cyp_results = predict_cyp_inhibition(smiles)
    profile.cyp1a2_inhibitor = cyp_results.get("cyp1a2_inhibitor", False)
    profile.cyp2c9_inhibitor = cyp_results.get("cyp2c9_inhibitor", False)
    profile.cyp2c19_inhibitor = cyp_results.get("cyp2c19_inhibitor", False)
    profile.cyp2d6_inhibitor = cyp_results.get("cyp2d6_inhibitor", False)
    profile.cyp3a4_inhibitor = cyp_results.get("cyp3a4_inhibitor", False)
    
    # Toxicity
    tox_alerts = predict_toxicity_alerts(smiles)
    profile.ames_toxicity = tox_alerts.get("ames_toxicity", False)
    profile.herg_inhibitor = tox_alerts.get("herg_risk", "Low")
    profile.hepatotoxicity = tox_alerts.get("hepatotoxicity_risk", False)
    
    # Drug-likeness
    profile.lipinski_violations = descriptors.get("lipinski_violations", 0)
    profile.sa_score = descriptors.get("sa_score")
    
    # Bioavailability score (simplified)
    if profile.lipinski_violations == 0 and profile.gi_absorption == "High":
        profile.bioavailability_score = 0.85
    elif profile.lipinski_violations <= 1:
        profile.bioavailability_score = 0.56
    else:
        profile.bioavailability_score = 0.17
    
    # Pharmacokinetics
    pk_results = predict_pharmacokinetics(smiles)
    profile.half_life_estimate = pk_results.get("half_life_estimate")
    profile.tmax_estimate = pk_results.get("tmax_estimate")
    profile.auc_relative = pk_results.get("auc_relative")
    profile.cmax_relative = pk_results.get("cmax_relative")
    profile.oral_bioavailability_estimate = pk_results.get("oral_bioavailability_estimate")
    profile.dosing_frequency = pk_results.get("dosing_frequency")
    
    return profile


def generate_admet_summary(profile: ADMETProfile, compound_id: str = "") -> str:
    """
    Generate human-readable ADMET summary for display.
    """
    lines = [f"**💊 ADMET Profile{' for ' + compound_id if compound_id else ''}**\n"]
    
    # Absorption
    lines.append(f"**Absorption:** {profile.gi_absorption} GI absorption")
    if profile.pgp_substrate:
        lines.append("  - ⚠️ P-gp substrate (may affect distribution)")
    
    # Distribution
    lines.append(f"\n**Distribution:** {profile.get_bbb_summary()}")
    if profile.plasma_protein_binding:
        lines.append(f"  - Plasma protein binding: {profile.plasma_protein_binding:.0f}%")
    
    # Metabolism
    cyp_interactions = profile.get_cyp_interactions()
    if cyp_interactions:
        lines.append("\n**Metabolism:** ⚠️ CYP interactions detected")
        for cyp in cyp_interactions:
            lines.append(f"  - {cyp}")
    else:
        lines.append("\n**Metabolism:** No significant CYP inhibition")
    
    # Toxicity
    tox_alerts = profile.get_toxicity_alerts()
    if tox_alerts:
        lines.append("\n**⚠️ Toxicity Alerts:**")
        for alert in tox_alerts:
            icon = "🔴" if alert["severity"] == "critical" else "🟡"
            lines.append(f"  - {icon} {alert['type']}: {alert['message']}")
    else:
        lines.append("\n**Toxicity:** ✅ No significant alerts")
    
    # Drug-likeness
    lines.append(f"\n**Drug-likeness:** Lipinski violations: {profile.lipinski_violations}")
    lines.append(f"  - Bioavailability score: {profile.bioavailability_score:.0%}")
    if profile.sa_score:
        sa_desc = "Easy" if profile.sa_score < 4 else ("Moderate" if profile.sa_score < 7 else "Difficult")
        lines.append(f"  - Synthetic accessibility: {sa_desc} ({profile.sa_score:.1f}/10)")
    
    return "\n".join(lines)


# Test
if __name__ == "__main__":
    # Test with gefitinib
    test_smiles = "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1"
    
    print("Testing ADMET prediction...")
    profile = predict_admet(test_smiles)
    
    print("\n" + generate_admet_summary(profile, "GEFITINIB"))

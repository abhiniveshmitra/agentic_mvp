"""
ADME/Toxicity Stage-2 Agent (RDKit-Only, NON-LLM).

Purpose: Flag developability risk with FULL explanations.

Rules (IMMUTABLE):
- Rule-based only
- No ML models
- No composite scoring
- Fully explainable
- Every flag must have a reason

Forbidden:
- Using ADME to alter ranking
- Hiding flagged molecules
- Creating composite "safety scores"
- Making clinical predictions

This is a NON-LLM, RDKit-only agent.
"""

from typing import Dict, List, Optional
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
    logger.warning("RDKit not available - ADME/Tox Stage-2 disabled")


class Stage2RiskLevel(str, Enum):
    """Stage-2 risk levels (same as Stage-1 for consistency)."""
    SAFE = "SAFE"
    FLAGGED = "FLAGGED"
    HIGH_RISK = "HIGH_RISK"


# =============================================================================
# STAGE-2 ADME RULES (From stage2-rules-plain-english.md)
# =============================================================================

STAGE2_ADME_RULES = {
    "HIGH_LIPOPHILICITY": {
        "condition": lambda p: p["logp"] > 7.5,
        "threshold": 7.5,
        "property": "logp",
        "scientific_rationale": (
            "Highly lipophilic molecules often exhibit poor aqueous solubility "
            "and increased nonspecific membrane binding, leading to higher "
            "plasma protein binding and potential accumulation in fatty tissues."
        ),
        "practical_implication": (
            "May require advanced formulation strategies. Increased risk of "
            "off-target effects. Potential bioavailability issues."
        ),
        "what_it_does_not_mean": (
            "Does not prove the molecule is toxic. Does not mean oral "
            "administration is impossible. Many approved drugs exceed this threshold."
        ),
    },
    "LOW_POLAR_SURFACE_SOLUBILITY": {
        "condition": lambda p: p["tpsa"] < 40 and p["logp"] > 6,
        "threshold": "TPSA < 40 AND LogP > 6",
        "property": ["tpsa", "logp"],
        "scientific_rationale": (
            "Low TPSA indicates reduced hydrogen bonding capacity. Combined "
            "with high LogP, suggests poor aqueous solubility. May partition "
            "into membranes rather than staying in solution."
        ),
        "practical_implication": (
            "Dissolution-limited absorption likely. May require solubilization technology."
        ),
        "what_it_does_not_mean": (
            "Does not predict absolute solubility. Some CNS drugs intentionally have low TPSA."
        ),
    },
    "HIGH_TPSA_PERMEABILITY": {
        "condition": lambda p: p["tpsa"] > 140,
        "threshold": 140,
        "property": "tpsa",
        "scientific_rationale": (
            "High TPSA correlates with poor passive membrane permeability. "
            "Molecules may not cross intestinal epithelium effectively."
        ),
        "practical_implication": (
            "May require active transport. Parenteral administration may be needed."
        ),
        "what_it_does_not_mean": (
            "Does not rule out efficacy. Biologics and some antibiotics work despite high TPSA."
        ),
    },
    "HIGH_FLEXIBILITY": {
        "condition": lambda p: p["rotatable_bonds"] > 12,
        "threshold": 12,
        "property": "rotatable_bonds",
        "scientific_rationale": (
            "Highly flexible molecules have larger conformational entropy "
            "penalty on binding, reduced oral bioavailability, and potential "
            "for multiple binding modes."
        ),
        "practical_implication": (
            "May show lower potency than expected. Consider rigidifying scaffold."
        ),
        "what_it_does_not_mean": (
            "Does NOT prevent binding. Some peptide-like drugs are flexible."
        ),
    },
    "LOW_FSP3_PROMISCUITY": {
        "condition": lambda p: p["fraction_csp3"] < 0.1,
        "threshold": 0.1,
        "property": "fraction_csp3",
        "scientific_rationale": (
            "Low Fsp3 indicates a flat, aromatic molecule. Flat molecules are "
            "associated with promiscuous binding, off-target effects, and poor solubility."
        ),
        "practical_implication": (
            "Consider adding 3D character in optimization."
        ),
        "what_it_does_not_mean": (
            "Many kinase inhibitors are aromatic and effective. Context matters."
        ),
    },
    "EXCESSIVE_MW": {
        "condition": lambda p: p["mw"] > 700,
        "threshold": 700,
        "property": "mw",
        "scientific_rationale": (
            "High molecular weight compounds often have poor oral absorption "
            "and reduced BBB penetration."
        ),
        "practical_implication": (
            "Consider scaffold simplification if oral delivery is required."
        ),
        "what_it_does_not_mean": (
            "Does not rule out efficacy. Many macrocyclic drugs are successful."
        ),
    },
}


# =============================================================================
# PAINS MOTIF DETAILS
# =============================================================================

PAINS_MOTIF_INFO = {
    "quinone": {
        "description": "Quinone-like substructure",
        "known_issue": (
            "Associated with redox cycling and assay interference, "
            "leading to frequent false positives in biochemical assays."
        ),
    },
    "rhodanine": {
        "description": "Rhodanine core",
        "known_issue": (
            "Known pan-assay interference compound with poor selectivity."
        ),
    },
    "enone": {
        "description": "Michael acceptor (α,β-unsaturated carbonyl)",
        "known_issue": (
            "Can react covalently with protein nucleophiles."
        ),
    },
}


@dataclass
class ADMEToxRule:
    """A triggered ADME/Tox rule with full explanation."""
    rule_id: str
    condition: str
    triggered: bool
    value: float
    threshold: float
    scientific_rationale: str
    practical_implication: str
    what_it_does_not_mean: str
    
    def to_dict(self) -> Dict:
        return {
            "rule_id": self.rule_id,
            "condition": self.condition,
            "triggered": self.triggered,
            "value": self.value,
            "threshold": self.threshold,
            "scientific_rationale": self.scientific_rationale,
            "practical_implication": self.practical_implication,
            "what_it_does_not_mean": self.what_it_does_not_mean,
        }


@dataclass
class PAINSDetail:
    """Detailed PAINS alert information."""
    pains_alert: str
    matched_motif: str
    known_issue: str
    important_note: str = (
        "PAINS alerts indicate risk, not proof of inactivity. "
        "Orthogonal validation assays are recommended."
    )
    
    def to_dict(self) -> Dict:
        return {
            "pains_alert": self.pains_alert,
            "matched_motif": self.matched_motif,
            "known_issue": self.known_issue,
            "important_note": self.important_note,
        }


@dataclass
class ADMEToxStage2Explanation:
    """Full Stage-2 ADME/Tox explanation (follows unified schema)."""
    
    result: Dict = field(default_factory=dict)
    raw_values: Dict = field(default_factory=dict)
    rules_triggered: List[Dict] = field(default_factory=list)
    pains_details: Optional[Dict] = None
    limitations: List[str] = field(default_factory=list)
    summary: str = ""
    
    def to_dict(self) -> Dict:
        result = {
            "result": self.result,
            "raw_values": self.raw_values,
            "rules_triggered": self.rules_triggered,
            "limitations": self.limitations,
            "summary": self.summary,
        }
        if self.pains_details:
            result["pains_details"] = self.pains_details
        return result


@dataclass
class ADMEToxStage2Result:
    """Complete Stage-2 ADME/Tox assessment result."""
    
    smiles: str
    compound_id: str
    label: Stage2RiskLevel
    explanation: ADMEToxStage2Explanation = field(default_factory=ADMEToxStage2Explanation)
    
    def to_dict(self) -> Dict:
        return {
            "smiles": self.smiles,
            "compound_id": self.compound_id,
            "adme_tox_label": self.label.value,
            "explanation": self.explanation.to_dict(),
        }


class ADMEToxStage2:
    """
    ADME/Toxicity Stage-2 Agent.
    
    Extends Stage-1 ADME/Tox with FULL explanations following the
    unified explanation schema. Every flag has a scientific rationale.
    """
    
    def __init__(self):
        """Initialize ADME/Tox Stage-2 agent."""
        if not HAS_RDKIT:
            logger.error("RDKit required for ADME/Tox Stage-2")
        logger.info("ADMEToxStage2 initialized")
    
    def assess(self, smiles: str, compound_id: str = "") -> ADMEToxStage2Result:
        """
        Assess ADME/Tox with full explanations.
        
        Args:
            smiles: SMILES string
            compound_id: Compound identifier
        
        Returns:
            ADMEToxStage2Result with label and full explanation
        """
        if not HAS_RDKIT:
            return self._unavailable_result(smiles, compound_id)
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return self._invalid_smiles_result(smiles, compound_id)
        
        # Calculate all properties
        properties = self._calculate_properties(mol)
        
        # Apply all rules and collect explanations
        triggered_rules = self._apply_rules(properties)
        
        # Check PAINS
        pains_details = self._check_pains_detailed(mol)
        
        # Determine final label
        risk_score = len(triggered_rules)
        if pains_details:
            risk_score += 1  # PAINS adds to risk
        
        if risk_score == 0:
            label = Stage2RiskLevel.SAFE
        elif risk_score <= 2:
            label = Stage2RiskLevel.FLAGGED
        else:
            label = Stage2RiskLevel.HIGH_RISK
        
        # Build explanation
        explanation = self._build_explanation(
            properties=properties,
            triggered_rules=triggered_rules,
            pains_details=pains_details,
            label=label,
        )
        
        return ADMEToxStage2Result(
            smiles=smiles,
            compound_id=compound_id,
            label=label,
            explanation=explanation,
        )
    
    def assess_batch(
        self,
        candidates: List[Dict],
    ) -> List[ADMEToxStage2Result]:
        """
        Assess ADME/Tox for a batch of candidates.
        
        Args:
            candidates: List of dicts with 'smiles' and 'compound_id'
        
        Returns:
            List of ADMEToxStage2Result objects
        """
        results = []
        
        for candidate in candidates:
            smiles = candidate.get("smiles", "")
            compound_id = candidate.get("compound_id", "")
            
            result = self.assess(smiles, compound_id)
            results.append(result)
        
        # Log summary
        safe = sum(1 for r in results if r.label == Stage2RiskLevel.SAFE)
        flagged = sum(1 for r in results if r.label == Stage2RiskLevel.FLAGGED)
        high_risk = sum(1 for r in results if r.label == Stage2RiskLevel.HIGH_RISK)
        
        logger.info(f"ADME/Tox Stage-2: {safe} SAFE, {flagged} FLAGGED, {high_risk} HIGH_RISK")
        
        return results
    
    def _calculate_properties(self, mol) -> Dict:
        """Calculate all ADME-relevant properties."""
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
    
    def _apply_rules(self, properties: Dict) -> List[ADMEToxRule]:
        """Apply all ADME rules and return triggered ones with explanations."""
        triggered = []
        
        for rule_id, rule_def in STAGE2_ADME_RULES.items():
            try:
                is_triggered = rule_def["condition"](properties)
                
                if is_triggered:
                    # Get the value that triggered the rule
                    prop = rule_def["property"]
                    if isinstance(prop, list):
                        value_str = ", ".join(f"{p}={properties[p]}" for p in prop)
                        value = properties[prop[0]]  # Use first for threshold comparison
                    else:
                        value = properties[prop]
                        value_str = str(value)
                    
                    threshold = rule_def["threshold"]
                    if isinstance(threshold, str):
                        condition_str = threshold
                    else:
                        condition_str = f"{prop} {value} > {threshold}"
                    
                    triggered.append(ADMEToxRule(
                        rule_id=rule_id,
                        condition=condition_str,
                        triggered=True,
                        value=value,
                        threshold=threshold if not isinstance(threshold, str) else 0,
                        scientific_rationale=rule_def["scientific_rationale"],
                        practical_implication=rule_def["practical_implication"],
                        what_it_does_not_mean=rule_def["what_it_does_not_mean"],
                    ))
                    
            except Exception as e:
                logger.warning(f"Rule {rule_id} evaluation failed: {e}")
        
        return triggered
    
    def _check_pains_detailed(self, mol) -> Optional[PAINSDetail]:
        """Check PAINS with detailed motif information."""
        try:
            params = FilterCatalogParams()
            params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
            catalog = FilterCatalog.FilterCatalog(params)
            
            entries = catalog.GetMatches(mol)
            
            if entries:
                # Get first matching alert
                alert = entries[0].GetDescription()
                
                # Try to identify motif type
                motif_info = None
                for motif_key, info in PAINS_MOTIF_INFO.items():
                    if motif_key.lower() in alert.lower():
                        motif_info = info
                        break
                
                return PAINSDetail(
                    pains_alert=alert,
                    matched_motif=motif_info["description"] if motif_info else alert,
                    known_issue=motif_info["known_issue"] if motif_info else (
                        "This motif is associated with assay interference and false positives."
                    ),
                )
            
            return None
            
        except Exception as e:
            logger.warning(f"PAINS check failed: {e}")
            return None
    
    def _build_explanation(
        self,
        properties: Dict,
        triggered_rules: List[ADMEToxRule],
        pains_details: Optional[PAINSDetail],
        label: Stage2RiskLevel,
    ) -> ADMEToxStage2Explanation:
        """Build full explanation following unified schema."""
        
        # Build summary
        if label == Stage2RiskLevel.SAFE:
            summary = "No ADME/Tox flags detected. Compound passes developability screening."
        else:
            issues = [r.rule_id.replace("_", " ").title() for r in triggered_rules]
            if pains_details:
                issues.append("PAINS Alert")
            summary = f"{label.value}: {', '.join(issues)}"
        
        return ADMEToxStage2Explanation(
            result={
                "label": label.value,
            },
            raw_values=properties,
            rules_triggered=[r.to_dict() for r in triggered_rules],
            pains_details=pains_details.to_dict() if pains_details else None,
            limitations=[
                "Rule-based assessment only, not predictive modeling",
                "Does not account for metabolites",
                "Clinical outcome not predicted",
                "Drug-drug interactions not assessed",
            ],
            summary=summary,
        )
    
    def _unavailable_result(self, smiles: str, compound_id: str) -> ADMEToxStage2Result:
        """Result when RDKit is not available."""
        return ADMEToxStage2Result(
            smiles=smiles,
            compound_id=compound_id,
            label=Stage2RiskLevel.FLAGGED,
            explanation=ADMEToxStage2Explanation(
                result={"label": "FLAGGED"},
                raw_values={},
                rules_triggered=[],
                limitations=["RDKit not available - cannot assess"],
                summary="Assessment unavailable - RDKit required",
            ),
        )
    
    def _invalid_smiles_result(self, smiles: str, compound_id: str) -> ADMEToxStage2Result:
        """Result for invalid SMILES."""
        return ADMEToxStage2Result(
            smiles=smiles,
            compound_id=compound_id,
            label=Stage2RiskLevel.HIGH_RISK,
            explanation=ADMEToxStage2Explanation(
                result={"label": "HIGH_RISK"},
                raw_values={},
                rules_triggered=[],
                limitations=["Invalid SMILES - cannot parse structure"],
                summary="HIGH_RISK: Invalid molecular structure",
            ),
        )


def create_adme_tox_stage2() -> ADMEToxStage2:
    """Factory function to create ADMEToxStage2 agent."""
    return ADMEToxStage2()

"""
Patent Risk Stub Agent.

Purpose: Placeholder for patent risk assessment (deferred in MVP).

This stub explicitly returns NOT_EVALUATED to show intentional design,
not omission. Patent analysis will be enabled in Phase 2.1.
"""

from typing import Dict, Optional
from dataclasses import dataclass, field
from datetime import datetime

from utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PatentRiskExplanation:
    """Explanation object for patent risk (follows unified schema)."""
    
    result: Dict = field(default_factory=dict)
    raw_values: Dict = field(default_factory=dict)
    rules_triggered: list = field(default_factory=list)
    limitations: list = field(default_factory=list)
    disclaimer: str = ""
    summary: str = ""
    
    def to_dict(self) -> Dict:
        return {
            "result": self.result,
            "raw_values": self.raw_values,
            "rules_triggered": self.rules_triggered,
            "limitations": self.limitations,
            "disclaimer": self.disclaimer,
            "summary": self.summary,
        }


@dataclass
class PatentRiskResult:
    """Patent risk assessment result."""
    
    smiles: str
    compound_id: str
    inchikey: Optional[str]
    patent_risk: str  # "CLEAR", "POTENTIAL_RISK", "LIKELY_ENCUMBERED", "NOT_EVALUATED"
    explanation: PatentRiskExplanation = field(default_factory=PatentRiskExplanation)
    
    def to_dict(self) -> Dict:
        return {
            "smiles": self.smiles,
            "compound_id": self.compound_id,
            "inchikey": self.inchikey,
            "patent_risk": self.patent_risk,
            "explanation": self.explanation.to_dict(),
        }


class PatentRiskStub:
    """
    Patent Risk Stub Agent.
    
    Returns NOT_EVALUATED for all compounds.
    This is intentional design for MVP - patent analysis deferred to Phase 2.1.
    """
    
    def __init__(self):
        """Initialize Patent Risk Stub."""
        logger.info("PatentRiskStub initialized (patent analysis deferred in MVP)")
    
    def assess(self, smiles: str, compound_id: str = "") -> PatentRiskResult:
        """
        Return NOT_EVALUATED status for patent risk.
        
        Args:
            smiles: SMILES string
            compound_id: Compound identifier
        
        Returns:
            PatentRiskResult with NOT_EVALUATED status
        """
        # Try to generate InChIKey if RDKit available
        inchikey = self._generate_inchikey(smiles)
        
        return PatentRiskResult(
            smiles=smiles,
            compound_id=compound_id,
            inchikey=inchikey,
            patent_risk="NOT_EVALUATED",
            explanation=PatentRiskExplanation(
                result={
                    "label": "NOT_EVALUATED",
                    "reason": "Patent analysis deferred in Stage-2 MVP",
                },
                raw_values={
                    "inchikey": inchikey,
                    "assessment_date": datetime.now().isoformat(),
                },
                rules_triggered=[],
                limitations=[
                    "Patent search not implemented in MVP",
                    "Manual IP review recommended for candidates of interest",
                    "This stub ensures intentional design, not omission",
                ],
                disclaimer=(
                    "This assessment is intentionally deferred. Patent analysis "
                    "will be available in Phase 2.1. For compounds advancing to "
                    "development, professional IP counsel is required."
                ),
                summary="Patent risk NOT_EVALUATED (deferred in MVP)",
            ),
        )
    
    def assess_batch(self, candidates: list) -> list:
        """
        Assess patent risk for a batch of candidates.
        
        Args:
            candidates: List of dicts with 'smiles' and 'compound_id'
        
        Returns:
            List of PatentRiskResult objects
        """
        results = []
        
        for candidate in candidates:
            smiles = candidate.get("smiles", "")
            compound_id = candidate.get("compound_id", "")
            result = self.assess(smiles, compound_id)
            results.append(result)
        
        logger.info(f"Patent risk stub: {len(results)} compounds marked NOT_EVALUATED")
        
        return results
    
    def _generate_inchikey(self, smiles: str) -> Optional[str]:
        """Generate InChIKey from SMILES if RDKit available."""
        try:
            from rdkit import Chem
            from rdkit.Chem.inchi import MolToInchiKey
            
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                return MolToInchiKey(mol)
            return None
        except Exception:
            return None


def create_patent_risk_stub() -> PatentRiskStub:
    """Factory function to create PatentRiskStub."""
    return PatentRiskStub()

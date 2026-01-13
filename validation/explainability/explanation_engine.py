"""
Explanation Engine for Drug Discovery Pipeline.

Generates human-readable explanations for why compounds were
accepted or rejected, with supporting visualizations.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import json

from utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class AcceptanceExplanation:
    """Explanation for why a compound was accepted."""
    compound_id: str
    compound_name: str
    smiles: str
    decision: str = "ACCEPTED"
    
    # Scores
    binding_score: float = 0.0
    confidence: float = 0.0
    percentile: float = 0.0
    
    # Chemistry checks passed
    chemistry_checks: List[Dict] = None
    
    # Binding analysis
    key_binding_features: List[str] = None
    pharmacophore_matches: List[Dict] = None
    
    # Atom contributions (for heatmap)
    atom_contributions: List[float] = None
    highlight_atoms: List[int] = None
    
    def __post_init__(self):
        if self.chemistry_checks is None:
            self.chemistry_checks = []
        if self.key_binding_features is None:
            self.key_binding_features = []
        if self.pharmacophore_matches is None:
            self.pharmacophore_matches = []
        if self.atom_contributions is None:
            self.atom_contributions = []
        if self.highlight_atoms is None:
            self.highlight_atoms = []
    
    def to_dict(self) -> Dict:
        return {
            "compound_id": self.compound_id,
            "compound_name": self.compound_name,
            "smiles": self.smiles,
            "decision": self.decision,
            "binding_score": self.binding_score,
            "confidence": self.confidence,
            "percentile": self.percentile,
            "chemistry_checks": self.chemistry_checks,
            "key_binding_features": self.key_binding_features,
            "pharmacophore_matches": self.pharmacophore_matches,
            "atom_contributions": self.atom_contributions,
            "highlight_atoms": self.highlight_atoms,
        }


@dataclass
class RejectionExplanation:
    """Explanation for why a compound was rejected."""
    compound_id: str
    compound_name: str
    smiles: str
    decision: str = "REJECTED"
    
    # Primary rejection reason
    rejection_reason: str = ""
    rejection_category: str = ""  # "chemistry_filter", "adme_tox", "pains", etc.
    
    # Failed checks
    failed_filters: List[Dict] = None
    
    # Problematic atoms/features
    problematic_atoms: List[int] = None
    problematic_features: List[str] = None
    
    # Remediation hints
    remediation_hints: List[str] = None
    
    def __post_init__(self):
        if self.failed_filters is None:
            self.failed_filters = []
        if self.problematic_atoms is None:
            self.problematic_atoms = []
        if self.problematic_features is None:
            self.problematic_features = []
        if self.remediation_hints is None:
            self.remediation_hints = []
    
    def to_dict(self) -> Dict:
        return {
            "compound_id": self.compound_id,
            "compound_name": self.compound_name,
            "smiles": self.smiles,
            "decision": self.decision,
            "rejection_reason": self.rejection_reason,
            "rejection_category": self.rejection_category,
            "failed_filters": self.failed_filters,
            "problematic_atoms": self.problematic_atoms,
            "problematic_features": self.problematic_features,
            "remediation_hints": self.remediation_hints,
        }


class ExplanationEngine:
    """
    Generate comprehensive explanations for compound decisions.
    
    This engine analyzes compounds and produces human-readable
    explanations with supporting data for visualizations.
    """
    
    def __init__(self):
        """Initialize the explanation engine."""
        self.chemistry_filter_names = {
            "mw": "Molecular Weight",
            "logp": "LogP (Lipophilicity)",
            "hbd": "H-Bond Donors",
            "hba": "H-Bond Acceptors",
            "tpsa": "Polar Surface Area",
            "rotatable_bonds": "Rotatable Bonds",
            "pains": "PAINS Filter",
            "formal_charge": "Formal Charge",
        }
    
    def explain_accepted(self, compound: Dict) -> AcceptanceExplanation:
        """
        Generate explanation for an accepted compound.
        
        Args:
            compound: Compound dictionary with scores and properties
        
        Returns:
            AcceptanceExplanation with detailed reasoning
        """
        explanation = AcceptanceExplanation(
            compound_id=compound.get("compound_id", ""),
            compound_name=compound.get("compound_name", "Unknown"),
            smiles=compound.get("smiles", ""),
            binding_score=compound.get("raw_score", 0.0),
            confidence=1.0 - compound.get("uncertainty", 0.0),
            percentile=compound.get("percentile", 0.0),
        )
        
        # Add chemistry checks that passed
        explanation.chemistry_checks = self._get_passed_chemistry_checks(compound)
        
        # Add key binding features
        explanation.key_binding_features = self._identify_binding_features(compound)
        
        # Add pharmacophore matches
        explanation.pharmacophore_matches = self._get_pharmacophore_matches(compound)
        
        # Calculate atom contributions (if possible)
        if compound.get("smiles"):
            try:
                contributions = self._calculate_atom_contributions(compound["smiles"])
                explanation.atom_contributions = contributions["scores"]
                explanation.highlight_atoms = contributions["highlight"]
            except Exception as e:
                logger.debug(f"Could not calculate atom contributions: {e}")
        
        return explanation
    
    def explain_rejected(self, compound: Dict) -> RejectionExplanation:
        """
        Generate explanation for a rejected compound.
        
        Args:
            compound: Compound dictionary with rejection info
        
        Returns:
            RejectionExplanation with detailed reasoning
        """
        rejection_reason = compound.get("rejection_reason", "Unknown reason")
        
        explanation = RejectionExplanation(
            compound_id=compound.get("compound_id", ""),
            compound_name=compound.get("compound_name", "Unknown"),
            smiles=compound.get("smiles", ""),
            rejection_reason=rejection_reason,
            rejection_category=self._categorize_rejection(rejection_reason),
        )
        
        # Parse failed filters from rejection reason
        explanation.failed_filters = self._parse_failed_filters(rejection_reason, compound)
        
        # Identify problematic structural features
        if compound.get("smiles"):
            try:
                problems = self._identify_problematic_features(
                    compound["smiles"], 
                    rejection_reason
                )
                explanation.problematic_atoms = problems["atoms"]
                explanation.problematic_features = problems["features"]
            except Exception as e:
                logger.debug(f"Could not identify problematic features: {e}")
        
        # Generate remediation hints
        explanation.remediation_hints = self._generate_remediation_hints(
            rejection_reason, 
            compound
        )
        
        return explanation
    
    def _get_passed_chemistry_checks(self, compound: Dict) -> List[Dict]:
        """Get list of chemistry checks that passed."""
        checks = []
        
        # Molecular weight
        mw = compound.get("mw") or compound.get("molecular_weight")
        if mw:
            checks.append({
                "name": "Molecular Weight",
                "value": f"{mw:.1f} Da",
                "status": "PASS",
                "range": "120-700 Da"
            })
        
        # LogP
        logp = compound.get("logp")
        if logp is not None:
            checks.append({
                "name": "LogP",
                "value": f"{logp:.2f}",
                "status": "PASS",
                "range": "-1 to 8"
            })
        
        # TPSA
        tpsa = compound.get("tpsa")
        if tpsa is not None:
            checks.append({
                "name": "TPSA",
                "value": f"{tpsa:.1f} Å²",
                "status": "PASS",
                "range": "< 140 Å²"
            })
        
        # Add more checks based on available data
        if not compound.get("pains_alert"):
            checks.append({
                "name": "PAINS Filter",
                "value": "No alerts",
                "status": "PASS",
                "range": "No interference patterns"
            })
        
        return checks
    
    def _identify_binding_features(self, compound: Dict) -> List[str]:
        """Identify key features contributing to binding."""
        features = []
        
        # Based on score percentile
        percentile = compound.get("percentile", 0)
        if percentile >= 90:
            features.append("Top 10% binding affinity prediction")
        elif percentile >= 75:
            features.append("Strong binding affinity (top 25%)")
        elif percentile >= 50:
            features.append("Good binding affinity (top 50%)")
        
        # Based on confidence
        uncertainty = compound.get("uncertainty", 1.0)
        if uncertainty < 0.2:
            features.append("High prediction confidence")
        elif uncertainty < 0.4:
            features.append("Moderate prediction confidence")
        
        # Known drug status
        if compound.get("is_known_drug"):
            drug_name = compound.get("known_drug_name", "")
            features.append(f"Approved drug: {drug_name}")
        
        return features
    
    def _get_pharmacophore_matches(self, compound: Dict) -> List[Dict]:
        """Get pharmacophore feature matches."""
        matches = []
        
        try:
            from rdkit import Chem
            from rdkit.Chem import Descriptors, rdMolDescriptors
            
            smiles = compound.get("smiles")
            if not smiles:
                return matches
            
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return matches
            
            # H-bond donors
            hbd = rdMolDescriptors.CalcNumHBD(mol)
            if hbd > 0:
                matches.append({
                    "feature": "H-Bond Donors",
                    "count": hbd,
                    "importance": "high" if hbd <= 5 else "moderate"
                })
            
            # H-bond acceptors
            hba = rdMolDescriptors.CalcNumHBA(mol)
            if hba > 0:
                matches.append({
                    "feature": "H-Bond Acceptors",
                    "count": hba,
                    "importance": "high" if hba <= 10 else "moderate"
                })
            
            # Aromatic rings
            num_aromatic = rdMolDescriptors.CalcNumAromaticRings(mol)
            if num_aromatic > 0:
                matches.append({
                    "feature": "Aromatic Rings",
                    "count": num_aromatic,
                    "importance": "high"
                })
            
        except ImportError:
            logger.debug("RDKit not available for pharmacophore analysis")
        except Exception as e:
            logger.debug(f"Pharmacophore analysis failed: {e}")
        
        return matches
    
    def _calculate_atom_contributions(self, smiles: str) -> Dict:
        """
        Calculate per-atom contributions to binding.
        
        Uses a simplified heuristic based on atom properties.
        For more accurate results, use gradient-based methods with the model.
        """
        from rdkit import Chem
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {"scores": [], "highlight": []}
        
        scores = []
        highlight = []
        
        for atom in mol.GetAtoms():
            score = 0.0
            
            # Heteroatoms often contribute more to binding
            if atom.GetAtomicNum() == 7:  # Nitrogen
                score = 0.3
            elif atom.GetAtomicNum() == 8:  # Oxygen
                score = 0.25
            elif atom.GetAtomicNum() == 16:  # Sulfur
                score = 0.2
            elif atom.GetAtomicNum() == 9:  # Fluorine
                score = 0.15
            elif atom.GetAtomicNum() == 17:  # Chlorine
                score = 0.1
            elif atom.GetAtomicNum() == 6:  # Carbon
                if atom.GetIsAromatic():
                    score = 0.1
                else:
                    score = 0.0
            
            # Boost for H-bond donors/acceptors
            if atom.GetTotalNumHs() > 0 and atom.GetAtomicNum() in [7, 8]:
                score += 0.2
            
            scores.append(round(score, 2))
            
            if score >= 0.2:
                highlight.append(atom.GetIdx())
        
        return {"scores": scores, "highlight": highlight}
    
    def _categorize_rejection(self, reason: str) -> str:
        """Categorize the rejection reason."""
        reason_lower = reason.lower()
        
        if "pains" in reason_lower:
            return "pains_filter"
        elif "mw" in reason_lower or "molecular weight" in reason_lower:
            return "molecular_weight"
        elif "logp" in reason_lower:
            return "lipophilicity"
        elif "smiles" in reason_lower or "invalid" in reason_lower:
            return "invalid_structure"
        elif "adme" in reason_lower or "tox" in reason_lower:
            return "adme_tox"
        elif "charge" in reason_lower:
            return "formal_charge"
        else:
            return "chemistry_filter"
    
    def _parse_failed_filters(self, reason: str, compound: Dict) -> List[Dict]:
        """Parse rejection reason into structured filter failures."""
        filters = []
        
        reason_lower = reason.lower()
        
        if "mw" in reason_lower or "molecular weight" in reason_lower:
            mw = compound.get("mw") or compound.get("molecular_weight", "N/A")
            filters.append({
                "filter": "Molecular Weight",
                "actual": f"{mw}" if isinstance(mw, (int, float)) else mw,
                "limit": "120-700 Da",
                "severity": "high"
            })
        
        if "logp" in reason_lower:
            logp = compound.get("logp", "N/A")
            filters.append({
                "filter": "LogP",
                "actual": f"{logp}" if isinstance(logp, (int, float)) else logp,
                "limit": "-1 to 8",
                "severity": "high"
            })
        
        if "pains" in reason_lower:
            filters.append({
                "filter": "PAINS Alert",
                "actual": "Pattern detected",
                "limit": "No PAINS patterns",
                "severity": "critical"
            })
        
        if "charge" in reason_lower:
            charge = compound.get("formal_charge", "N/A")
            filters.append({
                "filter": "Formal Charge",
                "actual": str(charge),
                "limit": "±2",
                "severity": "moderate"
            })
        
        # If no specific filter parsed, add generic
        if not filters:
            filters.append({
                "filter": "Chemistry Filter",
                "actual": "Failed",
                "limit": reason,
                "severity": "unknown"
            })
        
        return filters
    
    def _identify_problematic_features(self, smiles: str, reason: str) -> Dict:
        """Identify atoms/features causing the rejection."""
        from rdkit import Chem
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {"atoms": [], "features": []}
        
        atoms = []
        features = []
        reason_lower = reason.lower()
        
        # For LogP issues, highlight hydrophobic chains
        if "logp" in reason_lower:
            features.append("Excessive hydrophobic regions")
            for atom in mol.GetAtoms():
                if atom.GetAtomicNum() == 6 and not atom.GetIsAromatic():
                    # Count connected carbons (lipophilic chain)
                    carbon_neighbors = sum(
                        1 for n in atom.GetNeighbors() 
                        if n.GetAtomicNum() == 6
                    )
                    if carbon_neighbors >= 2:
                        atoms.append(atom.GetIdx())
        
        # For MW issues, the whole molecule is the problem
        if "mw" in reason_lower or "molecular weight" in reason_lower:
            features.append("Molecule too large/small")
        
        # For PAINS, try to identify the pattern
        if "pains" in reason_lower:
            features.append("Pan-assay interference pattern detected")
            # Could use PAINS SMARTS patterns to identify specific atoms
        
        return {"atoms": atoms[:10], "features": features}  # Limit atoms
    
    def _generate_remediation_hints(self, reason: str, compound: Dict) -> List[str]:
        """Generate suggestions for fixing the compound."""
        hints = []
        reason_lower = reason.lower()
        
        if "logp" in reason_lower and "high" in reason_lower:
            hints.append("Add polar groups (OH, NH2) to reduce lipophilicity")
            hints.append("Replace alkyl chains with ether linkages")
            hints.append("Consider adding a carboxylic acid or sulfonamide")
        
        if "logp" in reason_lower and "low" in reason_lower:
            hints.append("Add hydrophobic groups to improve membrane permeability")
        
        if "mw" in reason_lower and ("high" in reason_lower or ">" in reason_lower):
            hints.append("Consider fragment-based approach")
            hints.append("Remove non-essential substituents")
            hints.append("Replace heavy atoms with lighter bioisosteres")
        
        if "pains" in reason_lower:
            hints.append("Avoid known PAINS substructures")
            hints.append("Validate activity with orthogonal assays")
            hints.append("Consider alternative core scaffolds")
        
        if not hints:
            hints.append("Review structure for drug-likeness improvements")
        
        return hints


def create_explanation_engine() -> ExplanationEngine:
    """Factory function to create ExplanationEngine."""
    return ExplanationEngine()

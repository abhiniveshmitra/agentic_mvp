"""
Retrosynthesis Planning Module.

Provides synthesis route prediction for drug candidates using RDKit's
BRICS decomposition and reaction templates.

Features:
- BRICS decomposition for fragment analysis
- Building block identification
- Synthetic accessibility scoring
- Reaction step estimation
"""

from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field

from utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class BuildingBlock:
    """Represents a purchasable building block."""
    smiles: str
    name: Optional[str] = None
    availability: str = "likely"  # 'purchasable', 'likely', 'custom'
    estimated_cost: str = "low"  # 'low', 'medium', 'high'


@dataclass
class ReactionStep:
    """Represents a single reaction step in synthesis."""
    reactants: List[str]
    product: str
    reaction_type: str
    conditions: Optional[str] = None
    yield_estimate: str = "moderate"  # 'high', 'moderate', 'low'


@dataclass
class SynthesisRoute:
    """Represents a complete synthesis route."""
    target_smiles: str
    building_blocks: List[BuildingBlock]
    steps: List[ReactionStep]
    total_steps: int
    synthetic_accessibility: float  # 1 (easy) to 10 (hard)
    estimated_feasibility: str  # 'high', 'moderate', 'low', 'challenging'
    notes: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "target_smiles": self.target_smiles,
            "building_blocks": [{"smiles": bb.smiles, "availability": bb.availability} 
                               for bb in self.building_blocks],
            "total_steps": self.total_steps,
            "synthetic_accessibility": self.synthetic_accessibility,
            "estimated_feasibility": self.estimated_feasibility,
            "notes": self.notes,
        }


class RetrosynthesisPlanner:
    """
    Plan synthesis routes for drug molecules.
    
    Uses RDKit's BRICS decomposition to identify:
    1. Potential disconnection sites
    2. Building block fragments
    3. Synthetic accessibility
    """
    
    # Common reaction types based on BRICS rules
    BRICS_REACTIONS = {
        "L1": "Amide bond formation",
        "L2": "Amine alkylation",
        "L3": "Lactam formation",
        "L4": "Aromatic N-coupling",
        "L5": "Aromatic C-coupling",
        "L6": "Aromatic substitution",
        "L7": "Ether formation",
        "L8": "Carbonyl addition",
    }
    
    def __init__(self):
        """Initialize retrosynthesis planner."""
        self._rdkit_available = self._check_rdkit()
    
    def _check_rdkit(self) -> bool:
        """Check if RDKit is available."""
        try:
            from rdkit import Chem
            from rdkit.Chem import BRICS
            return True
        except ImportError:
            logger.warning("RDKit not available - retrosynthesis disabled")
            return False
    
    def analyze(self, smiles: str) -> Optional[SynthesisRoute]:
        """
        Analyze a molecule and plan its synthesis route.
        
        Args:
            smiles: Target molecule SMILES
            
        Returns:
            SynthesisRoute object or None if analysis fails
        """
        if not self._rdkit_available:
            return None
            
        try:
            from rdkit import Chem
            from rdkit.Chem import BRICS, Descriptors
            from rdkit.Chem.Scaffolds import MurckoScaffold
            
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            
            # Calculate synthetic accessibility score (if available)
            try:
                from rdkit.Chem import RDConfig
                import os
                import sys
                sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
                from sascorer import calculateScore
                sa_score = calculateScore(mol)
            except Exception:
                # Estimate based on complexity
                sa_score = self._estimate_sa_score(mol)
            
            # Get BRICS fragments (building blocks)
            fragments = list(BRICS.BRICSDecompose(mol))
            
            # Convert fragments to building blocks
            building_blocks = []
            for frag in fragments[:10]:  # Limit to top 10
                # Clean up fragment SMILES (remove BRICS labels)
                clean_frag = self._clean_fragment(frag)
                if clean_frag:
                    availability = self._estimate_availability(clean_frag)
                    building_blocks.append(BuildingBlock(
                        smiles=clean_frag,
                        availability=availability,
                    ))
            
            # Estimate steps based on fragments
            n_fragments = len(building_blocks)
            total_steps = max(1, n_fragments - 1) if n_fragments > 1 else 1
            
            # Determine feasibility
            if sa_score <= 3:
                feasibility = "high"
            elif sa_score <= 5:
                feasibility = "moderate"
            elif sa_score <= 7:
                feasibility = "low"
            else:
                feasibility = "challenging"
            
            # Generate reaction steps
            steps = self._generate_reaction_steps(building_blocks, smiles)
            
            # Generate notes
            notes = self._generate_notes(mol, sa_score, building_blocks)
            
            return SynthesisRoute(
                target_smiles=smiles,
                building_blocks=building_blocks,
                steps=steps,
                total_steps=total_steps,
                synthetic_accessibility=sa_score,
                estimated_feasibility=feasibility,
                notes=notes,
            )
            
        except Exception as e:
            logger.error(f"Retrosynthesis analysis failed: {e}")
            return None
    
    def _clean_fragment(self, fragment: str) -> Optional[str]:
        """Remove BRICS labels from fragment SMILES."""
        try:
            from rdkit import Chem
            
            # Remove dummy atoms [n*] patterns
            import re
            cleaned = re.sub(r'\[\d+\*\]', '', fragment)
            cleaned = re.sub(r'\*', '', cleaned)
            
            # Validate
            mol = Chem.MolFromSmiles(cleaned)
            if mol and mol.GetNumAtoms() >= 2:
                return Chem.MolToSmiles(mol)
            return None
            
        except Exception:
            return None
    
    def _estimate_availability(self, smiles: str) -> str:
        """Estimate commercial availability of a fragment."""
        try:
            from rdkit import Chem
            from rdkit.Chem import Descriptors
            
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return "custom"
            
            mw = Descriptors.MolWt(mol)
            n_atoms = mol.GetNumAtoms()
            n_rings = Descriptors.RingCount(mol)
            
            # Simple heuristics for availability
            if mw < 150 and n_rings <= 1:
                return "purchasable"
            elif mw < 300 and n_rings <= 2:
                return "likely"
            else:
                return "custom"
                
        except Exception:
            return "custom"
    
    def _estimate_sa_score(self, mol) -> float:
        """Estimate synthetic accessibility without sascorer."""
        try:
            from rdkit.Chem import Descriptors, Lipinski
            
            # Factors affecting synthesis difficulty
            mw = Descriptors.MolWt(mol)
            n_rings = Descriptors.RingCount(mol)
            n_stereo = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True)) if hasattr(Chem, 'FindMolChiralCenters') else 0
            n_rot = Descriptors.NumRotatableBonds(mol)
            n_hbd = Lipinski.NumHDonors(mol)
            n_hba = Lipinski.NumHAcceptors(mol)
            
            # Calculate score (1-10 scale)
            score = 2.0  # Base difficulty
            score += mw / 200  # Weight penalty
            score += n_rings * 0.3  # Ring penalty
            score += n_stereo * 0.8  # Stereocenter penalty
            score += max(0, (n_rot - 5)) * 0.1  # Flexibility penalty
            
            return min(10.0, max(1.0, score))
            
        except Exception:
            return 5.0  # Default moderate difficulty
    
    def _generate_reaction_steps(
        self, 
        building_blocks: List[BuildingBlock], 
        target: str
    ) -> List[ReactionStep]:
        """Generate synthetic steps from building blocks."""
        steps = []
        
        if len(building_blocks) < 2:
            return steps
        
        # Simple linear synthesis model
        current_intermediate = building_blocks[0].smiles
        
        for i, bb in enumerate(building_blocks[1:], 1):
            # Determine reaction type based on fragment analysis
            reaction_type = self._guess_reaction_type(current_intermediate, bb.smiles)
            
            steps.append(ReactionStep(
                reactants=[current_intermediate, bb.smiles],
                product=f"Intermediate_{i}" if i < len(building_blocks) - 1 else target,
                reaction_type=reaction_type,
                conditions=self._get_conditions(reaction_type),
                yield_estimate="moderate",
            ))
            
            current_intermediate = f"Intermediate_{i}"
        
        return steps
    
    def _guess_reaction_type(self, frag1: str, frag2: str) -> str:
        """Guess the reaction type needed to join fragments."""
        # Simple heuristics based on functional groups
        if "N" in frag1 or "N" in frag2:
            if "C(=O)" in frag1 or "C(=O)" in frag2:
                return "Amide coupling"
            return "N-alkylation"
        elif "O" in frag1 or "O" in frag2:
            return "Ether formation"
        elif "c" in frag1 and "c" in frag2:
            return "C-C coupling (Suzuki/Buchwald)"
        else:
            return "Alkylation"
    
    def _get_conditions(self, reaction_type: str) -> str:
        """Get typical reaction conditions."""
        conditions = {
            "Amide coupling": "EDC/HOBt, DCM, rt",
            "N-alkylation": "K2CO3, DMF, 60°C",
            "Ether formation": "K2CO3, acetone, reflux",
            "C-C coupling (Suzuki/Buchwald)": "Pd(PPh3)4, K2CO3, dioxane, 80°C",
            "Alkylation": "NaH, THF, 0°C to rt",
        }
        return conditions.get(reaction_type, "Standard conditions")
    
    def _generate_notes(
        self, 
        mol, 
        sa_score: float, 
        building_blocks: List[BuildingBlock]
    ) -> List[str]:
        """Generate synthesis notes and warnings."""
        notes = []
        
        try:
            from rdkit.Chem import Descriptors
            
            # Check for challenging features
            n_stereo = len([a for a in mol.GetAtoms() if a.GetChiralTag() != 0])
            if n_stereo > 0:
                notes.append(f"⚠️ Contains {n_stereo} stereocenter(s) - may require asymmetric synthesis")
            
            n_rings = Descriptors.RingCount(mol)
            if n_rings >= 4:
                notes.append(f"⚠️ Complex ring system ({n_rings} rings) - may require multi-step ring construction")
            
            # Check building block availability
            custom_count = sum(1 for bb in building_blocks if bb.availability == "custom")
            if custom_count > 0:
                notes.append(f"⚠️ {custom_count} building block(s) may require custom synthesis")
            
            purchasable_count = sum(1 for bb in building_blocks if bb.availability == "purchasable")
            if purchasable_count == len(building_blocks):
                notes.append("✅ All building blocks likely commercially available")
            
            # SA score interpretation
            if sa_score <= 3:
                notes.append("✅ High synthetic accessibility - straightforward synthesis expected")
            elif sa_score <= 5:
                notes.append("ℹ️ Moderate synthetic accessibility - standard medicinal chemistry")
            elif sa_score <= 7:
                notes.append("⚠️ Low synthetic accessibility - may require experienced chemists")
            else:
                notes.append("⛔ Challenging synthesis - consider structural simplification")
                
        except Exception:
            pass
        
        return notes
    
    def get_building_blocks_summary(self, smiles: str) -> Dict[str, Any]:
        """
        Get a quick summary of building blocks needed.
        
        Args:
            smiles: Target molecule SMILES
            
        Returns:
            Dictionary with building block summary
        """
        route = self.analyze(smiles)
        if route is None:
            return {"error": "Analysis failed"}
        
        return {
            "target": smiles,
            "n_building_blocks": len(route.building_blocks),
            "n_steps": route.total_steps,
            "sa_score": route.synthetic_accessibility,
            "feasibility": route.estimated_feasibility,
            "building_blocks": [
                {"smiles": bb.smiles, "availability": bb.availability}
                for bb in route.building_blocks
            ],
        }


def plan_synthesis(smiles: str) -> Optional[Dict]:
    """
    Convenience function for synthesis planning.
    
    Args:
        smiles: Target molecule SMILES
        
    Returns:
        Synthesis route dictionary
    """
    planner = RetrosynthesisPlanner()
    route = planner.analyze(smiles)
    return route.to_dict() if route else None


# Test function
if __name__ == "__main__":
    # Test with ibuprofen
    test_smiles = "CC(C)Cc1ccc(cc1)C(C)C(=O)O"  # Ibuprofen
    
    planner = RetrosynthesisPlanner()
    
    print(f"Analyzing synthesis for: {test_smiles}")
    
    route = planner.analyze(test_smiles)
    if route:
        print(f"\n✅ Synthesis Route:")
        print(f"   SA Score: {route.synthetic_accessibility:.1f}/10")
        print(f"   Feasibility: {route.estimated_feasibility}")
        print(f"   Steps: {route.total_steps}")
        print(f"   Building Blocks: {len(route.building_blocks)}")
        for i, bb in enumerate(route.building_blocks, 1):
            print(f"      {i}. {bb.smiles} ({bb.availability})")
        print("\n   Notes:")
        for note in route.notes:
            print(f"      {note}")
    else:
        print("❌ Analysis failed")

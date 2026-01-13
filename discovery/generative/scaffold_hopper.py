"""
Scaffold Hopping Module for Drug Discovery.

Generates novel molecular analogs by modifying scaffolds of hit compounds.
Uses RDKit's MCS (Maximum Common Substructure) and fragment-based approaches.

Features:
- Scaffold extraction using Murcko decomposition
- R-group enumeration for lead optimization
- Fragment-based analog generation
- Similarity-guided filtering
"""

from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
import random

from utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class GeneratedAnalog:
    """Represents a generated molecular analog."""
    smiles: str
    parent_smiles: str
    scaffold: str
    modification_type: str
    similarity_to_parent: float
    valid: bool = True
    properties: Dict[str, Any] = None
    
    def to_dict(self) -> Dict:
        return {
            "smiles": self.smiles,
            "parent_smiles": self.parent_smiles,
            "scaffold": self.scaffold,
            "modification_type": self.modification_type,
            "similarity_to_parent": self.similarity_to_parent,
            "valid": self.valid,
            "properties": self.properties or {},
        }


class ScaffoldHopper:
    """
    Generate novel molecular analogs through scaffold hopping techniques.
    
    Uses RDKit-based methods for:
    1. Murcko scaffold extraction
    2. R-group decoration
    3. Fragment replacement
    4. Bioisosteric substitution
    """
    
    # Common bioisosteric replacements
    BIOISOSTERES = {
        "C(=O)O": ["C(=O)N", "S(=O)(=O)N", "P(=O)(O)O"],  # Carboxylic acid
        "c1ccccc1": ["c1ccncc1", "c1ccc2ccccc2c1", "c1ccoc1"],  # Benzene
        "F": ["Cl", "CF3", "OCF3"],  # Fluorine
        "O": ["S", "NH"],  # Oxygen
        "N": ["O", "S"],  # Nitrogen
        "Cl": ["F", "Br", "CF3"],  # Chlorine
    }
    
    # Common R-groups for decoration
    R_GROUPS = [
        "C",      # Methyl
        "CC",     # Ethyl
        "C(C)C",  # Isopropyl
        "C1CC1",  # Cyclopropyl
        "OC",     # Methoxy
        "F",      # Fluoro
        "Cl",     # Chloro
        "N",      # Amino
        "NC",     # Methylamino
        "O",      # Hydroxy
        "C#N",    # Cyano
        "C(F)(F)F",  # Trifluoromethyl
    ]
    
    def __init__(self, similarity_threshold: float = 0.4):
        """
        Initialize scaffold hopper.
        
        Args:
            similarity_threshold: Minimum Tanimoto similarity to parent (0.0-1.0)
        """
        self.similarity_threshold = similarity_threshold
        self._rdkit_available = self._check_rdkit()
    
    def _check_rdkit(self) -> bool:
        """Check if RDKit is available."""
        try:
            from rdkit import Chem
            return True
        except ImportError:
            logger.warning("RDKit not available - scaffold hopping disabled")
            return False
    
    def extract_scaffold(self, smiles: str) -> Optional[str]:
        """
        Extract Murcko scaffold from a molecule.
        
        Args:
            smiles: SMILES string
            
        Returns:
            Scaffold SMILES or None if failed
        """
        if not self._rdkit_available:
            return None
            
        try:
            from rdkit import Chem
            from rdkit.Chem.Scaffolds import MurckoScaffold
            
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            return Chem.MolToSmiles(scaffold)
            
        except Exception as e:
            logger.error(f"Failed to extract scaffold: {e}")
            return None
    
    def get_generic_scaffold(self, smiles: str) -> Optional[str]:
        """
        Get generic (framework) scaffold with all atoms as carbon.
        
        Args:
            smiles: SMILES string
            
        Returns:
            Generic scaffold SMILES
        """
        if not self._rdkit_available:
            return None
            
        try:
            from rdkit import Chem
            from rdkit.Chem.Scaffolds import MurckoScaffold
            
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            
            framework = MurckoScaffold.MakeScaffoldGeneric(
                MurckoScaffold.GetScaffoldForMol(mol)
            )
            return Chem.MolToSmiles(framework)
            
        except Exception as e:
            logger.error(f"Failed to get generic scaffold: {e}")
            return None
    
    def generate_analogs(
        self,
        smiles: str,
        n_analogs: int = 10,
        methods: Optional[List[str]] = None,
    ) -> List[GeneratedAnalog]:
        """
        Generate molecular analogs using multiple methods.
        
        Args:
            smiles: Parent SMILES string
            n_analogs: Number of analogs to generate
            methods: List of methods to use ('decoration', 'bioisostere', 'fragment')
                    If None, uses all methods
        
        Returns:
            List of GeneratedAnalog objects
        """
        if not self._rdkit_available:
            return []
        
        if methods is None:
            methods = ["decoration", "bioisostere", "fragment"]
        
        analogs = []
        
        # Distribute analogs across methods
        per_method = max(1, n_analogs // len(methods))
        
        if "decoration" in methods:
            analogs.extend(self._generate_decorations(smiles, per_method))
        
        if "bioisostere" in methods:
            analogs.extend(self._generate_bioisosteres(smiles, per_method))
        
        if "fragment" in methods:
            analogs.extend(self._generate_fragment_variants(smiles, per_method))
        
        # Filter by similarity and validity
        valid_analogs = []
        for analog in analogs:
            if analog.valid and analog.similarity_to_parent >= self.similarity_threshold:
                valid_analogs.append(analog)
        
        # Deduplicate by SMILES
        seen = set()
        unique_analogs = []
        for analog in valid_analogs:
            if analog.smiles not in seen and analog.smiles != smiles:
                seen.add(analog.smiles)
                unique_analogs.append(analog)
        
        return unique_analogs[:n_analogs]
    
    def _generate_decorations(self, smiles: str, n: int) -> List[GeneratedAnalog]:
        """Generate analogs by R-group decoration."""
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem
            
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return []
            
            scaffold = self.extract_scaffold(smiles)
            analogs = []
            
            # Find attachment points (carbons with implicit hydrogens)
            for atom in mol.GetAtoms():
                if atom.GetSymbol() == "C" and atom.GetTotalNumHs() > 0:
                    for r_group in random.sample(self.R_GROUPS, min(len(self.R_GROUPS), 3)):
                        try:
                            # Create modified molecule
                            new_mol = self._attach_r_group(mol, atom.GetIdx(), r_group)
                            if new_mol:
                                new_smiles = Chem.MolToSmiles(new_mol)
                                if self._is_valid_molecule(new_smiles):
                                    similarity = self._calculate_similarity(smiles, new_smiles)
                                    analogs.append(GeneratedAnalog(
                                        smiles=new_smiles,
                                        parent_smiles=smiles,
                                        scaffold=scaffold or "",
                                        modification_type="R-group decoration",
                                        similarity_to_parent=similarity,
                                        valid=True,
                                    ))
                        except Exception:
                            continue
                        
                        if len(analogs) >= n:
                            return analogs
            
            return analogs
            
        except Exception as e:
            logger.error(f"Decoration generation failed: {e}")
            return []
    
    def _generate_bioisosteres(self, smiles: str, n: int) -> List[GeneratedAnalog]:
        """Generate analogs by bioisosteric replacement."""
        try:
            from rdkit import Chem
            
            scaffold = self.extract_scaffold(smiles)
            analogs = []
            
            for pattern, replacements in self.BIOISOSTERES.items():
                for replacement in replacements:
                    try:
                        new_smiles = smiles.replace(pattern, replacement)
                        if new_smiles != smiles and self._is_valid_molecule(new_smiles):
                            similarity = self._calculate_similarity(smiles, new_smiles)
                            analogs.append(GeneratedAnalog(
                                smiles=new_smiles,
                                parent_smiles=smiles,
                                scaffold=scaffold or "",
                                modification_type=f"Bioisostere: {pattern}→{replacement}",
                                similarity_to_parent=similarity,
                                valid=True,
                            ))
                    except Exception:
                        continue
                    
                    if len(analogs) >= n:
                        return analogs
            
            return analogs
            
        except Exception as e:
            logger.error(f"Bioisostere generation failed: {e}")
            return []
    
    def _generate_fragment_variants(self, smiles: str, n: int) -> List[GeneratedAnalog]:
        """Generate analogs by fragment shuffling."""
        try:
            from rdkit import Chem
            from rdkit.Chem import BRICS
            
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return []
            
            scaffold = self.extract_scaffold(smiles)
            analogs = []
            
            # Fragment the molecule
            try:
                frags = list(BRICS.BRICSDecompose(mol))
                if len(frags) < 2:
                    return []
                
                # Generate combinations by replacing fragments
                for i, frag in enumerate(frags[:3]):
                    for replacement in self.R_GROUPS[:5]:
                        try:
                            # Simple fragment replacement
                            new_smiles = smiles.replace(frag[:10], replacement) if len(frag) > 10 else smiles
                            if new_smiles != smiles and self._is_valid_molecule(new_smiles):
                                similarity = self._calculate_similarity(smiles, new_smiles)
                                analogs.append(GeneratedAnalog(
                                    smiles=new_smiles,
                                    parent_smiles=smiles,
                                    scaffold=scaffold or "",
                                    modification_type="Fragment replacement",
                                    similarity_to_parent=similarity,
                                    valid=True,
                                ))
                        except Exception:
                            continue
                        
                        if len(analogs) >= n:
                            return analogs
                            
            except Exception:
                pass
            
            return analogs
            
        except Exception as e:
            logger.error(f"Fragment variant generation failed: {e}")
            return []
    
    def _attach_r_group(self, mol, atom_idx: int, r_group: str):
        """Attach an R-group to a specific atom."""
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem
            
            # Create editable molecule
            rw_mol = Chem.RWMol(mol)
            
            # Parse R-group
            r_mol = Chem.MolFromSmiles(r_group)
            if r_mol is None:
                return None
            
            # Combine molecules
            combined = Chem.CombineMols(rw_mol, r_mol)
            combined = Chem.RWMol(combined)
            
            # Find attachment point in R-group
            r_attach = mol.GetNumAtoms()  # First atom of R-group
            
            # Add bond
            combined.AddBond(atom_idx, r_attach, Chem.BondType.SINGLE)
            
            # Sanitize
            Chem.SanitizeMol(combined)
            
            return combined.GetMol()
            
        except Exception:
            return None
    
    def _is_valid_molecule(self, smiles: str) -> bool:
        """Check if SMILES represents a valid molecule."""
        try:
            from rdkit import Chem
            mol = Chem.MolFromSmiles(smiles)
            return mol is not None
        except Exception:
            return False
    
    def _calculate_similarity(self, smiles1: str, smiles2: str) -> float:
        """Calculate Tanimoto similarity between two molecules."""
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem
            from rdkit import DataStructs
            
            mol1 = Chem.MolFromSmiles(smiles1)
            mol2 = Chem.MolFromSmiles(smiles2)
            
            if mol1 is None or mol2 is None:
                return 0.0
            
            fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=2048)
            fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=2048)
            
            return DataStructs.TanimotoSimilarity(fp1, fp2)
            
        except Exception:
            return 0.0
    
    def optimize_properties(
        self,
        smiles: str,
        target_mw: Optional[Tuple[float, float]] = None,
        target_logp: Optional[Tuple[float, float]] = None,
        n_iterations: int = 3,
    ) -> List[GeneratedAnalog]:
        """
        Iteratively optimize molecular properties.
        
        Args:
            smiles: Starting SMILES
            target_mw: Target molecular weight range (min, max)
            target_logp: Target logP range (min, max)
            n_iterations: Number of optimization iterations
            
        Returns:
            List of optimized analogs
        """
        if not self._rdkit_available:
            return []
        
        try:
            from rdkit import Chem
            from rdkit.Chem import Descriptors
            
            current = smiles
            all_analogs = []
            
            for iteration in range(n_iterations):
                # Generate analogs from current best
                analogs = self.generate_analogs(current, n_analogs=10)
                
                # Score analogs based on property targets
                scored = []
                for analog in analogs:
                    mol = Chem.MolFromSmiles(analog.smiles)
                    if mol is None:
                        continue
                    
                    score = 0.0
                    mw = Descriptors.MolWt(mol)
                    logp = Descriptors.MolLogP(mol)
                    
                    # Penalize out-of-range properties
                    if target_mw:
                        if target_mw[0] <= mw <= target_mw[1]:
                            score += 1.0
                        else:
                            score -= abs(mw - sum(target_mw)/2) / 100
                    
                    if target_logp:
                        if target_logp[0] <= logp <= target_logp[1]:
                            score += 1.0
                        else:
                            score -= abs(logp - sum(target_logp)/2)
                    
                    # Add similarity bonus
                    score += analog.similarity_to_parent
                    
                    analog.properties = {
                        "mw": mw,
                        "logp": logp,
                        "optimization_score": score,
                        "iteration": iteration + 1,
                    }
                    scored.append((score, analog))
                
                # Sort by score and keep best
                scored.sort(key=lambda x: x[0], reverse=True)
                
                if scored:
                    all_analogs.extend([a for _, a in scored[:3]])
                    current = scored[0][1].smiles  # Use best for next iteration
            
            return all_analogs
            
        except Exception as e:
            logger.error(f"Property optimization failed: {e}")
            return []


def generate_analogs_for_compound(
    smiles: str,
    n_analogs: int = 10,
    similarity_threshold: float = 0.4,
) -> List[Dict]:
    """
    Convenience function for generating analogs.
    
    Args:
        smiles: Parent molecule SMILES
        n_analogs: Number of analogs to generate
        similarity_threshold: Minimum similarity to parent
        
    Returns:
        List of analog dictionaries
    """
    hopper = ScaffoldHopper(similarity_threshold=similarity_threshold)
    analogs = hopper.generate_analogs(smiles, n_analogs=n_analogs)
    return [a.to_dict() for a in analogs]


# Test function
if __name__ == "__main__":
    # Test with imatinib-like structure
    test_smiles = "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1"
    
    hopper = ScaffoldHopper()
    
    print(f"Testing scaffold hopping for: {test_smiles[:50]}...")
    
    # Extract scaffold
    scaffold = hopper.extract_scaffold(test_smiles)
    print(f"\nMurcko scaffold: {scaffold}")
    
    # Generate analogs
    analogs = hopper.generate_analogs(test_smiles, n_analogs=5)
    print(f"\nGenerated {len(analogs)} analogs:")
    for i, analog in enumerate(analogs, 1):
        print(f"  {i}. {analog.smiles[:50]}...")
        print(f"     Modification: {analog.modification_type}")
        print(f"     Similarity: {analog.similarity_to_parent:.2f}")

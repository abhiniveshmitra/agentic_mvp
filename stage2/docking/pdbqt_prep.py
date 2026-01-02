"""
PDBQT Preparation Module.

Purpose: Convert molecules to PDBQT format for AutoDock Vina.

Uses OpenBabel for:
- Adding hydrogens
- Assigning Gasteiger partial charges
- PDBQT format conversion

Terminology: Uses "Vina score" not "binding energy" to prevent
confusion with thermodynamic ΔG.
"""

from typing import Optional, Tuple
from pathlib import Path
import tempfile
import os

from utils.logging import get_logger

logger = get_logger(__name__)

# Cross-target warning to include in all outputs
CROSS_TARGET_WARNING = (
    "Vina score thresholds are heuristic and target-dependent. "
    "Scores are interpreted qualitatively and should not be compared "
    "across different protein targets."
)


def check_openbabel_available() -> bool:
    """
    Check if OpenBabel is available.
    
    Returns:
        True if OpenBabel Python bindings are available
    """
    try:
        from openbabel import openbabel
        version = openbabel.OBReleaseVersion()
        logger.info(f"OpenBabel available (version {version})")
        return True
    except ImportError:
        logger.warning("OpenBabel not available - install with: pip install openbabel-wheel")
        return False


def prepare_ligand_pdbqt(smiles: str, output_path: Optional[Path] = None) -> Optional[Path]:
    """
    Convert SMILES to PDBQT format for docking.
    
    Uses RDKit for 3D generation (more reliable) and OpenBabel
    for PDBQT conversion with Gasteiger charges.
    
    Args:
        smiles: Input SMILES string
        output_path: Optional path for output file
    
    Returns:
        Path to PDBQT file, or None if conversion failed
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        from openbabel import openbabel, pybel
        
        # Step 1: Generate 3D structure with RDKit
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.error(f"Failed to parse SMILES: {smiles[:50]}...")
            return None
        
        # Add hydrogens
        mol = Chem.AddHs(mol)
        
        # Generate 3D coordinates
        result = AllChem.EmbedMolecule(mol, randomSeed=42)
        if result == -1:
            # Try with random coordinates
            result = AllChem.EmbedMolecule(mol, useRandomCoords=True, randomSeed=42)
            if result == -1:
                logger.error("Failed to generate 3D coordinates")
                return None
        
        # Optimize geometry with MMFF
        try:
            AllChem.MMFFOptimizeMolecule(mol)
        except:
            logger.warning("MMFF optimization failed, using unoptimized geometry")
        
        # Step 2: Write to temporary MOL file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.mol', delete=False) as f:
            mol_path = f.name
            Chem.MolToMolFile(mol, mol_path)
        
        # Step 3: Convert to PDBQT using OpenBabel
        if output_path is None:
            output_path = Path(tempfile.mktemp(suffix='.pdbqt'))
        else:
            output_path = Path(output_path)
        
        # Use pybel for conversion
        ob_mol = next(pybel.readfile("mol", mol_path))
        
        # Add hydrogens if missing and assign Gasteiger charges
        ob_mol.OBMol.AddHydrogens()
        
        # Assign Gasteiger charges
        charge_model = openbabel.OBChargeModel.FindType("Gasteiger")
        if charge_model:
            charge_model.ComputeCharges(ob_mol.OBMol)
        
        # Write PDBQT
        ob_mol.write("pdbqt", str(output_path), overwrite=True)
        
        # Cleanup temp file
        try:
            os.unlink(mol_path)
        except:
            pass
        
        logger.info(f"Ligand PDBQT prepared: {output_path}")
        return output_path
        
    except ImportError as e:
        logger.error(f"Missing dependency for PDBQT preparation: {e}")
        return None
    except Exception as e:
        logger.error(f"PDBQT preparation failed: {e}")
        return None


def prepare_receptor_pdbqt(pdb_path: Path, output_path: Optional[Path] = None) -> Optional[Path]:
    """
    Convert protein PDB to PDBQT format for docking.
    
    Steps:
    1. Remove water molecules
    2. Add hydrogens at pH 7.4
    3. Assign Gasteiger partial charges
    4. Write PDBQT
    
    Args:
        pdb_path: Path to input PDB file
        output_path: Optional path for output file
    
    Returns:
        Path to PDBQT file, or None if conversion failed
    """
    try:
        from openbabel import openbabel, pybel
        
        if not pdb_path.exists():
            logger.error(f"PDB file not found: {pdb_path}")
            return None
        
        # Read PDB
        ob_mol = next(pybel.readfile("pdb", str(pdb_path)))
        
        # Remove water molecules (residue name HOH, WAT)
        atoms_to_delete = []
        for atom in openbabel.OBMolAtomIter(ob_mol.OBMol):
            res = atom.GetResidue()
            if res and res.GetName().strip() in ['HOH', 'WAT', 'H2O']:
                atoms_to_delete.append(atom)
        
        for atom in reversed(atoms_to_delete):
            ob_mol.OBMol.DeleteAtom(atom)
        
        # Add hydrogens (polar only for speed, or all for accuracy)
        ob_mol.OBMol.AddHydrogens(True, True, 7.4)  # pH 7.4
        
        # Assign Gasteiger charges
        charge_model = openbabel.OBChargeModel.FindType("Gasteiger")
        if charge_model:
            charge_model.ComputeCharges(ob_mol.OBMol)
        
        # Prepare output path
        if output_path is None:
            output_path = pdb_path.with_suffix('.pdbqt')
        else:
            output_path = Path(output_path)
        
        # Write PDBQT
        ob_mol.write("pdbqt", str(output_path), overwrite=True)
        
        logger.info(f"Receptor PDBQT prepared: {output_path}")
        return output_path
        
    except ImportError as e:
        logger.error(f"Missing dependency for receptor preparation: {e}")
        return None
    except Exception as e:
        logger.error(f"Receptor PDBQT preparation failed: {e}")
        return None


def get_pdbqt_preparation_explanation(
    success: bool,
    ligand: bool = True,
    error: str = None,
) -> dict:
    """
    Generate explanation for PDBQT preparation step.
    
    Args:
        success: Whether preparation succeeded
        ligand: True for ligand, False for receptor
        error: Error message if preparation failed
    
    Returns:
        Explanation dictionary
    """
    mol_type = "ligand" if ligand else "receptor"
    
    if success:
        return {
            "step": f"{mol_type}_preparation",
            "status": "SUCCESS",
            "actions": [
                "Generated 3D coordinates" if ligand else "Parsed PDB structure",
                "Added hydrogens" + (" at pH 7.4" if not ligand else ""),
                "Assigned Gasteiger partial charges",
                "Converted to PDBQT format",
            ],
            "charge_model": "Gasteiger",
            "limitations": [
                "Gasteiger charges are empirical, not QM-derived",
                "Protonation states assigned heuristically" if not ligand else None,
                "No explicit treatment of metal ions" if not ligand else None,
            ],
        }
    else:
        return {
            "step": f"{mol_type}_preparation",
            "status": "FAIL",
            "error": error,
            "implications": [
                f"{mol_type.capitalize()} PDBQT not generated",
                "Docking cannot proceed for this molecule",
            ],
        }


# Quick test
if __name__ == "__main__":
    if check_openbabel_available():
        # Test ligand preparation
        test_smiles = "CC(=O)Nc1ccc(O)cc1"  # Acetaminophen
        pdbqt = prepare_ligand_pdbqt(test_smiles)
        if pdbqt:
            print(f"Ligand PDBQT created: {pdbqt}")
            with open(pdbqt, 'r') as f:
                print(f.read()[:500])
    else:
        print("OpenBabel not available")

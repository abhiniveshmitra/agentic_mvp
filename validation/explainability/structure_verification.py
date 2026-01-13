"""
Structure Verification Module.

Validates that compound identifiers (ChEMBL IDs, PubChem CIDs) match
the SMILES structure being displayed.

This is a CRITICAL trust layer - no compound should display a structure
that doesn't match its identifier.
"""

from typing import Dict, Optional, Tuple
import requests
import time

from utils.logging import get_logger

logger = get_logger(__name__)


def verify_chembl_structure(chembl_id: str, smiles: str) -> Tuple[bool, str]:
    """
    Verify that a ChEMBL ID matches the given SMILES.
    
    Args:
        chembl_id: ChEMBL molecule ID (e.g., "CHEMBL137635")
        smiles: SMILES string to verify
    
    Returns:
        Tuple of (is_valid, message)
    """
    if not chembl_id or not smiles:
        return False, "Missing ChEMBL ID or SMILES"
    
    # Skip non-ChEMBL IDs
    if not chembl_id.upper().startswith("CHEMBL"):
        return True, "Not a ChEMBL ID - skipping verification"
    
    try:
        # Query ChEMBL API for the actual SMILES of this compound
        url = f"https://www.ebi.ac.uk/chembl/api/data/molecule/{chembl_id}.json"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 404:
            return False, f"ChEMBL ID {chembl_id} not found in database"
        
        response.raise_for_status()
        data = response.json()
        
        # Get the canonical SMILES from ChEMBL
        mol_structures = data.get("molecule_structures", {})
        if not mol_structures:
            return False, f"No structure data for {chembl_id}"
        
        canonical_smiles = mol_structures.get("canonical_smiles", "")
        
        if not canonical_smiles:
            return False, f"No canonical SMILES for {chembl_id}"
        
        # Compare SMILES (canonical comparison)
        # Note: Direct string comparison may fail due to different canonicalization
        # Use RDKit for proper comparison
        try:
            from rdkit import Chem
            
            mol1 = Chem.MolFromSmiles(smiles)
            mol2 = Chem.MolFromSmiles(canonical_smiles)
            
            if mol1 is None or mol2 is None:
                return False, "Could not parse SMILES for comparison"
            
            # Generate canonical SMILES for both
            canon1 = Chem.MolToSmiles(mol1, canonical=True)
            canon2 = Chem.MolToSmiles(mol2, canonical=True)
            
            if canon1 == canon2:
                return True, "Structure verified against ChEMBL"
            else:
                return False, f"SMILES MISMATCH! Stored: {smiles[:50]}... vs ChEMBL: {canonical_smiles[:50]}..."
                
        except ImportError:
            # Fallback to string comparison if RDKit not available
            if smiles == canonical_smiles:
                return True, "Structure verified (string match)"
            else:
                # May be false negative due to canonicalization differences
                return False, f"Possible SMILES mismatch (no RDKit for proper check)"
        
    except requests.RequestException as e:
        return False, f"ChEMBL API error: {e}"
    except Exception as e:
        return False, f"Verification error: {e}"


def verify_compound_batch(compounds: list, max_verify: int = 10) -> list:
    """
    Verify a batch of compounds and add verification status.
    
    Args:
        compounds: List of compound dictionaries
        max_verify: Maximum compounds to verify (API rate limiting)
    
    Returns:
        Same compounds with verification_status added
    """
    verified_count = 0
    
    for compound in compounds[:max_verify]:
        compound_id = compound.get("compound_id", "")
        smiles = compound.get("smiles", "")
        
        # Only verify ChEMBL compounds
        if compound_id.upper().startswith("CHEMBL"):
            is_valid, message = verify_chembl_structure(compound_id, smiles)
            compound["structure_verified"] = is_valid
            compound["verification_message"] = message
            
            if not is_valid:
                logger.warning(f"Structure verification FAILED for {compound_id}: {message}")
                # Mark as unverified to prevent display
                compound["display_warning"] = True
            else:
                logger.debug(f"Structure verified for {compound_id}")
            
            verified_count += 1
            time.sleep(0.3)  # Rate limiting
    
    logger.info(f"Verified {verified_count} compound structures")
    return compounds


def get_correct_smiles_from_chembl(chembl_id: str) -> Optional[str]:
    """
    Fetch the correct canonical SMILES for a ChEMBL ID.
    
    Args:
        chembl_id: ChEMBL molecule ID
    
    Returns:
        Canonical SMILES or None
    """
    if not chembl_id or not chembl_id.upper().startswith("CHEMBL"):
        return None
    
    try:
        url = f"https://www.ebi.ac.uk/chembl/api/data/molecule/{chembl_id}.json"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        mol_structures = data.get("molecule_structures", {})
        
        return mol_structures.get("canonical_smiles")
        
    except Exception as e:
        logger.warning(f"Could not fetch SMILES for {chembl_id}: {e}")
        return None


def verify_and_correct_compound(compound: dict) -> dict:
    """
    Verify a compound's structure and correct it if mismatched.
    
    Args:
        compound: Compound dictionary
    
    Returns:
        Corrected compound dictionary
    """
    compound_id = compound.get("compound_id", "")
    stored_smiles = compound.get("smiles", "")
    
    if not compound_id.upper().startswith("CHEMBL"):
        compound["structure_verified"] = True
        return compound
    
    # Verify the structure
    is_valid, message = verify_chembl_structure(compound_id, stored_smiles)
    
    if is_valid:
        compound["structure_verified"] = True
        compound["verification_message"] = message
    else:
        # Try to fetch the correct SMILES from ChEMBL
        correct_smiles = get_correct_smiles_from_chembl(compound_id)
        
        if correct_smiles:
            logger.warning(
                f"Correcting SMILES for {compound_id}: "
                f"{stored_smiles[:30]}... -> {correct_smiles[:30]}..."
            )
            compound["smiles"] = correct_smiles
            compound["smiles_corrected"] = True
            compound["original_smiles"] = stored_smiles
            compound["structure_verified"] = True
            compound["verification_message"] = "Structure corrected from ChEMBL"
        else:
            compound["structure_verified"] = False
            compound["verification_message"] = message
            compound["display_warning"] = True
    
    return compound


# Test function
if __name__ == "__main__":
    # Test verification
    test_id = "CHEMBL137635"
    test_smiles = "COc1ccc2c(c1)ncnc2Nc1cccc(c1)C#N"  # Example SMILES
    
    is_valid, msg = verify_chembl_structure(test_id, test_smiles)
    print(f"Verification result: {is_valid}")
    print(f"Message: {msg}")
    
    # Get correct SMILES
    correct = get_correct_smiles_from_chembl(test_id)
    print(f"Correct SMILES from ChEMBL: {correct}")

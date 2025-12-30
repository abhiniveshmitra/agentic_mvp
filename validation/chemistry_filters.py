"""
Chemistry Filters - Hard Gate for Chemical Validity.

CRITICAL CONSTRAINTS (from agent knowledge base):
- This agent OVERRIDES all others
- Rejected molecules are logged, preserved, NEVER deleted
- If it fails chemistry, nothing else matters

Filters:
- RDKit parse validity
- MW: 150-700 Da
- LogP: -1 to 6
- Rotatable bonds: ≤ 10
- Absolute formal charge: ≤ 2
- PAINS filter (target-conditional)
"""

from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field

from utils.logging import get_logger
from utils.provenance import ProvenanceTracker, FilterInfo, FilterStatus

logger = get_logger(__name__)

# Try to import RDKit
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors, FilterCatalog
    from rdkit.Chem.FilterCatalog import FilterCatalogParams
    from rdkit.Chem.SaltRemover import SaltRemover
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False
    logger.warning("RDKit not available - chemistry filters disabled")


def normalize_smiles(smiles: str) -> str:
    """
    Normalize SMILES by removing salts and keeping largest fragment.
    
    This is critical for processing PubChem SMILES which often contain:
    - Salt forms (HCl, sodium, etc.)
    - Multiple fragments
    - Counterions
    
    Args:
        smiles: Raw SMILES string
    
    Returns:
        Normalized SMILES (largest fragment, salts removed)
    """
    if not HAS_RDKIT:
        return smiles
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return smiles
        
        # Remove common salts
        try:
            remover = SaltRemover()
            mol = remover.StripMol(mol, dontRemoveEverything=True)
        except:
            pass
        
        # If multiple fragments, keep the largest one
        fragments = Chem.GetMolFrags(mol, asMols=True)
        if len(fragments) > 1:
            # Keep largest by heavy atom count
            mol = max(fragments, key=lambda m: m.GetNumHeavyAtoms())
            logger.debug(f"Kept largest fragment from {len(fragments)} fragments")
        
        # Return canonical SMILES
        return Chem.MolToSmiles(mol, canonical=True)
        
    except Exception as e:
        logger.warning(f"Salt normalization failed: {e}")
        return smiles


@dataclass
class FilterResult:
    """Result of applying filters to a compound."""
    passed: bool
    passed_filters: List[str] = field(default_factory=list)
    failed_filters: List[str] = field(default_factory=list)
    details: Dict = field(default_factory=dict)


def apply_all_filters(
    compounds: List[Dict],
    config,  # ChemistryFilterConfig
    provenance: Optional[ProvenanceTracker] = None,
    known_drugs: Optional[List[str]] = None,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Apply all chemistry filters to compounds.
    
    Args:
        compounds: List of compound dictionaries with 'smiles' key
        config: ChemistryFilterConfig with thresholds
        provenance: Optional provenance tracker
        known_drugs: Optional list of known drug names that bypass filters
    
    Returns:
        Tuple of (passed_compounds, rejected_compounds)
    """
    if not HAS_RDKIT:
        logger.error("RDKit not available - cannot filter compounds")
        return compounds, []
    
    # Known drug names (lowercase for matching)
    known_drug_set = set(d.lower() for d in (known_drugs or []))
    
    passed = []
    rejected = []
    
    for compound in compounds:
        smiles = compound.get("smiles")
        compound_id = compound.get("compound_id", "unknown")
        compound_name = compound.get("compound_name", "").lower()
        
        if not smiles:
            # No SMILES - reject
            compound["rejection_reason"] = "NO_SMILES"
            rejected.append(compound)
            continue
        
        # Normalize SMILES (remove salts, keep largest fragment)
        original_smiles = smiles
        normalized_smiles = normalize_smiles(smiles)
        if normalized_smiles != original_smiles:
            logger.debug(f"Normalized SMILES for {compound_id}: {original_smiles} -> {normalized_smiles}")
            compound["smiles"] = normalized_smiles
            compound["original_smiles"] = original_smiles
        
        # Check if this is a known drug (pass-through for calibration)
        is_known_drug = compound_name in known_drug_set or compound.get("is_seed", False)
        
        # Apply filters
        result = _apply_filters_single(normalized_smiles, config)
        
        if result.passed:
            compound["filter_status"] = "PASSED"
            compound["filter_details"] = result.details
            passed.append(compound)
            
            # Update provenance
            if provenance:
                provenance.update_filter_info(
                    compound_id,
                    FilterInfo(
                        status=FilterStatus.PASSED,
                        passed_filters=result.passed_filters,
                        failed_filters=[],
                        filter_details=result.details,
                    )
                )
        elif is_known_drug:
            # Known-ligand pass-through: allow for calibration but mark as such
            compound["filter_status"] = "KNOWN_CONTROL_BYPASS"
            compound["filter_details"] = result.details
            compound["bypass_reason"] = f"Known drug ({compound_name}) - bypassed for calibration"
            compound["original_filter_failures"] = result.failed_filters
            passed.append(compound)
            logger.info(f"Known drug '{compound_name}' bypassed filters: {result.failed_filters}")
            
            if provenance:
                provenance.update_filter_info(
                    compound_id,
                    FilterInfo(
                        status=FilterStatus.PASSED,
                        passed_filters=result.passed_filters + ["KNOWN_DRUG_BYPASS"],
                        failed_filters=result.failed_filters,
                        filter_details=result.details,
                    )
                )
        else:
            compound["filter_status"] = "REJECTED"
            compound["rejection_reason"] = ", ".join(result.failed_filters)
            compound["filter_details"] = result.details
            rejected.append(compound)
            
            # Log rejection reason for debugging
            logger.info(f"Rejected {compound_id} ({compound_name}): {result.failed_filters}")
            logger.info(f"  Details: MW={result.details.get('molecular_weight')}, LogP={result.details.get('logp')}")
            
            # Update provenance
            if provenance:
                provenance.update_filter_info(
                    compound_id,
                    FilterInfo(
                        status=FilterStatus.REJECTED,
                        passed_filters=result.passed_filters,
                        failed_filters=result.failed_filters,
                        filter_details=result.details,
                    )
                )
    
    logger.info(f"Filters: {len(passed)} passed, {len(rejected)} rejected")
    return passed, rejected


def _apply_filters_single(smiles: str, config) -> FilterResult:
    """
    Apply all filters to a single molecule.
    
    Args:
        smiles: SMILES string
        config: ChemistryFilterConfig
    
    Returns:
        FilterResult with pass/fail status and details
    """
    passed_filters = []
    failed_filters = []
    details = {"smiles": smiles}
    
    # Filter 1: RDKit Parse Validity
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return FilterResult(
            passed=False,
            failed_filters=["RDKIT_PARSE"],
            details={"error": "Failed to parse SMILES"}
        )
    passed_filters.append("RDKIT_PARSE")
    
    # Filter 2: Molecular Weight
    mw = Descriptors.MolWt(mol)
    details["molecular_weight"] = round(mw, 2)
    
    if mw < config.mw_min or mw > config.mw_max:
        failed_filters.append(f"MW_{mw:.0f}")
    else:
        passed_filters.append("MW")
    
    # Filter 3: LogP
    logp = Descriptors.MolLogP(mol)
    details["logp"] = round(logp, 2)
    
    if logp < config.logp_min or logp > config.logp_max:
        failed_filters.append(f"LOGP_{logp:.1f}")
    else:
        passed_filters.append("LOGP")
    
    # Filter 4: Rotatable Bonds
    rot_bonds = rdMolDescriptors.CalcNumRotatableBonds(mol)
    details["rotatable_bonds"] = rot_bonds
    
    if rot_bonds > config.rotatable_bonds_max:
        failed_filters.append(f"ROTBONDS_{rot_bonds}")
    else:
        passed_filters.append("ROTBONDS")
    
    # Filter 5: Formal Charge
    formal_charge = Chem.GetFormalCharge(mol)
    details["formal_charge"] = formal_charge
    
    if abs(formal_charge) > config.formal_charge_max_abs:
        failed_filters.append(f"CHARGE_{formal_charge}")
    else:
        passed_filters.append("CHARGE")
    
    # Filter 6: PAINS (if enabled)
    if config.enable_pains:
        pains_result = _check_pains(mol)
        details["pains_alerts"] = pains_result["alerts"]
        
        if pains_result["has_pains"]:
            failed_filters.append(f"PAINS_{len(pains_result['alerts'])}")
        else:
            passed_filters.append("PAINS")
    
    # Additional useful descriptors (not filters, just info)
    details["hbd"] = rdMolDescriptors.CalcNumHBD(mol)
    details["hba"] = rdMolDescriptors.CalcNumHBA(mol)
    details["tpsa"] = round(Descriptors.TPSA(mol), 2)
    details["num_heavy_atoms"] = mol.GetNumHeavyAtoms()
    
    return FilterResult(
        passed=len(failed_filters) == 0,
        passed_filters=passed_filters,
        failed_filters=failed_filters,
        details=details,
    )


def _check_pains(mol) -> Dict:
    """
    Check molecule for PAINS (Pan-Assay Interference Compounds) alerts.
    
    Args:
        mol: RDKit molecule object
    
    Returns:
        Dictionary with has_pains flag and alert list
    """
    try:
        params = FilterCatalogParams()
        params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
        catalog = FilterCatalog.FilterCatalog(params)
        
        entries = catalog.GetMatches(mol)
        alerts = [entry.GetDescription() for entry in entries]
        
        return {
            "has_pains": len(alerts) > 0,
            "alerts": alerts,
        }
    except Exception as e:
        logger.warning(f"PAINS check failed: {e}")
        return {"has_pains": False, "alerts": []}


def validate_smiles(smiles: str) -> Tuple[bool, Optional[str]]:
    """
    Validate a SMILES string and return canonical form.
    
    Args:
        smiles: SMILES string to validate
    
    Returns:
        Tuple of (is_valid, canonical_smiles or None)
    """
    if not HAS_RDKIT:
        return True, smiles  # Can't validate without RDKit
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False, None
        
        canonical = Chem.MolToSmiles(mol, canonical=True)
        return True, canonical
        
    except Exception:
        return False, None


def calculate_properties(smiles: str) -> Optional[Dict]:
    """
    Calculate molecular properties for a SMILES.
    
    Args:
        smiles: SMILES string
    
    Returns:
        Dictionary of properties or None if invalid
    """
    if not HAS_RDKIT:
        return None
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    return {
        "molecular_weight": round(Descriptors.MolWt(mol), 2),
        "logp": round(Descriptors.MolLogP(mol), 2),
        "tpsa": round(Descriptors.TPSA(mol), 2),
        "hbd": rdMolDescriptors.CalcNumHBD(mol),
        "hba": rdMolDescriptors.CalcNumHBA(mol),
        "rotatable_bonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
        "num_heavy_atoms": mol.GetNumHeavyAtoms(),
        "num_rings": rdMolDescriptors.CalcNumRings(mol),
        "formal_charge": Chem.GetFormalCharge(mol),
    }

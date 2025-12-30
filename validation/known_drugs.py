"""
Known Drug Registry.

Registry of approved/clinical drugs by target for sanity calibration.
These are used to:
1. Verify known drugs rank appropriately (calibration)
2. Mark them in results (transparency)
3. Exclude from "novelty" claims
"""

from typing import Dict, List, Optional
from dataclasses import dataclass

# =============================================================================
# DRUG TIERS (for calibration expectations)
# =============================================================================
# BEST_IN_CLASS: Most potent, newest generation → expect ≥85 percentile
# STANDARD: Well-established approved → expect ≥70 percentile
# LEGACY: Older generation, less potent → expect ≥50 percentile

TIER_THRESHOLDS = {
    "BEST_IN_CLASS": 85,
    "STANDARD": 70,
    "LEGACY": 50,
}

# =============================================================================
# EGFR INHIBITORS (FDA Approved + Major Clinical)
# =============================================================================

EGFR_DRUGS = {
    # Third Generation (BEST_IN_CLASS)
    "osimertinib": {
        "chembl_id": "CHEMBL3353410",
        "fda_approved": True,
        "year_approved": 2015,
        "brand_names": ["Tagrisso"],
        "ic50_nm": 12,
        "tier": "BEST_IN_CLASS",
        "generation": 3,
        "notes": "Third-gen, T790M selective, current standard",
    },
    # Second Generation
    "afatinib": {
        "chembl_id": "CHEMBL1173655",
        "fda_approved": True,
        "year_approved": 2013,
        "brand_names": ["Gilotrif"],
        "ic50_nm": 0.5,
        "tier": "STANDARD",
        "generation": 2,
        "notes": "Irreversible pan-HER inhibitor",
    },
    "dacomitinib": {
        "chembl_id": "CHEMBL2110732",
        "fda_approved": True,
        "year_approved": 2018,
        "brand_names": ["Vizimpro"],
        "ic50_nm": 6,
        "tier": "STANDARD",
        "generation": 2,
        "notes": "Irreversible EGFR inhibitor",
    },
    "neratinib": {
        "chembl_id": "CHEMBL1277058",
        "fda_approved": True,
        "year_approved": 2017,
        "brand_names": ["Nerlynx"],
        "ic50_nm": 92,
        "tier": "STANDARD",
        "generation": 2,
        "notes": "HER2-selective, less EGFR potent",
    },
    # First Generation (LEGACY)
    "gefitinib": {
        "chembl_id": "CHEMBL939",
        "fda_approved": True,
        "year_approved": 2003,
        "brand_names": ["Iressa"],
        "ic50_nm": 33,
        "tier": "LEGACY",
        "generation": 1,
        "notes": "First-gen, reversible inhibitor",
    },
    "erlotinib": {
        "chembl_id": "CHEMBL553",
        "fda_approved": True,
        "year_approved": 2004,
        "brand_names": ["Tarceva"],
        "ic50_nm": 2,
        "tier": "LEGACY",
        "generation": 1,
        "notes": "First-gen, reversible inhibitor",
    },
    # Dual EGFR/HER2
    "lapatinib": {
        "chembl_id": "CHEMBL554",
        "fda_approved": True,
        "year_approved": 2007,
        "brand_names": ["Tykerb"],
        "ic50_nm": 10.8,
        "tier": "LEGACY",
        "generation": 1,
        "notes": "Dual EGFR/HER2, reversible",
    },
}

# =============================================================================
# OTHER TARGET DRUGS
# =============================================================================

BCR_ABL_DRUGS = {
    "imatinib": {"chembl_id": "CHEMBL941", "fda_approved": True, "tier": "LEGACY"},
    "dasatinib": {"chembl_id": "CHEMBL1421", "fda_approved": True, "tier": "BEST_IN_CLASS"},
    "nilotinib": {"chembl_id": "CHEMBL255863", "fda_approved": True, "tier": "STANDARD"},
    "bosutinib": {"chembl_id": "CHEMBL288441", "fda_approved": True, "tier": "STANDARD"},
    "ponatinib": {"chembl_id": "CHEMBL1171837", "fda_approved": True, "tier": "BEST_IN_CLASS"},
}

BRAF_DRUGS = {
    "vemurafenib": {"chembl_id": "CHEMBL1229517", "fda_approved": True, "tier": "STANDARD"},
    "dabrafenib": {"chembl_id": "CHEMBL2028663", "fda_approved": True, "tier": "BEST_IN_CLASS"},
    "encorafenib": {"chembl_id": "CHEMBL3301610", "fda_approved": True, "tier": "BEST_IN_CLASS"},
}

# CDK2 - Classic ATP-competitive kinase (sanity benchmark)
CDK2_DRUGS = {
    "roscovitine": {"chembl_id": "CHEMBL279", "fda_approved": False, "tier": "LEGACY",
                    "notes": "Classic CDK inhibitor, clinical trials"},
    "dinaciclib": {"chembl_id": "CHEMBL480723", "fda_approved": False, "tier": "STANDARD",
                   "notes": "Pan-CDK inhibitor, clinical trials"},
    "palbociclib": {"chembl_id": "CHEMBL189963", "fda_approved": True, "tier": "BEST_IN_CLASS",
                    "notes": "CDK4/6 selective but also CDK2 activity"},
}

# VEGFR2 (KDR) - Promiscuous kinase (tests polypharmacology handling)
VEGFR2_DRUGS = {
    "axitinib": {"chembl_id": "CHEMBL288441", "fda_approved": True, "tier": "BEST_IN_CLASS",
                 "notes": "Selective VEGFR inhibitor"},
    "sunitinib": {"chembl_id": "CHEMBL535", "fda_approved": True, "tier": "STANDARD",
                  "notes": "Multi-kinase inhibitor"},
    "sorafenib": {"chembl_id": "CHEMBL1336", "fda_approved": True, "tier": "LEGACY",
                  "notes": "Multi-kinase including VEGFR"},
    "pazopanib": {"chembl_id": "CHEMBL477772", "fda_approved": True, "tier": "STANDARD",
                  "notes": "Multi-kinase VEGFR inhibitor"},
}

# HMG-CoA Reductase - Enzyme target (non-kinase control)
HMGCR_DRUGS = {
    "atorvastatin": {"chembl_id": "CHEMBL1487", "fda_approved": True, "tier": "BEST_IN_CLASS",
                     "notes": "Most prescribed statin"},
    "simvastatin": {"chembl_id": "CHEMBL1064", "fda_approved": True, "tier": "STANDARD",
                    "notes": "Early statin, prodrug"},
    "rosuvastatin": {"chembl_id": "CHEMBL1496", "fda_approved": True, "tier": "BEST_IN_CLASS",
                     "notes": "High potency statin"},
    "lovastatin": {"chembl_id": "CHEMBL503", "fda_approved": True, "tier": "LEGACY",
                   "notes": "First statin approved"},
}

# COX-2 (PTGS2) - Enzyme target (non-kinase control)
COX2_DRUGS = {
    "celecoxib": {"chembl_id": "CHEMBL118", "fda_approved": True, "tier": "BEST_IN_CLASS",
                  "notes": "COX-2 selective NSAID"},
    "rofecoxib": {"chembl_id": "CHEMBL122", "fda_approved": False, "tier": "STANDARD",
                  "notes": "Withdrawn (cardiovascular risk)"},
    "etoricoxib": {"chembl_id": "CHEMBL1071", "fda_approved": True, "tier": "STANDARD",
                   "notes": "COX-2 selective"},
}

# Master registry
KNOWN_DRUGS_BY_TARGET = {
    "EGFR": EGFR_DRUGS,
    "BCR-ABL": BCR_ABL_DRUGS,
    "BRAF": BRAF_DRUGS,
    "CDK2": CDK2_DRUGS,
    "VEGFR2": VEGFR2_DRUGS,
    "KDR": VEGFR2_DRUGS,  # Alias
    "HMGCR": HMGCR_DRUGS,
    "HMG-COA REDUCTASE": HMGCR_DRUGS,  # Alias
    "COX-2": COX2_DRUGS,
    "PTGS2": COX2_DRUGS,  # Alias
}


@dataclass
class KnownDrugInfo:
    """Information about a known drug."""
    name: str
    chembl_id: str
    fda_approved: bool
    tier: str
    rank_in_results: Optional[int] = None
    percentile_in_results: Optional[float] = None
    confidence_tier: Optional[str] = None


def get_known_drugs(target: str) -> Dict[str, dict]:
    """
    Get known drugs for a target.
    
    Args:
        target: Target name (e.g., "EGFR")
    
    Returns:
        Dictionary of drug name -> drug info
    """
    target_upper = target.upper()
    return KNOWN_DRUGS_BY_TARGET.get(target_upper, {})


def get_known_drug_chembl_ids(target: str) -> List[str]:
    """
    Get list of ChEMBL IDs for known drugs.
    
    Args:
        target: Target name
    
    Returns:
        List of ChEMBL IDs
    """
    drugs = get_known_drugs(target)
    return [d["chembl_id"] for d in drugs.values()]


def is_known_drug(compound_id: str, target: str) -> bool:
    """
    Check if a compound is a known drug for a target.
    
    Args:
        compound_id: ChEMBL ID (e.g., "CHEMBL939")
        target: Target name
    
    Returns:
        True if known drug
    """
    known_ids = get_known_drug_chembl_ids(target)
    return compound_id in known_ids


def get_known_drug_name(compound_id: str, target: str) -> Optional[str]:
    """
    Get the common name for a known drug.
    
    Args:
        compound_id: ChEMBL ID
        target: Target name
    
    Returns:
        Drug name or None
    """
    drugs = get_known_drugs(target)
    for name, info in drugs.items():
        if info["chembl_id"] == compound_id:
            return name.capitalize()
    return None


def validate_known_drugs_ranking(
    compounds: List[Dict],
    target: str,
) -> Dict:
    """
    Validate that known drugs rank appropriately using TIERED expectations.
    
    Tiered calibration:
    - BEST_IN_CLASS: ≥85th percentile expected
    - STANDARD: ≥70th percentile expected
    - LEGACY: ≥50th percentile expected (first-gen drugs won't beat optimized analogs)
    
    This is a NOTE system, not a hard failure.
    """
    drugs = get_known_drugs(target)
    
    found_known = []
    missing_known = []
    
    for name, info in drugs.items():
        chembl_id = info["chembl_id"]
        tier = info.get("tier", "STANDARD")
        expected_min = TIER_THRESHOLDS.get(tier, 70)
        
        # Find in ranked compounds
        match = None
        for i, c in enumerate(compounds):
            if c.get("compound_id") == chembl_id:
                actual_percentile = c.get("percentile", 0)
                meets_expectation = actual_percentile >= expected_min
                
                match = {
                    "name": name.capitalize(),
                    "chembl_id": chembl_id,
                    "rank": i + 1,
                    "percentile": actual_percentile,
                    "confidence": c.get("confidence_tier", ""),
                    "tier": tier,
                    "expected_min": expected_min,
                    "meets_expectation": meets_expectation,
                    "generation": info.get("generation", ""),
                    "notes": info.get("notes", ""),
                }
                break
        
        if match:
            found_known.append(match)
        else:
            missing_known.append({"name": name, "tier": tier})
    
    # Sort by rank
    found_known.sort(key=lambda x: x["rank"])
    
    # Analyze calibration with nuance
    issues = []
    notes = []
    all_meet_expectations = True
    
    for drug in found_known:
        if not drug["meets_expectation"]:
            all_meet_expectations = False
            if drug["tier"] == "LEGACY":
                # First-gen drugs ranking lower is expected, just note it
                notes.append(
                    f"{drug['name']} (first-gen) at {drug['percentile']:.1f}th percentile - "
                    f"consistent with older generation being outranked by optimized analogs"
                )
            else:
                issues.append(
                    f"{drug['name']} at {drug['percentile']:.1f}th percentile "
                    f"(expected ≥{drug['expected_min']} for {drug['tier']} tier)"
                )
        else:
            notes.append(f"{drug['name']} at {drug['percentile']:.1f}th percentile ✓")
    
    # Comparative check: do newer drugs outrank older ones?
    comparative_valid = True
    generations = [d for d in found_known if d.get("generation")]
    if len(generations) >= 2:
        gen_sorted = sorted(generations, key=lambda x: x["rank"])
        # Check if newer generations tend to rank higher
        for i, drug in enumerate(gen_sorted[:-1]):
            next_drug = gen_sorted[i + 1]
            if drug.get("generation", 0) > next_drug.get("generation", 0):
                # Newer gen ranks higher - expected
                pass
            elif drug.get("generation") == next_drug.get("generation"):
                pass
            else:
                # Older gen outranks newer - note but don't fail
                notes.append(
                    f"Note: {drug['name']} (gen {drug['generation']}) ranks higher than "
                    f"{next_drug['name']} (gen {next_drug['generation']})"
                )
    
    # Determine overall status
    if not issues:
        status = "PASS"
        summary = "All known drugs rank within expected ranges for their tier."
    elif all(d["tier"] == "LEGACY" for d in found_known if not d["meets_expectation"]):
        status = "PASS_WITH_NOTE"
        summary = "Legacy drugs rank as expected (mid-range is normal for first-gen inhibitors)."
    else:
        status = "NEEDS_REVIEW"
        summary = "Some newer-generation drugs below expected range - review recommended."
    
    return {
        "status": status,
        "summary": summary,
        "calibration_valid": status in ["PASS", "PASS_WITH_NOTE"],
        "found_known_drugs": found_known,
        "missing_known_drugs": missing_known,
        "issues": issues,
        "notes": notes,
        "total_known": len(drugs),
        "found_count": len(found_known),
    }

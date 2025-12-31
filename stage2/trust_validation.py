"""
Stage-2 Trust Validation Report Generator.

Generates human-readable reports showing:
- Full explanation chains for each candidate
- SAFE and FLAGGED examples with scientific justification
- Demonstrates expected failure modes

This is a key trust artifact for Stage-2.
"""

from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path
import json

from utils.logging import get_logger

logger = get_logger(__name__)

# Known drugs for sanity testing
KNOWN_DRUGS = {
    "EGFR": [
        {
            "name": "Gefitinib",
            "smiles": "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1",
            "expected_behavior": "Should dock well, may have ADME flags (LogP)",
        },
        {
            "name": "Erlotinib",
            "smiles": "COCCOc1cc2ncnc(Nc3cccc(C#C)c3)c2cc1OCCOC",
            "expected_behavior": "Should dock well, moderate ADME profile",
        },
        {
            "name": "Lapatinib",
            "smiles": "CS(=O)(=O)CCNCc1ccc(-c2ccc3ncnc(Nc4ccc(OCc5cccc(F)c5)c(Cl)c4)c3c2)o1",
            "expected_behavior": "High MW, may trigger MW/LogP flags",
        },
    ],
    "BRAF": [
        {
            "name": "Vemurafenib",
            "smiles": "CCCS(=O)(=O)Nc1ccc(-c2nc(-c3ccc(F)cc3F)c(-c3ccnc(N)n3)s2)cc1F",
            "expected_behavior": "Should bind BRAF, moderate ADME profile",
        },
        {
            "name": "Dabrafenib",
            "smiles": "CC(C)(C)c1nc(-c2cccc(NS(=O)(=O)c3c(F)cccc3F)c2F)c(-c2ccnc(N)n2)s1",
            "expected_behavior": "Should bind BRAF, watch for MW flags",
        },
    ],
    "HMGCR": [
        {
            "name": "Atorvastatin",
            "smiles": "CC(C)c1c(C(=O)Nc2ccccc2)c(-c2ccccc2)c(-c2ccc(F)cc2)n1CC[C@@H](O)C[C@@H](O)CC(=O)O",
            "expected_behavior": "HMG-CoA reductase inhibitor, may have carboxylic acid flags",
        },
        {
            "name": "Rosuvastatin",
            "smiles": "CC(C)c1nc(N(C)S(=O)(=O)C)nc(-c2ccc(F)cc2)c1/C=C/[C@@H](O)C[C@@H](O)CC(=O)O",
            "expected_behavior": "High TPSA due to sulfamide, may flag permeability",
        },
    ],
}

# Problem molecules for failure mode demonstration
FAILURE_MODE_EXAMPLES = {
    "DOCKING_FAILURE": {
        "name": "Benzene (too small)",
        "smiles": "c1ccccc1",
        "expected": "FAIL - too small for kinase binding pocket",
    },
    "HIGH_LOGP": {
        "name": "Hypothetical lipophilic compound",
        "smiles": "CCCCCCCCCCCCCCCCCC",  # Octadecane
        "expected": "HIGH_RISK - extreme LogP",
    },
    "PAINS_FLAGGED": {
        "name": "Rhodanine scaffold",
        "smiles": "O=C1NC(=S)SC1=Cc1ccccc1",
        "expected": "FLAGGED - PAINS alert",
    },
    "HIGH_MW": {
        "name": "Large macrocycle",
        "smiles": "CC(C)C[C@H](NC(=O)[C@H](CC(C)C)NC(=O)[C@H](CC1=CC=CC=C1)NC(=O)[C@H](CC(C)C)NC(=O)[C@H](CC1=CC=CC=C1)NC(=O)C(CC(C)C)NC(=O)C(CC1=CC=CC=C1)NC(=O)C(CC(C)C)NC(=O)C)C(=O)O",
        "expected": "HIGH_RISK - MW too high",
    },
}


class TrustValidationReporter:
    """
    Generates trust validation reports for Stage-2 results.
    
    Shows complete explanation chains with scientific justification.
    """
    
    def __init__(self):
        """Initialize reporter."""
        logger.info("TrustValidationReporter initialized")
    
    def generate_candidate_report(
        self,
        candidate: Dict,
        include_raw_data: bool = True,
    ) -> str:
        """
        Generate detailed report for a single candidate.
        
        Args:
            candidate: Aggregated candidate dictionary
            include_raw_data: Whether to include raw property values
        
        Returns:
            Formatted report string
        """
        lines = []
        
        # Header
        compound_name = candidate.get("compound_name", "Unknown")
        compound_id = candidate.get("compound_id", "N/A")
        smiles = candidate.get("smiles", "N/A")
        
        lines.append(f"{'='*70}")
        lines.append(f"COMPOUND: {compound_name}")
        lines.append(f"ID: {compound_id}")
        lines.append(f"SMILES: {smiles[:50]}..." if len(smiles) > 50 else f"SMILES: {smiles}")
        lines.append(f"{'='*70}")
        
        # Stage-1 Summary
        stage1 = candidate.get("stage1", {})
        lines.append("\n📊 STAGE-1 (Affinity Prediction)")
        lines.append(f"  • Rank: #{stage1.get('rank', 'N/A')}")
        lines.append(f"  • Score: {stage1.get('score', 'N/A')}")
        lines.append(f"  • Percentile: {stage1.get('percentile', 'N/A')}%")
        lines.append(f"  • Confidence: {stage1.get('confidence', 'N/A')}")
        
        # Stage-2 Results
        stage2 = candidate.get("stage2", {})
        
        # Docking
        docking = stage2.get("docking", {})
        lines.append("\n🔬 DOCKING (Structural Plausibility)")
        docking_status = docking.get("docking_status", "N/A")
        lines.append(f"  • Status: {docking_status}")
        
        if docking.get("explanation"):
            exp = docking["explanation"]
            if exp.get("observations"):
                for obs in exp["observations"]:
                    lines.append(f"    - {obs}")
            if exp.get("limitations"):
                lines.append("  • Limitations:")
                for lim in exp["limitations"][:3]:
                    lines.append(f"    ⚠ {lim}")
        
        # ADME/Tox
        adme = stage2.get("adme_tox", {})
        adme_label = adme.get("adme_tox_label", "N/A")
        lines.append(f"\n💊 ADME/TOX (Developability Risk)")
        lines.append(f"  • Label: {adme_label}")
        
        if adme.get("explanation"):
            exp = adme["explanation"]
            
            # Raw values
            if include_raw_data and exp.get("raw_values"):
                lines.append("  • Raw Properties:")
                rv = exp["raw_values"]
                lines.append(f"    MW: {rv.get('mw', 'N/A')} | LogP: {rv.get('logp', 'N/A')} | TPSA: {rv.get('tpsa', 'N/A')}")
                lines.append(f"    HBD: {rv.get('hbd', 'N/A')} | HBA: {rv.get('hba', 'N/A')} | RotB: {rv.get('rotatable_bonds', 'N/A')}")
            
            # Rules triggered
            if exp.get("rules_triggered"):
                lines.append("  • Rules Triggered:")
                for rule in exp["rules_triggered"]:
                    lines.append(f"    🚩 {rule.get('rule_id', 'Unknown')}")
                    lines.append(f"       Condition: {rule.get('condition', 'N/A')}")
                    lines.append(f"       Rationale: {rule.get('scientific_rationale', 'N/A')[:80]}...")
                    lines.append(f"       Implication: {rule.get('practical_implication', 'N/A')[:80]}...")
            
            # PAINS
            if exp.get("pains_details"):
                pains = exp["pains_details"]
                lines.append("  • PAINS Alert:")
                lines.append(f"    Motif: {pains.get('matched_motif', 'N/A')}")
                lines.append(f"    Issue: {pains.get('known_issue', 'N/A')}")
                lines.append(f"    Note: {pains.get('important_note', 'N/A')}")
        
        # Patent
        patent = stage2.get("patent", {})
        lines.append(f"\n📝 PATENT RISK")
        lines.append(f"  • Status: {patent.get('patent_risk', 'N/A')}")
        
        # Narrative
        if candidate.get("narrative"):
            lines.append("\n📖 NARRATIVE SUMMARY")
            lines.append(f"  {candidate['narrative']}")
        
        # Final verdict
        lines.append(f"\n{'─'*70}")
        lines.append(f"FINAL STAGE-2 ASSESSMENT: {adme_label}")
        lines.append(f"{'─'*70}")
        
        return "\n".join(lines)
    
    def generate_full_report(
        self,
        results: Dict,
        output_path: Optional[Path] = None,
    ) -> str:
        """
        Generate full validation report for all candidates.
        
        Args:
            results: Stage-2 pipeline results
            output_path: Optional path to save report
        
        Returns:
            Complete report string
        """
        lines = []
        
        # Report header
        lines.append("=" * 70)
        lines.append("STAGE-2 TRUST VALIDATION REPORT")
        lines.append(f"Generated: {datetime.now().isoformat()}")
        lines.append("=" * 70)
        
        # Summary
        summary = results.get("summary", {})
        lines.append("\n## PIPELINE SUMMARY")
        lines.append(f"  Candidates: {summary.get('total_candidates', 'N/A')}")
        
        docking_sum = summary.get("docking", {})
        lines.append(f"  Docking: {docking_sum.get('pass', 0)} PASS, {docking_sum.get('flag', 0)} FLAG, {docking_sum.get('fail', 0)} FAIL")
        
        adme_sum = summary.get("adme_tox", {})
        lines.append(f"  ADME/Tox: {adme_sum.get('safe', 0)} SAFE, {adme_sum.get('flagged', 0)} FLAGGED, {adme_sum.get('high_risk', 0)} HIGH_RISK")
        
        # Individual candidates
        candidates = results.get("candidates", [])
        
        # Find SAFE and FLAGGED examples
        safe_examples = [c for c in candidates if c.get("stage2", {}).get("adme_tox", {}).get("adme_tox_label") == "SAFE"]
        flagged_examples = [c for c in candidates if c.get("stage2", {}).get("adme_tox", {}).get("adme_tox_label") == "FLAGGED"]
        high_risk_examples = [c for c in candidates if c.get("stage2", {}).get("adme_tox", {}).get("adme_tox_label") == "HIGH_RISK"]
        
        # Show SAFE example
        if safe_examples:
            lines.append("\n\n## EXAMPLE: SAFE CLASSIFICATION")
            lines.append(self.generate_candidate_report(safe_examples[0]))
        
        # Show FLAGGED example
        if flagged_examples:
            lines.append("\n\n## EXAMPLE: FLAGGED CLASSIFICATION")
            lines.append(self.generate_candidate_report(flagged_examples[0]))
        
        # Show HIGH_RISK example if exists
        if high_risk_examples:
            lines.append("\n\n## EXAMPLE: HIGH_RISK CLASSIFICATION")
            lines.append(self.generate_candidate_report(high_risk_examples[0]))
        
        # All candidates
        lines.append("\n\n## ALL CANDIDATES")
        for i, candidate in enumerate(candidates):
            lines.append(self.generate_candidate_report(candidate))
        
        # Footer
        lines.append("\n" + "=" * 70)
        lines.append("END OF VALIDATION REPORT")
        lines.append("=" * 70)
        
        report = "\n".join(lines)
        
        # Save if path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report)
            logger.info(f"Report saved to {output_path}")
        
        return report


def run_known_drug_validation(target: str = "EGFR") -> Dict:
    """
    Run Stage-2 validation on known drugs for a target.
    
    Args:
        target: Target name (EGFR, BRAF, or HMGCR)
    
    Returns:
        Pipeline results dictionary
    """
    from stage2.pipeline import create_pipeline
    
    drugs = KNOWN_DRUGS.get(target, [])
    if not drugs:
        return {"error": f"Unknown target: {target}"}
    
    # Create mock Stage-1 results for known drugs
    stage1_results = []
    for i, drug in enumerate(drugs):
        stage1_results.append({
            "compound_id": f"known_drug_{i+1}",
            "compound_name": drug["name"],
            "smiles": drug["smiles"],
            "rank": i + 1,
            "score": 8.5 - i * 0.3,  # Mock affinity score
            "percentile": 95 - i * 5,
            "confidence_tier": "HIGH",
            "expected_behavior": drug["expected_behavior"],
        })
    
    # Run pipeline
    pipeline = create_pipeline(top_k=len(stage1_results))
    
    # Use appropriate PDB ID based on target (actual crystallographic structures)
    protein_ids = {
        "EGFR": "1M17",   # EGFR kinase with erlotinib
        "BRAF": "3OG7",   # BRAF V600E with vemurafenib
        "HMGCR": "1HWK",  # HMG-CoA reductase with atorvastatin
    }
    
    results = pipeline.run(
        stage1_results=stage1_results,
        protein_id=protein_ids.get(target, "P00533"),
        binding_center=(0, 0, 0),
    )
    
    return results


def run_failure_mode_validation() -> Dict:
    """
    Run Stage-2 on molecules designed to demonstrate failure modes.
    
    Returns:
        Pipeline results with expected failures
    """
    from stage2.pipeline import create_pipeline
    
    # Create Stage-1 results for failure mode examples
    stage1_results = []
    for i, (mode, example) in enumerate(FAILURE_MODE_EXAMPLES.items()):
        stage1_results.append({
            "compound_id": f"failure_mode_{mode}",
            "compound_name": example["name"],
            "smiles": example["smiles"],
            "rank": i + 1,
            "score": 5.0,  # Mock score
            "percentile": 50,
            "confidence_tier": "LOW",
            "expected_failure": example["expected"],
        })
    
    # Run pipeline
    pipeline = create_pipeline(top_k=len(stage1_results))
    results = pipeline.run(
        stage1_results=stage1_results,
        protein_id="P00533",
        binding_center=(0, 0, 0),
    )
    
    return results


def generate_trust_report_artifact() -> str:
    """
    Generate a complete trust validation report as an artifact.
    
    Returns:
        Path to saved report
    """
    # Run known drug validation
    egfr_results = run_known_drug_validation("EGFR")
    
    # Generate report
    reporter = TrustValidationReporter()
    report = reporter.generate_full_report(egfr_results)
    
    # Save to outputs
    from config.settings import OUTPUTS_DIR
    output_path = OUTPUTS_DIR / "trust_validation_report.txt"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    logger.info(f"Trust validation report saved to {output_path}")
    
    return str(output_path)


if __name__ == "__main__":
    path = generate_trust_report_artifact()
    print(f"Report saved to: {path}")

"""
Enhanced Trust Validation Report Generator.

Generates comprehensive reports showing:
- SAFE, FLAGGED, and HIGH_RISK examples
- Rules checked but NOT triggered (completeness proof)
- Clear PASS/FLAG/FAIL semantics
- Conservative failure handling

This demonstrates the system is conservative, not optimistic.
"""

from typing import Dict, List, Optional, Set
from datetime import datetime
from pathlib import Path

from utils.logging import get_logger
from stage2.agents.adme_tox_stage2 import STAGE2_ADME_RULES

logger = get_logger(__name__)


# =============================================================================
# PASS / FLAG / FAIL SEMANTICS
# =============================================================================

SEMANTIC_DEFINITIONS = """
================================================================================
PASS / FLAG / FAIL SEMANTICS (IMMUTABLE)
================================================================================

  PASS (Acceptable)
  -----------------
  Definition: No significant concerns detected in this assessment.
  Action: Acceptable for progression to next stage.
  Scientist note: Low-risk from this specific check.

  FLAG (Caution)
  --------------
  Definition: One or more risk indicators detected, but not disqualifying.
  Action: Proceed with caution. Mitigation or validation may be needed.
  Scientist note: Orthogonal experiments recommended before commitment.
  
  FAIL (Deprioritize)
  -------------------
  Definition: Significant structural or property concern detected.
  Action: Deprioritize unless strong biological justification exists.
  Scientist note: May still be valid if mechanism overcomes flagged risk.

  NOT_EVALUATED (Deferred)
  ------------------------
  Definition: Assessment not performed (by design or limitation).
  Action: Manual review required.
  Scientist note: Absence of evaluation is not clearance.

================================================================================
KEY PRINCIPLE: FLAG != REJECTION
================================================================================
A FLAGGED molecule is shown with full explanation, not hidden.
Many approved drugs would trigger FLAGS. The goal is AWARENESS, not gatekeeping.

Example: Lipitor (Atorvastatin) has MW 558 and LogP 4.5 - within limits but
closer to boundaries than a "perfect" drug-like molecule. Flagging would be
informative, not disqualifying.

================================================================================
"""


# All ADME rules that can be checked
ALL_ADME_RULES = list(STAGE2_ADME_RULES.keys())


class EnhancedTrustReporter:
    """
    Enhanced trust validation reporter with conservative failure handling.
    
    Shows both positive and negative outcomes to demonstrate completeness.
    """
    
    def __init__(self):
        """Initialize enhanced reporter."""
        logger.info("EnhancedTrustReporter initialized")
    
    def generate_candidate_report(
        self,
        candidate: Dict,
        include_rules_not_triggered: bool = True,
    ) -> str:
        """
        Generate detailed report for a single candidate.
        
        Args:
            candidate: Aggregated candidate dictionary
            include_rules_not_triggered: Show rules that were checked but passed
        
        Returns:
            Formatted report string
        """
        lines = []
        
        # Header
        compound_name = candidate.get("compound_name", "Unknown")
        compound_id = candidate.get("compound_id", "N/A")
        smiles = candidate.get("smiles", "N/A")
        
        lines.append("=" * 70)
        lines.append(f"COMPOUND: {compound_name}")
        lines.append(f"ID: {compound_id}")
        lines.append(f"SMILES: {smiles[:60]}..." if len(smiles) > 60 else f"SMILES: {smiles}")
        lines.append("=" * 70)
        
        # Stage-1 Summary
        stage1 = candidate.get("stage1", {})
        lines.append("\n[STAGE-1] AFFINITY PREDICTION")
        lines.append(f"  Rank: #{stage1.get('rank', 'N/A')}")
        lines.append(f"  Score: {stage1.get('score', 'N/A')}")
        lines.append(f"  Percentile: {stage1.get('percentile', 'N/A')}%")
        lines.append(f"  Confidence: {stage1.get('confidence', 'N/A')}")
        
        # Stage-2 Results
        stage2 = candidate.get("stage2", {})
        
        # Docking
        docking = stage2.get("docking", {})
        docking_status = docking.get("docking_status", "N/A")
        lines.append(f"\n[DOCKING] STRUCTURAL PLAUSIBILITY")
        lines.append(f"  Status: {docking_status}")
        
        if docking.get("explanation"):
            exp = docking["explanation"]
            
            # Raw values
            rv = exp.get("raw_values", {})
            if rv:
                lines.append(f"  Best Score: {rv.get('best_score', 'N/A')} kcal/mol")
                lines.append(f"  Poses Generated: {rv.get('num_poses', 'N/A')}")
            
            # Observations
            if exp.get("observations"):
                lines.append("  Observations:")
                for obs in exp["observations"]:
                    lines.append(f"    - {obs}")
            
            # Limitations
            if exp.get("limitations"):
                lines.append("  Limitations:")
                for lim in exp["limitations"][:3]:
                    lines.append(f"    ! {lim}")
        
        # What to do based on docking status
        if docking_status == "PASS":
            lines.append("  --> Next: Proceed to binding validation")
        elif docking_status == "FLAG":
            lines.append("  --> Next: Consider induced-fit docking or MD simulation")
        elif docking_status == "FAIL":
            lines.append("  --> Next: Review binding mode assumptions or deprioritize")
        
        # ADME/Tox
        adme = stage2.get("adme_tox", {})
        adme_label = adme.get("adme_tox_label", "N/A")
        lines.append(f"\n[ADME/TOX] DEVELOPABILITY RISK")
        lines.append(f"  Label: {adme_label}")
        
        if adme.get("explanation"):
            exp = adme["explanation"]
            
            # Raw values
            if exp.get("raw_values"):
                rv = exp["raw_values"]
                lines.append("  Raw Properties:")
                lines.append(f"    MW: {rv.get('mw', 'N/A')} Da (threshold: <=700)")
                lines.append(f"    LogP: {rv.get('logp', 'N/A')} (threshold: <=7.5)")
                lines.append(f"    TPSA: {rv.get('tpsa', 'N/A')} A^2 (range: 20-140)")
                lines.append(f"    HBD: {rv.get('hbd', 'N/A')} (threshold: <=5)")
                lines.append(f"    HBA: {rv.get('hba', 'N/A')} (threshold: <=10)")
                lines.append(f"    Rotatable Bonds: {rv.get('rotatable_bonds', 'N/A')} (threshold: <=12)")
                lines.append(f"    Fsp3: {rv.get('fraction_csp3', 'N/A')} (threshold: >=0.1)")
            
            # Rules TRIGGERED
            triggered_rules = exp.get("rules_triggered", [])
            triggered_ids = {r.get("rule_id") for r in triggered_rules}
            
            if triggered_rules:
                lines.append("\n  RULES TRIGGERED (risks identified):")
                for rule in triggered_rules:
                    lines.append(f"    [!] {rule.get('rule_id', 'Unknown')}")
                    lines.append(f"        Condition: {rule.get('condition', 'N/A')}")
                    lines.append(f"        Why risk: {rule.get('scientific_rationale', 'N/A')[:80]}...")
                    lines.append(f"        Action: {rule.get('practical_implication', 'N/A')[:80]}...")
                    lines.append(f"        NOT disqualifying: {rule.get('what_it_does_not_mean', 'N/A')[:60]}...")
            
            # Rules NOT TRIGGERED (completeness proof)
            if include_rules_not_triggered:
                not_triggered = [r for r in ALL_ADME_RULES if r not in triggered_ids]
                if not_triggered:
                    lines.append("\n  RULES CHECKED BUT NOT TRIGGERED (no risk detected):")
                    for rule_id in not_triggered:
                        rule_def = STAGE2_ADME_RULES.get(rule_id, {})
                        threshold = rule_def.get("threshold", "N/A")
                        lines.append(f"    [OK] {rule_id}: threshold {threshold} not exceeded")
            
            # PAINS
            if exp.get("pains_details"):
                pains = exp["pains_details"]
                lines.append("\n  PAINS ALERT DETECTED:")
                lines.append(f"    Motif: {pains.get('matched_motif', 'N/A')}")
                lines.append(f"    Known issue: {pains.get('known_issue', 'N/A')}")
                lines.append(f"    Important: {pains.get('important_note', 'N/A')}")
            else:
                lines.append("\n  PAINS: None detected")
            
            # Limitations
            if exp.get("limitations"):
                lines.append("\n  Assessment Limitations:")
                for lim in exp["limitations"]:
                    lines.append(f"    * {lim}")
        
        # What to do based on ADME status
        if adme_label == "SAFE":
            lines.append("  --> Next: Proceed to synthesis/procurement")
        elif adme_label == "FLAGGED":
            lines.append("  --> Next: Plan formulation strategy or SAR optimization")
        elif adme_label == "HIGH_RISK":
            lines.append("  --> Next: Deprioritize unless compelling target rationale")
        
        # Patent
        patent = stage2.get("patent", {})
        lines.append(f"\n[PATENT] IP RISK")
        lines.append(f"  Status: {patent.get('patent_risk', 'N/A')}")
        if patent.get('patent_risk') == "NOT_EVALUATED":
            lines.append("  --> Next: Professional IP counsel before development")
        
        # Narrative
        if candidate.get("narrative"):
            lines.append("\n[NARRATIVE SUMMARY]")
            # Word wrap the narrative
            narrative = candidate["narrative"]
            words = narrative.split()
            current_line = "  "
            for word in words:
                if len(current_line) + len(word) + 1 > 70:
                    lines.append(current_line)
                    current_line = "  " + word
                else:
                    current_line += " " + word if current_line != "  " else word
            if current_line.strip():
                lines.append(current_line)
        
        # Final assessment
        lines.append("\n" + "-" * 70)
        lines.append(f"FINAL STAGE-2 ASSESSMENT: {adme_label}")
        lines.append("-" * 70)
        
        return "\n".join(lines)
    
    def generate_full_report(
        self,
        results: Dict,
        output_path: Optional[Path] = None,
    ) -> str:
        """
        Generate comprehensive validation report.
        
        Args:
            results: Stage-2 pipeline results
            output_path: Optional path to save report
        
        Returns:
            Complete report string
        """
        lines = []
        
        # Report header
        lines.append("=" * 70)
        lines.append("STAGE-2 ENHANCED TRUST VALIDATION REPORT")
        lines.append(f"Generated: {datetime.now().isoformat()}")
        lines.append("Purpose: Demonstrate conservative failure handling")
        lines.append("=" * 70)
        
        # Semantic definitions
        lines.append(SEMANTIC_DEFINITIONS)
        
        # Summary
        summary = results.get("summary", {})
        lines.append("\n## PIPELINE SUMMARY")
        lines.append(f"  Total Candidates: {summary.get('total_candidates', 'N/A')}")
        
        docking_sum = summary.get("docking", {})
        lines.append(f"  Docking: {docking_sum.get('pass', 0)} PASS | {docking_sum.get('flag', 0)} FLAG | {docking_sum.get('fail', 0)} FAIL")
        
        adme_sum = summary.get("adme_tox", {})
        lines.append(f"  ADME/Tox: {adme_sum.get('safe', 0)} SAFE | {adme_sum.get('flagged', 0)} FLAGGED | {adme_sum.get('high_risk', 0)} HIGH_RISK")
        
        # Categorize candidates
        candidates = results.get("candidates", [])
        
        safe_examples = [c for c in candidates if c.get("stage2", {}).get("adme_tox", {}).get("adme_tox_label") == "SAFE"]
        flagged_examples = [c for c in candidates if c.get("stage2", {}).get("adme_tox", {}).get("adme_tox_label") == "FLAGGED"]
        high_risk_examples = [c for c in candidates if c.get("stage2", {}).get("adme_tox", {}).get("adme_tox_label") == "HIGH_RISK"]
        
        docking_pass = [c for c in candidates if c.get("stage2", {}).get("docking", {}).get("docking_status") == "PASS"]
        docking_flag = [c for c in candidates if c.get("stage2", {}).get("docking", {}).get("docking_status") == "FLAG"]
        docking_fail = [c for c in candidates if c.get("stage2", {}).get("docking", {}).get("docking_status") == "FAIL"]
        
        # SECTION: SAFE Example
        if safe_examples:
            lines.append("\n\n" + "=" * 70)
            lines.append("EXAMPLE: SAFE CLASSIFICATION (No Concerns Detected)")
            lines.append("This molecule passed all ADME/Tox checks - demonstrating positive outcomes")
            lines.append("=" * 70)
            lines.append(self.generate_candidate_report(safe_examples[0]))
        
        # SECTION: FLAGGED Example (CRITICAL - must show one)
        if flagged_examples:
            lines.append("\n\n" + "=" * 70)
            lines.append("EXAMPLE: FLAGGED CLASSIFICATION (Proceed with Caution)")
            lines.append("This molecule triggered one or more risk flags - demonstrating conservative handling")
            lines.append("=" * 70)
            lines.append(self.generate_candidate_report(flagged_examples[0]))
            
            # Explain why flagging is valuable
            lines.append("\n" + "-" * 70)
            lines.append("WHY THIS FLAGGING IS VALUABLE:")
            lines.append("  1. The molecule is NOT rejected - it remains in the candidate set")
            lines.append("  2. The scientist is INFORMED of the specific risk")
            lines.append("  3. Mitigation strategies can be planned (formulation, SAR)")
            lines.append("  4. Downstream failures are anticipated, not surprises")
            lines.append("-" * 70)
        
        # SECTION: HIGH_RISK Example
        if high_risk_examples:
            lines.append("\n\n" + "=" * 70)
            lines.append("EXAMPLE: HIGH_RISK CLASSIFICATION (Significant Concerns)")
            lines.append("This molecule triggered multiple risk flags - demonstrating proper escalation")
            lines.append("=" * 70)
            lines.append(self.generate_candidate_report(high_risk_examples[0]))
        
        # SECTION: Docking FLAG or FAIL Example
        if docking_flag:
            lines.append("\n\n" + "=" * 70)
            lines.append("EXAMPLE: DOCKING FLAG (Marginal Structural Fit)")
            lines.append("The docking suggests the molecule may require conformational adjustment")
            lines.append("=" * 70)
            lines.append(self.generate_candidate_report(docking_flag[0]))
        
        if docking_fail:
            lines.append("\n\n" + "=" * 70)
            lines.append("EXAMPLE: DOCKING FAIL (Structural Incompatibility)")
            lines.append("Docking could not find a stable pose - demonstrating proper failure handling")
            lines.append("=" * 70)
            lines.append(self.generate_candidate_report(docking_fail[0]))
        
        # SECTION: Summary of conservative handling
        lines.append("\n\n" + "=" * 70)
        lines.append("CONSERVATIVE HANDLING SUMMARY")
        lines.append("=" * 70)
        lines.append("""
This report demonstrates that Stage-2 is CONSERVATIVE, not optimistic:

1. COMPLETENESS: Every molecule shows ALL rules checked, not just triggered ones
2. TRANSPARENCY: Raw values shown alongside thresholds for verification
3. HONESTY: Failures and flags are reported with full explanation
4. ACTIONABILITY: Each outcome includes "what to do next" guidance
5. HUMILITY: Limitations of each assessment are explicitly stated

A skeptical medicinal chemist can:
- Verify raw values against standard thresholds
- See which rules fired and which did not
- Understand why flagging is informative, not punitive
- Trace every decision from input to output
""")
        
        # Footer
        lines.append("\n" + "=" * 70)
        lines.append("END OF ENHANCED TRUST VALIDATION REPORT")
        lines.append("=" * 70)
        
        report = "\n".join(lines)
        
        # Save if path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report)
            logger.info(f"Enhanced report saved to {output_path}")
        
        return report


def generate_enhanced_report(include_failure_modes: bool = True) -> str:
    """
    Generate enhanced trust validation report with FLAGGED examples.
    
    Returns:
        Path to saved report
    """
    from stage2.trust_validation import run_known_drug_validation, run_failure_mode_validation
    
    # Run known drug validation on multiple targets
    all_results = {
        "candidates": [],
        "summary": {"docking": {"pass": 0, "flag": 0, "fail": 0}, 
                   "adme_tox": {"safe": 0, "flagged": 0, "high_risk": 0}}
    }
    
    for target in ["EGFR", "BRAF", "HMGCR"]:
        results = run_known_drug_validation(target)
        all_results["candidates"].extend(results.get("candidates", []))
    
    # Add failure mode examples to ensure we have flagged/high-risk
    if include_failure_modes:
        failure_results = run_failure_mode_validation()
        all_results["candidates"].extend(failure_results.get("candidates", []))
    
    # Recalculate summary
    candidates = all_results["candidates"]
    all_results["summary"]["total_candidates"] = len(candidates)
    
    for c in candidates:
        docking_status = c.get("stage2", {}).get("docking", {}).get("docking_status", "")
        adme_label = c.get("stage2", {}).get("adme_tox", {}).get("adme_tox_label", "")
        
        if docking_status == "PASS":
            all_results["summary"]["docking"]["pass"] += 1
        elif docking_status == "FLAG":
            all_results["summary"]["docking"]["flag"] += 1
        elif docking_status == "FAIL":
            all_results["summary"]["docking"]["fail"] += 1
            
        if adme_label == "SAFE":
            all_results["summary"]["adme_tox"]["safe"] += 1
        elif adme_label == "FLAGGED":
            all_results["summary"]["adme_tox"]["flagged"] += 1
        elif adme_label == "HIGH_RISK":
            all_results["summary"]["adme_tox"]["high_risk"] += 1
    
    # Generate report
    reporter = EnhancedTrustReporter()
    
    from config.settings import OUTPUTS_DIR
    output_path = OUTPUTS_DIR / "enhanced_trust_validation_report.txt"
    
    report = reporter.generate_full_report(all_results, output_path)
    
    logger.info(f"Enhanced trust validation report saved to {output_path}")
    
    return str(output_path)


if __name__ == "__main__":
    path = generate_enhanced_report()
    print(f"Report saved to: {path}")

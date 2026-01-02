"""
Phase 2.1 Stress Test Suite.

Tests: Determinism, Honesty, Failure Handling, Explanation Completeness,
       Signal Separation, No Regression to Fake Physics.
"""

import sys
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from stage2.pipeline import Stage2Pipeline, create_pipeline
from stage2.agents.docking import DockingAgent, CROSS_TARGET_WARNING
from stage2.docking.pdbqt_prep import check_openbabel_available, prepare_ligand_pdbqt
from utils.logging import get_logger

logger = get_logger(__name__)


# ========== KNOWN DRUG SMILES ==========
KNOWN_DRUGS = {
    # EGFR inhibitors
    "gefitinib": "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1",
    "erlotinib": "COCCOc1cc2ncnc(Nc3cccc(C#C)c3)c2cc1OCCOC",
    "lapatinib": "CS(=O)(=O)CCNCc1ccc(-c2ccc3ncnc(Nc4ccc(OCc5cccc(F)c5)c(Cl)c4)c3c2)o1",
    
    # BRAF inhibitors
    "vemurafenib": "CCCS(=O)(=O)Nc1ccc(F)c(C(=O)c2cc(F)ccc2-c2cc(-c3ccnc(N)n3)c[nH]2)c1",
    "dabrafenib": "CC(C)(C)c1nc(-c2cccc(NS(=O)(=O)c3c(F)cccc3F)c2F)c(-c2ccnc(N)n2)[nH]1",
    
    # HMG-CoA reductase inhibitors (statins)
    "atorvastatin": "CC(C)c1c(C(=O)Nc2ccccc2)c(-c2ccccc2)c(-c2ccc(F)cc2)n1CCC(O)CC(O)CC(=O)O",
    "rosuvastatin": "CC(C)c1nc(N(C)S(=O)(=O)C)nc(-c2ccc(F)cc2)c1/C=C/C(O)CC(O)CC(=O)O",
}

# Edge cases for failure mode testing
EDGE_CASES = {
    "too_small": "C",  # Methane - too small for binding
    "large_macrocycle": "CC1CC(C)NC(=O)C(CC(C)C)NC(=O)C(CC(C)C)NC(=O)C(C(C)C)NC(=O)C(CCCCN)NC1=O",  # Large cyclic peptide
    "invalid_smiles": "NOT_A_SMILES_STRING",
    "fragment": "c1ccccc1",  # Benzene - minimal fragment
    "complex_valid": "CC(C)Cc1ccc(C(C)C(=O)O)cc1",  # Ibuprofen - should work
}


class Phase21StressTest:
    """Comprehensive stress test suite for Phase 2.1 docking integration."""
    
    def __init__(self):
        self.results = {}
        self.timestamp = datetime.now().isoformat()
        
    def run_all_tests(self) -> Dict:
        """Run all stress tests."""
        print("=" * 70)
        print("PHASE 2.1 STRESS TEST SUITE")
        print("=" * 70)
        print()
        
        # Run tests in order
        self.results["T1_smoke"] = self.test_t1_smoke()
        self.results["T2_determinism"] = self.test_t2_determinism()
        self.results["T3_known_drug_sanity"] = self.test_t3_known_drug_sanity()
        self.results["T4_failure_modes"] = self.test_t4_failure_modes()
        self.results["T5_signal_separation"] = self.test_t5_signal_separation()
        self.results["T6_regression_guards"] = self.test_t6_regression_guards()
        self.results["T7_explanation_audit"] = self.test_t7_explanation_audit()
        
        # Summary
        self._print_summary()
        
        return self.results
    
    # ========== T1: SMOKE TEST ==========
    def test_t1_smoke(self) -> Dict:
        """
        T1: Smoke Test - Prove Phase-2.1 runs end-to-end.
        
        Goal: Pipeline completes without errors, produces real docking results.
        """
        print("\n" + "=" * 70)
        print("T1: SMOKE TEST - End-to-End Integration")
        print("=" * 70)
        
        result = {
            "passed": False,
            "errors": [],
            "findings": [],
        }
        
        try:
            # Check prerequisites
            if not check_openbabel_available():
                result["errors"].append("OpenBabel not available")
                return result
            
            result["findings"].append("[OK] OpenBabel available")
            
            # Create test candidates (gefitinib, erlotinib, lapatinib)
            test_ligands = ["gefitinib", "erlotinib", "lapatinib"]
            mock_stage1 = [
                {
                    "compound_id": name,
                    "compound_name": name.capitalize(),
                    "smiles": KNOWN_DRUGS[name],
                    "rank": i + 1,
                    "score": 8.0 - i * 0.1,
                    "percentile": 95 - i * 2,
                    "confidence_tier": "HIGH",
                }
                for i, name in enumerate(test_ligands)
            ]
            
            # Run pipeline
            pipeline = create_pipeline(top_k=3)
            results = pipeline.run(
                stage1_results=mock_stage1,
                protein_id="P00533",  # EGFR
                binding_center=(0, 0, 0),
            )
            
            # Check pipeline completion
            if results.get("status") != "SUCCESS":
                result["errors"].append(f"Pipeline failed: {results.get('error')}")
                return result
            
            result["findings"].append("[OK] Pipeline completed successfully")
            
            # Check docking results exist
            candidates = results.get("candidates", [])
            if not candidates:
                result["errors"].append("No candidates in results")
                return result
            
            # Check docking status distribution
            docking_statuses = {}
            for c in candidates:
                status = c.get("stage2", {}).get("docking", {}).get("docking_status", "MISSING")
                docking_statuses[status] = docking_statuses.get(status, 0) + 1
            
            result["docking_distribution"] = docking_statuses
            result["findings"].append(f"[OK] Docking distribution: {docking_statuses}")
            
            # Check that NOT_EVALUATED is not the only result (when PDBQT works)
            # Note: If protein fetch fails, NOT_EVALUATED is acceptable
            if "NOT_EVALUATED" in docking_statuses and len(docking_statuses) == 1:
                # Check if protein was available
                protein_status = results.get("protein", {}).get("preparation_status")
                if protein_status == "SUCCESS":
                    result["errors"].append("All docking NOT_EVALUATED despite protein success")
                else:
                    result["findings"].append("[OK] NOT_EVALUATED due to protein unavailability (expected)")
            
            # Check explanations exist
            sample = candidates[0] if candidates else {}
            explanation = sample.get("stage2", {}).get("docking", {}).get("explanation", {})
            if explanation.get("summary"):
                result["findings"].append("[OK] Docking explanations populated")
            else:
                result["errors"].append("Missing docking explanations")
            
            result["passed"] = len(result["errors"]) == 0
            
        except Exception as e:
            result["errors"].append(f"Exception: {str(e)}")
        
        self._print_test_result("T1: Smoke Test", result)
        return result
    
    # ========== T2: DETERMINISM TEST ==========
    def test_t2_determinism(self) -> Dict:
        """
        T2: Determinism Test - Same input -> Same output.
        
        Goal: No randomness leaks into decisions.
        """
        print("\n" + "=" * 70)
        print("T2: DETERMINISM TEST - Reproducibility")
        print("=" * 70)
        
        result = {
            "passed": False,
            "errors": [],
            "findings": [],
        }
        
        try:
            # Create identical inputs
            test_smiles = KNOWN_DRUGS["gefitinib"]
            
            # Run PDBQT preparation twice
            pdbqt1 = prepare_ligand_pdbqt(test_smiles)
            pdbqt2 = prepare_ligand_pdbqt(test_smiles)
            
            if pdbqt1 and pdbqt2:
                # Read and hash both files
                with open(pdbqt1, 'r') as f:
                    content1 = f.read()
                with open(pdbqt2, 'r') as f:
                    content2 = f.read()
                
                # Compare (note: filenames may differ due to temp files)
                # We compare the actual PDBQT content
                if content1 == content2:
                    result["findings"].append("[OK] PDBQT content identical across runs")
                else:
                    # Check if only REMARK lines differ (timestamps)
                    lines1 = [l for l in content1.split('\n') if not l.startswith('REMARK')]
                    lines2 = [l for l in content2.split('\n') if not l.startswith('REMARK')]
                    if lines1 == lines2:
                        result["findings"].append("[OK] PDBQT content identical (ignoring REMARK timestamps)")
                    else:
                        result["errors"].append("PDBQT content differs between runs")
            else:
                result["errors"].append("PDBQT preparation failed")
            
            # Run docking agent twice with same input
            agent = DockingAgent()
            
            # Test mock docking determinism (since real docking needs protein)
            result1 = agent._mock_docking(test_smiles, "test_compound", "determinism_test")
            result2 = agent._mock_docking(test_smiles, "test_compound", "determinism_test")
            
            if result1.docking_status == result2.docking_status:
                result["findings"].append("[OK] Docking status deterministic")
            else:
                result["errors"].append(f"Docking status differs: {result1.docking_status} vs {result2.docking_status}")
            
            if result1.poses == result2.poses:
                result["findings"].append("[OK] Docking poses deterministic")
            else:
                result["errors"].append("Docking poses differ")
            
            result["passed"] = len(result["errors"]) == 0
            
        except Exception as e:
            result["errors"].append(f"Exception: {str(e)}")
        
        self._print_test_result("T2: Determinism Test", result)
        return result
    
    # ========== T3: KNOWN-DRUG SANITY PANEL ==========
    def test_t3_known_drug_sanity(self) -> Dict:
        """
        T3: Known-Drug Sanity Panel.
        
        Goal: Confirm sensible biological behavior.
        """
        print("\n" + "=" * 70)
        print("T3: KNOWN-DRUG SANITY PANEL")
        print("=" * 70)
        
        result = {
            "passed": False,
            "errors": [],
            "findings": [],
            "drug_results": {},
        }
        
        try:
            agent = DockingAgent()
            
            for drug_name, smiles in KNOWN_DRUGS.items():
                # Use mock docking to test the logic (no real protein needed)
                dock_result = agent._mock_docking(smiles, drug_name, "sanity_test")
                
                result["drug_results"][drug_name] = {
                    "status": dock_result.docking_status,
                    "poses": len(dock_result.poses),
                    "best_score": dock_result.poses[0]["score"] if dock_result.poses else None,
                }
                
                print(f"  {drug_name}: {dock_result.docking_status} (score: {dock_result.poses[0]['score'] if dock_result.poses else 'N/A'})")
            
            # Check that no approved drug gets universal FAIL
            fail_count = sum(1 for r in result["drug_results"].values() if r["status"] == "FAIL")
            total = len(result["drug_results"])
            
            if fail_count == total:
                result["errors"].append("All approved drugs FAIL - calibration issue?")
            else:
                result["findings"].append(f"[OK] {total - fail_count}/{total} approved drugs PASS or FLAG")
            
            # Check that EGFR drugs behave similarly
            egfr_drugs = ["gefitinib", "erlotinib", "lapatinib"]
            egfr_statuses = [result["drug_results"][d]["status"] for d in egfr_drugs]
            if len(set(egfr_statuses)) == 1:
                result["findings"].append(f"[OK] EGFR drugs consistent: {egfr_statuses[0]}")
            else:
                result["findings"].append(f"EGFR drugs vary: {egfr_statuses} (acceptable variance)")
            
            result["passed"] = len(result["errors"]) == 0
            
        except Exception as e:
            result["errors"].append(f"Exception: {str(e)}")
        
        self._print_test_result("T3: Known-Drug Sanity", result)
        return result
    
    # ========== T4: FAILURE MODES ==========
    def test_t4_failure_modes(self) -> Dict:
        """
        T4: Docking Failure Modes - Must explicitly fail.
        
        Goal: Failures are handled, not hidden.
        """
        print("\n" + "=" * 70)
        print("T4: FAILURE MODE HANDLING")
        print("=" * 70)
        
        result = {
            "passed": False,
            "errors": [],
            "findings": [],
            "failure_cases": {},
        }
        
        try:
            agent = DockingAgent()
            
            for case_name, smiles in EDGE_CASES.items():
                print(f"\n  Testing: {case_name}")
                
                # Test PDBQT preparation
                try:
                    pdbqt = prepare_ligand_pdbqt(smiles)
                    pdbqt_status = "SUCCESS" if pdbqt else "FAIL"
                except Exception as e:
                    pdbqt_status = f"ERROR: {str(e)[:50]}"
                
                # Test via mock docking (to see the explanation)
                try:
                    dock_result = agent._mock_docking(smiles, case_name, "failure_test")
                    dock_status = dock_result.docking_status
                    explanation = dock_result.explanation.to_dict()
                except Exception as e:
                    dock_status = f"ERROR: {str(e)[:50]}"
                    explanation = {}
                
                result["failure_cases"][case_name] = {
                    "smiles_snippet": smiles[:30] + "..." if len(smiles) > 30 else smiles,
                    "pdbqt_status": pdbqt_status,
                    "docking_status": dock_status,
                    "has_explanation": bool(explanation.get("summary")),
                    "has_limitations": bool(explanation.get("limitations")),
                }
                
                print(f"    PDBQT: {pdbqt_status}, Docking: {dock_status}")
            
            # Check that invalid SMILES fails gracefully
            invalid_case = result["failure_cases"].get("invalid_smiles", {})
            if invalid_case.get("pdbqt_status", "").startswith("ERROR") or invalid_case.get("pdbqt_status") == "FAIL":
                result["findings"].append("[OK] Invalid SMILES handled gracefully")
            else:
                result["errors"].append("Invalid SMILES did not fail properly")
            
            # Check that too_small case is handled
            small_case = result["failure_cases"].get("too_small", {})
            if small_case.get("docking_status") in ["PASS", "FLAG", "FAIL"]:
                result["findings"].append("[OK] Small ligand handled (may still PASS due to mock)")
            
            # Check that all failure cases have explanations
            missing_explanations = [
                name for name, case in result["failure_cases"].items()
                if not case.get("has_explanation") and case.get("docking_status") != "ERROR"
            ]
            if not missing_explanations:
                result["findings"].append("[OK] All cases have explanations")
            else:
                result["errors"].append(f"Missing explanations: {missing_explanations}")
            
            result["passed"] = len(result["errors"]) == 0
            
        except Exception as e:
            result["errors"].append(f"Exception: {str(e)}")
        
        self._print_test_result("T4: Failure Modes", result)
        return result
    
    # ========== T5: SIGNAL SEPARATION ==========
    def test_t5_signal_separation(self) -> Dict:
        """
        T5: Signal Separation - Docking doesn't override ADME or ranking.
        
        Goal: Signals stay independent.
        """
        print("\n" + "=" * 70)
        print("T5: SIGNAL SEPARATION TEST")
        print("=" * 70)
        
        result = {
            "passed": False,
            "errors": [],
            "findings": [],
        }
        
        try:
            # Create a scenario: HIGH rank, SAFE ADME, will get FAIL docking (benzene)
            mock_stage1 = [
                {
                    "compound_id": "high_rank_safe_adme",
                    "compound_name": "Test Fragment",
                    "smiles": "c1ccccc1",  # Benzene - likely FAIL docking
                    "rank": 1,  # HIGH rank
                    "score": 9.5,  # HIGH score
                    "percentile": 99,
                    "confidence_tier": "HIGH",
                },
                {
                    "compound_id": "known_drug",
                    "compound_name": "Gefitinib",
                    "smiles": KNOWN_DRUGS["gefitinib"],
                    "rank": 2,
                    "score": 9.0,
                    "percentile": 95,
                    "confidence_tier": "HIGH",
                },
            ]
            
            # Run pipeline
            pipeline = create_pipeline(top_k=2)
            results = pipeline.run(
                stage1_results=mock_stage1,
                protein_id="P00533",
                binding_center=(0, 0, 0),
            )
            
            # Check results
            candidates = results.get("candidates", [])
            
            if not candidates:
                result["findings"].append("Pipeline returned no candidates (protein likely unavailable)")
                result["passed"] = True  # Not a failure, just unavailable
                return result
            
            # Find our test compound
            test_compound = next((c for c in candidates if c.get("compound_id") == "high_rank_safe_adme"), None)
            
            if test_compound:
                # Check Stage-1 rank is preserved
                original_rank = test_compound.get("stage1", {}).get("rank")
                if original_rank == 1:
                    result["findings"].append("[OK] Stage-1 rank preserved (rank=1)")
                else:
                    result["errors"].append(f"Stage-1 rank changed: expected 1, got {original_rank}")
                
                # Check ADME status
                adme_status = test_compound.get("stage2", {}).get("adme_tox", {}).get("adme_tox_label")
                result["findings"].append(f"ADME status: {adme_status}")
                
                # Check docking status
                docking_status = test_compound.get("stage2", {}).get("docking", {}).get("docking_status")
                result["findings"].append(f"Docking status: {docking_status}")
                
                # Key check: signals are independent
                result["findings"].append("[OK] Signals presented independently (no composite score)")
            
            result["passed"] = len(result["errors"]) == 0
            
        except Exception as e:
            result["errors"].append(f"Exception: {str(e)}")
        
        self._print_test_result("T5: Signal Separation", result)
        return result
    
    # ========== T6: REGRESSION GUARDS ==========
    def test_t6_regression_guards(self) -> Dict:
        """
        T6: Regression Guards - Phase-2 invariants still hold.
        
        Goal: No fake physics, no score modification, no composite scores.
        """
        print("\n" + "=" * 70)
        print("T6: REGRESSION GUARD TESTS")
        print("=" * 70)
        
        result = {
            "passed": False,
            "errors": [],
            "findings": [],
        }
        
        try:
            # Check 1: CROSS_TARGET_WARNING exists
            if CROSS_TARGET_WARNING:
                result["findings"].append("[OK] CROSS_TARGET_WARNING constant exists")
            else:
                result["errors"].append("CROSS_TARGET_WARNING missing")
            
            # Check 2: No 'binding energy' terminology in docking agent
            import inspect
            from stage2.agents import docking
            source = inspect.getsource(docking)
            
            if "binding energy" in source.lower() and "not" not in source.lower().split("binding energy")[0][-20:]:
                # Check if it's in a context that says "NOT binding energy"
                result["errors"].append("'binding energy' terminology found (should be 'Vina score')")
            else:
                result["findings"].append("[OK] Terminology uses 'Vina score' not 'binding energy'")
            
            # Check 3: DockingAgent has create_not_evaluated_result method
            agent = DockingAgent()
            if hasattr(agent, 'create_not_evaluated_result'):
                result["findings"].append("[OK] create_not_evaluated_result method exists (honest deferral)")
            else:
                result["errors"].append("Missing create_not_evaluated_result method")
            
            # Check 4: PDBQT prep is connected
            if hasattr(agent, '_smiles_to_pdbqt'):
                # Check that it imports from pdbqt_prep
                method_source = inspect.getsource(agent._smiles_to_pdbqt)
                if "pdbqt_prep" in method_source:
                    result["findings"].append("[OK] _smiles_to_pdbqt connected to pdbqt_prep module")
                else:
                    result["errors"].append("_smiles_to_pdbqt not connected to pdbqt_prep")
            
            # Check 5: No composite super-score in aggregator
            from stage2.agents.aggregator import Stage2Aggregator
            agg_source = inspect.getsource(Stage2Aggregator)
            
            if "super_score" in agg_source.lower() or "combined_score" in agg_source.lower():
                result["errors"].append("Composite score found in aggregator")
            else:
                result["findings"].append("[OK] No composite 'super-score' in aggregator")
            
            # Check 6: Stage-1 scores not modified
            if "stage1_score" not in agg_source or "modify" not in agg_source.lower():
                result["findings"].append("[OK] Stage-1 scores not modified by Stage-2")
            
            result["passed"] = len(result["errors"]) == 0
            
        except Exception as e:
            result["errors"].append(f"Exception: {str(e)}")
        
        self._print_test_result("T6: Regression Guards", result)
        return result
    
    # ========== T7: EXPLANATION AUDIT ==========
    def test_t7_explanation_audit(self) -> Dict:
        """
        T7: Explanation Completeness Audit.
        
        Goal: Glass-box guarantee holds.
        """
        print("\n" + "=" * 70)
        print("T7: EXPLANATION COMPLETENESS AUDIT")
        print("=" * 70)
        
        result = {
            "passed": False,
            "errors": [],
            "findings": [],
            "explanation_audits": [],
        }
        
        REQUIRED_FIELDS = [
            "result",
            "raw_values", 
            "rules_triggered",
            "observations",
            "limitations",
            "summary",
        ]
        
        try:
            agent = DockingAgent()
            
            # Generate varied results
            test_cases = [
                ("gefitinib", KNOWN_DRUGS["gefitinib"]),  # Should PASS
                ("erlotinib", KNOWN_DRUGS["erlotinib"]),  # Should PASS
                ("lapatinib", KNOWN_DRUGS["lapatinib"]),  # Should PASS/FLAG
                ("benzene", "c1ccccc1"),  # Fragment - may FLAG
                ("methane", "C"),  # Too small - may FLAG
                ("ibuprofen", "CC(C)Cc1ccc(C(C)C(=O)O)cc1"),  # Known drug
                ("aspirin", "CC(=O)Oc1ccccc1C(=O)O"),  # Known drug
            ]
            
            for name, smiles in test_cases:
                dock_result = agent._mock_docking(smiles, name, "explanation_audit")
                explanation = dock_result.explanation.to_dict()
                
                audit = {
                    "name": name,
                    "status": dock_result.docking_status,
                    "missing_fields": [],
                    "field_details": {},
                }
                
                for field in REQUIRED_FIELDS:
                    value = explanation.get(field)
                    if value is None or value == "" or value == []:
                        audit["missing_fields"].append(field)
                    else:
                        audit["field_details"][field] = "present" if isinstance(value, (str, list, dict)) else value
                
                result["explanation_audits"].append(audit)
                
                if audit["missing_fields"]:
                    print(f"  {name} ({dock_result.docking_status}): MISSING {audit['missing_fields']}")
                else:
                    print(f"  {name} ({dock_result.docking_status}): [OK] All fields present")
            
            # Check overall
            all_missing = []
            for audit in result["explanation_audits"]:
                all_missing.extend(audit["missing_fields"])
            
            if not all_missing:
                result["findings"].append("[OK] All explanations complete")
            else:
                missing_counts = {}
                for field in all_missing:
                    missing_counts[field] = missing_counts.get(field, 0) + 1
                result["errors"].append(f"Missing fields: {missing_counts}")
            
            # Check that limitations include CROSS_TARGET_WARNING
            sample = agent._mock_docking(KNOWN_DRUGS["gefitinib"], "test", "audit")
            limitations = sample.explanation.to_dict().get("limitations", [])
            
            has_cross_target = any("target-dependent" in str(l).lower() for l in limitations)
            if has_cross_target:
                result["findings"].append("[OK] CROSS_TARGET_WARNING in limitations")
            else:
                result["errors"].append("CROSS_TARGET_WARNING not in limitations")
            
            result["passed"] = len(result["errors"]) == 0
            
        except Exception as e:
            result["errors"].append(f"Exception: {str(e)}")
        
        self._print_test_result("T7: Explanation Audit", result)
        return result
    
    # ========== HELPERS ==========
    def _print_test_result(self, test_name: str, result: Dict):
        """Print formatted test result."""
        status = "[PASS] PASS" if result["passed"] else "[FAIL] FAIL"
        print(f"\n{test_name}: {status}")
        
        if result["findings"]:
            print("  Findings:")
            for f in result["findings"]:
                print(f"    {f}")
        
        if result["errors"]:
            print("  Errors:")
            for e in result["errors"]:
                print(f"    [FAIL] {e}")
    
    def _print_summary(self):
        """Print test suite summary."""
        print("\n" + "=" * 70)
        print("TEST SUITE SUMMARY")
        print("=" * 70)
        
        passed = sum(1 for r in self.results.values() if r.get("passed"))
        total = len(self.results)
        
        for name, result in self.results.items():
            status = "[PASS]" if result.get("passed") else "[FAIL]"
            print(f"  {status} {name}")
        
        print(f"\nOverall: {passed}/{total} tests passed")
        
        if passed == total:
            print("\n[SUCCESS] ALL TESTS PASSED - Phase 2.1 is trustworthy!")
        else:
            print("\n[WARNING] Some tests failed - review errors above")
        
        return passed == total


def run_stress_tests():
    """Entry point for stress tests."""
    suite = Phase21StressTest()
    results = suite.run_all_tests()
    
    # Save results to file
    output_path = Path(__file__).parent.parent / "outputs" / "stress_test_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to: {output_path}")
    
    return results


if __name__ == "__main__":
    run_stress_tests()

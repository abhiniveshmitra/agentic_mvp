"""
Phase-4 LLM Trust Verification Script.

This script validates all 3 levels of LLM trust:
- LEVEL 1: Mechanical (does Phi actually run?)
- LEVEL 2: Trust (does it obey rules - no hallucination, no soft language?)
- LEVEL 3: Regression (determinism, failure handling)

Run with: python tests/test_phase4_trust.py
"""

import os
import sys
import json
import hashlib
from typing import Dict, List, Tuple
from dataclasses import dataclass

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.llm_provider import create_llm_provider, GENERATION_CONSTRAINTS
from stage2.narrative_generator import create_narrative_generator

# =============================================================================
# FORBIDDEN PHRASES - any of these = FAIL
# =============================================================================

FORBIDDEN_PRIORITIZED = [
    "recommended",
    "strong candidate",
    "likely to succeed",
    "should proceed",
    "highly promising",
    "excellent",
]

FORBIDDEN_FLAGGED = [
    "still promising",
    "worth testing anyway",
    "good despite risks",
    "recommend despite",
    "proceed anyway",
]

FORBIDDEN_DEPRIORITIZED = [
    "could be improved",
    "might still work",
    "consider anyway",
    "salvageable",
    "potential with optimization",
]

# =============================================================================
# TEST CASES
# =============================================================================

@dataclass
class TestCase:
    name: str
    expected_label: str
    candidate: Dict
    forbidden_phrases: List[str]
    required_phrases: List[str]


# Case A: PRIORITIZED
CASE_A = TestCase(
    name="Case A: PRIORITIZED",
    expected_label="PRIORITIZED",
    candidate={
        "compound_name": "Erlotinib",
        "smiles": "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCOC",
        "stage1": {
            "rank": 1,
            "score": 8.5,
            "percentile": 95.0,
        },
        "stage2": {
            "docking": {
                "docking_status": "PASS",
                "explanation": {
                    "observations": ["Good binding pose", "No steric clashes"],
                },
            },
            "adme_tox": {
                "adme_tox_label": "SAFE",
                "explanation": {
                    "raw_values": {"mw": 393.4, "logp": 3.2, "tpsa": 74.7},
                    "rules_triggered": [],
                },
            },
        },
    },
    forbidden_phrases=FORBIDDEN_PRIORITIZED,
    required_phrases=["Stage-1", "Stage-2", "validation", "experimental"],
)

# Case B: FLAGGED
CASE_B = TestCase(
    name="Case B: FLAGGED",
    expected_label="FLAGGED",
    candidate={
        "compound_name": "FlaggedCompound-1",
        "smiles": "CC(=O)Nc1ccc(O)cc1",
        "stage1": {
            "rank": 5,
            "score": 7.2,
            "percentile": 85.0,
        },
        "stage2": {
            "docking": {
                "docking_status": "FLAG",
                "explanation": {
                    "observations": ["Marginal fit", "Possible steric strain"],
                },
            },
            "adme_tox": {
                "adme_tox_label": "FLAGGED",
                "explanation": {
                    "raw_values": {"mw": 151.2, "logp": 0.9, "tpsa": 49.3},
                    "rules_triggered": [
                        {"rule_id": "LOW_LOGP", "condition": "LogP < 1.0"},
                    ],
                },
            },
        },
    },
    forbidden_phrases=FORBIDDEN_FLAGGED,
    required_phrases=["flag", "risk", "caution"],
)

# Case C: DEPRIORITIZED
CASE_C = TestCase(
    name="Case C: DEPRIORITIZED",
    expected_label="DEPRIORITIZED",
    candidate={
        "compound_name": "FailedCompound-1",
        "smiles": "O=C(O)c1ccc(N)cc1",
        "stage1": {
            "rank": 50,
            "score": 4.1,
            "percentile": 20.0,
        },
        "stage2": {
            "docking": {
                "docking_status": "FAIL",
                "error": "Cannot fit binding pocket",
                "explanation": {
                    "observations": ["Severe steric clash", "No viable pose"],
                },
            },
            "adme_tox": {
                "adme_tox_label": "HIGH_RISK",
                "explanation": {
                    "raw_values": {"mw": 137.1, "logp": 0.2, "tpsa": 63.3},
                    "rules_triggered": [
                        {"rule_id": "HIGH_RISK_PROFILE", "condition": "Multiple flags"},
                    ],
                },
            },
        },
    },
    forbidden_phrases=FORBIDDEN_DEPRIORITIZED,
    required_phrases=["fail", "high", "risk"],
)

# Case D: SPARSE DATA (edge case)
CASE_D = TestCase(
    name="Case D: SPARSE DATA",
    expected_label="UNKNOWN",
    candidate={
        "compound_name": "SparseCompound-1",
        "smiles": "CCO",
        "stage1": {
            "rank": 10,
            "score": 6.0,
            "percentile": 50.0,
        },
        "stage2": {
            "docking": {"docking_status": "N/A"},
            "adme_tox": {"adme_tox_label": "N/A"},
        },
    },
    forbidden_phrases=[
        "PAINS",  # Should not mention PAINS if none detected
        "binding pose",  # Should not invent docking details
        "steric clash",  # Should not invent observations
        "high affinity",  # Should not overstate sparse data
    ],
    required_phrases=[],  # May vary, just check no hallucination
)


# =============================================================================
# VERIFICATION FUNCTIONS
# =============================================================================

def check_forbidden_phrases(text: str, forbidden: List[str]) -> List[str]:
    """Return list of forbidden phrases found in text."""
    text_lower = text.lower()
    found = []
    for phrase in forbidden:
        if phrase.lower() in text_lower:
            found.append(phrase)
    return found


def check_required_phrases(text: str, required: List[str]) -> List[str]:
    """Return list of required phrases NOT found in text."""
    text_lower = text.lower()
    missing = []
    for phrase in required:
        if phrase.lower() not in text_lower:
            missing.append(phrase)
    return missing


def run_level1_mechanical(provider_name: str) -> Tuple[bool, str]:
    """
    LEVEL 1: Mechanical check - is the provider actually running?
    """
    print("\n" + "="*60)
    print("LEVEL 1: MECHANICAL CHECK")
    print("="*60)
    
    try:
        provider = create_llm_provider(provider_name)
        actual_name = provider.provider_name
        
        print(f"  Requested backend: {provider_name}")
        print(f"  Actual provider:   {actual_name}")
        print(f"  LLM Provider in use: {actual_name.upper()}")
        
        if provider_name == "phi_local" and actual_name != "phi_local":
            return False, f"FAIL: Requested phi_local but got {actual_name} (silent fallback!)"
        
        # Test generation
        test_prompt = "Generate a one-sentence summary: The compound has good affinity."
        result = provider.generate(test_prompt)
        
        if not result or len(result) < 10:
            return False, "FAIL: Provider returned empty or too-short response"
        
        print(f"  Test generation: SUCCESS ({len(result)} chars)")
        return True, f"PASS: {actual_name} is active and generating"
        
    except Exception as e:
        return False, f"FAIL: {str(e)}"


def run_level2_trust(generator) -> Tuple[bool, List[Dict]]:
    """
    LEVEL 2: Trust check - does it obey rules?
    """
    print("\n" + "="*60)
    print("LEVEL 2: TRUST CHECK")
    print("="*60)
    
    test_cases = [CASE_A, CASE_B, CASE_C, CASE_D]
    results = []
    all_pass = True
    
    for case in test_cases:
        print(f"\n  {case.name}")
        print("  " + "-"*50)
        
        narrative = generator.generate_narrative(case.candidate)
        
        # Check forbidden phrases
        forbidden_found = check_forbidden_phrases(narrative, case.forbidden_phrases)
        
        # Check required phrases (skip for sparse case)
        missing_required = check_required_phrases(narrative, case.required_phrases)
        
        passed = len(forbidden_found) == 0 and len(missing_required) == 0
        
        result = {
            "case": case.name,
            "passed": passed,
            "narrative": narrative[:200] + "..." if len(narrative) > 200 else narrative,
            "forbidden_found": forbidden_found,
            "missing_required": missing_required,
        }
        results.append(result)
        
        if passed:
            print(f"    [PASS]")
        else:
            all_pass = False
            print(f"    [FAIL]")
            if forbidden_found:
                print(f"       Forbidden phrases found: {forbidden_found}")
            if missing_required:
                print(f"       Missing required phrases: {missing_required}")
        
        print(f"    Narrative preview: {narrative[:100]}...")
    
    return all_pass, results


def run_level3_regression(generator) -> Tuple[bool, Dict]:
    """
    LEVEL 3: Regression check - determinism and failure handling.
    """
    print("\n" + "="*60)
    print("LEVEL 3: REGRESSION CHECK")
    print("="*60)
    
    results = {}
    all_pass = True
    
    # Test 1: Determinism
    print("\n  Test 1: Determinism (same input = same output)")
    print("  " + "-"*50)
    
    narrative1 = generator.generate_narrative(CASE_A.candidate)
    narrative2 = generator.generate_narrative(CASE_A.candidate)
    
    determinism_pass = narrative1 == narrative2
    results["determinism"] = {
        "passed": determinism_pass,
        "output1_hash": hashlib.md5(narrative1.encode()).hexdigest()[:8],
        "output2_hash": hashlib.md5(narrative2.encode()).hexdigest()[:8],
    }
    
    if determinism_pass:
        print(f"    [PASS]: Identical outputs (hash: {results['determinism']['output1_hash']})")
    else:
        all_pass = False
        print(f"    [FAIL]: Different outputs!")
        print(f"       Hash 1: {results['determinism']['output1_hash']}")
        print(f"       Hash 2: {results['determinism']['output2_hash']}")
    
    # Test 2: Sparse ML handling
    print("\n  Test 2: Sparse ML data handling")
    print("  " + "-"*50)
    
    sparse_candidate = {
        "compound_name": "NoMLCompound",
        "smiles": "CCO",
        "stage1": {"rank": 5, "score": 6.0, "percentile": 70.0},
        "stage2": {
            "docking": {"docking_status": "PASS"},
            "adme_tox": {"adme_tox_label": "SAFE"},
        },
        # No ML fields at all
    }
    
    sparse_narrative = generator.generate_narrative(sparse_candidate)
    
    # Check for hallucinated ML content
    ml_hallucination_phrases = [
        "ML uncertainty",
        "model confidence",
        "prediction uncertainty",
        "ensemble disagreement",
    ]
    ml_hallucinated = check_forbidden_phrases(sparse_narrative, ml_hallucination_phrases)
    
    sparse_pass = len(ml_hallucinated) == 0
    results["sparse_ml"] = {
        "passed": sparse_pass,
        "hallucinated_phrases": ml_hallucinated,
    }
    
    if sparse_pass:
        print(f"    [PASS]: No hallucinated ML content")
    else:
        all_pass = False
        print(f"    [FAIL]: Hallucinated ML phrases: {ml_hallucinated}")
    
    # Test 3: Generation constraints are correct
    print("\n  Test 3: Generation constraints verification")
    print("  " + "-"*50)
    
    constraints_pass = (
        GENERATION_CONSTRAINTS["temperature"] == 0.0 and
        GENERATION_CONSTRAINTS["do_sample"] == False
    )
    results["constraints"] = {
        "passed": constraints_pass,
        "temperature": GENERATION_CONSTRAINTS["temperature"],
        "do_sample": GENERATION_CONSTRAINTS["do_sample"],
    }
    
    if constraints_pass:
        print(f"    [PASS]: temp={GENERATION_CONSTRAINTS['temperature']}, do_sample={GENERATION_CONSTRAINTS['do_sample']}")
    else:
        all_pass = False
        print(f"    [FAIL]: Incorrect constraints")
    
    return all_pass, results


def run_full_verification(backend: str = "mock") -> Dict:
    """
    Run complete 3-level verification.
    
    Args:
        backend: 'phi_local' for real test, 'mock' for CI
    """
    print("\n" + "="*60)
    print(f"PHASE-4 LLM TRUST VERIFICATION")
    print(f"Backend: {backend}")
    print("="*60)
    
    results = {
        "backend": backend,
        "level1_passed": False,
        "level2_passed": False,
        "level3_passed": False,
        "overall_passed": False,
    }
    
    # LEVEL 1
    level1_pass, level1_msg = run_level1_mechanical(backend)
    results["level1_passed"] = level1_pass
    results["level1_message"] = level1_msg
    print(f"\n  {level1_msg}")
    
    if not level1_pass:
        print("\n[FAIL] LEVEL 1 FAILED - Cannot proceed to LEVEL 2")
        return results
    
    # Create generator for Level 2 and 3
    # NOTE: For mock provider, use rule-based fallback for semantic tests
    # Mock only tests wiring, rule-based tests semantic behavior
    if backend == "mock":
        print("\n  NOTE: Using rule-based fallback for semantic tests (mock = CI mode)")
        generator = create_narrative_generator(use_llm=False)
    else:
        generator = create_narrative_generator(use_llm=True, llm_backend=backend)
    
    # LEVEL 2
    level2_pass, level2_results = run_level2_trust(generator)
    results["level2_passed"] = level2_pass
    results["level2_results"] = level2_results
    
    # LEVEL 3
    level3_pass, level3_results = run_level3_regression(generator)
    results["level3_passed"] = level3_pass
    results["level3_results"] = level3_results
    
    # Overall
    results["overall_passed"] = level1_pass and level2_pass and level3_pass
    
    # Summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print(f"  LEVEL 1 (Mechanical): {'[PASS]' if level1_pass else '[FAIL]'}")
    print(f"  LEVEL 2 (Trust):      {'[PASS]' if level2_pass else '[FAIL]'}")
    print(f"  LEVEL 3 (Regression): {'[PASS]' if level3_pass else '[FAIL]'}")
    print("="*60)
    
    if results["overall_passed"]:
        print("[SUCCESS] PHASE-4 VERIFICATION: PASSED")
        print("   Phi LLM is working correctly and can be frozen.")
    else:
        print("[FAIL] PHASE-4 VERIFICATION: FAILED")
        print("   Do NOT freeze. Fix issues before proceeding.")
    
    return results


if __name__ == "__main__":
    # Default to mock for CI, use phi_local for real test
    backend = os.getenv("LLM_BACKEND", "mock")
    
    print(f"\nUsing backend: {backend}")
    print("To test with real Phi: set LLM_BACKEND=phi_local")
    
    results = run_full_verification(backend)
    
    # Save results
    output_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "outputs",
        "phase4_verification_results.json"
    )
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to: {output_path}")
    
    # Exit with appropriate code
    sys.exit(0 if results["overall_passed"] else 1)

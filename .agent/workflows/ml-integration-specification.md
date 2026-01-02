---
description: ML Integration Specification - Authoritative governance contract for ML model integration
---

# ML INTEGRATION SPECIFICATION

> **Document Type**: Governance Contract  
> **Version**: 1.0  
> **Effective Date**: 2026-01-02  
> **Status**: AUTHORITATIVE - Violations invalidate the integration

This specification defines the boundaries, permissions, and constraints for any Machine Learning model integrated into the Agentic Drug Discovery Platform.

---

## SECTION 1 — SYSTEM OUTPUTS (AUTHORITATIVE DEFINITION)

### 1.1 Stage-1 Outputs (Affinity Ranking)

| Output | Definition | Question Answered | What It Is NOT |
|--------|------------|-------------------|----------------|
| **Affinity Score** | DeepDTA-predicted binding affinity (pKd scale) | "How strongly might this bind?" | NOT a measured Kd value |
| **Rank** | Position in sorted candidate list | "Where does this sit among all candidates?" | NOT an absolute quality measure |
| **Percentile** | Rank normalized to 0-100 | "What fraction of candidates are below this?" | NOT a probability |
| **Confidence Tier** | HIGH/MEDIUM/LOW based on model uncertainty | "How reliable is this prediction?" | NOT a guarantee |

**Assumptions**:
- Target protein sequence is valid
- SMILES represents a real molecule
- Training data distribution applies

---

### 1.2 Stage-2 Outputs (ADME/Tox Validation)

| Output | Definition | Labels | When Triggered |
|--------|------------|--------|----------------|
| **ADME/Tox Label** | Developability risk assessment | `SAFE` / `FLAGGED` / `HIGH_RISK` | Always |
| **PAINS Alerts** | Pan-assay interference substructure matches | List of matched patterns | If PAINS present |
| **Rule Violations** | Specific property violations | List with values and thresholds | If any threshold exceeded |

**Label Semantics**:

| Label | Meaning | Action |
|-------|---------|--------|
| `SAFE` | No developability flags detected | Proceed normally |
| `FLAGGED` | Some concerns, but not fatal | Proceed with caution, investigate |
| `HIGH_RISK` | Multiple serious concerns | Deprioritize or require orthogonal validation |

**What ADME/Tox Does NOT Claim**:
- Does NOT predict clinical safety
- Does NOT replace in vivo testing
- Does NOT guarantee ADMET properties

---

### 1.3 Stage-2.1 Outputs (Docking Validation)

| Output | Definition | Values | Interpretation |
|--------|------------|--------|----------------|
| **Docking Status** | Structural plausibility verdict | `PASS` / `FLAG` / `FAIL` / `NOT_EVALUATED` | Filter signal only |
| **Vina Score** | AutoDock Vina scoring-function output | kcal/mol (negative = better) | Heuristic, not thermodynamic |
| **Pose Available** | Whether 3D pose was generated | true/false | Structural information only |

**Status Thresholds (LOCKED)**:

| Status | Vina Score | Interpretation |
|--------|------------|----------------|
| `PASS` | ≤ -7.0 | Strong predicted binding |
| `FLAG` | -7.0 to -5.0 | Marginal, proceed with caution |
| `FAIL` | > -5.0 | Poor predicted fit |
| `NOT_EVALUATED` | N/A | Tools unavailable (honest deferral) |

**Explicit Limitations of Docking**:
- Protein treated as rigid (no induced fit)
- Solvent effects approximated
- Gasteiger charges are empirical, not QM-derived
- Scores are target-dependent and NOT comparable across proteins
- This is a FILTER, not an affinity predictor

---

## SECTION 2 — DECISION AUTHORITY HIERARCHY

```
┌─────────────────────────────────────────────────────────┐
│  LEVEL 1: Chemistry + ADME Hard Constraints (TERMINAL)  │
│  → RDKit validity, MW, LogP, PAINS (if configured)      │
│  → Cannot be overridden by any downstream component     │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│  LEVEL 2: Docking Structural Plausibility (FILTER)     │
│  → Validates physical fit to binding pocket             │
│  → FAIL is informative, not final rejection             │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│  LEVEL 3: Affinity Ranking (Stage-1 Score)              │
│  → Primary discovery signal                             │
│  → Never modified by Stage-2 validation                 │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│  LEVEL 4: ML Refinement (FUTURE - THIS CONTRACT)        │
│  → Pattern recognition and calibration                  │
│  → May not override any upstream FAIL                   │
└─────────────────────────────────────────────────────────┘
```

**Authority Rules**:
1. No downstream component may override an upstream hard constraint
2. FAIL states are terminal unless manually overridden by a human
3. ML operates ONLY at Level 4 - the refinement layer

---

## SECTION 3 — ML MODEL ACCESS CONTRACT

### 3.1 Allowed Inputs (Exhaustive List)

| Input | Type | Source | Notes |
|-------|------|--------|-------|
| Canonical SMILES | string | RDKit | Validated, canonical form |
| Molecular Graph | PyG/DGL object | RDKit | Node/edge features |
| Stage-1 Affinity Score | float | DeepDTA | Clearly labeled as "DeepDTA score" |
| Stage-1 Percentile | float (0-100) | Normalization | Rank position |
| Stage-1 Confidence Tier | enum | Uncertainty | HIGH/MEDIUM/LOW |
| Docking Status | enum | Vina pipeline | PASS/FLAG/FAIL/NOT_EVALUATED |
| Docking Vina Score | float (kcal/mol) | Vina | Labeled as "Vina score (heuristic)" |
| ADME/Tox Label | enum | Stage-2 | SAFE/FLAGGED/HIGH_RISK |
| ADME/Tox Raw Properties | dict | RDKit | MW, LogP, HBD, HBA, TPSA, etc. |
| PAINS Flags | list | RDKit | Matched pattern names |
| Target Protein ID | string | User input | UniProt/PDB identifier |
| Target Protein Sequence | string | Retrieved | Amino acid sequence |

### 3.2 Access Constraints

- ML sees **computed annotations**, not hidden heuristics
- ML does NOT infer missing values - missing = missing
- All inputs are read-only to ML
- ML may NOT request additional data beyond this contract

---

## SECTION 4 — ML MODEL OUTPUT CONTRACT

### 4.1 Allowed Outputs (Exhaustive List)

| Output | Type | Label Requirement | Usage |
|--------|------|-------------------|-------|
| ML Affinity Score | float | `"ml_affinity_score"` | Comparable to Stage-1 score |
| ML Uncertainty | float | `"ml_uncertainty"` | Calibrated confidence |
| Ensemble Variance | float | `"ensemble_variance"` | Model agreement |
| Feature Importance | dict | `"ml_feature_importance"` | Explainability |

### 4.2 Output Requirements

Every ML output MUST:
1. Be clearly labeled with `"ml_"` prefix
2. Include uncertainty/confidence estimate
3. Provide feature attribution for explainability
4. Never replace rule-based labels
5. Never silently drop or suppress candidates
6. Be reproducible given same inputs and model version

### 4.3 Output Structure (Mandatory)

```json
{
  "ml_affinity_score": 7.2,
  "ml_uncertainty": 0.4,
  "ml_confidence_tier": "MEDIUM",
  "ml_model_version": "gat_v1.2.0",
  "ml_explanation": {
    "top_features": [...],
    "what_this_means": "...",
    "what_this_does_not_mean": "..."
  }
}
```

---

## SECTION 5 — PROHIBITED ML BEHAVIOR (NON-NEGOTIABLE)

### 5.1 Absolute Prohibitions

| Prohibition | Rationale |
|-------------|-----------|
| ❌ Override ADME/Tox FAIL or HIGH_RISK | Trust hierarchy violation |
| ❌ Override Docking FAIL | Physics > pattern recognition |
| ❌ Re-rank across FAIL vs PASS boundaries | Status boundaries are semantic |
| ❌ Combine scores into single opaque "final score" | Glass-box violation |
| ❌ Suppress or hide explanations | Transparency is non-negotiable |
| ❌ Invent structural or chemical features | Hallucination risk |
| ❌ Modify Stage-1 ranking order | Discovery/validation separation |
| ❌ Remove candidates from output | No silent suppression |
| ❌ Claim binding affinity in physical units | Only models, not measurements |

### 5.2 Enforcement

Violations of Section 5 prohibitions **invalidate the ML model integration**.

The model will be rejected and not deployed.

---

## SECTION 6 — INTEGRATION PHILOSOPHY ("BEST OF BOTH WORLDS")

### 6.1 Guiding Principle

**Rules + Physics provide trust and guardrails.**
**ML provides pattern recognition and refinement.**
**Humans retain final decision authority.**

The existing pipeline (Stage-1 + Stage-2 + Stage-2.1) establishes a foundation of:
- Hard chemical validity gates
- Empirical ADME/Tox constraints
- Physics-based structural validation
- Deterministic, reproducible ranking

ML complements this foundation by:
- Learning non-obvious structure-activity patterns
- Improving calibration on specific target families
- Providing uncertainty quantification
- Identifying molecular features predictive of success

ML never REPLACES the existing system - it AUGMENTS it.

### 6.2 Visual Representation

```
┌──────────────────────────────────────────────────────────┐
│                    TRUST FOUNDATION                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐   │
│  │  Chemistry  │  │  ADME/Tox   │  │     Docking     │   │
│  │   Filters   │  │   Rules     │  │   Validation    │   │
│  └─────────────┘  └─────────────┘  └─────────────────┘   │
│                 (Cannot be overridden)                    │
└──────────────────────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────┐
│                    ML REFINEMENT LAYER                    │
│  ┌─────────────────────────────────────────────────────┐ │
│  │  Pattern Recognition  │  Calibration  │  Uncertainty│ │
│  └─────────────────────────────────────────────────────┘ │
│               (Complements, never replaces)              │
└──────────────────────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────┐
│                    HUMAN DECISION                         │
│        (Final authority, interprets all signals)          │
└──────────────────────────────────────────────────────────┘
```

---

## SECTION 7 — ACCEPTANCE CRITERIA FOR ML MODELS

### 7.1 Mandatory Requirements

An ML model is acceptable for integration ONLY if it:

| Criterion | Test Method |
|-----------|-------------|
| Improves calibration on known-drug panels | Sanity test on EGFR, BRAF, HMG-CoA drugs |
| Preserves all explainability guarantees | Every output has feature attribution |
| Does not degrade determinism | Same input → same output (given same model version) |
| Survives Phase 2.1 stress tests | T1-T7 regression suite passes |
| Produces explainable outputs | Scientist can understand why score is high/low |
| Labels all outputs as ML-derived | No unlabeled predictions |
| Respects authority hierarchy | Never overrides upstream FAIL |

### 7.2 Validation Process

1. Model trained on approved dataset
2. Model submitted with version and training metadata
3. Regression test suite executed
4. Known-drug sanity panel evaluated
5. Explainability audit performed
6. Human review of sample predictions
7. Integration approved OR rejected

### 7.3 Rejection Criteria

Model is REJECTED if:
- Any Section 5 prohibition is violated
- Determinism test fails
- Explainability is missing
- Known-drug sanity panel shows unreasonable results
- Authority hierarchy is violated

---

## SUMMARY

This specification defines the **immutable contract** between the existing validated pipeline and any future ML integration.

**The one-sentence summary:**

> ML models may refine, calibrate, and add uncertainty quantification, but may never override rule-based constraints, suppress explanations, or claim authority over physics-based validation.

---

**Document Authority**: This specification is authoritative and governs all ML integration work.

**Violation Consequences**: Models violating this specification will not be integrated.

**Amendment Process**: Changes require explicit approval and version increment.

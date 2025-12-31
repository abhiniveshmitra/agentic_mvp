---
description: Unified explanation schema for all Stage-2 agents - mandatory format
---

# STAGE-2 UNIFIED EXPLANATION SCHEMA

> All Stage-2 agents **MUST** use this schema.
> No agent returns a label without a full explanation object.

---

## Core Explanation Object

Every Stage-2 output must include this structure:

```json
{
  "result": {
    "label": "PASS | FLAG | FAIL | SAFE | FLAGGED | HIGH_RISK",
    "confidence": "HIGH | MEDIUM | LOW"
  },
  "raw_values": {
    // All computed properties used in decision
  },
  "rules_triggered": [
    {
      "rule_id": "string",
      "condition": "string (e.g., 'LogP > 7.5')",
      "triggered": true | false,
      "scientific_rationale": "string",
      "practical_implication": "string"
    }
  ],
  "limitations": [
    "string - what this assessment cannot determine"
  ],
  "summary": "string - human-readable 1-2 sentence explanation"
}
```

---

## Agent-Specific Schemas

### Top-K Selection Explanation

```json
{
  "result": {
    "selected": true,
    "rank": 7,
    "total_candidates": 68
  },
  "raw_values": {
    "stage1_score": 7.82,
    "stage1_percentile": 89.7
  },
  "rules_triggered": [
    {
      "rule_id": "TOPK_RANK",
      "condition": "rank <= 20",
      "triggered": true,
      "scientific_rationale": "Top-K selection prioritizes highest-confidence candidates for deeper analysis",
      "practical_implication": "Reduces computational cost while preserving best candidates"
    }
  ],
  "limitations": [
    "Lower-ranked molecules are not scientifically rejected",
    "This is resource prioritization, not quality judgment"
  ],
  "summary": "Selected for Stage-2 (rank #7 of 68, top 20 cutoff)"
}
```

### Docking Explanation

```json
{
  "result": {
    "label": "PASS | FLAG | FAIL",
    "pose_available": true
  },
  "raw_values": {
    "docking_score": -8.2,
    "pose_count": 3,
    "best_rmsd": 1.4
  },
  "rules_triggered": [
    {
      "rule_id": "STERIC_FIT",
      "condition": "no major clashes",
      "triggered": true,
      "scientific_rationale": "Ligand must fit within binding pocket without severe overlap",
      "practical_implication": "Suggests structural compatibility with target"
    }
  ],
  "observations": [
    "Ligand fits within ATP-binding pocket",
    "Core scaffold aligns with hinge region"
  ],
  "limitations": [
    "Protein treated as rigid",
    "Solvent effects not modeled",
    "Score is not binding free energy"
  ],
  "summary": "Docking PASS - ligand fits binding pocket without major clashes"
}
```

### ADME/Tox Explanation

```json
{
  "result": {
    "label": "SAFE | FLAGGED | HIGH_RISK"
  },
  "raw_values": {
    "mw": 512,
    "logp": 7.8,
    "tpsa": 38,
    "hbd": 1,
    "hba": 6,
    "rotatable_bonds": 5,
    "aromatic_rings": 3,
    "fraction_csp3": 0.25,
    "pains_count": 1
  },
  "rules_triggered": [
    {
      "rule_id": "HIGH_LIPOPHILICITY",
      "condition": "LogP 7.8 > 7.5",
      "triggered": true,
      "scientific_rationale": "Highly lipophilic molecules exhibit poor solubility and nonspecific binding",
      "practical_implication": "May require formulation strategies, increased off-target risk"
    },
    {
      "rule_id": "PAINS_ALERT",
      "condition": "PAINS count >= 1",
      "triggered": true,
      "scientific_rationale": "PAINS motifs associated with assay interference",
      "practical_implication": "Orthogonal validation required"
    }
  ],
  "pains_details": {
    "matched_motifs": ["quinone"],
    "important_note": "PAINS alerts indicate risk, not proof of inactivity"
  },
  "limitations": [
    "Rule-based only, not predictive",
    "Does not account for metabolites",
    "Clinical outcome not predicted"
  ],
  "summary": "FLAGGED - high lipophilicity and PAINS alert suggest elevated developability risk"
}
```

### Patent Risk Explanation

```json
{
  "result": {
    "label": "CLEAR | POTENTIAL_RISK | LIKELY_ENCUMBERED | NOT_EVALUATED"
  },
  "raw_values": {
    "inchikey": "XXXXX-XXXXX-X",
    "hit_count": 5
  },
  "rules_triggered": [
    {
      "rule_id": "SIMILAR_SCAFFOLD",
      "condition": "similar scaffolds in patent literature",
      "triggered": true,
      "scientific_rationale": "Structural similarity to patented compounds indicates potential IP overlap",
      "practical_implication": "Professional IP analysis recommended before development"
    }
  ],
  "limitations": [
    "Heuristic assessment only",
    "Not a legal determination",
    "Based on public patent text similarity"
  ],
  "disclaimer": "This is not legal advice. Professional IP counsel required.",
  "summary": "POTENTIAL_RISK - similar quinazoline scaffolds found in patent literature"
}
```

### Aggregation Final Output

```json
{
  "smiles": "CC(=O)Nc1ccc2...",
  "stage1": {
    "rank": 7,
    "score": 7.82,
    "percentile": 89.7,
    "confidence_tier": "HIGH"
  },
  "stage2": {
    "docking": { /* full explanation object */ },
    "adme_tox": { /* full explanation object */ },
    "patent": { /* full explanation object */ }
  },
  "narrative": "This compound ranks in the top 10% for predicted EGFR affinity...",
  "provenance": {
    "stage1_version": "phase1-stable",
    "stage2_version": "1.0.0",
    "timestamp": "2025-12-31T12:00:00Z"
  }
}
```

---

## FAIL vs FLAG Semantics

| Status | Meaning | Molecule Shown? | User Action |
|--------|---------|-----------------|-------------|
| **PASS** | Passed all checks | Yes | Proceed |
| **FLAG** | Warning, but not blocking | Yes (with warning) | Review advised |
| **FAIL** | Failed structural check | Yes (with explanation) | Investigate |
| **SAFE** | No ADME/Tox concerns | Yes | Proceed |
| **FLAGGED** | Moderate risk | Yes (with details) | Consider risk |
| **HIGH_RISK** | Significant risk | Yes (with full explanation) | Serious consideration |

**Critical Rule**: No molecule is ever hidden. All are shown with explanation.

---

## Enforcement Rule

> No agent returns a label unless it can also return a full explanation object.

If an agent cannot explain, it must return:

```json
{
  "result": { "label": "UNKNOWN" },
  "raw_values": {},
  "rules_triggered": [],
  "limitations": ["Unable to compute - reason"],
  "summary": "Assessment could not be completed"
}
```

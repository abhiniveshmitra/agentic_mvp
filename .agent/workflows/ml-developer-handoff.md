---
description: ML Developer Handoff - Everything you need to build the ML model for this platform
---

# ML DEVELOPER HANDOFF PACKAGE

> **For**: ML Engineer/Data Scientist  
> **From**: Platform Team  
> **Purpose**: Clear, actionable instructions for ML model development

---

## YOUR TASK (ONE SENTENCE)

> **"Your job is to add an ML model that refines ranking and confidence *within* already-approved candidates, without overriding rules, docking, or explanations."**

---

## WHAT YOU RECEIVE

### 1. The Governance Contract

Read this FIRST: [ml-integration-specification.md](./ml-integration-specification.md)

**This is not guidance — it's the boundary.**

Key points:
- ML is Level 4 in the authority hierarchy (cannot override chemistry, ADME, or docking)
- All outputs must have `ml_` prefix and include uncertainty
- 9 absolute prohibitions that will invalidate your model if violated

---

### 2. Input Schema (What You Get)

Your model receives this structure for each candidate:

```json
{
  "smiles": "CCOc1ccc2nc(S(N)(=O)=O)sc2c1",
  "target_id": "P00533",
  "target_sequence": "MRPSGTAGAALLALLAALCPAS...",
  
  "stage1": {
    "affinity_score": 8.1,
    "percentile": 87.5,
    "confidence_tier": "HIGH"
  },
  
  "docking": {
    "status": "PASS",
    "vina_score": -7.4
  },
  
  "adme_tox": {
    "label": "SAFE",
    "properties": {
      "mw": 480,
      "logp": 3.2,
      "hbd": 2,
      "hba": 5,
      "tpsa": 71.0,
      "rotatable_bonds": 6
    }
  },
  
  "pains": []
}
```

**Notes:**
- All values are pre-computed and read-only
- Missing values = missing (do not infer)
- You may NOT request additional data beyond this

---

### 3. Output Schema (What You Produce)

Your model must output this structure:

```json
{
  "ml_affinity_score": 7.2,
  "ml_uncertainty": 0.4,
  "ml_confidence_tier": "MEDIUM",
  "ml_model_version": "gat_v1.0.0",
  "ml_explanation": {
    "top_features": [
      {"feature": "logp", "contribution": 0.15},
      {"feature": "hba", "contribution": -0.08}
    ],
    "what_this_means": "Model predicts moderate affinity with reasonable confidence",
    "what_this_does_not_mean": "This is NOT a measured binding constant"
  }
}
```

**Requirements:**
- All outputs prefixed with `ml_`
- Uncertainty is mandatory
- Explainability is mandatory
- Deterministic: same input → same output (given same model version)

---

### 4. Known-Drug Sanity Set

Your model must produce sensible results for these approved drugs:

| Target | Drug | Expected | Notes |
|--------|------|----------|-------|
| EGFR | Gefitinib | Reasonable score | FDA approved |
| EGFR | Erlotinib | Reasonable score | FDA approved |
| BRAF | Vemurafenib | Reasonable score | FDA approved |
| HMG-CoA | Atorvastatin | Reasonable score | FDA approved |
| HMG-CoA | Rosuvastatin | Reasonable score | FDA approved |

**If your model makes these look insane, we stop.**

SMILES for testing:
```python
KNOWN_DRUGS = {
    "gefitinib": "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1",
    "erlotinib": "COCCOc1cc2ncnc(Nc3cccc(C#C)c3)c2cc1OCCOC",
    "vemurafenib": "CCCS(=O)(=O)Nc1ccc(F)c(C(=O)c2cc(F)ccc2-c2cc(-c3ccnc(N)n3)c[nH]2)c1",
    "atorvastatin": "CC(C)c1c(C(=O)Nc2ccccc2)c(-c2ccccc2)c(-c2ccc(F)cc2)n1CCC(O)CC(O)CC(=O)O",
    "rosuvastatin": "CC(C)c1nc(N(C)S(=O)(=O)C)nc(-c2ccc(F)cc2)c1/C=C/C(O)CC(O)CC(=O)O",
}
```

---

## YOUR FIRST MILESTONE

**ML Milestone 0 — Feasibility + Interface Test**

Do NOT start with full training. First prove:

1. ✓ Load sample data (use the known drugs above)
2. ✓ Produce `ml_affinity_score` and `ml_uncertainty`
3. ✓ Outputs are deterministic (same input → same output)
4. ✓ Outputs respect FAIL boundaries (never flip FAIL to PASS)
5. ✓ Output format matches the schema above

**Timeline**: A few days, not weeks.
**Deliverable**: Working prediction function, no integration needed yet.

---

## WHAT YOU MAY NOT DO

From the governance contract, these are absolute prohibitions:

| ❌ Prohibition | Why |
|----------------|-----|
| Override ADME/Tox FAIL or HIGH_RISK | Trust hierarchy violation |
| Override Docking FAIL | Physics > pattern recognition |
| Re-rank across FAIL vs PASS boundaries | Status boundaries are semantic |
| Create single opaque "final score" | Glass-box violation |
| Suppress or hide explanations | Transparency is non-negotiable |
| Invent structural features | Hallucination risk |
| Modify Stage-1 ranking order | Discovery/validation separation |
| Remove candidates from output | No silent suppression |
| Claim binding affinity in physical units | Only models, not measurements |

**Violations = model rejection.**

---

## MODEL ARCHITECTURE FLEXIBILITY

You have freedom in:
- Model type (GNN, GAT, XGBoost, ensemble, etc.)
- Training approach
- Feature engineering within allowed inputs
- Uncertainty quantification method

You do NOT have freedom in:
- Output format (locked)
- Authority hierarchy (locked)
- Explainability requirements (locked)

---

## ACCEPTANCE TESTS YOUR MODEL MUST PASS

Before integration, your model must:

1. **Sanity test**: Known drugs produce reasonable scores
2. **Determinism test**: Same input → same output
3. **Respect test**: Never flips FAIL to PASS
4. **Explainability test**: Every prediction has feature attribution
5. **Stress test**: Survives T1-T7 regression suite

---

## QUESTIONS?

If you need clarification on:
- **Input data**: Ask for sample files
- **Output format**: Reference the schema above
- **Boundaries**: Read ml-integration-specification.md
- **Architecture choices**: You decide, but output contract is fixed

---

## ONE-LINER TO REMEMBER

> **"ML complements — but never replaces — the existing validated pipeline."**

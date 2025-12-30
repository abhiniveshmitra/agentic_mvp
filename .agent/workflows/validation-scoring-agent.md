---
description: Knowledge base for Validation/Scoring Agent - ML-based affinity prediction
---

# VALIDATION / SCORING AGENT

## Purpose

Predict binding affinity **without context bias**.

## Inputs

* Canonical SMILES
* Protein sequence/ID
* Fixed model versions

## Output Contract (Immutable)

```python
{
    "score": float,
    "uncertainty": float
}
```

## Constraints

* No access to:
  * Literature context
  * Discovery source
  * Human expectations
* No post-processing beyond ensemble rules
* No manual weighting
* No score smoothing or adjustment

## Ensemble Rule (Phase 2+)

* Final score = mean(model_scores)
* Uncertainty = std(model_scores)

## Mental Model

> "I score molecules blindly. Context is irrelevant."

---

## Operational Workflow (Phase 1 - DeepDTA Only)

1. Receive filtered compounds + protein target
2. For each compound:
   a. Encode SMILES (character-level)
   b. Encode protein sequence
   c. Run DeepDTA inference
   d. Return score + uncertainty
3. Pass all scores to Normalization Agent

---
description: Knowledge base for Stage-2 Aggregation Agent (NON-LLM) - evidence presentation
---

# STAGE-2 AGGREGATION AGENT (NON-LLM)

## Purpose

Present **all evidence side-by-side** without blending.

## Output Contract (Per Ligand)

```json
{
  "smiles": "string",
  "stage1": {
    "rank": int,
    "score": float,
    "percentile": float,
    "confidence": "string"
  },
  "stage2": {
    "docking": {
      "status": "PASS | FLAG | FAIL",
      "pose_available": bool,
      "confidence": float | null
    },
    "adme_tox": {
      "label": "SAFE | FLAGGED | HIGH_RISK",
      "reasons": ["string"],
      "properties": {...}
    },
    "patent": {
      "risk": "CLEAR | POTENTIAL_RISK | LIKELY_ENCUMBERED",
      "notes": "string"
    }
  },
  "provenance": {
    "stage1_version": "string",
    "stage2_version": "string",
    "timestamp": "ISO8601"
  }
}
```

## Rules (Immutable)

* No cross-layer math
* No final "super score"
* Preserve interpretability
* No re-ranking
* No suppression

## Forbidden Behaviors

* Combining Stage-1 and Stage-2 into a single score
* Hiding any evidence
* Re-ordering based on Stage-2 results
* Applying any weighting

## Implementation Notes

* This is a **NON-LLM** agent
* Pure data aggregation
* No reasoning or interpretation
* Complete provenance preservation

## Mental Model

> "I present all evidence. I do not blend or judge."

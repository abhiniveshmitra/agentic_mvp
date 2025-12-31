---
description: Knowledge base for Docking Agent (NON-LLM) - structural plausibility validation
---

# DOCKING AGENT (NON-LLM)

## Purpose

Validate **structural plausibility**, not affinity ranking.

## Inputs

* Prepared protein
* Canonical ligand SMILES (Top-K only)

## Responsibilities

* Run docking
* Generate 1–3 representative poses
* Extract docking confidence

## Output Contract (Per Ligand)

```json
{
  "ligand_smiles": "string",
  "docking_status": "PASS | FLAG | FAIL",
  "pose_available": true | false,
  "confidence": float | null,
  "poses": [
    {
      "pose_id": int,
      "score": float,
      "rmsd": float | null
    }
  ],
  "error": "string | null"
}
```

## Rules (Immutable)

* Docking score must **never modify Stage-1 score**
* Docking failure is allowed and logged
* Docking is a **validation signal only**
* Runs in parallel, non-blocking

## Forbidden Behaviors

* Ranking molecules by docking score
* Combining docking scores with affinity
* Re-ordering candidates based on docking
* Suppressing failed docking results

## Implementation Notes

* This is a **NON-LLM** agent
* If docking fails → mark FAIL, continue pipeline
* Pipeline continues regardless of docking outcome

## Mental Model

> "I check if the molecule can physically fit. I do not judge its quality."

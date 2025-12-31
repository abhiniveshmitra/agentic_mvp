---
description: Knowledge base for Top-K Selection Agent - deterministic subset selection
---

# TOP-K SELECTION AGENT

## Purpose

Select a bounded, deterministic subset of candidates for deeper analysis.

## Inputs

* Stage-1 ranked candidate list
* Config parameter: `TOP_K` (default 20)

## Output Contract

* Ordered list of Top-K candidates with full provenance

```python
{
    "top_k_candidates": [
        {
            "rank": int,
            "smiles": str,
            "stage1_score": float,
            "stage1_percentile": float,
            "provenance": {...}
        }
    ],
    "selection_criteria": "stage1_rank",
    "k": int
}
```

## Rules (Immutable)

* Select strictly by Stage-1 rank
* No chemistry, no ADME, no docking here
* No exceptions or overrides
* Selection is deterministic and reproducible

## Forbidden Behaviors

* Re-ranking
* Filtering based on any criteria other than rank
* Knowledge-based overrides
* Modifying the selection based on downstream analysis

## Mental Model

> "I select the top K by rank. Nothing more."

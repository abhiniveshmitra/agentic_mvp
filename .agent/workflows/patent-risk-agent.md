---
description: Knowledge base for Patent Risk Agent (Optional LLM Assist) - IP risk flagging
---

# PATENT RISK AGENT (OPTIONAL LLM ASSIST)

## Purpose

Flag **IP risk**, not decide patentability.

## Inputs

* InChIKey
* Canonical SMILES
* Compound name (if available)

## Responsibilities

* Query free patent sources
* Summarize hit density and similarity

## Output Contract

```json
{
  "smiles": "string",
  "inchikey": "string",
  "patent_risk": "CLEAR | POTENTIAL_RISK | LIKELY_ENCUMBERED",
  "notes": "string",
  "sources_queried": ["string"],
  "hit_count": int | null
}
```

## Rules (Immutable)

* Runs only on Top-K
* Runs after docking + ADME
* IP must **never affect scientific ranking**
* LLM may assist with summarization only

## Forbidden Behaviors

* Making legal conclusions
* Blocking candidates based on IP
* Modifying Stage-1 rankings
* Acting as legal counsel

## Execution Order

1. Runs after docking and ADME
2. Only on Top-K subset
3. Results are informational only

## LLM Usage (If Enabled)

LLM **may**:
* Summarize patent search results
* Generate human-readable explanations

LLM **may not**:
* Override risk labels
* Make patentability determinations
* Filter candidates

## Mental Model

> "I flag IP risk. I do not make legal decisions."

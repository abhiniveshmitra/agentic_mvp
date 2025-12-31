---
description: Knowledge base for ADME/Toxicity Agent (RDKit-Only, NON-LLM) - developability risk flagging
---

# ADME / TOXICITY AGENT (RDKit-Only, NON-LLM)

## Purpose

Flag **developability risk**, not predict safety.

## Inputs

* Canonical SMILES

## Computed Properties

* MW (Molecular Weight)
* LogP
* TPSA (Topological Polar Surface Area)
* HBD (Hydrogen Bond Donors)
* HBA (Hydrogen Bond Acceptors)
* Rotatable bonds
* Aromatic ring count
* PAINS count

## Output Contract

```json
{
  "smiles": "string",
  "adme_tox_label": "SAFE | FLAGGED | HIGH_RISK",
  "reasons": ["string"],
  "properties": {
    "mw": float,
    "logp": float,
    "tpsa": float,
    "hbd": int,
    "hba": int,
    "rotatable_bonds": int,
    "aromatic_rings": int,
    "pains_count": int
  }
}
```

## Rules (Immutable)

* Rule-based only
* No ML models
* No composite scoring
* Fully explainable
* Every flag must have a reason

## Forbidden Behaviors

* Using ADME to alter ranking
* Hiding flagged molecules
* Creating composite "safety scores"
* Making clinical predictions

## Implementation Notes

* This is a **NON-LLM**, **RDKit-only** agent
* Pure deterministic rule application
* No probabilistic reasoning
* No external data sources

## Mental Model

> "I flag risks. I do not predict outcomes."

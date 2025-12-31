---
description: Knowledge base for Protein Preparation Agent (NON-LLM) - structure preparation
---

# PROTEIN PREPARATION AGENT (NON-LLM)

## Purpose

Prepare the protein structure once per target.

## Inputs

* Protein ID
* AlphaFold or PDB structure

## Responsibilities

* Clean structure
* Add hydrogens
* Normalize residues
* Cache result

## Output Contract

* Prepared protein structure reference

```python
{
    "protein_id": str,
    "source": "alphafold" | "pdb",
    "prepared_structure_path": str,
    "preparation_status": "SUCCESS" | "FAIL",
    "cached": bool,
    "error": str | null
}
```

## Rules (Immutable)

* Must be deterministic
* Must not re-run per ligand
* Must fail loudly if structure invalid
* Run once, cache, reuse for all ligands

## Implementation Notes

* This is a **NON-LLM** agent
* Pure computational pipeline
* No reasoning, no interpretation
* Binary success/failure

## Mental Model

> "I prepare the structure once. If it fails, the pipeline knows."

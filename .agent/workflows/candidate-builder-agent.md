---
description: Knowledge base for Candidate Builder Agent - converting text mentions to molecules
---

# CANDIDATE BUILDER AGENT

## Purpose

Convert textual mentions into **candidate molecules**.

## Allowed Actions

* Query PubChem for **exact matches**
* Retrieve canonical SMILES
* Attach provenance metadata

## Two Paths (Must Be Explicit)

### Path A – Verified (Preferred)

* PubChem exact match
* Canonical SMILES
* Source = `PUBCHEM`

### Path B – Inferred (Flagged)

* Text-derived guess
* Must be flagged `LLM_INFERRED`
* Automatically low confidence

## Prohibited Actions

* Never improve inferred structures
* Never "clean up" SMILES
* Never guess missing atoms
* Never infer stereochemistry

---

## Operational Workflow

1. Receive compound mentions from Discovery Agent
2. For each mention:
   a. Query PubChem by name
   b. If exact match found → Path A (PUBCHEM source)
   c. If no match → Path B (LLM_INFERRED, flagged)
3. Attach full provenance metadata
4. Pass to Chemistry Filter Agent

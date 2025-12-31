---
description: How to run the complete Stage-2 pipeline - exact execution steps
---

# STAGE-2 EXECUTION PIPELINE

> This is the **literal run order**. No agent may deviate.

---

## STEP 0 — INPUT LOCK

* Load Stage-1 results
* Verify Stage-1 hash / version
* **Abort if Stage-1 changed**

```python
# Pseudocode
stage1_results = load_stage1_results()
if not verify_stage1_hash(stage1_results):
    abort("Stage-1 results modified, cannot proceed")
```

---

## STEP 1 — TOP-K SELECTION

* Select Top-K by rank
* Freeze list
* Pass forward

```python
# Pseudocode
top_k = select_top_k(stage1_results, k=TOP_K)
freeze(top_k)  # Immutable from this point
```

---

## STEP 2 — PROTEIN PREPARATION

* Prepare protein once
* Cache
* Reuse for all ligands

```python
# Pseudocode
prepared_protein = prepare_protein(protein_id)
cache(prepared_protein)
```

---

## STEP 3 — DOCKING (PARALLEL, NON-BLOCKING)

For each ligand:
* Attempt docking
* If fails → mark FAIL
* **Continue pipeline regardless**

```python
# Pseudocode
for ligand in top_k:
    try:
        result = dock(prepared_protein, ligand.smiles)
        ligand.docking = result
    except DockingError:
        ligand.docking = {"status": "FAIL", "error": str(e)}
    # Pipeline continues regardless
```

---

## STEP 4 — ADME / TOX FLAGS

For each ligand:
* Compute RDKit properties
* Apply rule thresholds
* Assign label + reasons

```python
# Pseudocode
for ligand in top_k:
    properties = compute_adme_properties(ligand.smiles)
    ligand.adme_tox = apply_adme_rules(properties)
```

---

## STEP 5 — PATENT RISK (OPTIONAL)

For each ligand:
* Generate InChIKey
* Query patents
* Assign risk label

```python
# Pseudocode
for ligand in top_k:
    inchikey = generate_inchikey(ligand.smiles)
    ligand.patent = query_patent_risk(inchikey, ligand.smiles)
```

---

## STEP 6 — AGGREGATION & OUTPUT

* Combine Stage-1 + Stage-2 outputs
* **No re-ranking**
* **No suppression**
* Persist full provenance

```python
# Pseudocode
final_output = []
for ligand in top_k:
    final_output.append({
        "stage1": ligand.stage1_data,
        "stage2": {
            "docking": ligand.docking,
            "adme_tox": ligand.adme_tox,
            "patent": ligand.patent
        },
        "provenance": generate_provenance()
    })
persist(final_output)
```

---

## KEY INVARIANTS

1. Stage-1 scores are **never modified**
2. All Top-K candidates appear in output (no filtering)
3. Failed docking does not remove candidates
4. High-risk ADME does not remove candidates
5. Patent encumbered does not remove candidates
6. Full provenance is preserved

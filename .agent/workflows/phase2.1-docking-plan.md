---
description: Phase 2.1 Docking Integration Plan - OpenBabel implementation for real structural validation
---

# PHASE 2.1: REAL DOCKING INTEGRATION PLAN

> **Purpose**: Add real structural validation using AutoDock Vina with OpenBabel for PDBQT preparation.  
> **Prerequisite**: Phase 2 MVP complete (branch: `phase2-mvp-complete`)  
> **Status**: PLANNING (No implementation yet)

---

## Executive Summary

Phase 2.1 adds real molecular docking while preserving:
- All existing ADME/Tox logic (unchanged)
- All explainability infrastructure (unchanged)
- All provenance tracking (unchanged)

The only change: `NOT_EVALUATED` → real `PASS/FLAG/FAIL` with full explanations.

---

## 1. OpenBabel PDBQT Preparation

### 1.1 Why OpenBabel

| Tool | Pros | Cons |
|------|------|------|
| **OpenBabel** | CLI-friendly, open source, widely used | Slight charge inaccuracies |
| MGLTools | Canonical for AutoDock, accurate | Requires GUI/Python 2, harder to automate |

**Decision**: Use OpenBabel for MVP. Document charge model limitations.

### 1.2 Installation

```bash
# Windows (conda recommended)
conda install -c conda-forge openbabel

# Or via pip
pip install openbabel-wheel

# Verify
obabel -V
```

### 1.3 PDBQT Preparation Steps

#### Receptor (Protein) Preparation

```bash
# Step 1: Clean PDB (remove water, ligands, alternate conformations)
obabel receptor.pdb -O receptor_clean.pdb -d

# Step 2: Add hydrogens at physiological pH
obabel receptor_clean.pdb -O receptor_h.pdb -p 7.4

# Step 3: Convert to PDBQT (adds Gasteiger charges)
obabel receptor_h.pdb -O receptor.pdbqt -xr
```

**Explanation Output**:
```json
{
  "step": "receptor_preparation",
  "actions": [
    "Removed water molecules and alternate conformations",
    "Added hydrogens at pH 7.4",
    "Assigned Gasteiger partial charges"
  ],
  "charge_model": "Gasteiger",
  "limitations": [
    "Gasteiger charges are approximate, not QM-derived",
    "No explicit treatment of metal ions"
  ]
}
```

#### Ligand Preparation

```bash
# Step 1: Generate 3D coordinates from SMILES
obabel -:\"CCO\" --gen3d -O ligand.mol2

# Step 2: Add hydrogens and optimize geometry
obabel ligand.mol2 -O ligand_opt.mol2 --minimize --steps 500 --ff MMFF94

# Step 3: Convert to PDBQT
obabel ligand_opt.mol2 -O ligand.pdbqt
```

**Implementation in Python**:
```python
def prepare_ligand_pdbqt(smiles: str) -> Path:
    """
    Convert SMILES to PDBQT using OpenBabel.
    
    Returns:
        Path to prepared PDBQT file
    """
    import subprocess
    
    # RDKit for initial 3D generation (more reliable)
    from rdkit import Chem
    from rdkit.Chem import AllChem
    
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)
    AllChem.MMFFOptimizeMolecule(mol)
    
    # Write temporary MOL file
    mol_path = Path(tempfile.mktemp(suffix='.mol'))
    Chem.MolToMolFile(mol, str(mol_path))
    
    # Convert to PDBQT using OpenBabel
    pdbqt_path = mol_path.with_suffix('.pdbqt')
    subprocess.run([
        'obabel', str(mol_path), '-O', str(pdbqt_path)
    ], check=True)
    
    return pdbqt_path
```

### 1.4 Detection of OpenBabel Availability

**Update to `pipeline.py`**:
```python
def _can_prepare_pdbqt(self) -> bool:
    """
    Check if PDBQT preparation is available.
    """
    import shutil
    obabel_path = shutil.which('obabel')
    
    if obabel_path:
        logger.info(f"OpenBabel available at {obabel_path}")
        return True
    else:
        logger.warning("OpenBabel not found - docking will be NOT_EVALUATED")
        return False
```

---

## 2. Docking PASS / FLAG / FAIL Semantics

### 2.1 Status Definitions

| Status | Vina Score | Interpretation | Action |
|--------|------------|----------------|--------|
| **PASS** | ≤ -7.0 kcal/mol | Strong predicted fit | Proceed confidently |
| **FLAG** | -7.0 to -5.0 kcal/mol | Marginal fit | Proceed with caution |
| **FAIL** | > -5.0 kcal/mol OR no pose | Poor/no fit | Deprioritize |
| **NOT_EVALUATED** | N/A | Tools unavailable | Honest deferral |

> ⚠️ **Critical Terminology**: Always use "Vina score (scoring-function estimate)" — never "binding energy". This prevents confusion with thermodynamic ΔG.

### 2.2 Why These Thresholds

| Threshold | Scientific Rationale |
|-----------|---------------------|
| -7.0 kcal/mol | Corresponds to ~μM affinity (Kd ≈ 7 μM) in Vina's scoring model |
| -5.0 kcal/mol | Corresponds to ~mM affinity (Kd ≈ 150 μM) in Vina's scoring model |

**Source**: Vina scoring function calibration literature.

> ⚠️ **Cross-Target Warning** (include in all outputs):
> "Vina score thresholds are heuristic and target-dependent. Scores are interpreted qualitatively and should not be compared across different protein targets."

### 2.3 What Docking Status Does NOT Mean

| Status | Does NOT Mean |
|--------|---------------|
| PASS | Compound will bind in vivo |
| FLAG | Compound should be rejected |
| FAIL | Compound cannot bind at all |

**Key Principle**: Docking is a filter for structural plausibility, NOT an affinity predictor.

### 2.4 Status Assignment Logic

```python
def _assign_docking_status(self, best_score: float) -> str:
    """
    Assign docking status based on best Vina score.
    
    CRITICAL: These thresholds are for FILTERING, not RANKING.
    """
    if best_score <= -7.0:
        return "PASS"  # Strong predicted binding
    elif best_score <= -5.0:
        return "FLAG"  # Marginal, proceed with caution
    else:
        return "FAIL"  # Poor binding predicted
```

---

## 3. Docking Failure Modes

### 3.1 Expected Failure Categories

| Failure Mode | Example | Status | Explanation |
|--------------|---------|--------|-------------|
| **Ligand too small** | Benzene (MW 78) | FAIL | Ligand cannot fill binding pocket |
| **Ligand too large** | Macrocycle (MW 1200) | FAIL/FLAG | Steric clashes, poor fit |
| **No converged pose** | Highly flexible ligand | FAIL | Conformational sampling limit |
| **PDBQT conversion fail** | Invalid SMILES | FAIL | Input error |
| **Timeout** | Complex search space | FLAG | Exploration incomplete |

### 3.2 Failure Explanations

For each failure mode, the explanation schema includes:

```json
{
  "docking_status": "FAIL",
  "failure_mode": "LIGAND_TOO_SMALL",
  "raw_values": {
    "ligand_mw": 78.1,
    "best_score": null,
    "poses_generated": 0
  },
  "explanation": {
    "why_failed": "Ligand molecular volume (~90 Å³) is smaller than binding pocket (~400 Å³)",
    "scientific_rationale": "Small ligands cannot establish sufficient contacts for stable binding",
    "what_it_does_not_mean": "Does not prove binding impossible via alternative mechanisms",
    "recommended_action": "Consider as fragment for fragment-based drug design"
  }
}
```

### 3.3 Failure Documentation Template

| Field | Description |
|-------|-------------|
| `failure_mode` | Categorical: `LIGAND_TOO_SMALL`, `STERIC_CLASH`, `NO_POSE`, etc. |
| `why_failed` | Human-readable explanation |
| `scientific_rationale` | Brief scientific basis |
| `what_it_does_not_mean` | Clarifies limitations |
| `recommended_action` | What scientist should do next |

---

## 4. Docking Explanation Schema Integration

### 4.1 Schema Extension

The existing unified explanation schema will be extended for docking:

```json
{
  "result": {
    "label": "PASS | FLAG | FAIL | NOT_EVALUATED",
    "pose_available": true,
    "best_score": -8.2
  },
  "raw_values": {
    "vina_scores": [-8.2, -7.5, -7.1],
    "num_poses": 3,
    "exhaustiveness": 8,
    "center": [15.2, 42.1, 33.8],
    "box_size": [20, 20, 20]
  },
  "rules_triggered": [
    {
      "rule_id": "STERIC_FIT",
      "condition": "Ligand fits binding pocket without severe clashes",
      "triggered": true,
      "scientific_rationale": "Van der Waals complementarity is necessary for binding",
      "practical_implication": "Structural compatibility confirmed"
    }
  ],
  "geometry_observations": [
    "Ligand occupies ATP-binding site",
    "Hinge region hydrogen bond observed",
    "Hydrophobic pocket partially filled"
  ],
  "limitations": [
    "Protein treated as rigid",
    "Solvent effects approximated",
    "Entropy contribution not calculated",
    "Charge model: Gasteiger (not QM)"
  ],
  "summary": "PASS - Strong predicted binding (-8.2 kcal/mol), ligand occupies binding pocket with favorable geometry."
}
```

### 4.2 Key Fields for Docking

| Field | Purpose |
|-------|---------|
| `geometry_observations` | Human-readable structural insights |
| `vina_scores` | All pose scores for transparency |
| `exhaustiveness` | Search thoroughness parameter |
| `center/box_size` | Binding site definition (reproducibility) |

### 4.3 Integration with Aggregator

The aggregator will handle docking explanations exactly like ADME/Tox:

```python
# In Stage2Aggregator.aggregate()
if docking_result:
    stage2["docking"] = {
        "docking_status": docking_result.docking_status,
        "best_score": docking_result.best_score,
        "explanation": docking_result.explanation.to_dict(),
    }
```

### 4.4 Narrative Integration

The narrative generator will include docking summary:

```python
if docking_result.docking_status == "PASS":
    parts.append(
        f"Docking analysis shows favorable binding "
        f"({docking_result.best_score:.1f} kcal/mol) with "
        f"the ligand occupying the target binding pocket."
    )
elif docking_result.docking_status == "FLAG":
    parts.append(
        f"Docking suggests marginal binding "
        f"({docking_result.best_score:.1f} kcal/mol). "
        "Binding may require induced fit or conformational changes."
    )
elif docking_result.docking_status == "FAIL":
    parts.append(
        f"Docking could not identify a favorable pose: "
        f"{docking_result.failure_mode}. This is structural information, "
        "not a definitive rejection."
    )
```

---

## 5. Implementation Checklist (Phase 2.1 Gate)

### Prerequisites

- [ ] OpenBabel installed and in PATH
- [ ] `obabel -V` returns version
- [ ] Test PDBQT generation on simple molecule
- [ ] Test receptor preparation on cached PDB

### Code Changes

- [ ] Update `_can_prepare_pdbqt()` to detect OpenBabel
- [ ] Implement `prepare_ligand_pdbqt()` with OpenBabel
- [ ] Implement `prepare_receptor_pdbqt()` with OpenBabel
- [ ] Update `dock_single()` to use real Vina
- [ ] Add geometry observation extraction from poses
- [ ] Update explanation schema with docking fields

### Verification

- [ ] Known drugs show PASS/FLAG (Gefitinib, Erlotinib)
- [ ] Known fragments show FAIL (Benzene)
- [ ] All failure modes documented
- [ ] Explanation schema validated

### Documentation

- [ ] Update `stage2-failure-modes.md` with docking failures
- [ ] Update `stage2-explanation-schema.md` with docking fields
- [ ] Update `phase2-exit-checklist.md` → Phase 2.1 checklist

---

## 6. Risk Assessment

| Risk | Mitigation |
|------|------------|
| Gasteiger charges inaccurate | Document limitation, compare to AM1-BCC if needed |
| Rigid protein misses induced fit | Note in explanation, recommend MD follow-up |
| Score threshold too strict | Tune based on validation set |
| OpenBabel version incompatibility | Pin version in requirements |

---

## 7. Timeline Estimate

| Task | Estimated Time |
|------|----------------|
| OpenBabel installation + verification | 30 min |
| PDBQT preparation implementation | 2 hours |
| Vina integration updates | 1 hour |
| Explanation schema extension | 1 hour |
| Testing on known drugs | 1 hour |
| Documentation updates | 1 hour |
| **Total** | **~6-7 hours** |

---

## 8. Decision Points

Before implementing, confirm:

1. **OpenBabel is acceptable** for charge model (vs AM1-BCC)
2. **Thresholds** (-7.0 / -5.0) are appropriate for target classes
3. **Geometry observations** detail level is sufficient
4. **No re-ranking by docking scores** (maintain Stage-1 order)

---

## One-Line Rule

> **Docking validates structural plausibility. It does not predict affinity or rank molecules.**

---

**Author**: Antigravity  
**Date**: 2025-12-31  
**Status**: PLANNING COMPLETE - Ready for implementation when approved

---
description: Stage-1 frozen thresholds and rules snapshot - baseline for Stage-2
---

# STAGE-1 FROZEN SNAPSHOT

> **Git Tag**: `phase1-stable`
> **Commit**: `bc78b03`
> **Date**: 2025-12-31

This document captures all Stage-1 thresholds and rules at the time of Stage-2 development.
**Stage-2 must never modify these values.**

---

## Chemistry Filter Thresholds (Hard Gate)

| Property | Min | Max | Notes |
|----------|-----|-----|-------|
| Molecular Weight | 120 Da | 700 Da | Relaxed for fragment-like binders |
| LogP | -1 | 8.0 | Widened for lipophilic kinase inhibitors |
| Rotatable Bonds | - | 10 | Flexibility limit |
| Formal Charge (abs) | - | 2 | Avoid charged species |
| PAINS Filter | - | - | **Disabled** in Phase 1 |

**Source**: `config/settings.py` lines 40-56

---

## ADME/Tox Thresholds (Post-Ranking Flags)

| Property | Threshold | Issue |
|----------|-----------|-------|
| MW | > 700 | High molecular weight |
| LogP | > 8.0 | High lipophilicity |
| HBD | > 5 | Too many H-bond donors |
| HBA | > 10 | Too many H-bond acceptors |
| TPSA | < 20 | Low TPSA → poor solubility |
| TPSA | > 140 | High TPSA → poor permeability |
| Rotatable Bonds | > 12 | High flexibility |
| Aromatic Rings | > 5 | Too many aromatic rings |
| Fraction Csp3 | < 0.1 | Flat/promiscuous risk |

**Source**: `validation/adme_tox.py` lines 60-73

---

## Toxicity Alert SMARTS

| Alert | SMARTS |
|-------|--------|
| Acyl Halide | `[CX3](=[OX1])[ClBrIF]` |
| Aldehyde | `[CX3H1](=O)[#6]` |
| Epoxide | `C1OC1` |
| Michael Acceptor | `[CX3]=[CX3][CX3]=O` |
| Nitro Aromatic | `[cR1][N+](=O)[O-]` |
| Alkyl Halide | `[CX4][Cl,Br,I]` |
| Azo | `[#6]N=N[#6]` |
| Hydrazine | `[NX3][NX3]` |
| Isocyanate | `[NX2]=C=O` |
| Sulfonyl Halide | `[SX4](=[OX1])(=[OX1])[Cl,Br,I]` |
| Peroxide | `[OX2][OX2]` |
| Nitroso | `[#6][NX2]=O` |

**Source**: `validation/adme_tox.py` lines 80-93

---

## Control Validation Thresholds

| Setting | Value |
|---------|-------|
| Positive Control Min Percentile | 80% |
| Negative Control Max Percentile | 20% |
| Minimum Separation Margin | 0.3 |

**Source**: `config/settings.py` lines 71-75

---

## Normalization Rules

| Setting | Value |
|---------|-------|
| Min Batch for Z-Score | 30 |
| Exclude Controls from Stats | True |

**Source**: `config/settings.py` lines 62-65

---

## Risk Level Determination (ADME/Tox)

```python
risk_score = len(adme_issues) + len(tox_issues) * 2

if risk_score == 0:
    status = SAFE
elif risk_score <= 2:
    status = FLAGGED
else:
    status = HIGH_RISK
```

**Source**: `validation/adme_tox.py` lines 142-147

---

## What This Means for Stage-2

Stage-2 agents:
- **Cannot** change these thresholds
- **Cannot** re-rank based on ADME/Tox
- **Can** add new flags (with explanation)
- **Must** cite Stage-1 values when explaining

If thresholds change in future, create a **Phase-1.1** tag.

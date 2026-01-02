---
description: Phase 2.1 frozen thresholds and configuration - DO NOT MODIFY without explicit approval
---

# PHASE 2.1 FROZEN SNAPSHOT

> **Freeze Date**: 2026-01-02
> **Status**: LOCKED - Do not modify without explicit approval
> **Validation**: 7/7 stress tests passed

---

## Locked Version Information

| Component | Version | Source |
|-----------|---------|--------|
| OpenBabel | 3.1.0 | `pip install openbabel-wheel` |
| AutoDock Vina | 1.2.5 | `tools/vina/vina.exe` |
| RDKit | (system) | 3D coordinate generation |

---

## Locked Thresholds

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| PASS threshold | ≤ -7.0 kcal/mol | Corresponds to ~μM affinity in Vina scoring |
| FLAG threshold | -7.0 to -5.0 kcal/mol | Borderline / proceed with caution |
| FAIL threshold | > -5.0 kcal/mol | Poor predicted fit |

> **WARNING**: Do not adjust thresholds to "make known drugs look nicer."
> FLAG for approved drugs is acceptable and expected (induced fit, kinetics, etc.)

---

## Locked Charge Model

| Parameter | Value |
|-----------|-------|
| Charge model | Gasteiger |
| Protonation pH | 7.4 |
| Hydrogen treatment | Added via OpenBabel |

**Limitations** (documented, not hidden):
- Gasteiger charges are empirical, not QM-derived
- No explicit metal ion treatment
- Protein treated as rigid

---

## Locked Terminology

| Use | Do Not Use |
|-----|------------|
| "Vina score" | "binding energy" |
| "scoring-function estimate" | "thermodynamic ΔG" |
| "structural plausibility" | "binding affinity" |

**CROSS_TARGET_WARNING** (must appear in all outputs):
> "Vina score thresholds are heuristic and target-dependent. Scores are interpreted qualitatively and should not be compared across different protein targets."

---

## Locked Behaviors

### MUST DO
- [x] Docking returns PASS/FLAG/FAIL/NOT_EVALUATED
- [x] NOT_EVALUATED when tools unavailable (honest deferral)
- [x] Full explanation for every result
- [x] Cross-target warning in limitations

### MUST NOT DO
- [ ] Re-rank by docking score
- [ ] Average docking with affinity
- [ ] Add ML "dock score correction"
- [ ] Relax thresholds to reduce FLAG count
- [ ] Hide FAIL results
- [ ] Substitute heuristics for physics

---

## Validation Evidence

Stress test results (2026-01-02):
```
T1_smoke:              PASS
T2_determinism:        PASS
T3_known_drug_sanity:  PASS
T4_failure_modes:      PASS
T5_signal_separation:  PASS
T6_regression_guards:  PASS
T7_explanation_audit:  PASS
```

---

## One-Line Summary

> **Phase 2.1 does not promise correctness — it promises clarity. Every prediction, risk, and limitation is explicit, reproducible, and stress-tested.**

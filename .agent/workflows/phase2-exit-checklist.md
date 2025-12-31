---
description: Phase 2 (MVP) Exit Checklist - Gate document before adding real docking
---

# PHASE 2 (MVP) EXIT CHECKLIST

> **Purpose of Phase 2:**  
> Make decisions explainable, risks explicit, and limitations honest — *without adding new physics.*

**Gate Date**: 2025-12-31  
**Status**: ✅ VERIFIED  

---

## 1. 🔒 Scope & Honesty (Non-Negotiable)

| Checkpoint | Status | Evidence |
|------------|--------|----------|
| Docking status is **NOT_EVALUATED** everywhere unless real PDBQT prep is available | ✅ | `pipeline.py:121-133`, `docking.py:517-568` |
| No simulated / heuristic docking scores exist anywhere in code or output | ✅ | Removed `_mock_docking()` usage from pipeline |
| Every NOT_EVALUATED result clearly states **why** | ✅ | "PDBQT preparation requires MGLTools or OpenBabel" |
| Every NOT_EVALUATED result states **implication** | ✅ | "Structural plausibility not assessed in this run" |
| Every NOT_EVALUATED result states **not an error** | ✅ | "This is an honest deferral, not a failure" |

**Pass condition**: ✅ A reviewer cannot accuse the system of "fake docking" or disguised heuristics.

---

## 2. 🧠 Explainability Completeness

For **every molecule**, the system can answer:

| Question | Status | Implementation |
|----------|--------|----------------|
| Why it ranked where it did (Stage-1) | ✅ | `TopKSelector._build_explanation()` |
| Why it is SAFE / FLAGGED / HIGH_RISK | ✅ | `ADMEToxStage2._build_explanation()` |
| Which exact rules were triggered | ✅ | `rules_triggered` list in every explanation |
| Which rules were checked but *not* triggered | ✅ | `enhanced_trust_validation.py:200-210` |
| What assumptions and limitations apply | ✅ | `limitations` list in every explanation |

**Pass condition**: ✅ A scientist can defend any single decision without referencing code.

---

## 3. 🧪 ADME / Tox Trust Criteria

| Checkpoint | Status | Evidence |
|------------|--------|----------|
| Raw properties always visible (MW, LogP, TPSA, HBD/HBA, RotB, PAINS) | ✅ | `raw_values` dict in every explanation |
| Each rule includes Condition | ✅ | `"condition"` field in `rules_triggered` |
| Each rule includes Scientific rationale | ✅ | `"scientific_rationale"` field |
| Each rule includes Practical implication | ✅ | `"practical_implication"` field |
| FLAGGED ≠ rejected (language reflects caution) | ✅ | "proceed with caution", "mitigation needed" |
| Known edge cases behave correctly | ✅ | Rosuvastatin: FLAGGED (TPSA) as expected |

**Pass condition**: ✅ Medicinal chemist agrees flags are conservative, not arbitrary.

---

## 4. 🧬 Known-Drug Sanity Panel

| Target | Drugs Tested | ADME Result | Status |
|--------|--------------|-------------|--------|
| EGFR | Gefitinib, Erlotinib, Lapatinib | All SAFE | ✅ |
| BRAF | Vemurafenib, Dabrafenib | All SAFE | ✅ |
| HMG-CoA Reductase | Atorvastatin, Rosuvastatin | SAFE + FLAGGED | ✅ |

| Checkpoint | Status |
|------------|--------|
| At least 3 targets tested | ✅ |
| Approved drugs rank high in Stage-1 | ✅ (mock Stage-1 for sanity) |
| ADME flags match known properties | ✅ |
| No special-casing or hard-coding | ✅ |

**Pass condition**: ✅ System respects known biology without memorizing it.

---

## 5. 📖 Narrative Layer Discipline

| Checkpoint | Status | Evidence |
|------------|--------|----------|
| Narrative summaries exist for every molecule | ✅ | `NarrativeSummaryGenerator` |
| Narratives only summarize existing explanations | ✅ | Uses `_build_context()` from schema |
| Narratives never introduce new reasoning | ✅ | Prompt: "STRICTLY on the provided data" |
| Narratives never alter labels or scores | ✅ | Labels come from agent outputs |
| Fallback exists if LLM is unavailable | ✅ | `_generate_rule_based()` method |

**Pass condition**: ✅ Removing the LLM does not change scientific conclusions.

---

## 6. 🚨 Failure Mode Transparency

| Failure Mode | Documented | Explains Why | Explains Value | Explains Next Steps |
|--------------|------------|--------------|----------------|---------------------|
| ADME FLAG | ✅ | ✅ | ✅ | ✅ |
| PAINS FLAG | ✅ | ✅ | ✅ | ✅ |
| Docking NOT_EVALUATED | ✅ | ✅ | ✅ | ✅ |
| High MW | ✅ | ✅ | ✅ | ✅ |
| High LogP | ✅ | ✅ | ✅ | ✅ |

**Documentation**: `stage2-failure-modes.md`

**Pass condition**: ✅ Failures increase trust instead of raising alarms.

---

## 7. 🧾 Provenance & Reproducibility

| Checkpoint | Status | Implementation |
|------------|--------|----------------|
| Stage-1 version hash recorded | ✅ | `STAGE1_VERSION = "phase1-stable"` |
| Phase-2 run config recorded | ✅ | `provenance` dict in pipeline output |
| Protein source logged (AlphaFold / PDB) | ✅ | `PreparedProtein.source` field |
| All outputs traceable to inputs | ✅ | Candidate ID → Stage-1 → Stage-2 chain |

**Pass condition**: ✅ Same inputs → same outputs, every time.

---

## 8. 🧭 Product Positioning (Critical)

| Checkpoint | Status | Location |
|------------|--------|----------|
| Phase 2 improves **decision clarity**, not accuracy | ✅ | Workflow docs, README |
| Structural validation is **explicitly deferred** | ✅ | NOT_EVALUATED with explanation |
| No slide/README/demo implies docking occurred | ✅ | Verified |

**Pass condition**: ✅ No mismatch between what the system does and what it claims.

---

# 🚪 GATE DECISION

## Phase 2 Exit Criteria

| Criterion | Status |
|-----------|--------|
| All boxes above are checked | ✅ |
| System is honest and conservative | ✅ |
| No hidden assumptions exist | ✅ |

## VERDICT: ✅ PHASE 2 COMPLETE

**Phase 2 meets the bar without docking.**  
Proceed to Phase 2.1 when ready — not because of pressure.

---

# 🔜 PHASE 2.1 — REAL DOCKING (NEXT GATE)

**Prerequisites (separate checklist):**

- [ ] OpenBabel or MGLTools installed
- [ ] PDBQT generation verified
- [ ] Charge model documented
- [ ] Docking outputs labeled PASS / FLAG / FAIL
- [ ] Docking explanations added (geometry, not scores)

---

## One-Line Rule

> **Never add a capability until you can explain its failure modes better than its successes.**

---

## Artifact References

| Document | Path |
|----------|------|
| Stage-1 Frozen Snapshot | `.agent/workflows/stage1-frozen-snapshot.md` |
| Explanation Schema | `.agent/workflows/stage2-explanation-schema.md` |
| Rules in Plain English | `.agent/workflows/stage2-rules-plain-english.md` |
| Failure Modes | `.agent/workflows/stage2-failure-modes.md` |
| UI Proposal | `.agent/workflows/stage2-ui-proposal.md` |
| Trust Validation Report | `outputs/enhanced_trust_validation_report.txt` |

---

**Signed off**: 2025-12-31  
**Git tag recommendation**: `phase2-mvp-complete`

---
description: Stage-2 locked configuration decisions - do not modify without explicit approval
---

# STAGE 2 — LOCKED CONFIGURATION

> These decisions are **final for MVP**.
> Do not deviate without explicit founder/PI approval.

---

## Locked Decisions

| Component | Decision | Rationale |
|-----------|----------|-----------|
| **Docking Engine** | AutoDock Vina (CLI) | Industry-standard, scriptable, deterministic, explainable |
| **Protein Source** | AlphaFold (primary), PDB (fallback) | Free, consistent, no API key needed |
| **Structure Caching** | Local cache mandatory | Reproducibility, speed, auditability |
| **Patent Search** | Deferred (stub only) | Noisy, not decision-critical for MVP |
| **LLM Usage** | Gemini (summarization only) | Narrator role, never decision-maker |
| **DeepDTA/PyTorch** | No changes | Stage-1 is frozen, Phase-2 is about trust |

---

## Execution Order (Immutable)

```
1. Top-K Selection
2. Protein prep + caching
3. Docking with Vina
4. ADME/Tox rule expansion + explanations
5. Aggregation
6. Narrative explanations (LLM)
7. Patent stub (deferred)
```

**Do NOT reorder these.**

---

## Technical Setup Requirements

### AutoDock Vina
- **Version**: 1.2.5 (stable)
- **Location**: `mvp/tools/vina/` (local, not PATH)
- **Download**: https://vina.scripps.edu/downloads/

### AlphaFold API
- **No API key needed**
- **Endpoint**: `https://alphafold.ebi.ac.uk/api/`
- **Cache by**: `{target_id}_{structure_source}_{version}`

### Gemini API
- **Already configured in `.env`**
- **Role**: Summarization ONLY
- **Forbidden**: Decisions, label changes, chemistry inference

---

## LLM Constraints (Reinforced)

LLM **may**:
- Turn rule outputs into readable explanations
- Summarize why a molecule passed/failed
- Produce user-facing narratives

LLM **may NOT**:
- Make pass/fail calls
- Change labels
- Infer chemistry
- Fill missing data

**Fallback**: If Gemini unavailable → show raw explanations (still acceptable)

---

## Phase 2.1 (Post-MVP)

After MVP validation, consider:
- **Patent search**: Lens.org API integration
- **Model improvements**: DeepDTA fine-tuning
- **Enhanced docking**: Flexible receptor, water molecules

---

## One Line Summary

> **Stage-2 is about explainability and risk flagging, not model upgrades.**

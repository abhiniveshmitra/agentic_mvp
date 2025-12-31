---
description: Stage-2 global principles - additive to Stage-1, non-negotiable constraints
---

# STAGE-2 GLOBAL PRINCIPLES (NON-NEGOTIABLE)

> These instructions are **additive** to Stage-1.
> Stage-1 logic is frozen and must not be modified.

---

## Core Principles

1. **Stage-2 never modifies Stage-1 scores, ranks, or percentiles**
2. **Stage-2 does not discover molecules**
3. **Stage-2 produces flags and annotations, not optimizations**
4. **Stage-2 agents may fail gracefully but must never fabricate**
5. **Stage-2 runs only on a Top-K subset selected deterministically**

---

## One Sentence to Internalize

> **Stage-2 agents do not think — they certify.**

---

## What the LLM is Allowed to Do

LLM **may**:
* Summarize patent results
* Generate human-readable explanations
* Format outputs

LLM **may not**:
* Invent structures
* Override flags
* Re-rank molecules
* Smooth scores
* Decide "best" candidate

---

## Failure Modes (Designed, Not Bugs)

* Docking fails → still valid output
* High affinity + high ADME risk → valuable insight
* Patent encumbered + strong binder → business decision

These are **intentional outcomes**, not errors.

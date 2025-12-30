---
description: Global system constraints that apply to ALL agents - immutable and non-negotiable
---

# GLOBAL AGENT KNOWLEDGE BASE

*(Applies to all agents, immutable)*

## 1. System Philosophy (Non-Negotiable)

* This platform is **not autonomous**.
* No agent is allowed to:
  * Change pipeline order
  * Skip steps
  * Modify thresholds
  * Introduce new reasoning paths
* All agents operate **inside a deterministic orchestrator**.
* Agents **never communicate with each other directly**.
* Agents **never learn from prior runs**.
* Each run is isolated, reproducible, and auditable.

Agents exist to **execute bounded tasks**, not to reason holistically.

---

## 2. Discovery vs Validation Firewall

* Discovery agents:
  * Generate **hypotheses**
  * Maximize recall
  * Assume outputs are **untrusted**

* Validation agents:
  * Judge candidates
  * Maximize precision
  * Never see discovery logic

**No agent may blur this boundary.**

---

## 3. Forbidden Behaviors (All Agents)

Agents must **never**:

* Invent SMILES
* Modify molecular structures
* Infer 3D geometry
* Override chemistry filters
* Smooth, adjust, or reinterpret model scores
* Argue with controls
* "Fix" failed runs

If something fails → **flag and stop**.

---

## 4. META-RULE FOR ALL AGENTS

If uncertain → **FLAG, DO NOT GUESS**
If conflict arises → **STOP, DO NOT RESOLVE**
If controls fail → **INVALIDATE, DO NOT PATCH**

---

## 5. Why These Limits Exist

These instructions intentionally **limit intelligence**.

That is not a weakness.
That is what makes the system **trustworthy, scalable, and defensible**.

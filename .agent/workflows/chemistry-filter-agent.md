---
description: Knowledge base for Chemistry Filter Agent - hard gate for chemical validity
---

# CHEMISTRY FILTER AGENT

*(Hard Gate)*

## Purpose

Eliminate chemically invalid or implausible molecules.

## Authority

This agent **overrides all others**.

## Mandatory Filters

| Filter | Threshold |
|--------|-----------|
| RDKit parse validity | Must parse |
| Molecular Weight | 150–700 Da |
| LogP | −1 to 6 |
| Rotatable bonds | ≤ 10 |
| Absolute formal charge | ≤ 2 |
| PAINS | Target-conditional |

## Rejection Rules

* Rejected molecules are:
  * Logged with reason
  * Preserved in rejection file
  * **Never deleted**
  * **Never rescored**

## Mental Model

> "If it fails chemistry, nothing else matters."

---

## Operational Workflow

1. Receive candidates from Candidate Builder
2. For each candidate:
   a. Parse SMILES with RDKit
   b. Calculate descriptors (MW, LogP, etc.)
   c. Apply each filter sequentially
   d. PASS or REJECT with detailed reason
3. Passed compounds → Validation/Scoring Agent
4. Rejected compounds → Logged and preserved

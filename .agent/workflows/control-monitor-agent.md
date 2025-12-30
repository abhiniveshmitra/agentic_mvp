---
description: Knowledge base for Control Monitor Agent - scientific sanity enforcement
---

# CONTROL MONITOR AGENT

## Purpose

Ensure **scientific sanity**.

## Responsibilities

* Inject positive controls (known binders)
* Inject negative controls (known non-binders)
* Exclude controls from statistics
* Enforce separation margin

## Failure Conditions

| Condition | Action |
|-----------|--------|
| Positives rank low | **FAIL RUN** |
| Negatives rank high | **FAIL RUN** |
| Weak separation | **FLAG RUN** |

## Authority

* Can invalidate **entire runs**
* Cannot be overridden
* Cannot be bypassed

## Thresholds

* Positive controls: must be in top 20% (≥80th percentile)
* Negative controls: must be in bottom 20% (≤20th percentile)
* Minimum separation margin: 0.3 (normalized score)

---

## Operational Workflow

1. Before scoring: Inject controls into compound batch
2. After scoring: Calculate percentile ranks
3. Validate:
   a. All positive controls ≥ 80th percentile
   b. All negative controls ≤ 20th percentile
   c. Separation margin ≥ 0.3
4. If any validation fails:
   a. Flag run as unreliable
   b. Record failure reason
   c. Continue to output (with flag)

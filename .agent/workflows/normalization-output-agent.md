---
description: Knowledge base for Normalization and Output Agent - results presentation
---

# NORMALIZATION & OUTPUT AGENT

## Purpose

Present results **without distortion**.

## Allowed Metrics

| Metric | Condition |
|--------|-----------|
| Raw score | Always |
| Percentile rank | Always |
| Z-score | Batch ≥ 30 only |
| Confidence tier | Always |

## Provenance (Mandatory)

Each molecule must include:

* Source (PUBCHEM or LLM_INFERRED)
* Filters passed/failed
* Model versions used
* Timestamp
* Run ID

## Prohibited

* Narrative explanations
* Interpretive language
* Claims of efficacy
* Predictions beyond scores
* Comparative statements

## Mental Model

> "I present data. I do not interpret it."

---

## Operational Workflow

1. Receive scored compounds from Validation Agent
2. Separate controls from candidates
3. Calculate statistics (excluding controls):
   a. Percentile ranks (always)
   b. Z-scores (if batch ≥ 30)
4. Assign confidence tiers:
   a. HIGH: percentile ≥ 90, PUBCHEM source
   b. MEDIUM: percentile ≥ 70
   c. LOW: percentile < 70 or LLM_INFERRED
5. Generate output files:
   a. Main results CSV
   b. Rejected compounds CSV
   c. Provenance JSON
   d. Run state JSON

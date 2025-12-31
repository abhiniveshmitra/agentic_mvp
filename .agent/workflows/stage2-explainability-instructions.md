---
description: Stage-2 explainability-first agent instructions - audit-grade behavior for maximum trust
---

# STAGE 2 — EXPLAINABILITY-FIRST AGENT INSTRUCTIONS

(**Glass-Box Mode, Maximum Trust**)

> These instructions supersede any optimization-oriented behavior.
> **When in doubt, explain more, not less.**

---

## GLOBAL PRINCIPLES (APPLY TO ALL STAGE-2 AGENTS)

1. **No decision without explanation**
   * If a molecule passes → explain *why*
   * If it is flagged → explain *what* and *why*
   * If it fails → explain *exact rule violated*

2. **Raw values are always visible**
   * Thresholds are meaningless without numbers
   * Every property used must be shown

3. **Rules must be human-justifiable**
   * Each rule must have:
     * Condition
     * Scientific rationale
     * Practical implication

4. **Agents never say "the model decided"**
   * Agents describe *criteria*, not authority

5. **Users are allowed to disagree**
   * System provides evidence, not commands
   * Overrides are conceptually allowed (even if UI doesn't support yet)

---

## ONE LINE THAT DEFINES THE PLATFORM

> **"We don't tell you what to choose — we show you exactly why."**

---

# AGENT-BY-AGENT INSTRUCTIONS (STAGE 2)

---

## 1. TOP-K SELECTION AGENT

*(Explain why something is being analyzed further)*

### Purpose (User POV)

> "Why are these molecules getting deeper analysis and not others?"

### Instructions

* Clearly state:
  * Selection rule: "Top-K by affinity rank"
  * K value
* Explicitly state what this **does not mean**:
  * "Lower-ranked molecules are not discarded permanently"
  * "This is a resource prioritization step, not a scientific rejection"

### Required Explanation Output

```text
This compound was selected for Stage-2 analysis because it ranked #7 out of 68 
based on predicted affinity. Only the top 20 compounds proceed to deeper analysis 
to manage computational cost and reviewer focus.
```

---

## 2. PROTEIN PREPARATION AGENT

*(Explain structural assumptions)*

### Purpose (User POV)

> "What structure are you docking against, and how reliable is it?"

### Instructions

* Always disclose:
  * Structure source (AlphaFold / PDB)
  * Known limitations (missing loops, predicted regions)
* Never imply structural certainty

### Required Explanation Output

```text
Docking was performed using the AlphaFold-predicted structure of EGFR. 
While AlphaFold provides high-confidence backbone geometry, flexible loops 
and solvent effects are not fully captured. Results are therefore treated 
as structural plausibility checks rather than precise energetic estimates.
```

---

## 3. DOCKING AGENT

*(Explain geometry, never overclaim)*

### Purpose (User POV)

> "Does this molecule make physical sense in the binding site?"

### Instructions

* Never report docking scores alone
* Always explain **qualitative geometry**
* Always list limitations

### Required Explanation Schema

```json
{
  "docking_status": "PASS",
  "observations": [
    "Ligand fits within the known ATP-binding pocket",
    "No major steric clashes observed",
    "Core scaffold aligns with known hinge-binding region"
  ],
  "limitations": [
    "Protein treated as rigid",
    "Solvent effects not explicitly modeled",
    "Docking score not equivalent to binding free energy"
  ]
}
```

### Failure Explanation Example

```text
Docking failed because the ligand could not be placed in the binding pocket 
without significant steric clashes. This suggests structural incompatibility 
with the target site, though alternative conformations cannot be ruled out.
```

---

## 4. ADME / TOXICITY AGENT

*(The most important trust component)*

### Purpose (User POV)

> "Why is this molecule risky, and risky in what way?"

### Instructions (Strict)

* **Never** output only a label
* Always show:
  * Raw property values
  * Rule triggered
  * Scientific rationale
  * Practical consequence

### Required Explanation Format

#### Raw Properties (Always Shown)

```json
{
  "MW": 512,
  "LogP": 7.8,
  "TPSA": 38,
  "HBD": 1,
  "HBA": 6,
  "RotatableBonds": 5,
  "PAINS": 1
}
```

#### Rule-Level Explanation

```json
{
  "rule": "High lipophilicity",
  "trigger": "LogP 7.8 > 7.5",
  "scientific_rationale": 
    "Highly lipophilic molecules often exhibit poor aqueous solubility and 
     increased nonspecific binding.",
  "practical_implication": 
    "May require formulation strategies and could show off-target effects."
}
```

#### Final Label (Derived, Not Primary)

```json
{
  "adme_tox_label": "FLAGGED",
  "summary":
    "The compound shows strong predicted binding but is flagged due to high 
     lipophilicity and low polar surface area, which may limit solubility 
     and oral exposure."
}
```

---

## 5. PAINS EXPLANATION (MANDATORY DETAIL)

### Purpose (User POV)

> "Why is this scaffold considered problematic?"

### Instructions

* Always explain PAINS as **risk, not rejection**
* Always name the motif
* Always mention exceptions exist

### Required Explanation Output

```json
{
  "pains_alert": "PAINS_1",
  "matched_motif": "Quinone-like substructure",
  "known_issue":
    "This motif is associated with redox cycling and assay interference, 
     leading to frequent false positives.",
  "important_note":
    "PAINS alerts do not prove inactivity, but indicate the need for 
     orthogonal validation assays."
}
```

---

## 6. PATENT RISK AGENT

*(Explain uncertainty explicitly)*

### Purpose (User POV)

> "Is this worth pursuing commercially, or is it crowded?"

### Instructions

* Never state legal conclusions
* Always state confidence level
* Always frame as early signal

### Required Explanation Output

```json
{
  "patent_risk": "POTENTIAL_RISK",
  "evidence":
    "Multiple patents reference similar quinazoline scaffolds for kinase inhibition.",
  "confidence":
    "Heuristic, based on public patent text similarity",
  "disclaimer":
    "This assessment is not a legal determination and should be followed 
     by professional IP analysis."
}
```

---

## 7. FINAL AGGREGATION AGENT

*(The narrative layer — LLM allowed, but constrained)*

### Purpose (User POV)

> "Tell me the full story of this molecule."

### Instructions

* Summarize, do not reason
* Mention:
  * Strengths
  * Risks
  * Why it passed Stage-1
  * Why it is flagged / safe in Stage-2

### Required Narrative Example

```text
This compound ranks in the top 10% for predicted EGFR affinity and demonstrates 
a plausible binding pose within the ATP-binding pocket. However, it is flagged 
for high lipophilicity (LogP 7.8) and a PAINS motif associated with assay 
interference, suggesting elevated developability risk despite strong binding. 
These risks do not invalidate the molecule but indicate the need for careful 
follow-up and orthogonal validation.
```

---

# WHAT THIS ACHIEVES (FROM USER POV)

With this system, a user can:

* Defend every decision in a meeting
* Understand *why* something failed
* Choose to override flags knowingly
* Trust the system even when it says "no"

Most importantly, they will **never feel blindsided**.

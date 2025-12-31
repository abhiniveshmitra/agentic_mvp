# Stage-2 Expected Failure Modes

> **Purpose**: Document expected failure modes so they are understood as *informative outcomes*, not bugs.

---

## Failure Mode 1: Docking Failure (Structural Incompatibility)

### Example

**Compound**: Benzene (too small)  
**SMILES**: `c1ccccc1`

### What Happens

- Docking agent cannot find stable pose
- Binding site is too large for benzene
- Result: `DOCKING_STATUS: FAIL`

### Raw Values

```json
{
  "docking_status": "FAIL",
  "pose_available": false,
  "error": "Ligand too small for binding pocket"
}
```

### Why This is Expected

Benzene is a fragment-like molecule with MW ~78 Da. Kinase binding pockets are designed for larger drug-like molecules (300-700 Da). The failure is informative:

- ✓ Confirms the docking agent correctly identifies structural incompatibility
- ✓ Demonstrates that not all SMILES produce poses
- ✓ Shows failure is logged with explanation, not hidden

### Scientist Takeaway

> "Docking failure for fragments is expected. The system correctly identifies that benzene cannot occupy the ATP-binding site of EGFR."

---

## Failure Mode 2: High LogP (Lipophilicity Risk)

### Example

**Compound**: Octadecane (hypothetical lipophilic compound)  
**SMILES**: `CCCCCCCCCCCCCCCCCC`

### What Happens

- ADME/Tox agent calculates LogP
- LogP value exceeds threshold (7.5)
- Rule `HIGH_LIPOPHILICITY` triggers
- Result: `ADME_TOX_LABEL: HIGH_RISK`

### Raw Values

```json
{
  "mw": 254.5,
  "logp": 9.58,
  "tpsa": 0.0,
  "hbd": 0,
  "hba": 0,
  "rotatable_bonds": 15
}
```

### Rules Triggered

```json
{
  "rule_id": "HIGH_LIPOPHILICITY",
  "condition": "LogP 9.58 > 7.5",
  "triggered": true,
  "scientific_rationale": "Highly lipophilic molecules often exhibit poor aqueous solubility, increased nonspecific membrane binding, and potential accumulation in fatty tissues.",
  "practical_implication": "May require advanced formulation strategies. Increased risk of off-target effects.",
  "what_it_does_not_mean": "Does NOT prove the molecule is toxic. Many approved drugs exceed this threshold."
}
```

### Why This is Expected

Octadecane is an 18-carbon alkane with no polar groups. It is not a drug candidate - this is a stress test demonstrating that:

- ✓ LogP calculation is working correctly
- ✓ Threshold triggers at expected value
- ✓ Full explanation is provided with scientific rationale
- ✓ "What it does NOT mean" clarifies limitations

### Scientist Takeaway

> "HIGH_RISK for LogP is a developability flag, not a toxicity prediction. The system correctly identifies that formulation would be challenging."

---

## Failure Mode 3: PAINS-Flagged Scaffold

### Example

**Compound**: Rhodanine scaffold  
**SMILES**: `O=C1NC(=S)SC1=Cc1ccccc1`

### What Happens

- ADME/Tox agent runs PAINS filter
- Rhodanine motif is detected
- Result: `ADME_TOX_LABEL: FLAGGED`

### Raw Values

```json
{
  "mw": 221.3,
  "logp": 1.89,
  "tpsa": 41.1,
  "pains_count": 1
}
```

### PAINS Details

```json
{
  "pains_alert": "Rhodanine_A",
  "matched_motif": "Rhodanine core",
  "known_issue": "Known pan-assay interference compound with poor selectivity.",
  "important_note": "PAINS alerts indicate risk, not proof of inactivity. Orthogonal validation assays are recommended."
}
```

### Why This is Expected

Rhodanine is a well-documented PAINS motif:

- Associated with assay interference and false positives
- Often shows promiscuous binding
- Does NOT mean the molecule is inactive

The flag is informative:

- ✓ Signals need for orthogonal validation (SPR, ITC, cellular assays)
- ✓ Does not reject the molecule outright
- ✓ Provides clear explanation of what PAINS means

### Scientist Takeaway

> "PAINS flag for rhodanine is expected. This doesn't reject the molecule, but signals that binding should be confirmed with orthogonal methods."

---

## Failure Mode 4: High Molecular Weight

### Example

**Compound**: Large macrocycle (cyclic peptide-like)  
**SMILES**: Long cyclic peptide

### What Happens

- ADME/Tox agent calculates MW
- MW exceeds 700 Da threshold
- Rule `EXCESSIVE_MW` triggers
- Result: `ADME_TOX_LABEL: HIGH_RISK`

### Raw Values

```json
{
  "mw": 952.3,
  "logp": 4.2,
  "rotatable_bonds": 18,
  "hbd": 8,
  "hba": 12
}
```

### Why This is Expected

Large peptidic molecules:

- Often have poor oral bioavailability
- May require parenteral administration
- Can still be successful drugs (cyclosporin, daptomycin)

The flag is informative:

- ✓ Identifies that oral delivery is challenging
- ✓ Does not claim the molecule is undevelopable
- ✓ Suggests considering delivery route alternatives

### Scientist Takeaway

> "HIGH_RISK for MW suggests oral bioavailability challenges. Cyclosporin (MW 1,203) is proof that MW alone doesn't disqualify a molecule."

---

## Summary: Failures Are Features

| Failure Mode | What It Means | What It Does NOT Mean |
|--------------|---------------|----------------------|
| Docking FAIL | Structural incompatibility detected | Molecule cannot bind |
| High LogP | Formulation challenges likely | Molecule is toxic |
| PAINS Alert | Orthogonal validation needed | Molecule is inactive |
| High MW | Oral delivery challenging | Molecule is undevelopable |

**Key Principle**: Every failure produces an explanation. Silent failures are bugs; explained failures are features.

---

## Definition of Trust

A skeptical scientist can trace every decision:

1. **Input**: What data entered the pipeline?
2. **Rules**: What thresholds were applied?
3. **Trigger**: Which rules fired and why?
4. **Output**: What label was assigned?
5. **Limitations**: What can this assessment NOT tell us?

Stage-2 is trusted when all five questions are answered for every molecule.

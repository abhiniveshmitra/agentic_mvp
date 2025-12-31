---
description: Stage-2 ADME/Tox and Docking rules in plain English with rationale
---

# STAGE-2 RULES IN PLAIN ENGLISH

> All rules documented before coding.
> Each rule has: condition, rationale, and what it does NOT mean.

---

## ADME/TOX RULES

### Rule 1: High Lipophilicity

**Condition**: LogP > 7.5

**Scientific Rationale**: Highly lipophilic molecules often exhibit:
- Poor aqueous solubility
- Increased nonspecific membrane binding
- Higher plasma protein binding
- Potential for accumulation in fatty tissues

**Practical Implication**: 
- May require advanced formulation strategies
- Increased risk of off-target effects
- Potential bioavailability issues

**What This Does NOT Mean**:
- Does not prove the molecule is toxic
- Does not mean oral administration is impossible
- Many approved drugs exceed this threshold (e.g., some statins)

---

### Rule 2: Low Polar Surface Area + High LogP

**Condition**: TPSA < 40 AND LogP > 6

**Scientific Rationale**:
- Low TPSA indicates reduced hydrogen bonding capacity
- Combined with high LogP, suggests poor aqueous solubility
- May partition into membranes rather than staying in solution

**Practical Implication**:
- Dissolution-limited absorption likely
- May require solubilization technology

**What This Does NOT Mean**:
- Does not predict absolute solubility
- Some CNS drugs intentionally have low TPSA

---

### Rule 3: High TPSA (Permeability Risk)

**Condition**: TPSA > 140

**Scientific Rationale**:
- High TPSA correlates with poor passive membrane permeability
- Molecules may not cross intestinal epithelium effectively

**Practical Implication**:
- May require active transport
- Parenteral administration may be needed

**What This Does NOT Mean**:
- Does not rule out efficacy
- Biologics and some antibiotics work despite high TPSA

---

### Rule 4: PAINS Alert

**Condition**: PAINS count ≥ 1

**Scientific Rationale**:
- PAINS (Pan-Assay Interference Compounds) contain substructures that:
  - May react nonspecifically
  - Can interfere with assay readouts
  - Often produce false positives

**Practical Implication**:
- Orthogonal validation assays required
- SPR, ITC, or cellular assays recommended

**What This Does NOT Mean**:
- Does NOT prove the molecule is inactive
- Many approved drugs contain PAINS motifs
- PAINS is a flag for caution, not rejection

---

### Rule 5: Toxicity Alert (Reactive Groups)

**Condition**: Contains any of 12 reactive SMARTS patterns

**Scientific Rationale**:
- Reactive functional groups can:
  - Covalently modify proteins
  - Cause idiosyncratic toxicity
  - Lead to genotoxicity

**Practical Implication**:
- Requires careful selectivity assessment
- May need Ames test and reactive metabolite screening

**What This Does NOT Mean**:
- Does NOT mean the molecule is toxic in humans
- Many covalent drugs are successful (e.g., aspirin, penicillins)
- Risk depends on selectivity, not mere presence

---

### Rule 6: High Flexibility

**Condition**: Rotatable bonds > 12

**Scientific Rationale**:
- Highly flexible molecules have:
  - Larger conformational entropy penalty on binding
  - Reduced oral bioavailability
  - Potential for multiple binding modes

**Practical Implication**:
- May show lower potency than expected
- Consider rigidifying scaffold

**What This Does NOT Mean**:
- Does NOT prevent binding
- Some peptide-like drugs are flexible

---

### Rule 7: Low Fsp3

**Condition**: Fraction Csp3 < 0.1

**Scientific Rationale**:
- Low Fsp3 indicates a flat, aromatic molecule
- Flat molecules associated with:
  - Promiscuous binding
  - Off-target effects
  - Poor solubility

**Practical Implication**:
- Consider 3D character in optimization

**What This Does NOT Mean**:
- Many kinase inhibitors are aromatic and effective
- Context matters

---

## DOCKING RULES

### Rule D1: Steric Clash

**Condition**: Major atomic overlap with protein

**Scientific Rationale**:
- Severe steric clashes indicate the ligand cannot physically occupy the binding site without deforming the protein

**Practical Implication**:
- Suggests structural incompatibility
- May still bind with induced fit (not modeled)

**Limitations**:
- Protein treated as rigid
- Does not account for conformational flexibility

---

### Rule D2: Binding Pocket Fit

**Condition**: Ligand fits within defined binding pocket

**Scientific Rationale**:
- Ligand volume must be compatible with pocket volume
- Scaffold should align with known pharmacophore features

**What This Does NOT Mean**:
- Fit does NOT prove high affinity
- Docking score is NOT binding free energy

---

### Rule D3: Hydrogen Bonding Network

**Condition**: H-bonds formed with key residues (if applicable)

**Scientific Rationale**:
- For kinases, hinge region H-bonds are critical
- Absence may indicate weak or alternative binding

**What This Does NOT Mean**:
- Some inhibitors bind without traditional H-bonds
- Hydrophobic interactions can compensate

---

## PATENT RISK RULES

### Rule P1: Scaffold Similarity

**Condition**: Similar scaffold found in patent literature

**Scientific Rationale**:
- Structural similarity to patented compounds may indicate IP overlap

**Practical Implication**:
- Professional IP analysis required before development

**What This Does NOT Mean**:
- Similarity does NOT mean infringement
- Patents have specific claims, not blanket coverage

---

## ONE-LINER SUMMARY

> **Every rule is a flag, not a verdict. Every flag comes with an explanation.**

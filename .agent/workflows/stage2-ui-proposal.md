# Stage-2 Streamlit UI Layout Proposal

> **Purpose**: Design a UI that maximizes trust through transparency.  
> **Principle**: Explain everything, collapse by default.

---

## Layout Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                    🧬 Drug Discovery Platform                        │
│                         Stage-2 Results                              │
├─────────────────────────────────────────────────────────────────────┤
│ ┌─────────────────────────────────────────────────────────────────┐ │
│ │ 📊 Pipeline Summary                                              │ │
│ │   Candidates: 20 | SAFE: 8 | FLAGGED: 10 | HIGH_RISK: 2         │ │
│ │   Docking: 15 PASS | 3 FLAG | 2 FAIL                            │ │
│ └─────────────────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│ ┌───────────────────────────────────────────────────────────────┐   │
│ │ 🏆 Candidate #1: Gefitinib                          [SAFE] ✓  │   │
│ │ Stage-1 Rank: #1 | Percentile: 95% | Score: 8.5               │   │
│ ├───────────────────────────────────────────────────────────────┤   │
│ │ 📖 Narrative Summary                                          │   │
│ │ "Gefitinib ranks #1 in Stage-1 affinity prediction..."        │   │
│ ├───────────────────────────────────────────────────────────────┤   │
│ │ ▶ Stage-2 Details (collapsed by default)                      │   │
│ │   ├─ 🔬 Docking [PASS]                                        │   │
│ │   ├─ 💊 ADME/Tox [SAFE]                                       │   │
│ │   └─ 📝 Patent [NOT_EVALUATED]                                │   │
│ └───────────────────────────────────────────────────────────────┘   │
│                                                                     │
│ ┌───────────────────────────────────────────────────────────────┐   │
│ │ 🏆 Candidate #2: Lapatinib                      [FLAGGED] ⚠   │   │
│ │ Stage-1 Rank: #3 | Percentile: 85% | Score: 7.9               │   │
│ ├───────────────────────────────────────────────────────────────┤   │
│ │ 📖 Narrative Summary                                          │   │
│ │ "Lapatinib flags for high molecular weight..."                │   │
│ ├───────────────────────────────────────────────────────────────┤   │
│ │ ▶ Stage-2 Details (expand to see rules)                       │   │
│ └───────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Component Hierarchy

### 1. Header (Always Visible)

- Platform title
- Target name (e.g., "EGFR Inhibitors")
- Run timestamp

### 2. Pipeline Summary (Always Visible)

- Total candidates analyzed
- Breakdown by Stage-2 label (SAFE / FLAGGED / HIGH_RISK)
- Breakdown by docking status (PASS / FLAG / FAIL)

### 3. Candidate Cards (List View)

Each candidate shown as a card with:

#### Card Header (Always Visible)
- Rank badge (#1, #2, etc.)
- Compound name
- Stage-2 label with color:
  - 🟢 SAFE (green)
  - 🟡 FLAGGED (yellow)
  - 🔴 HIGH_RISK (red)
- Stage-1 metrics (rank, percentile, score)

#### Narrative Summary (Always Visible)
- 3-5 sentence summary
- Highlights key findings

#### Expandable Sections (Collapsed by Default)

```
▶ Docking Details
   ├─ Status: PASS / FLAG / FAIL
   ├─ Best Score: -8.2 kcal/mol
   ├─ Observations: ["Ligand fits binding pocket"]
   └─ Limitations: ["Protein treated as rigid"]

▶ ADME/Tox Details
   ├─ Label: SAFE / FLAGGED / HIGH_RISK
   ├─ Raw Properties:
   │   MW: 446.9 | LogP: 3.75 | TPSA: 68.7
   │   HBD: 1 | HBA: 7 | RotB: 6
   ├─ Rules Triggered: (if any)
   │   └─ [Rule ID]: [Condition]
   │      Rationale: [Scientific explanation]
   │      Implication: [What this means]
   ├─ PAINS: None detected
   └─ Limitations: ["Rule-based only"]

▶ Patent Risk
   ├─ Status: NOT_EVALUATED
   └─ Note: "Patent analysis deferred in MVP"

▶ Full Explanation JSON (for power users)
   └─ [Raw JSON dump]
```

---

## Streamlit Components

### Main Page Structure

```python
import streamlit as st

st.set_page_config(page_title="Stage-2 Results", layout="wide")

# Header
st.title("🧬 Drug Discovery Platform")
st.subheader("Stage-2 Validation Results")

# Summary metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("Candidates", 20)
col2.metric("SAFE", 8, delta=None)
col3.metric("FLAGGED", 10, delta=None)
col4.metric("HIGH_RISK", 2, delta=None)

# Candidate cards
for candidate in candidates:
    with st.expander(
        f"#{candidate['rank']} {candidate['name']} [{candidate['label']}]",
        expanded=False
    ):
        # Narrative
        st.markdown(f"**Summary**: {candidate['narrative']}")
        
        # Tabs for details
        tab1, tab2, tab3 = st.tabs(["Docking", "ADME/Tox", "Patent"])
        
        with tab1:
            st.json(candidate['docking'])
        
        with tab2:
            st.json(candidate['adme_tox'])
        
        with tab3:
            st.json(candidate['patent'])
```

---

## Design Principles

### 1. Stage-1 Ranking is Primary

- Candidates are always shown in Stage-1 rank order
- Stage-2 labels are overlays, not re-rankings
- No sorting by Stage-2 labels

### 2. Progressive Disclosure

- Narrative visible by default (high-level overview)
- Details hidden until requested
- Full JSON available for power users

### 3. Color Semantics

| Color | Label | Meaning |
|-------|-------|---------|
| 🟢 Green | SAFE | No concerns detected |
| 🟡 Yellow | FLAGGED | Moderate risk, review advised |
| 🔴 Red | HIGH_RISK | Significant concerns |
| ⬜ Gray | NOT_EVALUATED | Assessment not performed |

### 4. Trace Everything

Every UI element should answer:
- What data produced this?
- What rule was applied?
- What does this mean?

### 5. No Hidden Information

- All failures shown with explanation
- No silently dropped candidates
- Limitations always visible

---

## Interactions

### Filter by Stage-2 Label

```
[Show All] [SAFE Only] [FLAGGED Only] [HIGH_RISK Only]
```

Filters visibility, never changes order.

### Search by Compound Name

```
🔍 Search compounds...
```

### Export Options

```
[Export JSON] [Export CSV] [Export Report (PDF)]
```

---

## Mobile Considerations

- Cards stack vertically
- Expanders work natively
- Summary bar becomes scrollable

---

## Implementation Notes

### Files to Create

1. `interface/streamlit_stage2.py` - Main Stage-2 results page
2. `interface/components/candidate_card.py` - Reusable card component
3. `interface/components/explanation_panel.py` - Expandable explanation panel

### Integration Points

- Load results from `stage2/pipeline.py`
- Use `narrative_generator.py` for summaries
- Apply `trust_validation.py` format for reports

---

## Definition of Done (UI)

- [ ] Every candidate has visible Stage-1 rank
- [ ] Every candidate shows Stage-2 label with color
- [ ] Narrative summary visible without clicking
- [ ] Full explanation available on expand
- [ ] Raw values visible for each assessment
- [ ] Rules triggered shown with rationale
- [ ] Limitations clearly stated
- [ ] No candidate is hidden

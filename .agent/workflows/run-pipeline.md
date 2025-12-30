---
description: How to run the complete drug discovery pipeline
---

# Running the Drug Discovery Pipeline

## Prerequisites

1. Python 3.10+ installed
2. Virtual environment activated
3. Dependencies installed: `pip install -r requirements.txt`
4. `.env` file configured with API keys

## Quick Start

```bash
# Navigate to project directory
cd c:\Users\abhin\OneDrive\Desktop\antigravity-mvp\mvp

# Activate virtual environment
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Run the Streamlit dashboard
streamlit run interface/streamlit_app.py
```

## Pipeline Execution Order (Immutable)

1. **Initialization** - Lock configuration, generate run ID
2. **Discovery** - PubMed search, Gemini extraction, PubChem matching
3. **Chemistry Filters** - RDKit validation, property filters, PAINS
4. **Scoring** - DeepDTA inference
5. **Normalization** - Percentiles, Z-scores, control validation
6. **Output** - CSV, JSON, provenance files

## Output Files

All outputs are saved to `outputs/` directory:

* `results_{run_id}.csv` - Main ranked results
* `rejected_{run_id}.csv` - Rejected compounds
* `provenance_{run_id}.json` - Full provenance data
* `run_state_{run_id}.json` - Execution state
* `run_config_{run_id}.json` - Locked configuration

## Interpreting Results

| Column | Description |
|--------|-------------|
| rank | Position by raw score |
| raw_score | DeepDTA affinity prediction |
| percentile | Rank among batch (0-100) |
| z_score | Standard deviations from mean |
| uncertainty | Model confidence |
| confidence_tier | HIGH/MEDIUM/LOW |
| source | PUBCHEM or LLM_INFERRED |

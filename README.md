# AI Drug Discovery Platform - Phase 1 (Trustworthy Pilot)

A scientifically-defensible drug discovery MVP with strict Discovery/Validation separation.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the Streamlit dashboard
streamlit run interface/streamlit_app.py
```

## Architecture

```
                    ┌─────────────────┐
                    │   Streamlit UI  │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │   Orchestrator  │
                    │  (Deterministic)│
                    └────────┬────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
┌───────▼───────┐   ┌────────▼────────┐   ┌───────▼───────┐
│   Discovery   │   │    Validation   │   │    Output     │
│  (High Recall)│   │  (High Trust)   │   │  (Provenance) │
└───────────────┘   └─────────────────┘   └───────────────┘
```

## Pipeline Steps

1. **Initialization** - Lock configuration, generate run ID
2. **Discovery** - PubMed search → Gemini extraction → PubChem matching
3. **Chemistry Filters** - RDKit validation (MW, LogP, PAINS)
4. **Scoring** - DeepDTA affinity prediction
5. **Normalization** - Percentiles, Z-scores, control validation
6. **Output** - CSV results, provenance JSON

## Key Principles

- **Deterministic orchestration** - No LLMs in pipeline control
- **Discovery/Validation firewall** - Strict separation
- **Control-based validation** - Known binders/non-binders
- **Full provenance** - Every compound traced to source

## Configuration

Copy `.env.example` to `.env` and add your API keys:

```
GOOGLE_API_KEY=your_gemini_api_key
NCBI_EMAIL=your_email@example.com
```

## Testing

```bash
pytest tests/ -v
```

## License

MIT

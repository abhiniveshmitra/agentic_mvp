"""
Centralized Configuration for AI Drug Discovery Platform

All thresholds, paths, and settings are defined here.
This configuration is IMMUTABLE during a run.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# =============================================================================
# PATHS
# =============================================================================

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"
OUTPUTS_DIR = BASE_DIR / "outputs"

# Ensure directories exist
for dir_path in [DATA_DIR, MODELS_DIR, LOGS_DIR, OUTPUTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# =============================================================================
# API KEYS
# =============================================================================

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
NCBI_EMAIL = os.getenv("NCBI_EMAIL", "")

# =============================================================================
# CHEMISTRY FILTER THRESHOLDS (IMMUTABLE)
# =============================================================================

CHEMISTRY_FILTERS = {
    "molecular_weight": {
        "min": 120,  # Relaxed from 150 to allow fragment-like binders (Phase 1)
        "max": 700,
    },
    "logp": {
        "min": -1,
        "max": 8.0,  # Final widening for lipophilic kinase inhibitors
    },
    "rotatable_bonds": {
        "max": 10,
    },
    "formal_charge": {
        "max_abs": 2,
    },
    "pains_filter": False,  # Disabled for Phase 1 discovery (too aggressive for early literature)
}

# =============================================================================
# NORMALIZATION SETTINGS
# =============================================================================

NORMALIZATION = {
    "min_batch_for_zscore": 30,  # Z-score only if batch >= 30
    "exclude_controls_from_stats": True,
}

# =============================================================================
# CONTROL VALIDATION THRESHOLDS
# =============================================================================

CONTROL_VALIDATION = {
    "positive_control_min_percentile": 80,  # Must be in top 20%
    "negative_control_max_percentile": 20,  # Must be in bottom 20%
    "min_separation_margin": 0.3,  # Minimum separation in normalized score
}

# =============================================================================
# MODEL CONFIGURATIONS
# =============================================================================

MODELS = {
    "deepdta": {
        "version": "1.0",
        "checkpoint_path": MODELS_DIR / "checkpoints" / "deepdta_pretrained.pt",
        "max_drug_len": 100,
        "max_protein_len": 1000,
    },
    "gat": {
        "version": "1.0",
        "checkpoint_path": MODELS_DIR / "checkpoints" / "gat.pt",
        "num_layers": 2,
        "num_heads": 4,
        "hidden_channels": 64,
    },
    "xgboost": {
        "version": "1.0",
        "checkpoint_path": MODELS_DIR / "checkpoints" / "xgb_ecfp.joblib",
        "fp_radius": 2,
        "fp_bits": 2048,
    },
    "ensemble": {
        "enabled": True,
        "models": ["deepdta", "gat", "xgboost"],
    },
}

# =============================================================================
# API ENDPOINTS
# =============================================================================

PUBMED_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
PUBCHEM_BASE_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"

# =============================================================================
# PUBMED SETTINGS
# =============================================================================

PUBMED = {
    "max_results": 200,  # Increased from 100 for better discovery recall
    "retry_count": 3,
    "retry_delay": 1.0,  # seconds
}

# =============================================================================
# LOGGING
# =============================================================================

LOGGING = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": LOGS_DIR / "platform.log",
}


def validate_config():
    """Validate critical configuration on startup."""
    errors = []
    
    if not GOOGLE_API_KEY:
        errors.append("GOOGLE_API_KEY is not set in environment")
    
    if errors:
        raise ValueError(f"Configuration errors: {', '.join(errors)}")
    
    return True

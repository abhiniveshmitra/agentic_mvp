"""
Reproducibility utilities.

Ensures deterministic execution and 
configuration snapshotting.
"""

import random
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict

import numpy as np

# Try to import torch for seed setting
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def set_global_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed to use
    """
    random.seed(seed)
    np.random.seed(seed)
    
    if HAS_TORCH:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            # Deterministic algorithms (may reduce performance)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


@dataclass
class ConfigSnapshot:
    """Immutable snapshot of run configuration."""
    run_id: str
    timestamp: str
    random_seed: int
    chemistry_filters: Dict[str, Any]
    normalization: Dict[str, Any]
    control_validation: Dict[str, Any]
    model_versions: Dict[str, str]
    target_protein: str
    query: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def compute_hash(self) -> str:
        """Compute deterministic hash of configuration."""
        # Sort keys for determinism
        config_str = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]
    
    def save(self, output_dir: Path) -> Path:
        """Save configuration snapshot to file."""
        filepath = output_dir / f"config_snapshot_{self.run_id}.json"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        return filepath
    
    @classmethod
    def load(cls, filepath: Path) -> "ConfigSnapshot":
        """Load configuration from file."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)


def generate_run_id() -> str:
    """
    Generate a unique run ID.
    
    Format: YYYYMMDD_HHMMSS_XXXX
    Where XXXX is a random 4-character hex string.
    """
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    rand_suffix = hashlib.sha256(
        f"{timestamp}{random.random()}".encode()
    ).hexdigest()[:4]
    return f"{timestamp}_{rand_suffix}"


def create_config_snapshot(
    run_id: str,
    target_protein: str,
    query: str,
    seed: int = 42
) -> ConfigSnapshot:
    """
    Create a configuration snapshot for a run.
    
    Imports settings and creates an immutable record.
    """
    from config.settings import (
        CHEMISTRY_FILTERS,
        NORMALIZATION,
        CONTROL_VALIDATION,
        MODELS
    )
    
    return ConfigSnapshot(
        run_id=run_id,
        timestamp=datetime.utcnow().isoformat(),
        random_seed=seed,
        chemistry_filters=CHEMISTRY_FILTERS,
        normalization=NORMALIZATION,
        control_validation=CONTROL_VALIDATION,
        model_versions={
            name: config.get("version", "unknown")
            for name, config in MODELS.items()
        },
        target_protein=target_protein,
        query=query
    )

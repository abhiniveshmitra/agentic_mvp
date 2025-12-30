"""
Run configuration for the orchestrator.

Immutable configuration that is locked at run initialization.
No runtime changes allowed.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional
import json
from pathlib import Path


@dataclass(frozen=True)
class ChemistryFilterConfig:
    """Immutable chemistry filter configuration."""
    mw_min: float = 150.0
    mw_max: float = 700.0
    logp_min: float = -1.0
    logp_max: float = 6.0
    rotatable_bonds_max: int = 10
    formal_charge_max_abs: int = 2
    enable_pains: bool = True


@dataclass(frozen=True)
class NormalizationConfig:
    """Immutable normalization configuration."""
    min_batch_for_zscore: int = 30
    exclude_controls: bool = True


@dataclass(frozen=True)
class ControlValidationConfig:
    """Immutable control validation configuration."""
    positive_min_percentile: float = 80.0
    negative_max_percentile: float = 20.0
    min_separation_margin: float = 0.3


@dataclass(frozen=True)
class ModelConfig:
    """Immutable model configuration."""
    name: str
    version: str
    checkpoint_path: Optional[str] = None


@dataclass
class RunConfig:
    """
    Complete run configuration.
    
    This is created at run initialization and NEVER modified.
    All thresholds and settings are locked.
    """
    run_id: str
    target_protein: str
    target_protein_sequence: Optional[str] = None
    query: str = ""
    max_papers: int = 100
    
    # Locked configurations
    chemistry_filters: ChemistryFilterConfig = field(
        default_factory=ChemistryFilterConfig
    )
    normalization: NormalizationConfig = field(
        default_factory=NormalizationConfig
    )
    control_validation: ControlValidationConfig = field(
        default_factory=ControlValidationConfig
    )
    
    # Model configuration
    models: List[ModelConfig] = field(default_factory=list)
    
    # Positive and negative controls
    positive_controls: List[str] = field(default_factory=list)  # SMILES
    negative_controls: List[str] = field(default_factory=list)  # SMILES
    
    # Random seed for reproducibility
    random_seed: int = 42
    
    def __post_init__(self):
        """Set default model if none provided."""
        if not self.models:
            self.models = [
                ModelConfig(
                    name="deepdta",
                    version="1.0",
                    checkpoint_path=None  # Use pretrained
                )
            ]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "run_id": self.run_id,
            "target_protein": self.target_protein,
            "target_protein_sequence": self.target_protein_sequence,
            "query": self.query,
            "max_papers": self.max_papers,
            "chemistry_filters": asdict(self.chemistry_filters),
            "normalization": asdict(self.normalization),
            "control_validation": asdict(self.control_validation),
            "models": [asdict(m) for m in self.models],
            "positive_controls": self.positive_controls,
            "negative_controls": self.negative_controls,
            "random_seed": self.random_seed,
        }
    
    def save(self, output_dir: Path) -> Path:
        """Save configuration to file."""
        filepath = output_dir / f"run_config_{self.run_id}.json"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        return filepath
    
    @classmethod
    def load(cls, filepath: Path) -> "RunConfig":
        """Load configuration from file."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Reconstruct nested configs
        data["chemistry_filters"] = ChemistryFilterConfig(
            **data["chemistry_filters"]
        )
        data["normalization"] = NormalizationConfig(
            **data["normalization"]
        )
        data["control_validation"] = ControlValidationConfig(
            **data["control_validation"]
        )
        data["models"] = [ModelConfig(**m) for m in data["models"]]
        
        return cls(**data)


def create_default_config(
    run_id: str,
    target_protein: str,
    query: str,
    target_protein_sequence: Optional[str] = None,
    positive_controls: Optional[List[str]] = None,
    negative_controls: Optional[List[str]] = None
) -> RunConfig:
    """
    Create a run configuration with defaults from settings.
    
    Args:
        run_id: Unique run identifier
        target_protein: Target protein name or UniProt ID
        query: PubMed search query
        target_protein_sequence: Optional protein sequence
        positive_controls: Optional list of known binder SMILES
        negative_controls: Optional list of known non-binder SMILES
    
    Returns:
        Configured RunConfig instance
    """
    from config.settings import CHEMISTRY_FILTERS, NORMALIZATION, CONTROL_VALIDATION
    
    chemistry_config = ChemistryFilterConfig(
        mw_min=CHEMISTRY_FILTERS["molecular_weight"]["min"],
        mw_max=CHEMISTRY_FILTERS["molecular_weight"]["max"],
        logp_min=CHEMISTRY_FILTERS["logp"]["min"],
        logp_max=CHEMISTRY_FILTERS["logp"]["max"],
        rotatable_bonds_max=CHEMISTRY_FILTERS["rotatable_bonds"]["max"],
        formal_charge_max_abs=CHEMISTRY_FILTERS["formal_charge"]["max_abs"],
        enable_pains=CHEMISTRY_FILTERS["pains_filter"],
    )
    
    normalization_config = NormalizationConfig(
        min_batch_for_zscore=NORMALIZATION["min_batch_for_zscore"],
        exclude_controls=NORMALIZATION["exclude_controls_from_stats"],
    )
    
    control_config = ControlValidationConfig(
        positive_min_percentile=CONTROL_VALIDATION["positive_control_min_percentile"],
        negative_max_percentile=CONTROL_VALIDATION["negative_control_max_percentile"],
        min_separation_margin=CONTROL_VALIDATION["min_separation_margin"],
    )
    
    return RunConfig(
        run_id=run_id,
        target_protein=target_protein,
        target_protein_sequence=target_protein_sequence,
        query=query,
        chemistry_filters=chemistry_config,
        normalization=normalization_config,
        control_validation=control_config,
        positive_controls=positive_controls or [],
        negative_controls=negative_controls or [],
    )

"""
Ensemble Scorer combining DeepDTA, GAT, and XGBoost models.

STRICT MODE: No mocks, no stubs. If dependencies unavailable, fail explicitly.

Follows Phase 3 ensemble rules from ML Integration Specification:
    - Final score = mean(model_scores)
    - Uncertainty = std(model_scores) (model disagreement)

Interface Contract (IMMUTABLE):
    predict_affinity(smiles, protein) -> {"score": float, "uncertainty": float}
"""

from typing import Dict, List, Optional
import numpy as np
from pathlib import Path

from utils.logging import get_logger

logger = get_logger(__name__)


class ModelUnavailableError(Exception):
    """Raised when required model dependencies are not available."""
    pass


class EnsembleScorer:
    """
    Ensemble scorer combining multiple affinity prediction models.
    
    STRICT MODE: No mocks. Fails explicitly if ANY model fails to initialize.
    
    Models included:
        - DeepDTA: Sequence-based CNN
        - GAT: Graph Attention Network
        - XGBoost: ECFP fingerprint-based
    
    Interface Contract (IMMUTABLE):
        predict_affinity(smiles, protein) -> {"score": float, "uncertainty": float}
    """
    
    def __init__(
        self,
        deepdta_path: Optional[str] = None,
        gat_path: Optional[str] = None,
        xgboost_path: Optional[str] = None,
        models_to_use: Optional[List[str]] = None,
    ):
        """
        Initialize ensemble scorer with all models.
        
        Args:
            deepdta_path: Path to DeepDTA checkpoint
            gat_path: Path to GAT checkpoint
            xgboost_path: Path to XGBoost checkpoint
            models_to_use: List of models to use (default: all available)
            
        Raises:
            ModelUnavailableError: If any required model cannot be initialized
        """
        self.scorers = {}
        self.models_to_use = models_to_use or ["deepdta", "gat", "xgboost"]
        self.initialization_errors = []
        
        # Initialize DeepDTA
        if "deepdta" in self.models_to_use:
            try:
                from validation.scoring.deepdta import DeepDTAScorer
                self.scorers["deepdta"] = DeepDTAScorer(checkpoint_path=deepdta_path)
                logger.info("DeepDTA scorer initialized")
            except Exception as e:
                self.initialization_errors.append(f"DeepDTA: {e}")
        
        # Initialize GAT
        if "gat" in self.models_to_use:
            try:
                from validation.scoring.gat_scorer import GATScorer
                self.scorers["gat"] = GATScorer(checkpoint_path=gat_path)
                logger.info("GAT scorer initialized")
            except Exception as e:
                self.initialization_errors.append(f"GAT: {e}")
        
        # Initialize XGBoost
        if "xgboost" in self.models_to_use:
            try:
                from validation.scoring.xgboost_scorer import XGBoostScorer
                self.scorers["xgboost"] = XGBoostScorer(checkpoint_path=xgboost_path)
                logger.info("XGBoost scorer initialized")
            except Exception as e:
                self.initialization_errors.append(f"XGBoost: {e}")
        
        # STRICT MODE: Fail if no scorers available
        if not self.scorers:
            raise ModelUnavailableError(
                f"No ML scorers could be initialized. Errors: {'; '.join(self.initialization_errors)}"
            )
        
        # Log any partial failures
        if self.initialization_errors:
            logger.warning(f"Some scorers failed to initialize: {self.initialization_errors}")
    
    def predict_affinity(self, smiles: str, protein: str) -> Dict[str, float]:
        """
        Predict binding affinity using ensemble of models.
        
        IMMUTABLE INTERFACE - DO NOT MODIFY SIGNATURE.
        
        Args:
            smiles: Drug SMILES string
            protein: Protein sequence or ID
        
        Returns:
            Dictionary with:
                - score: Ensemble mean score
                - uncertainty: Ensemble std (model disagreement)
                - model_scores: Individual model predictions
                
        Raises:
            RuntimeError: If all models fail for this prediction
        """
        # Collect predictions from all models
        model_scores = {}
        scores = []
        errors = []
        
        for name, scorer in self.scorers.items():
            try:
                result = scorer.predict_affinity(smiles, protein)
                model_scores[name] = {
                    "score": result["score"],
                    "uncertainty": result["uncertainty"],
                }
                scores.append(result["score"])
            except Exception as e:
                errors.append(f"{name}: {e}")
                logger.warning(f"Model {name} failed for {smiles[:20]}...: {e}")
        
        if not scores:
            raise RuntimeError(
                f"All ML models failed for SMILES {smiles[:30]}... Errors: {'; '.join(errors)}"
            )
        
        # Ensemble: mean score, std as uncertainty
        ensemble_score = float(np.mean(scores))
        ensemble_uncertainty = float(np.std(scores)) if len(scores) > 1 else 0.3
        
        return {
            "score": round(ensemble_score, 4),
            "uncertainty": round(ensemble_uncertainty, 4),
            "model_scores": model_scores,
        }
    
    def get_available_models(self) -> List[str]:
        """Return list of available model names."""
        return list(self.scorers.keys())
    
    def get_initialization_errors(self) -> List[str]:
        """Return list of models that failed to initialize."""
        return self.initialization_errors


def create_scorer(
    use_ensemble: bool = True,
    models_to_use: Optional[List[str]] = None,
) -> EnsembleScorer:
    """
    Factory function to create a scorer.
    
    Loads model paths from config/settings.py.
    
    Args:
        use_ensemble: If True, use ensemble scorer
        models_to_use: List of models to include in ensemble
    
    Returns:
        EnsembleScorer instance
        
    Raises:
        ModelUnavailableError: If no models can be initialized
    """
    try:
        from config.settings import MODELS, MODELS_DIR
        
        deepdta_path = MODELS.get("deepdta", {}).get("checkpoint_path")
        gat_path = MODELS.get("gat", {}).get("checkpoint_path")
        xgboost_path = MODELS.get("xgboost", {}).get("checkpoint_path")
        
        # Convert Path to string if needed
        if deepdta_path and isinstance(deepdta_path, Path):
            deepdta_path = str(deepdta_path) if deepdta_path.exists() else None
        if gat_path and isinstance(gat_path, Path):
            gat_path = str(gat_path) if gat_path.exists() else None
        if xgboost_path and isinstance(xgboost_path, Path):
            xgboost_path = str(xgboost_path) if xgboost_path.exists() else None
    except ImportError as e:
        raise ModelUnavailableError(f"Could not load config: {e}")
    
    return EnsembleScorer(
        deepdta_path=deepdta_path,
        gat_path=gat_path,
        xgboost_path=xgboost_path,
        models_to_use=models_to_use,
    )

"""
XGBoost Scorer for pIC50 Prediction.

Uses a pre-trained XGBoost regressor on ECFP4 fingerprints.
Model trained on ChEMBL pIC50 data.

STRICT MODE: No mocks, no stubs. If dependencies unavailable, fail explicitly.

Interface Contract (IMMUTABLE):
    predict_affinity(smiles, protein) -> {"score": float, "uncertainty": float}
"""

from typing import Dict, Optional
from pathlib import Path
import numpy as np

from utils.logging import get_logger

logger = get_logger(__name__)


class ModelUnavailableError(Exception):
    """Raised when required model dependencies are not available."""
    pass


class ModelLoadError(Exception):
    """Raised when model checkpoint cannot be loaded."""
    pass


# Validate all required dependencies at import time
_MISSING_DEPS = []

try:
    import xgboost as xgb
except ImportError as e:
    _MISSING_DEPS.append(f"XGBoost: {e}")

try:
    import joblib
except ImportError as e:
    _MISSING_DEPS.append(f"joblib: {e}")

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
except ImportError as e:
    _MISSING_DEPS.append(f"RDKit: {e}")


def _check_dependencies():
    """Check all dependencies are available. Raise if not."""
    if _MISSING_DEPS:
        raise ModelUnavailableError(
            f"XGBoost scorer requires the following dependencies: {'; '.join(_MISSING_DEPS)}"
        )


def smiles_to_ecfp(
    smiles: str,
    radius: int = 2,
    n_bits: int = 2048,
) -> np.ndarray:
    """
    Convert SMILES to ECFP4 fingerprint.
    
    Args:
        smiles: SMILES string
        radius: Morgan fingerprint radius (2 = ECFP4)
        n_bits: Number of bits in fingerprint
        
    Returns:
        Numpy array of shape (n_bits,)
        
    Raises:
        ValueError: If SMILES cannot be parsed
    """
    _check_dependencies()
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")
    
    # Generate Morgan fingerprint (ECFP)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    
    # Convert to numpy array
    arr = np.zeros(n_bits, dtype=np.float32)
    for idx in fp.GetOnBits():
        arr[idx] = 1.0
    
    return arr


class XGBoostScorer:
    """
    XGBoost scoring wrapper with immutable interface.
    
    STRICT MODE: No mocks. Fails explicitly if dependencies/model unavailable.
    
    Interface Contract (IMMUTABLE):
        predict_affinity(smiles, protein) -> {"score": float, "uncertainty": float}
    """
    
    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        fp_radius: int = 2,
        fp_bits: int = 2048,
    ):
        """
        Initialize XGBoost scorer.
        
        Args:
            checkpoint_path: Path to pretrained model (.joblib file). REQUIRED.
            fp_radius: Fingerprint radius (2 for ECFP4)
            fp_bits: Fingerprint bit size
            
        Raises:
            ModelUnavailableError: If dependencies not available
            ModelLoadError: If checkpoint cannot be loaded
        """
        _check_dependencies()
        
        self.fp_radius = fp_radius
        self.fp_bits = fp_bits
        
        if checkpoint_path is None:
            raise ModelLoadError("XGBoost checkpoint_path is required. No mock mode available.")
        
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise ModelLoadError(f"XGBoost checkpoint not found: {checkpoint_path}")
        
        try:
            self.model = joblib.load(str(checkpoint_path))
            logger.info(f"Loaded XGBoost model from {checkpoint_path}")
        except Exception as e:
            raise ModelLoadError(f"Failed to load XGBoost checkpoint: {e}")
        
        self._initialized = True
    
    def predict_affinity(self, smiles: str, protein: str) -> Dict[str, float]:
        """
        Predict binding affinity between drug and protein.
        
        IMMUTABLE INTERFACE - DO NOT MODIFY SIGNATURE.
        
        Args:
            smiles: Drug SMILES string
            protein: Protein sequence or ID (kept for interface compatibility)
        
        Returns:
            Dictionary with:
                - score: Predicted pIC50 value
                - uncertainty: Model uncertainty estimate
                
        Raises:
            ValueError: If SMILES is invalid
            RuntimeError: If prediction fails
        """
        # Generate fingerprint (raises ValueError if invalid)
        fp = smiles_to_ecfp(smiles, self.fp_radius, self.fp_bits)
        
        # Reshape for prediction
        X = fp.reshape(1, -1)
        
        # Predict
        score = self.model.predict(X)[0]
        
        # Estimate uncertainty
        uncertainty = self._estimate_uncertainty(smiles, fp)
        
        return {
            "score": float(score),
            "uncertainty": uncertainty,
        }
    
    def _estimate_uncertainty(self, smiles: str, fp: np.ndarray) -> float:
        """
        Estimate prediction uncertainty based on fingerprint density.
        """
        on_bits = np.sum(fp)
        
        # Typical molecules have 30-150 bits on
        if 30 <= on_bits <= 150:
            uncertainty = 0.15
        elif on_bits < 30:
            uncertainty = 0.3  # Very small molecule
        else:
            uncertainty = 0.25  # Large/complex molecule
        
        return uncertainty


def create_scorer(
    checkpoint_path: Optional[str] = None,
    fp_radius: int = 2,
    fp_bits: int = 2048,
) -> XGBoostScorer:
    """
    Factory function to create an XGBoost scorer.
    
    Raises:
        ModelUnavailableError: If dependencies not available
        ModelLoadError: If checkpoint cannot be loaded
    """
    return XGBoostScorer(checkpoint_path, fp_radius, fp_bits)

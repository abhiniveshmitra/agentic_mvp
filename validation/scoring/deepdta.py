"""
DeepDTA Scoring Model.

CRITICAL CONSTRAINTS (from agent knowledge base):
- Immutable interface: predict_affinity(smiles, protein) -> {score, uncertainty}
- No access to literature context or discovery source
- No post-processing beyond ensemble rules
- No manual weighting

Phase 1: Uses pretrained DeepDTA weights.
"""

from typing import Dict, Optional
import numpy as np

from utils.logging import get_logger

logger = get_logger(__name__)

# Try to import PyTorch
try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    logger.warning("PyTorch not available - using mock scorer")


# Character encoding for SMILES and protein sequences
SMILES_CHARS = "CNOPSFIBrcnops=#-+12345678()[]@"
PROTEIN_CHARS = "ACDEFGHIKLMNPQRSTVWY"


def encode_smiles(smiles: str, max_len: int = 100) -> np.ndarray:
    """Encode SMILES string to integer sequence."""
    encoding = np.zeros(max_len, dtype=np.int32)
    for i, char in enumerate(smiles[:max_len]):
        if char in SMILES_CHARS:
            encoding[i] = SMILES_CHARS.index(char) + 1
    return encoding


def encode_protein(sequence: str, max_len: int = 1000) -> np.ndarray:
    """Encode protein sequence to integer sequence."""
    encoding = np.zeros(max_len, dtype=np.int32)
    for i, char in enumerate(sequence[:max_len]):
        if char in PROTEIN_CHARS:
            encoding[i] = PROTEIN_CHARS.index(char) + 1
    return encoding


if HAS_TORCH:
    class DeepDTAModel(nn.Module):
        """
        DeepDTA architecture for drug-target affinity prediction.
        
        Based on: Öztürk et al., "DeepDTA: deep drug-target binding affinity prediction"
        Bioinformatics, 2018.
        """
        
        def __init__(
            self,
            smiles_vocab_size: int = 64,
            protein_vocab_size: int = 26,
            embed_dim: int = 128,
            num_filters: int = 32,
            filter_lengths: list = None,
        ):
            super().__init__()
            
            filter_lengths = filter_lengths or [4, 6, 8]
            
            # Drug (SMILES) embedding and CNN
            self.drug_embed = nn.Embedding(smiles_vocab_size, embed_dim)
            self.drug_convs = nn.ModuleList([
                nn.Conv1d(embed_dim, num_filters * 2, k, padding=k // 2)
                for k in filter_lengths
            ])
            self.drug_pool = nn.AdaptiveMaxPool1d(1)
            
            # Protein embedding and CNN
            self.protein_embed = nn.Embedding(protein_vocab_size, embed_dim)
            self.protein_convs = nn.ModuleList([
                nn.Conv1d(embed_dim, num_filters * 2, k, padding=k // 2)
                for k in filter_lengths
            ])
            self.protein_pool = nn.AdaptiveMaxPool1d(1)
            
            # Combined layers
            combined_size = len(filter_lengths) * num_filters * 2 * 2
            self.fc1 = nn.Linear(combined_size, 1024)
            self.fc2 = nn.Linear(1024, 512)
            self.fc3 = nn.Linear(512, 1)
            
            self.dropout = nn.Dropout(0.1)
            self.relu = nn.ReLU()
        
        def forward(self, drug: torch.Tensor, protein: torch.Tensor) -> torch.Tensor:
            # Drug branch
            drug_x = self.drug_embed(drug)  # (batch, seq_len, embed)
            drug_x = drug_x.permute(0, 2, 1)  # (batch, embed, seq_len)
            
            drug_features = []
            for conv in self.drug_convs:
                x = self.relu(conv(drug_x))
                x = self.drug_pool(x).squeeze(-1)
                drug_features.append(x)
            drug_out = torch.cat(drug_features, dim=1)
            
            # Protein branch
            prot_x = self.protein_embed(protein)
            prot_x = prot_x.permute(0, 2, 1)
            
            prot_features = []
            for conv in self.protein_convs:
                x = self.relu(conv(prot_x))
                x = self.protein_pool(x).squeeze(-1)
                prot_features.append(x)
            prot_out = torch.cat(prot_features, dim=1)
            
            # Combined
            combined = torch.cat([drug_out, prot_out], dim=1)
            x = self.dropout(self.relu(self.fc1(combined)))
            x = self.dropout(self.relu(self.fc2(x)))
            out = self.fc3(x)
            
            return out
else:
    # Placeholder when torch is not available
    DeepDTAModel = None


class DeepDTAScorer:
    """
    DeepDTA scoring wrapper with immutable interface.
    
    Interface Contract (IMMUTABLE):
        predict_affinity(smiles, protein) -> {"score": float, "uncertainty": float}
    """
    
    def __init__(self, checkpoint_path: Optional[str] = None):
        """
        Initialize DeepDTA scorer.
        
        Args:
            checkpoint_path: Path to pretrained weights (optional)
        """
        self.model = None
        self.device = "cpu"
        
        if HAS_TORCH:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = DeepDTAModel()
            
            if checkpoint_path:
                try:
                    state_dict = torch.load(checkpoint_path, map_location=self.device)
                    self.model.load_state_dict(state_dict)
                    logger.info(f"Loaded DeepDTA weights from {checkpoint_path}")
                except Exception as e:
                    logger.warning(f"Could not load weights: {e}. Using random initialization.")
            
            self.model.to(self.device)
            self.model.eval()
        else:
            logger.warning("PyTorch not available - using mock predictions")
    
    def predict_affinity(self, smiles: str, protein: str) -> Dict[str, float]:
        """
        Predict binding affinity between drug and protein.
        
        IMMUTABLE INTERFACE - DO NOT MODIFY SIGNATURE.
        
        Args:
            smiles: Drug SMILES string
            protein: Protein sequence or ID
        
        Returns:
            Dictionary with:
                - score: Predicted pKd/pKi value
                - uncertainty: Model uncertainty estimate
        """
        if self.model is None:
            # Mock prediction for testing without PyTorch
            return self._mock_predict(smiles, protein)
        
        try:
            # Encode inputs
            drug_enc = encode_smiles(smiles)
            prot_enc = encode_protein(protein)
            
            # Convert to tensors
            drug_tensor = torch.tensor(drug_enc, dtype=torch.long).unsqueeze(0).to(self.device)
            prot_tensor = torch.tensor(prot_enc, dtype=torch.long).unsqueeze(0).to(self.device)
            
            # Predict
            with torch.no_grad():
                score = self.model(drug_tensor, prot_tensor)
                score_value = score.item()
            
            # Estimate uncertainty (using simple heuristic for Phase 1)
            # In Phase 2+, this will come from ensemble std
            uncertainty = self._estimate_uncertainty(smiles, protein)
            
            return {
                "score": score_value,
                "uncertainty": uncertainty,
            }
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return {"score": 0.0, "uncertainty": 1.0}
    
    def _estimate_uncertainty(self, smiles: str, protein: str) -> float:
        """
        Estimate prediction uncertainty.
        
        Phase 1: Simple heuristic based on input length.
        Phase 2+: Will use ensemble standard deviation.
        """
        # Simple uncertainty heuristic
        smiles_len = len(smiles)
        prot_len = len(protein)
        
        # Longer/shorter molecules have higher uncertainty
        smiles_unc = 0.1 if 20 < smiles_len < 80 else 0.3
        prot_unc = 0.1 if 100 < prot_len < 800 else 0.2
        
        return (smiles_unc + prot_unc) / 2
    
    def _mock_predict(self, smiles: str, protein: str) -> Dict[str, float]:
        """Mock prediction for testing without PyTorch."""
        # Generate deterministic but varied scores based on input
        hash_val = hash(smiles + protein)
        score = 5.0 + (hash_val % 1000) / 200  # Range ~5-10
        uncertainty = 0.2 + (hash_val % 100) / 500  # Range ~0.2-0.4
        
        return {
            "score": round(score, 3),
            "uncertainty": round(uncertainty, 3),
        }


def create_scorer(checkpoint_path: Optional[str] = None) -> DeepDTAScorer:
    """Factory function to create a DeepDTA scorer."""
    return DeepDTAScorer(checkpoint_path)

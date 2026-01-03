"""
GAT (Graph Attention Network) Scorer.

Uses a pre-trained 2-layer GAT with 4 attention heads per layer.
Model trained on ChEMBL pIC50 data with PyTorch Geometric.

STRICT MODE: No mocks, no stubs. If dependencies unavailable, fail explicitly.

Interface Contract (IMMUTABLE):
    predict_affinity(smiles, protein) -> {"score": float, "uncertainty": float}
"""

from typing import Dict, Optional
from pathlib import Path

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
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError as e:
    _MISSING_DEPS.append(f"PyTorch: {e}")

try:
    from torch_geometric.nn import GATConv, global_mean_pool
    from torch_geometric.data import Data
except ImportError as e:
    _MISSING_DEPS.append(f"PyTorch Geometric: {e}")

try:
    from rdkit import Chem
except ImportError as e:
    _MISSING_DEPS.append(f"RDKit: {e}")


def _check_dependencies():
    """Check all dependencies are available. Raise if not."""
    if _MISSING_DEPS:
        raise ModelUnavailableError(
            f"GAT scorer requires the following dependencies: {'; '.join(_MISSING_DEPS)}"
        )


def smiles_to_graph(smiles: str):
    """
    Convert SMILES to PyTorch Geometric graph.
    
    Args:
        smiles: SMILES string
        
    Returns:
        PyTorch Geometric Data object with node features and edge index
        
    Raises:
        ValueError: If SMILES cannot be parsed
    """
    _check_dependencies()
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")
    
    # Node features (5 features to match trained model):
    # atomic number, degree, formal charge, hybridization, is_aromatic
    node_features = []
    for atom in mol.GetAtoms():
        features = [
            atom.GetAtomicNum(),
            atom.GetDegree(),
            atom.GetFormalCharge(),
            int(atom.GetHybridization()),
            int(atom.GetIsAromatic()),
        ]
        node_features.append(features)
    
    x = torch.tensor(node_features, dtype=torch.float)
    
    # Edge index (bonds)
    edge_index = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_index.append([i, j])
        edge_index.append([j, i])  # Undirected
    
    if len(edge_index) == 0:
        # Single atom molecule
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    
    return Data(x=x, edge_index=edge_index)


# Only define model class if dependencies available
if not _MISSING_DEPS:
    class GATModel(nn.Module):
        """
        Graph Attention Network for pIC50 prediction.
        
        Architecture (matching trained checkpoint):
            - 2 GAT layers (gat1, gat2) with 4 attention heads each
            - 64 hidden channels per head -> 256 total after concat
            - Global mean pooling
            - MLP for final prediction (fc1, fc2)
        """
        
        def __init__(
            self,
            in_channels: int = 5,
            hidden_channels: int = 64,
            num_heads: int = 4,
            dropout: float = 0.2,
        ):
            super().__init__()
            
            self.gat1 = GATConv(in_channels, hidden_channels, heads=num_heads, dropout=dropout)
            self.gat2 = GATConv(
                hidden_channels * num_heads,
                hidden_channels,
                heads=num_heads,
                dropout=dropout
            )
            
            self.fc1 = nn.Linear(hidden_channels * num_heads, 128)
            self.fc2 = nn.Linear(128, 1)
            self.dropout = nn.Dropout(dropout)
        
        def forward(self, x, edge_index, batch=None):
            x = self.gat1(x, edge_index)
            x = F.elu(x)
            x = self.dropout(x)
            
            x = self.gat2(x, edge_index)
            x = F.elu(x)
            
            if batch is None:
                batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            x = global_mean_pool(x, batch)
            
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
            
            return x


class GATScorer:
    """
    GAT scoring wrapper with immutable interface.
    
    STRICT MODE: No mocks. Fails explicitly if dependencies/model unavailable.
    
    Interface Contract (IMMUTABLE):
        predict_affinity(smiles, protein) -> {"score": float, "uncertainty": float}
    """
    
    def __init__(self, checkpoint_path: Optional[str] = None):
        """
        Initialize GAT scorer.
        
        Args:
            checkpoint_path: Path to pretrained weights (.pt file). REQUIRED.
            
        Raises:
            ModelUnavailableError: If dependencies not available
            ModelLoadError: If checkpoint cannot be loaded
        """
        _check_dependencies()
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = GATModel()
        
        if checkpoint_path is None:
            raise ModelLoadError("GAT checkpoint_path is required. No mock mode available.")
        
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise ModelLoadError(f"GAT checkpoint not found: {checkpoint_path}")
        
        try:
            state_dict = torch.load(str(checkpoint_path), map_location=self.device, weights_only=True)
            self.model.load_state_dict(state_dict)
            logger.info(f"Loaded GAT weights from {checkpoint_path}")
        except Exception as e:
            raise ModelLoadError(f"Failed to load GAT checkpoint: {e}")
        
        self.model.to(self.device)
        self.model.eval()
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
        # Convert SMILES to graph (raises ValueError if invalid)
        graph = smiles_to_graph(smiles)
        
        # Move to device
        graph = graph.to(self.device)
        
        # Predict
        with torch.no_grad():
            score = self.model(graph.x, graph.edge_index)
            score_value = score.item()
        
        # Estimate uncertainty based on molecule complexity
        uncertainty = self._estimate_uncertainty(smiles)
        
        return {
            "score": score_value,
            "uncertainty": uncertainty,
        }
    
    def _estimate_uncertainty(self, smiles: str) -> float:
        """Estimate prediction uncertainty based on molecule complexity."""
        num_atoms = smiles.count('C') + smiles.count('N') + smiles.count('O')
        num_rings = smiles.count('1') + smiles.count('2')
        
        base_uncertainty = 0.15
        size_factor = min(num_atoms / 50, 0.15)
        ring_factor = min(num_rings / 10, 0.1)
        
        return base_uncertainty + size_factor + ring_factor


def create_scorer(checkpoint_path: Optional[str] = None) -> GATScorer:
    """
    Factory function to create a GAT scorer.
    
    Raises:
        ModelUnavailableError: If dependencies not available
        ModelLoadError: If checkpoint cannot be loaded
    """
    return GATScorer(checkpoint_path)

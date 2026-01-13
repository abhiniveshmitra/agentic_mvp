"""
Molecule Visualizer for XAI Explanations.

Generates 2D molecule images with atom-level heatmaps showing
contribution to binding predictions.

Color scheme:
- Red/Orange: High contribution to binding
- Blue/Green: Low contribution to binding
"""

from typing import Dict, List, Optional, Tuple
import base64
from io import BytesIO

from utils.logging import get_logger

logger = get_logger(__name__)


def draw_molecule_with_heatmap(
    smiles: str,
    atom_scores: Optional[List[float]] = None,
    highlight_atoms: Optional[List[int]] = None,
    size: Tuple[int, int] = (400, 300),
    title: Optional[str] = None,
) -> Optional[str]:
    """
    Draw a molecule with atoms colored by contribution score.
    
    Args:
        smiles: SMILES string of the molecule
        atom_scores: Per-atom contribution scores (-1 to 1 scale)
        highlight_atoms: Specific atoms to highlight
        size: Image size (width, height)
        title: Optional title for the image
    
    Returns:
        Base64-encoded PNG image string, or None if failed
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import Draw
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.warning(f"Invalid SMILES: {smiles[:50]}...")
            return None
        
        # Determine which atoms to highlight
        highlight_atom_list = []
        highlight_atom_colors = {}
        
        if atom_scores and len(atom_scores) == mol.GetNumAtoms():
            for idx, score in enumerate(atom_scores):
                if score > 0.1:  # Only highlight significant atoms
                    highlight_atom_list.append(idx)
                    
                    # Color based on score (use RGB tuples)
                    if score > 0.3:
                        # Red for high contribution
                        highlight_atom_colors[idx] = (1.0, 0.3, 0.2)
                    elif score > 0.2:
                        # Orange for moderate
                        highlight_atom_colors[idx] = (1.0, 0.6, 0.2)
                    else:
                        # Yellow for low positive
                        highlight_atom_colors[idx] = (1.0, 0.9, 0.3)
        
        elif highlight_atoms:
            highlight_atom_list = [a for a in highlight_atoms if a < mol.GetNumAtoms()]
            for idx in highlight_atom_list:
                highlight_atom_colors[idx] = (1.0, 0.4, 0.0)  # Orange
        
        # Use the simpler MolToImage with highlights
        img = Draw.MolToImage(
            mol,
            size=size,
            highlightAtoms=highlight_atom_list if highlight_atom_list else None,
            highlightAtomColors=highlight_atom_colors if highlight_atom_colors else None,
        )
        
        # Convert to base64
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        b64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        return b64_image
        
    except ImportError as e:
        logger.error(f"RDKit not available: {e}")
        return None
    except Exception as e:
        logger.error(f"Failed to draw molecule: {e}")
        return None


def draw_molecule_simple(
    smiles: str,
    size: Tuple[int, int] = (300, 200),
) -> Optional[str]:
    """
    Draw a simple 2D molecule image without heatmap.
    
    Args:
        smiles: SMILES string
        size: Image size
    
    Returns:
        Base64-encoded PNG image string
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import Draw
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        # Generate image
        img = Draw.MolToImage(mol, size=size)
        
        # Convert to base64
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        b64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        return b64_image
        
    except Exception as e:
        logger.error(f"Failed to draw molecule: {e}")
        return None


def draw_molecule_with_highlights(
    smiles: str,
    highlight_atoms: List[int],
    highlight_color: Tuple[float, float, float] = (1.0, 0.3, 0.3),
    size: Tuple[int, int] = (400, 300),
) -> Optional[str]:
    """
    Draw molecule with specific atoms highlighted (e.g., problematic atoms).
    
    Args:
        smiles: SMILES string
        highlight_atoms: Atom indices to highlight
        highlight_color: RGB color for highlights
        size: Image size
    
    Returns:
        Base64-encoded PNG image string
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import Draw
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        # Filter valid atom indices
        valid_atoms = [idx for idx in highlight_atoms if idx < mol.GetNumAtoms()]
        
        # Create color dict for each atom
        atom_colors = {idx: highlight_color for idx in valid_atoms}
        
        # Use simple MolToImage with highlights
        img = Draw.MolToImage(
            mol,
            size=size,
            highlightAtoms=valid_atoms if valid_atoms else None,
            highlightAtomColors=atom_colors if atom_colors else None,
        )
        
        # Convert to base64
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        b64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        return b64_image
        
    except Exception as e:
        logger.error(f"Failed to draw highlighted molecule: {e}")
        return None


def get_color_legend() -> Dict:
    """
    Get the color legend for heatmap interpretation.
    
    Returns:
        Dictionary with color descriptions
    """
    return {
        "high_contribution": {
            "color": "#FF6B35",  # Orange-red
            "description": "High contribution to binding"
        },
        "moderate_contribution": {
            "color": "#F7C566",  # Yellow-orange
            "description": "Moderate contribution"
        },
        "neutral": {
            "color": "#4ECDC4",  # Teal
            "description": "Neutral/low contribution"
        },
        "negative_contribution": {
            "color": "#45B7D1",  # Blue
            "description": "May reduce binding"
        }
    }


def create_explanation_card_html(
    compound_name: str,
    smiles: str,
    score: float,
    confidence: float,
    features: List[str],
    molecule_image_b64: str,
) -> str:
    """
    Create HTML for an explanation card with molecule image.
    
    Args:
        compound_name: Name of the compound
        smiles: SMILES string
        score: Binding score (pIC50)
        confidence: Confidence percentage
        features: List of key features
        molecule_image_b64: Base64-encoded molecule image
    
    Returns:
        HTML string for the card
    """
    features_html = "\n".join([f"<li>{f}</li>" for f in features])
    
    return f"""
    <div style="border: 1px solid #ddd; border-radius: 12px; padding: 20px; 
                background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);">
        <h3 style="margin: 0 0 15px 0; color: #2c3e50;">{compound_name}</h3>
        
        <div style="display: flex; gap: 20px;">
            <div style="flex: 1;">
                <img src="data:image/png;base64,{molecule_image_b64}" 
                     style="max-width: 100%; border-radius: 8px; background: white; padding: 10px;">
                <p style="font-size: 12px; color: #7f8c8d; margin-top: 5px;">
                    <span style="color: #e74c3c;">●</span> High binding contribution
                    <span style="color: #3498db; margin-left: 10px;">●</span> Low contribution
                </p>
            </div>
            
            <div style="flex: 1;">
                <div style="background: white; padding: 15px; border-radius: 8px; margin-bottom: 10px;">
                    <strong>Predicted Bioactivity</strong><br>
                    <span style="font-size: 24px; color: #27ae60;">pIC50 = {score:.2f}</span>
                </div>
                
                <div style="background: white; padding: 15px; border-radius: 8px; margin-bottom: 10px;">
                    <strong>Confidence Score</strong><br>
                    <span style="font-size: 24px; color: #3498db;">{confidence:.0%}</span>
                </div>
                
                <div style="background: white; padding: 15px; border-radius: 8px;">
                    <strong>Key Features</strong>
                    <ul style="margin: 10px 0 0 0; padding-left: 20px;">
                        {features_html}
                    </ul>
                </div>
            </div>
        </div>
    </div>
    """


# Test function
if __name__ == "__main__":
    # Test with aspirin
    test_smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"
    
    # Simple drawing
    img = draw_molecule_simple(test_smiles)
    if img:
        print(f"Simple image generated: {len(img)} chars")
    
    # With heatmap
    scores = [0.3, 0.5, 0.2, 0.1, -0.1, -0.2, 0.0, 0.4, 0.6, 0.1, -0.1, 0.2, 0.3]
    img_heatmap = draw_molecule_with_heatmap(test_smiles, scores)
    if img_heatmap:
        print(f"Heatmap image generated: {len(img_heatmap)} chars")

"""
3D Molecule Viewer using Py3Dmol.

Provides interactive 3D visualization of molecules for the drug discovery platform.
Renders molecules in multiple styles with rotation/zoom capabilities.

Features:
- Interactive 3D molecular structures
- Multiple rendering styles (stick, sphere, cartoon)
- Binding pose visualization with protein
- Pharmacophore feature highlighting
- Exportable HTML for Streamlit embedding
"""

from typing import Dict, List, Optional, Tuple, Any
import json

from utils.logging import get_logger

logger = get_logger(__name__)


def smiles_to_3d_html(
    smiles: str,
    style: str = "stick",
    width: int = 400,
    height: int = 350,
    background_color: str = "#1a1a2e",
    show_surface: bool = False,
) -> Optional[str]:
    """
    Convert SMILES to interactive 3D HTML viewer.
    
    Args:
        smiles: SMILES string of the molecule
        style: Rendering style ('stick', 'sphere', 'line', 'cartoon')
        width: Viewer width in pixels
        height: Viewer height in pixels
        background_color: Background color (hex)
        show_surface: Whether to show molecular surface
    
    Returns:
        HTML string for embedding, or None if failed
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        
        # Parse SMILES and generate 3D coordinates
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.warning(f"Invalid SMILES: {smiles[:50]}...")
            return None
        
        # Add hydrogens and generate 3D conformation
        mol = Chem.AddHs(mol)
        result = AllChem.EmbedMolecule(mol, randomSeed=42)
        if result == -1:
            # Fallback to distance geometry if ETKDG fails
            result = AllChem.EmbedMolecule(mol, useRandomCoords=True, randomSeed=42)
            if result == -1:
                logger.warning(f"Could not generate 3D coordinates for: {smiles[:50]}...")
                return None
        
        # Optimize geometry
        try:
            AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
        except Exception:
            # MMFF may fail for some molecules, try UFF
            try:
                AllChem.UFFOptimizeMolecule(mol, maxIters=200)
            except Exception:
                pass  # Use unoptimized coordinates
        
        # Get Mol block for Py3Dmol
        mol_block = Chem.MolToMolBlock(mol)
        
        # Get style config
        style_config = _get_style_config(style)
        import json
        style_json = json.dumps(style_config)
        
        # Escape mol_block for JavaScript
        mol_block_escaped = mol_block.replace('\\', '\\\\').replace('\n', '\\n').replace("'", "\\'")
        
        # Generate standalone HTML with proper centering
        html = f'''
        <div id="mol3d_container_{id(smiles)}" style="width: {width}px; height: {height}px; position: relative; margin: 0 auto;">
            <div id="mol3d_viewer_{id(smiles)}" style="width: 100%; height: 100%;"></div>
        </div>
        <script src="https://3dmol.org/build/3Dmol-min.js"></script>
        <script>
        (function() {{
            var viewer = $3Dmol.createViewer("mol3d_viewer_{id(smiles)}", {{
                backgroundColor: "{background_color}",
                width: {width},
                height: {height}
            }});
            var molData = '{mol_block_escaped}';
            viewer.addModel(molData, "mol");
            viewer.setStyle({{}}, {style_json});
            viewer.zoomTo();
            viewer.center();
            viewer.render();
        }})();
        </script>
        '''
        
        return html
        
    except ImportError as e:
        logger.error(f"RDKit not available: {e}")
        return _fallback_html(smiles, width, height)
    except Exception as e:
        logger.error(f"Failed to create 3D view: {e}")
        return _fallback_html(smiles, width, height)


def render_binding_pose(
    ligand_sdf: str,
    protein_pdb: str,
    width: int = 600,
    height: int = 450,
    ligand_style: str = "stick",
    protein_style: str = "cartoon",
    highlight_residues: Optional[List[int]] = None,
) -> Optional[str]:
    """
    Render ligand-protein binding pose.
    
    Args:
        ligand_sdf: Ligand structure in SDF format
        protein_pdb: Protein structure in PDB format
        width: Viewer width
        height: Viewer height
        ligand_style: Style for ligand ('stick', 'sphere')
        protein_style: Style for protein ('cartoon', 'line', 'stick')
        highlight_residues: Residue numbers to highlight in binding site
    
    Returns:
        HTML string for embedding
    """
    try:
        import py3Dmol
        
        viewer = py3Dmol.view(width=width, height=height)
        
        # Add protein
        viewer.addModel(protein_pdb, "pdb")
        
        # Style protein
        if protein_style == "cartoon":
            viewer.setStyle({"model": 0}, {"cartoon": {"color": "spectrum"}})
        elif protein_style == "line":
            viewer.setStyle({"model": 0}, {"line": {"color": "#888888"}})
        else:
            viewer.setStyle({"model": 0}, {"stick": {"colorscheme": "grayCarbon"}})
        
        # Highlight binding site residues
        if highlight_residues:
            for resi in highlight_residues:
                viewer.addStyle(
                    {"model": 0, "resi": resi},
                    {"stick": {"color": "#FFD700"}, "opacity": 0.8}
                )
        
        # Add ligand
        viewer.addModel(ligand_sdf, "sdf")
        
        # Style ligand (bright for visibility)
        ligand_style_config = {
            "stick": {"colorscheme": "greenCarbon", "radius": 0.3},
            "sphere": {"colorscheme": "greenCarbon", "radius": 0.6}
        }.get(ligand_style, {"stick": {"colorscheme": "greenCarbon"}})
        
        viewer.setStyle({"model": 1}, ligand_style_config)
        
        # Add ligand surface
        viewer.addSurface(
            py3Dmol.VDW,
            {"opacity": 0.5, "color": "#00ff88"},
            {"model": 1}
        )
        
        # Set view
        viewer.setBackgroundColor("#0f0f1a")
        viewer.zoomTo({"model": 1})  # Focus on ligand
        
        return viewer._make_html()
        
    except Exception as e:
        logger.error(f"Failed to render binding pose: {e}")
        return None


def render_comparison_grid(
    smiles_list: List[str],
    names: Optional[List[str]] = None,
    scores: Optional[List[float]] = None,
    cols: int = 3,
    cell_width: int = 280,
    cell_height: int = 250,
) -> str:
    """
    Create a grid of 3D molecule viewers for comparison.
    
    Args:
        smiles_list: List of SMILES strings
        names: Optional compound names
        scores: Optional binding scores
        cols: Number of columns in grid
        cell_width: Width of each viewer
        cell_height: Height of each viewer
    
    Returns:
        HTML string with grid of viewers
    """
    if names is None:
        names = [f"Compound {i+1}" for i in range(len(smiles_list))]
    
    if scores is None:
        scores = [None] * len(smiles_list)
    
    # Build grid HTML
    html_parts = [
        f"""
        <style>
            .mol3d-grid {{
                display: grid;
                grid-template-columns: repeat({cols}, 1fr);
                gap: 15px;
                padding: 15px;
            }}
            .mol3d-cell {{
                background: linear-gradient(145deg, #1a1a2e, #16213e);
                border-radius: 12px;
                overflow: hidden;
                box-shadow: 0 4px 15px rgba(0,0,0,0.3);
            }}
            .mol3d-header {{
                padding: 10px 15px;
                background: rgba(255,255,255,0.05);
                border-bottom: 1px solid rgba(255,255,255,0.1);
            }}
            .mol3d-name {{
                color: #fff;
                font-weight: 600;
                font-size: 14px;
            }}
            .mol3d-score {{
                color: #4ecdc4;
                font-size: 12px;
            }}
        </style>
        <div class="mol3d-grid">
        """
    ]
    
    for i, (smi, name, score) in enumerate(zip(smiles_list, names, scores)):
        viewer_html = smiles_to_3d_html(
            smi, 
            width=cell_width, 
            height=cell_height - 40,
            style="stick"
        )
        
        score_text = f"pIC50: {score:.2f}" if score is not None else ""
        
        if viewer_html:
            html_parts.append(f"""
                <div class="mol3d-cell">
                    <div class="mol3d-header">
                        <div class="mol3d-name">{name}</div>
                        <div class="mol3d-score">{score_text}</div>
                    </div>
                    {viewer_html}
                </div>
            """)
        else:
            html_parts.append(f"""
                <div class="mol3d-cell">
                    <div class="mol3d-header">
                        <div class="mol3d-name">{name}</div>
                    </div>
                    <div style="padding: 20px; color: #888;">
                        Could not generate 3D view
                    </div>
                </div>
            """)
    
    html_parts.append("</div>")
    
    return "".join(html_parts)


def render_pharmacophore_features(
    smiles: str,
    features: Dict[str, List[int]],
    width: int = 450,
    height: int = 400,
) -> Optional[str]:
    """
    Render molecule with pharmacophore features highlighted.
    
    Args:
        smiles: SMILES string
        features: Dict mapping feature type to atom indices
                  e.g., {"HBA": [1,3], "HBD": [5], "Hydrophobic": [2,4,6]}
        width: Viewer width
        height: Viewer height
    
    Returns:
        HTML string with pharmacophore visualization
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        import py3Dmol
        
        # Feature color scheme
        feature_colors = {
            "HBA": "#ff6b6b",      # Red - H-bond acceptor
            "HBD": "#4ecdc4",      # Teal - H-bond donor
            "Hydrophobic": "#ffd93d",  # Yellow - Hydrophobic
            "Aromatic": "#95e1d3",     # Light teal - Aromatic
            "PosCharge": "#0066ff",    # Blue - Positive charge
            "NegCharge": "#ff6600",    # Orange - Negative charge
        }
        
        # Generate 3D mol
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        try:
            AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
        except Exception:
            pass
        
        mol_block = Chem.MolToMolBlock(mol)
        
        # Create viewer
        viewer = py3Dmol.view(width=width, height=height)
        viewer.addModel(mol_block, "mol")
        
        # Base style
        viewer.setStyle({"stick": {"colorscheme": "grayCarbon", "radius": 0.15}})
        
        # Highlight pharmacophore features
        for feature_type, atom_indices in features.items():
            color = feature_colors.get(feature_type, "#ffffff")
            for idx in atom_indices:
                viewer.addSphere({
                    "center": {"serial": idx + 1},  # 1-indexed
                    "radius": 0.8,
                    "color": color,
                    "opacity": 0.6,
                })
        
        viewer.setBackgroundColor("#1a1a2e")
        viewer.zoomTo()
        
        # Build legend
        legend_html = "<div style='padding: 10px; display: flex; gap: 15px; flex-wrap: wrap;'>"
        for feature_type, atom_indices in features.items():
            if atom_indices:
                color = feature_colors.get(feature_type, "#ffffff")
                legend_html += f"""
                    <span style='display: inline-flex; align-items: center; gap: 5px;'>
                        <span style='width: 12px; height: 12px; border-radius: 50%; 
                                     background: {color};'></span>
                        <span style='color: #ccc; font-size: 12px;'>{feature_type}</span>
                    </span>
                """
        legend_html += "</div>"
        
        return viewer._make_html() + legend_html
        
    except Exception as e:
        logger.error(f"Failed to render pharmacophore features: {e}")
        return None


def _get_style_config(style: str) -> Dict[str, Any]:
    """Get Py3Dmol style configuration."""
    styles = {
        "stick": {"stick": {"colorscheme": "greenCarbon", "radius": 0.2}},
        "sphere": {"sphere": {"colorscheme": "greenCarbon", "scale": 0.25}},
        "line": {"line": {"colorscheme": "greenCarbon"}},
        "cartoon": {"cartoon": {"color": "spectrum"}},
        "ball_stick": {
            "stick": {"colorscheme": "greenCarbon", "radius": 0.15},
            "sphere": {"colorscheme": "greenCarbon", "scale": 0.2}
        },
    }
    return styles.get(style, styles["stick"])


def _fallback_html(smiles: str, width: int, height: int) -> str:
    """Generate fallback HTML when py3Dmol is not available."""
    return f"""
    <div style="width: {width}px; height: {height}px; 
                background: linear-gradient(145deg, #1a1a2e, #16213e);
                border-radius: 12px; display: flex; 
                align-items: center; justify-content: center;
                flex-direction: column; color: #888;">
        <div style="font-size: 48px; margin-bottom: 10px;">🧪</div>
        <div style="font-size: 14px;">3D Viewer Unavailable</div>
        <div style="font-size: 11px; margin-top: 5px; color: #666;">
            Install py3Dmol: pip install py3Dmol
        </div>
        <code style="font-size: 10px; margin-top: 10px; color: #4ecdc4; 
                     max-width: 90%; overflow: hidden; text-overflow: ellipsis;">
            {smiles[:60]}{'...' if len(smiles) > 60 else ''}
        </code>
    </div>
    """


# Convenience function for Streamlit
def get_streamlit_3d_viewer(
    smiles: str,
    title: Optional[str] = None,
    score: Optional[float] = None,
    width: int = 400,
    height: int = 350,
) -> str:
    """
    Get a complete 3D viewer widget for Streamlit.
    
    Args:
        smiles: SMILES string
        title: Optional title to display
        score: Optional score to display
        width: Viewer width
        height: Viewer height
    
    Returns:
        Complete HTML string ready for st.components.html()
    """
    viewer_html = smiles_to_3d_html(smiles, width=width, height=height - 50)
    
    if viewer_html is None:
        viewer_html = _fallback_html(smiles, width, height - 50)
    
    title_html = ""
    if title or score is not None:
        title_text = title or "Molecule"
        score_text = f" | pIC50: {score:.2f}" if score is not None else ""
        title_html = f"""
        <div style="padding: 10px; background: rgba(0,0,0,0.3); 
                    border-radius: 10px 10px 0 0; color: #fff; 
                    font-weight: 600; text-align: center;">
            {title_text}{score_text}
        </div>
        """
    
    return f"""
    <div style="background: linear-gradient(145deg, #1a1a2e, #16213e);
                border-radius: 12px; overflow: hidden; 
                box-shadow: 0 4px 20px rgba(0,0,0,0.3);">
        {title_html}
        {viewer_html}
        <div style="padding: 8px; text-align: center; 
                    background: rgba(0,0,0,0.2); color: #666; font-size: 11px;">
            🖱️ Click and drag to rotate | Scroll to zoom
        </div>
    </div>
    """


# Test function
if __name__ == "__main__":
    # Test with caffeine
    test_smiles = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"  # Caffeine
    
    html = smiles_to_3d_html(test_smiles)
    if html:
        print(f"✓ 3D viewer HTML generated: {len(html)} chars")
        # Save to file for testing
        with open("/tmp/test_mol3d.html", "w") as f:
            f.write(f"<html><body>{html}</body></html>")
        print("  Saved to /tmp/test_mol3d.html")
    
    # Test comparison grid
    test_molecules = [
        "CC(=O)OC1=CC=CC=C1C(=O)O",  # Aspirin
        "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",  # Ibuprofen
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
    ]
    
    grid_html = render_comparison_grid(
        test_molecules,
        names=["Aspirin", "Ibuprofen", "Caffeine"],
        scores=[7.2, 6.8, 5.5]
    )
    print(f"✓ Comparison grid HTML generated: {len(grid_html)} chars")

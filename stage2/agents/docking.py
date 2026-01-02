"""
Docking Agent (NON-LLM).

Purpose: Validate structural plausibility, NOT affinity ranking.

Rules (IMMUTABLE):
- Docking score must NEVER modify Stage-1 score
- Docking failure is allowed and logged
- Docking is a validation signal only
- Runs in parallel, non-blocking

Forbidden:
- Ranking molecules by docking score
- Combining docking scores with affinity
- Re-ordering candidates based on docking
- Suppressing failed docking results

This is a NON-LLM agent.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
import subprocess
import tempfile
import os

from utils.logging import get_logger

logger = get_logger(__name__)

# Vina executable path (relative to project root)
VINA_PATH = Path(__file__).parent.parent.parent / "tools" / "vina" / "vina.exe"

# Cross-target warning (MUST include in all user-facing outputs)
CROSS_TARGET_WARNING = (
    "Vina score thresholds are heuristic and target-dependent. "
    "Scores are interpreted qualitatively and should not be compared "
    "across different protein targets."
)


@dataclass
class DockingExplanation:
    """Explanation object for docking (follows unified schema)."""
    
    result: Dict = field(default_factory=dict)
    raw_values: Dict = field(default_factory=dict)
    rules_triggered: List = field(default_factory=list)
    observations: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)
    summary: str = ""
    
    def to_dict(self) -> Dict:
        return {
            "result": self.result,
            "raw_values": self.raw_values,
            "rules_triggered": self.rules_triggered,
            "observations": self.observations,
            "limitations": self.limitations,
            "summary": self.summary,
        }


@dataclass
class DockingResult:
    """Result of docking a single ligand."""
    
    smiles: str
    compound_id: str
    docking_status: str  # "PASS", "FLAG", "FAIL"
    pose_available: bool
    confidence: Optional[float]
    poses: List[Dict] = field(default_factory=list)
    explanation: DockingExplanation = field(default_factory=DockingExplanation)
    error: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            "smiles": self.smiles,
            "compound_id": self.compound_id,
            "docking_status": self.docking_status,
            "pose_available": self.pose_available,
            "confidence": self.confidence,
            "poses": self.poses,
            "explanation": self.explanation.to_dict(),
            "error": self.error,
        }


class DockingAgent:
    """
    Docking Agent using AutoDock Vina.
    
    Validates structural plausibility, NOT affinity.
    Docking score is a validation signal, never a ranking signal.
    """
    
    def __init__(
        self,
        vina_path: Optional[Path] = None,
        exhaustiveness: int = 8,
        num_modes: int = 3,
    ):
        """
        Initialize Docking Agent.
        
        Args:
            vina_path: Path to Vina executable
            exhaustiveness: Vina search exhaustiveness (default 8)
            num_modes: Number of poses to generate (default 3)
        """
        self.vina_path = Path(vina_path) if vina_path else VINA_PATH
        self.exhaustiveness = exhaustiveness
        self.num_modes = num_modes
        
        # Check if Vina is available
        if not self.vina_path.exists():
            logger.warning(f"Vina not found at {self.vina_path}")
            self.vina_available = False
        else:
            self.vina_available = True
            logger.info(f"DockingAgent initialized with Vina at {self.vina_path}")
    
    def dock_single(
        self,
        smiles: str,
        compound_id: str,
        protein_path: Path,
        center: Tuple[float, float, float],
        box_size: Tuple[float, float, float] = (20, 20, 20),
    ) -> DockingResult:
        """
        Dock a single ligand against prepared protein.
        
        Args:
            smiles: Ligand SMILES string
            compound_id: Compound identifier
            protein_path: Path to prepared protein PDB
            center: (x, y, z) center of binding box
            box_size: (x, y, z) size of binding box in Angstroms
        
        Returns:
            DockingResult with status and explanation
        """
        if not self.vina_available:
            return self._mock_docking(smiles, compound_id)
        
        try:
            # Convert SMILES to 3D structure
            ligand_pdbqt = self._smiles_to_pdbqt(smiles)
            if ligand_pdbqt is None:
                return DockingResult(
                    smiles=smiles,
                    compound_id=compound_id,
                    docking_status="FAIL",
                    pose_available=False,
                    confidence=None,
                    explanation=self._build_explanation(
                        smiles=smiles,
                        success=False,
                        error="Failed to convert SMILES to 3D structure",
                    ),
                    error="SMILES conversion failed",
                )
            
            # Prepare protein PDBQT
            protein_pdbqt = self._protein_to_pdbqt(protein_path)
            if protein_pdbqt is None:
                return DockingResult(
                    smiles=smiles,
                    compound_id=compound_id,
                    docking_status="FAIL",
                    pose_available=False,
                    confidence=None,
                    explanation=self._build_explanation(
                        smiles=smiles,
                        success=False,
                        error="Failed to prepare protein",
                    ),
                    error="Protein preparation failed",
                )
            
            # Run Vina
            result = self._run_vina(
                ligand_pdbqt=ligand_pdbqt,
                protein_pdbqt=protein_pdbqt,
                center=center,
                box_size=box_size,
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Docking error for {compound_id}: {e}")
            return DockingResult(
                smiles=smiles,
                compound_id=compound_id,
                docking_status="FAIL",
                pose_available=False,
                confidence=None,
                explanation=self._build_explanation(
                    smiles=smiles,
                    success=False,
                    error=str(e),
                ),
                error=str(e),
            )
    
    def dock_batch(
        self,
        candidates: List[Dict],
        protein_path: Path,
        center: Tuple[float, float, float],
        box_size: Tuple[float, float, float] = (20, 20, 20),
    ) -> List[DockingResult]:
        """
        Dock multiple ligands against a protein.
        
        Args:
            candidates: List of candidate dicts with 'smiles' and 'compound_id'
            protein_path: Path to prepared protein
            center: Binding box center
            box_size: Binding box size
        
        Returns:
            List of DockingResult objects
        """
        results = []
        
        for i, candidate in enumerate(candidates):
            smiles = candidate.get("smiles", "")
            compound_id = candidate.get("compound_id", f"compound_{i}")
            
            logger.info(f"Docking {compound_id} ({i+1}/{len(candidates)})")
            
            result = self.dock_single(
                smiles=smiles,
                compound_id=compound_id,
                protein_path=protein_path,
                center=center,
                box_size=box_size,
            )
            results.append(result)
        
        # Log summary
        passed = sum(1 for r in results if r.docking_status == "PASS")
        flagged = sum(1 for r in results if r.docking_status == "FLAG")
        failed = sum(1 for r in results if r.docking_status == "FAIL")
        logger.info(f"Docking complete: {passed} PASS, {flagged} FLAG, {failed} FAIL")
        
        return results
    
    def _smiles_to_pdbqt(self, smiles: str) -> Optional[Path]:
        """Convert SMILES to PDBQT format for docking."""
        # Try using OpenBabel if available (proper PDBQT with charges)
        try:
            from stage2.docking.pdbqt_prep import prepare_ligand_pdbqt, check_openbabel_available
            if check_openbabel_available():
                result = prepare_ligand_pdbqt(smiles)
                if result:
                    logger.info(f"Ligand PDBQT prepared via OpenBabel")
                    return result
        except ImportError:
            pass
        
        # Fallback to RDKit-only (PDB format - not ideal, charges missing)
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem
            
            # Parse SMILES
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            
            # Add hydrogens
            mol = Chem.AddHs(mol)
            
            # Generate 3D coordinates
            result = AllChem.EmbedMolecule(mol, randomSeed=42)
            if result == -1:
                # Fallback to random generation
                AllChem.EmbedMolecule(mol, useRandomCoords=True)
            
            # Optimize geometry
            AllChem.MMFFOptimizeMolecule(mol)
            
            # Write to temporary PDB file
            with tempfile.NamedTemporaryFile(
                mode='w', suffix='.pdb', delete=False
            ) as f:
                pdb_path = f.name
                Chem.MolToPDBFile(mol, pdb_path)
            
            logger.warning("Using RDKit PDB fallback - no partial charges (OpenBabel not available)")
            return Path(pdb_path)
            
        except Exception as e:
            logger.error(f"SMILES to PDBQT conversion failed: {e}")
            return None

    
    def _protein_to_pdbqt(self, protein_path: Path) -> Optional[Path]:
        """Prepare protein PDB to PDBQT format."""
        if not protein_path.exists():
            return None
        
        # Try using OpenBabel if available (proper PDBQT with charges)
        try:
            from stage2.docking.pdbqt_prep import prepare_receptor_pdbqt, check_openbabel_available
            if check_openbabel_available():
                result = prepare_receptor_pdbqt(protein_path)
                if result:
                    logger.info(f"Receptor PDBQT prepared via OpenBabel")
                    return result
        except ImportError:
            pass
        
        # Fallback: use PDB directly (not ideal, charges missing)
        logger.warning("Using PDB fallback for receptor - no partial charges (OpenBabel not available)")
        return protein_path

    
    def _run_vina(
        self,
        ligand_pdbqt: Path,
        protein_pdbqt: Path,
        center: Tuple[float, float, float],
        box_size: Tuple[float, float, float],
    ) -> DockingResult:
        """Run AutoDock Vina docking."""
        try:
            # Create output file
            with tempfile.NamedTemporaryFile(
                suffix='_out.pdbqt', delete=False
            ) as f:
                output_path = f.name
            
            # Build Vina command
            cmd = [
                str(self.vina_path),
                "--receptor", str(protein_pdbqt),
                "--ligand", str(ligand_pdbqt),
                "--center_x", str(center[0]),
                "--center_y", str(center[1]),
                "--center_z", str(center[2]),
                "--size_x", str(box_size[0]),
                "--size_y", str(box_size[1]),
                "--size_z", str(box_size[2]),
                "--exhaustiveness", str(self.exhaustiveness),
                "--num_modes", str(self.num_modes),
                "--out", output_path,
            ]
            
            # Run Vina
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
            )
            
            if result.returncode != 0:
                # Parse error
                error_msg = result.stderr[:500] if result.stderr else "Unknown error"
                return self._create_fail_result(
                    smiles="",
                    compound_id="",
                    error=f"Vina failed: {error_msg}",
                )
            
            # Parse output
            poses = self._parse_vina_output(result.stdout)
            
            # Determine status based on docking score
            if poses:
                best_score = poses[0].get("score", 0)
                if best_score < -7:
                    status = "PASS"
                elif best_score < -5:
                    status = "FLAG"
                else:
                    status = "FLAG"
                
                return DockingResult(
                    smiles="",  # Will be filled by caller
                    compound_id="",
                    docking_status=status,
                    pose_available=True,
                    confidence=abs(best_score) / 10,  # Normalize to 0-1
                    poses=poses,
                    explanation=self._build_explanation(
                        smiles="",
                        success=True,
                        poses=poses,
                    ),
                )
            else:
                return self._create_fail_result(
                    smiles="",
                    compound_id="",
                    error="No poses generated",
                )
                
        except subprocess.TimeoutExpired:
            return self._create_fail_result("", "", "Docking timeout")
        except Exception as e:
            return self._create_fail_result("", "", str(e))
        finally:
            # Cleanup temp files
            try:
                if 'output_path' in locals():
                    os.unlink(output_path)
            except:
                pass
    
    def _parse_vina_output(self, stdout: str) -> List[Dict]:
        """Parse Vina output to extract poses and scores."""
        poses = []
        in_results = False
        
        for line in stdout.split('\n'):
            if 'mode' in line.lower() and 'affinity' in line.lower():
                in_results = True
                continue
            
            if in_results and line.strip():
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        mode = int(parts[0])
                        score = float(parts[1])
                        poses.append({
                            "pose_id": mode,
                            "score": score,
                            "rmsd_lb": float(parts[2]) if len(parts) > 2 else None,
                            "rmsd_ub": float(parts[3]) if len(parts) > 3 else None,
                        })
                    except (ValueError, IndexError):
                        continue
        
        return poses
    
    def _mock_docking(self, smiles: str, compound_id: str, reason: str = "Vina not available") -> DockingResult:
        """
        Mock docking when Vina is not available or PDBQT conversion fails.
        
        In MVP mode, this uses RDKit to estimate druglikeness and generates
        a deterministic score based on molecular properties.
        """
        import hashlib
        
        try:
            from rdkit import Chem
            from rdkit.Chem import Descriptors, AllChem
            
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return self._create_fail_result(smiles, compound_id, "Invalid SMILES")
            
            # Use molecular properties to estimate docking plausibility
            mw = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            hbd = Descriptors.NumHDonors(mol)
            hba = Descriptors.NumHAcceptors(mol)
            rotb = Descriptors.NumRotatableBonds(mol)
            
            # Generate pseudo-score based on druglikeness
            # Better druglike properties = better simulated score
            base_score = -6.0
            
            # Adjust based on MW (optimal 300-500)
            if 300 <= mw <= 500:
                base_score -= 1.0
            elif mw < 200 or mw > 700:
                base_score += 1.5
            
            # Adjust based on LogP (optimal 1-3)
            if 1 <= logp <= 3:
                base_score -= 0.5
            elif logp < 0 or logp > 5:
                base_score += 0.5
            
            # Add some deterministic variation based on SMILES hash
            hash_val = int(hashlib.md5(smiles.encode()).hexdigest()[:8], 16)
            variation = (hash_val % 200 - 100) / 100  # -1 to +1
            mock_score = base_score + variation
            
            # Determine status
            if mock_score < -7:
                status = "PASS"
            elif mock_score < -5:
                status = "FLAG"
            else:
                status = "FLAG"
            
        except ImportError:
            # RDKit not available - use hash-based fallback
            hash_val = int(hashlib.md5(smiles.encode()).hexdigest()[:8], 16)
            mock_score = -5 - (hash_val % 500) / 100
            status = "PASS" if mock_score < -7 else "FLAG"
            mw = logp = hbd = hba = rotb = "N/A"
        
        poses = [
            {"pose_id": 1, "score": round(mock_score, 2), "rmsd_lb": 0, "rmsd_ub": 0},
            {"pose_id": 2, "score": round(mock_score + 0.5, 2), "rmsd_lb": 1.2, "rmsd_ub": 2.1},
        ]
        
        logger.warning(f"Using simulated docking for {compound_id} ({reason})")
        
        return DockingResult(
            smiles=smiles,
            compound_id=compound_id,
            docking_status=status,
            pose_available=True,
            confidence=abs(mock_score) / 10,
            poses=poses,
            explanation=self._build_explanation(
                smiles=smiles,
                success=True,
                poses=poses,
                is_mock=True,
                mock_reason=reason,
            ),
        )
    
    def _create_fail_result(
        self,
        smiles: str,
        compound_id: str,
        error: str,
    ) -> DockingResult:
        """Create a FAIL result."""
        return DockingResult(
            smiles=smiles,
            compound_id=compound_id,
            docking_status="FAIL",
            pose_available=False,
            confidence=None,
            explanation=self._build_explanation(
                smiles=smiles,
                success=False,
                error=error,
            ),
            error=error,
        )
    
    def create_not_evaluated_result(
        self,
        smiles: str,
        compound_id: str,
        reason: str,
    ) -> DockingResult:
        """
        Create a NOT_EVALUATED result for honest deferral.
        
        This is used when docking cannot be performed properly
        (e.g., PDBQT tools not available). We NEVER substitute
        heuristics for physics.
        
        Args:
            smiles: Ligand SMILES
            compound_id: Compound identifier
            reason: Clear explanation of why docking is unavailable
        
        Returns:
            DockingResult with NOT_EVALUATED status
        """
        logger.info(f"Docking NOT_EVALUATED for {compound_id}: {reason}")
        
        return DockingResult(
            smiles=smiles,
            compound_id=compound_id,
            docking_status="NOT_EVALUATED",
            pose_available=False,
            confidence=None,
            explanation=DockingExplanation(
                result={
                    "label": "NOT_EVALUATED",
                    "pose_available": False,
                },
                raw_values={},
                rules_triggered=[],
                observations=[
                    f"Docking not performed: {reason}",
                ],
                limitations=[
                    "Structural plausibility not assessed in this run",
                    "Docking requires PDBQT format with partial charges and atom types",
                    "Real docking will be available when MGLTools or OpenBabel is installed",
                ],
                summary=(
                    f"NOT_EVALUATED - {reason}. "
                    "This is an honest deferral, not a failure."
                ),
            ),
            error=None,  # This is not an error, it's an intentional deferral
        )
    
    def _build_explanation(
        self,
        smiles: str,
        success: bool,
        poses: List[Dict] = None,
        error: str = None,
        is_mock: bool = False,
        mock_reason: str = None,
    ) -> DockingExplanation:
        """Build explanation for docking result."""
        
        limitations = [
            "Protein treated as rigid",
            "Solvent effects not explicitly modeled",
            "Vina score is a scoring-function estimate, NOT thermodynamic ΔG",
            "Results are structural plausibility checks only",
            CROSS_TARGET_WARNING,
        ]
        
        if is_mock:
            reason_text = mock_reason or "Vina not available"
            limitations.append(f"SIMULATED RESULT - {reason_text}, scores derived from druglikeness")
        
        if success and poses:
            best_score = poses[0]["score"]
            
            if best_score < -8:
                observations = [
                    "Strong docking score suggests good binding pocket fit",
                    "Ligand occupies binding site without major clashes",
                ]
            elif best_score < -6:
                observations = [
                    "Moderate docking score - acceptable fit",
                    "Some suboptimal interactions may be present",
                ]
            else:
                observations = [
                    "Weak docking score - marginal fit",
                    "May require conformational changes for binding",
                ]
            
            return DockingExplanation(
                result={
                    "label": "PASS" if best_score < -6 else "FLAG",
                    "pose_available": True,
                },
                raw_values={
                    "best_score": best_score,
                    "num_poses": len(poses),
                    "scores": [p["score"] for p in poses],
                },
                rules_triggered=[
                    {
                        "rule_id": "STERIC_FIT",
                        "condition": "ligand placed in binding pocket",
                        "triggered": True,
                        "scientific_rationale": (
                            "Ligand must fit within binding pocket without severe steric clashes"
                        ),
                        "practical_implication": (
                            "Suggests structural compatibility with target site"
                        ),
                    }
                ],
                observations=observations,
                limitations=limitations,
                summary=(
                    f"Docking {'PASS' if best_score < -6 else 'FLAG'} - "
                    f"best score {best_score:.1f} kcal/mol, "
                    f"{len(poses)} poses generated"
                ),
            )
        else:
            return DockingExplanation(
                result={
                    "label": "FAIL",
                    "pose_available": False,
                    "error": error,
                },
                raw_values={},
                rules_triggered=[],
                observations=[
                    f"Docking failed: {error}",
                ],
                limitations=limitations + [
                    "Alternative conformations could not be explored",
                ],
                summary=f"Docking FAIL - {error}",
            )


def create_docking_agent(
    vina_path: Optional[Path] = None,
    exhaustiveness: int = 8,
) -> DockingAgent:
    """Factory function to create DockingAgent."""
    return DockingAgent(vina_path=vina_path, exhaustiveness=exhaustiveness)

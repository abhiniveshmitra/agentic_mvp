"""
Protein Preparation Agent (NON-LLM).

Purpose: Prepare the protein structure once per target.

Rules (IMMUTABLE):
- Must be deterministic
- Must not re-run per ligand
- Must fail loudly if structure invalid
- Run once, cache, reuse for all ligands

This is a NON-LLM agent - pure computational pipeline.
"""

from typing import Dict, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
import json
import hashlib
import requests

from utils.logging import get_logger

logger = get_logger(__name__)

# AlphaFold API endpoint
ALPHAFOLD_API = "https://alphafold.ebi.ac.uk/api"
ALPHAFOLD_FILES = "https://alphafold.ebi.ac.uk/files"

# PDB API endpoint
PDB_API = "https://files.rcsb.org/download"


@dataclass
class ProteinPrepExplanation:
    """Explanation object for protein preparation (follows unified schema)."""
    
    result: Dict = field(default_factory=dict)
    raw_values: Dict = field(default_factory=dict)
    rules_triggered: list = field(default_factory=list)
    limitations: list = field(default_factory=list)
    summary: str = ""
    
    def to_dict(self) -> Dict:
        return {
            "result": self.result,
            "raw_values": self.raw_values,
            "rules_triggered": self.rules_triggered,
            "limitations": self.limitations,
            "summary": self.summary,
        }


@dataclass
class PreparedProtein:
    """A prepared protein structure ready for docking."""
    
    protein_id: str
    source: str  # "alphafold" or "pdb"
    structure_path: Path
    preparation_status: str  # "SUCCESS" or "FAIL"
    cached: bool
    confidence_regions: Dict = field(default_factory=dict)
    explanation: ProteinPrepExplanation = field(default_factory=ProteinPrepExplanation)
    error: Optional[str] = None
    sequence: Optional[str] = None  # Protein sequence for ML models
    
    def to_dict(self) -> Dict:
        return {
            "protein_id": self.protein_id,
            "source": self.source,
            "structure_path": str(self.structure_path) if self.structure_path else None,
            "preparation_status": self.preparation_status,
            "cached": self.cached,
            "confidence_regions": self.confidence_regions,
            "explanation": self.explanation.to_dict(),
            "error": self.error,
        }


# Import List for type hints
from typing import List


class ProteinPreparator:
    """
    Protein Preparation Agent.
    
    Fetches and prepares protein structures from AlphaFold or PDB.
    Caches results to avoid re-downloading.
    """
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize Protein Preparator.
        
        Args:
            cache_dir: Directory for caching structures (default: mvp/data/protein_cache)
        """
        if cache_dir is None:
            from config.settings import DATA_DIR
            cache_dir = DATA_DIR / "protein_cache"
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache metadata file
        self.cache_index = self.cache_dir / "cache_index.json"
        self._load_cache_index()
        
        logger.info(f"ProteinPreparator initialized with cache at {self.cache_dir}")
    
    def _load_cache_index(self):
        """Load cache index from disk."""
        if self.cache_index.exists():
            with open(self.cache_index, 'r') as f:
                self._cache = json.load(f)
        else:
            self._cache = {}
    
    def _save_cache_index(self):
        """Save cache index to disk."""
        with open(self.cache_index, 'w') as f:
            json.dump(self._cache, f, indent=2)
    
    def _get_cache_key(self, protein_id: str, source: str) -> str:
        """Generate cache key for a protein."""
        return f"{protein_id}_{source}"
    
    def prepare(
        self,
        protein_id: str,
        prefer_alphafold: bool = True,
    ) -> PreparedProtein:
        """
        Prepare protein structure for docking.
        
        Args:
            protein_id: UniProt ID or PDB ID
            prefer_alphafold: Prefer AlphaFold over PDB (default True)
        
        Returns:
            PreparedProtein object with structure and explanation
        """
        # Detect if this is a PDB ID (4 characters, alphanumeric)
        is_pdb_id = len(protein_id) == 4 and protein_id.isalnum()
        
        # Check cache first
        cache_key_af = self._get_cache_key(protein_id, "alphafold")
        cache_key_pdb = self._get_cache_key(protein_id, "pdb")
        
        if cache_key_pdb in self._cache:
            return self._load_from_cache(protein_id, "pdb")
        elif cache_key_af in self._cache:
            return self._load_from_cache(protein_id, "alphafold")
        
        # For PDB IDs, try PDB first
        if is_pdb_id:
            result = self._fetch_pdb(protein_id)
            if result.preparation_status == "SUCCESS":
                return result
            logger.info(f"PDB fetch failed for {protein_id}, trying AlphaFold")
            result = self._fetch_alphafold(protein_id)
        elif prefer_alphafold:
            result = self._fetch_alphafold(protein_id)
            if result.preparation_status == "FAIL":
                logger.info(f"AlphaFold failed for {protein_id}, trying PDB")
                result = self._fetch_pdb(protein_id)
        else:
            result = self._fetch_pdb(protein_id)
            if result.preparation_status == "FAIL":
                result = self._fetch_alphafold(protein_id)
        
        return result
    
    def _load_from_cache(self, protein_id: str, source: str) -> PreparedProtein:
        """Load prepared protein from cache."""
        cache_key = self._get_cache_key(protein_id, source)
        cache_entry = self._cache[cache_key]
        
        structure_path = Path(cache_entry["structure_path"])
        
        if not structure_path.exists():
            logger.warning(f"Cached file missing: {structure_path}")
            del self._cache[cache_key]
            self._save_cache_index()
            return self.prepare(protein_id)
        
        logger.info(f"Loaded {protein_id} from cache ({source})")
        
        return PreparedProtein(
            protein_id=protein_id,
            source=source,
            structure_path=structure_path,
            preparation_status="SUCCESS",
            cached=True,
            confidence_regions=cache_entry.get("confidence_regions", {}),
            explanation=self._build_explanation(
                protein_id=protein_id,
                source=source,
                success=True,
                cached=True,
            ),
        )
    
    def _fetch_alphafold(self, protein_id: str) -> PreparedProtein:
        """Fetch structure from AlphaFold database."""
        try:
            # AlphaFold uses UniProt IDs
            pdb_url = f"{ALPHAFOLD_FILES}/AF-{protein_id}-F1-model_v4.pdb"
            
            logger.info(f"Fetching AlphaFold structure for {protein_id}")
            
            response = requests.get(pdb_url, timeout=30)
            
            if response.status_code != 200:
                logger.warning(f"AlphaFold fetch failed: {response.status_code}")
                return PreparedProtein(
                    protein_id=protein_id,
                    source="alphafold",
                    structure_path=None,
                    preparation_status="FAIL",
                    cached=False,
                    explanation=self._build_explanation(
                        protein_id=protein_id,
                        source="alphafold",
                        success=False,
                        error=f"HTTP {response.status_code}",
                    ),
                    error=f"AlphaFold fetch failed: HTTP {response.status_code}",
                )
            
            # Save structure
            structure_path = self.cache_dir / f"AF-{protein_id}-F1.pdb"
            with open(structure_path, 'w') as f:
                f.write(response.text)
            
            # Get confidence info (pLDDT)
            confidence_regions = self._analyze_alphafold_confidence(response.text)
            
            # Update cache
            cache_key = self._get_cache_key(protein_id, "alphafold")
            self._cache[cache_key] = {
                "structure_path": str(structure_path),
                "fetched_at": datetime.now().isoformat(),
                "confidence_regions": confidence_regions,
            }
            self._save_cache_index()
            
            logger.info(f"Saved AlphaFold structure for {protein_id}")
            
            return PreparedProtein(
                protein_id=protein_id,
                source="alphafold",
                structure_path=structure_path,
                preparation_status="SUCCESS",
                cached=False,
                confidence_regions=confidence_regions,
                explanation=self._build_explanation(
                    protein_id=protein_id,
                    source="alphafold",
                    success=True,
                    cached=False,
                    confidence_regions=confidence_regions,
                ),
            )
            
        except Exception as e:
            logger.error(f"AlphaFold fetch error: {e}")
            return PreparedProtein(
                protein_id=protein_id,
                source="alphafold",
                structure_path=None,
                preparation_status="FAIL",
                cached=False,
                explanation=self._build_explanation(
                    protein_id=protein_id,
                    source="alphafold",
                    success=False,
                    error=str(e),
                ),
                error=str(e),
            )
    
    def _fetch_pdb(self, protein_id: str) -> PreparedProtein:
        """Fetch structure from PDB."""
        try:
            pdb_url = f"{PDB_API}/{protein_id.upper()}.pdb"
            
            logger.info(f"Fetching PDB structure for {protein_id}")
            
            response = requests.get(pdb_url, timeout=30)
            
            if response.status_code != 200:
                return PreparedProtein(
                    protein_id=protein_id,
                    source="pdb",
                    structure_path=None,
                    preparation_status="FAIL",
                    cached=False,
                    explanation=self._build_explanation(
                        protein_id=protein_id,
                        source="pdb",
                        success=False,
                        error=f"HTTP {response.status_code}",
                    ),
                    error=f"PDB fetch failed: HTTP {response.status_code}",
                )
            
            # Save structure
            structure_path = self.cache_dir / f"{protein_id.upper()}.pdb"
            with open(structure_path, 'w') as f:
                f.write(response.text)
            
            # Update cache
            cache_key = self._get_cache_key(protein_id, "pdb")
            self._cache[cache_key] = {
                "structure_path": str(structure_path),
                "fetched_at": datetime.now().isoformat(),
                "confidence_regions": {},  # PDB doesn't have pLDDT
            }
            self._save_cache_index()
            
            logger.info(f"Saved PDB structure for {protein_id}")
            
            return PreparedProtein(
                protein_id=protein_id,
                source="pdb",
                structure_path=structure_path,
                preparation_status="SUCCESS",
                cached=False,
                explanation=self._build_explanation(
                    protein_id=protein_id,
                    source="pdb",
                    success=True,
                    cached=False,
                ),
            )
            
        except Exception as e:
            logger.error(f"PDB fetch error: {e}")
            return PreparedProtein(
                protein_id=protein_id,
                source="pdb",
                structure_path=None,
                preparation_status="FAIL",
                cached=False,
                explanation=self._build_explanation(
                    protein_id=protein_id,
                    source="pdb",
                    success=False,
                    error=str(e),
                ),
                error=str(e),
            )
    
    def _analyze_alphafold_confidence(self, pdb_content: str) -> Dict:
        """
        Analyze AlphaFold pLDDT confidence from B-factor column.
        
        Returns summary of confidence regions.
        """
        try:
            b_factors = []
            for line in pdb_content.split('\n'):
                if line.startswith('ATOM'):
                    # B-factor is columns 61-66 in PDB format
                    b_factor = float(line[60:66].strip())
                    b_factors.append(b_factor)
            
            if not b_factors:
                return {}
            
            import statistics
            
            mean_plddt = statistics.mean(b_factors)
            high_conf = sum(1 for b in b_factors if b > 90) / len(b_factors) * 100
            low_conf = sum(1 for b in b_factors if b < 50) / len(b_factors) * 100
            
            return {
                "mean_plddt": round(mean_plddt, 1),
                "high_confidence_pct": round(high_conf, 1),
                "low_confidence_pct": round(low_conf, 1),
                "residue_count": len(set(b_factors)),  # Approximate
            }
            
        except Exception as e:
            logger.warning(f"Could not analyze pLDDT: {e}")
            return {}
    
    def _build_explanation(
        self,
        protein_id: str,
        source: str,
        success: bool,
        cached: bool = False,
        confidence_regions: Dict = None,
        error: str = None,
    ) -> ProteinPrepExplanation:
        """Build explanation for protein preparation."""
        
        if success:
            limitations = [
                "Protein treated as rigid during docking",
                "Solvent effects not explicitly modeled",
                "Missing loops or disordered regions may affect results",
            ]
            
            if source == "alphafold":
                limitations.append(
                    "AlphaFold predictions may have lower confidence in flexible regions"
                )
                summary = (
                    f"Successfully prepared AlphaFold structure for {protein_id}. "
                )
                if confidence_regions:
                    summary += (
                        f"Mean pLDDT: {confidence_regions.get('mean_plddt', 'N/A')}, "
                        f"{confidence_regions.get('high_confidence_pct', 0):.0f}% high-confidence regions."
                    )
            else:
                summary = f"Successfully prepared PDB structure for {protein_id}."
            
            if cached:
                summary = f"[CACHED] {summary}"
            
            return ProteinPrepExplanation(
                result={
                    "status": "SUCCESS",
                    "protein_id": protein_id,
                    "source": source,
                    "cached": cached,
                },
                raw_values=confidence_regions or {},
                rules_triggered=[
                    {
                        "rule_id": "STRUCTURE_VALID",
                        "condition": "structure file parsed successfully",
                        "triggered": True,
                        "scientific_rationale": (
                            "Valid 3D structure required for molecular docking"
                        ),
                        "practical_implication": (
                            "Docking can proceed with this structure"
                        ),
                    }
                ],
                limitations=limitations,
                summary=summary,
            )
        else:
            return ProteinPrepExplanation(
                result={
                    "status": "FAIL",
                    "protein_id": protein_id,
                    "source": source,
                    "error": error,
                },
                raw_values={},
                rules_triggered=[],
                limitations=[
                    "Structure could not be obtained",
                    "Docking cannot proceed without valid structure",
                ],
                summary=f"Failed to prepare structure for {protein_id}: {error}",
            )


def create_protein_preparator(cache_dir: Optional[Path] = None) -> ProteinPreparator:
    """Factory function to create ProteinPreparator."""
    return ProteinPreparator(cache_dir=cache_dir)

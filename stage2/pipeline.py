"""
Stage-2 Pipeline Orchestrator.

Runs the complete Stage-2 pipeline in the correct order:
1. Input Lock (verify Stage-1)
2. Top-K Selection
3. Protein Preparation
4. Docking (parallel)
5. ADME/Tox Flags
6. Patent Risk (stub)
7. Aggregation & Output

This orchestrator enforces execution order and provenance.
"""

from typing import Dict, List, Optional, Tuple
from pathlib import Path
from datetime import datetime
import json

from utils.logging import get_logger
from stage2.agents.topk_selection import TopKSelector, TopKCandidate
from stage2.agents.protein_prep import ProteinPreparator, PreparedProtein
from stage2.agents.docking import DockingAgent, DockingResult
from stage2.agents.adme_tox_stage2 import ADMEToxStage2, ADMEToxStage2Result
from stage2.agents.patent_stub import PatentRiskStub, PatentRiskResult
from stage2.agents.aggregator import Stage2Aggregator, AggregatedCandidate

logger = get_logger(__name__)


class Stage2Pipeline:
    """
    Stage-2 Pipeline Orchestrator.
    
    Runs all Stage-2 agents in the correct order.
    Enforces: no re-ranking, no score modification, full provenance.
    """
    
    def __init__(
        self,
        top_k: int = 20,
        protein_cache_dir: Optional[Path] = None,
        vina_path: Optional[Path] = None,
    ):
        """
        Initialize Stage-2 Pipeline.
        
        Args:
            top_k: Number of top candidates to analyze
            protein_cache_dir: Directory for protein structure cache
            vina_path: Path to Vina executable
        """
        self.top_k = top_k
        
        # Initialize all agents
        self.selector = TopKSelector(k=top_k)
        self.protein_prep = ProteinPreparator(cache_dir=protein_cache_dir)
        self.docking = DockingAgent(vina_path=vina_path)
        self.adme_tox = ADMEToxStage2()
        self.patent = PatentRiskStub()
        self.aggregator = Stage2Aggregator()
        
        logger.info(f"Stage2Pipeline initialized (Top-K={top_k})")
    
    def run(
        self,
        stage1_results: List[Dict],
        protein_id: str,
        binding_center: Tuple[float, float, float] = (0, 0, 0),
        binding_box_size: Tuple[float, float, float] = (20, 20, 20),
    ) -> Dict:
        """
        Run the complete Stage-2 pipeline.
        
        Args:
            stage1_results: List of Stage-1 ranked candidates
            protein_id: Target protein UniProt/PDB ID
            binding_center: (x, y, z) center of binding site
            binding_box_size: (x, y, z) size of binding box
        
        Returns:
            Dictionary with complete Stage-2 results
        """
        start_time = datetime.now()
        
        logger.info("=" * 60)
        logger.info("STAGE-2 PIPELINE STARTING")
        logger.info("=" * 60)
        
        # Step 0: Input Lock
        logger.info("Step 0: Input Lock")
        if not stage1_results:
            return self._error_result("No Stage-1 results provided")
        
        # Step 1: Top-K Selection
        logger.info(f"Step 1: Top-K Selection (k={self.top_k})")
        topk_candidates = self.selector.select(stage1_results)
        logger.info(f"  Selected {len(topk_candidates)} candidates")
        
        if not topk_candidates:
            return self._error_result("No candidates selected")
        
        # Step 2: Protein Preparation
        logger.info(f"Step 2: Protein Preparation ({protein_id})")
        prepared_protein = self.protein_prep.prepare(protein_id)
        
        if prepared_protein.preparation_status == "FAIL":
            logger.warning(f"  Protein prep failed: {prepared_protein.error}")
            # Continue anyway - docking will use mock
        else:
            logger.info(f"  Protein ready: {prepared_protein.source}")
        
        # Step 3: Docking
        logger.info("Step 3: Docking")
        docking_results = []
        
        # CRITICAL: Real docking requires PDBQT format (charges + atom types)
        # which requires MGLTools or OpenBabel. If unavailable, we mark as
        # NOT_EVALUATED - we NEVER substitute heuristics for physics.
        can_dock = (
            prepared_protein.preparation_status == "SUCCESS" and
            self.docking.vina_available and
            self._can_prepare_pdbqt()
        )
        
        for i, candidate in enumerate(topk_candidates):
            logger.info(f"  Docking {i+1}/{len(topk_candidates)}: {candidate.compound_id}")
            
            if can_dock:
                result = self.docking.dock_single(
                    smiles=candidate.smiles,
                    compound_id=candidate.compound_id,
                    protein_path=prepared_protein.structure_path,
                    center=binding_center,
                    box_size=binding_box_size,
                )
            else:
                # Honest deferral - do NOT fake docking with heuristics
                result = self.docking.create_not_evaluated_result(
                    smiles=candidate.smiles,
                    compound_id=candidate.compound_id,
                    reason=self._get_docking_unavailable_reason(prepared_protein),
                )
            
            # Fill in compound info
            result.smiles = candidate.smiles
            result.compound_id = candidate.compound_id
            docking_results.append(result)
        
        # Step 4: ADME/Tox
        logger.info("Step 4: ADME/Tox Flags")
        adme_results = []
        
        for candidate in topk_candidates:
            result = self.adme_tox.assess(
                smiles=candidate.smiles,
                compound_id=candidate.compound_id,
            )
            adme_results.append(result)
        
        # Step 5: Patent Risk (Stub)
        logger.info("Step 5: Patent Risk (Stub)")
        patent_results = []
        
        for candidate in topk_candidates:
            result = self.patent.assess(
                smiles=candidate.smiles,
                compound_id=candidate.compound_id,
            )
            patent_results.append(result)
        
        # Step 6: Aggregation
        logger.info("Step 6: Aggregation")
        aggregated = self.aggregator.aggregate_batch(
            topk_candidates=topk_candidates,
            docking_results=docking_results,
            adme_tox_results=adme_results,
            patent_results=patent_results,
        )
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Build final result
        summary = self.aggregator.get_summary(aggregated)
        
        result = {
            "status": "SUCCESS",
            "pipeline_version": "2.0.0",
            "execution_time_seconds": round(duration, 2),
            "protein": prepared_protein.to_dict(),
            "summary": summary,
            "candidates": [c.to_dict() for c in aggregated],
            "provenance": {
                "stage1_input_count": len(stage1_results),
                "topk_value": self.top_k,
                "protein_id": protein_id,
                "execution_start": start_time.isoformat(),
                "execution_end": end_time.isoformat(),
            },
        }
        
        logger.info("=" * 60)
        logger.info(f"STAGE-2 PIPELINE COMPLETE ({duration:.1f}s)")
        logger.info(f"  Docking: {summary['docking']}")
        logger.info(f"  ADME/Tox: {summary['adme_tox']}")
        logger.info("=" * 60)
        
        return result
    
    def _error_result(self, error: str) -> Dict:
        """Create error result."""
        return {
            "status": "ERROR",
            "error": error,
            "pipeline_version": "2.0.0",
            "timestamp": datetime.now().isoformat(),
        }
    
    def _can_prepare_pdbqt(self) -> bool:
        """
        Check if PDBQT preparation is available.
        
        Real docking requires PDBQT format with partial charges and atom types.
        This requires MGLTools (prepare_receptor/prepare_ligand) or OpenBabel.
        
        Returns:
            True if OpenBabel is available, False otherwise
        """
        try:
            from stage2.docking.pdbqt_prep import check_openbabel_available
            return check_openbabel_available()
        except ImportError:
            return False
    
    def _get_docking_unavailable_reason(self, prepared_protein) -> str:
        """
        Get the specific reason why docking cannot be performed.
        
        Returns a clear, honest explanation.
        """
        reasons = []
        
        if prepared_protein.preparation_status != "SUCCESS":
            reasons.append("protein structure not available")
        
        if not self.docking.vina_available:
            reasons.append("AutoDock Vina not found")
        
        if not self._can_prepare_pdbqt():
            reasons.append(
                "PDBQT preparation requires MGLTools or OpenBabel (not installed). "
                "Docking needs partial charges and atom types."
            )
        
        if not reasons:
            reasons.append("unknown reason")
        
        return "; ".join(reasons)
    
    def save_results(self, results: Dict, output_path: Path):
        """
        Save pipeline results to JSON file.
        
        Args:
            results: Pipeline results dictionary
            output_path: Path to save JSON file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Results saved to {output_path}")


def create_pipeline(
    top_k: int = 20,
    protein_cache_dir: Optional[Path] = None,
) -> Stage2Pipeline:
    """Factory function to create Stage2Pipeline."""
    return Stage2Pipeline(top_k=top_k, protein_cache_dir=protein_cache_dir)


# Quick test function
def test_pipeline():
    """Quick test of Stage-2 pipeline with mock data."""
    # Create mock Stage-1 results
    mock_stage1 = [
        {
            "compound_id": f"compound_{i}",
            "compound_name": f"Test Compound {i}",
            "smiles": "CC(=O)Nc1ccc(O)cc1" if i % 2 == 0 else "c1ccccc1",  # Acetaminophen or benzene
            "rank": i,
            "score": 8.0 - i * 0.2,
            "percentile": 100 - i * 2,
            "confidence_tier": "HIGH" if i < 10 else "MEDIUM",
        }
        for i in range(1, 31)
    ]
    
    # Run pipeline
    pipeline = create_pipeline(top_k=5)
    results = pipeline.run(
        stage1_results=mock_stage1,
        protein_id="P00533",  # EGFR
        binding_center=(0, 0, 0),
    )
    
    print(f"\nPipeline Status: {results['status']}")
    print(f"Execution Time: {results.get('execution_time_seconds', 'N/A')}s")
    print(f"Candidates Analyzed: {len(results.get('candidates', []))}")
    
    if results.get('summary'):
        print(f"\nSummary:")
        print(f"  Docking: {results['summary'].get('docking', {})}")
        print(f"  ADME/Tox: {results['summary'].get('adme_tox', {})}")
    
    return results


if __name__ == "__main__":
    test_pipeline()

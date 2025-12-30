"""
Deterministic Pipeline Orchestrator.

CRITICAL: This is a DETERMINISTIC STATE MACHINE.
- No LLMs
- No probabilistic decisions
- No autonomy

The pipeline executes in a fixed order:
1. Initialization
2. Discovery
3. Chemistry Filters
4. Scoring
5. Normalization
6. Output
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path

from orchestrator.state import RunState, PipelineStep
from orchestrator.run_config import RunConfig, create_default_config
from utils.logging import setup_logger
from utils.provenance import ProvenanceTracker
from utils.reproducibility import generate_run_id, set_global_seed


@dataclass
class PipelineResult:
    """Final result of a pipeline run."""
    run_id: str
    success: bool
    is_flagged: bool
    flag_reason: Optional[str]
    compounds_processed: int
    compounds_passed: int
    output_file: Optional[Path]
    state: RunState
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "success": self.success,
            "is_flagged": self.is_flagged,
            "flag_reason": self.flag_reason,
            "compounds_processed": self.compounds_processed,
            "compounds_passed": self.compounds_passed,
            "output_file": str(self.output_file) if self.output_file else None,
        }


class Pipeline:
    """
    Deterministic drug discovery pipeline.
    
    Executes a fixed sequence of steps with no autonomous decisions.
    All configuration is locked at initialization.
    """
    
    def __init__(
        self,
        target_protein: str,
        query: str,
        target_protein_sequence: Optional[str] = None,
        positive_controls: Optional[List[str]] = None,
        negative_controls: Optional[List[str]] = None,
        output_dir: Optional[Path] = None,
    ):
        """
        Initialize the pipeline with locked configuration.
        
        Args:
            target_protein: Target protein name or UniProt ID
            query: PubMed search query
            target_protein_sequence: Optional protein sequence
            positive_controls: Known binder SMILES for validation
            negative_controls: Known non-binder SMILES for validation
            output_dir: Directory for output files
        """
        # Generate run ID
        self.run_id = generate_run_id()
        
        # Set up logging
        from config.settings import LOGS_DIR
        self.logger = setup_logger(
            __name__,
            log_file=LOGS_DIR / f"run_{self.run_id}.log",
            run_id=self.run_id
        )
        
        # Lock configuration
        self.config = create_default_config(
            run_id=self.run_id,
            target_protein=target_protein,
            query=query,
            target_protein_sequence=target_protein_sequence,
            positive_controls=positive_controls,
            negative_controls=negative_controls,
        )
        
        # Initialize state
        self.state = RunState(
            run_id=self.run_id,
            target_protein=target_protein,
            query=query,
        )
        
        # Initialize provenance tracker
        self.provenance = ProvenanceTracker(self.run_id)
        
        # Output directory
        from config.settings import OUTPUTS_DIR
        self.output_dir = output_dir or OUTPUTS_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set random seed
        set_global_seed(self.config.random_seed)
        
        self.logger.info(f"Pipeline initialized with run_id: {self.run_id}")
        self.logger.info(f"Target: {target_protein}")
        self.logger.info(f"Query: {query}")
    
    def run(self) -> PipelineResult:
        """
        Execute the complete pipeline.
        
        Returns:
            PipelineResult with final status and outputs
        """
        self.logger.info("=" * 60)
        self.logger.info("STARTING PIPELINE EXECUTION")
        self.logger.info("=" * 60)
        
        try:
            # Step 0: Initialization
            self._step_initialization()
            
            # Step 1: Discovery
            papers, compounds = self._step_discovery()
            
            # Step 2: Chemistry Filters
            passed_compounds, rejected_compounds = self._step_chemistry_filters(
                compounds
            )
            
            # Step 3: Scoring
            scored_compounds = self._step_scoring(passed_compounds)
            
            # Step 4: Normalization & Sanity Checks
            final_compounds = self._step_normalization(scored_compounds)
            
            # Step 5: Output
            output_file = self._step_output(final_compounds, rejected_compounds)
            
            # Mark completion
            self.state.current_step = PipelineStep.COMPLETED
            
            return PipelineResult(
                run_id=self.run_id,
                success=True,
                is_flagged=self.state.is_flagged,
                flag_reason=self.state.flag_reason,
                compounds_processed=len(compounds),
                compounds_passed=len(passed_compounds),
                output_file=output_file,
                state=self.state,
            )
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            self.state.fail_step(self.state.current_step, str(e))
            
            return PipelineResult(
                run_id=self.run_id,
                success=False,
                is_flagged=True,
                flag_reason=f"Pipeline failure: {str(e)}",
                compounds_processed=0,
                compounds_passed=0,
                output_file=None,
                state=self.state,
            )
    
    def _step_initialization(self) -> None:
        """Step 0: Initialize run and lock configuration."""
        self.logger.info("STEP 0: Initialization")
        self.state.start_step(PipelineStep.INITIALIZATION)
        
        # Save configuration snapshot
        self.config.save(self.output_dir)
        
        self.logger.info(f"Configuration locked")
        self.logger.info(f"Positive controls: {len(self.config.positive_controls)}")
        self.logger.info(f"Negative controls: {len(self.config.negative_controls)}")
        
        self.state.complete_step(PipelineStep.INITIALIZATION)
    
    def _step_discovery(self) -> Tuple[List[Dict], List[Dict]]:
        """Step 1: Discovery - Literature mining and candidate extraction."""
        self.logger.info("STEP 1: Discovery")
        self.state.start_step(PipelineStep.DISCOVERY)
        
        from discovery.literature_ingest import fetch_pubmed_papers
        from discovery.text_mining import extract_compounds
        from discovery.candidate_builder import build_candidates
        
        # 1.1 Fetch papers from PubMed
        papers = fetch_pubmed_papers(
            query=self.config.query,
            max_results=self.config.max_papers,
        )
        self.state.papers_fetched = len(papers)
        self.logger.info(f"Fetched {len(papers)} papers from PubMed")
        
        # 1.2 Extract compound mentions using Gemini (with seeding)
        raw_compounds = extract_compounds(papers, target=self.config.target_protein)
        self.logger.info(f"Extracted {len(raw_compounds)} compound mentions")
        
        # 1.3 Build candidates with SMILES
        candidates = build_candidates(raw_compounds, self.provenance)
        self.state.compounds_extracted = len(candidates)
        self.logger.info(f"Built {len(candidates)} candidates with SMILES")
        
        self.state.complete_step(
            PipelineStep.DISCOVERY,
            metrics={
                "papers_fetched": len(papers),
                "compounds_extracted": len(candidates),
            }
        )
        
        return papers, candidates
    
    def _step_chemistry_filters(
        self,
        compounds: List[Dict]
    ) -> Tuple[List[Dict], List[Dict]]:
        """Step 2: Chemistry Filters - Apply hard gates."""
        self.logger.info("STEP 2: Chemistry Filters")
        self.state.start_step(PipelineStep.CHEMISTRY_FILTERS)
        
        from validation.chemistry_filters import apply_all_filters
        
        # Get known drugs for this target (for calibration pass-through)
        try:
            from discovery.query_templates import get_query_templates
            templates = get_query_templates(self.config.target_protein)
            known_drugs = templates.get("known_drugs", [])
        except:
            known_drugs = []
        
        passed, rejected = apply_all_filters(
            compounds=compounds,
            config=self.config.chemistry_filters,
            provenance=self.provenance,
            known_drugs=known_drugs,
        )
        
        self.state.compounds_passed_filters = len(passed)
        self.state.compounds_rejected = len(rejected)
        
        self.logger.info(f"Passed filters: {len(passed)}")
        self.logger.info(f"Rejected: {len(rejected)}")
        
        self.state.complete_step(
            PipelineStep.CHEMISTRY_FILTERS,
            metrics={
                "passed": len(passed),
                "rejected": len(rejected),
            }
        )
        
        return passed, rejected
    
    def _step_scoring(self, compounds: List[Dict]) -> List[Dict]:
        """Step 3: Scoring - Apply ML models."""
        self.logger.info("STEP 3: Scoring")
        self.state.start_step(PipelineStep.SCORING)
        
        from validation.scoring.deepdta import DeepDTAScorer
        
        # Initialize scorer
        scorer = DeepDTAScorer()
        
        # Score all compounds (+ controls)
        all_to_score = compounds.copy()
        
        # Add controls for scoring
        for smiles in self.config.positive_controls:
            all_to_score.append({
                "smiles": smiles,
                "compound_id": f"positive_control_{len(all_to_score)}",
                "is_control": True,
                "control_type": "positive",
            })
        
        for smiles in self.config.negative_controls:
            all_to_score.append({
                "smiles": smiles,
                "compound_id": f"negative_control_{len(all_to_score)}",
                "is_control": True,
                "control_type": "negative",
            })
        
        # Score each compound
        scored = []
        for compound in all_to_score:
            result = scorer.predict_affinity(
                smiles=compound["smiles"],
                protein=self.config.target_protein_sequence or self.config.target_protein,
            )
            compound["raw_score"] = result["score"]
            compound["uncertainty"] = result["uncertainty"]
            scored.append(compound)
            
            # Update provenance
            from utils.provenance import ScoringInfo
            self.provenance.update_scoring_info(
                compound["compound_id"],
                ScoringInfo(
                    model_name="deepdta",
                    model_version="1.0",
                    raw_score=result["score"],
                    uncertainty=result["uncertainty"],
                )
            )
        
        self.state.compounds_scored = len(scored)
        self.logger.info(f"Scored {len(scored)} compounds")
        
        self.state.complete_step(
            PipelineStep.SCORING,
            metrics={"scored": len(scored)}
        )
        
        return scored
    
    def _step_normalization(self, compounds: List[Dict]) -> List[Dict]:
        """Step 4: Normalization and Sanity Checks."""
        self.logger.info("STEP 4: Normalization & Sanity Checks")
        self.state.start_step(PipelineStep.NORMALIZATION)
        
        from validation.normalization import normalize_scores
        from validation.sanity_checks import validate_controls
        
        # Normalize scores
        normalized = normalize_scores(
            compounds=compounds,
            config=self.config.normalization,
        )
        
        # Validate controls
        is_valid, validation_msg = validate_controls(
            compounds=normalized,
            config=self.config.control_validation,
        )
        
        if not is_valid:
            self.state.flag_run(validation_msg)
            self.logger.warning(f"Run flagged: {validation_msg}")
        
        self.state.complete_step(
            PipelineStep.NORMALIZATION,
            metrics={"controls_valid": is_valid}
        )
        
        return normalized
    
    def _step_output(
        self,
        compounds: List[Dict],
        rejected: List[Dict]
    ) -> Path:
        """Step 5: Generate output files."""
        self.logger.info("STEP 5: Output Generation")
        self.state.start_step(PipelineStep.OUTPUT)
        
        import pandas as pd
        
        # Filter out controls from main output
        main_compounds = [c for c in compounds if not c.get("is_control")]
        
        # Sort by score (descending)
        main_compounds.sort(key=lambda x: x.get("raw_score", 0), reverse=True)
        
        # Create output DataFrame
        output_data = []
        for c in main_compounds:
            output_data.append({
                "rank": len(output_data) + 1,
                "compound_id": c.get("compound_id"),
                "smiles": c.get("smiles"),
                "compound_name": c.get("compound_name", ""),
                "raw_score": c.get("raw_score"),
                "percentile": c.get("percentile"),
                "z_score": c.get("z_score"),
                "uncertainty": c.get("uncertainty"),
                "confidence_tier": c.get("confidence_tier", ""),
                "source": c.get("source", ""),
                "paper_id": c.get("paper_id", ""),
            })
        
        df = pd.DataFrame(output_data)
        
        # Save main output
        output_file = self.output_dir / f"results_{self.run_id}.csv"
        df.to_csv(output_file, index=False)
        
        # Save rejected compounds
        if rejected:
            rejected_df = pd.DataFrame(rejected)
            rejected_file = self.output_dir / f"rejected_{self.run_id}.csv"
            rejected_df.to_csv(rejected_file, index=False)
        
        # Save provenance
        provenance_file = self.output_dir / f"provenance_{self.run_id}.json"
        self.provenance.export_json(str(provenance_file))
        
        # Save state
        self.state.save(self.output_dir)
        
        self.logger.info(f"Output saved to: {output_file}")
        
        self.state.complete_step(
            PipelineStep.OUTPUT,
            metrics={"output_file": str(output_file)}
        )
        
        return output_file

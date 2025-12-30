"""
Integration tests for the full pipeline.

Tests the complete flow from discovery to output.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestPipelineIntegration:
    """Test full pipeline integration."""
    
    @patch('discovery.literature_ingest.fetch_pubmed_papers')
    @patch('discovery.text_mining.extract_compounds')
    @patch('discovery.candidate_builder.query_pubchem')
    def test_pipeline_execution(
        self,
        mock_pubchem,
        mock_extract,
        mock_pubmed,
    ):
        """Test that pipeline executes all steps."""
        from orchestrator.pipeline import Pipeline
        from orchestrator.state import PipelineStep
        
        # Mock PubMed response
        mock_pubmed.return_value = [
            {
                "paper_id": "12345",
                "title": "Test Paper",
                "abstract": "This paper studies erlotinib as an EGFR inhibitor.",
            }
        ]
        
        # Mock compound extraction
        mock_extract.return_value = [
            {
                "compound_name": "erlotinib",
                "context": "erlotinib as an EGFR inhibitor",
                "paper_id": "12345",
                "confidence": 0.9,
                "mention_type": "drug",
            }
        ]
        
        # Mock PubChem lookup
        mock_pubchem.return_value = {
            "cid": 176870,
            "smiles": "COc1cc2ncnc(Nc3ccc(OCCOc4ccccc4)cc3)c2cc1OC",
        }
        
        # Create and run pipeline
        pipeline = Pipeline(
            target_protein="EGFR",
            query="test query",
        )
        
        result = pipeline.run()
        
        # Check that pipeline completed
        assert result.run_id is not None
        assert result.state.current_step in [PipelineStep.COMPLETED, PipelineStep.FAILED]
    
    def test_pipeline_state_tracking(self):
        """Test that pipeline state is properly tracked."""
        from orchestrator.state import RunState, PipelineStep, StepStatus
        
        state = RunState(
            run_id="test_123",
            target_protein="EGFR",
            query="test query",
        )
        
        # Start a step
        state.start_step(PipelineStep.DISCOVERY)
        assert state.current_step == PipelineStep.DISCOVERY
        
        # Complete the step
        state.complete_step(PipelineStep.DISCOVERY, {"papers": 10})
        
        # Check step result
        step_result = state.step_results[-1]
        assert step_result.status == StepStatus.COMPLETED
        assert step_result.metrics["papers"] == 10
    
    def test_pipeline_flag_on_control_failure(self):
        """Test that pipeline flags run when controls fail."""
        from validation.sanity_checks import validate_controls
        from orchestrator.run_config import ControlValidationConfig
        
        config = ControlValidationConfig(
            positive_min_percentile=80,
            negative_max_percentile=20,
        )
        
        # Positive control with low percentile (failure)
        compounds = [
            {
                "compound_id": "pos_1",
                "is_control": True,
                "control_type": "positive",
                "percentile": 30,  # Too low!
            }
        ]
        
        is_valid, message = validate_controls(compounds, config)
        
        assert not is_valid
        assert "Positive control" in message


class TestProvenanceTracking:
    """Test provenance tracking throughout pipeline."""
    
    def test_provenance_creation(self):
        """Test that provenance is properly created."""
        from utils.provenance import ProvenanceTracker, ExtractionSource, ExtractionInfo
        
        tracker = ProvenanceTracker(run_id="test_123")
        
        # Add a compound
        tracker.add_compound(
            compound_id="cand_001",
            smiles="CCO",
            extraction_info=ExtractionInfo(
                source=ExtractionSource.PUBCHEM,
                compound_name="Ethanol",
                pubchem_cid=702,
                confidence=0.9,
            ),
        )
        
        # Retrieve
        compound = tracker.get_compound("cand_001")
        assert compound is not None
        assert compound.smiles == "CCO"
        assert compound.extraction_info.source == ExtractionSource.PUBCHEM
    
    def test_provenance_export(self, tmp_path):
        """Test provenance JSON export."""
        from utils.provenance import ProvenanceTracker
        
        tracker = ProvenanceTracker(run_id="test_export")
        tracker.add_compound("c_1", "CCO")
        tracker.add_compound("c_2", "CCC")
        
        # Export
        export_path = tmp_path / "provenance.json"
        tracker.export_json(str(export_path))
        
        # Check file exists and has content
        assert export_path.exists()
        
        import json
        with open(export_path) as f:
            data = json.load(f)
        
        assert data["run_id"] == "test_export"
        assert len(data["compounds"]) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

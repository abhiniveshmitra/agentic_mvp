"""
Tests for normalization.

Validates percentile calculation, z-score, and confidence tiers.
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestNormalization:
    """Test score normalization functionality."""
    
    def test_percentile_calculation(self):
        """Test percentile rank calculation."""
        from validation.normalization import normalize_scores
        from orchestrator.run_config import NormalizationConfig
        
        config = NormalizationConfig()
        
        # Create compounds with known scores
        compounds = [
            {"compound_id": f"c_{i}", "raw_score": float(i), "is_control": False}
            for i in range(10)
        ]
        
        result = normalize_scores(compounds, config)
        
        # Highest score should have highest percentile
        highest = max(result, key=lambda x: x["raw_score"])
        assert highest["percentile"] > 80
        
        # Lowest score should have low percentile
        lowest = min(result, key=lambda x: x["raw_score"])
        assert lowest["percentile"] < 20
    
    def test_zscore_threshold(self):
        """Test that z-score is only calculated for batches >= 30."""
        from validation.normalization import normalize_scores
        from orchestrator.run_config import NormalizationConfig
        
        config = NormalizationConfig(min_batch_for_zscore=30)
        
        # Small batch - no z-score
        small_batch = [
            {"compound_id": f"c_{i}", "raw_score": float(i), "is_control": False}
            for i in range(10)
        ]
        result_small = normalize_scores(small_batch, config)
        assert result_small[0]["z_score"] is None
        
        # Large batch - should have z-score
        large_batch = [
            {"compound_id": f"c_{i}", "raw_score": float(i), "is_control": False}
            for i in range(50)
        ]
        result_large = normalize_scores(large_batch, config)
        assert result_large[0]["z_score"] is not None
    
    def test_controls_excluded_from_stats(self):
        """Test that controls are excluded from statistics."""
        from validation.normalization import normalize_scores
        from orchestrator.run_config import NormalizationConfig
        
        config = NormalizationConfig(exclude_controls=True)
        
        # Candidates with normal scores
        compounds = [
            {"compound_id": f"c_{i}", "raw_score": 5.0, "is_control": False}
            for i in range(5)
        ]
        
        # Add a control with extreme score
        compounds.append({
            "compound_id": "control_1",
            "raw_score": 100.0,  # Extreme value
            "is_control": True,
            "control_type": "positive",
        })
        
        result = normalize_scores(compounds, config)
        
        # Control should still have percentile calculated
        control = [c for c in result if c["is_control"]][0]
        assert "percentile" in control


class TestConfidenceTiers:
    """Test confidence tier assignment."""
    
    def test_high_confidence_tier(self):
        """Test HIGH tier assignment."""
        from validation.normalization import normalize_scores
        from orchestrator.run_config import NormalizationConfig
        
        config = NormalizationConfig()
        
        # Create compounds where one has very high score
        compounds = [
            {"compound_id": f"c_{i}", "raw_score": float(i), "is_control": False, "source": "PUBCHEM"}
            for i in range(20)
        ]
        
        result = normalize_scores(compounds, config)
        
        # Highest scorer with PUBCHEM source should be HIGH
        highest = max(result, key=lambda x: x["raw_score"])
        assert highest["confidence_tier"] == "HIGH"
    
    def test_llm_inferred_always_low(self):
        """Test that LLM_INFERRED source is always LOW confidence."""
        from validation.normalization import normalize_scores
        from orchestrator.run_config import NormalizationConfig
        
        config = NormalizationConfig()
        
        compounds = [
            {"compound_id": "high_score", "raw_score": 100.0, "is_control": False, "source": "LLM_INFERRED"},
        ]
        
        result = normalize_scores(compounds, config)
        
        # Even with high score, LLM_INFERRED should be LOW
        assert result[0]["confidence_tier"] == "LOW"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

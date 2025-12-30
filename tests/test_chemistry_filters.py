"""
Tests for chemistry filters.

Validates that the hard gate filters work correctly.
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestChemistryFilters:
    """Test chemistry filter functionality."""
    
    def test_valid_smiles_parsing(self):
        """Test that valid SMILES are parsed correctly."""
        from validation.chemistry_filters import validate_smiles
        
        # Valid drug-like SMILES
        valid_smiles = [
            "CC(=O)Nc1ccc(O)cc1",  # Acetaminophen
            "Cn1cnc2c1c(=O)n(C)c(=O)n2C",  # Caffeine
            "CC(C)Cc1ccc(C(C)C(=O)O)cc1",  # Ibuprofen
        ]
        
        for smiles in valid_smiles:
            is_valid, canonical = validate_smiles(smiles)
            assert is_valid, f"SMILES should be valid: {smiles}"
            assert canonical is not None
    
    def test_invalid_smiles_parsing(self):
        """Test that invalid SMILES are rejected."""
        from validation.chemistry_filters import validate_smiles
        
        invalid_smiles = [
            "not_a_smiles",
            "XXXXX",
            "C(C(C",  # Unbalanced parentheses
        ]
        
        for smiles in invalid_smiles:
            is_valid, canonical = validate_smiles(smiles)
            assert not is_valid, f"SMILES should be invalid: {smiles}"
    
    def test_molecular_weight_filter(self):
        """Test molecular weight filtering."""
        from validation.chemistry_filters import calculate_properties
        
        # Small molecule (should fail MW > 150)
        methane_props = calculate_properties("C")
        if methane_props:
            assert methane_props["molecular_weight"] < 150
        
        # Drug-like molecule
        caffeine_props = calculate_properties("Cn1cnc2c1c(=O)n(C)c(=O)n2C")
        if caffeine_props:
            assert 150 <= caffeine_props["molecular_weight"] <= 700
    
    def test_logp_calculation(self):
        """Test LogP calculation."""
        from validation.chemistry_filters import calculate_properties
        
        # Hydrophilic molecule (low LogP)
        glucose_props = calculate_properties("OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O")
        if glucose_props:
            assert glucose_props["logp"] < 0
        
        # Hydrophobic molecule (high LogP)
        # Long alkane chain
        props = calculate_properties("CCCCCCCCCCCCCCCC")
        if props:
            assert props["logp"] > 3


class TestFilterApplication:
    """Test filter application to compound batches."""
    
    def test_filter_batch(self):
        """Test filtering a batch of compounds."""
        from validation.chemistry_filters import apply_all_filters
        from orchestrator.run_config import ChemistryFilterConfig
        
        config = ChemistryFilterConfig()
        
        compounds = [
            {"compound_id": "test_1", "smiles": "CC(=O)Nc1ccc(O)cc1"},  # Valid
            {"compound_id": "test_2", "smiles": "invalid"},  # Invalid
            {"compound_id": "test_3", "smiles": "C"},  # Too small
        ]
        
        passed, rejected = apply_all_filters(compounds, config)
        
        # At least one should pass, at least one should be rejected
        assert len(passed) >= 1
        assert len(rejected) >= 1
    
    def test_rejected_compounds_preserved(self):
        """Test that rejected compounds are preserved (not deleted)."""
        from validation.chemistry_filters import apply_all_filters
        from orchestrator.run_config import ChemistryFilterConfig
        
        config = ChemistryFilterConfig()
        
        compounds = [
            {"compound_id": "reject_me", "smiles": "invalid_smiles"},
        ]
        
        passed, rejected = apply_all_filters(compounds, config)
        
        # Compound should be in rejected list
        assert len(rejected) == 1
        assert rejected[0]["compound_id"] == "reject_me"
        assert "rejection_reason" in rejected[0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

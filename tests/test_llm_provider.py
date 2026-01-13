"""
Tests for LLM Provider Abstraction.

These tests use MockProvider ONLY - NO model downloads required.
This keeps CI fast, free, and stable.
"""

import unittest
from unittest.mock import patch, MagicMock
import os
import sys

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.llm_provider import (
    LLMProvider,
    MockProvider,
    PhiLocalProvider,
    GeminiProvider,
    create_llm_provider,
    GENERATION_CONSTRAINTS,
)


class TestGenerationConstraints(unittest.TestCase):
    """Test that generation constraints enforce determinism."""
    
    def test_temperature_is_zero(self):
        """Temperature MUST be 0.0 for deterministic output."""
        self.assertEqual(GENERATION_CONSTRAINTS["temperature"], 0.0)
    
    def test_do_sample_is_false(self):
        """Sampling MUST be disabled for determinism."""
        self.assertEqual(GENERATION_CONSTRAINTS["do_sample"], False)
    
    def test_max_tokens_capped(self):
        """Max tokens MUST be capped to prevent runaway."""
        self.assertLessEqual(GENERATION_CONSTRAINTS["max_new_tokens"], 512)


class TestMockProvider(unittest.TestCase):
    """Test MockProvider for CI testing."""
    
    def test_mock_is_always_available(self):
        """MockProvider should always be available."""
        provider = MockProvider()
        self.assertTrue(provider.is_available())
    
    def test_mock_provider_name(self):
        """MockProvider should identify itself correctly."""
        provider = MockProvider()
        self.assertEqual(provider.provider_name, "mock")
    
    def test_mock_generates_output(self):
        """MockProvider should generate non-empty output."""
        provider = MockProvider()
        result = provider.generate("Test prompt")
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)
    
    def test_mock_is_deterministic(self):
        """MockProvider should give same output for same prompt."""
        provider = MockProvider()
        prompt = "Summarize this compound data"
        result1 = provider.generate(prompt)
        result2 = provider.generate(prompt)
        self.assertEqual(result1, result2)
    
    def test_mock_different_prompts_different_output(self):
        """MockProvider should give different output for different prompts."""
        provider = MockProvider()
        result1 = provider.generate("Prompt A")
        result2 = provider.generate("Prompt B")
        self.assertNotEqual(result1, result2)


class TestProviderFactory(unittest.TestCase):
    """Test create_llm_provider factory function."""
    
    def test_create_mock_provider(self):
        """Factory should create MockProvider when requested."""
        provider = create_llm_provider("mock")
        self.assertIsInstance(provider, MockProvider)
        self.assertEqual(provider.provider_name, "mock")
    
    def test_factory_returns_provider(self):
        """Factory should always return a valid LLMProvider."""
        provider = create_llm_provider("mock")
        self.assertTrue(hasattr(provider, "generate"))
        self.assertTrue(hasattr(provider, "is_available"))
        self.assertTrue(hasattr(provider, "provider_name"))
    
    @patch.dict(os.environ, {"LLM_BACKEND": "mock"})
    def test_factory_respects_env_var(self):
        """Factory should respect LLM_BACKEND env variable."""
        provider = create_llm_provider()
        self.assertEqual(provider.provider_name, "mock")


class TestProviderInterface(unittest.TestCase):
    """Test that all providers implement the interface correctly."""
    
    def test_mock_implements_interface(self):
        """MockProvider should implement LLMProvider interface."""
        provider = MockProvider()
        # Check all required methods exist
        self.assertTrue(callable(provider.generate))
        self.assertTrue(callable(provider.is_available))
        self.assertIsInstance(provider.provider_name, str)


class TestNarrativeGeneratorIntegration(unittest.TestCase):
    """Test narrative generator with mock provider."""
    
    def test_narrative_generator_with_mock(self):
        """NarrativeGenerator should work with mock provider."""
        from stage2.narrative_generator import create_narrative_generator
        
        generator = create_narrative_generator(use_llm=True, llm_backend="mock")
        
        # Create a minimal test candidate
        candidate = {
            "compound_name": "Test-Compound",
            "smiles": "CCO",
            "stage1": {
                "rank": 1,
                "score": 8.5,
                "percentile": 95.0,
            },
            "stage2": {
                "docking": {"docking_status": "PASS"},
                "adme_tox": {"adme_tox_label": "SAFE"},
            },
        }
        
        narrative = generator.generate_narrative(candidate)
        self.assertIsInstance(narrative, str)
        self.assertGreater(len(narrative), 0)


if __name__ == "__main__":
    unittest.main()

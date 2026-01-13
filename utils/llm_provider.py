"""
LLM Provider Abstraction for Deterministic Summarization.

This provider is configured for deterministic summarization only.
Any reasoning-style output is considered a bug.

Rules (IMMUTABLE):
    Temperature = 0.0 (no randomness)
    Max tokens capped (prevent runaway generation)
    No chain-of-thought prompting
    No system prompts that invite reasoning
    Summarize ONLY, never add new information
"""

from abc import ABC, abstractmethod
from typing import Optional
import os
import hashlib

from utils.logging import get_logger

logger = get_logger(__name__)


# =============================================================================
# DETERMINISTIC GENERATION CONSTRAINTS (IMMUTABLE)
# =============================================================================

GENERATION_CONSTRAINTS = {
    "temperature": 0.0,
    "max_new_tokens": 256,
    "top_p": 1.0,
    "do_sample": False,
    "repetition_penalty": 1.0,
}


# =============================================================================
# ABSTRACT BASE CLASS
# =============================================================================

class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate text from a prompt (deterministic)."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the provider is available."""
        pass
    
    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider name."""
        pass


# =============================================================================
# MOCK PROVIDER (for CI/testing - NO model download required)
# =============================================================================

class MockProvider(LLMProvider):
    """
    Mock provider for CI testing.
    
    Returns deterministic hash-based output. Does NOT require any model.
    Used for:
     - CI tests (T13-T17) that validate behavior, not language quality
     - Unit tests for provider wiring
    """
    
    def __init__(self, prefix: str = "Summary:"):
        self._prefix = prefix
    
    def generate(self, prompt: str) -> str:
        """Return deterministic mock output based on prompt hash."""
        # Hash the prompt for deterministic output
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()[:8]
        return f"{self._prefix} Mock summary [{prompt_hash}]. This compound shows predicted affinity based on available data."
    
    def is_available(self) -> bool:
        return True
    
    @property
    def provider_name(self) -> str:
        return "mock"


# =============================================================================
# PHI LOCAL PROVIDER (Hugging Face Transformers)
# =============================================================================

class PhiLocalProvider(LLMProvider):
    """
    Phi-3 model via Hugging Face Transformers for local inference.
    
    This provider is configured for deterministic summarization only.
    Any reasoning-style output is considered a bug.
    """
    
    MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"
    
    def __init__(self):
        """Initialize Phi-3 model (lazy loading)."""
        self._model = None
        self._tokenizer = None
        self._available = None
        
    def _load_model(self):
        """Lazy load model on first use."""
        if self._model is not None:
            return
            
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            logger.info(f"Loading Phi-3 model: {self.MODEL_ID}")
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.float16 if device == "cuda" else torch.float32
            
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.MODEL_ID, trust_remote_code=True
            )
            
            self._model = AutoModelForCausalLM.from_pretrained(
                self.MODEL_ID,
                torch_dtype=dtype,
                device_map="auto" if device == "cuda" else None,
                trust_remote_code=True
            )
            
            if device == "cpu":
                self._model = self._model.to(device)
                
            self._available = True
            logger.info(f"Phi-3 loaded successfully on {device}")
            
        except Exception as e:
            logger.error(f"Failed to load Phi-3: {e}")
            self._available = False
            raise
    
    def generate(self, prompt: str) -> str:
        """
        Generate text using Phi-3 with deterministic settings.
        Temperature is fixed at 0.0 - intentional and immutable.
        """
        self._load_model()
        
        # Format for Phi-3 instruct - NO reasoning prompts
        messages = [
            {"role": "user", "content": prompt}
        ]
        
        # Apply chat template
        text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = self._tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
        
        # Generate with STRICT deterministic settings
        outputs = self._model.generate(
            **inputs,
            max_new_tokens=GENERATION_CONSTRAINTS["max_new_tokens"],
            do_sample=GENERATION_CONSTRAINTS["do_sample"],
            temperature=GENERATION_CONSTRAINTS["temperature"],
            top_p=GENERATION_CONSTRAINTS["top_p"],
            repetition_penalty=GENERATION_CONSTRAINTS["repetition_penalty"],
            pad_token_id=self._tokenizer.eos_token_id,
        )
        
        # Decode and extract response
        full_text = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract assistant response (after the prompt)
        if "<|assistant|>" in full_text:
            response = full_text.split("<|assistant|>")[-1].strip()
        else:
            response = full_text[len(prompt):].strip()
        
        return response
    
    def is_available(self) -> bool:
        if self._available is None:
            try:
                from transformers import AutoTokenizer
                AutoTokenizer.from_pretrained(self.MODEL_ID, trust_remote_code=True)
                self._available = True
            except Exception:
                self._available = False
        return self._available
    
    @property
    def provider_name(self) -> str:
        return "phi_local"


# =============================================================================
# GEMINI PROVIDER (fallback for backward compatibility)
# =============================================================================

class GeminiProvider(LLMProvider):
    """
    Gemini API provider (backward compatible fallback).
    
    Uses temperature=0 for deterministic output.
    """
    
    def __init__(self):
        self._model = None
        self._available = None
        self._api_key = os.getenv("GOOGLE_API_KEY")
        
    def _init_model(self):
        if self._model is not None:
            return
            
        if not self._api_key:
            raise ValueError("GOOGLE_API_KEY not set")
            
        try:
            import google.generativeai as genai
            genai.configure(api_key=self._api_key)
            
            # Use generation config for determinism
            generation_config = {
                "temperature": 0.0,
                "max_output_tokens": 256,
            }
            
            self._model = genai.GenerativeModel(
                'gemini-2.0-flash',
                generation_config=generation_config
            )
            self._available = True
            logger.info("Gemini provider initialized")
            
        except Exception as e:
            logger.error(f"Failed to init Gemini: {e}")
            self._available = False
            raise
    
    def generate(self, prompt: str) -> str:
        """Generate using Gemini with temperature=0."""
        self._init_model()
        
        response = self._model.generate_content(prompt)
        
        if response.text:
            return response.text.strip()
        return ""
    
    def is_available(self) -> bool:
        if self._available is None:
            self._available = bool(self._api_key)
        return self._available
    
    @property
    def provider_name(self) -> str:
        return "gemini"


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_llm_provider(backend: Optional[str] = None) -> LLMProvider:
    """
    Create an LLM provider based on configuration.
    
    Args:
        backend: One of 'phi_local', 'gemini', 'mock'. 
                 If None, uses LLM_BACKEND env var or defaults to 'phi_local'.
    
    Returns:
        An LLMProvider instance.
    
    Priority for local development:
        1. phi_local (requires GPU/CPU with transformers)
        2. gemini (requires GOOGLE_API_KEY)
        3. mock (always works, for testing)
    """
    if backend is None:
        backend = os.getenv("LLM_BACKEND", "phi_local")
    
    backend = backend.lower()
    
    if backend == "mock":
        logger.info("Using MockProvider (testing mode)")
        return MockProvider()
    
    if backend == "phi_local":
        try:
            provider = PhiLocalProvider()
            if provider.is_available():
                logger.info("Using PhiLocalProvider")
                return provider
        except Exception as e:
            logger.warning(f"Phi not available: {e}, trying Gemini fallback")
    
    if backend == "gemini" or backend == "phi_local":
        try:
            provider = GeminiProvider()
            if provider.is_available():
                logger.info("Using GeminiProvider (fallback)")
                return provider
        except Exception as e:
            logger.warning(f"Gemini not available: {e}")
    
    # Final fallback to mock
    logger.warning("No LLM available, using MockProvider")
    return MockProvider()


# Quick test
if __name__ == "__main__":
    provider = create_llm_provider("mock")
    print(f"Provider: {provider.provider_name}")
    result = provider.generate("Summarize: Test molecule with high affinity")
    print(f"Result: {result}")

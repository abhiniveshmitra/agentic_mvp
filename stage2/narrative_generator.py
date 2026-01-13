"""
Narrative Summary Generator (LLM-Only Summarization).

Purpose: Generate 3-5 sentence human-readable summaries for each molecule.

Rules (IMMUTABLE):
- SUMMARIZE ONLY - no new reasoning
- Based strictly on explanation schema data
- Must mention: Stage-1 ranking, Stage-2 status, key risks
- NEVER alter labels or scores
- NEVER fabricate information not in the explanation

This is LLM summarization mode only.
"""

from typing import Dict, List, Optional
import os

from utils.logging import get_logger

logger = get_logger(__name__)


class NarrativeSummaryGenerator:
    """
    Generates human-readable narrative summaries for Stage-2 candidates.
    
    Uses OpenAI or Gemini API for natural language generation, but ONLY for summarization.
    Falls back to rule-based summaries if API unavailable.
    """
    
    def __init__(self, use_llm: bool = True):
        """
        Initialize narrative generator.
        
        Args:
            use_llm: Whether to use LLM for generation (requires API key)
        """
        from config.settings import LLM_PROVIDER, LLM_CONFIG, OPENAI_API_KEY, GOOGLE_API_KEY
        
        self.llm_provider = LLM_PROVIDER if use_llm else None
        self.use_llm = use_llm and bool(LLM_PROVIDER)
        self.model = None
        self.client = None
        
        if self.use_llm:
            try:
                if self.llm_provider == "openai":
                    from openai import OpenAI
                    self.client = OpenAI(api_key=OPENAI_API_KEY)
                    self.llm_config = LLM_CONFIG["openai"]
                    logger.info(f"NarrativeSummaryGenerator initialized with OpenAI {self.llm_config['model']}")
                else:  # gemini
                    import google.generativeai as genai
                    genai.configure(api_key=GOOGLE_API_KEY)
                    self.model = genai.GenerativeModel(LLM_CONFIG["gemini"]["model"])
                    self.llm_config = LLM_CONFIG["gemini"]
                    logger.info(f"NarrativeSummaryGenerator initialized with Gemini {self.llm_config['model']}")
            except Exception as e:
                logger.warning(f"LLM init failed: {e}, using rule-based fallback")
                self.use_llm = False
        else:
            logger.info("NarrativeSummaryGenerator initialized (rule-based mode)")
    
    def generate_narrative(self, candidate: Dict) -> str:
        """
        Generate narrative summary for a candidate.
        
        Args:
            candidate: Aggregated candidate dictionary
        
        Returns:
            3-5 sentence narrative summary
        """
        if self.use_llm:
            return self._generate_with_llm(candidate)
        else:
            return self._generate_rule_based(candidate)
    
    def generate_batch(self, candidates: List[Dict]) -> List[str]:
        """
        Generate narratives for a batch of candidates.
        
        Args:
            candidates: List of aggregated candidate dictionaries
        
        Returns:
            List of narrative summaries
        """
        narratives = []
        for candidate in candidates:
            narrative = self.generate_narrative(candidate)
            narratives.append(narrative)
        return narratives
    
    def _generate_with_llm(self, candidate: Dict) -> str:
        """Generate narrative using OpenAI or Gemini API."""
        try:
            # Build context from explanation schema
            context = self._build_context(candidate)
            
            prompt = f"""You are a scientific summarizer for drug discovery pipelines.

Generate a 3-5 sentence summary for this molecule based STRICTLY on the provided data.
Do NOT add any new information or reasoning not present in the data.
Do NOT alter any labels or scores.
Focus on:
1. Why it ranked well in Stage-1 (affinity prediction)
2. Why it passed or was flagged in Stage-2 (structural/ADME checks)
3. Key risks and limitations that a medicinal chemist should know

Here is the data:
{context}

Generate a clear, scientific, but accessible summary:"""
            
            if self.llm_provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.llm_config["model"],
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.llm_config["temperature"],
                    max_tokens=self.llm_config["max_tokens"],
                )
                text = response.choices[0].message.content
            else:  # gemini
                response = self.model.generate_content(prompt)
                text = response.text
            
            if text:
                return text.strip()
            else:
                return self._generate_rule_based(candidate)
                
        except Exception as e:
            logger.warning(f"LLM generation failed: {e}")
            return self._generate_rule_based(candidate)
    
    def _generate_rule_based(self, candidate: Dict) -> str:
        """Generate rule-based narrative (fallback)."""
        parts = []
        
        # Compound identity
        name = candidate.get("compound_name", "This compound")
        
        # Stage-1 summary
        stage1 = candidate.get("stage1", {})
        rank = stage1.get("rank", "N/A")
        percentile = stage1.get("percentile", "N/A")
        score = stage1.get("score", "N/A")
        
        parts.append(
            f"{name} ranks #{rank} in Stage-1 affinity prediction "
            f"(top {100 - percentile:.0f}% percentile) with a predicted affinity score of {score:.2f}."
        )
        
        # Stage-2: Docking
        stage2 = candidate.get("stage2", {})
        docking = stage2.get("docking", {})
        docking_status = docking.get("docking_status", "N/A")
        
        if docking_status == "PASS":
            parts.append(
                "Docking analysis confirms structural compatibility with the target - "
                "the molecule fits the binding pocket without major steric clashes."
            )
        elif docking_status == "FLAG":
            parts.append(
                "Docking suggests marginal fit - binding may require conformational "
                "adjustments or induced fit effects not captured in rigid docking."
            )
        elif docking_status == "FAIL":
            error = docking.get("error", "structural incompatibility")
            parts.append(f"Docking failed ({error}), indicating structural concerns.")
        
        # Stage-2: ADME/Tox
        adme = stage2.get("adme_tox", {})
        adme_label = adme.get("adme_tox_label", "N/A")
        
        if adme_label == "SAFE":
            parts.append(
                "No ADME/Tox flags were triggered - the compound passes developability screening."
            )
        elif adme_label == "FLAGGED":
            # Get triggered rules
            exp = adme.get("explanation", {})
            rules = exp.get("rules_triggered", [])
            if rules:
                rule_names = [r.get("rule_id", "").replace("_", " ").lower() for r in rules[:2]]
                parts.append(
                    f"ADME/Tox assessment flags potential concerns: {', '.join(rule_names)}. "
                    "These suggest elevated developability risk but do not invalidate the compound."
                )
            else:
                parts.append(
                    "ADME/Tox assessment flags moderate developability risk. "
                    "Further optimization may be needed."
                )
        elif adme_label == "HIGH_RISK":
            parts.append(
                "HIGH-RISK ADME/Tox profile detected. Multiple flags suggest significant "
                "developability challenges requiring careful assessment."
            )
        
        # Limitations
        parts.append(
            "Note: Stage-2 assessments are structural plausibility checks only - "
            "experimental validation is required to confirm binding and safety."
        )
        
        return " ".join(parts)
    
    def _build_context(self, candidate: Dict) -> str:
        """Build context string from candidate data for LLM."""
        lines = []
        
        # Basic info
        lines.append(f"Compound: {candidate.get('compound_name', 'Unknown')}")
        lines.append(f"SMILES: {candidate.get('smiles', 'N/A')}")
        
        # Stage-1
        stage1 = candidate.get("stage1", {})
        lines.append(f"\nSTAGE-1 AFFINITY PREDICTION:")
        lines.append(f"  Rank: #{stage1.get('rank')}")
        lines.append(f"  Score: {stage1.get('score')}")
        lines.append(f"  Percentile: {stage1.get('percentile')}%")
        
        # Stage-2: Docking
        stage2 = candidate.get("stage2", {})
        docking = stage2.get("docking", {})
        lines.append(f"\nDOCKING:")
        lines.append(f"  Status: {docking.get('docking_status')}")
        if docking.get("explanation", {}).get("observations"):
            for obs in docking["explanation"]["observations"]:
                lines.append(f"  - {obs}")
        
        # Stage-2: ADME/Tox
        adme = stage2.get("adme_tox", {})
        lines.append(f"\nADME/TOX:")
        lines.append(f"  Label: {adme.get('adme_tox_label')}")
        exp = adme.get("explanation", {})
        if exp.get("raw_values"):
            rv = exp["raw_values"]
            lines.append(f"  Properties: MW={rv.get('mw')}, LogP={rv.get('logp')}, TPSA={rv.get('tpsa')}")
        if exp.get("rules_triggered"):
            lines.append("  Rules triggered:")
            for rule in exp["rules_triggered"]:
                lines.append(f"    - {rule.get('rule_id')}: {rule.get('condition')}")
        if exp.get("pains_details"):
            lines.append(f"  PAINS Alert: {exp['pains_details'].get('matched_motif')}")
        
        # Limitations
        if exp.get("limitations"):
            lines.append("\nLIMITATIONS:")
            for lim in exp["limitations"]:
                lines.append(f"  - {lim}")
        
        return "\n".join(lines)


def create_narrative_generator(use_llm: bool = True) -> NarrativeSummaryGenerator:
    """Factory function to create NarrativeSummaryGenerator."""
    return NarrativeSummaryGenerator(use_llm=use_llm)


# Quick test
if __name__ == "__main__":
    from stage2.trust_validation import run_known_drug_validation
    
    results = run_known_drug_validation("EGFR")
    generator = create_narrative_generator(use_llm=True)
    
    for candidate in results.get("candidates", []):
        print(f"\n{'='*60}")
        print(f"Compound: {candidate['compound_name']}")
        print(f"{'='*60}")
        narrative = generator.generate_narrative(candidate)
        print(narrative)

"""
LLM Decision Summary Agent.

Purpose: Generate human-readable explanations for compound decisions.
LLM = NARRATOR ONLY - explains decisions, never changes them.

Rules (IMMUTABLE):
- decision_label is pre-computed and MUST be echoed verbatim
- LLM fills templates, never decides
- ml_status NOT_APPLICABLE → do not reference ML scores/uncertainty
- No recommendations, no optimism, no comparisons

Anchor Statement:
> Rules decide. Docking constrains. ML refines. LLM explains. Humans choose.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum
import json

from stage2.agents.decision_label import (
    compute_decision_label,
    validate_llm_echoes_label,
    DecisionLabel,
    DecisionLabelResult,
)
from utils.logging import get_logger

logger = get_logger(__name__)


# Forbidden phrases in LLM output (checked by T17)
FORBIDDEN_PHRASES = [
    "recommended",
    "should be tested",
    "likely to succeed",
    "promising drug candidate",
    "promising candidate",
    "we recommend",
    "suggest testing",
    "worth exploring",
    "high potential",
]


@dataclass
class LLMSummaryInput:
    """
    Allowed inputs for LLM summary generation (READ-ONLY).
    
    The LLM may ONLY read these fields.
    Forbidden: thresholds, filtering logic, ranking logic, cross-compound comparisons.
    """
    
    # Stage-1
    stage1_rank: Optional[int] = None
    stage1_percentile: Optional[float] = None
    stage1_affinity_score: Optional[float] = None
    stage1_confidence_tier: Optional[str] = None
    
    # Stage-2 (ADME/Tox)
    adme_status: str = "SAFE"
    adme_rules_triggered: List[str] = field(default_factory=list)
    pains_alerts: List[str] = field(default_factory=list)
    
    # Stage-2.1 (Docking)
    docking_status: str = "NOT_EVALUATED"
    vina_score: Optional[float] = None
    docking_observations: List[str] = field(default_factory=list)
    docking_failure_reason: Optional[str] = None
    
    # Stage-3 (ML Refinement)
    ml_status: str = "NOT_APPLICABLE"
    ml_affinity_score: Optional[float] = None
    ml_uncertainty: Optional[float] = None
    
    # Metadata
    target_id: Optional[str] = None
    compound_id: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            "stage1": {
                "rank": self.stage1_rank,
                "percentile": self.stage1_percentile,
                "affinity_score": self.stage1_affinity_score,
                "confidence_tier": self.stage1_confidence_tier,
            },
            "adme": {
                "status": self.adme_status,
                "rules_triggered": self.adme_rules_triggered,
                "pains_alerts": self.pains_alerts,
            },
            "docking": {
                "status": self.docking_status,
                "vina_score": self.vina_score,
                "observations": self.docking_observations,
                "failure_reason": self.docking_failure_reason,
            },
            "ml": {
                "status": self.ml_status,
                "affinity_score": self.ml_affinity_score,
                "uncertainty": self.ml_uncertainty,
            },
            "metadata": {
                "target_id": self.target_id,
                "compound_id": self.compound_id,
            }
        }


@dataclass
class LLMSummaryOutput:
    """
    Required output schema for LLM summary.
    
    The decision_label MUST match the pre-computed label exactly.
    """
    
    decision_label: str
    summary: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "decision_label": self.decision_label,
            "summary": self.summary,
        }


class LLMSummaryAgent:
    """
    LLM Decision Summary Agent.
    
    NARRATOR ONLY - generates explanations, never changes decisions.
    
    Behavior Constraints:
        ✅ Summarize existing signals
        ✅ Use conservative, factual language
        ✅ Reflect uncertainty honestly
        ❌ Never recommend actions
        ❌ Never contradict decision_label
        ❌ Never invent reasons
        ❌ Never suppress risks
        ❌ Never compare compounds
    """
    
    def __init__(self, use_llm: bool = False):
        """
        Initialize LLM Summary Agent.
        
        Args:
            use_llm: If True, use actual LLM for generation.
                     If False, use template-based generation (deterministic).
        """
        self.use_llm = use_llm
        logger.info(f"LLMSummaryAgent initialized (use_llm={use_llm})")
    
    def generate_summary(
        self,
        inputs: LLMSummaryInput,
    ) -> LLMSummaryOutput:
        """
        Generate decision summary for a compound.
        
        The decision_label is computed FIRST using deterministic rules.
        The summary is then generated to EXPLAIN (not decide).
        
        Args:
            inputs: LLMSummaryInput with allowed fields only
        
        Returns:
            LLMSummaryOutput with decision_label and summary
            
        Raises:
            ValueError: If LLM attempts to change decision_label
        """
        # Step 1: Compute decision label DETERMINISTICALLY
        label_result = compute_decision_label(
            adme_status=inputs.adme_status,
            docking_status=inputs.docking_status,
            ml_status=inputs.ml_status,
        )
        
        decision_label = label_result.decision_label
        
        # Step 2: Generate summary (template-based or LLM)
        if self.use_llm:
            summary = self._generate_with_llm(inputs, decision_label)
        else:
            summary = self._generate_with_template(inputs, decision_label)
        
        # Step 3: Validate output
        output = LLMSummaryOutput(
            decision_label=decision_label,
            summary=summary,
        )
        
        self._validate_output(output, decision_label)
        
        return output
    
    def _generate_with_template(
        self,
        inputs: LLMSummaryInput,
        decision_label: str,
    ) -> Dict:
        """
        Generate summary using deterministic templates.
        
        Templates are anchored to decision labels.
        ML references are ONLY included if ml_status == APPLIED.
        """
        if decision_label == DecisionLabel.PRIORITIZED.value:
            return self._template_prioritized(inputs)
        elif decision_label == DecisionLabel.FLAGGED.value:
            return self._template_flagged(inputs)
        elif decision_label == DecisionLabel.DEPRIORITIZED.value:
            return self._template_deprioritized(inputs)
        else:
            raise ValueError(f"Unknown decision_label: {decision_label}")
    
    def _template_prioritized(self, inputs: LLMSummaryInput) -> Dict:
        """Template for PRIORITIZED compounds."""
        primary = (
            "This compound was prioritized because it satisfies chemical and "
            "developability constraints and shows structural plausibility for the target."
        )
        
        secondary = [
            "Predicted affinity places it among higher-ranked candidates.",
            "Docking indicates a plausible binding pose.",
        ]
        
        # ML reference ONLY if applied
        if inputs.ml_status == "APPLIED":
            secondary.append(
                "ML refinement provides additional context without overriding rule-based signals."
            )
            uncertainty_note = (
                "ML uncertainty reflects model disagreement and data coverage, "
                "not experimental error."
            )
        else:
            secondary.append("ML refinement was not applied to this compound.")
            uncertainty_note = "No ML uncertainty is reported because refinement was not applicable."
        
        limitations = [
            "Docking scores are heuristic and target-specific.",
            "No experimental validation has been performed.",
        ]
        
        return {
            "primary_reason": primary,
            "secondary_considerations": secondary,
            "uncertainty_note": uncertainty_note,
            "limitations": limitations,
        }
    
    def _template_flagged(self, inputs: LLMSummaryInput) -> Dict:
        """Template for FLAGGED compounds."""
        primary = (
            "This compound was flagged due to identified risk factors "
            "despite showing potential binding relevance."
        )
        
        secondary = []
        
        # Add specific reasons
        if inputs.adme_status == "FLAGGED":
            secondary.append("One or more developability rules were triggered.")
            if inputs.adme_rules_triggered:
                secondary.append(
                    f"ADME flags: {', '.join(inputs.adme_rules_triggered[:3])}"
                )
        
        if inputs.docking_status == "FLAG":
            secondary.append("Structural plausibility is borderline or context-dependent.")
            if inputs.docking_observations:
                secondary.append(
                    f"Docking observations: {', '.join(inputs.docking_observations[:2])}"
                )
        
        # ML reference ONLY if applied
        if inputs.ml_status == "APPLIED":
            secondary.append("ML refinement suggests mixed confidence.")
            uncertainty_note = (
                "ML uncertainty highlights disagreement between models "
                "and should be interpreted cautiously."
            )
        else:
            secondary.append("ML refinement was not applied to this compound.")
            uncertainty_note = "No ML uncertainty is reported because refinement was not applicable."
        
        limitations = [
            "Flagged status does not imply inactivity.",
            "Additional validation or mitigation strategies may be required.",
        ]
        
        return {
            "primary_reason": primary,
            "secondary_considerations": secondary,
            "uncertainty_note": uncertainty_note,
            "limitations": limitations,
        }
    
    def _template_deprioritized(self, inputs: LLMSummaryInput) -> Dict:
        """Template for DEPRIORITIZED compounds."""
        primary = (
            "This compound was deprioritized due to failure of hard constraints "
            "related to developability or structural compatibility."
        )
        
        secondary = [
            "Predicted affinity alone was insufficient to offset rule-based risks.",
        ]
        
        # Add specific reasons
        if inputs.adme_status == "HIGH_RISK":
            secondary.append(
                f"ADME/Tox status is HIGH_RISK due to critical developability failures."
            )
            if inputs.adme_rules_triggered:
                secondary.append(
                    f"Critical flags: {', '.join(inputs.adme_rules_triggered[:3])}"
                )
        
        if inputs.docking_status == "FAIL":
            secondary.append("Docking indicates structural incompatibility with the target.")
            if inputs.docking_failure_reason:
                secondary.append(f"Failure reason: {inputs.docking_failure_reason}")
        
        # ML was NOT applied (per hard fail boundaries)
        secondary.append("ML refinement was not applied due to prior failures.")
        uncertainty_note = "No ML uncertainty is reported because refinement was not applicable."
        
        limitations = [
            "Deprioritization reflects current rule-based constraints, not definitive inactivity.",
        ]
        
        return {
            "primary_reason": primary,
            "secondary_considerations": secondary,
            "uncertainty_note": uncertainty_note,
            "limitations": limitations,
        }
    
    def _generate_with_llm(
        self,
        inputs: LLMSummaryInput,
        decision_label: str,
    ) -> Dict:
        """
        Generate summary using actual LLM.
        
        The LLM receives:
        - The pre-computed decision_label (MUST echo verbatim)
        - The allowed inputs
        - The template structure to fill
        
        LLM is NOT allowed to change the decision_label.
        """
        # For now, fall back to template-based
        # TODO: Implement actual LLM call with strict validation
        logger.warning("LLM generation not implemented - using template fallback")
        return self._generate_with_template(inputs, decision_label)
    
    def _validate_output(
        self,
        output: LLMSummaryOutput,
        computed_label: str,
    ) -> None:
        """
        Validate LLM output meets all constraints.
        
        Raises ValueError if any constraint is violated.
        """
        # 1. Decision label must match
        validate_llm_echoes_label(computed_label, output.decision_label)
        
        # 2. Check for forbidden phrases
        summary_text = json.dumps(output.summary).lower()
        for phrase in FORBIDDEN_PHRASES:
            if phrase.lower() in summary_text:
                raise ValueError(
                    f"LLM output contains forbidden phrase: '{phrase}'. "
                    f"LLM must not recommend or express optimism."
                )
        
        # 3. Required fields present
        if "primary_reason" not in output.summary:
            raise ValueError("Missing required field: primary_reason")
        if "limitations" not in output.summary:
            raise ValueError("Missing required field: limitations")
        
        logger.debug("LLM output validation passed")
    
    def generate_batch(
        self,
        inputs_list: List[LLMSummaryInput],
    ) -> List[LLMSummaryOutput]:
        """Generate summaries for a batch of compounds."""
        outputs = []
        for inputs in inputs_list:
            try:
                output = self.generate_summary(inputs)
                outputs.append(output)
            except Exception as e:
                logger.error(f"Summary generation failed for {inputs.compound_id}: {e}")
                # Fail loudly - do not silently skip
                raise
        return outputs


def create_agent(use_llm: bool = False) -> LLMSummaryAgent:
    """Factory function to create LLM Summary Agent."""
    return LLMSummaryAgent(use_llm=use_llm)

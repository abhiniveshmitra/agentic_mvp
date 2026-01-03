"""
Decision Label Computer (NON-LLM).

Purpose: Compute deterministic decision labels BEFORE LLM runs.
The decision_label is the SINGLE SOURCE OF TRUTH.

Rules (IMMUTABLE):
- ADME HIGH_RISK → DEPRIORITIZED
- Docking FAIL → DEPRIORITIZED
- ADME FLAGGED OR Docking FLAG → FLAGGED
- ADME SAFE AND Docking PASS → PRIORITIZED

The LLM MUST echo decision_label verbatim. Any mismatch fails validation.
"""

from typing import Dict, Optional
from dataclasses import dataclass
from enum import Enum

from utils.logging import get_logger

logger = get_logger(__name__)


class DecisionLabel(str, Enum):
    """Deterministic decision labels - computed before LLM."""
    PRIORITIZED = "PRIORITIZED"
    FLAGGED = "FLAGGED"
    DEPRIORITIZED = "DEPRIORITIZED"


@dataclass
class DecisionLabelResult:
    """Result from decision label computation."""
    
    decision_label: str
    reason: str
    adme_status: str
    docking_status: str
    ml_status: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            "decision_label": self.decision_label,
            "reason": self.reason,
            "inputs": {
                "adme_status": self.adme_status,
                "docking_status": self.docking_status,
                "ml_status": self.ml_status,
            }
        }


def compute_decision_label(
    adme_status: str,
    docking_status: str,
    ml_status: Optional[str] = None,
) -> DecisionLabelResult:
    """
    Compute deterministic decision label.
    
    This is the SINGLE SOURCE OF TRUTH for decision labels.
    The LLM must echo this verbatim - any mismatch fails validation.
    
    Decision Rules (evaluated in order):
        1. ADME HIGH_RISK → DEPRIORITIZED
        2. Docking FAIL → DEPRIORITIZED
        3. ADME FLAGGED OR Docking FLAG → FLAGGED
        4. ADME SAFE AND Docking PASS → PRIORITIZED
    
    Args:
        adme_status: ADME/Tox status (SAFE, FLAGGED, HIGH_RISK)
        docking_status: Docking status (PASS, FLAG, FAIL, NOT_EVALUATED)
        ml_status: ML refinement status (APPLIED, NOT_APPLICABLE) - informational only
    
    Returns:
        DecisionLabelResult with deterministic label
    """
    # Normalize inputs
    adme_status = adme_status.upper() if adme_status else "SAFE"
    docking_status = docking_status.upper() if docking_status else "NOT_EVALUATED"
    
    # Rule 1: ADME HIGH_RISK → DEPRIORITIZED (highest priority)
    if adme_status == "HIGH_RISK":
        return DecisionLabelResult(
            decision_label=DecisionLabel.DEPRIORITIZED.value,
            reason="ADME/Tox status is HIGH_RISK - hard constraint failure",
            adme_status=adme_status,
            docking_status=docking_status,
            ml_status=ml_status,
        )
    
    # Rule 2: Docking FAIL → DEPRIORITIZED
    if docking_status == "FAIL":
        return DecisionLabelResult(
            decision_label=DecisionLabel.DEPRIORITIZED.value,
            reason="Docking status is FAIL - structural compatibility failure",
            adme_status=adme_status,
            docking_status=docking_status,
            ml_status=ml_status,
        )
    
    # Rule 3: ADME FLAGGED OR Docking FLAG → FLAGGED
    if adme_status == "FLAGGED" or docking_status == "FLAG":
        reasons = []
        if adme_status == "FLAGGED":
            reasons.append("ADME/Tox flagged")
        if docking_status == "FLAG":
            reasons.append("Docking flagged")
        
        return DecisionLabelResult(
            decision_label=DecisionLabel.FLAGGED.value,
            reason=" and ".join(reasons) + " - requires attention",
            adme_status=adme_status,
            docking_status=docking_status,
            ml_status=ml_status,
        )
    
    # Rule 4: ADME SAFE AND Docking PASS → PRIORITIZED
    if adme_status == "SAFE" and docking_status in ("PASS", "NOT_EVALUATED"):
        return DecisionLabelResult(
            decision_label=DecisionLabel.PRIORITIZED.value,
            reason="Satisfies chemical and developability constraints",
            adme_status=adme_status,
            docking_status=docking_status,
            ml_status=ml_status,
        )
    
    # Default fallback: FLAGGED (conservative)
    logger.warning(
        f"Unexpected status combination: adme={adme_status}, docking={docking_status}"
    )
    return DecisionLabelResult(
        decision_label=DecisionLabel.FLAGGED.value,
        reason="Status combination requires review",
        adme_status=adme_status,
        docking_status=docking_status,
        ml_status=ml_status,
    )


def validate_llm_echoes_label(
    computed_label: str,
    llm_output_label: str,
) -> bool:
    """
    Validate that LLM echoed the decision_label verbatim.
    
    This is a critical validation - ANY mismatch fails.
    
    Args:
        computed_label: The deterministic label computed before LLM
        llm_output_label: The label returned by the LLM
    
    Returns:
        True if labels match, False otherwise
        
    Raises:
        ValueError: If labels don't match (LLM attempted to override)
    """
    if computed_label != llm_output_label:
        raise ValueError(
            f"LLM decision_label mismatch! "
            f"Computed: {computed_label}, LLM returned: {llm_output_label}. "
            f"LLM is NOT allowed to change decision labels."
        )
    return True

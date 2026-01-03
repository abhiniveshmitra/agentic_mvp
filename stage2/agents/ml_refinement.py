"""
Stage-3 ML Refinement Agent (NON-LLM).

Purpose: Apply ML ensemble scoring as Level-4 refinement signal.

Authority Level: 4 (lowest) - Cannot override upstream decisions.
Governed by: ML Integration Specification

Rules (IMMUTABLE):
- NEVER override ADME/Tox HIGH_RISK
- NEVER override Docking FAIL
- NEVER re-rank Stage-1 order
- NEVER merge scores into composite
- All outputs prefixed with ml_
- ML is visible, NOT authoritative

This is a NON-LLM agent - pure ML inference.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from utils.logging import get_logger

logger = get_logger(__name__)


# ML Status values (standardized)
class MLStatus(str, Enum):
    APPLIED = "APPLIED"
    NOT_APPLICABLE = "NOT_APPLICABLE"


# Cross-target warning (mirrors docking)
ML_CROSS_TARGET_WARNING = (
    "ML affinity scores are target-specific and should not be compared across proteins."
)

# Uncertainty interpretation note
ML_UNCERTAINTY_NOTE = (
    "ML uncertainty reflects model disagreement and data coverage, not experimental error."
)


@dataclass
class MLRefinementExplanation:
    """Structured explanation for ML refinement output."""
    
    result: str  # Brief result statement
    raw_values: Dict = field(default_factory=dict)  # Ensemble scores
    rules_triggered: List[str] = field(default_factory=list)
    observations: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)
    summary: str = ""
    
    def to_dict(self) -> Dict:
        return {
            "result": self.result,
            "raw_values": self.raw_values,
            "rules_triggered": self.rules_triggered,
            "observations": self.observations,
            "limitations": self.limitations,
            "summary": self.summary,
        }


@dataclass
class MLRefinementResult:
    """Result from ML refinement scoring."""
    
    compound_id: str
    ml_status: str  # "APPLIED" or "NOT_APPLICABLE"
    ml_affinity_score: Optional[float] = None
    ml_uncertainty: Optional[float] = None
    ml_model_scores: Dict = field(default_factory=dict)
    reason: Optional[str] = None  # Only if NOT_APPLICABLE
    explanation: MLRefinementExplanation = None
    
    def __post_init__(self):
        if self.explanation is None:
            self.explanation = MLRefinementExplanation(result="Not evaluated")
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        result = {
            "ml_status": self.ml_status,
            "ml_affinity_score": self.ml_affinity_score,
            "ml_uncertainty": self.ml_uncertainty,
            "ml_model_scores": self.ml_model_scores,
            "explanation": self.explanation.to_dict(),
        }
        if self.reason:
            result["reason"] = self.reason
        return result


class MLRefinementAgent:
    """
    Stage-3 ML Refinement Agent.
    
    STRICT MODE: No mocks. Fails explicitly if ML scorer cannot be initialized.
    
    Applies ensemble ML scoring as a Level-4 refinement signal.
    Respects hard fail boundaries from ADME/Tox and Docking.
    
    Authority Hierarchy (from ML Integration Specification):
        Level 1: Chemistry + ADME Hard Constraints (TERMINAL)
        Level 2: Docking Structural Plausibility (FILTER)
        Level 3: Affinity Ranking (Stage-1 Score)
        Level 4: ML Refinement (THIS AGENT) ← Lowest authority
    """
    
    def __init__(self, require_scorer: bool = True):
        """
        Initialize ML Refinement Agent.
        
        Args:
            require_scorer: If True (default), fail if scorer cannot be initialized.
                            If False, scorer failures result in NOT_APPLICABLE status.
        
        Raises:
            RuntimeError: If require_scorer=True and scorer cannot be initialized
        """
        self.scorer = None
        self.scorer_error = None
        self.require_scorer = require_scorer
        self._init_scorer()
        logger.info(f"MLRefinementAgent initialized (scorer available: {self.scorer is not None})")
    
    def _init_scorer(self):
        """
        Initialize the ensemble scorer.
        
        STRICT MODE: Raises error if require_scorer=True and initialization fails.
        """
        try:
            from validation.scoring.ensemble_scorer import create_scorer
            self.scorer = create_scorer(use_ensemble=True)
            available = self.scorer.get_available_models()
            logger.info(f"Ensemble scorer initialized with models: {available}")
        except Exception as e:
            self.scorer_error = str(e)
            logger.error(f"Ensemble scorer initialization failed: {e}")
            
            if self.require_scorer:
                raise RuntimeError(
                    f"ML scorer initialization failed (STRICT MODE). "
                    f"Install required dependencies or disable ML refinement. Error: {e}"
                )
            else:
                logger.warning("Continuing without ML scorer - all results will be NOT_APPLICABLE")
    
    def score(
        self,
        smiles: str,
        compound_id: str,
        protein_sequence: str,
        adme_status: str,
        docking_status: str,
    ) -> MLRefinementResult:
        """
        Apply ML refinement scoring to a compound.
        
        Hard Fail Boundaries (ENFORCED):
            - IF adme_status == "HIGH_RISK" → SKIP ML
            - IF docking_status == "FAIL" → SKIP ML
        
        Args:
            smiles: Compound SMILES string
            compound_id: Compound identifier
            protein_sequence: Target protein sequence
            adme_status: ADME/Tox status ("SAFE", "FLAGGED", "HIGH_RISK")
            docking_status: Docking status ("PASS", "FLAG", "FAIL", "NOT_EVALUATED")
        
        Returns:
            MLRefinementResult with ml_ prefixed outputs
        """
        # ========== HARD FAIL BOUNDARY CHECK ==========
        
        # Check 1: ADME HIGH_RISK → Skip ML
        if adme_status == "HIGH_RISK":
            return self._create_not_applicable_result(
                compound_id=compound_id,
                reason="ADME/Tox status is HIGH_RISK - ML refinement not applicable",
                rule="ADME HIGH_RISK overrides ML (Level 1 > Level 4)",
            )
        
        # Check 2: Docking FAIL → Skip ML
        if docking_status == "FAIL":
            return self._create_not_applicable_result(
                compound_id=compound_id,
                reason="Docking status is FAIL - ML refinement not applicable",
                rule="Docking FAIL overrides ML (Level 2 > Level 4)",
            )
        
        # ========== APPLY ML SCORING ==========
        
        if self.scorer is None:
            reason = f"ML scorer not available: {self.scorer_error}" if self.scorer_error else "ML scorer not available"
            return self._create_not_applicable_result(
                compound_id=compound_id,
                reason=reason,
                rule="Scorer initialization failed - install required dependencies",
            )
        
        try:
            # Get ensemble prediction
            result = self.scorer.predict_affinity(smiles, protein_sequence)
            
            score = result.get("score", 0.0)
            uncertainty = result.get("uncertainty", 1.0)
            model_scores = result.get("model_scores", {})
            
            # Create explanation
            explanation = self._create_explanation(
                score=score,
                uncertainty=uncertainty,
                model_scores=model_scores,
                adme_status=adme_status,
                docking_status=docking_status,
            )
            
            return MLRefinementResult(
                compound_id=compound_id,
                ml_status=MLStatus.APPLIED.value,
                ml_affinity_score=round(score, 4),
                ml_uncertainty=round(uncertainty, 4),
                ml_model_scores=model_scores,
                explanation=explanation,
            )
            
        except Exception as e:
            logger.error(f"ML scoring failed for {compound_id}: {e}")
            return self._create_not_applicable_result(
                compound_id=compound_id,
                reason=f"ML scoring error: {str(e)[:100]}",
                rule="Scoring exception",
            )
    
    def score_batch(
        self,
        compounds: List[Dict],
        protein_sequence: str,
    ) -> List[MLRefinementResult]:
        """
        Apply ML refinement scoring to a batch of compounds.
        
        Each compound dict must have:
            - smiles: str
            - compound_id: str
            - adme_status: str (from ADME/Tox agent)
            - docking_status: str (from Docking agent)
        
        Args:
            compounds: List of compound dictionaries
            protein_sequence: Target protein sequence
        
        Returns:
            List of MLRefinementResult objects
        """
        results = []
        
        applied_count = 0
        skipped_count = 0
        
        for compound in compounds:
            result = self.score(
                smiles=compound.get("smiles", ""),
                compound_id=compound.get("compound_id", "unknown"),
                protein_sequence=protein_sequence,
                adme_status=compound.get("adme_status", "SAFE"),
                docking_status=compound.get("docking_status", "NOT_EVALUATED"),
            )
            results.append(result)
            
            if result.ml_status == MLStatus.APPLIED.value:
                applied_count += 1
            else:
                skipped_count += 1
        
        logger.info(
            f"ML refinement: {applied_count} APPLIED, {skipped_count} NOT_APPLICABLE"
        )
        
        return results
    
    def _create_not_applicable_result(
        self,
        compound_id: str,
        reason: str,
        rule: str,
    ) -> MLRefinementResult:
        """Create a NOT_APPLICABLE result for skipped compounds."""
        explanation = MLRefinementExplanation(
            result="ML refinement not applied",
            raw_values={},
            rules_triggered=[rule],
            observations=[reason],
            limitations=[
                "ML is a Level-4 signal and cannot override upstream FAIL status.",
                ML_UNCERTAINTY_NOTE,
            ],
            summary=reason,
        )
        
        return MLRefinementResult(
            compound_id=compound_id,
            ml_status=MLStatus.NOT_APPLICABLE.value,
            ml_affinity_score=None,
            ml_uncertainty=None,
            ml_model_scores={},
            reason=reason,
            explanation=explanation,
        )
    
    def _create_explanation(
        self,
        score: float,
        uncertainty: float,
        model_scores: Dict,
        adme_status: str,
        docking_status: str,
    ) -> MLRefinementExplanation:
        """Create structured explanation for ML scoring result."""
        
        # Determine confidence tier
        if uncertainty < 0.2:
            confidence = "HIGH"
        elif uncertainty < 0.4:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"
        
        # Build observations
        observations = [
            f"Ensemble score: {score:.2f} (pIC50 scale)",
            f"Model disagreement (uncertainty): {uncertainty:.3f}",
            f"Confidence tier: {confidence}",
        ]
        
        # Add model-specific observations
        for model_name, model_result in model_scores.items():
            if isinstance(model_result, dict):
                model_score = model_result.get("score", "N/A")
                observations.append(f"{model_name}: {model_score:.2f}" if isinstance(model_score, (int, float)) else f"{model_name}: {model_score}")
        
        # Context about upstream signals
        observations.append(f"Upstream ADME status: {adme_status}")
        observations.append(f"Upstream Docking status: {docking_status}")
        
        return MLRefinementExplanation(
            result=f"ML ensemble predicts pIC50 = {score:.2f} ± {uncertainty:.2f}",
            raw_values={
                "ml_affinity_score": score,
                "ml_uncertainty": uncertainty,
                "model_count": len(model_scores),
            },
            rules_triggered=[],  # No hard rules triggered if we got here
            observations=observations,
            limitations=[
                ML_UNCERTAINTY_NOTE,
                ML_CROSS_TARGET_WARNING,
                "ML scores supplement but do not replace Stage-1 ranking.",
                "This is a refinement signal, not a decision label.",
            ],
            summary=(
                f"ML ensemble predicts pIC50 = {score:.2f} with {confidence.lower()} confidence. "
                + ML_UNCERTAINTY_NOTE
            ),
        )


def create_agent() -> MLRefinementAgent:
    """Factory function to create MLRefinementAgent."""
    return MLRefinementAgent()

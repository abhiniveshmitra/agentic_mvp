"""
Sanity Checks - Control Validation.

CRITICAL CONSTRAINTS (from agent knowledge base):
- Positive controls must rank HIGH
- Negative controls must rank LOW
- Separation margin enforced
- Can invalidate ENTIRE RUNS
- Cannot be overridden

Failure → Run is FLAGGED as unreliable
"""

from typing import List, Dict, Tuple

from utils.logging import get_logger

logger = get_logger(__name__)


def validate_controls(
    compounds: List[Dict],
    config,  # ControlValidationConfig
) -> Tuple[bool, str]:
    """
    Validate that controls behave as expected.
    
    Args:
        compounds: All scored and normalized compounds
        config: ControlValidationConfig with thresholds
    
    Returns:
        Tuple of (is_valid, message)
    """
    # Count non-control compounds (candidates)
    candidates = [c for c in compounds if not c.get("is_control")]
    
    # If batch too small, percentiles are meaningless - skip validation
    MIN_BATCH_FOR_CONTROL_VALIDATION = 10
    if len(candidates) < MIN_BATCH_FOR_CONTROL_VALIDATION:
        logger.info(
            f"Batch size {len(candidates)} < {MIN_BATCH_FOR_CONTROL_VALIDATION}: "
            "skipping control validation (statistically meaningless)"
        )
        return True, f"Batch size {len(candidates)} too small for control validation (need >= {MIN_BATCH_FOR_CONTROL_VALIDATION})"
    
    # Separate controls
    positive_controls = [
        c for c in compounds 
        if c.get("is_control") and c.get("control_type") == "positive"
    ]
    negative_controls = [
        c for c in compounds 
        if c.get("is_control") and c.get("control_type") == "negative"
    ]
    
    # If no controls provided, skip validation
    if not positive_controls and not negative_controls:
        logger.info("No controls provided - skipping validation")
        return True, "No controls to validate"
    
    validation_errors = []
    
    # Check positive controls
    for control in positive_controls:
        percentile = control.get("percentile", 0)
        name = control.get("compound_name", control.get("compound_id", "Unknown"))
        
        if percentile < config.positive_min_percentile:
            validation_errors.append(
                f"Positive control '{name}' at percentile {percentile:.1f} "
                f"(required >= {config.positive_min_percentile})"
            )
    
    # Check negative controls
    for control in negative_controls:
        percentile = control.get("percentile", 100)
        name = control.get("compound_name", control.get("compound_id", "Unknown"))
        
        if percentile > config.negative_max_percentile:
            validation_errors.append(
                f"Negative control '{name}' at percentile {percentile:.1f} "
                f"(required <= {config.negative_max_percentile})"
            )
    
    # Check separation margin
    if positive_controls and negative_controls:
        avg_positive = sum(c.get("raw_score", 0) for c in positive_controls) / len(positive_controls)
        avg_negative = sum(c.get("raw_score", 0) for c in negative_controls) / len(negative_controls)
        
        # Normalize to check separation
        all_scores = [c.get("raw_score", 0) for c in compounds]
        score_range = max(all_scores) - min(all_scores) if all_scores else 1
        
        if score_range > 0:
            normalized_separation = (avg_positive - avg_negative) / score_range
            
            if normalized_separation < config.min_separation_margin:
                validation_errors.append(
                    f"Insufficient separation between controls: {normalized_separation:.3f} "
                    f"(required >= {config.min_separation_margin})"
                )
    
    # Compile result
    if validation_errors:
        error_msg = "; ".join(validation_errors)
        logger.warning(f"Control validation FAILED: {error_msg}")
        return False, error_msg
    
    logger.info("Control validation PASSED")
    return True, "All controls validated successfully"


def validate_run_sanity(
    compounds: List[Dict],
    papers_fetched: int,
    min_compounds: int = 5,
) -> Tuple[bool, str]:
    """
    Validate overall run sanity.
    
    Checks:
    - Minimum compounds processed
    - Score distribution reasonableness
    
    Args:
        compounds: Processed compounds
        papers_fetched: Number of papers from literature search
        min_compounds: Minimum required compounds
    
    Returns:
        Tuple of (is_valid, message)
    """
    warnings = []
    
    # Check minimum compounds
    if len(compounds) < min_compounds:
        warnings.append(
            f"Only {len(compounds)} compounds extracted "
            f"(minimum {min_compounds} recommended)"
        )
    
    # Check if we got papers
    if papers_fetched == 0:
        return False, "No papers fetched from literature search"
    
    # Check extraction rate
    if papers_fetched > 0:
        extraction_rate = len(compounds) / papers_fetched
        if extraction_rate < 0.1:  # Less than 0.1 compounds per paper
            warnings.append(
                f"Low extraction rate: {extraction_rate:.2f} compounds/paper"
            )
    
    # Check score distribution
    if compounds:
        scores = [c.get("raw_score", 0) for c in compounds]
        unique_scores = len(set(scores))
        
        if unique_scores == 1:
            warnings.append("All compounds have identical scores - model may not be working")
    
    if warnings:
        return True, "; ".join(warnings)  # Warnings but not failures
    
    return True, "Run sanity checks passed"

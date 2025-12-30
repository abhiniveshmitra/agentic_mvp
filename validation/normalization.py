"""
Score Normalization.

CRITICAL CONSTRAINTS (from agent knowledge base):
- Controls excluded from statistics
- Percentile rank: always
- Z-score: only if batch >= 30
- No interpretive language
"""

from typing import List, Dict, Tuple
import numpy as np

from utils.logging import get_logger

logger = get_logger(__name__)


def normalize_scores(
    compounds: List[Dict],
    config,  # NormalizationConfig
) -> List[Dict]:
    """
    Normalize scores and compute percentiles/z-scores.
    
    Args:
        compounds: List of scored compound dictionaries
        config: NormalizationConfig with settings
    
    Returns:
        Compounds with added percentile, z_score, and confidence_tier
    """
    # Separate controls and candidates
    candidates = [c for c in compounds if not c.get("is_control", False)]
    controls = [c for c in compounds if c.get("is_control", False)]
    
    if config.exclude_controls:
        # Calculate statistics from candidates only
        stats_pool = candidates
    else:
        stats_pool = compounds
    
    if not stats_pool:
        logger.warning("No compounds for normalization")
        return compounds
    
    # Extract scores
    scores = [c.get("raw_score", 0) for c in stats_pool]
    scores_array = np.array(scores)
    
    # Calculate statistics
    mean_score = np.mean(scores_array)
    std_score = np.std(scores_array) if len(scores_array) > 1 else 1.0
    
    # Normalize all compounds (including controls)
    for compound in compounds:
        raw_score = compound.get("raw_score", 0)
        
        # Percentile rank (always)
        percentile = _calculate_percentile(raw_score, scores_array)
        compound["percentile"] = round(percentile, 2)
        
        # Z-score (only if batch >= threshold)
        if len(stats_pool) >= config.min_batch_for_zscore and std_score > 0:
            z_score = (raw_score - mean_score) / std_score
            compound["z_score"] = round(z_score, 3)
        else:
            compound["z_score"] = None
        
        # Assign confidence tier
        compound["confidence_tier"] = _assign_confidence_tier(
            compound, percentile
        )
    
    logger.info(
        f"Normalized {len(candidates)} candidates, {len(controls)} controls. "
        f"Mean: {mean_score:.3f}, Std: {std_score:.3f}"
    )
    
    return compounds


def _calculate_percentile(value: float, all_values: np.ndarray) -> float:
    """
    Calculate the percentile rank of a value.
    
    Args:
        value: Score to rank
        all_values: Array of all scores
    
    Returns:
        Percentile (0-100)
    """
    if len(all_values) == 0:
        return 50.0
    
    # Count how many scores are below this value
    below = np.sum(all_values < value)
    equal = np.sum(all_values == value)
    
    # Percentile = (below + 0.5 * equal) / total * 100
    percentile = (below + 0.5 * equal) / len(all_values) * 100
    
    return percentile


def _assign_confidence_tier(compound: Dict, percentile: float) -> str:
    """
    Assign confidence tier based on percentile and source.
    
    Tiers:
        HIGH: percentile >= 90, PUBCHEM source
        MEDIUM: percentile >= 70
        LOW: percentile < 70 or LLM_INFERRED source
    """
    source = compound.get("source", "UNKNOWN")
    
    # LLM_INFERRED is always LOW confidence
    if source == "LLM_INFERRED":
        return "LOW"
    
    # Assign based on percentile
    if percentile >= 90:
        return "HIGH"
    elif percentile >= 70:
        return "MEDIUM"
    else:
        return "LOW"


def calculate_batch_statistics(compounds: List[Dict]) -> Dict:
    """
    Calculate aggregate statistics for a batch.
    
    Args:
        compounds: List of scored compounds
    
    Returns:
        Dictionary of statistics
    """
    scores = [c.get("raw_score", 0) for c in compounds if not c.get("is_control")]
    
    if not scores:
        return {"count": 0}
    
    scores_array = np.array(scores)
    
    return {
        "count": len(scores),
        "mean": round(float(np.mean(scores_array)), 3),
        "std": round(float(np.std(scores_array)), 3),
        "min": round(float(np.min(scores_array)), 3),
        "max": round(float(np.max(scores_array)), 3),
        "median": round(float(np.median(scores_array)), 3),
    }

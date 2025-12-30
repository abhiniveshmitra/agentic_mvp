"""
Control Loaders - Load controls for specific targets.

Provides a unified interface for getting positive and negative
controls for any target.
"""

from typing import List, Tuple

from validation.controls.positive import get_positive_smiles, get_positive_controls
from validation.controls.negative import get_negative_smiles, get_negative_controls


def load_controls_for_target(target: str) -> Tuple[List[str], List[str]]:
    """
    Load positive and negative controls for a target.
    
    Args:
        target: Target protein name
    
    Returns:
        Tuple of (positive_smiles_list, negative_smiles_list)
    """
    positive = get_positive_smiles(target)
    negative = get_negative_smiles()
    
    return positive, negative


def get_all_control_info(target: str) -> dict:
    """
    Get complete control information for a target.
    
    Args:
        target: Target protein name
    
    Returns:
        Dictionary with positive and negative control details
    """
    return {
        "target": target,
        "positive_controls": get_positive_controls(target),
        "negative_controls": get_negative_controls(),
    }

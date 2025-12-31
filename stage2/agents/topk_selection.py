"""
Top-K Selection Agent.

Purpose: Select a bounded, deterministic subset of candidates for deeper analysis.

Rules (IMMUTABLE):
- Select strictly by Stage-1 rank
- No chemistry, no ADME, no docking here
- No exceptions or overrides

Forbidden:
- Re-ranking
- Filtering based on any criteria other than rank
- Knowledge-based overrides
"""

from typing import List, Dict, Optional
from dataclasses import dataclass, field
from datetime import datetime

from utils.logging import get_logger

logger = get_logger(__name__)

# Default Top-K value
DEFAULT_TOP_K = 20


@dataclass
class TopKExplanation:
    """Explanation object for Top-K selection (follows unified schema)."""
    
    result: Dict = field(default_factory=dict)
    raw_values: Dict = field(default_factory=dict)
    rules_triggered: List[Dict] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)
    summary: str = ""
    
    def to_dict(self) -> Dict:
        return {
            "result": self.result,
            "raw_values": self.raw_values,
            "rules_triggered": self.rules_triggered,
            "limitations": self.limitations,
            "summary": self.summary,
        }


@dataclass
class TopKCandidate:
    """A candidate selected for Stage-2 analysis."""
    
    smiles: str
    compound_id: str
    compound_name: str
    stage1_rank: int
    stage1_score: float
    stage1_percentile: float
    stage1_confidence: str
    total_candidates: int
    explanation: TopKExplanation = field(default_factory=TopKExplanation)
    original_data: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "smiles": self.smiles,
            "compound_id": self.compound_id,
            "compound_name": self.compound_name,
            "stage1": {
                "rank": self.stage1_rank,
                "score": self.stage1_score,
                "percentile": self.stage1_percentile,
                "confidence": self.stage1_confidence,
            },
            "total_candidates": self.total_candidates,
            "explanation": self.explanation.to_dict(),
            "original_data": self.original_data,
        }


class TopKSelector:
    """
    Top-K Selection Agent.
    
    Selects a deterministic subset of candidates based on Stage-1 rank.
    Does NOT apply any scientific judgment - purely rank-based.
    """
    
    def __init__(self, k: int = DEFAULT_TOP_K):
        """
        Initialize Top-K Selector.
        
        Args:
            k: Number of top candidates to select (default 20)
        """
        self.k = k
        logger.info(f"TopKSelector initialized with k={k}")
    
    def select(
        self,
        candidates: List[Dict],
        rank_key: str = "rank",
        score_key: str = "score",
        percentile_key: str = "percentile",
    ) -> List[TopKCandidate]:
        """
        Select Top-K candidates by Stage-1 rank.
        
        Args:
            candidates: List of candidate dictionaries with ranking info
            rank_key: Key for rank in candidate dict
            score_key: Key for score in candidate dict
            percentile_key: Key for percentile in candidate dict
        
        Returns:
            List of TopKCandidate objects with full explanations
        """
        if not candidates:
            logger.warning("No candidates provided to TopKSelector")
            return []
        
        total_candidates = len(candidates)
        
        # Sort by rank (ascending - rank 1 is best)
        sorted_candidates = sorted(
            candidates,
            key=lambda c: c.get(rank_key, float('inf'))
        )
        
        # Select Top-K
        selected = sorted_candidates[:self.k]
        
        logger.info(
            f"Selected {len(selected)} of {total_candidates} candidates "
            f"(Top-{self.k} by Stage-1 rank)"
        )
        
        # Build TopKCandidate objects with explanations
        result = []
        for candidate in selected:
            rank = candidate.get(rank_key, 0)
            score = candidate.get(score_key, 0.0)
            percentile = candidate.get(percentile_key, 0.0)
            
            # Build explanation following unified schema
            explanation = self._build_explanation(
                rank=rank,
                total=total_candidates,
                score=score,
                percentile=percentile,
            )
            
            topk_candidate = TopKCandidate(
                smiles=candidate.get("smiles", ""),
                compound_id=candidate.get("compound_id", f"compound_{rank}"),
                compound_name=candidate.get("compound_name", "Unknown"),
                stage1_rank=rank,
                stage1_score=score,
                stage1_percentile=percentile,
                stage1_confidence=candidate.get("confidence_tier", "UNKNOWN"),
                total_candidates=total_candidates,
                explanation=explanation,
                original_data=candidate,
            )
            
            result.append(topk_candidate)
        
        return result
    
    def _build_explanation(
        self,
        rank: int,
        total: int,
        score: float,
        percentile: float,
    ) -> TopKExplanation:
        """
        Build explanation object for selection decision.
        
        Every selection must have a full explanation.
        """
        return TopKExplanation(
            result={
                "selected": True,
                "rank": rank,
                "total_candidates": total,
            },
            raw_values={
                "stage1_score": score,
                "stage1_percentile": percentile,
                "k_value": self.k,
            },
            rules_triggered=[
                {
                    "rule_id": "TOPK_RANK",
                    "condition": f"rank {rank} <= {self.k}",
                    "triggered": True,
                    "scientific_rationale": (
                        "Top-K selection prioritizes highest-confidence candidates "
                        "from Stage-1 for computational efficiency in deeper analysis"
                    ),
                    "practical_implication": (
                        "Reduces computational cost while preserving best candidates "
                        "based on predicted affinity"
                    ),
                }
            ],
            limitations=[
                "Lower-ranked molecules are not scientifically rejected",
                "This is resource prioritization, not quality judgment",
                "Stage-1 ranking may have uncertainty",
            ],
            summary=(
                f"Selected for Stage-2 analysis (rank #{rank} of {total}, "
                f"top {self.k} cutoff, percentile {percentile:.1f}%)"
            ),
        )
    
    def get_selection_summary(self, selected: List[TopKCandidate]) -> Dict:
        """
        Generate summary of selection for reporting.
        
        Args:
            selected: List of selected TopKCandidate objects
        
        Returns:
            Summary dictionary
        """
        if not selected:
            return {
                "status": "NO_CANDIDATES",
                "selected_count": 0,
                "k_value": self.k,
            }
        
        total = selected[0].total_candidates if selected else 0
        
        return {
            "status": "SUCCESS",
            "selected_count": len(selected),
            "total_candidates": total,
            "k_value": self.k,
            "selection_method": "stage1_rank",
            "rank_range": {
                "min": min(c.stage1_rank for c in selected),
                "max": max(c.stage1_rank for c in selected),
            },
            "percentile_range": {
                "min": round(min(c.stage1_percentile for c in selected), 1),
                "max": round(max(c.stage1_percentile for c in selected), 1),
            },
            "timestamp": datetime.now().isoformat(),
        }


def create_topk_selector(k: int = DEFAULT_TOP_K) -> TopKSelector:
    """Factory function to create TopKSelector."""
    return TopKSelector(k=k)

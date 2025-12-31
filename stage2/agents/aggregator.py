"""
Stage-2 Aggregation Agent (NON-LLM).

Purpose: Present ALL evidence side-by-side without blending.

Rules (IMMUTABLE):
- No cross-layer math
- No final "super score"
- Preserve interpretability
- No re-ranking
- No suppression

This is a NON-LLM agent - pure data aggregation.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime

from utils.logging import get_logger
from stage2.agents.topk_selection import TopKCandidate
from stage2.agents.docking import DockingResult
from stage2.agents.adme_tox_stage2 import ADMEToxStage2Result
from stage2.agents.patent_stub import PatentRiskResult

logger = get_logger(__name__)

# Version info
STAGE1_VERSION = "phase1-stable"
STAGE2_VERSION = "1.0.0"


@dataclass
class AggregatedCandidate:
    """Complete Stage-2 aggregated result for a candidate."""
    
    smiles: str
    compound_id: str
    compound_name: str
    stage1: Dict = field(default_factory=dict)
    stage2: Dict = field(default_factory=dict)
    narrative: str = ""
    provenance: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "smiles": self.smiles,
            "compound_id": self.compound_id,
            "compound_name": self.compound_name,
            "stage1": self.stage1,
            "stage2": self.stage2,
            "narrative": self.narrative,
            "provenance": self.provenance,
        }


class Stage2Aggregator:
    """
    Stage-2 Aggregation Agent.
    
    Combines Stage-1 and Stage-2 outputs without blending or re-ranking.
    Preserves full interpretability and provenance.
    """
    
    def __init__(self):
        """Initialize Stage-2 Aggregator."""
        logger.info("Stage2Aggregator initialized")
    
    def aggregate(
        self,
        topk_candidate: TopKCandidate,
        docking_result: Optional[DockingResult] = None,
        adme_tox_result: Optional[ADMEToxStage2Result] = None,
        patent_result: Optional[PatentRiskResult] = None,
    ) -> AggregatedCandidate:
        """
        Aggregate all Stage-2 results for a candidate.
        
        Args:
            topk_candidate: Top-K selection result
            docking_result: Docking result (optional)
            adme_tox_result: ADME/Tox result (optional)
            patent_result: Patent risk result (optional)
        
        Returns:
            AggregatedCandidate with all evidence
        """
        # Stage-1 data (immutable)
        stage1 = {
            "rank": topk_candidate.stage1_rank,
            "score": topk_candidate.stage1_score,
            "percentile": topk_candidate.stage1_percentile,
            "confidence": topk_candidate.stage1_confidence,
        }
        
        # Stage-2 data (evidence only, no blending)
        stage2 = {}
        
        if docking_result:
            stage2["docking"] = docking_result.to_dict()
        else:
            stage2["docking"] = {"status": "NOT_RUN"}
        
        if adme_tox_result:
            stage2["adme_tox"] = adme_tox_result.to_dict()
        else:
            stage2["adme_tox"] = {"status": "NOT_RUN"}
        
        if patent_result:
            stage2["patent"] = patent_result.to_dict()
        else:
            stage2["patent"] = {"status": "NOT_Run"}
        
        # Generate narrative (can be enhanced with Gemini later)
        narrative = self._generate_narrative(
            topk_candidate=topk_candidate,
            docking_result=docking_result,
            adme_tox_result=adme_tox_result,
            patent_result=patent_result,
        )
        
        # Provenance
        provenance = {
            "stage1_version": STAGE1_VERSION,
            "stage2_version": STAGE2_VERSION,
            "timestamp": datetime.now().isoformat(),
            "total_candidates_analyzed": topk_candidate.total_candidates,
        }
        
        return AggregatedCandidate(
            smiles=topk_candidate.smiles,
            compound_id=topk_candidate.compound_id,
            compound_name=topk_candidate.compound_name,
            stage1=stage1,
            stage2=stage2,
            narrative=narrative,
            provenance=provenance,
        )
    
    def aggregate_batch(
        self,
        topk_candidates: List[TopKCandidate],
        docking_results: Optional[List[DockingResult]] = None,
        adme_tox_results: Optional[List[ADMEToxStage2Result]] = None,
        patent_results: Optional[List[PatentRiskResult]] = None,
    ) -> List[AggregatedCandidate]:
        """
        Aggregate results for a batch of candidates.
        
        Args:
            topk_candidates: List of Top-K candidates
            docking_results: List of docking results (same order)
            adme_tox_results: List of ADME/Tox results (same order)
            patent_results: List of patent results (same order)
        
        Returns:
            List of AggregatedCandidate objects
        """
        results = []
        
        for i, topk in enumerate(topk_candidates):
            docking = docking_results[i] if docking_results and i < len(docking_results) else None
            adme = adme_tox_results[i] if adme_tox_results and i < len(adme_tox_results) else None
            patent = patent_results[i] if patent_results and i < len(patent_results) else None
            
            aggregated = self.aggregate(
                topk_candidate=topk,
                docking_result=docking,
                adme_tox_result=adme,
                patent_result=patent,
            )
            results.append(aggregated)
        
        logger.info(f"Aggregated {len(results)} candidates")
        
        return results
    
    def _generate_narrative(
        self,
        topk_candidate: TopKCandidate,
        docking_result: Optional[DockingResult],
        adme_tox_result: Optional[ADMEToxStage2Result],
        patent_result: Optional[PatentRiskResult],
    ) -> str:
        """
        Generate human-readable narrative summary.
        
        This is a simple rule-based narrative. Can be enhanced with
        Gemini API for more natural language in Phase 2.1.
        """
        parts = []
        
        # Stage-1 summary
        parts.append(
            f"This compound ranks #{topk_candidate.stage1_rank} "
            f"(top {100 - topk_candidate.stage1_percentile:.0f}%) "
            f"with predicted affinity score {topk_candidate.stage1_score:.2f}."
        )
        
        # Docking summary
        if docking_result:
            if docking_result.docking_status == "PASS":
                parts.append(
                    "Docking analysis shows the molecule fits the binding pocket "
                    "without major steric clashes."
                )
            elif docking_result.docking_status == "FLAG":
                parts.append(
                    "Docking analysis indicates marginal fit - "
                    "binding may require conformational adjustments."
                )
            elif docking_result.docking_status == "FAIL":
                parts.append(
                    f"Docking failed: {docking_result.error or 'structural incompatibility'}."
                )
            elif docking_result.docking_status == "NOT_EVALUATED":
                parts.append(
                    "Docking not performed (requires PDBQT tools not yet installed). "
                    "Structural plausibility not assessed."
                )
        
        # ADME/Tox summary
        if adme_tox_result:
            from stage2.agents.adme_tox_stage2 import Stage2RiskLevel
            
            if adme_tox_result.label == Stage2RiskLevel.SAFE:
                parts.append(
                    "No ADME/Tox flags detected - compound passes developability screening."
                )
            elif adme_tox_result.label == Stage2RiskLevel.FLAGGED:
                issues = adme_tox_result.explanation.summary
                parts.append(
                    f"ADME/Tox assessment: {issues}. These flags suggest elevated "
                    "developability risk but do not invalidate the compound."
                )
            elif adme_tox_result.label == Stage2RiskLevel.HIGH_RISK:
                parts.append(
                    "HIGH-RISK ADME/Tox profile detected. Careful assessment and "
                    "orthogonal validation recommended before proceeding."
                )
        
        # Patent summary
        if patent_result:
            if patent_result.patent_risk == "NOT_EVALUATED":
                parts.append(
                    "Patent analysis deferred in MVP. Professional IP review recommended "
                    "before development."
                )
        
        return " ".join(parts)
    
    def get_summary(self, aggregated: List[AggregatedCandidate]) -> Dict:
        """
        Generate summary statistics for aggregated results.
        
        Args:
            aggregated: List of aggregated candidates
        
        Returns:
            Summary dictionary
        """
        if not aggregated:
            return {"status": "NO_CANDIDATES"}
        
        # Count docking statuses
        docking_pass = sum(
            1 for a in aggregated 
            if a.stage2.get("docking", {}).get("docking_status") == "PASS"
        )
        docking_flag = sum(
            1 for a in aggregated 
            if a.stage2.get("docking", {}).get("docking_status") == "FLAG"
        )
        docking_fail = sum(
            1 for a in aggregated 
            if a.stage2.get("docking", {}).get("docking_status") == "FAIL"
        )
        docking_not_evaluated = sum(
            1 for a in aggregated 
            if a.stage2.get("docking", {}).get("docking_status") == "NOT_EVALUATED"
        )
        
        # Count ADME/Tox statuses
        adme_safe = sum(
            1 for a in aggregated 
            if a.stage2.get("adme_tox", {}).get("adme_tox_label") == "SAFE"
        )
        adme_flagged = sum(
            1 for a in aggregated 
            if a.stage2.get("adme_tox", {}).get("adme_tox_label") == "FLAGGED"
        )
        adme_high_risk = sum(
            1 for a in aggregated 
            if a.stage2.get("adme_tox", {}).get("adme_tox_label") == "HIGH_RISK"
        )
        
        return {
            "total_candidates": len(aggregated),
            "docking": {
                "pass": docking_pass,
                "flag": docking_flag,
                "fail": docking_fail,
                "not_evaluated": docking_not_evaluated,
            },
            "adme_tox": {
                "safe": adme_safe,
                "flagged": adme_flagged,
                "high_risk": adme_high_risk,
            },
            "provenance": {
                "stage1_version": STAGE1_VERSION,
                "stage2_version": STAGE2_VERSION,
                "timestamp": datetime.now().isoformat(),
            },
        }


def create_aggregator() -> Stage2Aggregator:
    """Factory function to create Stage2Aggregator."""
    return Stage2Aggregator()

"""
Stage-2 Agents Package.

Stage-2 agents produce flags and annotations, never modify Stage-1 scores.
All agents must use the unified explanation schema.
"""

from stage2.agents.topk_selection import TopKSelector
from stage2.agents.protein_prep import ProteinPreparator
from stage2.agents.docking import DockingAgent
from stage2.agents.adme_tox_stage2 import ADMEToxStage2
from stage2.agents.aggregator import Stage2Aggregator
from stage2.agents.patent_stub import PatentRiskStub

__all__ = [
    "TopKSelector",
    "ProteinPreparator",
    "DockingAgent",
    "ADMEToxStage2",
    "Stage2Aggregator",
    "PatentRiskStub",
]

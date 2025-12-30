"""
Provenance tracking for drug discovery pipeline.

Tracks the complete lineage of each compound:
- Source paper
- Extraction method
- Filter status
- Model versions used
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional, Dict, Any, List
from enum import Enum
import json


class ExtractionSource(Enum):
    """Source of compound SMILES."""
    PUBCHEM = "PUBCHEM"
    LLM_INFERRED = "LLM_INFERRED"
    MANUAL = "MANUAL"


class FilterStatus(Enum):
    """Status after chemistry filters."""
    PASSED = "PASSED"
    REJECTED = "REJECTED"
    NOT_PROCESSED = "NOT_PROCESSED"


@dataclass
class PaperSource:
    """Source paper information."""
    paper_id: str
    title: str
    abstract: Optional[str] = None
    pubmed_url: Optional[str] = None
    query_used: Optional[str] = None


@dataclass
class ExtractionInfo:
    """Information about compound extraction."""
    source: ExtractionSource
    compound_name: str
    pubchem_cid: Optional[int] = None
    confidence: Optional[float] = None
    extraction_timestamp: str = field(
        default_factory=lambda: datetime.utcnow().isoformat()
    )


@dataclass
class FilterInfo:
    """Information about chemistry filtering."""
    status: FilterStatus
    passed_filters: List[str] = field(default_factory=list)
    failed_filters: List[str] = field(default_factory=list)
    filter_details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScoringInfo:
    """Information about scoring."""
    model_name: str
    model_version: str
    raw_score: float
    uncertainty: Optional[float] = None
    scoring_timestamp: str = field(
        default_factory=lambda: datetime.utcnow().isoformat()
    )


@dataclass
class CompoundProvenance:
    """Complete provenance for a single compound."""
    compound_id: str
    smiles: str
    paper_source: Optional[PaperSource] = None
    extraction_info: Optional[ExtractionInfo] = None
    filter_info: Optional[FilterInfo] = None
    scoring_info: Optional[ScoringInfo] = None
    run_id: Optional[str] = None
    created_at: str = field(
        default_factory=lambda: datetime.utcnow().isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2, default=str)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CompoundProvenance":
        """Create from dictionary."""
        # Convert nested dataclasses
        if data.get("paper_source"):
            data["paper_source"] = PaperSource(**data["paper_source"])
        if data.get("extraction_info"):
            data["extraction_info"]["source"] = ExtractionSource(
                data["extraction_info"]["source"]
            )
            data["extraction_info"] = ExtractionInfo(**data["extraction_info"])
        if data.get("filter_info"):
            data["filter_info"]["status"] = FilterStatus(
                data["filter_info"]["status"]
            )
            data["filter_info"] = FilterInfo(**data["filter_info"])
        if data.get("scoring_info"):
            data["scoring_info"] = ScoringInfo(**data["scoring_info"])
        
        return cls(**data)


class ProvenanceTracker:
    """
    Track provenance for all compounds in a run.
    
    Thread-safe tracking of compound lineage with
    export capabilities.
    """
    
    def __init__(self, run_id: str):
        self.run_id = run_id
        self._compounds: Dict[str, CompoundProvenance] = {}
    
    def add_compound(
        self,
        compound_id: str,
        smiles: str,
        paper_source: Optional[PaperSource] = None,
        extraction_info: Optional[ExtractionInfo] = None
    ) -> CompoundProvenance:
        """Add a new compound to tracking."""
        provenance = CompoundProvenance(
            compound_id=compound_id,
            smiles=smiles,
            paper_source=paper_source,
            extraction_info=extraction_info,
            run_id=self.run_id
        )
        self._compounds[compound_id] = provenance
        return provenance
    
    def update_filter_info(
        self,
        compound_id: str,
        filter_info: FilterInfo
    ) -> None:
        """Update filter information for a compound."""
        if compound_id in self._compounds:
            self._compounds[compound_id].filter_info = filter_info
    
    def update_scoring_info(
        self,
        compound_id: str,
        scoring_info: ScoringInfo
    ) -> None:
        """Update scoring information for a compound."""
        if compound_id in self._compounds:
            self._compounds[compound_id].scoring_info = scoring_info
    
    def get_compound(self, compound_id: str) -> Optional[CompoundProvenance]:
        """Get provenance for a specific compound."""
        return self._compounds.get(compound_id)
    
    def get_all_compounds(self) -> List[CompoundProvenance]:
        """Get all tracked compounds."""
        return list(self._compounds.values())
    
    def export_json(self, filepath: str) -> None:
        """Export all provenance to JSON file."""
        data = {
            "run_id": self.run_id,
            "compounds": [c.to_dict() for c in self._compounds.values()]
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

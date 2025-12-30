"""
Run state management for the orchestrator.

Tracks the current state of a pipeline run with:
- Run metadata
- Step tracking
- Status management
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Dict, Any, List, Optional
import json
from pathlib import Path


class PipelineStep(Enum):
    """Pipeline execution steps."""
    INITIALIZATION = "initialization"
    DISCOVERY = "discovery"
    CHEMISTRY_FILTERS = "chemistry_filters"
    SCORING = "scoring"
    NORMALIZATION = "normalization"
    OUTPUT = "output"
    COMPLETED = "completed"
    FAILED = "failed"


class StepStatus(Enum):
    """Status of individual steps."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class StepResult:
    """Result of a pipeline step."""
    step: PipelineStep
    status: StepStatus
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error_message: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["step"] = self.step.value
        data["status"] = self.status.value
        return data


@dataclass
class RunState:
    """
    Complete state for a pipeline run.
    
    This is stateful per run but stateless per step.
    """
    run_id: str
    target_protein: str
    query: str
    created_at: str = field(
        default_factory=lambda: datetime.utcnow().isoformat()
    )
    current_step: PipelineStep = PipelineStep.INITIALIZATION
    step_results: List[StepResult] = field(default_factory=list)
    is_flagged: bool = False
    flag_reason: Optional[str] = None
    
    # Counters
    papers_fetched: int = 0
    compounds_extracted: int = 0
    compounds_passed_filters: int = 0
    compounds_rejected: int = 0
    compounds_scored: int = 0
    
    def start_step(self, step: PipelineStep) -> StepResult:
        """Mark a step as started."""
        self.current_step = step
        result = StepResult(
            step=step,
            status=StepStatus.RUNNING,
            started_at=datetime.utcnow().isoformat()
        )
        self.step_results.append(result)
        return result
    
    def complete_step(
        self,
        step: PipelineStep,
        metrics: Optional[Dict[str, Any]] = None
    ) -> None:
        """Mark a step as completed."""
        for result in self.step_results:
            if result.step == step and result.status == StepStatus.RUNNING:
                result.status = StepStatus.COMPLETED
                result.completed_at = datetime.utcnow().isoformat()
                if metrics:
                    result.metrics = metrics
                break
    
    def fail_step(self, step: PipelineStep, error: str) -> None:
        """Mark a step as failed."""
        for result in self.step_results:
            if result.step == step and result.status == StepStatus.RUNNING:
                result.status = StepStatus.FAILED
                result.completed_at = datetime.utcnow().isoformat()
                result.error_message = error
                break
        self.current_step = PipelineStep.FAILED
    
    def flag_run(self, reason: str) -> None:
        """Flag the run as unreliable."""
        self.is_flagged = True
        self.flag_reason = reason
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data["current_step"] = self.current_step.value
        data["step_results"] = [r.to_dict() for r in self.step_results]
        return data
    
    def save(self, output_dir: Path) -> Path:
        """Save state to file."""
        filepath = output_dir / f"run_state_{self.run_id}.json"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        return filepath
    
    @classmethod
    def load(cls, filepath: Path) -> "RunState":
        """Load state from file."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        data["current_step"] = PipelineStep(data["current_step"])
        data["step_results"] = [
            StepResult(
                step=PipelineStep(r["step"]),
                status=StepStatus(r["status"]),
                started_at=r.get("started_at"),
                completed_at=r.get("completed_at"),
                error_message=r.get("error_message"),
                metrics=r.get("metrics", {})
            )
            for r in data["step_results"]
        ]
        return cls(**data)

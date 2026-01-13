"""
Generative Chemistry Module.

Provides tools for molecular analog generation and optimization:
- Scaffold hopping
- R-group enumeration
- Bioisosteric replacement
- Property optimization
- Retrosynthesis planning
- Synthesis route validation
"""

from .scaffold_hopper import (
    ScaffoldHopper,
    GeneratedAnalog,
    generate_analogs_for_compound,
)
from .retrosynthesis import (
    RetrosynthesisPlanner,
    SynthesisRoute,
    BuildingBlock,
    plan_synthesis,
)
from .synthesis_validator import (
    ReactionValidator,
    LLMSynthesisValidator,
    ValidationResult,
    validate_synthesis_route,
)

__all__ = [
    "ScaffoldHopper",
    "GeneratedAnalog", 
    "generate_analogs_for_compound",
    "RetrosynthesisPlanner",
    "SynthesisRoute",
    "BuildingBlock",
    "plan_synthesis",
    "ReactionValidator",
    "LLMSynthesisValidator",
    "ValidationResult",
    "validate_synthesis_route",
]

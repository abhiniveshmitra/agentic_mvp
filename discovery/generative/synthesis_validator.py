"""
Retrosynthesis Validation Module.

Validates proposed synthesis routes using:
1. Reaction template matching
2. Chemical feasibility rules
3. LLM-assisted expert review
4. Known reaction database lookup
"""

from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field

from utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ValidationResult:
    """Result of synthesis route validation."""
    is_valid: bool
    confidence: float  # 0.0 - 1.0
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    reaction_validations: List[Dict] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "is_valid": self.is_valid,
            "confidence": self.confidence,
            "issues": self.issues,
            "warnings": self.warnings,
            "suggestions": self.suggestions,
        }


class ReactionValidator:
    """
    Validates individual chemical reactions.
    
    Uses reaction templates and chemical rules to assess feasibility.
    """
    
    # Known feasible reaction transformations
    VALID_REACTIONS = {
        "amide_coupling": {
            "reactants": ["carboxylic_acid", "amine"],
            "conditions": ["EDC", "DCC", "HATU", "HOBt"],
            "functional_groups": ["C(=O)O", "N"],
            "typical_yield": (60, 90),
        },
        "suzuki_coupling": {
            "reactants": ["aryl_halide", "boronic_acid"],
            "conditions": ["Pd(0)", "base"],
            "functional_groups": ["Br", "Cl", "I", "B(O"],
            "typical_yield": (70, 95),
        },
        "buchwald_hartwig": {
            "reactants": ["aryl_halide", "amine"],
            "conditions": ["Pd catalyst", "base", "ligand"],
            "functional_groups": ["Br", "Cl", "N"],
            "typical_yield": (50, 85),
        },
        "alkylation": {
            "reactants": ["nucleophile", "alkyl_halide"],
            "conditions": ["base"],
            "functional_groups": ["N", "O", "S", "Br", "Cl"],
            "typical_yield": (60, 90),
        },
        "ether_formation": {
            "reactants": ["alcohol", "alkyl_halide"],
            "conditions": ["base", "Williamson"],
            "functional_groups": ["O", "Br", "Cl"],
            "typical_yield": (50, 80),
        },
        "reduction": {
            "reactants": ["carbonyl", "reducing_agent"],
            "conditions": ["NaBH4", "LiAlH4", "H2/Pd"],
            "functional_groups": ["C=O", "C#N", "NO2"],
            "typical_yield": (70, 95),
        },
    }
    
    # Forbidden functional group combinations
    INCOMPATIBLE_GROUPS = [
        ("C(=O)Cl", "O"),   # Acid chloride + unprotected alcohol
        ("N", "C(=O)Cl"),   # Can react uncontrollably
        ("[N+]", "O-"),     # Charged groups stability
    ]
    
    def __init__(self):
        """Initialize reaction validator."""
        self._rdkit_available = self._check_rdkit()
    
    def _check_rdkit(self) -> bool:
        try:
            from rdkit import Chem
            return True
        except ImportError:
            return False
    
    def validate_reaction(
        self,
        reactant1: str,
        reactant2: str,
        reaction_type: str,
    ) -> Dict[str, Any]:
        """
        Validate a single reaction step.
        
        Args:
            reactant1: SMILES of first reactant
            reactant2: SMILES of second reactant
            reaction_type: Type of reaction proposed
            
        Returns:
            Validation result dictionary
        """
        result = {
            "valid": True,
            "confidence": 0.5,
            "issues": [],
            "warnings": [],
            "matched_template": None,
        }
        
        if not self._rdkit_available:
            result["warnings"].append("RDKit not available for validation")
            return result
        
        try:
            from rdkit import Chem
            
            mol1 = Chem.MolFromSmiles(reactant1)
            mol2 = Chem.MolFromSmiles(reactant2)
            
            if mol1 is None or mol2 is None:
                result["valid"] = False
                result["issues"].append("Invalid reactant SMILES")
                return result
            
            # Check for reactive functional groups
            fg1 = self._identify_functional_groups(mol1)
            fg2 = self._identify_functional_groups(mol2)
            
            # Check incompatible groups
            for group1, group2 in self.INCOMPATIBLE_GROUPS:
                if (group1 in fg1 and group2 in fg2) or (group2 in fg1 and group1 in fg2):
                    result["warnings"].append(
                        f"Potentially incompatible functional groups: {group1} + {group2}"
                    )
            
            # Try to match reaction template
            matched = self._match_reaction_template(fg1, fg2, reaction_type)
            if matched:
                result["matched_template"] = matched
                result["confidence"] = 0.8
            else:
                result["warnings"].append(
                    f"No validated template for {reaction_type}"
                )
                result["confidence"] = 0.4
            
            # Check molecular complexity
            n_atoms1 = mol1.GetNumAtoms()
            n_atoms2 = mol2.GetNumAtoms()
            if n_atoms1 > 50 or n_atoms2 > 50:
                result["warnings"].append(
                    "Large molecules - may have selectivity issues"
                )
            
        except Exception as e:
            result["warnings"].append(f"Validation error: {e}")
        
        return result
    
    def _identify_functional_groups(self, mol) -> List[str]:
        """Identify functional groups in a molecule."""
        from rdkit import Chem
        
        smiles = Chem.MolToSmiles(mol)
        groups = []
        
        # Simple pattern matching
        patterns = {
            "carboxylic_acid": "C(=O)O",
            "amine": "N",
            "alcohol": "CO",
            "ketone": "CC(=O)C",
            "aldehyde": "[CH]=O",
            "halide": ["Br", "Cl", "I", "F"],
            "ether": "COC",
            "aromatic": "c",
        }
        
        for name, pattern in patterns.items():
            if isinstance(pattern, list):
                if any(p in smiles for p in pattern):
                    groups.append(name)
            elif pattern in smiles:
                groups.append(name)
        
        return groups
    
    def _match_reaction_template(
        self,
        fg1: List[str],
        fg2: List[str],
        reaction_type: str,
    ) -> Optional[str]:
        """Match functional groups to known reaction templates."""
        reaction_type_lower = reaction_type.lower()
        
        for template_name, template in self.VALID_REACTIONS.items():
            # Check if reaction type matches template
            if template_name in reaction_type_lower or any(
                kw in reaction_type_lower for kw in template_name.split("_")
            ):
                # Check if functional groups match
                required = template["reactants"]
                all_groups = fg1 + fg2
                if any(r in all_groups for r in required):
                    return template_name
        
        return None


class LLMSynthesisValidator:
    """
    Uses LLM to validate synthesis routes.
    
    Provides expert-level feedback on:
    - Reaction feasibility
    - Selectivity concerns
    - Alternative routes
    - Practical considerations
    """
    
    VALIDATION_PROMPT = """You are an expert medicinal chemist reviewing a proposed synthesis route.

TARGET MOLECULE: {target_smiles}

PROPOSED ROUTE:
{route_description}

BUILDING BLOCKS:
{building_blocks}

Please evaluate this synthesis and provide:

1. **FEASIBILITY** (HIGH/MODERATE/LOW): Is this route chemically feasible?

2. **ISSUES**: List any major problems with this route (e.g., impossible reactions, selectivity issues)

3. **WARNINGS**: List potential difficulties that could be overcome

4. **SUGGESTIONS**: How could this route be improved?

5. **CONFIDENCE**: Rate your confidence in this assessment (0-100%)

Format your response as:
FEASIBILITY: [rating]
ISSUES: [comma-separated list or "None"]
WARNINGS: [comma-separated list or "None"]  
SUGGESTIONS: [comma-separated list or "None"]
CONFIDENCE: [0-100]%
"""
    
    def __init__(self):
        """Initialize LLM validator."""
        self._llm_available = self._check_llm()
    
    def _check_llm(self) -> bool:
        """Check if LLM is available."""
        try:
            from config.settings import LLM_PROVIDER
            return LLM_PROVIDER is not None
        except ImportError:
            return False
    
    def validate_route(
        self,
        target_smiles: str,
        building_blocks: List[str],
        reaction_steps: List[Dict],
    ) -> ValidationResult:
        """
        Validate a complete synthesis route using LLM.
        
        Args:
            target_smiles: Target molecule SMILES
            building_blocks: List of building block SMILES
            reaction_steps: List of reaction step dictionaries
            
        Returns:
            ValidationResult object
        """
        if not self._llm_available:
            return ValidationResult(
                is_valid=True,
                confidence=0.5,
                warnings=["LLM validation not available - using heuristics only"],
            )
        
        try:
            # Format route for LLM
            route_desc = self._format_route(reaction_steps)
            bb_desc = "\n".join(f"- {bb}" for bb in building_blocks[:5])
            
            prompt = self.VALIDATION_PROMPT.format(
                target_smiles=target_smiles,
                route_description=route_desc,
                building_blocks=bb_desc,
            )
            
            # Call LLM
            response = self._call_llm(prompt)
            
            # Parse response
            return self._parse_response(response)
            
        except Exception as e:
            logger.error(f"LLM validation failed: {e}")
            return ValidationResult(
                is_valid=True,
                confidence=0.3,
                warnings=[f"LLM validation error: {e}"],
            )
    
    def _format_route(self, steps: List[Dict]) -> str:
        """Format reaction steps for LLM prompt."""
        lines = []
        for i, step in enumerate(steps, 1):
            reactants = step.get("reactants", [])
            reaction = step.get("reaction_type", "Unknown")
            conditions = step.get("conditions", "Standard")
            lines.append(f"Step {i}: {' + '.join(reactants[:2])} → [{reaction}] → Product")
            lines.append(f"   Conditions: {conditions}")
        return "\n".join(lines)
    
    def _call_llm(self, prompt: str) -> str:
        """Call LLM API."""
        try:
            from config.settings import LLM_PROVIDER, GOOGLE_API_KEY, OPENAI_API_KEY
            
            if LLM_PROVIDER == "openai" and OPENAI_API_KEY:
                import openai
                client = openai.OpenAI(api_key=OPENAI_API_KEY)
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=500,
                )
                return response.choices[0].message.content
                
            elif LLM_PROVIDER == "gemini" and GOOGLE_API_KEY:
                import google.generativeai as genai
                genai.configure(api_key=GOOGLE_API_KEY)
                model = genai.GenerativeModel("gemini-1.5-flash")
                response = model.generate_content(prompt)
                return response.text
            
            return ""
            
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return ""
    
    def _parse_response(self, response: str) -> ValidationResult:
        """Parse LLM response into ValidationResult."""
        result = ValidationResult(
            is_valid=True,
            confidence=0.5,
        )
        
        if not response:
            return result
        
        try:
            lines = response.upper().split("\n")
            
            for line in lines:
                if "FEASIBILITY:" in line:
                    if "LOW" in line:
                        result.is_valid = False
                        result.confidence = 0.3
                    elif "MODERATE" in line:
                        result.confidence = 0.6
                    elif "HIGH" in line:
                        result.confidence = 0.85
                
                elif "ISSUES:" in line:
                    issues_text = line.split(":", 1)[1].strip()
                    if issues_text and issues_text != "NONE":
                        result.issues = [i.strip() for i in issues_text.split(",")]
                
                elif "WARNINGS:" in line:
                    warnings_text = line.split(":", 1)[1].strip()
                    if warnings_text and warnings_text != "NONE":
                        result.warnings = [w.strip() for w in warnings_text.split(",")]
                
                elif "SUGGESTIONS:" in line:
                    suggestions_text = line.split(":", 1)[1].strip()
                    if suggestions_text and suggestions_text != "NONE":
                        result.suggestions = [s.strip() for s in suggestions_text.split(",")]
                
                elif "CONFIDENCE:" in line:
                    import re
                    match = re.search(r"(\d+)", line)
                    if match:
                        result.confidence = int(match.group(1)) / 100.0
                        
        except Exception as e:
            logger.error(f"Failed to parse LLM response: {e}")
        
        return result


def validate_synthesis_route(
    target_smiles: str,
    building_blocks: List[str],
    reaction_steps: List[Dict],
    use_llm: bool = True,
) -> ValidationResult:
    """
    Comprehensive validation of a synthesis route.
    
    Args:
        target_smiles: Target molecule SMILES
        building_blocks: List of building block SMILES
        reaction_steps: List of reaction dictionaries
        use_llm: Whether to use LLM validation
        
    Returns:
        ValidationResult with all findings
    """
    # Rule-based validation
    rule_validator = ReactionValidator()
    
    all_issues = []
    all_warnings = []
    reaction_validations = []
    
    for step in reaction_steps:
        reactants = step.get("reactants", [])
        if len(reactants) >= 2:
            validation = rule_validator.validate_reaction(
                reactants[0],
                reactants[1],
                step.get("reaction_type", "Unknown")
            )
            reaction_validations.append(validation)
            all_issues.extend(validation.get("issues", []))
            all_warnings.extend(validation.get("warnings", []))
    
    # Base result from rule validation
    rule_valid = len(all_issues) == 0
    rule_confidence = 0.6 if rule_valid else 0.3
    
    # LLM validation (optional)
    if use_llm:
        llm_validator = LLMSynthesisValidator()
        llm_result = llm_validator.validate_route(
            target_smiles,
            building_blocks,
            reaction_steps,
        )
        
        # Combine results
        all_issues.extend(llm_result.issues)
        all_warnings.extend(llm_result.warnings)
        
        # Weighted confidence
        final_confidence = (rule_confidence * 0.4) + (llm_result.confidence * 0.6)
    else:
        final_confidence = rule_confidence
        llm_result = None
    
    return ValidationResult(
        is_valid=len(all_issues) == 0,
        confidence=final_confidence,
        issues=list(set(all_issues)),
        warnings=list(set(all_warnings)),
        suggestions=llm_result.suggestions if llm_result else [],
        reaction_validations=reaction_validations,
    )

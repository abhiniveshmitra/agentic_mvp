"""
Text Mining with Gemini AI.

Extracts compound mentions from literature abstracts.

CRITICAL CONSTRAINTS (from agent knowledge base):
- Allowed: compound names, experimental codes, context
- FORBIDDEN: SMILES generation, structural inference
- Output is UNTRUSTED HYPOTHESIS, not chemistry
"""

from typing import List, Dict, Optional
from dataclasses import dataclass
import json

from utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class CompoundMention:
    """A compound mention extracted from text."""
    compound_name: str
    context: str
    paper_id: str
    confidence: float
    mention_type: str  # "drug", "inhibitor", "compound", "experimental"
    
    def to_dict(self) -> Dict:
        return {
            "compound_name": self.compound_name,
            "context": self.context,
            "paper_id": self.paper_id,
            "confidence": self.confidence,
            "mention_type": self.mention_type,
        }


def extract_compounds(papers: List[Dict], target: str = None) -> List[Dict]:
    """
    Extract compound mentions from paper abstracts using Gemini.
    
    IMPORTANT: This function ONLY extracts names and context.
    It NEVER generates SMILES or infers structures.
    
    Args:
        papers: List of paper dictionaries with abstracts
        target: Optional target name for seeding known compounds
    
    Returns:
        List of compound mention dictionaries
    """
    import google.generativeai as genai
    from config.settings import GOOGLE_API_KEY
    
    if not GOOGLE_API_KEY:
        logger.error("GOOGLE_API_KEY not configured")
        return []
    
    # Configure Gemini
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel('gemini-2.0-flash')
    
    all_mentions = []
    
    # Add discovery seeding if target is specified
    if target:
        try:
            from discovery.query_templates import get_seeding_compounds
            seeds = get_seeding_compounds(target)
            all_mentions.extend(seeds)
            logger.info(f"Added {len(seeds)} seeding compounds for target {target}")
        except Exception as e:
            logger.warning(f"Could not add seeding compounds: {e}")
    
    for paper in papers:
        abstract = paper.get("abstract", "")
        paper_id = paper.get("paper_id", "")
        
        if not abstract:
            continue
        
        try:
            mentions = _extract_from_abstract(
                model=model,
                abstract=abstract,
                paper_id=paper_id,
            )
            all_mentions.extend(mentions)
            
        except Exception as e:
            logger.warning(f"Extraction failed for paper {paper_id}: {e}")
            continue
    
    logger.info(f"Extracted {len(all_mentions)} compound mentions from {len(papers)} papers")
    return all_mentions


def _extract_from_abstract(
    model,
    abstract: str,
    paper_id: str,
) -> List[Dict]:
    """
    Extract compound mentions from a single abstract.
    
    Uses Gemini with strict constraints to prevent hallucination.
    """
    # Improved prompt with examples for better extraction
    prompt = f"""You are a biomedical text mining expert. Extract ALL drug and compound names from this scientific abstract.

WHAT TO EXTRACT:
- FDA-approved drug names (e.g., "erlotinib", "gefitinib", "imatinib")
- Experimental compound names (e.g., "compound 1", "compound 12a")
- Chemical names (e.g., "4-aminoquinazoline derivative")
- Code names (e.g., "OSI-774", "ZD1839", "STI-571")
- Generic drug names and brand names

RULES:
1. Extract the compound NAME only - never generate chemical structures or SMILES
2. Include a brief context phrase where each compound is mentioned
3. Classify as: drug, inhibitor, compound, or experimental_code
4. Confidence 0.8-1.0 for clear drug names, 0.5-0.7 for less certain mentions

Return a JSON array. Example format:
[
  {{"compound_name": "erlotinib", "context": "erlotinib showed potent EGFR inhibition", "mention_type": "drug", "confidence": 0.95}},
  {{"compound_name": "compound 5a", "context": "compound 5a exhibited IC50 of 10nM", "mention_type": "experimental_code", "confidence": 0.8}}
]

If truly no drug/compound names are found, return: []

ABSTRACT:
{abstract}

JSON OUTPUT:"""

    try:
        response = model.generate_content(prompt)
        
        # Parse response
        text = response.text.strip()
        
        # Clean up markdown code blocks if present
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
        
        # Handle empty or invalid response
        if not text or text == "[]":
            return []
        
        # Parse JSON
        mentions_data = json.loads(text)
        
        # Ensure it's a list
        if not isinstance(mentions_data, list):
            mentions_data = [mentions_data]
        
        # Convert to CompoundMention objects
        mentions = []
        for m in mentions_data:
            if not isinstance(m, dict):
                continue
            compound_name = m.get("compound_name", "").strip()
            if compound_name and len(compound_name) > 1:  # Skip empty or single-char names
                mention = CompoundMention(
                    compound_name=compound_name,
                    context=m.get("context", ""),
                    paper_id=paper_id,
                    confidence=float(m.get("confidence", 0.5)),
                    mention_type=m.get("mention_type", "compound"),
                )
                mentions.append(mention.to_dict())
        
        logger.info(f"Paper {paper_id}: extracted {len(mentions)} compounds")
        return mentions
        
    except json.JSONDecodeError as e:
        logger.warning(f"JSON parse error for paper {paper_id}: {e}")
        return []
    except Exception as e:
        logger.warning(f"Gemini extraction error for paper {paper_id}: {e}")
        return []


def extract_target_context(
    abstract: str,
    target_name: str,
) -> Optional[Dict]:
    """
    Extract context about a specific target protein from abstract.
    
    Args:
        abstract: Paper abstract text
        target_name: Name of the target protein
    
    Returns:
        Context dictionary or None
    """
    import google.generativeai as genai
    from config.settings import GOOGLE_API_KEY
    
    if not GOOGLE_API_KEY:
        return None
    
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel('gemini-2.0-flash')
    
    prompt = f"""Analyze this abstract for information about {target_name}.

Extract:
1. What compounds interact with this target?
2. What type of interaction (inhibitor, activator, modulator)?
3. Any reported activity values?

Return JSON:
{{
  "target_mentioned": true/false,
  "compounds_mentioned": ["list of compound names"],
  "interaction_types": ["list of interaction types"],
  "activity_values": ["any IC50, Ki, EC50 values mentioned"]
}}

ABSTRACT:
{abstract}

JSON OUTPUT:"""

    try:
        response = model.generate_content(prompt)
        text = response.text.strip()
        
        # Clean markdown
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        
        return json.loads(text.strip())
        
    except Exception as e:
        logger.warning(f"Target context extraction failed: {e}")
        return None

"""
LLM-Based Structure Verification.

Uses LLM to verify that compound names/IDs match their SMILES structures.
This catches cases where:
1. Wrong SMILES got associated with a compound ID
2. Compound names don't match the displayed structure
3. Data corruption occurred in the pipeline

This is an additional trust layer on top of database verification.
"""

from typing import Dict, Optional, Tuple
import json

from utils.logging import get_logger

logger = get_logger(__name__)


def verify_structure_with_llm(
    compound_name: str,
    compound_id: str,
    smiles: str,
    molecular_formula: Optional[str] = None,
) -> Tuple[bool, str, Dict]:
    """
    Use LLM to verify that a compound's SMILES matches its name/ID.
    
    Args:
        compound_name: Name of the compound (e.g., "Phentolamine")
        compound_id: ChEMBL ID or other identifier
        smiles: SMILES string to verify
        molecular_formula: Optional formula for additional verification
    
    Returns:
        Tuple of (is_valid, message, details)
    """
    from config.settings import LLM_PROVIDER, LLM_CONFIG, OPENAI_API_KEY, GOOGLE_API_KEY
    
    if not LLM_PROVIDER:
        return True, "No LLM configured - skipping verification", {}
    
    # Build verification prompt
    prompt = f"""You are a pharmaceutical chemistry expert tasked with verifying compound data integrity.

TASK: Verify if the given SMILES structure matches the compound name/ID.

COMPOUND INFORMATION:
- Name: {compound_name}
- ID: {compound_id}
- SMILES: {smiles}
{f'- Molecular Formula: {molecular_formula}' if molecular_formula else ''}

INSTRUCTIONS:
1. Analyze the SMILES structure
2. Compare it with your knowledge of the named compound
3. Check if ChEMBL ID (if provided) typically refers to this structure
4. Identify any discrepancies

Return a JSON response:
{{
    "is_valid": true/false,
    "confidence": 0.0-1.0,
    "expected_structure": "brief description of what the compound should look like",
    "observed_structure": "brief description of what the SMILES represents",
    "discrepancy": "description of mismatch if any, or null",
    "correct_name": "if you can identify what the SMILES actually represents, provide it here"
}}

JSON RESPONSE:"""

    try:
        if LLM_PROVIDER == "openai":
            from openai import OpenAI
            client = OpenAI(api_key=OPENAI_API_KEY)
            config = LLM_CONFIG["openai"]
            
            response = client.chat.completions.create(
                model=config["model"],
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,  # Low temperature for factual response
                max_tokens=500,
            )
            text = response.choices[0].message.content.strip()
        else:
            import google.generativeai as genai
            genai.configure(api_key=GOOGLE_API_KEY)
            model = genai.GenerativeModel(LLM_CONFIG["gemini"]["model"])
            response = model.generate_content(prompt)
            text = response.text.strip()
        
        # Parse JSON response
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
        
        result = json.loads(text)
        
        is_valid = result.get("is_valid", True)
        confidence = result.get("confidence", 0.5)
        
        if is_valid:
            message = f"LLM verification passed (confidence: {confidence:.0%})"
        else:
            discrepancy = result.get("discrepancy", "Unknown mismatch")
            correct_name = result.get("correct_name", "")
            message = f"LLM detected mismatch: {discrepancy}"
            if correct_name:
                message += f" (SMILES appears to be: {correct_name})"
        
        logger.info(f"LLM verification for {compound_name}: valid={is_valid}")
        return is_valid, message, result
        
    except Exception as e:
        logger.warning(f"LLM verification failed: {e}")
        return True, f"LLM verification error: {e}", {}


def verify_compound_batch_with_llm(
    compounds: list,
    max_verify: int = 5,
) -> list:
    """
    Verify a batch of compounds using LLM.
    
    Args:
        compounds: List of compound dictionaries
        max_verify: Maximum compounds to verify (to limit API calls)
    
    Returns:
        Same compounds with llm_verified field added
    """
    verified_count = 0
    
    for compound in compounds[:max_verify]:
        compound_name = compound.get("compound_name", "")
        compound_id = compound.get("compound_id", "")
        smiles = compound.get("smiles", "")
        
        if not smiles:
            continue
        
        # Skip if name is just the ID (nothing to verify against)
        if compound_name == compound_id:
            compound["llm_verified"] = True
            compound["llm_verification_message"] = "Name equals ID - no semantic verification needed"
            continue
        
        is_valid, message, details = verify_structure_with_llm(
            compound_name=compound_name,
            compound_id=compound_id,
            smiles=smiles,
        )
        
        compound["llm_verified"] = is_valid
        compound["llm_verification_message"] = message
        compound["llm_verification_details"] = details
        
        if not is_valid:
            compound["display_warning"] = True
            logger.warning(f"LLM verification failed for {compound_name}: {message}")
        
        verified_count += 1
    
    logger.info(f"LLM verified {verified_count} compounds")
    return compounds


def get_correct_compound_info_from_llm(
    compound_name: str,
) -> Optional[Dict]:
    """
    Ask LLM for the correct structure information for a known compound.
    
    Args:
        compound_name: Name of the compound (e.g., "Phentolamine")
    
    Returns:
        Dictionary with compound information, or None
    """
    from config.settings import LLM_PROVIDER, LLM_CONFIG, OPENAI_API_KEY, GOOGLE_API_KEY
    
    if not LLM_PROVIDER:
        return None
    
    prompt = f"""You are a pharmaceutical chemistry expert.

For the drug/compound named "{compound_name}", provide:
1. The correct ChEMBL ID (if known)
2. The molecular formula
3. A brief structural description
4. Key functional groups

Return JSON:
{{
    "compound_name": "{compound_name}",
    "chembl_id": "CHEMBL...",
    "molecular_formula": "C...",
    "structure_description": "...",
    "key_features": ["..."]
}}

JSON RESPONSE:"""

    try:
        if LLM_PROVIDER == "openai":
            from openai import OpenAI
            client = OpenAI(api_key=OPENAI_API_KEY)
            config = LLM_CONFIG["openai"]
            
            response = client.chat.completions.create(
                model=config["model"],
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=300,
            )
            text = response.choices[0].message.content.strip()
        else:
            import google.generativeai as genai
            genai.configure(api_key=GOOGLE_API_KEY)
            model = genai.GenerativeModel(LLM_CONFIG["gemini"]["model"])
            response = model.generate_content(prompt)
            text = response.text.strip()
        
        # Parse JSON
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        
        return json.loads(text.strip())
        
    except Exception as e:
        logger.warning(f"Failed to get compound info from LLM: {e}")
        return None


# Test function
if __name__ == "__main__":
    # Test verification
    print("Testing LLM structure verification...")
    
    # This should FAIL - wrong SMILES for Phentolamine
    is_valid, msg, details = verify_structure_with_llm(
        compound_name="Phentolamine",
        compound_id="CHEMBL137635",
        smiles="CN(c1ccccc1)c1ncnc2ccc(N/N=N/Cc3ccccn3)cc12",
    )
    
    print(f"Is Valid: {is_valid}")
    print(f"Message: {msg}")
    print(f"Details: {details}")

"""
LLM-Based Compound Explainer.

Uses LLM (OpenAI GPT-4o-mini or Gemini) to generate human-readable
explanations of compound structures and acceptance/rejection reasoning.

Example output:
"CHEMBL137635 contains a quinazoline core attached to an azo linker (N=N),
which is a known PAINS pattern associated with pan-assay interference.
The compound was correctly rejected due to this structural alert."
"""

from typing import Dict, Optional, Tuple
import json

from utils.logging import get_logger

logger = get_logger(__name__)


def generate_compound_explanation(
    compound_id: str,
    smiles: str,
    decision: str,  # "ACCEPTED" or "REJECTED"
    rejection_reason: Optional[str] = None,
    binding_score: Optional[float] = None,
    percentile: Optional[float] = None,
) -> Dict:
    """
    Generate a human-readable explanation of a compound using LLM.
    
    Args:
        compound_id: ChEMBL ID or compound identifier
        smiles: SMILES string
        decision: "ACCEPTED" or "REJECTED"
        rejection_reason: If rejected, the reason (e.g., "PAINS_1")
        binding_score: Predicted binding score
        percentile: Score percentile
    
    Returns:
        Dictionary with:
        - structure_description: Plain English description of the molecule
        - key_features: List of important structural features
        - decision_rationale: Why accepted/rejected
        - drug_likeness_notes: Comments on drug-likeness
    """
    from config.settings import LLM_PROVIDER, LLM_CONFIG, OPENAI_API_KEY, GOOGLE_API_KEY
    
    if not LLM_PROVIDER:
        return _fallback_explanation(compound_id, smiles, decision, rejection_reason)
    
    # Build prompt
    prompt = f"""You are a medicinal chemistry expert. Analyze this compound and provide a clear explanation.

COMPOUND DATA:
- ID: {compound_id}
- SMILES: {smiles}
- Decision: {decision}
{f'- Rejection Reason: {rejection_reason}' if rejection_reason else ''}
{f'- Binding Score: {binding_score:.2f}' if binding_score else ''}
{f'- Percentile: Top {100-percentile:.0f}%' if percentile else ''}

INSTRUCTIONS:
1. Describe the molecular structure in plain English (core scaffold, key functional groups)
2. Identify key pharmacophore features (H-bond donors/acceptors, aromatic rings, etc.)
3. Explain WHY this compound was {decision}
4. Note any drug-likeness considerations

Return JSON:
{{
    "structure_description": "One sentence describing the core structure and key groups",
    "key_features": ["feature 1", "feature 2", "feature 3"],
    "decision_rationale": "Clear explanation of why accepted/rejected",
    "drug_likeness_notes": "Brief comment on drug-likeness properties"
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
                temperature=0.3,
                max_tokens=500,
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
        
        result = json.loads(text.strip())
        logger.info(f"Generated LLM explanation for {compound_id}")
        return result
        
    except Exception as e:
        logger.warning(f"LLM explanation failed: {e}")
        return _fallback_explanation(compound_id, smiles, decision, rejection_reason)


def generate_acceptance_explanation(
    compound_id: str,
    smiles: str,
    binding_score: float,
    percentile: float,
    confidence: float,
) -> str:
    """
    Generate a natural language explanation for why a compound was accepted.
    
    Returns a formatted string ready for display.
    """
    result = generate_compound_explanation(
        compound_id=compound_id,
        smiles=smiles,
        decision="ACCEPTED",
        binding_score=binding_score,
        percentile=percentile,
    )
    
    # Format as readable text
    explanation = f"""**🧬 Structure Analysis:**
{result.get('structure_description', 'Structure analysis unavailable')}

**🔬 Key Features:**
{chr(10).join(['• ' + f for f in result.get('key_features', [])])}

**✅ Why Accepted:**
{result.get('decision_rationale', 'Passed all chemistry filters and showed good binding prediction.')}

**💊 Drug-Likeness:**
{result.get('drug_likeness_notes', 'Meets standard drug-likeness criteria.')}
"""
    return explanation


def generate_rejection_explanation(
    compound_id: str,
    smiles: str,
    rejection_reason: str,
) -> str:
    """
    Generate a natural language explanation for why a compound was rejected.
    
    Returns a formatted string ready for display.
    """
    result = generate_compound_explanation(
        compound_id=compound_id,
        smiles=smiles,
        decision="REJECTED",
        rejection_reason=rejection_reason,
    )
    
    # Format as readable text
    explanation = f"""**🧬 Structure Analysis:**
{result.get('structure_description', 'Structure analysis unavailable')}

**🔬 Key Features:**
{chr(10).join(['• ' + f for f in result.get('key_features', [])])}

**❌ Why Rejected:**
{result.get('decision_rationale', f'Failed filter: {rejection_reason}')}

**💡 Drug-Likeness Notes:**
{result.get('drug_likeness_notes', 'Does not meet drug-likeness criteria.')}
"""
    return explanation


def _fallback_explanation(
    compound_id: str,
    smiles: str,
    decision: str,
    rejection_reason: Optional[str] = None,
) -> Dict:
    """
    Generate a basic explanation without LLM.
    Uses RDKit for basic analysis.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, rdMolDescriptors
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {
                "structure_description": "Could not parse molecular structure",
                "key_features": [],
                "decision_rationale": f"{decision} based on filter results",
                "drug_likeness_notes": "Analysis unavailable",
            }
        
        # Basic analysis
        num_atoms = mol.GetNumAtoms()
        num_rings = rdMolDescriptors.CalcNumRings(mol)
        num_aromatic = rdMolDescriptors.CalcNumAromaticRings(mol)
        hbd = rdMolDescriptors.CalcNumHBD(mol)
        hba = rdMolDescriptors.CalcNumHBA(mol)
        mw = Descriptors.MolWt(mol)
        
        # Build description
        if num_aromatic > 0:
            core = f"aromatic compound with {num_aromatic} aromatic ring(s)"
        else:
            core = f"non-aromatic compound with {num_rings} ring(s)"
        
        structure_desc = f"This is a {core} containing {num_atoms} atoms (MW: {mw:.0f} Da)."
        
        features = []
        if hbd > 0:
            features.append(f"{hbd} H-bond donor(s)")
        if hba > 0:
            features.append(f"{hba} H-bond acceptor(s)")
        if num_aromatic > 0:
            features.append(f"{num_aromatic} aromatic ring(s)")
        
        # Decision rationale
        if decision == "REJECTED":
            if rejection_reason and "PAINS" in rejection_reason:
                rationale = "Contains a PAINS (Pan-Assay Interference) pattern which may cause false positives in biochemical assays."
            elif rejection_reason and "MW" in rejection_reason:
                rationale = f"Molecular weight ({mw:.0f} Da) outside acceptable range for drug-like compounds."
            elif rejection_reason and "LOGP" in rejection_reason:
                rationale = "Lipophilicity (LogP) outside acceptable range, may have poor solubility or membrane permeability."
            else:
                rationale = f"Failed chemistry filter: {rejection_reason}"
        else:
            rationale = "Passed all chemistry filters and showed favorable binding prediction."
        
        return {
            "structure_description": structure_desc,
            "key_features": features,
            "decision_rationale": rationale,
            "drug_likeness_notes": f"MW: {mw:.0f} Da, HBD: {hbd}, HBA: {hba}",
        }
        
    except Exception as e:
        logger.warning(f"Fallback explanation failed: {e}")
        return {
            "structure_description": "Structure analysis unavailable",
            "key_features": [],
            "decision_rationale": f"{decision}",
            "drug_likeness_notes": "",
        }


def generate_admet_explanation(
    compound_id: str,
    admet_profile: Dict,
) -> str:
    """
    Generate a human-readable summary of the ADMET profile.
    """
    from config.settings import LLM_PROVIDER, LLM_CONFIG, OPENAI_API_KEY, GOOGLE_API_KEY
    
    if not LLM_PROVIDER:
        return "**ADMET Summary:** No LLM available for detailed analysis."
    
    # Extract key metrics
    abs_data = admet_profile.get("absorption", {})
    dist_data = admet_profile.get("distribution", {})
    met_data = admet_profile.get("metabolism", {})
    tox_data = admet_profile.get("toxicity", {})
    
    prompt = f"""You are a DMPK scientist. Summarize this ADMET profile for a drug candidate.

COMPOUND: {compound_id}

DATA:
- Absorption: GI={abs_data.get('gi_absorption', 'Unknown')}, Bioavailability Score={abs_data.get('bioavailability_score', 0):.2f}
- Distribution: BBB Permeant={dist_data.get('bbb_permeant', 'Unknown')}, LogBB={dist_data.get('log_bb', 0):.2f}
- Metabolism (CYP Inhibition): {', '.join([k for k, v in met_data.items() if v == 'Inhibitor']) or 'None predicted'}
- Toxicity Alerts: {', '.join([k for k, v in tox_data.items() if v == 'High Risk']) or 'None predicted'}
- Synthetic Accessibility: {admet_profile.get('synthetic_accessibility', 0):.1f}/10 (1=Easy)

INSTRUCTIONS:
Provide a concise 3-4 sentence summary highlighted key strengths and risks. 
Focus on oral bioavailability, CNS penetration, and safety.

Summary:"""

    try:
        if LLM_PROVIDER == "openai":
            from openai import OpenAI
            client = OpenAI(api_key=OPENAI_API_KEY)
            response = client.chat.completions.create(
                model=LLM_CONFIG["openai"]["model"],
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=200,
            )
            return response.choices[0].message.content.strip()
        else:
            import google.generativeai as genai
            genai.configure(api_key=GOOGLE_API_KEY)
            model = genai.GenerativeModel(LLM_CONFIG["gemini"]["model"])
            response = model.generate_content(prompt)
            return response.text.strip()
    except Exception as e:
        logger.warning(f"ADMET explanation failed: {e}")
        return "ADMET summary unavailable."


def generate_ddi_explanation(
    compound_id: str,
    ddi_report: Dict,
) -> str:
    """
    Generate a human-readable summary of Drug-Drug Interaction risks.
    """
    from config.settings import LLM_PROVIDER, LLM_CONFIG, OPENAI_API_KEY, GOOGLE_API_KEY
    
    if not LLM_PROVIDER:
        return "**DDI Summary:** No LLM available for detailed analysis."
    
    interactions = ddi_report.get("interactions", [])
    severe = [i for i in interactions if i["severity"] in ["Contraindicated", "Major"]]
    
    if not severe:
        return "✅ **Low Interaction Risk:** No major drug-drug interactions predicted with common co-medications."
        
    prompt = f"""You are a clinical pharmacist. Summarize the Drug-Drug Interaction (DDI) risks.

COMPOUND: {compound_id}

INTERACTIONS DETECTED:
{chr(10).join([f"- {i['drug']} ({i['severity']}): {i['mechanism']} -> {i['effect']}" for i in severe[:5]])}

INSTRUCTIONS:
Provide a concise warning about the main risks, especially for diabetic/cardiac patients.
Suggest what to avoid.

Summary:"""

    try:
        if LLM_PROVIDER == "openai":
            from openai import OpenAI
            client = OpenAI(api_key=OPENAI_API_KEY)
            response = client.chat.completions.create(
                model=LLM_CONFIG["openai"]["model"],
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=200,
            )
            return response.choices[0].message.content.strip()
        else:
            import google.generativeai as genai
            genai.configure(api_key=GOOGLE_API_KEY)
            model = genai.GenerativeModel(LLM_CONFIG["gemini"]["model"])
            response = model.generate_content(prompt)
            return response.text.strip()
    except Exception as e:
        logger.warning(f"DDI explanation failed: {e}")
        return "DDI summary unavailable."


def generate_pk_explanation(
    compound_id: str,
    smiles: str,
    pk_profile: Dict,
) -> str:
    """
    Generate a human-readable pharmacokinetics explanation.
    
    Args:
        compound_id: Compound identifier
        smiles: SMILES string
        pk_profile: Dictionary with PK parameters (from ADMETProfile.to_dict()["pharmacokinetics"])
    
    Returns:
        Formatted string with PK interpretation and dosing recommendations
    """
    from config.settings import LLM_PROVIDER, LLM_CONFIG, OPENAI_API_KEY, GOOGLE_API_KEY
    
    if not LLM_PROVIDER:
        return _fallback_pk_explanation(compound_id, pk_profile)
    
    # Extract PK data
    half_life = pk_profile.get("half_life_hours", "Unknown")
    tmax = pk_profile.get("tmax_hours", "Unknown")
    auc = pk_profile.get("auc_relative", "Unknown")
    cmax = pk_profile.get("cmax_relative", "Unknown")
    f_oral = pk_profile.get("oral_bioavailability", 0)
    dosing = pk_profile.get("dosing_frequency", "Unknown")
    
    prompt = f"""You are a clinical pharmacologist. Provide a clear pharmacokinetics interpretation for this drug candidate.

COMPOUND: {compound_id}

PHARMACOKINETICS DATA:
- Estimated Half-life: {half_life} hours
- Time to Peak (Tmax): {tmax} hours
- Relative AUC: {auc}
- Relative Cmax: {cmax}
- Oral Bioavailability: {f_oral*100 if f_oral else 0:.0f}%
- Suggested Dosing: {dosing}

INSTRUCTIONS:
Provide a 3-4 sentence clinical interpretation that:
1. Explains what the half-life means for dosing (e.g., "suitable for once-daily dosing")
2. Comments on expected absorption profile based on Tmax
3. Notes any implications for patient convenience or compliance
4. Highlights any potential PK concerns

Be concise and clinical. No jargon without explanation.

Interpretation:"""

    try:
        if LLM_PROVIDER == "openai":
            from openai import OpenAI
            client = OpenAI(api_key=OPENAI_API_KEY)
            response = client.chat.completions.create(
                model=LLM_CONFIG["openai"]["model"],
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=250,
            )
            return response.choices[0].message.content.strip()
        else:
            import google.generativeai as genai
            genai.configure(api_key=GOOGLE_API_KEY)
            model = genai.GenerativeModel(LLM_CONFIG["gemini"]["model"])
            response = model.generate_content(prompt)
            return response.text.strip()
    except Exception as e:
        logger.warning(f"PK explanation failed: {e}")
        return _fallback_pk_explanation(compound_id, pk_profile)


def _fallback_pk_explanation(compound_id: str, pk_profile: Dict) -> str:
    """Generate basic PK explanation without LLM."""
    half_life = pk_profile.get("half_life_hours")
    dosing = pk_profile.get("dosing_frequency", "Unknown")
    f_oral = pk_profile.get("oral_bioavailability", 0)
    
    lines = [f"**📊 Pharmacokinetics Summary for {compound_id}:**"]
    
    if half_life:
        if half_life >= 12:
            lines.append(f"• Half-life of ~{half_life:.0f} hours supports once-daily dosing, improving patient compliance.")
        elif half_life >= 6:
            lines.append(f"• Half-life of ~{half_life:.0f} hours suggests twice-daily dosing may be required.")
        else:
            lines.append(f"• Short half-life (~{half_life:.0f} hours) may require multiple daily doses or extended-release formulation.")
    
    if f_oral:
        if f_oral >= 0.7:
            lines.append(f"• Good predicted oral bioavailability ({f_oral*100:.0f}%) suggests reliable absorption.")
        elif f_oral >= 0.4:
            lines.append(f"• Moderate bioavailability ({f_oral*100:.0f}%) may affect dose optimization.")
        else:
            lines.append(f"• Limited bioavailability ({f_oral*100:.0f}%) may require higher doses or alternative routes.")
    
    lines.append(f"• Suggested dosing: {dosing}")
    
    return "\n".join(lines)

# Test
if __name__ == "__main__":
    # Test with CHEMBL137635 (the PAINS compound)
    result = generate_compound_explanation(
        compound_id="CHEMBL137635",
        smiles="CN(c1ccccc1)c1ncnc2ccc(N/N=N/Cc3ccccn3)cc12",
        decision="REJECTED",
        rejection_reason="PAINS_1",
    )
    
    print("Structure:", result.get("structure_description"))
    print("Features:", result.get("key_features"))
    print("Rationale:", result.get("decision_rationale"))

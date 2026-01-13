"""
Off-Target Prediction Module.

Predicts potential off-target proteins that a compound may bind to,
beyond its intended primary target.

This addresses the PhD's key concern: "We know it inhibits HDAC but 
might be inhibiting other molecules we don't know about."

Uses SwissTargetPrediction API for target fishing.
"""

from typing import Dict, List, Optional, Tuple
import requests
import time
from dataclasses import dataclass, field

from utils.logging import get_logger

logger = get_logger(__name__)


# Known safety-critical targets that warrant alerts
SAFETY_CRITICAL_TARGETS = {
    "HERG": {"severity": "critical", "effect": "Cardiac arrhythmia risk (QT prolongation)"},
    "KCNH2": {"severity": "critical", "effect": "Cardiac arrhythmia risk (QT prolongation)"},
    "DRD2": {"severity": "warning", "effect": "Potential CNS side effects"},
    "HTR2A": {"severity": "warning", "effect": "Potential CNS/psychiatric effects"},
    "HTR2B": {"severity": "critical", "effect": "Cardiac valvulopathy risk"},
    "OPRM1": {"severity": "warning", "effect": "Opioid receptor - addiction potential"},
    "NR3C1": {"severity": "info", "effect": "Glucocorticoid effects"},
    "CYP3A4": {"severity": "warning", "effect": "Drug-drug interaction potential"},
    "CYP2D6": {"severity": "warning", "effect": "Drug-drug interaction potential"},
    "ABCB1": {"severity": "info", "effect": "P-glycoprotein - affects distribution"},
}


@dataclass
class TargetPrediction:
    """Represents a predicted protein target."""
    target_name: str
    gene_name: str
    uniprot_id: str
    probability: float  # 0-1
    known_activity: Optional[str] = None
    target_class: Optional[str] = None
    safety_alert: Optional[Dict] = None


@dataclass
class OffTargetReport:
    """Complete off-target analysis report."""
    compound_id: str
    smiles: str
    primary_target: Optional[str]
    off_targets: List[TargetPrediction] = field(default_factory=list)
    safety_alerts: List[Dict] = field(default_factory=list)
    analysis_timestamp: Optional[str] = None
    confidence_note: str = ""
    
    def to_dict(self) -> Dict:
        return {
            "compound_id": self.compound_id,
            "smiles": self.smiles,
            "primary_target": self.primary_target,
            "off_targets": [
                {
                    "target_name": t.target_name,
                    "gene_name": t.gene_name,
                    "uniprot_id": t.uniprot_id,
                    "probability": t.probability,
                    "target_class": t.target_class,
                    "safety_alert": t.safety_alert,
                }
                for t in self.off_targets
            ],
            "safety_alerts": self.safety_alerts,
            "confidence_note": self.confidence_note,
        }


def predict_off_targets_llm(
    smiles: str,
    primary_target: Optional[str] = None,
) -> List[TargetPrediction]:
    """
    Predict off-targets using LLM (GPT-4o-mini) chemistry knowledge.
    
    This approach uses the LLM's understanding of:
    - Molecular structure-activity relationships
    - Known pharmacophore patterns
    - Target family similarities
    
    Args:
        smiles: Compound SMILES string
        primary_target: Known primary target (to exclude)
    
    Returns:
        List of TargetPrediction objects
    """
    from config.settings import LLM_PROVIDER, LLM_CONFIG, OPENAI_API_KEY, GOOGLE_API_KEY
    
    if not LLM_PROVIDER:
        logger.warning("No LLM provider configured for off-target prediction")
        return []
    
    prompt = f"""You are an expert medicinal chemist and pharmacologist. Analyze this molecule's structure and predict potential protein targets it might bind to.

SMILES: {smiles}
{f'Primary Target: {primary_target} (exclude this from predictions)' if primary_target else ''}

Based on the molecular structure, identify:
1. Key pharmacophore features (aromatic systems, H-bond donors/acceptors, charged groups)
2. Structural motifs associated with specific target classes
3. Potential off-target proteins this molecule might interact with

Return a JSON object with this exact format:
{{
    "structure_analysis": "Brief description of key structural features",
    "predictions": [
        {{"target_name": "Protein Name", "gene_name": "GENE", "probability": 0.0-1.0, "target_class": "Class", "rationale": "Why this target"}},
        ...
    ]
}}

Consider these common off-target families:
- Kinases (if contains ATP-mimetic motifs)
- GPCRs (aminergic if contains basic amine + aromatic)
- Ion channels (hERG if lipophilic + basic nitrogen)
- Nuclear receptors
- Proteases
- Transporters (P-gp, OCT)

Provide 5-10 most likely off-targets with probability estimates. Be scientifically rigorous.

JSON RESPONSE:"""

    try:
        if LLM_PROVIDER == "openai":
            from openai import OpenAI
            client = OpenAI(api_key=OPENAI_API_KEY)
            
            response = client.chat.completions.create(
                model=LLM_CONFIG["openai"]["model"],
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1000,
            )
            text = response.choices[0].message.content.strip()
        else:
            import google.generativeai as genai
            genai.configure(api_key=GOOGLE_API_KEY)
            model = genai.GenerativeModel(LLM_CONFIG["gemini"]["model"])
            response = model.generate_content(prompt)
            text = response.text.strip()
        
        # Parse JSON response
        import json
        
        # Clean up response
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        
        result = json.loads(text.strip())
        
        predictions = []
        for pred_data in result.get("predictions", []):
            target_name = pred_data.get("target_name", "Unknown")
            gene_name = pred_data.get("gene_name", "")
            
            # Skip primary target
            if primary_target and primary_target.upper() in target_name.upper():
                continue
            
            pred = TargetPrediction(
                target_name=target_name,
                gene_name=gene_name,
                uniprot_id="",
                probability=float(pred_data.get("probability", 0.5)),
                target_class=pred_data.get("target_class", ""),
                known_activity=pred_data.get("rationale", ""),
            )
            
            # Check for safety-critical targets
            gene_upper = gene_name.upper()
            if gene_upper in SAFETY_CRITICAL_TARGETS:
                pred.safety_alert = SAFETY_CRITICAL_TARGETS[gene_upper]
            
            predictions.append(pred)
        
        predictions.sort(key=lambda x: x.probability, reverse=True)
        logger.info(f"LLM predicted {len(predictions)} off-targets")
        
        return predictions
        
    except Exception as e:
        logger.error(f"LLM off-target prediction failed: {e}")
        return []


def predict_off_targets_similarity(
    smiles: str,
    primary_target: Optional[str] = None,
) -> List[TargetPrediction]:
    """
    Predict off-targets using ChEMBL similarity search.
    
    Finds compounds similar to the query and returns their known targets.
    This is a fallback when SwissTargetPrediction is unavailable.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return []
        
        # Generate fingerprint
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        fp_hex = fp.ToBitString()
        
        # Query ChEMBL for similar molecules
        chembl_url = "https://www.ebi.ac.uk/chembl/api/data/similarity"
        params = {
            "smiles": smiles,
            "similarity": 70,  # 70% similarity threshold
            "limit": 50,
        }
        
        response = requests.get(chembl_url, params=params, timeout=30)
        
        if response.status_code != 200:
            return []
        
        data = response.json()
        
        # Collect targets from similar molecules
        target_counts = {}
        for mol_data in data.get("molecules", []):
            mol_chembl_id = mol_data.get("molecule_chembl_id")
            
            # Get activities for this molecule
            activities_url = f"https://www.ebi.ac.uk/chembl/api/data/activity.json"
            act_response = requests.get(
                activities_url,
                params={"molecule_chembl_id": mol_chembl_id, "limit": 10},
                timeout=10,
            )
            
            if act_response.status_code == 200:
                activities = act_response.json().get("activities", [])
                for act in activities:
                    target_name = act.get("target_pref_name", "")
                    if target_name:
                        if target_name not in target_counts:
                            target_counts[target_name] = {
                                "count": 0,
                                "chembl_id": act.get("target_chembl_id", ""),
                            }
                        target_counts[target_name]["count"] += 1
            
            time.sleep(0.1)  # Rate limiting
        
        # Convert to predictions
        total = sum(t["count"] for t in target_counts.values())
        predictions = []
        
        for target_name, info in target_counts.items():
            # Skip primary target
            if primary_target and primary_target.upper() in target_name.upper():
                continue
            
            prob = info["count"] / total if total > 0 else 0
            pred = TargetPrediction(
                target_name=target_name,
                gene_name="",
                uniprot_id="",
                probability=prob,
                target_class="",
            )
            predictions.append(pred)
        
        predictions.sort(key=lambda x: x.probability, reverse=True)
        return predictions[:20]  # Top 20
        
    except Exception as e:
        logger.error(f"Similarity-based prediction error: {e}")
        return []


def analyze_off_targets(
    compound_id: str,
    smiles: str,
    primary_target: Optional[str] = None,
) -> OffTargetReport:
    """
    Complete off-target analysis for a compound.
    
    Args:
        compound_id: Compound identifier
        smiles: SMILES string
        primary_target: Known primary target (will be excluded from off-targets)
    
    Returns:
        OffTargetReport with all predictions and safety alerts
    """
    logger.info(f"Analyzing off-targets for {compound_id}")
    
    # Try LLM-based prediction first (uses GPT-4o-mini chemistry knowledge)
    predictions = predict_off_targets_llm(smiles, primary_target)
    
    # Fallback to similarity-based if LLM fails
    if not predictions:
        logger.info("Falling back to ChEMBL similarity-based prediction")
        predictions = predict_off_targets_similarity(smiles, primary_target)
    
    # Filter out primary target
    if primary_target:
        predictions = [
            p for p in predictions 
            if primary_target.upper() not in p.target_name.upper()
        ]
    
    # Collect safety alerts
    safety_alerts = []
    for pred in predictions:
        if pred.safety_alert and pred.probability > 0.3:
            safety_alerts.append({
                "target": pred.target_name,
                "gene": pred.gene_name,
                "probability": pred.probability,
                **pred.safety_alert,
            })
    
    # Generate report
    from datetime import datetime
    
    report = OffTargetReport(
        compound_id=compound_id,
        smiles=smiles,
        primary_target=primary_target,
        off_targets=predictions[:15],  # Top 15 off-targets
        safety_alerts=safety_alerts,
        analysis_timestamp=datetime.now().isoformat(),
        confidence_note="Based on similarity to known ligands. Experimental validation recommended.",
    )
    
    return report


def generate_off_target_summary_llm(
    report: OffTargetReport,
) -> str:
    """
    Generate a human-readable summary of off-target analysis using LLM.
    """
    from config.settings import LLM_PROVIDER, LLM_CONFIG, OPENAI_API_KEY, GOOGLE_API_KEY
    
    if not LLM_PROVIDER:
        return _format_off_target_summary_basic(report)
    
    # Format off-targets for prompt
    off_target_text = "\n".join([
        f"- {t.target_name} ({t.gene_name}): {t.probability:.0%} probability"
        for t in report.off_targets[:10]
    ])
    
    safety_text = "\n".join([
        f"- {a['target']}: {a['effect']} (severity: {a['severity']})"
        for a in report.safety_alerts
    ]) or "No critical safety alerts"
    
    prompt = f"""You are a pharmacology expert. Summarize this off-target analysis.

COMPOUND: {report.compound_id}
PRIMARY TARGET: {report.primary_target or "Not specified"}

PREDICTED OFF-TARGETS:
{off_target_text}

SAFETY ALERTS:
{safety_text}

Provide a brief summary (3-4 sentences) covering:
1. Key off-target concerns
2. Safety implications
3. Recommendation for the medicinal chemist

Summary:"""

    try:
        if LLM_PROVIDER == "openai":
            from openai import OpenAI
            client = OpenAI(api_key=OPENAI_API_KEY)
            response = client.chat.completions.create(
                model=LLM_CONFIG["openai"]["model"],
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=300,
            )
            return response.choices[0].message.content.strip()
        else:
            import google.generativeai as genai
            genai.configure(api_key=GOOGLE_API_KEY)
            model = genai.GenerativeModel(LLM_CONFIG["gemini"]["model"])
            response = model.generate_content(prompt)
            return response.text.strip()
            
    except Exception as e:
        logger.warning(f"LLM summary failed: {e}")
        return _format_off_target_summary_basic(report)


def _format_off_target_summary_basic(report: OffTargetReport) -> str:
    """Basic off-target summary without LLM."""
    lines = [f"**Off-Target Analysis for {report.compound_id}**\n"]
    
    if report.off_targets:
        lines.append("**Top Predicted Off-Targets:**")
        for t in report.off_targets[:5]:
            alert = " ⚠️" if t.safety_alert else ""
            lines.append(f"- {t.target_name}: {t.probability:.0%}{alert}")
    else:
        lines.append("No significant off-targets predicted.")
    
    if report.safety_alerts:
        lines.append("\n**⚠️ Safety Concerns:**")
        for a in report.safety_alerts:
            lines.append(f"- {a['target']}: {a['effect']}")
    
    return "\n".join(lines)


# Test
if __name__ == "__main__":
    # Test with gefitinib (EGFR inhibitor)
    test_smiles = "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1"
    
    print("Testing off-target prediction...")
    report = analyze_off_targets(
        compound_id="GEFITINIB",
        smiles=test_smiles,
        primary_target="EGFR",
    )
    
    print(f"\nFound {len(report.off_targets)} off-targets")
    for t in report.off_targets[:5]:
        print(f"  - {t.target_name}: {t.probability:.0%}")
    
    if report.safety_alerts:
        print(f"\nSafety alerts: {len(report.safety_alerts)}")

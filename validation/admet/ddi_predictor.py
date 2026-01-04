"""
Drug-Drug Interaction (DDI) Prediction Module.

Predicts potential interactions between a candidate compound and
common co-medications, critical for polypharmacy scenarios.

Key concern from PhD: "In India polypharmacy is common - which drug 
will go best with another drug?"

Predicts:
1. CYP450-mediated pharmacokinetic interactions
2. P-glycoprotein interactions
3. Known clinical DDI patterns
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

from utils.logging import get_logger

logger = get_logger(__name__)


class DDISeverity(Enum):
    """Severity levels for drug interactions."""
    CONTRAINDICATED = "contraindicated"
    MAJOR = "major"
    MODERATE = "moderate"
    MINOR = "minor"
    NONE = "none"


class DDIMechanism(Enum):
    """Mechanisms of drug interactions."""
    CYP3A4_INHIBITION = "CYP3A4 inhibition"
    CYP3A4_INDUCTION = "CYP3A4 induction"
    CYP2D6_INHIBITION = "CYP2D6 inhibition"
    CYP2C9_INHIBITION = "CYP2C9 inhibition"
    CYP2C19_INHIBITION = "CYP2C19 inhibition"
    CYP1A2_INHIBITION = "CYP1A2 inhibition"
    PGP_INHIBITION = "P-glycoprotein inhibition"
    PGP_INDUCTION = "P-glycoprotein induction"
    PHARMACODYNAMIC = "Pharmacodynamic interaction"
    PROTEIN_BINDING = "Protein binding displacement"
    RENAL_CLEARANCE = "Renal clearance competition"
    UNKNOWN = "Unknown mechanism"


@dataclass
class DrugProfile:
    """Profile of a drug for DDI analysis."""
    name: str
    drug_class: str
    cyp_substrates: List[str] = field(default_factory=list)  # CYPs it's metabolized by
    cyp_inhibitors: List[str] = field(default_factory=list)  # CYPs it inhibits
    cyp_inducers: List[str] = field(default_factory=list)    # CYPs it induces
    pgp_substrate: bool = False
    pgp_inhibitor: bool = False
    common_indications: List[str] = field(default_factory=list)


@dataclass
class DDIPrediction:
    """A predicted drug-drug interaction."""
    drug_name: str
    drug_class: str
    severity: DDISeverity
    mechanism: DDIMechanism
    description: str
    clinical_effect: str
    recommendation: str
    confidence: float = 0.0


@dataclass
class DDIReport:
    """Complete DDI analysis report."""
    compound_id: str
    smiles: str
    compound_cyp_profile: Dict[str, bool]  # Which CYPs it affects
    interactions: List[DDIPrediction] = field(default_factory=list)
    safe_combinations: List[str] = field(default_factory=list)
    
    def get_severe_interactions(self) -> List[DDIPrediction]:
        return [i for i in self.interactions 
                if i.severity in (DDISeverity.CONTRAINDICATED, DDISeverity.MAJOR)]
    
    def get_moderate_interactions(self) -> List[DDIPrediction]:
        return [i for i in self.interactions if i.severity == DDISeverity.MODERATE]
        
    def to_dict(self) -> Dict:
        return {
            "compound_id": self.compound_id,
            "smiles": self.smiles,
            "interactions": [
                {
                    "drug": i.drug_name,
                    "class": i.drug_class,
                    "severity": i.severity.value,
                    "mechanism": i.mechanism.value,
                    "effect": i.clinical_effect,
                    "recommendation": i.recommendation,
                }
                for i in self.interactions
            ],
            "safe_combinations": self.safe_combinations,
        }


# Common drugs database for DDI analysis
# Focus on diabetes + cardiac drugs (comorbidity mentioned by PhD)
COMMON_DRUGS_DB: Dict[str, DrugProfile] = {
    # Diabetes medications
    "metformin": DrugProfile(
        name="Metformin",
        drug_class="Antidiabetic (Biguanide)",
        cyp_substrates=[],  # Not CYP metabolized
        common_indications=["Type 2 Diabetes"],
    ),
    "glimepiride": DrugProfile(
        name="Glimepiride",
        drug_class="Antidiabetic (Sulfonylurea)",
        cyp_substrates=["CYP2C9"],
        common_indications=["Type 2 Diabetes"],
    ),
    "pioglitazone": DrugProfile(
        name="Pioglitazone",
        drug_class="Antidiabetic (Thiazolidinedione)",
        cyp_substrates=["CYP2C8", "CYP3A4"],
        common_indications=["Type 2 Diabetes"],
    ),
    "sitagliptin": DrugProfile(
        name="Sitagliptin",
        drug_class="Antidiabetic (DPP-4 Inhibitor)",
        cyp_substrates=["CYP3A4"],
        pgp_substrate=True,
        common_indications=["Type 2 Diabetes"],
    ),
    
    # Cardiac medications
    "atorvastatin": DrugProfile(
        name="Atorvastatin",
        drug_class="Statin",
        cyp_substrates=["CYP3A4"],
        pgp_substrate=True,
        common_indications=["Hyperlipidemia", "Cardiovascular Disease"],
    ),
    "simvastatin": DrugProfile(
        name="Simvastatin",
        drug_class="Statin",
        cyp_substrates=["CYP3A4"],
        common_indications=["Hyperlipidemia"],
    ),
    "amlodipine": DrugProfile(
        name="Amlodipine",
        drug_class="Calcium Channel Blocker",
        cyp_substrates=["CYP3A4"],
        common_indications=["Hypertension", "Angina"],
    ),
    "losartan": DrugProfile(
        name="Losartan",
        drug_class="ARB (Angiotensin Receptor Blocker)",
        cyp_substrates=["CYP2C9", "CYP3A4"],
        common_indications=["Hypertension"],
    ),
    "warfarin": DrugProfile(
        name="Warfarin",
        drug_class="Anticoagulant",
        cyp_substrates=["CYP2C9", "CYP3A4", "CYP1A2"],
        common_indications=["Anticoagulation", "Atrial Fibrillation"],
    ),
    "clopidogrel": DrugProfile(
        name="Clopidogrel",
        drug_class="Antiplatelet",
        cyp_substrates=["CYP2C19", "CYP3A4"],
        common_indications=["Antiplatelet Therapy", "Cardiovascular Disease"],
    ),
    "digoxin": DrugProfile(
        name="Digoxin",
        drug_class="Cardiac Glycoside",
        cyp_substrates=[],  # Not CYP metabolized
        pgp_substrate=True,
        common_indications=["Heart Failure", "Atrial Fibrillation"],
    ),
    "metoprolol": DrugProfile(
        name="Metoprolol",
        drug_class="Beta Blocker",
        cyp_substrates=["CYP2D6"],
        common_indications=["Hypertension", "Heart Failure"],
    ),
    
    # Common co-medications
    "omeprazole": DrugProfile(
        name="Omeprazole",
        drug_class="Proton Pump Inhibitor",
        cyp_substrates=["CYP2C19", "CYP3A4"],
        cyp_inhibitors=["CYP2C19"],
        common_indications=["GERD", "Peptic Ulcer"],
    ),
    "ketoconazole": DrugProfile(
        name="Ketoconazole",
        drug_class="Antifungal",
        cyp_inhibitors=["CYP3A4"],
        pgp_inhibitor=True,
        common_indications=["Fungal Infections"],
    ),
    "rifampicin": DrugProfile(
        name="Rifampicin",
        drug_class="Antibiotic",
        cyp_inducers=["CYP3A4", "CYP2C9", "CYP2C19"],
        common_indications=["Tuberculosis"],
    ),
    "fluconazole": DrugProfile(
        name="Fluconazole",
        drug_class="Antifungal",
        cyp_inhibitors=["CYP2C9", "CYP2C19", "CYP3A4"],
        common_indications=["Fungal Infections"],
    ),
}


def get_compound_cyp_profile(smiles: str) -> Dict[str, bool]:
    """
    Determine which CYP enzymes a compound may inhibit/induce.
    Uses the ADMET predictor.
    """
    try:
        from validation.admet.admet_predictor import predict_cyp_inhibition
        return predict_cyp_inhibition(smiles)
    except:
        return {}


def predict_ddi_with_drug(
    compound_smiles: str,
    compound_cyp_profile: Dict[str, bool],
    drug: DrugProfile,
) -> Optional[DDIPrediction]:
    """
    Predict interaction between candidate compound and a specific drug.
    """
    interactions_found = []
    
    # Check CYP-mediated interactions
    for cyp in drug.cyp_substrates:
        cyp_key = f"{cyp.lower()}_inhibitor"
        if compound_cyp_profile.get(cyp_key, False):
            # Compound inhibits CYP that metabolizes this drug
            severity = DDISeverity.MAJOR if cyp == "CYP3A4" else DDISeverity.MODERATE
            
            if drug.name == "Warfarin":
                severity = DDISeverity.MAJOR  # Always major for warfarin
            
            return DDIPrediction(
                drug_name=drug.name,
                drug_class=drug.drug_class,
                severity=severity,
                mechanism=DDIMechanism[f"{cyp.upper()}_INHIBITION"],
                description=f"May inhibit {cyp}-mediated metabolism of {drug.name}",
                clinical_effect=f"Increased {drug.name} levels, potential toxicity",
                recommendation=f"Monitor for {drug.name} toxicity, consider dose reduction",
                confidence=0.75,
            )
    
    # Check P-gp interactions
    if drug.pgp_substrate and compound_cyp_profile.get("pgp_inhibitor", False):
        return DDIPrediction(
            drug_name=drug.name,
            drug_class=drug.drug_class,
            severity=DDISeverity.MODERATE,
            mechanism=DDIMechanism.PGP_INHIBITION,
            description=f"May inhibit P-glycoprotein, affecting {drug.name} transport",
            clinical_effect=f"Increased {drug.name} absorption/decreased elimination",
            recommendation="Monitor drug levels if applicable",
            confidence=0.65,
        )
    
    return None


def predict_all_ddi(
    compound_id: str,
    smiles: str,
    focus_drugs: Optional[List[str]] = None,
) -> DDIReport:
    """
    Predict drug-drug interactions with common medications.
    
    Args:
        compound_id: Compound identifier
        smiles: SMILES string
        focus_drugs: Optional list of specific drugs to check
    
    Returns:
        DDIReport with all predicted interactions
    """
    logger.info(f"Predicting DDI for {compound_id}")
    
    # Get compound's CYP profile
    cyp_profile = get_compound_cyp_profile(smiles)
    
    interactions = []
    safe_combinations = []
    
    # Check against common drugs
    drugs_to_check = focus_drugs or list(COMMON_DRUGS_DB.keys())
    
    for drug_key in drugs_to_check:
        if drug_key.lower() not in COMMON_DRUGS_DB:
            continue
        
        drug = COMMON_DRUGS_DB[drug_key.lower()]
        interaction = predict_ddi_with_drug(smiles, cyp_profile, drug)
        
        if interaction:
            interactions.append(interaction)
        else:
            safe_combinations.append(drug.name)
    
    # Sort by severity
    severity_order = {
        DDISeverity.CONTRAINDICATED: 0,
        DDISeverity.MAJOR: 1,
        DDISeverity.MODERATE: 2,
        DDISeverity.MINOR: 3,
        DDISeverity.NONE: 4,
    }
    interactions.sort(key=lambda x: severity_order[x.severity])
    
    return DDIReport(
        compound_id=compound_id,
        smiles=smiles,
        compound_cyp_profile=cyp_profile,
        interactions=interactions,
        safe_combinations=safe_combinations,
    )


def generate_ddi_summary(report: DDIReport) -> str:
    """
    Generate human-readable DDI summary for display.
    """
    lines = [f"**💊 Drug Interaction Analysis for {report.compound_id}**\n"]
    
    # Severe interactions
    severe = report.get_severe_interactions()
    if severe:
        lines.append("**⚠️ Major Interactions:**")
        for ddi in severe:
            icon = "❌" if ddi.severity == DDISeverity.CONTRAINDICATED else "🔴"
            lines.append(f"{icon} **{ddi.drug_name}** ({ddi.drug_class})")
            lines.append(f"   - Mechanism: {ddi.mechanism.value}")
            lines.append(f"   - Effect: {ddi.clinical_effect}")
            lines.append(f"   - Recommendation: {ddi.recommendation}")
        lines.append("")
    
    # Moderate interactions
    moderate = report.get_moderate_interactions()
    if moderate:
        lines.append("**🟡 Moderate Interactions:**")
        for ddi in moderate:
            lines.append(f"- {ddi.drug_name}: {ddi.description}")
        lines.append("")
    
    # Safe combinations
    if report.safe_combinations:
        lines.append("**✅ No Significant Interaction Expected:**")
        lines.append(", ".join(report.safe_combinations[:10]))
    
    return "\n".join(lines)


def get_diabetes_cardiac_ddi(smiles: str, compound_id: str = "") -> DDIReport:
    """
    Focused DDI analysis for diabetes + cardiac polypharmacy.
    Common comorbidity in India as noted by PhD.
    """
    focus_drugs = [
        # Diabetes
        "metformin", "glimepiride", "pioglitazone", "sitagliptin",
        # Cardiac
        "atorvastatin", "amlodipine", "losartan", "metoprolol",
        "warfarin", "clopidogrel",
    ]
    
    return predict_all_ddi(compound_id, smiles, focus_drugs)


# Test
if __name__ == "__main__":
    # Test with a CYP3A4 inhibitor (like ketoconazole-like structure)
    test_smiles = "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1"
    
    print("Testing DDI prediction...")
    report = get_diabetes_cardiac_ddi(test_smiles, "TEST_COMPOUND")
    
    print(generate_ddi_summary(report))

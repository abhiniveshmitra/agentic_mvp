"""
Streamlit Dashboard for Drug Discovery Platform.

Phase 1 MVP: Trustworthy Pilot Interface

Features:
- Target protein input
- PubMed query builder
- Real-time pipeline progress
- Results table with scores and provenance
- CSV export
"""

import streamlit as st
import pandas as pd
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.settings import validate_config, LLM_PROVIDER, OPENAI_API_KEY, GOOGLE_API_KEY
from orchestrator.pipeline import Pipeline
from validation.controls.loaders import load_controls_for_target

# Page configuration
st.set_page_config(
    page_title="Drug Discovery Platform",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for premium look
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(120deg, #1a73e8 0%, #8e44ad 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        color: #5f6368;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
    }
    .status-running {
        color: #f59e0b;
        font-weight: 600;
    }
    .status-complete {
        color: #10b981;
        font-weight: 600;
    }
    .status-failed {
        color: #ef4444;
        font-weight: 600;
    }
    .warning-box {
        background-color: #fef3c7;
        border-left: 4px solid #f59e0b;
        padding: 1rem;
        border-radius: 4px;
    }
    .success-box {
        background-color: #d1fae5;
        border-left: 4px solid #10b981;
        padding: 1rem;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)


def main():
    """Main application entry point."""
    
    # Header
    st.markdown('<h1 class="main-header">🧬 Drug Discovery Platform</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Compound Discovery & Validation</p>', unsafe_allow_html=True)
    
    # Check API key (either OpenAI or Gemini)
    if not LLM_PROVIDER:
        st.error("⚠️ No LLM API key configured. Please set OPENAI_API_KEY or GOOGLE_API_KEY in your .env file.")
        st.stop()
    
    # Show which LLM is active
    if LLM_PROVIDER == "openai":
        st.success("🤖 Using OpenAI GPT-4o-mini for AI extraction")
    else:
        st.info("🧠 Using Google Gemini for AI extraction")
    
    # Sidebar - Configuration
    with st.sidebar:
        st.header("🎯 Configuration")
        
        # Target protein input
        st.subheader("Target Protein")
        target_protein = st.text_input(
            "Target Name/ID",
            value="EGFR",
            help="Enter the target protein name (e.g., EGFR, BCR-ABL)"
        )
        
        protein_sequence = st.text_area(
            "Protein Sequence (optional)",
            height=100,
            help="Enter the amino acid sequence if available"
        )
        
        # Discovery mode toggle - CRITICAL
        st.subheader("🔬 Discovery Mode")
        
        discovery_mode = st.radio(
            "Compound Source",
            options=["database", "literature"],
            index=0,  # Default to database
            format_func=lambda x: {
                "database": "📊 ChEMBL Database (recommended)",
                "literature": "📚 Literature Mining (experimental)"
            }[x],
            help="Database-first is recommended: molecules come from ChEMBL with guaranteed SMILES"
        )
        
        if discovery_mode == "database":
            max_compounds = st.slider(
                "Max Compounds",
                min_value=50,
                max_value=500,
                value=100,
                step=50,
                help="Maximum compounds to retrieve from ChEMBL"
            )
            st.success("✅ Using ChEMBL for reliable molecule discovery")
        else:
            st.warning("⚠️ Literature-only mode may produce few results")
            max_compounds = 100
        
        # PubMed query (only if literature mode or for context)
        if discovery_mode == "literature":
            st.subheader("Literature Search")
            
            # Get recommended query for target
            try:
                from discovery.query_templates import get_combined_query
                recommended_query = get_combined_query(target_protein)
            except:
                recommended_query = f"{target_protein} inhibitor IC50 compound"
            
            query = st.text_input(
                "PubMed Query",
                value=recommended_query,
                help="Use compound-focused queries for better extraction"
            )
            
            max_papers = st.slider(
                "Max Papers",
                min_value=10,
                max_value=300,
                value=100,
                step=10,
                help="Maximum number of papers to fetch"
            )
        else:
            query = ""
            max_papers = 0
        
        # Controls
        st.subheader("Controls")
        use_controls = st.checkbox("Use validation controls", value=True)
        
        if use_controls:
            positive, negative = load_controls_for_target(target_protein)
            st.info(f"📊 {len(positive)} positive, {len(negative)} negative controls loaded")
        
        # Run button
        st.divider()
        run_button = st.button(
            "🚀 Run Discovery Pipeline",
            type="primary",
            use_container_width=True
        )
    
    # Main content area
    # Main content area
    if run_button:
        # Clear previous results
        if "pipeline_results" in st.session_state:
            del st.session_state["pipeline_results"]
        
        run_pipeline(
            target_protein=target_protein,
            protein_sequence=protein_sequence if protein_sequence else None,
            query=query,
            max_papers=max_papers,
            use_controls=use_controls,
            discovery_mode=discovery_mode,
            max_compounds=max_compounds,
        )
    elif "pipeline_results" in st.session_state:
        # Display stored results
        results = st.session_state["pipeline_results"]
        display_database_results(
            candidates=results["candidates"],
            rejected=results["rejected"],
            target=results["target"],
            discovery_mode=results["discovery_mode"],
        )
    else:
        show_welcome_screen()


def show_welcome_screen():
    """Display welcome screen with instructions."""
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🔬 How it Works")
        st.markdown("""
        1. **Enter Target** - Specify your protein target
        2. **Search Literature** - Query PubMed for relevant papers
        3. **Extract Compounds** - AI identifies drug candidates
        4. **Validate** - Chemistry filters and ML scoring
        5. **Rank** - Get ranked, scored results
        """)
    
    with col2:
        st.markdown("### 📋 Pipeline Steps")
        st.markdown("""
        | Step | Description |
        |------|-------------|
        | Discovery | PubMed + Gemini extraction |
        | Filters | RDKit chemistry validation |
        | Scoring | DeepDTA affinity prediction |
        | Normalization | Percentiles & Z-scores |
        """)
    
    st.divider()
    
    st.markdown("### 🛡️ Trustworthy by Design")
    
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.metric("Architecture", "Deterministic")
        st.caption("Fixed pipeline, no autonomous decisions")
    
    with c2:
        st.metric("Validation", "Control-Based")
        st.caption("Known binders/non-binders for sanity checks")
    
    with c3:
        st.metric("Provenance", "Full Tracking")
        st.caption("Every compound traced to source")


def run_pipeline(
    target_protein: str,
    protein_sequence: str,
    query: str,
    max_papers: int,
    use_controls: bool,
    discovery_mode: str = "database",
    max_compounds: int = 100,
):
    """Execute the discovery pipeline with progress tracking."""
    
    st.divider()
    st.subheader("🔄 Pipeline Execution")
    
    # Progress container
    progress_container = st.container()
    
    with progress_container:
        progress_bar = st.progress(0)
        status_text = st.empty()
    
    # Results container
    results_container = st.container()
    
    try:
        # Load controls if enabled
        positive_controls = []
        negative_controls = []
        if use_controls:
            positive, negative = load_controls_for_target(target_protein)
            positive_controls = positive
            negative_controls = negative
        
        # Step 1: Initialize
        status_text.markdown("**Step 1/5:** Initializing pipeline...")
        progress_bar.progress(10)
        
        # CRITICAL: Database-first vs Literature discovery
        if discovery_mode == "database":
            status_text.markdown("**Step 2/5:** Querying ChEMBL database...")
            progress_bar.progress(25)
            
            from discovery.chembl_discovery import discover_from_database
            
            # Get compounds directly from ChEMBL
            candidates = discover_from_database(
                target_name=target_protein,
                max_compounds=max_compounds,
            )
            
            st.info(f"📊 Retrieved {len(candidates)} compounds from ChEMBL")
            
            # Skip literature completely for database mode
            papers_fetched = 0
            
        else:
            # Literature mode (experimental)
            status_text.markdown("**Step 2/5:** Searching literature...")
            progress_bar.progress(20)
            
            from discovery.literature_ingest import fetch_pubmed_papers
            from discovery.text_mining import extract_compounds
            from discovery.candidate_builder import build_candidates
            from utils.provenance import ProvenanceTracker
            
            # Fetch papers
            papers = fetch_pubmed_papers(query=query, max_results=max_papers)
            papers_fetched = len(papers)
            
            # Extract and build candidates
            raw_compounds = extract_compounds(papers, target=target_protein)
            provenance = ProvenanceTracker(run_id="lit_run")
            candidates = build_candidates(raw_compounds, provenance)
        
        # Step 3: Apply chemistry filters
        status_text.markdown("**Step 3/5:** Applying chemistry filters...")
        progress_bar.progress(50)
        
        from validation.chemistry_filters import apply_all_filters
        from orchestrator.run_config import ChemistryFilterConfig
        from discovery.query_templates import get_query_templates
        
        # Get known drugs for bypass
        try:
            templates = get_query_templates(target_protein)
            known_drugs = templates.get("known_drugs", [])
        except:
            known_drugs = []
        
        filter_config = ChemistryFilterConfig()
        passed_compounds, rejected_compounds = apply_all_filters(
            compounds=candidates,
            config=filter_config,
            known_drugs=known_drugs,
        )
        
        st.info(f"✅ {len(passed_compounds)} passed filters, {len(rejected_compounds)} rejected")
        
        # Step 4: Scoring
        status_text.markdown("**Step 4/5:** Scoring compounds...")
        progress_bar.progress(70)
        
        from validation.scoring.deepdta import DeepDTAScorer
        
        scorer = DeepDTAScorer()
        
        # Add controls to scoring
        all_to_score = passed_compounds.copy()
        for smiles in positive_controls:
            all_to_score.append({
                "smiles": smiles,
                "compound_id": f"positive_control_{len(all_to_score)}",
                "compound_name": "Positive Control",
                "is_control": True,
                "control_type": "positive",
            })
        for smiles in negative_controls:
            all_to_score.append({
                "smiles": smiles,
                "compound_id": f"negative_control_{len(all_to_score)}",
                "compound_name": "Negative Control",
                "is_control": True,
                "control_type": "negative",
            })
        
        # Score each compound
        scored = []
        for compound in all_to_score:
            if compound.get("smiles"):
                result = scorer.predict_affinity(
                    smiles=compound["smiles"],
                    protein=protein_sequence or target_protein,
                )
                compound["raw_score"] = result["score"]
                compound["uncertainty"] = result["uncertainty"]
            scored.append(compound)
        
        # Step 5: Normalize and output
        status_text.markdown("**Step 5/6:** Normalizing and ranking...")
        progress_bar.progress(80)
        
        from validation.normalization import normalize_scores
        from orchestrator.run_config import NormalizationConfig
        
        norm_config = NormalizationConfig()
        normalized = normalize_scores(scored, norm_config)
        
        # Sort by score
        candidates_only = [c for c in normalized if not c.get("is_control")]
        candidates_only.sort(key=lambda x: x.get("raw_score", 0), reverse=True)
        
        # Step 6: ADME/Toxicity Safety Check (post-ranking guardrail)
        status_text.markdown("**Step 6/6:** Assessing ADME/Toxicity risks...")
        progress_bar.progress(90)
        
        try:
            from validation.adme_tox import assess_batch, get_risk_summary
            from validation.patent_check import check_patent_batch, get_patent_summary
            
            # Assess ADME/Tox for all candidates
            candidates_only = assess_batch(candidates_only)
            
            # Patent check for top 20 only
            candidates_only = check_patent_batch(candidates_only, top_k=20)
            
            has_safety_checks = True
        except ImportError:
            has_safety_checks = False
        
        progress_bar.progress(100)
        status_text.markdown("**✓ Complete**")
        
        # Save results to session state
        st.session_state["pipeline_results"] = {
            "candidates": candidates_only,
            "rejected": rejected_compounds,
            "target": target_protein,
            "discovery_mode": discovery_mode,
        }
        
        # Display results
        with results_container:
            display_database_results(
                candidates=candidates_only,
                rejected=rejected_compounds,
                target=target_protein,
                discovery_mode=discovery_mode,
            )
        
    except Exception as e:
        st.error(f"❌ Pipeline failed: {str(e)}")
        st.exception(e)


def display_database_results(candidates, rejected, target, discovery_mode):
    """Display results from pipeline run with Phase-1 hardening features."""
    
    st.divider()
    
    if candidates:
        st.success(f"✅ Found {len(candidates)} ranked compounds for {target}!")
        
        # Import Phase-1 hardening modules
        try:
            from validation.known_drugs import (
                is_known_drug, get_known_drug_name, 
                validate_known_drugs_ranking, get_known_drugs
            )
            from validation.scaffold_diversity import (
                add_scaffold_info, calculate_diversity_metrics
            )
            has_hardening = True
        except ImportError:
            has_hardening = False
        
        # Add scaffold info and known drug markers
        if has_hardening:
            candidates = add_scaffold_info(candidates)
            
            # Mark known drugs
            known_drugs = get_known_drugs(target)
            for c in candidates:
                compound_id = c.get("compound_id", "")
                drug_name = get_known_drug_name(compound_id, target)
                if drug_name:
                    c["is_known_drug"] = True
                    c["known_drug_name"] = drug_name
                else:
                    c["is_known_drug"] = False
        
        # Score interpretation tooltip
        st.caption(
            "💡 **Score interpretation:** Lower (more negative) scores indicate "
            "stronger predicted binding affinity. This is model-specific and represents "
            "relative ranking, not absolute affinity values."
        )
        
        # Sanity Calibration Check
        if has_hardening and known_drugs:
            calibration = validate_known_drugs_ranking(candidates, target)
            
            with st.expander("🧪 Sanity Calibration (Known Drug Check)", expanded=True):
                if calibration["found_known_drugs"]:
                    # Show status with appropriate color
                    status = calibration.get("status", "UNKNOWN")
                    summary = calibration.get("summary", "")
                    
                    if status == "PASS":
                        st.success(f"✅ {summary}")
                    elif status == "PASS_WITH_NOTE":
                        st.info(f"ℹ️ {summary}")
                    else:
                        st.warning(f"⚠️ {summary}")
                    
                    # Show tier legend
                    st.caption("**Tier expectations:** BEST_IN_CLASS ≥85th | STANDARD ≥70th | LEGACY ≥50th")
                    
                    # Create display dataframe with key columns
                    cal_data = []
                    for drug in calibration["found_known_drugs"]:
                        cal_data.append({
                            "Drug": drug["name"],
                            "Rank": drug["rank"],
                            "Percentile": f"{drug['percentile']:.1f}",
                            "Tier": drug["tier"],
                            "Expected ≥": drug["expected_min"],
                            "Status": "✓" if drug["meets_expectation"] else "○",
                            "Gen": drug.get("generation", ""),
                        })
                    
                    st.dataframe(pd.DataFrame(cal_data), use_container_width=True)
                    
                    # Show notes if any
                    if calibration.get("notes"):
                        st.markdown("**Notes:**")
                        for note in calibration["notes"][:5]:
                            st.caption(f"• {note}")
                else:
                    st.info(f"ℹ️ No known {target} drugs found in results.")
        
        # Scaffold Diversity Metrics
        if has_hardening:
            diversity = calculate_diversity_metrics(candidates)
            
            with st.expander("🔬 Scaffold Diversity Analysis"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Unique Scaffolds", diversity["n_unique_scaffolds"])
                with col2:
                    st.metric("Diversity Ratio", f"{diversity['diversity_ratio']:.2f}")
                with col3:
                    st.metric("Evenness", f"{diversity['scaffold_evenness']:.2f}")
                
                st.markdown("**Most Common Scaffolds:**")
                if diversity["top_scaffolds"]:
                    for i, s in enumerate(diversity["top_scaffolds"], 1):
                        st.caption(f"{i}. {s['scaffold']} ({s['count']} compounds)")
        
        # Create dataframe with markers
        df_data = []
        for i, c in enumerate(candidates[:50]):  # Top 50
            name = c.get("compound_name", c.get("compound_id", "Unknown"))
            
            # Add known drug badge
            if c.get("is_known_drug"):
                name = f"⭐ {c.get('known_drug_name', name)} (APPROVED)"
            
            # Get ADME/Tox status with icon
            adme_status = c.get("adme_tox", {}).get("status", "")
            if adme_status == "SAFE":
                adme_display = "✅ SAFE"
            elif adme_status == "FLAGGED":
                adme_display = "⚠️ FLAGGED"
            elif adme_status == "HIGH_RISK":
                adme_display = "🚫 HIGH_RISK"
            else:
                adme_display = ""
            
            # Get Patent status with icon (only for top 20)
            patent_status = c.get("patent", {}).get("status", "")
            if patent_status == "CLEAR":
                patent_display = "✅ CLEAR"
            elif patent_status == "POTENTIAL_RISK":
                patent_display = "⚠️ RISK"
            elif patent_status == "LIKELY_ENCUMBERED":
                patent_display = "🚫 ENCUMBERED"
            else:
                patent_display = i < 20 and "—" or ""
            
            df_data.append({
                "Rank": i + 1,
                "Name": name,
                "Score": round(c.get("raw_score", 0), 3),
                "Percentile": round(c.get("percentile", 0), 1),
                "Confidence": c.get("confidence_tier", ""),
                "ADME/Tox": adme_display,
                "Patent": patent_display if i < 20 else "",
            })
        
        df = pd.DataFrame(df_data)
        
        st.subheader("📊 Ranked Results")
        st.dataframe(df, use_container_width=True, height=400)
        
        # Tab view for different perspectives
        tab1, tab2 = st.tabs(["📋 Full Results", "🎯 Top Per Scaffold"])
        
        with tab1:
            # Download button for full results
            full_df = pd.DataFrame(candidates)
            csv = full_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Full Results (CSV)",
                data=csv,
                file_name=f"drug_discovery_{target}.csv",
                mime="text/csv",
            )
        
        with tab2:
            # Diverse view - top per scaffold
            if has_hardening:
                from validation.scaffold_diversity import get_top_per_scaffold
                diverse = get_top_per_scaffold(candidates, n_per_scaffold=1)
                
                div_data = []
                for i, c in enumerate(diverse[:20]):
                    name = c.get("compound_name", c.get("compound_id", "Unknown"))
                    if c.get("is_known_drug"):
                        name = f"⭐ {c.get('known_drug_name', name)}"
                    
                    div_data.append({
                        "Rank": i + 1,
                        "Name": name,
                        "Score": round(c.get("raw_score", 0), 3),
                        "Scaffold": c.get("scaffold", "")[:30] + "...",
                        "Cluster Size": c.get("scaffold_cluster_size", 0),
                    })
                
                st.dataframe(pd.DataFrame(div_data), use_container_width=True)
                st.caption(f"Showing top compound from each of {len(diverse)} unique scaffolds")
    else:
        st.warning("No compounds passed all filters.")
    
    # =========================================================================
    # ADMET & OFF-TARGET ANALYSIS SECTION (New!)
    # =========================================================================
    if candidates:
        st.divider()
        st.subheader("🧬 ADMET & Off-Target Analysis")
        st.caption("Advanced pharmacokinetic and safety predictions for top compounds")
        
        # Try to import ADMET modules
        # Try to import ADMET and Explainer modules
        try:
            from validation.admet.admet_predictor import predict_admet
            from validation.admet.off_target_predictor import analyze_off_targets, generate_off_target_summary_llm
            from validation.admet.ddi_predictor import get_diabetes_cardiac_ddi, DDISeverity
            from validation.explainability.compound_explainer import generate_admet_explanation, generate_ddi_explanation, generate_pk_explanation
            has_admet = True
        except ImportError as e:
            has_admet = False
            st.info(f"ADMET modules not available: {e}")
        
        if has_admet:
            # Let user select a compound to analyze
            compound_options = {
                f"{c.get('compound_name', c.get('compound_id', 'Unknown'))}": i
                for i, c in enumerate(candidates[:10])
            }
            
            selected_compound = st.selectbox(
                "Select compound for detailed analysis:",
                list(compound_options.keys()),
                key="admet_compound_selector"
            )
            
            if selected_compound:
                idx = compound_options[selected_compound]
                compound = candidates[idx]
                smiles = compound.get("smiles", "")
                compound_id = compound.get("compound_id", "")
                
                if smiles:
                    col_admet, col_offtarget = st.columns(2)
                    
                    # ADMET Analysis
                    with col_admet:
                        st.markdown("### 💊 ADMET Profile")
                        
                        with st.spinner("Calculating ADMET properties..."):
                            admet_profile = predict_admet(smiles)
                        
                        # Display in tabs
                        tab_abs, tab_dist, tab_met, tab_tox, tab_pk = st.tabs(
                            ["Absorption", "Distribution", "Metabolism", "Toxicity", "⏱️ Pharmacokinetics"]
                        )
                        
                        with tab_abs:
                            st.metric("GI Absorption", admet_profile.gi_absorption)
                            if admet_profile.pgp_substrate:
                                st.warning("⚠️ P-glycoprotein substrate")
                            st.metric("Bioavailability Score", f"{admet_profile.bioavailability_score:.0%}")
                        
                        with tab_dist:
                            bbb_status = "✅ BBB+" if admet_profile.bbb_permeant else "❌ BBB-"
                            st.metric("Blood-Brain Barrier", bbb_status)
                            if admet_profile.log_bb:
                                st.caption(f"log BB: {admet_profile.log_bb:.2f}")
                        
                        with tab_met:
                            cyp_list = admet_profile.get_cyp_interactions()
                            if cyp_list:
                                st.warning("⚠️ CYP450 Interactions Detected")
                                for cyp in cyp_list:
                                    st.markdown(f"- {cyp}")
                            else:
                                st.success("✅ No significant CYP inhibition")
                        
                        with tab_tox:
                            tox_alerts = admet_profile.get_toxicity_alerts()
                            if tox_alerts:
                                for alert in tox_alerts:
                                    if alert["severity"] == "critical":
                                        st.error(f"🔴 {alert['type']}: {alert['message']}")
                                    else:
                                        st.warning(f"🟡 {alert['type']}: {alert['message']}")
                            else:
                                st.success("✅ No significant toxicity alerts")
                            
                            # hERG status
                            st.metric("hERG Risk", admet_profile.herg_inhibitor)
                            
                            # Synthetic accessibility
                            if admet_profile.sa_score:
                                sa_desc = "Easy" if admet_profile.sa_score < 4 else ("Moderate" if admet_profile.sa_score < 7 else "Difficult")
                                st.metric("Synthetic Accessibility", f"{sa_desc} ({admet_profile.sa_score:.1f}/10)")
                        
                        with tab_pk:
                            st.markdown("**⏱️ Pharmacokinetics Predictions**")
                            st.caption("Estimated PK parameters based on molecular properties")
                            
                            # Half-life with interpretation
                            col_hl, col_tmax = st.columns(2)
                            with col_hl:
                                if admet_profile.half_life_estimate:
                                    hl_val = admet_profile.half_life_estimate
                                    if hl_val >= 12:
                                        hl_status = "Long (once daily)"
                                    elif hl_val >= 6:
                                        hl_status = "Moderate (twice daily)"
                                    else:
                                        hl_status = "Short (multiple daily)"
                                    st.metric("Half-life", f"{hl_val:.1f} hrs", delta=hl_status)
                                else:
                                    st.metric("Half-life", "N/A")
                            
                            with col_tmax:
                                if admet_profile.tmax_estimate:
                                    tmax_val = admet_profile.tmax_estimate
                                    if tmax_val <= 1:
                                        tmax_status = "Rapid absorption"
                                    elif tmax_val <= 2:
                                        tmax_status = "Moderate absorption"
                                    else:
                                        tmax_status = "Slow absorption"
                                    st.metric("Tmax (Peak)", f"{tmax_val:.1f} hrs", delta=tmax_status)
                                else:
                                    st.metric("Tmax (Peak)", "N/A")
                            
                            # AUC and Cmax
                            col_auc, col_cmax = st.columns(2)
                            with col_auc:
                                auc_val = admet_profile.auc_relative or "N/A"
                                auc_icon = "🟢" if auc_val == "High" else ("🟡" if auc_val == "Moderate" else "🔴")
                                st.metric("Relative AUC", f"{auc_icon} {auc_val}")
                            
                            with col_cmax:
                                cmax_val = admet_profile.cmax_relative or "N/A"
                                cmax_icon = "🟢" if cmax_val == "High" else ("🟡" if cmax_val == "Moderate" else "🔴")
                                st.metric("Relative Cmax", f"{cmax_icon} {cmax_val}")
                            
                            # Oral bioavailability and dosing
                            st.divider()
                            col_f, col_dose = st.columns(2)
                            with col_f:
                                if admet_profile.oral_bioavailability_estimate:
                                    f_pct = admet_profile.oral_bioavailability_estimate * 100
                                    if f_pct >= 70:
                                        f_color = "normal"
                                    elif f_pct >= 40:
                                        f_color = "off"
                                    else:
                                        f_color = "inverse"
                                    st.metric("Oral Bioavailability", f"{f_pct:.0f}%")
                                else:
                                    st.metric("Oral Bioavailability", "N/A")
                            
                            with col_dose:
                                dosing = admet_profile.dosing_frequency or "N/A"
                                st.metric("💊 Dosing Recommendation", dosing)
                            
                            # LLM Explanation for PK
                            st.divider()
                            with st.spinner("🤖 Generating PK interpretation..."):
                                pk_profile = admet_profile.to_dict().get("pharmacokinetics", {})
                                pk_explanation = generate_pk_explanation(
                                    compound_id=compound_id,
                                    smiles=smiles,
                                    pk_profile=pk_profile,
                                )
                            st.info(pk_explanation)
                                
                        # LLM Explanation for ADMET
                        st.divider()
                        admet_summary = generate_admet_explanation(compound_id, admet_profile.to_dict())
                        st.info(admet_summary)
                    
                    # Off-Target Analysis
                    with col_offtarget:
                        st.markdown("### 🎯 Off-Target Prediction")
                        st.caption("Potential protein targets beyond primary target")
                        
                        with st.spinner("Analyzing off-targets..."):
                            off_target_report = analyze_off_targets(
                                compound_id=compound_id,
                                smiles=smiles,
                                primary_target=target,
                            )
                        
                        if off_target_report.off_targets:
                            # Show top predicted targets
                            for i, target in enumerate(off_target_report.off_targets[:5]):
                                prob_pct = target.probability * 100
                                icon = "🔴" if target.safety_alert else ("🟡" if prob_pct > 50 else "")
                                st.markdown(f"**{i+1}. {target.target_name}** - {prob_pct:.0f}% {icon}")
                                if target.safety_alert:
                                    st.caption(f"  ⚠️ {target.safety_alert.get('effect', '')}")
                            
                            # Safety alerts summary
                            if off_target_report.safety_alerts:
                                st.error(f"**⚠️ {len(off_target_report.safety_alerts)} Safety Alert(s)**")
                                for alert in off_target_report.safety_alerts[:3]:
                                    st.markdown(f"- {alert['target']}: {alert['effect']}")
                        else:
                            st.info("ℹ️ No significant off-targets predicted (or API unavailable)")
                        
                        st.caption(off_target_report.confidence_note)
                        
                        # LLM Explanation for Off-Target
                        if off_target_report.off_targets:
                            st.divider()
                            off_target_summary = generate_off_target_summary_llm(off_target_report)
                            st.info(off_target_summary)
                    
                    # Drug-Drug Interaction Analysis (New Section!)
                    st.divider()
                    st.markdown("### 💊 Drug-Drug Interaction Analysis")
                    st.caption("Polypharmacy safety check with common diabetes + cardiac medications")
                    
                    if True: # Already checked via has_admet
                        with st.spinner("Analyzing drug interactions..."):
                            ddi_report = get_diabetes_cardiac_ddi(smiles, compound_id)
                        
                        # Show interactions in columns
                        col_severe, col_safe = st.columns(2)
                        
                        with col_severe:
                            severe = ddi_report.get_severe_interactions()
                            moderate = ddi_report.get_moderate_interactions()
                            
                            if severe:
                                st.error(f"**⚠️ {len(severe)} Major Interaction(s)**")
                                for ddi in severe:
                                    icon = "❌" if ddi.severity == DDISeverity.CONTRAINDICATED else "🔴"
                                    st.markdown(f"{icon} **{ddi.drug_name}** ({ddi.drug_class})")
                                    st.caption(f"   Effect: {ddi.clinical_effect}")
                            
                            if moderate:
                                st.warning(f"**🟡 {len(moderate)} Moderate Interaction(s)**")
                                for ddi in moderate[:3]:
                                    st.markdown(f"- {ddi.drug_name}: {ddi.description}")
                            
                            if not severe and not moderate:
                                st.success("✅ No significant CYP-mediated interactions detected")
                        
                        with col_safe:
                            if ddi_report.safe_combinations:
                                st.success("**✅ Safe Combinations:**")
                                safe_text = ", ".join(ddi_report.safe_combinations[:8])
                                st.markdown(safe_text)
                        
                        # LLM Explanation for DDI
                        st.divider()
                        ddi_summary = generate_ddi_explanation(compound_id, ddi_report.to_dict())  # to_dict? check validity
                        st.info(ddi_summary)
                            
                        
                        # CYP Profile
                        if ddi_report.compound_cyp_profile:
                            st.markdown("**Compound CYP Profile:**")
                            for cyp, is_inhibitor in ddi_report.compound_cyp_profile.items():
                                if is_inhibitor:
                                    st.markdown(f"- ⚠️ {cyp.replace('_', ' ').title()}")
                    # Removed dead else block
                else:
                    st.warning("No SMILES available for this compound")
    
    # =========================================================================
    # GENERATIVE CHEMISTRY SECTION - Molecular Analog Generation
    # =========================================================================
    if candidates:
        st.divider()
        st.subheader("🧬 Generative Chemistry: Scaffold Hopping")
        st.caption("Generate novel molecular analogs from top compounds")
        
        try:
            from discovery.generative.scaffold_hopper import ScaffoldHopper, generate_analogs_for_compound
            from validation.explainability.mol3d_viewer import get_streamlit_3d_viewer
            import streamlit.components.v1 as components
            has_generative = True
        except ImportError as e:
            has_generative = False
            st.info(f"Generative chemistry module not available: {e}")
        
        if has_generative:
            # Let user select a compound for analog generation
            gen_compound_options = {
                f"{c.get('compound_name', c.get('compound_id', 'Unknown'))} (Rank #{i+1})": i
                for i, c in enumerate(candidates[:10])
            }
            
            col_select, col_settings = st.columns([2, 1])
            
            with col_select:
                selected_for_gen = st.selectbox(
                    "Select compound for analog generation:",
                    list(gen_compound_options.keys()),
                    key="gen_compound_selector"
                )
            
            with col_settings:
                n_analogs = st.slider("Number of analogs", 5, 20, 10, key="n_analogs_slider")
            
            if selected_for_gen:
                gen_idx = gen_compound_options[selected_for_gen]
                gen_compound = candidates[gen_idx]
                gen_smiles = gen_compound.get("smiles", "")
                
                if gen_smiles:
                    generate_btn = st.button("🧪 Generate Analogs", type="primary", key="generate_analogs_btn")
                    
                    if generate_btn:
                        with st.spinner(f"Generating {n_analogs} molecular analogs..."):
                            hopper = ScaffoldHopper(similarity_threshold=0.3)
                            
                            # Show scaffold
                            scaffold = hopper.extract_scaffold(gen_smiles)
                            if scaffold:
                                st.info(f"**Murcko Scaffold:** `{scaffold}`")
                            
                            # Generate analogs
                            analogs = hopper.generate_analogs(gen_smiles, n_analogs=n_analogs)
                        
                        if analogs:
                            st.success(f"✅ Generated {len(analogs)} novel analogs!")
                            
                            # Display analogs in columns
                            for i, analog in enumerate(analogs):
                                with st.expander(f"Analog #{i+1}: {analog.modification_type}", expanded=(i < 2)):
                                    col_mol, col_info = st.columns([1, 1])
                                    
                                    with col_mol:
                                        # Show 3D structure
                                        try:
                                            viewer_html = get_streamlit_3d_viewer(
                                                smiles=analog.smiles,
                                                title=f"Analog #{i+1}",
                                                width=300,
                                                height=250,
                                            )
                                            components.html(viewer_html, height=300, scrolling=False)
                                        except Exception:
                                            st.code(analog.smiles, language=None)
                                    
                                    with col_info:
                                        st.markdown(f"**SMILES:**")
                                        st.code(analog.smiles[:60] + "..." if len(analog.smiles) > 60 else analog.smiles)
                                        
                                        st.metric("Similarity to Parent", f"{analog.similarity_to_parent:.2%}")
                                        st.markdown(f"**Method:** {analog.modification_type}")
                                        
                                        # Copy button
                                        st.text_input(
                                            "Full SMILES (copy):",
                                            value=analog.smiles,
                                            key=f"copy_smiles_{i}",
                                            label_visibility="collapsed"
                                        )
                            
                            # Download all analogs as CSV
                            analog_data = [a.to_dict() for a in analogs]
                            analog_df = pd.DataFrame(analog_data)
                            csv = analog_df.to_csv(index=False)
                            
                            st.download_button(
                                label="📥 Download All Analogs (CSV)",
                                data=csv,
                                file_name=f"analogs_{gen_compound.get('compound_id', 'compound')}.csv",
                                mime="text/csv",
                            )
                        else:
                            st.warning("No valid analogs could be generated. Try a different compound.")
                else:
                    st.warning("No SMILES available for analog generation")
    
    # =========================================================================
    # RETROSYNTHESIS SECTION - Synthesis Route Planning
    # =========================================================================
    if candidates:
        st.divider()
        st.subheader("🧪 Retrosynthesis: Synthesis Route Planning")
        st.caption("Analyze synthetic accessibility and plan synthesis routes")
        
        try:
            from discovery.generative.retrosynthesis import RetrosynthesisPlanner
            has_retrosynthesis = True
        except ImportError as e:
            has_retrosynthesis = False
            st.info(f"Retrosynthesis module not available: {e}")
        
        if has_retrosynthesis:
            # Compound selector for retrosynthesis
            retro_compound_options = {
                f"{c.get('compound_name', c.get('compound_id', 'Unknown'))} (Rank #{i+1})": i
                for i, c in enumerate(candidates[:10])
            }
            
            selected_for_retro = st.selectbox(
                "Select compound for synthesis analysis:",
                list(retro_compound_options.keys()),
                key="retro_compound_selector"
            )
            
            if selected_for_retro:
                retro_idx = retro_compound_options[selected_for_retro]
                retro_compound = candidates[retro_idx]
                retro_smiles = retro_compound.get("smiles", "")
                
                if retro_smiles:
                    analyze_btn = st.button("🔬 Analyze Synthesis Route", type="primary", key="analyze_retro_btn")
                    
                    if analyze_btn:
                        with st.spinner("Analyzing synthesis route..."):
                            planner = RetrosynthesisPlanner()
                            route = planner.analyze(retro_smiles)
                            # Store in session state for validation button
                            st.session_state["current_retro_route"] = route
                            st.session_state["current_retro_smiles"] = retro_smiles
                    
                    # Get route from session state if available
                    route = st.session_state.get("current_retro_route")
                    retro_smiles_saved = st.session_state.get("current_retro_smiles", retro_smiles)
                    
                    if route:
                        # Display results in columns
                        col_score, col_feasibility, col_steps = st.columns(3)
                        
                        with col_score:
                            # SA Score with color coding
                            sa_color = "green" if route.synthetic_accessibility <= 4 else "orange" if route.synthetic_accessibility <= 6 else "red"
                            st.metric(
                                "Synthetic Accessibility",
                                f"{route.synthetic_accessibility:.1f}/10",
                                help="1=Easy, 10=Difficult"
                            )
                        
                        with col_feasibility:
                            feasibility_emoji = {"high": "✅", "moderate": "⚠️", "low": "🔶", "challenging": "⛔"}
                            st.metric(
                                "Feasibility",
                                f"{feasibility_emoji.get(route.estimated_feasibility, '')} {route.estimated_feasibility.title()}"
                            )
                        
                        with col_steps:
                            st.metric("Estimated Steps", route.total_steps)
                        
                        # Building blocks
                        st.markdown("### 🧱 Building Blocks")
                        if route.building_blocks:
                            bb_cols = st.columns(min(len(route.building_blocks), 4))
                            for i, bb in enumerate(route.building_blocks[:4]):
                                with bb_cols[i % 4]:
                                    avail_emoji = {"purchasable": "🟢", "likely": "🟡", "custom": "🔴"}
                                    st.markdown(f"**Block {i+1}** {avail_emoji.get(bb.availability, '')}")
                                    st.code(bb.smiles[:30] + "..." if len(bb.smiles) > 30 else bb.smiles)
                                    st.caption(bb.availability)
                        
                        # Reaction steps
                        if route.steps:
                            st.markdown("### ⚗️ Proposed Reaction Steps")
                            for i, step in enumerate(route.steps, 1):
                                with st.expander(f"Step {i}: {step.reaction_type}"):
                                    st.markdown(f"**Reactants:** `{step.reactants[0][:40]}...` + `{step.reactants[1][:40]}...`")
                                    st.markdown(f"**Conditions:** {step.conditions}")
                                    st.markdown(f"**Yield:** {step.yield_estimate}")
                        
                        # Notes
                        if route.notes:
                            st.markdown("### 📝 Synthesis Notes")
                            for note in route.notes:
                                st.markdown(note)
                        
                        # Validation section
                        st.markdown("### ✅ Route Validation")
                        
                        # Add validation toggle
                        use_llm_validation = st.checkbox(
                            "Use AI Expert Review (slower but more thorough)",
                            value=False,
                            key="use_llm_validation"
                        )
                        
                        if st.button("🔍 Validate Route", key="validate_route_btn"):
                            with st.spinner("Validating synthesis route..."):
                                try:
                                    from discovery.generative.synthesis_validator import (
                                        validate_synthesis_route,
                                        ReactionValidator
                                    )
                                    
                                    # Prepare data for validation
                                    bb_smiles = [bb.smiles for bb in route.building_blocks]
                                    steps_data = [
                                        {
                                            "reactants": step.reactants,
                                            "reaction_type": step.reaction_type,
                                            "conditions": step.conditions,
                                        }
                                        for step in route.steps
                                    ]
                                    
                                    # Run validation
                                    validation = validate_synthesis_route(
                                        retro_smiles_saved,
                                        bb_smiles,
                                        steps_data,
                                        use_llm=use_llm_validation,
                                    )
                                    
                                    # Display results
                                    col_v1, col_v2 = st.columns(2)
                                    
                                    with col_v1:
                                        if validation.is_valid:
                                            st.success(f"✅ Route Validated (Confidence: {validation.confidence:.0%})")
                                        else:
                                            st.error(f"⚠️ Issues Detected (Confidence: {validation.confidence:.0%})")
                                    
                                    with col_v2:
                                        conf_emoji = "🟢" if validation.confidence > 0.7 else "🟡" if validation.confidence > 0.4 else "🔴"
                                        st.markdown(f"**Validation Confidence:** {conf_emoji} {validation.confidence:.0%}")
                                    
                                    # Issues
                                    if validation.issues:
                                        st.error("**Issues Found:**")
                                        for issue in validation.issues:
                                            st.markdown(f"- ❌ {issue}")
                                    
                                    # Warnings
                                    if validation.warnings:
                                        st.warning("**Warnings:**")
                                        for warning in validation.warnings:
                                            st.markdown(f"- ⚠️ {warning}")
                                    
                                    # Suggestions
                                    if validation.suggestions:
                                        st.info("**Suggestions:**")
                                        for suggestion in validation.suggestions:
                                            st.markdown(f"- 💡 {suggestion}")
                                    
                                    if not validation.issues and not validation.warnings:
                                        st.success("✅ No issues detected - route appears chemically feasible")
                                        
                                except Exception as e:
                                    st.warning(f"Validation error: {e}")
                    else:
                        st.info("Click 'Analyze Synthesis Route' to see synthesis plan")
                else:
                    st.warning("No SMILES available for synthesis analysis")
    
    # =========================================================================
    # XAI EXPLANATION SECTION - Clickable compound details
    # =========================================================================
    st.divider()
    st.subheader("🔬 Explainable AI: Why Accepted/Rejected?")
    st.caption("Click on compounds below to see detailed explanations with molecular visualizations")
    
    # Import XAI modules
    try:
        from validation.explainability.explanation_engine import ExplanationEngine
        from validation.explainability.molecule_visualizer import (
            draw_molecule_with_heatmap, 
            draw_molecule_simple,
            draw_molecule_with_highlights,
        )
        from validation.explainability.mol3d_viewer import (
            smiles_to_3d_html,
            get_streamlit_3d_viewer,
            render_pharmacophore_features,
        )
        import streamlit.components.v1 as components
        has_3d_viewer = True
        from validation.explainability.structure_verification import (
            verify_and_correct_compound,
            verify_chembl_structure,
        )
        from validation.explainability.llm_verification import (
            verify_structure_with_llm,
        )
        from validation.explainability.compound_explainer import (
            generate_acceptance_explanation,
            generate_rejection_explanation,
        )
        has_xai = True
        has_verification = True
        has_llm_verification = True
        has_llm_explainer = True
        explainer = ExplanationEngine()
    except ImportError as e:
        has_xai = False
        has_verification = False
        has_llm_verification = False
        has_llm_explainer = False
        has_3d_viewer = False
        st.warning(f"XAI modules not available: {e}")
    
    if has_xai:
        col_accepted, col_rejected = st.columns(2)
        
        # =====================================================================
        # ACCEPTED COMPOUNDS EXPLANATIONS (Top 2)
        # =====================================================================
        with col_accepted:
            st.markdown("### ✅ Top Accepted Compounds")
            
            for i, compound in enumerate(candidates[:2]):
                compound_name = compound.get("compound_name", compound.get("compound_id", f"Compound {i+1}"))
                compound_id = compound.get("compound_id", "")
                smiles = compound.get("smiles", "")
                
                with st.expander(f"🔬 #{i+1}: {compound_name}", expanded=(i == 0)):
                    # VERIFY STRUCTURE BEFORE DISPLAY
                    # Step 1: ChEMBL API verification
                    if has_verification and compound_id.upper().startswith("CHEMBL"):
                        with st.spinner("Verifying structure via ChEMBL..."):
                            compound = verify_and_correct_compound(compound.copy())
                            smiles = compound.get("smiles", "")
                            
                            if compound.get("smiles_corrected"):
                                st.warning(
                                    f"⚠️ Structure corrected from ChEMBL database. "
                                    f"Original SMILES was incorrect."
                                )
                            elif not compound.get("structure_verified", True):
                                st.error(
                                    f"❌ Structure verification failed: "
                                    f"{compound.get('verification_message', 'Unknown error')}"
                                )
                    
                    # Step 2: LLM verification (if name differs from ID)
                    if has_llm_verification and compound_name.upper() != compound_id.upper():
                        with st.spinner("🤖 AI verifying compound identity..."):
                            is_valid, message, details = verify_structure_with_llm(
                                compound_name=compound_name,
                                compound_id=compound_id,
                                smiles=smiles,
                            )
                            
                            if not is_valid:
                                st.error(
                                    f"🤖 **AI Verification Failed:**\n{message}\n\n"
                                    f"Expected: {details.get('expected_structure', 'N/A')}\n"
                                    f"Observed: {details.get('observed_structure', 'N/A')}"
                                )
                                if details.get("correct_name"):
                                    st.info(f"💡 This structure appears to be: **{details.get('correct_name')}**")
                            else:
                                st.success(f"✅ {message}")
                    
                    # Generate explanation
                    explanation = explainer.explain_accepted(compound)
                    
                    # Two column layout
                    img_col, info_col = st.columns([1, 1])
                    
                    with img_col:
                        # Tabbed view for 2D and 3D molecules
                        mol_tab_2d, mol_tab_3d = st.tabs(["🎨 2D Heatmap", "🧬 3D Structure"])
                        
                        with mol_tab_2d:
                            # Draw molecule with heatmap
                            if smiles:
                                mol_img = draw_molecule_with_heatmap(
                                    smiles,
                                    atom_scores=explanation.atom_contributions,
                                    highlight_atoms=explanation.highlight_atoms,
                                    size=(350, 280)
                                )
                                if mol_img:
                                    st.image(
                                        f"data:image/png;base64,{mol_img}",
                                        caption="Atom Contribution Heatmap"
                                    )
                                else:
                                    st.code(smiles, language=None)
                            
                            st.markdown("""
                            <p style='font-size: 11px; color: #666;'>
                            🔴 <b>Red/Orange</b>: High binding contribution<br>
                            🔵 <b>Blue/Green</b>: Low contribution
                            </p>
                            """, unsafe_allow_html=True)
                        
                        with mol_tab_3d:
                            # Interactive 3D viewer
                            if smiles and has_3d_viewer:
                                try:
                                    viewer_html = get_streamlit_3d_viewer(
                                        smiles=smiles,
                                        title=compound_name[:30],
                                        score=compound.get("raw_score"),
                                        width=350,
                                        height=320,
                                    )
                                    components.html(viewer_html, height=380, scrolling=False)
                                    
                                    # Full screen button
                                    if st.button(f"🔍 Expand Full Screen", key=f"expand_3d_accepted_{i}"):
                                        st.session_state[f"show_fullscreen_3d_{compound_id}"] = True
                                    
                                    # Show fullscreen modal if triggered
                                    if st.session_state.get(f"show_fullscreen_3d_{compound_id}", False):
                                        @st.dialog(f"🧬 3D Structure: {compound_name[:40]}", width="large")
                                        def show_fullscreen_mol():
                                            large_viewer = get_streamlit_3d_viewer(
                                                smiles=smiles,
                                                title=compound_name,
                                                score=compound.get("raw_score"),
                                                width=750,
                                                height=550,
                                            )
                                            components.html(large_viewer, height=620, scrolling=False)
                                            st.caption("🖱️ Click and drag to rotate | Scroll to zoom | Press Esc to close")
                                            if st.button("Close", key=f"close_modal_{compound_id}"):
                                                st.session_state[f"show_fullscreen_3d_{compound_id}"] = False
                                                st.rerun()
                                        show_fullscreen_mol()
                                        st.session_state[f"show_fullscreen_3d_{compound_id}"] = False
                                except Exception as e:
                                    st.warning(f"3D viewer unavailable: {e}")
                                    st.code(smiles, language=None)
                            else:
                                st.info("Install py3Dmol for 3D visualization: `pip install py3Dmol`")
                    
                    with info_col:
                        # Predicted bioactivity
                        st.markdown("**Predicted Bioactivity**")
                        score = compound.get("raw_score", 0)
                        st.metric(
                            label="Binding Score",
                            value=f"{abs(score):.2f}",
                            delta=f"Top {100 - compound.get('percentile', 50):.0f}%"
                        )
                        
                        # Confidence
                        confidence = explanation.confidence
                        st.metric(
                            label="Confidence",
                            value=f"{confidence:.0%}"
                        )
                        
                        # Key features
                        st.markdown("**Why Accepted:**")
                        for feature in explanation.key_binding_features[:3]:
                            st.markdown(f"- ✅ {feature}")
                        
                        # Chemistry checks
                        if explanation.chemistry_checks:
                            st.markdown("**Passed Checks:**")
                            for check in explanation.chemistry_checks[:3]:
                                st.markdown(f"- {check['name']}: {check['value']}")
                    
                    # Pharmacophore features
                    if explanation.pharmacophore_matches:
                        st.markdown("**Pharmacophore Features:**")
                        pharm_cols = st.columns(len(explanation.pharmacophore_matches[:3]))
                        for j, match in enumerate(explanation.pharmacophore_matches[:3]):
                            with pharm_cols[j]:
                                st.metric(match["feature"], match["count"])
                    
                    # LLM-generated detailed explanation
                    if has_llm_explainer:
                        st.divider()
                        with st.spinner("🤖 Generating AI analysis..."):
                            llm_explanation = generate_acceptance_explanation(
                                compound_id=compound_id,
                                smiles=smiles,
                                binding_score=compound.get("raw_score", 0),
                                percentile=compound.get("percentile", 50),
                                confidence=explanation.confidence,
                            )
                            st.markdown(llm_explanation)
        
        # =====================================================================
        # REJECTED COMPOUNDS EXPLANATIONS (Top 2)
        # =====================================================================
        with col_rejected:
            st.markdown("### 🚫 Top Rejected Compounds")
            
            for i, compound in enumerate(rejected[:2]):
                compound_name = compound.get("compound_name", compound.get("compound_id", f"Rejected {i+1}"))
                compound_id = compound.get("compound_id", "")
                smiles = compound.get("smiles", "")
                
                with st.expander(f"❌ #{i+1}: {compound_name}", expanded=(i == 0)):
                    # VERIFY STRUCTURE BEFORE DISPLAY
                    if has_verification and compound_id.upper().startswith("CHEMBL"):
                        with st.spinner("Verifying structure..."):
                            compound = verify_and_correct_compound(compound.copy())
                            smiles = compound.get("smiles", "")
                            
                            if compound.get("smiles_corrected"):
                                st.info("ℹ️ Structure corrected from ChEMBL database.")
                    
                    # Step 2: LLM verification (if name differs from ID)
                    if has_llm_verification and compound_name.upper() != compound_id.upper():
                        with st.spinner("🤖 AI verifying compound identity..."):
                            is_valid, message, details = verify_structure_with_llm(
                                compound_name=compound_name,
                                compound_id=compound_id,
                                smiles=smiles,
                            )
                            
                            if not is_valid:
                                st.error(f"🤖 **AI Verification Failed:** {message}")
                                if details.get("correct_name"):
                                    st.info(f"💡 Structure appears to be: **{details.get('correct_name')}**")
                    
                    # Generate rejection explanation
                    explanation = explainer.explain_rejected(compound)
                    
                    # Two column layout
                    img_col, info_col = st.columns([1, 1])
                    
                    with img_col:
                        # Tabbed view for 2D and 3D molecules (matching accepted section)
                        rej_tab_2d, rej_tab_3d = st.tabs(["🎨 2D Highlights", "🧬 3D Structure"])
                        
                        with rej_tab_2d:
                            # Draw molecule with problematic atoms highlighted
                            if smiles:
                                mol_img = draw_molecule_with_highlights(
                                    smiles,
                                    highlight_atoms=explanation.problematic_atoms,
                                    highlight_color=(1.0, 0.2, 0.2),  # Red
                                    size=(350, 280)
                                )
                                if mol_img:
                                    st.image(
                                        f"data:image/png;base64,{mol_img}",
                                        caption="Problematic Features Highlighted"
                                    )
                                else:
                                    # Fallback to simple image
                                    simple_img = draw_molecule_simple(smiles, size=(350, 280))
                                    if simple_img:
                                        st.image(f"data:image/png;base64,{simple_img}")
                                    else:
                                        st.code(smiles, language=None)
                            
                            st.markdown("""
                            <p style='font-size: 11px; color: #666;'>
                            🔴 <b>Red atoms</b>: Problematic structural features
                            </p>
                            """, unsafe_allow_html=True)
                        
                        with rej_tab_3d:
                            # Interactive 3D viewer for rejected compounds
                            if smiles and has_3d_viewer:
                                try:
                                    viewer_html = get_streamlit_3d_viewer(
                                        smiles=smiles,
                                        title=f"❌ {compound_name[:25]}",
                                        width=350,
                                        height=320,
                                    )
                                    components.html(viewer_html, height=380, scrolling=False)
                                    
                                    # Full screen button
                                    if st.button(f"🔍 Expand Full Screen", key=f"expand_3d_rejected_{i}"):
                                        st.session_state[f"show_fullscreen_3d_rej_{compound_id}"] = True
                                    
                                    # Show fullscreen modal if triggered
                                    if st.session_state.get(f"show_fullscreen_3d_rej_{compound_id}", False):
                                        @st.dialog(f"🧬 3D Structure: {compound_name[:40]}", width="large")
                                        def show_fullscreen_mol_rej():
                                            large_viewer = get_streamlit_3d_viewer(
                                                smiles=smiles,
                                                title=f"❌ {compound_name}",
                                                width=750,
                                                height=550,
                                            )
                                            components.html(large_viewer, height=620, scrolling=False)
                                            st.caption("🖱️ Click and drag to rotate | Scroll to zoom | Press Esc to close")
                                            if st.button("Close", key=f"close_modal_rej_{compound_id}"):
                                                st.session_state[f"show_fullscreen_3d_rej_{compound_id}"] = False
                                                st.rerun()
                                        show_fullscreen_mol_rej()
                                        st.session_state[f"show_fullscreen_3d_rej_{compound_id}"] = False
                                except Exception as e:
                                    st.warning(f"3D viewer unavailable: {e}")
                                    st.code(smiles, language=None)
                            else:
                                st.info("Install py3Dmol for 3D visualization")
                    
                    with info_col:
                        # Rejection reason (prominent)
                        st.error(f"**Rejection Reason:**\n{explanation.rejection_reason}")
                        
                        # Category
                        st.markdown(f"**Category:** `{explanation.rejection_category}`")
                        
                        # Failed filters
                        if explanation.failed_filters:
                            st.markdown("**Failed Checks:**")
                            for filt in explanation.failed_filters:
                                severity_icon = "🔴" if filt.get("severity") == "critical" else "🟡"
                                st.markdown(
                                    f"- {severity_icon} **{filt['filter']}**: {filt['actual']} "
                                    f"(limit: {filt['limit']})"
                                )
                        
                        # Problematic features
                        if explanation.problematic_features:
                            st.markdown("**Problematic Features:**")
                            for feat in explanation.problematic_features:
                                st.markdown(f"- 🔴 {feat}")
                    
                    # Remediation hints
                    if explanation.remediation_hints:
                        st.info("**💡 Suggested Improvements:**\n" + 
                               "\n".join([f"• {h}" for h in explanation.remediation_hints[:3]]))
                    
                    # LLM-generated detailed explanation
                    if has_llm_explainer:
                        st.divider()
                        with st.spinner("🤖 Generating AI analysis..."):
                            llm_explanation = generate_rejection_explanation(
                                compound_id=compound_id,
                                smiles=smiles,
                                rejection_reason=compound.get("rejection_reason", "Unknown"),
                            )
                            st.markdown(llm_explanation)
    
    # =========================================================================
    # REJECTED COMPOUNDS TABLE (Full list)
    # =========================================================================
    if rejected:
        with st.expander(f"📋 View All {len(rejected)} Rejected Compounds"):
            # Categorize rejections
            pains_count = sum(1 for c in rejected if "PAINS" in c.get("rejection_reason", ""))
            mw_count = sum(1 for c in rejected if "MW" in c.get("rejection_reason", ""))
            logp_count = sum(1 for c in rejected if "LOGP" in c.get("rejection_reason", ""))
            
            st.caption(f"**Breakdown:** PAINS: {pains_count} | MW: {mw_count} | LogP: {logp_count}")
            
            rej_data = []
            for c in rejected[:20]:
                rej_data.append({
                    "Name": c.get("compound_name", c.get("compound_id", "Unknown")),
                    "Reason": c.get("rejection_reason", "Unknown"),
                })
            st.dataframe(pd.DataFrame(rej_data), use_container_width=True)


if __name__ == "__main__":
    main()

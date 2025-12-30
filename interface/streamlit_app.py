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

from config.settings import validate_config, GOOGLE_API_KEY
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
    
    # Check API key
    if not GOOGLE_API_KEY:
        st.error("⚠️ GOOGLE_API_KEY not configured. Please set it in your .env file.")
        st.stop()
    
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
    if run_button:
        run_pipeline(
            target_protein=target_protein,
            protein_sequence=protein_sequence if protein_sequence else None,
            query=query,
            max_papers=max_papers,
            use_controls=use_controls,
            discovery_mode=discovery_mode,
            max_compounds=max_compounds,
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
    
    # Show rejected if any
    if rejected:
        with st.expander(f"🚫 View {len(rejected)} Rejected Compounds"):
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

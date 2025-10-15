"""
Streamlit Dashboard for Cross-Lingual QA Results
===============================================

Interactive dashboard to visualize and analyze cross-lingual question answering
experiment results from mBERT and mT5 models.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import yaml
from pathlib import Path
import sys
import os

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from evaluation.analysis import ResultAnalyzer, StatisticalAnalyzer, ResultVisualizer
from utils.device_utils import DeviceManager

# Page configuration
st.set_page_config(
    page_title="Cross-Lingual QA Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS
st.markdown("""
<style>
    /* Main header styling */
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    /* Enhanced metric cards */
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #1f77b4;
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
        backdrop-filter: blur(4px);
        border: 1px solid rgba(255, 255, 255, 0.18);
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
    }
    
    /* Language cards */
    .language-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
        border: 1px solid rgba(0,0,0,0.05);
        transition: all 0.3s ease;
    }
    
    .language-card:hover {
        box-shadow: 0 8px 30px rgba(0,0,0,0.15);
        transform: translateY(-2px);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 10px 10px 0 0;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    
    /* Selectbox styling */
    .stSelectbox > div > div {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 10px;
    }
    
    /* Success/Error message styling */
    .stSuccess {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border: 1px solid #c3e6cb;
        border-radius: 10px;
    }
    
    .stError {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        border: 1px solid #f5c6cb;
        border-radius: 10px;
    }
    
    .stWarning {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        border: 1px solid #ffeaa7;
        border-radius: 10px;
    }
    
    .stInfo {
        background: linear-gradient(135deg, #d1ecf1 0%, #bee5eb 100%);
        border: 1px solid #bee5eb;
        border-radius: 10px;
    }
    
    /* Loading spinner */
    .stSpinner {
        border: 4px solid #f3f3f3;
        border-top: 4px solid #667eea;
        border-radius: 50%;
        width: 40px;
        height: 40px;
        animation: spin 2s linear infinite;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #5a6fd8 0%, #6a4190 100%);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_configurations():
    """Load all configuration files."""
    config_dir = Path(__file__).parent.parent / "config"
    
    configs = {}
    for config_file in ["model_configs.yaml", "training_configs.yaml", "experiment_configs.yaml"]:
        config_path = config_dir / config_file
        if config_path.exists():
            with open(config_path, 'r') as f:
                configs[config_file.replace('.yaml', '')] = yaml.safe_load(f)
    
    return configs

@st.cache_data
def load_results():
    """Load experiment results from JSON files."""
    results_dir = Path(__file__).parent.parent / "results"
    results = {}
    
    # Load zero-shot results
    for model in ["mbert", "mt5"]:
        zero_shot_file = results_dir / f"zero_shot_{model}_results.json"
        if zero_shot_file.exists():
            with open(zero_shot_file, 'r') as f:
                results[f"zero_shot_{model}"] = json.load(f)
    
    # Load few-shot results
    for model in ["mbert", "mt5"]:
        few_shot_file = results_dir / f"few_shot_{model}_results.json"
        if few_shot_file.exists():
            with open(few_shot_file, 'r') as f:
                results[f"few_shot_{model}"] = json.load(f)
    
    return results

@st.cache_data
def create_summary_metrics(results):
    """Create summary metrics from results."""
    summary = {}
    
    for key, data in results.items():
        if isinstance(data, dict):
            # Calculate average metrics across languages
            metrics = ['exact_match', 'f1', 'bleu']
            summary[key] = {}
            
            for metric in metrics:
                values = []
                for lang, lang_data in data.items():
                    if isinstance(lang_data, dict) and metric in lang_data:
                        values.append(lang_data[metric])
                
                if values:
                    summary[key][f"avg_{metric}"] = np.mean(values)
                    summary[key][f"std_{metric}"] = np.std(values)
                    summary[key][f"min_{metric}"] = np.min(values)
                    summary[key][f"max_{metric}"] = np.max(values)
    
    return summary

def create_language_family_mapping():
    """Create mapping of languages to their families."""
    return {
        'germanic': ['en', 'de'],
        'romance': ['es'],
        'slavic': ['ru'],
        'hellenic': ['el'],
        'turkic': ['tr'],
        'semitic': ['ar'],
        'austroasiatic': ['vi'],
        'tai': ['th'],
        'sino_tibetan': ['zh'],
        'indo_aryan': ['hi']
    }

def main():
    """Main dashboard application."""
    
    # Header with enhanced styling
    st.markdown('<h1 class="main-header">🌍 Cross-Lingual QA Dashboard</h1>', unsafe_allow_html=True)
    
    # Enhanced subtitle with better formatting
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <p style="font-size: 1.2rem; color: #666; font-weight: 500;">
            🚀 Interactive visualization of mBERT vs mT5 cross-lingual question answering results
        </p>
        <p style="font-size: 1rem; color: #888; margin-top: 0.5rem;">
            Explore performance metrics, statistical analysis, and comparative insights across 11 languages
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data with enhanced progress indicator
    with st.spinner("🔄 Loading configurations and results..."):
        configs = load_configurations()
        results = load_results()
        summary_metrics = create_summary_metrics(results)
        language_families = create_language_family_mapping()
    
    # Success message when data loads
    if results:
        st.success("✅ Data loaded successfully! Ready to explore your cross-lingual QA results.")
    
    # Enhanced Sidebar
    st.sidebar.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 1rem; border-radius: 10px; margin-bottom: 1rem;">
        <h2 style="color: white; text-align: center; margin: 0;">📊 Dashboard Controls</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Model selection with enhanced styling
    st.sidebar.markdown("### 🤖 Model Selection")
    available_models = []
    if "zero_shot_mbert" in results:
        available_models.append("mBERT")
    if "zero_shot_mt5" in results:
        available_models.append("mT5")
    
    if not available_models:
        st.sidebar.error("❌ No results found!")
        st.sidebar.info("💡 Run experiments first:\n- `python experiments/run_zero_shot.py`\n- `python experiments/run_few_shot.py`")
        return
    
    selected_models = st.sidebar.multiselect(
        "Choose models to compare",
        available_models,
        default=available_models,
        help="Select which models to include in the analysis"
    )
    
    st.sidebar.markdown("---")
    
    # Experiment type selection
    st.sidebar.markdown("### 🧪 Experiment Types")
    experiment_types = st.sidebar.multiselect(
        "Select experiment types",
        ["Zero-Shot", "Few-Shot"],
        default=["Zero-Shot", "Few-Shot"],
        help="Choose which experiment types to analyze"
    )
    
    st.sidebar.markdown("---")
    
    # Language selection with enhanced formatting
    st.sidebar.markdown("### 🌍 Language Selection")
    all_languages = ['en', 'es', 'de', 'el', 'ru', 'tr', 'ar', 'vi', 'th', 'zh', 'hi']
    language_names = {
        'en': 'English', 'es': 'Spanish', 'de': 'German', 'el': 'Greek',
        'ru': 'Russian', 'tr': 'Turkish', 'ar': 'Arabic', 'vi': 'Vietnamese',
        'th': 'Thai', 'zh': 'Chinese', 'hi': 'Hindi'
    }
    
    selected_languages = st.sidebar.multiselect(
        "Choose languages to analyze",
        all_languages,
        default=all_languages,
        format_func=lambda x: f"{x.upper()} - {language_names[x]}",
        help="Select which languages to include in the analysis"
    )
    
    st.sidebar.markdown("---")
    
    # Metric selection
    st.sidebar.markdown("### 📈 Metrics")
    metrics = st.sidebar.multiselect(
        "Select performance metrics",
        ["Exact Match", "F1 Score", "BLEU Score"],
        default=["F1 Score"],
        help="Choose which metrics to display and analyze"
    )
    
    # Add a quick stats section in sidebar
    if results and selected_models:
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📊 Quick Stats")
        
        total_experiments = len([k for k in results.keys() if any(model.lower() in k for model in selected_models)])
        total_languages = len(selected_languages)
        
        st.sidebar.metric("Experiments", total_experiments)
        st.sidebar.metric("Languages", total_languages)
        st.sidebar.metric("Models", len(selected_models))
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Overview", "🌍 Language Analysis", "📊 Model Comparison", 
        "🔬 Statistical Analysis", "⚙️ Configuration"
    ])
    
    with tab1:
        show_overview_tab(summary_metrics, selected_models, experiment_types)
    
    with tab2:
        show_language_analysis_tab(results, selected_models, selected_languages, 
                                 metrics, language_families, language_names)
    
    with tab3:
        show_model_comparison_tab(results, selected_models, selected_languages, metrics)
    
    with tab4:
        show_statistical_analysis_tab(results, selected_models, selected_languages)
    
    with tab5:
        show_configuration_tab(configs)

def show_overview_tab(summary_metrics, selected_models, experiment_types):
    """Show overview tab with summary metrics."""
    st.header("📈 Experiment Overview")
    
    if not summary_metrics:
        st.warning("⚠️ No results available for overview.")
        return
    
    # Add overview description
    st.markdown("""
    <div style="background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); 
                padding: 1.5rem; border-radius: 15px; margin-bottom: 2rem;">
        <h4 style="margin: 0 0 1rem 0; color: #333;">📊 Performance Summary</h4>
        <p style="margin: 0; color: #666;">
            This overview provides a high-level comparison of model performance across different 
            experiment types and metrics. Use the controls in the sidebar to filter and explore specific aspects.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create metrics columns with enhanced styling
    cols = st.columns(len(selected_models) if selected_models else 1)
    
    for i, model in enumerate(selected_models):
        with cols[i % len(cols)]:
            # Enhanced model header
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        padding: 1rem; border-radius: 10px; margin-bottom: 1rem; text-align: center;">
                <h3 style="color: white; margin: 0;">🤖 {model}</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Find results for this model
            model_results = {}
            for key, data in summary_metrics.items():
                if model.lower() in key:
                    model_results[key] = data
            
            if model_results:
                for exp_type, metrics in model_results.items():
                    # Enhanced experiment type header
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); 
                                padding: 0.8rem; border-radius: 8px; margin: 1rem 0;">
                        <h4 style="margin: 0; color: #495057;">🧪 {exp_type.replace('_', ' ').title()}</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Create metric cards
                    metric_cols = st.columns(3)
                    
                    with metric_cols[0]:
                        if 'avg_f1' in metrics:
                            st.metric(
                                "Average F1", 
                                f"{metrics['avg_f1']:.3f}", 
                                f"±{metrics['std_f1']:.3f}",
                                help="F1 Score measures the harmonic mean of precision and recall"
                            )
                    
                    with metric_cols[1]:
                        if 'avg_exact_match' in metrics:
                            st.metric(
                                "Average EM", 
                                f"{metrics['avg_exact_match']:.3f}", 
                                f"±{metrics['std_exact_match']:.3f}",
                                help="Exact Match measures the percentage of predictions that exactly match the ground truth"
                            )
                    
                    with metric_cols[2]:
                        if 'avg_bleu' in metrics:
                            st.metric(
                                "Average BLEU", 
                                f"{metrics['avg_bleu']:.3f}", 
                                f"±{metrics['std_bleu']:.3f}",
                                help="BLEU Score measures the quality of generated text compared to reference text"
                            )
                    
                    # Add performance insights
                    if 'avg_f1' in metrics:
                        f1_score = metrics['avg_f1']
                        if f1_score > 0.7:
                            st.success("🎯 Excellent performance!")
                        elif f1_score > 0.5:
                            st.info("✅ Good performance")
                        else:
                            st.warning("⚠️ Performance could be improved")
                    
                    st.markdown("---")
            else:
                st.info(f"No results found for {model}")
    
    # Add summary insights
    if summary_metrics:
        st.markdown("### 🔍 Key Insights")
        
        # Find best performing model
        best_model = None
        best_f1 = 0
        
        for key, data in summary_metrics.items():
            if 'avg_f1' in data and data['avg_f1'] > best_f1:
                best_f1 = data['avg_f1']
                best_model = key
        
        if best_model:
            st.success(f"🏆 **{best_model.replace('_', ' ').title()}** shows the best overall performance with an average F1 score of **{best_f1:.3f}**")
        
        # Performance range analysis
        all_f1_scores = [data.get('avg_f1', 0) for data in summary_metrics.values() if 'avg_f1' in data]
        if all_f1_scores:
            min_f1, max_f1 = min(all_f1_scores), max(all_f1_scores)
            st.info(f"📊 Performance range: {min_f1:.3f} - {max_f1:.3f} (F1 Score)")
            
            if max_f1 - min_f1 < 0.1:
                st.success("🎯 Models show consistent performance across experiments")
            else:
                st.warning("⚠️ Significant performance variation detected between experiments")

def show_language_analysis_tab(results, selected_models, selected_languages, 
                              metrics, language_families, language_names):
    """Show language analysis tab."""
    st.header("🌍 Language Performance Analysis")
    
    if not results:
        st.warning("No results available for language analysis.")
        return
    
    # Create performance by language plot
    st.subheader("Performance by Language")
    
    # Prepare data for plotting
    plot_data = []
    metric_mapping = {
        "Exact Match": "exact_match",
        "F1 Score": "f1", 
        "BLEU Score": "bleu"
    }
    
    for model in selected_models:
        for exp_type in ["zero_shot", "few_shot"]:
            key = f"{exp_type}_{model.lower()}"
            if key in results:
                for lang in selected_languages:
                    if lang in results[key]:
                        for metric_display in metrics:
                            metric_key = metric_mapping[metric_display]
                            if metric_key in results[key][lang]:
                                plot_data.append({
                                    'Language': f"{lang.upper()} - {language_names[lang]}",
                                    'Model': model,
                                    'Experiment': exp_type.replace('_', ' ').title(),
                                    'Metric': metric_display,
                                    'Score': results[key][lang][metric_key],
                                    'Language_Code': lang
                                })
    
    if plot_data:
        df = pd.DataFrame(plot_data)
        
        # Create interactive plot
        fig = px.bar(
            df, 
            x='Language', 
            y='Score', 
            color='Model',
            facet_col='Metric',
            facet_row='Experiment',
            title="Performance by Language and Model",
            hover_data=['Language_Code']
        )
        
        fig.update_layout(
            height=600,
            showlegend=True,
            xaxis_tickangle=-45
        )
        
        st.plotly_chart(fig, width='stretch')
        
        # Language family analysis
        st.subheader("Performance by Language Family")
        
        family_data = []
        for model in selected_models:
            for exp_type in ["zero_shot", "few_shot"]:
                key = f"{exp_type}_{model.lower()}"
                if key in results:
                    for family, langs in language_families.items():
                        family_scores = []
                        for lang in langs:
                            if lang in selected_languages and lang in results[key]:
                                if 'f1' in results[key][lang]:
                                    family_scores.append(results[key][lang]['f1'])
                        
                        if family_scores:
                            family_data.append({
                                'Family': family.title(),
                                'Model': model,
                                'Experiment': exp_type.replace('_', ' ').title(),
                                'Avg_F1': np.mean(family_scores),
                                'Count': len(family_scores)
                            })
        
        if family_data:
            family_df = pd.DataFrame(family_data)
            
            fig_family = px.bar(
                family_df,
                x='Family',
                y='Avg_F1',
                color='Model',
                facet_col='Experiment',
                title="Average F1 Score by Language Family",
                hover_data=['Count']
            )
            
            st.plotly_chart(fig_family, width='stretch')
    
    else:
        st.info("No data available for the selected filters.")

def show_model_comparison_tab(results, selected_models, selected_languages, metrics):
    """Show model comparison tab."""
    st.header("📊 Model Comparison")
    
    if len(selected_models) < 2:
        st.info("Select at least 2 models to see comparison.")
        return
    
    if not results:
        st.warning("No results available for model comparison.")
        return
    
    # Model comparison plot
    st.subheader("Direct Model Comparison")
    
    comparison_data = []
    metric_mapping = {
        "Exact Match": "exact_match",
        "F1 Score": "f1",
        "BLEU Score": "bleu"
    }
    
    for exp_type in ["zero_shot", "few_shot"]:
        for lang in selected_languages:
            for metric_display in metrics:
                metric_key = metric_mapping[metric_display]
                lang_scores = {}
                
                for model in selected_models:
                    key = f"{exp_type}_{model.lower()}"
                    if key in results and lang in results[key]:
                        if metric_key in results[key][lang]:
                            lang_scores[model] = results[key][lang][metric_key]
                
                if len(lang_scores) >= 2:  # At least 2 models have scores
                    for model, score in lang_scores.items():
                        comparison_data.append({
                            'Language': f"{lang.upper()}",
                            'Model': model,
                            'Experiment': exp_type.replace('_', ' ').title(),
                            'Metric': metric_display,
                            'Score': score
                        })
    
    if comparison_data:
        df = pd.DataFrame(comparison_data)
        
        # Create comparison plot
        fig = px.bar(
            df,
            x='Language',
            y='Score',
            color='Model',
            facet_col='Metric',
            facet_row='Experiment',
            title="Model Comparison Across Languages",
            barmode='group'
        )
        
        fig.update_layout(
            height=600,
            showlegend=True,
            xaxis_tickangle=-45
        )
        
        st.plotly_chart(fig, width='stretch')
        
        # Summary statistics
        st.subheader("Comparison Summary")
        
        summary_stats = df.groupby(['Model', 'Experiment', 'Metric'])['Score'].agg([
            'mean', 'std', 'min', 'max'
        ]).round(3)
        
        st.dataframe(summary_stats, width='stretch')
    
    else:
        st.info("No comparison data available for the selected models and languages.")

def show_statistical_analysis_tab(results, selected_models, selected_languages):
    """Show statistical analysis tab."""
    st.header("🔬 Statistical Analysis")
    
    if len(selected_models) < 2:
        st.info("Select at least 2 models to see statistical analysis.")
        return
    
    if not results:
        st.warning("No results available for statistical analysis.")
        return
    
    # Statistical significance testing
    st.subheader("Statistical Significance Tests")
    
    # Prepare data for statistical tests
    test_data = {}
    
    for exp_type in ["zero_shot", "few_shot"]:
        for metric in ["f1", "exact_match"]:
            model_scores = {}
            
            for model in selected_models:
                key = f"{exp_type}_{model.lower()}"
                if key in results:
                    scores = []
                    for lang in selected_languages:
                        if lang in results[key] and metric in results[key][lang]:
                            scores.append(results[key][lang][metric])
                    if scores:
                        model_scores[model] = scores
            
            if len(model_scores) >= 2:
                test_data[f"{exp_type}_{metric}"] = model_scores
    
    if test_data:
        # Perform t-tests
        from scipy import stats
        
        for test_name, model_scores in test_data.items():
            st.subheader(f"T-test: {test_name.replace('_', ' ').title()}")
            
            models = list(model_scores.keys())
            if len(models) >= 2:
                model1, model2 = models[0], models[1]
                scores1, scores2 = model_scores[model1], model_scores[model2]
                
                # Ensure equal length
                min_len = min(len(scores1), len(scores2))
                scores1, scores2 = scores1[:min_len], scores2[:min_len]
                
                if min_len > 1:
                    t_stat, p_value = stats.ttest_rel(scores1, scores2)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("T-statistic", f"{t_stat:.4f}")
                    with col2:
                        st.metric("P-value", f"{p_value:.4f}")
                    with col3:
                        significance = "Significant" if p_value < 0.05 else "Not Significant"
                        st.metric("Result", significance)
                    
                    # Interpretation
                    if p_value < 0.05:
                        better_model = model1 if np.mean(scores1) > np.mean(scores2) else model2
                        st.success(f"✅ {better_model} significantly outperforms the other model (p < 0.05)")
                    else:
                        st.info("ℹ️ No significant difference between models (p ≥ 0.05)")
                    
                    st.markdown("---")
    
    else:
        st.info("No data available for statistical analysis.")

def show_configuration_tab(configs):
    """Show configuration tab."""
    st.header("⚙️ Experiment Configuration")
    
    if not configs:
        st.warning("No configuration files found.")
        return
    
    for config_name, config_data in configs.items():
        with st.expander(f"📄 {config_name.replace('_', ' ').title()}"):
            st.json(config_data)

if __name__ == "__main__":
    main()

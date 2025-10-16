"""
Enhanced Streamlit Dashboard for Cross-Lingual QA Results
=======================================================

A modern, intuitive dashboard for visualizing and analyzing cross-lingual 
question answering experiment results from mBERT and mT5 models.
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
from datetime import datetime

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

# Page configuration
st.set_page_config(
    page_title="Cross-Lingual QA Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Modern CSS
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    .main {
        font-family: 'Inter', sans-serif;
    }
    
    /* Custom Header */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem 0;
        border-radius: 20px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
    }
    
    .main-header h1 {
        color: white;
        font-size: 3rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    .main-header p {
        color: rgba(255,255,255,0.9);
        font-size: 1.2rem;
        margin: 0.5rem 0 0 0;
        font-weight: 400;
    }
    
    /* Sidebar Styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    .sidebar-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin-bottom: 1.5rem;
        text-align: center;
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.2);
    }
    
    .sidebar-header h2 {
        color: white;
        margin: 0;
        font-size: 1.5rem;
        font-weight: 600;
    }
    
    /* Control Cards */
    .control-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin-bottom: 1rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        border: 1px solid rgba(0,0,0,0.05);
    }
    
    .control-card h3 {
        color: #495057;
        margin: 0 0 1rem 0;
        font-size: 1.1rem;
        font-weight: 600;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #667eea;
        box-shadow: 0 5px 20px rgba(0,0,0,0.1);
        margin: 1rem 0;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 30px rgba(0,0,0,0.15);
    }
    
    /* Status Indicators */
    .status-excellent {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border-left-color: #28a745;
    }
    
    .status-good {
        background: linear-gradient(135deg, #d1ecf1 0%, #bee5eb 100%);
        border-left-color: #17a2b8;
    }
    
    .status-warning {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        border-left-color: #ffc107;
    }
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 5px;
        background: #f8f9fa;
        padding: 5px;
        border-radius: 15px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: white;
        border-radius: 10px;
        padding: 0.8rem 1.5rem;
        font-weight: 500;
        transition: all 0.3s ease;
        border: 1px solid transparent;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: 1px solid #667eea;
        box-shadow: 0 3px 10px rgba(102, 126, 234, 0.3);
    }
    
    /* Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 1.5rem;
        font-weight: 500;
        transition: all 0.3s ease;
        box-shadow: 0 3px 10px rgba(102, 126, 234, 0.2);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Selectbox Styling */
    .stSelectbox > div > div {
        background: white;
        border-radius: 10px;
        border: 1px solid #dee2e6;
        transition: all 0.3s ease;
    }
    
    .stSelectbox > div > div:hover {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Multiselect Styling */
    .stMultiSelect > div > div {
        background: white;
        border-radius: 10px;
        border: 1px solid #dee2e6;
    }
    
    /* Alert Styling */
    .stSuccess {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border: 1px solid #c3e6cb;
        border-radius: 15px;
        padding: 1rem;
    }
    
    .stError {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        border: 1px solid #f5c6cb;
        border-radius: 15px;
        padding: 1rem;
    }
    
    .stWarning {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        border: 1px solid #ffeaa7;
        border-radius: 15px;
        padding: 1rem;
    }
    
    .stInfo {
        background: linear-gradient(135deg, #d1ecf1 0%, #bee5eb 100%);
        border: 1px solid #bee5eb;
        border-radius: 15px;
        padding: 1rem;
    }
    
    /* Loading Spinner */
    .stSpinner {
        border: 4px solid #f3f3f3;
        border-top: 4px solid #667eea;
        border-radius: 50%;
        width: 50px;
        height: 50px;
        animation: spin 1s linear infinite;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Content Sections */
    .content-section {
        background: white;
        padding: 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: 0 5px 20px rgba(0,0,0,0.1);
        border: 1px solid rgba(0,0,0,0.05);
    }
    
    .section-header {
        color: #495057;
        font-size: 1.8rem;
        font-weight: 600;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        gap: 0.8rem;
    }
    
    /* Quick Stats */
    .quick-stats {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 1.5rem 0;
    }
    
    .stat-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        border: 1px solid #dee2e6;
    }
    
    .stat-value {
        font-size: 2rem;
        font-weight: 700;
        color: #667eea;
        margin: 0;
    }
    
    .stat-label {
        color: #6c757d;
        font-size: 0.9rem;
        margin: 0.5rem 0 0 0;
        font-weight: 500;
    }
    
    /* Custom Scrollbar */
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
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .main-header h1 {
            font-size: 2rem;
        }
        
        .main-header p {
            font-size: 1rem;
        }
        
        .content-section {
            padding: 1rem;
        }
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

def get_language_names():
    """Get human-readable language names."""
    return {
        'en': 'English', 'es': 'Spanish', 'de': 'German', 'el': 'Greek',
        'ru': 'Russian', 'tr': 'Turkish', 'ar': 'Arabic', 'vi': 'Vietnamese',
        'th': 'Thai', 'zh': 'Chinese', 'hi': 'Hindi'
    }

def create_sidebar():
    """Create the enhanced sidebar with better organization."""
    
    # Sidebar Header
    st.sidebar.markdown("""
    <div class="sidebar-header">
        <h2>🎛️ Dashboard Controls</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data
    with st.spinner("🔄 Loading data..."):
        results = load_results()
        summary_metrics = create_summary_metrics(results)
    
    if not results:
        st.sidebar.error("❌ No results found!")
        st.sidebar.info("💡 Run experiments first:\n- `python experiments/run_zero_shot.py`\n- `python experiments/run_few_shot.py`")
        return None, None, None, None, None
    
    # Model Selection
    st.sidebar.markdown("""
    <div class="control-card">
        <h3>🤖 Model Selection</h3>
    </div>
    """, unsafe_allow_html=True)
    
    available_models = []
    if "zero_shot_mbert" in results:
        available_models.append("mBERT")
    if "zero_shot_mt5" in results:
        available_models.append("mT5")
    
    selected_models = st.sidebar.multiselect(
        "Choose models to compare",
        available_models,
        default=available_models,
        help="Select which models to include in the analysis"
    )
    
    # Experiment Type Selection
    st.sidebar.markdown("""
    <div class="control-card">
        <h3>🧪 Experiment Types</h3>
    </div>
    """, unsafe_allow_html=True)
    
    experiment_types = st.sidebar.multiselect(
        "Select experiment types",
        ["Zero-Shot", "Few-Shot"],
        default=["Zero-Shot", "Few-Shot"],
        help="Choose which experiment types to analyze"
    )
    
    # Language Selection
    st.sidebar.markdown("""
    <div class="control-card">
        <h3>🌍 Language Selection</h3>
    </div>
    """, unsafe_allow_html=True)
    
    all_languages = ['en', 'es', 'de', 'el', 'ru', 'tr', 'ar', 'vi', 'th', 'zh', 'hi']
    language_names = get_language_names()
    
    selected_languages = st.sidebar.multiselect(
        "Choose languages to analyze",
        all_languages,
        default=all_languages,
        format_func=lambda x: f"{x.upper()} - {language_names[x]}",
        help="Select which languages to include in the analysis"
    )
    
    # Metric Selection
    st.sidebar.markdown("""
    <div class="control-card">
        <h3>📈 Metrics</h3>
    </div>
    """, unsafe_allow_html=True)
    
    metrics = st.sidebar.multiselect(
        "Select performance metrics",
        ["Exact Match", "F1 Score", "BLEU Score"],
        default=["F1 Score"],
        help="Choose which metrics to display and analyze"
    )
    
    # Quick Stats
    if results and selected_models:
        st.sidebar.markdown("""
        <div class="control-card">
            <h3>📊 Quick Stats</h3>
        </div>
        """, unsafe_allow_html=True)
        
        total_experiments = len([k for k in results.keys() if any(model.lower() in k for model in selected_models)])
        total_languages = len(selected_languages)
        
        col1, col2 = st.sidebar.columns(2)
        with col1:
            st.metric("Experiments", total_experiments)
        with col2:
            st.metric("Languages", total_languages)
    
    return results, summary_metrics, selected_models, selected_languages, metrics

def create_header():
    """Create the enhanced header."""
    st.markdown("""
    <div class="main-header">
        <h1>🌍 Cross-Lingual QA Dashboard</h1>
        <p>Interactive analysis of mBERT vs mT5 performance across 11 languages</p>
    </div>
    """, unsafe_allow_html=True)

def show_dashboard_overview(results, summary_metrics, selected_models):
    """Show the main dashboard overview."""
    
    st.markdown("""
    <div class="content-section">
        <div class="section-header">
            📊 Dashboard Overview
        </div>
    """, unsafe_allow_html=True)
    
    # Quick Stats Grid
    st.markdown('<div class="quick-stats">', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-value">11</div>
            <div class="stat-label">Languages</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-value">2</div>
            <div class="stat-label">Models</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-value">2</div>
            <div class="stat-label">Experiment Types</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        total_experiments = len([k for k in results.keys() if any(model.lower() in k for model in selected_models)]) if selected_models else 0
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{total_experiments}</div>
            <div class="stat-label">Active Experiments</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Performance Summary
    if summary_metrics:
        st.markdown("### 🎯 Performance Summary")
        
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
    
    st.markdown('</div>', unsafe_allow_html=True)

def show_performance_analysis(results, selected_models, selected_languages, metrics):
    """Show detailed performance analysis."""
    
    st.markdown("""
    <div class="content-section">
        <div class="section-header">
            📈 Performance Analysis
        </div>
    """, unsafe_allow_html=True)
    
    if not results or not selected_models:
        st.warning("⚠️ No data available for performance analysis.")
        st.markdown('</div>', unsafe_allow_html=True)
        return
    
    # Create performance by language plot
    st.markdown("### 🌍 Performance by Language")
    
    # Prepare data for plotting
    plot_data = []
    metric_mapping = {
        "Exact Match": "exact_match",
        "F1 Score": "f1", 
        "BLEU Score": "bleu"
    }
    
    language_names = get_language_names()
    
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
            hover_data=['Language_Code'],
            color_discrete_sequence=['#667eea', '#764ba2']
        )
        
        fig.update_layout(
            height=600,
            showlegend=True,
            xaxis_tickangle=-45,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Language family analysis
        st.markdown("### 🏛️ Performance by Language Family")
        
        language_families = create_language_family_mapping()
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
                hover_data=['Count'],
                color_discrete_sequence=['#667eea', '#764ba2']
            )
            
            fig_family.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            
            st.plotly_chart(fig_family, use_container_width=True)
    
    else:
        st.info("No data available for the selected filters.")
    
    st.markdown('</div>', unsafe_allow_html=True)

def show_model_comparison(results, selected_models, selected_languages, metrics):
    """Show model comparison analysis."""
    
    st.markdown("""
    <div class="content-section">
        <div class="section-header">
            ⚖️ Model Comparison
        </div>
    """, unsafe_allow_html=True)
    
    if len(selected_models) < 2:
        st.info("Select at least 2 models to see comparison.")
        st.markdown('</div>', unsafe_allow_html=True)
        return
    
    if not results:
        st.warning("No results available for model comparison.")
        st.markdown('</div>', unsafe_allow_html=True)
        return
    
    # Model comparison plot
    st.markdown("### 📊 Direct Model Comparison")
    
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
            barmode='group',
            color_discrete_sequence=['#667eea', '#764ba2']
        )
        
        fig.update_layout(
            height=600,
            showlegend=True,
            xaxis_tickangle=-45,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Summary statistics
        st.markdown("### 📋 Comparison Summary")
        
        summary_stats = df.groupby(['Model', 'Experiment', 'Metric'])['Score'].agg([
            'mean', 'std', 'min', 'max'
        ]).round(3)
        
        st.dataframe(summary_stats, use_container_width=True)
    
    else:
        st.info("No comparison data available for the selected models and languages.")
    
    st.markdown('</div>', unsafe_allow_html=True)

def show_statistical_analysis(results, selected_models, selected_languages):
    """Show statistical analysis."""
    
    st.markdown("""
    <div class="content-section">
        <div class="section-header">
            🔬 Statistical Analysis
        </div>
    """, unsafe_allow_html=True)
    
    if len(selected_models) < 2:
        st.info("Select at least 2 models to see statistical analysis.")
        st.markdown('</div>', unsafe_allow_html=True)
        return
    
    if not results:
        st.warning("No results available for statistical analysis.")
        st.markdown('</div>', unsafe_allow_html=True)
        return
    
    # Statistical significance testing
    st.markdown("### 📊 Statistical Significance Tests")
    
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
            st.markdown(f"#### T-test: {test_name.replace('_', ' ').title()}")
            
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
    
    st.markdown('</div>', unsafe_allow_html=True)

def show_configuration(configs):
    """Show configuration information."""
    
    st.markdown("""
    <div class="content-section">
        <div class="section-header">
            ⚙️ Experiment Configuration
        </div>
    """, unsafe_allow_html=True)
    
    if not configs:
        st.warning("No configuration files found.")
        st.markdown('</div>', unsafe_allow_html=True)
        return
    
    for config_name, config_data in configs.items():
        with st.expander(f"📄 {config_name.replace('_', ' ').title()}"):
            st.json(config_data)
    
    st.markdown('</div>', unsafe_allow_html=True)

def main():
    """Main dashboard application."""
    
    # Create header
    create_header()
    
    # Create sidebar and get data
    results, summary_metrics, selected_models, selected_languages, metrics = create_sidebar()
    
    if results is None:
        return
    
    # Success message when data loads
    st.success("✅ Data loaded successfully! Ready to explore your cross-lingual QA results.")
    
    # Main content with better tab structure
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🏠 Dashboard", "📈 Performance", "⚖️ Comparison", 
        "🔬 Statistics", "⚙️ Configuration"
    ])
    
    with tab1:
        show_dashboard_overview(results, summary_metrics, selected_models)
    
    with tab2:
        show_performance_analysis(results, selected_models, selected_languages, metrics)
    
    with tab3:
        show_model_comparison(results, selected_models, selected_languages, metrics)
    
    with tab4:
        show_statistical_analysis(results, selected_models, selected_languages)
    
    with tab5:
        configs = load_configurations()
        show_configuration(configs)

if __name__ == "__main__":
    main()
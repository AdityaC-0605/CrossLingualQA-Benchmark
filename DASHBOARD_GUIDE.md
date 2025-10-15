# 🌍 Cross-Lingual QA Dashboard Guide

## Overview

The Cross-Lingual QA Dashboard is an interactive Streamlit application that provides comprehensive visualization and analysis of cross-lingual question answering experiment results. It allows researchers to explore, compare, and analyze the performance of mBERT and mT5 models across different languages and experimental settings.

## 🚀 Quick Start

### 1. Launch the Dashboard
```bash
# Simple launch
./run_dashboard.sh

# Or manually
source venv/bin/activate
streamlit run dashboard/app.py
```

### 2. Access the Dashboard
Open your browser to: **http://localhost:8501**

### 3. Explore Results
- Use the sidebar to select models, languages, and metrics
- Navigate through different tabs for various analyses
- Interact with charts and visualizations

## 📊 Dashboard Features

### 📈 Overview Tab
**Purpose**: High-level summary of experiment results

**Features**:
- Average performance metrics across all languages
- Model comparison summary
- Key performance indicators (F1, EM, BLEU)
- Standard deviation and confidence intervals

**Use Cases**:
- Quick performance assessment
- Model selection decisions
- Overall experiment success evaluation

### 🌍 Language Analysis Tab
**Purpose**: Detailed analysis of performance across languages

**Features**:
- Performance breakdown by individual languages
- Language family analysis (Germanic, Romance, Slavic, etc.)
- Interactive bar charts with hover information
- Language-specific insights

**Use Cases**:
- Identifying language-specific challenges
- Understanding cross-lingual transfer patterns
- Language family performance analysis

### 📊 Model Comparison Tab
**Purpose**: Direct comparison between mBERT and mT5

**Features**:
- Side-by-side performance metrics
- Statistical summary tables
- Model-specific strengths and weaknesses
- Performance difference analysis

**Use Cases**:
- Model selection for specific languages
- Understanding model trade-offs
- Performance gap analysis

### 🔬 Statistical Analysis Tab
**Purpose**: Statistical significance testing and analysis

**Features**:
- T-tests for model comparison
- P-values and confidence intervals
- Statistical significance indicators
- Effect size analysis

**Use Cases**:
- Validating performance differences
- Statistical rigor in evaluation
- Research publication support

### ⚙️ Configuration Tab
**Purpose**: View experiment configurations and settings

**Features**:
- Model parameters and hyperparameters
- Training configuration details
- Experiment setup information
- Configuration validation

**Use Cases**:
- Reproducibility verification
- Configuration debugging
- Experiment documentation

## 🎛️ Dashboard Controls

### Sidebar Options

#### Model Selection
- **mBERT**: Multilingual BERT model
- **mT5**: Multilingual T5 model
- **Both**: Compare both models

#### Experiment Types
- **Zero-Shot**: Train on English, test on target languages
- **Few-Shot**: Fine-tune with small number of examples
- **Both**: Compare zero-shot vs few-shot performance

#### Language Selection
- **All Languages**: All 11 XQuAD languages
- **Specific Languages**: Select individual languages
- **Language Families**: Group by linguistic families

#### Metrics
- **F1 Score**: Harmonic mean of precision and recall
- **Exact Match**: Exact answer matching
- **BLEU Score**: Text generation quality (for T5)

## 📁 Data Format

### Expected Results Structure
```
results/
├── zero_shot_mbert_results.json
├── zero_shot_mt5_results.json
├── few_shot_mbert_results.json
└── few_shot_mt5_results.json
```

### JSON Format
```json
{
  "en": {
    "exact_match": 0.75,
    "f1": 0.82,
    "bleu": 0.78
  },
  "es": {
    "exact_match": 0.68,
    "f1": 0.74,
    "bleu": 0.71
  }
}
```

## 🔧 Installation & Setup

### Prerequisites
- Python 3.8+
- Virtual environment activated
- Cross-Lingual QA project set up

### Install Dashboard Dependencies
```bash
pip install streamlit streamlit-plotly-events streamlit-aggrid streamlit-option-menu
```

### Create Sample Data (Optional)
```bash
python dashboard/create_sample_data.py
```

## 🎨 Customization

### Adding New Metrics
1. Update `metric_mapping` in `dashboard/app.py`
2. Add metric to sidebar selection
3. Update visualization functions

### Adding New Languages
1. Update `all_languages` list
2. Add to `language_names` dictionary
3. Update language family mappings

### Styling Customization
- Modify CSS in `st.markdown()` sections
- Update Plotly color schemes
- Customize page layout and themes

## 📊 Visualization Types

### Bar Charts
- Performance by language
- Model comparison
- Language family analysis

### Interactive Plots
- Hover information
- Zoom and pan capabilities
- Export functionality

### Statistical Plots
- Confidence intervals
- Error bars
- Significance indicators

## 🔍 Analysis Workflows

### 1. Model Selection Workflow
1. Go to **Overview Tab**
2. Compare average performance metrics
3. Check **Statistical Analysis Tab** for significance
4. Use **Model Comparison Tab** for detailed comparison

### 2. Language Analysis Workflow
1. Go to **Language Analysis Tab**
2. Select specific languages of interest
3. Analyze performance patterns
4. Check language family groupings

### 3. Experiment Validation Workflow
1. Check **Configuration Tab** for setup verification
2. Use **Statistical Analysis Tab** for significance testing
3. Validate results in **Overview Tab**

## 🚨 Troubleshooting

### Common Issues

#### "No results found"
- Ensure experiments have been run
- Check results directory exists
- Verify JSON file format

#### Import errors
- Activate virtual environment
- Install all dependencies
- Check Python path configuration

#### Performance issues
- Limit selected languages/models
- Use data sampling for large datasets
- Check system memory usage

#### Dashboard not loading
- Check port 8501 is available
- Verify Streamlit installation
- Check firewall settings

### Error Messages

#### "ModuleNotFoundError"
```bash
pip install -r requirements.txt
```

#### "Port already in use"
```bash
streamlit run dashboard/app.py --server.port 8502
```

#### "No results available"
```bash
python dashboard/create_sample_data.py
```

## 📈 Best Practices

### Dashboard Usage
1. **Start with Overview**: Get high-level understanding
2. **Filter Appropriately**: Use sidebar controls effectively
3. **Compare Systematically**: Use consistent filters for comparisons
4. **Validate Statistically**: Check significance of differences

### Data Interpretation
1. **Consider Language Families**: Group related languages
2. **Check Sample Sizes**: Ensure statistical power
3. **Look for Patterns**: Identify systematic trends
4. **Validate Results**: Cross-check with other analyses

### Performance Optimization
1. **Limit Selections**: Don't select all languages/models at once
2. **Use Caching**: Dashboard caches data automatically
3. **Close Unused Tabs**: Reduce memory usage
4. **Refresh Periodically**: Clear cache if needed

## 🔮 Advanced Features

### Export Functionality
- Download charts as PNG/PDF
- Export data as CSV/JSON
- Save filtered results

### Real-time Updates
- Automatic refresh when new results added
- Live monitoring of experiment progress
- Dynamic configuration updates

### Responsive Design
- Works on desktop and mobile
- Adaptive layout for different screen sizes
- Touch-friendly controls

## 📚 Additional Resources

### Documentation
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Plotly Documentation](https://plotly.com/python/)
- [Cross-Lingual QA Project README](README.md)

### Support
- Check project issues on GitHub
- Review dashboard logs for errors
- Consult experiment documentation

## 🎯 Use Cases

### Research
- Paper writing and analysis
- Conference presentation preparation
- Research collaboration and sharing

### Development
- Model debugging and optimization
- Performance monitoring
- A/B testing of different approaches

### Education
- Teaching cross-lingual NLP concepts
- Student project evaluation
- Research methodology demonstration

---

**The Cross-Lingual QA Dashboard provides a comprehensive, interactive platform for analyzing and visualizing cross-lingual question answering results. Use it to gain insights, make informed decisions, and communicate your findings effectively.**

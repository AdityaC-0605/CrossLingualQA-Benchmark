# 🌍 Cross-Lingual QA Dashboard

Interactive Streamlit dashboard for visualizing and analyzing cross-lingual question answering experiment results.

## Features

### 📈 Overview Tab
- Summary metrics across all experiments
- Average performance by model and experiment type
- Key performance indicators (F1, EM, BLEU)

### 🌍 Language Analysis Tab
- Performance breakdown by individual languages
- Language family analysis (Germanic, Romance, Slavic, etc.)
- Interactive bar charts and visualizations

### 📊 Model Comparison Tab
- Direct comparison between mBERT and mT5
- Side-by-side performance metrics
- Statistical summary tables

### 🔬 Statistical Analysis Tab
- Statistical significance testing (t-tests)
- P-values and confidence intervals
- Model performance significance analysis

### ⚙️ Configuration Tab
- View all experiment configurations
- Model parameters and training settings
- Experiment setup details

## Installation

1. Install additional dashboard dependencies:
```bash
pip install streamlit streamlit-plotly-events streamlit-aggrid streamlit-option-menu
```

2. Or install all requirements:
```bash
pip install -r requirements.txt
```

## Usage

### Start the Dashboard
```bash
streamlit run dashboard/app.py
```

### Access the Dashboard
Open your browser to: `http://localhost:8501`

## Dashboard Controls

### Sidebar Options
- **Select Models**: Choose mBERT, mT5, or both
- **Experiment Types**: Zero-shot, Few-shot, or both
- **Select Languages**: Choose specific languages to analyze
- **Select Metrics**: F1 Score, Exact Match, BLEU Score

### Interactive Features
- **Hover Information**: Detailed tooltips on all charts
- **Zoom and Pan**: Interactive plotly charts
- **Filtering**: Real-time filtering of results
- **Export**: Download charts and data

## Data Requirements

The dashboard expects results in the following format:
```
results/
├── zero_shot_mbert_results.json
├── zero_shot_mt5_results.json
├── few_shot_mbert_results.json
└── few_shot_mt5_results.json
```

## Example Results Format

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

## Customization

### Adding New Metrics
1. Update the `metric_mapping` dictionary in `app.py`
2. Add the metric to the sidebar selection
3. Update the visualization functions

### Adding New Languages
1. Update the `all_languages` list
2. Add language names to `language_names` dictionary
3. Update language family mappings

### Styling
- Modify the CSS in the `st.markdown()` section
- Update color schemes in plotly charts
- Customize the page layout

## Troubleshooting

### No Results Found
- Ensure experiments have been run first
- Check that results files exist in the `results/` directory
- Verify JSON file format is correct

### Import Errors
- Make sure all dependencies are installed
- Check that the `src/` directory is accessible
- Verify Python path configuration

### Performance Issues
- Use `@st.cache_data` for expensive operations
- Limit the number of selected languages/models
- Consider data sampling for large datasets

## Advanced Features

### Real-time Updates
The dashboard automatically refreshes when new results are added to the results directory.

### Export Functionality
- Download charts as PNG/PDF
- Export data as CSV/JSON
- Save filtered results

### Responsive Design
- Works on desktop and mobile devices
- Adaptive layout based on screen size
- Touch-friendly controls

## Contributing

To add new features to the dashboard:

1. Fork the repository
2. Create a new branch
3. Add your feature to `dashboard/app.py`
4. Update this README
5. Submit a pull request

## License

This dashboard is part of the Cross-Lingual QA project and follows the same license terms.

# Cross-Lingual Question Answering System

A comprehensive implementation of cross-lingual question answering using multilingual BERT (mBERT) and T5 (mT5) models, supporting zero-shot and few-shot learning across 11 languages.

## Overview

This project implements and compares cross-lingual question answering systems using:
- **mBERT**: Multilingual BERT for span extraction
- **mT5**: Multilingual T5 for text-to-text generation
- **Zero-shot learning**: Train on English, evaluate on target languages
- **Few-shot learning**: Fine-tune with 1, 5, or 10 examples per language

## Features

- ✅ **11 Languages**: English, Spanish, German, Greek, Russian, Turkish, Arabic, Vietnamese, Thai, Chinese, Hindi
- ✅ **Zero-shot Learning**: Direct transfer from English to target languages
- ✅ **Few-shot Learning**: 1-shot, 5-shot, and 10-shot experiments
- ✅ **Comprehensive Evaluation**: EM, F1, BLEU, ROUGE metrics
- ✅ **Statistical Analysis**: Significance testing and correlation analysis
- ✅ **Mac Optimization**: MPS support for Apple Silicon
- ✅ **Experiment Tracking**: Weights & Biases integration
- ✅ **Publication-ready Plots**: High-quality visualizations

## Installation

### Prerequisites

- Python 3.8+
- macOS (optimized for Apple Silicon with MPS support)
- 16GB+ RAM recommended
- 50GB+ disk space for models and datasets

### Setup

1. **Clone the repository**:
```bash
git clone <repository-url>
cd cross-lingual-qa-system
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Verify installation**:
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'MPS available: {torch.backends.mps.is_available()}')"
```

## Quick Start

### 1. Setup Data

The datasets (XQuAD and SQuAD 2.0) will be automatically downloaded when you run the experiments:

```bash
# Data will be cached in ./cache/ directory (excluded from git)
# XQuAD: ~141MB (11 languages)
# SQuAD 2.0: ~130MB
# Total: ~271MB (downloaded automatically)
```

**Note**: Cache files are excluded from the repository due to GitHub's 100MB file size limit. They will be downloaded automatically when you run experiments.

### 2. Data Exploration

Start with the data exploration notebook to understand the datasets:

```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

### 3. Run Zero-Shot Experiments

Train models on English SQuAD 2.0 and evaluate on all XQuAD languages:

```bash
# Using CLI script
python experiments/run_zero_shot.py --output-dir ./results --cache-dir ./cache

# Or using Jupyter notebook
jupyter notebook notebooks/04_zero_shot_experiments.ipynb
```

### 4. Run Few-Shot Experiments

Fine-tune models with few examples per language:

```bash
# Using CLI script
python experiments/run_few_shot.py --shots 1 5 10 --seeds 42 123 456

# Or using Jupyter notebook
jupyter notebook notebooks/05_few_shot_experiments.ipynb
```

### 4. Interactive Dashboard

Launch the Streamlit dashboard to visualize and analyze results:

```bash
# Quick launch
./run_dashboard.sh

# Or manually
source venv/bin/activate
streamlit run dashboard/app.py
```

Access the dashboard at: **http://localhost:8501**

**Dashboard Features:**
- 📈 **Overview**: Summary metrics and performance indicators
- 🌍 **Language Analysis**: Performance breakdown by language and language families
- 📊 **Model Comparison**: Direct comparison between mBERT and mT5
- 🔬 **Statistical Analysis**: Significance testing and statistical validation
- ⚙️ **Configuration**: View experiment settings and parameters

### 5. Analyze Results

Generate comprehensive analysis and visualizations:

```bash
jupyter notebook notebooks/06_comparative_analysis.ipynb
```

## Project Structure

```
cross-lingual-qa-system/
├── config/                          # Configuration files
│   ├── model_configs.yaml          # Model configurations
│   ├── training_configs.yaml       # Training parameters
│   └── experiment_configs.yaml     # Experiment settings
├── dashboard/                       # Interactive dashboard
│   ├── app.py                      # Main Streamlit application
│   ├── run_dashboard.py            # Dashboard runner script
│   ├── create_sample_data.py       # Sample data generator
│   └── README.md                   # Dashboard documentation
├── src/                            # Source code
│   ├── data/                       # Data handling
│   │   ├── loaders.py             # Dataset loaders
│   │   ├── preprocessors.py       # Text preprocessing
│   │   └── few_shot_sampler.py    # Few-shot sampling
│   ├── models/                     # Model implementations
│   │   ├── base_model.py          # Base model interface
│   │   ├── mbert_qa.py            # mBERT QA model
│   │   └── mt5_qa.py              # mT5 QA model
│   ├── training/                   # Training infrastructure
│   │   ├── trainer.py             # Unified trainer
│   │   ├── zero_shot.py           # Zero-shot training
│   │   └── few_shot.py            # Few-shot training
│   ├── evaluation/                 # Evaluation framework
│   │   ├── metrics.py             # Evaluation metrics
│   │   ├── evaluator.py           # Cross-lingual evaluator
│   │   └── analysis.py            # Statistical analysis
│   └── utils/                      # Utilities
│       ├── device_utils.py        # Device management
│       ├── logger.py              # Experiment logging
│       └── visualization.py       # Plotting utilities
├── notebooks/                      # Jupyter notebooks
│   ├── 01_data_exploration.ipynb  # Dataset exploration
│   ├── 04_zero_shot_experiments.ipynb  # Zero-shot experiments
│   ├── 05_few_shot_experiments.ipynb   # Few-shot experiments
│   └── 06_comparative_analysis.ipynb   # Results analysis
├── experiments/                    # CLI scripts
│   ├── run_zero_shot.py           # Zero-shot CLI
│   └── run_few_shot.py            # Few-shot CLI
├── models/                         # Saved model checkpoints
├── results/                        # Experiment results
│   ├── plots/                     # Generated plots
│   ├── tables/                    # Performance tables
│   └── reports/                   # Analysis reports
└── requirements.txt                # Dependencies
```

## Configuration

### Model Configuration (`config/model_configs.yaml`)

```yaml
models:
  mbert:
    name: "bert-base-multilingual-cased"
    max_length: 384
    num_labels: 2
  mt5:
    name: "google/mt5-base"
    max_length: 384
    max_target_length: 64
```

### Training Configuration (`config/training_configs.yaml`)

```yaml
zero_shot:
  train_batch_size: 16
  eval_batch_size: 32
  num_train_epochs: 3
  learning_rate: 3e-5

few_shot:
  train_batch_size: 8
  eval_batch_size: 16
  num_train_epochs: 10
  learning_rate: 1e-4
```

## Usage Examples

### Basic Zero-Shot Experiment

```python
from src.training.zero_shot import run_zero_shot_comparison
from src.utils.device_utils import DeviceManager

# Initialize device manager
device_manager = DeviceManager()

# Run zero-shot comparison
results = run_zero_shot_comparison(
    config={
        'model_type': 'mbert',
        'output_dir': './results',
        'cache_dir': './cache'
    },
    device_manager=device_manager
)
```

### Custom Few-Shot Experiment

```python
from src.training.few_shot import run_few_shot_experiment

# Run few-shot experiment
results = run_few_shot_experiment(
    model_type='mt5',
    model_name='google/mt5-base',
    config={
        'shots': [1, 5, 10],
        'seeds': [42, 123, 456],
        'sampling_strategy': 'random'
    }
)
```

### Evaluation

```python
from src.evaluation.evaluator import CrossLingualEvaluator
from src.models.base_model import ModelFactory

# Create model
model = ModelFactory.create_model('mbert', 'bert-base-multilingual-cased', config)
model.load_model()

# Evaluate
evaluator = CrossLingualEvaluator(model, config)
results = evaluator.evaluate_all_languages(['en', 'es', 'de', 'zh'])
```

## Results

### Expected Performance

Based on similar studies, expected performance ranges:

- **Zero-shot F1**: 35-45% average across languages
- **Few-shot F1 (10-shot)**: 45-55% average across languages
- **Best performing languages**: Spanish, German (closer to English)
- **Challenging languages**: Chinese, Arabic (distant from English)

### Key Findings

1. **mBERT vs mT5**: mBERT typically performs better for extractive QA
2. **Language Distance**: Performance correlates with linguistic distance from English
3. **Few-shot Learning**: Significant improvement with just 5-10 examples
4. **Sample Efficiency**: Diminishing returns beyond 10 shots

## Troubleshooting

### Common Issues

1. **MPS Not Available**:
   ```bash
   # Check MPS availability
   python -c "import torch; print(torch.backends.mps.is_available())"
   ```

2. **Memory Issues**:
   - Reduce batch size in configuration
   - Enable gradient checkpointing
   - Use CPU if GPU memory insufficient

3. **Dataset Loading Errors**:
   - Check internet connection
   - Verify cache directory permissions
   - Clear cache and retry

### Performance Optimization

1. **For Apple Silicon**:
   - Use MPS backend (automatic)
   - Enable mixed precision training
   - Use gradient accumulation

2. **For CPU**:
   - Reduce batch size
   - Use fewer workers
   - Enable model quantization

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Citation

If you use this code in your research, please cite:

```bibtex
@software{cross_lingual_qa_system,
  title={Cross-Lingual Question Answering System},
  author={Your Name},
  year={2024},
  url={https://github.com/your-repo/cross-lingual-qa-system}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [XQuAD Dataset](https://github.com/deepmind/xquad)
- [SQuAD Dataset](https://rajpurkar.github.io/SQuAD-explorer/)
- [PyTorch](https://pytorch.org/)

## Support

For questions and support:
- Create an issue on GitHub
- Check the documentation
- Review the example notebooks

---

**Note**: This implementation is optimized for macOS with Apple Silicon. For other platforms, you may need to adjust device configurations and memory settings.

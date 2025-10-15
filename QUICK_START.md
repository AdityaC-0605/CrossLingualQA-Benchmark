# Quick Start Guide

## 🚀 Get Started in 5 Minutes

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Test Installation
```bash
python test_installation.py
```

### 3. Explore Data
```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

### 4. Run Zero-Shot Experiments
```bash
python experiments/run_zero_shot.py --output-dir ./results
```

### 5. Run Few-Shot Experiments
```bash
python experiments/run_few_shot.py --shots 1 5 10
```

### 6. Analyze Results
```bash
jupyter notebook notebooks/06_comparative_analysis.ipynb
```

## 📊 Expected Results

- **Zero-shot F1**: 35-45% average across languages
- **Few-shot F1 (10-shot)**: 45-55% average across languages
- **Best languages**: Spanish, German (closer to English)
- **Challenging languages**: Chinese, Arabic (distant from English)

## 🎯 Key Features

- ✅ 11 languages (XQuAD dataset)
- ✅ Zero-shot and few-shot learning
- ✅ mBERT vs mT5 comparison
- ✅ Mac MPS optimization
- ✅ Publication-ready plots
- ✅ Statistical significance testing

## 📚 Documentation

- **README.md**: Complete setup guide
- **API_DOCUMENTATION.md**: Detailed API reference
- **PROJECT_SUMMARY.md**: Comprehensive project overview

## 🆘 Need Help?

1. Check the README.md for detailed instructions
2. Review the example notebooks
3. Run the test script to verify installation
4. Check the API documentation for usage examples

---

**Ready to start your cross-lingual QA research! 🎉**

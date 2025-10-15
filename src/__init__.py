"""
Cross-Lingual Question Answering System

A comprehensive implementation of cross-lingual question answering using
multilingual BERT (mBERT) and T5 (mT5) models.
"""

__version__ = "1.0.0"
__author__ = "Cross-Lingual QA Team"
__email__ = "contact@example.com"

# Import main components
from .models.base_model import BaseQAModel, ModelFactory
from .models.mbert_qa import MBertQAModel
from .models.mt5_qa import MT5QAModel

from .data.loaders import UnifiedDataLoader, XQuADLoader, SQuADLoader
from .data.preprocessors import DataPreprocessor
from .data.few_shot_sampler import FewShotSampler, CrossLingualFewShotSampler

from .training.trainer import QATrainer, ZeroShotTrainer, FewShotTrainer
from .training.zero_shot import run_zero_shot_experiment, run_zero_shot_comparison
from .training.few_shot import run_few_shot_experiment, run_few_shot_comparison

from .evaluation.metrics import QAMetrics, compute_qa_metrics
from .evaluation.evaluator import CrossLingualEvaluator, ComparativeEvaluator
from .evaluation.analysis import ResultAnalyzer, StatisticalAnalyzer, ResultVisualizer

from .utils.device_utils import DeviceManager
from .utils.logger import ExperimentLogger
from .utils.visualization import ResultVisualizer

__all__ = [
    # Models
    "BaseQAModel",
    "ModelFactory", 
    "MBertQAModel",
    "MT5QAModel",
    
    # Data
    "UnifiedDataLoader",
    "XQuADLoader",
    "SQuADLoader",
    "DataPreprocessor",
    "FewShotSampler",
    "CrossLingualFewShotSampler",
    
    # Training
    "QATrainer",
    "ZeroShotTrainer", 
    "FewShotTrainer",
    "run_zero_shot_experiment",
    "run_zero_shot_comparison",
    "run_few_shot_experiment",
    "run_few_shot_comparison",
    
    # Evaluation
    "QAMetrics",
    "compute_qa_metrics",
    "CrossLingualEvaluator",
    "ComparativeEvaluator",
    "ResultAnalyzer",
    "StatisticalAnalyzer",
    "ResultVisualizer",
    
    # Utils
    "DeviceManager",
    "ExperimentLogger",
    "ResultVisualizer"
]

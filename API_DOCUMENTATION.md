# API Documentation

## Cross-Lingual Question Answering System API

This document provides comprehensive API documentation for the cross-lingual QA system.

## Table of Contents

1. [Core Models](#core-models)
2. [Data Pipeline](#data-pipeline)
3. [Training Infrastructure](#training-infrastructure)
4. [Evaluation Framework](#evaluation-framework)
5. [Utilities](#utilities)
6. [CLI Interface](#cli-interface)

## Core Models

### BaseQAModel

Abstract base class for all QA models.

```python
from src.models.base_model import BaseQAModel

class BaseQAModel(ABC):
    def __init__(self, model_name: str, config: Dict[str, Any])
    def load_model(self) -> None
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]
    def predict(self, question: str, context: str) -> Dict[str, Any]
    def batch_predict(self, examples: List[Dict[str, str]]) -> List[Dict[str, Any]]
    def to_device(self, device: torch.device) -> None
    def save_model(self, output_dir: str) -> None
    def load_from_checkpoint(self, checkpoint_path: str) -> None
    def get_model_info(self) -> Dict[str, Any]
```

### MBertQAModel

Multilingual BERT model for question answering.

```python
from src.models.mbert_qa import MBertQAModel

model = MBertQAModel(
    model_name="bert-base-multilingual-cased",
    config={
        "max_length": 384,
        "num_labels": 2,
        "hidden_dropout_prob": 0.1
    }
)

# Load model
model.load_model()

# Make prediction
result = model.predict(
    question="What is the capital of France?",
    context="Paris is the capital and largest city of France."
)
# Returns: {"answer": "Paris", "confidence": 0.95, ...}

# Batch prediction
results = model.batch_predict([
    {"question": "Q1", "context": "C1"},
    {"question": "Q2", "context": "C2"}
])
```

### MT5QAModel

Multilingual T5 model for question answering.

```python
from src.models.mt5_qa import MT5QAModel

model = MT5QAModel(
    model_name="google/mt5-base",
    config={
        "max_length": 384,
        "max_target_length": 64,
        "num_beams": 4
    }
)

# Load model
model.load_model()

# Make prediction
result = model.predict(
    question="What is the capital of France?",
    context="Paris is the capital and largest city of France."
)
# Returns: {"answer": "Paris", "confidence": 0.92, ...}

# Generate with custom parameters
answer = model.generate_answer(
    question="Q",
    context="C",
    max_length=32,
    num_beams=8,
    temperature=0.7
)
```

### ModelFactory

Factory for creating model instances.

```python
from src.models.base_model import ModelFactory

# Create mBERT model
mbert_model = ModelFactory.create_model(
    model_type="mbert",
    model_name="bert-base-multilingual-cased",
    config=config
)

# Create mT5 model
mt5_model = ModelFactory.create_model(
    model_type="mt5",
    model_name="google/mt5-base",
    config=config
)

# Create from configuration file
model = ModelFactory.create_from_config("config/model_configs.yaml")
```

## Data Pipeline

### UnifiedDataLoader

Main data loader for XQuAD and SQuAD datasets.

```python
from src.data.loaders import UnifiedDataLoader

loader = UnifiedDataLoader(cache_dir="./cache")

# Load all datasets
data = loader.load_all_data()
# Returns: {"squad": {...}, "xquad": {...}}

# Get training data
train_data = loader.get_training_data(answerable_only=True)

# Get evaluation data
eval_data = loader.get_evaluation_data(languages=["en", "es", "de"])

# Get cross-lingual pairs
pairs = loader.get_cross_lingual_pairs(
    source_lang="en",
    target_languages=["es", "de", "zh"]
)
```

### DataPreprocessor

Text preprocessing and tokenization.

```python
from src.data.preprocessors import DataPreprocessor

preprocessor = DataPreprocessor(
    model_name="bert-base-multilingual-cased",
    model_type="bert",
    max_length=384
)

# Preprocess for training
train_dataset = preprocessor.preprocess_for_training(examples)

# Preprocess for evaluation
eval_dataset, qa_examples = preprocessor.preprocess_for_evaluation(examples)

# Create PyTorch dataset
dataset = preprocessor.create_dataset(qa_examples)
```

### FewShotSampler

Few-shot sampling strategies.

```python
from src.data.few_shot_sampler import FewShotSampler, SamplingConfig

config = SamplingConfig(
    num_shots=5,
    strategy="random",  # or "diverse", "stratified"
    seed=42
)

sampler = FewShotSampler(config)

# Sample examples
sampled = sampler.sample_examples(examples, language="es")

# Cross-lingual sampling
from src.data.few_shot_sampler import CrossLingualFewShotSampler

cross_sampler = CrossLingualFewShotSampler(config)
few_shot_datasets = cross_sampler.create_few_shot_datasets(
    language_data=data,
    shots=[1, 5, 10],
    seeds=[42, 123, 456]
)
```

## Training Infrastructure

### QATrainer

Unified trainer for QA models.

```python
from src.training.trainer import QATrainer

trainer = QATrainer(
    model=model,
    config=training_config,
    device_manager=device_manager
)

# Setup training
trainer.setup_training(
    train_dataloader=train_loader,
    eval_dataloader=eval_loader,
    num_training_steps=1000
)

# Train model
history = trainer.train(
    train_dataloader=train_loader,
    eval_dataloader=eval_loader,
    num_epochs=3
)

# Save model
trainer.save_model("./models/checkpoint")
```

### ZeroShotTrainer

Specialized trainer for zero-shot learning.

```python
from src.training.zero_shot import ZeroShotTrainer

trainer = ZeroShotTrainer(model, config, device_manager)

# Train for zero-shot
history = trainer.train_zero_shot(
    train_dataloader=train_loader,
    eval_dataloader=eval_loader
)
```

### FewShotTrainer

Specialized trainer for few-shot learning.

```python
from src.training.few_shot import FewShotTrainer

trainer = FewShotTrainer(model, config, device_manager)

# Train for few-shot
history = trainer.train_few_shot(
    train_dataloader=few_shot_loader,
    eval_dataloader=eval_loader,
    num_shots=5
)
```

## Evaluation Framework

### QAMetrics

Evaluation metrics for QA tasks.

```python
from src.evaluation.metrics import QAMetrics

metrics = QAMetrics(
    metrics=["exact_match", "f1", "bleu"],
    normalize=True
)

# Compute metrics
results = metrics.compute(predictions, references)
# Returns: {"exact_match": 0.75, "f1": 0.82, "bleu": 0.68}

# Per-sample metrics
per_sample = metrics.compute_per_sample(predictions, references)
```

### CrossLingualEvaluator

Cross-lingual evaluation framework.

```python
from src.evaluation.evaluator import CrossLingualEvaluator

evaluator = CrossLingualEvaluator(
    model=model,
    config=config,
    device_manager=device_manager
)

# Evaluate on all languages
results = evaluator.evaluate_all_languages()

# Evaluate on specific languages
results = evaluator.evaluate_all_languages(languages=["en", "es", "de"])

# Generate transfer matrix
matrix = evaluator.generate_transfer_matrix(results)

# Save results
evaluator.save_results(results, "./results/evaluation.json")
```

### ComparativeEvaluator

Model comparison evaluator.

```python
from src.evaluation.evaluator import ComparativeEvaluator

models = {
    "mbert": mbert_model,
    "mt5": mt5_model
}

comparator = ComparativeEvaluator(models, config, device_manager)

# Compare models
results = comparator.compare_models(languages=["en", "es", "de"])

# Save comparison results
comparator.save_comparison_results(results, "./results/comparison")
```

## Utilities

### DeviceManager

Device management for Mac MPS/CPU.

```python
from src.utils.device_utils import DeviceManager

device_manager = DeviceManager({
    "use_mps": True,
    "use_cpu": False,
    "mixed_precision": True
})

# Get device
device = device_manager.get_device()

# Get optimal batch size
batch_size = device_manager.get_optimal_batch_size(
    model_size_mb=500,
    sequence_length=384
)

# Optimize model for memory
optimized_model = device_manager.optimize_for_memory(model)

# Get memory usage
memory_info = device_manager.get_memory_usage()

# Clear cache
device_manager.clear_cache()
```

### ExperimentLogger

Experiment tracking and logging.

```python
from src.utils.logger import ExperimentLogger

logger = ExperimentLogger(
    project_name="cross-lingual-qa",
    experiment_name="mbert-vs-mt5",
    use_wandb=True,
    local_logging=True
)

# Log configuration
logger.log_config(config)

# Log metrics
logger.log_metrics({"f1": 0.82, "em": 0.75}, step=100)

# Log training step
logger.log_training_step(step=100, loss=0.25, learning_rate=3e-5)

# Log evaluation
logger.log_evaluation({"f1": 0.82}, language="es", step=100)

# Log cross-lingual results
logger.log_cross_lingual_results(results)

# Finish logging
logger.finish()
```

### ResultVisualizer

Visualization utilities.

```python
from src.utils.visualization import ResultVisualizer

visualizer = ResultVisualizer(
    output_dir="./results/plots",
    style="publication"
)

# Plot performance by language
plot_path = visualizer.plot_performance_by_language(
    results=results,
    metric="f1",
    title="Performance by Language"
)

# Plot model comparison
plot_path = visualizer.plot_model_comparison(
    comparison_results=results,
    metric="f1"
)

# Plot few-shot learning curve
plot_path = visualizer.plot_few_shot_learning_curve(
    few_shot_results=results,
    metric="f1"
)

# Create performance table
table_path = visualizer.create_performance_table(
    results=results,
    metric="f1"
)

# Create summary plot
plot_path = visualizer.create_summary_plot(
    zero_shot_results=zero_shot_results,
    few_shot_results=few_shot_results
)
```

## CLI Interface

### Zero-Shot Experiments

```bash
python experiments/run_zero_shot.py \
    --config ../config/experiment_configs.yaml \
    --model-config ../config/model_configs.yaml \
    --training-config ../config/training_configs.yaml \
    --output-dir ../results \
    --cache-dir ../cache \
    --languages en es de zh \
    --use-mps
```

### Few-Shot Experiments

```bash
python experiments/run_few_shot.py \
    --config ../config/experiment_configs.yaml \
    --model-config ../config/model_configs.yaml \
    --training-config ../config/training_configs.yaml \
    --output-dir ../results \
    --cache-dir ../cache \
    --shots 1 5 10 \
    --seeds 42 123 456 \
    --sampling-strategy random \
    --use-mps
```

## Configuration Examples

### Model Configuration

```yaml
models:
  mbert:
    name: "bert-base-multilingual-cased"
    model_type: "bert"
    max_length: 384
    num_labels: 2
    hidden_dropout_prob: 0.1
    attention_probs_dropout_prob: 0.1
    
  mt5:
    name: "google/mt5-base"
    model_type: "t5"
    max_length: 384
    max_target_length: 64
    num_beams: 4
    early_stopping: true

device:
  use_mps: true
  use_cpu: false
  mixed_precision: true
  gradient_checkpointing: true
  max_memory_gb: 16
```

### Training Configuration

```yaml
zero_shot:
  train_batch_size: 16
  eval_batch_size: 32
  gradient_accumulation_steps: 2
  num_train_epochs: 3
  learning_rate: 3e-5
  warmup_ratio: 0.1
  max_grad_norm: 1.0

few_shot:
  train_batch_size: 8
  eval_batch_size: 16
  gradient_accumulation_steps: 4
  num_train_epochs: 10
  learning_rate: 1e-4
  early_stopping_patience: 3
```

### Experiment Configuration

```yaml
languages:
  xquad_languages:
    - "en"
    - "es"
    - "de"
    - "el"
    - "ru"
    - "tr"
    - "ar"
    - "vi"
    - "th"
    - "zh"
    - "hi"

evaluation:
  metrics:
    - "exact_match"
    - "f1"
    - "bleu"
    - "rouge"
  
  significance_testing:
    method: "wilcoxon"
    alpha: 0.05
    multiple_comparison_correction: "bonferroni"

tracking:
  use_wandb: true
  project_name: "cross-lingual-qa"
  experiment_name: "mbert-vs-mt5"
  tags: ["cross-lingual", "qa", "mbert", "mt5"]
```

## Error Handling

### Common Exceptions

```python
# Model loading errors
try:
    model.load_model()
except Exception as e:
    print(f"Failed to load model: {e}")

# Device errors
try:
    model.to_device(device)
except RuntimeError as e:
    print(f"Device error: {e}")

# Data loading errors
try:
    data = loader.load_all_data()
except Exception as e:
    print(f"Data loading error: {e}")
```

### Best Practices

1. **Always check device availability**:
```python
device_manager = DeviceManager()
if device_manager.is_mps():
    print("Using MPS")
elif device_manager.is_cpu():
    print("Using CPU")
```

2. **Handle memory issues**:
```python
# Check memory usage
memory_info = device_manager.get_memory_usage()
if memory_info['allocated_gb'] > 10:
    device_manager.clear_cache()
```

3. **Validate configurations**:
```python
# Check required keys
required_keys = ['model_name', 'max_length', 'learning_rate']
for key in required_keys:
    if key not in config:
        raise ValueError(f"Missing required config key: {key}")
```

## Performance Tips

1. **Use appropriate batch sizes**:
```python
batch_size = device_manager.get_optimal_batch_size(
    model_size_mb=500,
    sequence_length=384
)
```

2. **Enable mixed precision**:
```python
config['mixed_precision'] = True
config['gradient_checkpointing'] = True
```

3. **Use gradient accumulation**:
```python
config['gradient_accumulation_steps'] = 4
config['train_batch_size'] = 8  # Effective batch size = 8 * 4 = 32
```

4. **Cache preprocessed data**:
```python
config['cache_dir'] = './cache'
config['overwrite_cache'] = False
```

## Troubleshooting

### Common Issues

1. **MPS not available**: Check PyTorch version and macOS version
2. **Memory errors**: Reduce batch size or enable gradient checkpointing
3. **Slow training**: Use mixed precision and appropriate batch sizes
4. **Dataset loading errors**: Check internet connection and cache permissions

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable debug logging
config['debug'] = True
config['logging_steps'] = 10
```

This API documentation provides comprehensive coverage of all components in the cross-lingual QA system. For more examples and use cases, refer to the Jupyter notebooks in the `notebooks/` directory.

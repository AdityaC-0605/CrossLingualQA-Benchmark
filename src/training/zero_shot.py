"""
Zero-shot training implementation for cross-lingual QA.
Handles training on source language and evaluation on target languages.
"""

import torch
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Any, Tuple
import logging
import json
from pathlib import Path
import time

from src.training.trainer import ZeroShotTrainer
from src.models.base_model import BaseQAModel
from src.data.loaders import UnifiedDataLoader
from src.data.preprocessors import DataPreprocessor
from src.utils.device_utils import DeviceManager

logger = logging.getLogger(__name__)


class ZeroShotExperiment:
    """Handles zero-shot learning experiments."""
    
    def __init__(self, 
                 model: BaseQAModel,
                 config: Dict[str, Any],
                 device_manager: Optional[DeviceManager] = None):
        """
        Initialize zero-shot experiment.
        
        Args:
            model: QA model to train
            config: Experiment configuration
            device_manager: Device manager instance
        """
        self.model = model
        self.config = config
        self.device_manager = device_manager or DeviceManager()
        
        # Initialize data components
        self.data_loader = UnifiedDataLoader(cache_dir=config.get('cache_dir', './cache'))
        self.preprocessor = DataPreprocessor(
            model.model_name,
            model_type=config.get('model_type', 'bert'),
            max_length=config.get('max_length', 384)
        )
        
        logger.info(f"Initialized zero-shot experiment for {model.model_name}")
    
    def run_experiment(self) -> Dict[str, Any]:
        """
        Run complete zero-shot experiment.
        
        Returns:
            Dictionary containing experiment results
        """
        logger.info("Starting zero-shot experiment")
        
        # Load training data (SQuAD 2.0)
        logger.info("Loading training data...")
        training_data = self.data_loader.get_training_data(answerable_only=True)
        
        # Load evaluation data (XQuAD)
        logger.info("Loading evaluation data...")
        evaluation_data = self.data_loader.get_evaluation_data()
        
        # Preprocess training data
        logger.info("Preprocessing training data...")
        train_dataset = self.preprocessor.preprocess_for_training(training_data)
        
        # Create data loaders
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.config.get('train_batch_size', 16),
            shuffle=True,
            num_workers=self.config.get('dataloader_num_workers', 4),
            pin_memory=self.config.get('dataloader_pin_memory', True)
        )
        
        # Train model
        logger.info("Training model...")
        trainer = ZeroShotTrainer(self.model, self.config, self.device_manager)
        training_history = trainer.train_zero_shot(train_dataloader)
        
        # Evaluate on all languages
        logger.info("Evaluating on all languages...")
        evaluation_results = self._evaluate_all_languages(evaluation_data)
        
        # Compile results
        results = {
            'training_history': training_history,
            'evaluation_results': evaluation_results,
            'model_info': self.model.get_model_info(),
            'config': self.config,
            'timestamp': time.time()
        }
        
        # Save results
        self._save_results(results)
        
        logger.info("Zero-shot experiment completed")
        return results
    
    def _evaluate_all_languages(self, evaluation_data: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Evaluate model on all languages.
        
        Args:
            evaluation_data: Dictionary mapping language codes to examples
            
        Returns:
            Dictionary containing evaluation results for each language
        """
        results = {}
        
        for language, examples in evaluation_data.items():
            logger.info(f"Evaluating on {language} ({len(examples)} examples)")
            
            try:
                # Preprocess evaluation data
                eval_dataset, qa_examples = self.preprocessor.preprocess_for_evaluation(examples)
                
                # Create data loader
                eval_dataloader = DataLoader(
                    eval_dataset,
                    batch_size=self.config.get('eval_batch_size', 32),
                    shuffle=False,
                    num_workers=self.config.get('dataloader_num_workers', 4),
                    pin_memory=self.config.get('dataloader_pin_memory', True)
                )
                
                # Evaluate
                eval_results = self._evaluate_language(eval_dataloader, qa_examples, language)
                results[language] = eval_results
                
            except Exception as e:
                logger.error(f"Failed to evaluate on {language}: {e}")
                results[language] = {'error': str(e)}
        
        return results
    
    def _evaluate_language(self, 
                          eval_dataloader: DataLoader,
                          qa_examples: List[Any],
                          language: str) -> Dict[str, Any]:
        """
        Evaluate model on a specific language.
        
        Args:
            eval_dataloader: Evaluation data loader
            qa_examples: Original QA examples
            language: Language code
            
        Returns:
            Dictionary containing evaluation metrics
        """
        self.model.eval_mode()
        
        predictions = []
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in eval_dataloader:
                # Move batch to device
                if self.device_manager.device is not None:
                    batch = {k: v.to(self.device_manager.device) for k, v in batch.items()}
                
                # Forward pass
                if hasattr(self.model, 'compute_loss'):
                    loss = self.model.compute_loss(**batch)
                    total_loss += loss.item()
                    num_batches += 1
                
                # Get predictions
                if hasattr(self.model, 'batch_predict'):
                    # Use batch prediction if available
                    batch_examples = []
                    for i in range(len(batch['input_ids'])):
                        # Extract question and context from batch
                        # This is a simplified version - in practice, you'd need to
                        # properly extract the original text
                        batch_examples.append({
                            'question': f"question_{i}",
                            'context': f"context_{i}"
                        })
                    
                    batch_predictions = self.model.batch_predict(batch_examples)
                    predictions.extend(batch_predictions)
        
        # Calculate metrics
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        # For now, return basic metrics
        # In a full implementation, you'd calculate EM, F1, etc.
        results = {
            'loss': avg_loss,
            'num_examples': len(qa_examples),
            'predictions': predictions[:10] if predictions else [],  # Sample predictions
            'language': language
        }
        
        return results
    
    def _save_results(self, results: Dict[str, Any]):
        """Save experiment results."""
        output_dir = self.config.get('output_dir', './results')
        output_path = Path(output_dir) / 'zero_shot_results'
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save main results
        results_file = output_path / f"{self.model.model_name.replace('/', '_')}_zero_shot.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save model
        model_dir = output_path / f"{self.model.model_name.replace('/', '_')}_model"
        self.model.save_model(str(model_dir))
        
        logger.info(f"Saved results to {output_path}")


def run_zero_shot_experiment(model_type: str,
                           model_name: str,
                           config: Dict[str, Any],
                           device_manager: Optional[DeviceManager] = None) -> Dict[str, Any]:
    """
    Run a zero-shot experiment.
    
    Args:
        model_type: Type of model ("mbert" or "mt5")
        model_name: Name of the model
        config: Experiment configuration
        device_manager: Device manager instance
        
    Returns:
        Experiment results
    """
    # Create model
    from ..models.base_model import ModelFactory
    model = ModelFactory.create_model(model_type, model_name, config)
    
    # Load model
    model.load_model()
    
    # Move to device
    if device_manager:
        model.to_device(device_manager.get_device())
    
    # Create experiment
    experiment = ZeroShotExperiment(model, config, device_manager)
    
    # Run experiment
    results = experiment.run_experiment()
    
    return results


def run_zero_shot_comparison(config: Dict[str, Any],
                           device_manager: Optional[DeviceManager] = None) -> Dict[str, Any]:
    """
    Run zero-shot comparison between mBERT and mT5.
    
    Args:
        config: Experiment configuration
        device_manager: Device manager instance
        
    Returns:
        Comparison results
    """
    logger.info("Starting zero-shot comparison between mBERT and mT5")
    
    results = {}
    
    # Run mBERT experiment
    logger.info("Running mBERT zero-shot experiment...")
    mbert_config = {**config, 'model_type': 'mbert'}
    mbert_results = run_zero_shot_experiment(
        'mbert', 
        'bert-base-multilingual-cased', 
        mbert_config, 
        device_manager
    )
    results['mbert'] = mbert_results
    
    # Run mT5 experiment
    logger.info("Running mT5 zero-shot experiment...")
    mt5_config = {**config, 'model_type': 'mt5'}
    mt5_results = run_zero_shot_experiment(
        'mt5', 
        'google/mt5-base', 
        mt5_config, 
        device_manager
    )
    results['mt5'] = mt5_results
    
    # Save comparison results
    output_dir = config.get('output_dir', './results')
    output_path = Path(output_dir) / 'zero_shot_comparison'
    output_path.mkdir(parents=True, exist_ok=True)
    
    comparison_file = output_path / 'zero_shot_comparison.json'
    with open(comparison_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"Saved comparison results to {comparison_file}")
    
    return results

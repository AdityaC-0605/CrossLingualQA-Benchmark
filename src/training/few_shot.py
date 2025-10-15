"""
Few-shot training implementation for cross-lingual QA.
Handles few-shot learning with different numbers of examples per language.
"""

import torch
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Any, Tuple
import logging
import json
from pathlib import Path
import time

from src.training.trainer import FewShotTrainer
from src.models.base_model import BaseQAModel
from src.data.loaders import UnifiedDataLoader
from src.data.preprocessors import DataPreprocessor
from src.data.few_shot_sampler import CrossLingualFewShotSampler, SamplingConfig
from src.utils.device_utils import DeviceManager

logger = logging.getLogger(__name__)


class FewShotExperiment:
    """Handles few-shot learning experiments."""
    
    def __init__(self, 
                 model: BaseQAModel,
                 config: Dict[str, Any],
                 device_manager: Optional[DeviceManager] = None):
        """
        Initialize few-shot experiment.
        
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
        
        # Initialize few-shot sampler
        sampling_config = SamplingConfig(
            num_shots=1,  # Will be updated per experiment
            strategy=config.get('sampling_strategy', 'random'),
            seed=config.get('seed', 42),
            max_samples_per_language=config.get('max_samples_per_language', 50)
        )
        self.sampler = CrossLingualFewShotSampler(sampling_config)
        
        logger.info(f"Initialized few-shot experiment for {model.model_name}")
    
    def run_experiment(self, 
                      shots: List[int] = [1, 5, 10],
                      seeds: List[int] = [42, 123, 456]) -> Dict[str, Any]:
        """
        Run complete few-shot experiment.
        
        Args:
            shots: List of shot numbers to test
            seeds: List of random seeds for reproducibility
            
        Returns:
            Dictionary containing experiment results
        """
        logger.info(f"Starting few-shot experiment with shots: {shots}, seeds: {seeds}")
        
        # Load evaluation data (XQuAD)
        logger.info("Loading evaluation data...")
        evaluation_data = self.data_loader.get_evaluation_data()
        
        # Create few-shot datasets
        logger.info("Creating few-shot datasets...")
        few_shot_datasets = self.sampler.create_few_shot_datasets(
            evaluation_data, shots, seeds
        )
        
        # Run experiments for each shot and seed combination
        results = {}
        
        for shot in shots:
            results[shot] = {}
            
            for seed in seeds:
                logger.info(f"Running {shot}-shot experiment with seed {seed}")
                
                # Get few-shot data for this shot and seed
                shot_data = few_shot_datasets[shot][seed]
                
                # Train and evaluate
                shot_results = self._run_shot_experiment(shot_data, evaluation_data, shot, seed)
                results[shot][seed] = shot_results
        
        # Compile final results
        final_results = {
            'few_shot_results': results,
            'model_info': self.model.get_model_info(),
            'config': self.config,
            'shots': shots,
            'seeds': seeds,
            'timestamp': time.time()
        }
        
        # Save results
        self._save_results(final_results)
        
        logger.info("Few-shot experiment completed")
        return final_results
    
    def _run_shot_experiment(self, 
                           shot_data: Dict[str, List[Dict[str, Any]]],
                           evaluation_data: Dict[str, List[Dict[str, Any]]],
                           shot: int,
                           seed: int) -> Dict[str, Any]:
        """
        Run experiment for a specific shot and seed.
        
        Args:
            shot_data: Few-shot training data
            evaluation_data: Full evaluation data
            shot: Number of shots
            seed: Random seed
            
        Returns:
            Experiment results
        """
        # Combine few-shot data from all languages
        combined_training_data = []
        for language, examples in shot_data.items():
            combined_training_data.extend(examples)
        
        if not combined_training_data:
            logger.warning(f"No training data for {shot}-shot experiment with seed {seed}")
            return {'error': 'No training data'}
        
        # Preprocess training data
        train_dataset = self.preprocessor.preprocess_for_training(combined_training_data)
        
        # Create training data loader
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.config.get('train_batch_size', 8),
            shuffle=True,
            num_workers=self.config.get('dataloader_num_workers', 2),
            pin_memory=self.config.get('dataloader_pin_memory', True)
        )
        
        # Train model
        logger.info(f"Training {shot}-shot model with seed {seed}...")
        trainer = FewShotTrainer(self.model, self.config, self.device_manager)
        training_history = trainer.train_few_shot(train_dataloader, num_shots=shot)
        
        # Evaluate on all languages
        logger.info(f"Evaluating {shot}-shot model on all languages...")
        evaluation_results = self._evaluate_all_languages(evaluation_data)
        
        return {
            'training_history': training_history,
            'evaluation_results': evaluation_results,
            'num_training_examples': len(combined_training_data),
            'shot': shot,
            'seed': seed
        }
    
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
                    batch_size=self.config.get('eval_batch_size', 16),
                    shuffle=False,
                    num_workers=self.config.get('dataloader_num_workers', 2),
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
        output_path = Path(output_dir) / 'few_shot_results'
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save main results
        results_file = output_path / f"{self.model.model_name.replace('/', '_')}_few_shot.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save model
        model_dir = output_path / f"{self.model.model_name.replace('/', '_')}_model"
        self.model.save_model(str(model_dir))
        
        logger.info(f"Saved results to {output_path}")


def run_few_shot_experiment(model_type: str,
                          model_name: str,
                          config: Dict[str, Any],
                          shots: List[int] = [1, 5, 10],
                          seeds: List[int] = [42, 123, 456],
                          device_manager: Optional[DeviceManager] = None) -> Dict[str, Any]:
    """
    Run a few-shot experiment.
    
    Args:
        model_type: Type of model ("mbert" or "mt5")
        model_name: Name of the model
        config: Experiment configuration
        shots: List of shot numbers to test
        seeds: List of random seeds for reproducibility
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
    experiment = FewShotExperiment(model, config, device_manager)
    
    # Run experiment
    results = experiment.run_experiment(shots, seeds)
    
    return results


def run_few_shot_comparison(config: Dict[str, Any],
                          shots: List[int] = [1, 5, 10],
                          seeds: List[int] = [42, 123, 456],
                          device_manager: Optional[DeviceManager] = None) -> Dict[str, Any]:
    """
    Run few-shot comparison between mBERT and mT5.
    
    Args:
        config: Experiment configuration
        shots: List of shot numbers to test
        seeds: List of random seeds for reproducibility
        device_manager: Device manager instance
        
    Returns:
        Comparison results
    """
    logger.info("Starting few-shot comparison between mBERT and mT5")
    
    results = {}
    
    # Run mBERT experiment
    logger.info("Running mBERT few-shot experiment...")
    mbert_config = {**config, 'model_type': 'mbert'}
    mbert_results = run_few_shot_experiment(
        'mbert', 
        'bert-base-multilingual-cased', 
        mbert_config, 
        shots, 
        seeds, 
        device_manager
    )
    results['mbert'] = mbert_results
    
    # Run mT5 experiment
    logger.info("Running mT5 few-shot experiment...")
    mt5_config = {**config, 'model_type': 'mt5'}
    mt5_results = run_few_shot_experiment(
        'mt5', 
        'google/mt5-base', 
        mt5_config, 
        shots, 
        seeds, 
        device_manager
    )
    results['mt5'] = mt5_results
    
    # Save comparison results
    output_dir = config.get('output_dir', './results')
    output_path = Path(output_dir) / 'few_shot_comparison'
    output_path.mkdir(parents=True, exist_ok=True)
    
    comparison_file = output_path / 'few_shot_comparison.json'
    with open(comparison_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"Saved comparison results to {comparison_file}")
    
    return results

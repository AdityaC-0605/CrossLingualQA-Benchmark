"""
Data loaders for XQuAD and SQuAD datasets.
Handles loading, preprocessing, and formatting of cross-lingual QA data.
"""

import os
from typing import Dict, List, Optional, Union, Any
from datasets import load_dataset, Dataset, DatasetDict
import pandas as pd
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class XQuADLoader:
    """Loader for XQuAD cross-lingual question answering dataset."""
    
    def __init__(self, languages: Optional[List[str]] = None, cache_dir: Optional[str] = None):
        """
        Initialize XQuAD loader.
        
        Args:
            languages: List of language codes to load. If None, loads all 11 languages.
            cache_dir: Directory to cache datasets
        """
        self.languages = languages or [
            'en', 'es', 'de', 'el', 'ru', 'tr', 'ar', 'vi', 'th', 'zh', 'hi'
        ]
        self.cache_dir = cache_dir or "./cache"
        self.datasets = {}
        
        logger.info(f"Initializing XQuAD loader for languages: {self.languages}")
    
    def load_all_languages(self) -> Dict[str, DatasetDict]:
        """
        Load XQuAD datasets for all specified languages.
        
        Returns:
            Dictionary mapping language codes to DatasetDict
        """
        for lang in self.languages:
            try:
                logger.info(f"Loading XQuAD dataset for language: {lang}")
                dataset = load_dataset("xquad", f"xquad.{lang}", cache_dir=self.cache_dir)
                self.datasets[lang] = dataset
                logger.info(f"Loaded {len(dataset['validation'])} examples for {lang}")
            except Exception as e:
                logger.error(f"Failed to load XQuAD dataset for {lang}: {e}")
                continue
        
        return self.datasets
    
    def get_language_dataset(self, language: str) -> Optional[DatasetDict]:
        """
        Get dataset for a specific language.
        
        Args:
            language: Language code
            
        Returns:
            DatasetDict for the language or None if not found
        """
        if language not in self.datasets:
            try:
                logger.info(f"Loading XQuAD dataset for language: {language}")
                dataset = load_dataset("xquad", f"xquad.{language}", cache_dir=self.cache_dir)
                self.datasets[language] = dataset
            except Exception as e:
                logger.error(f"Failed to load XQuAD dataset for {language}: {e}")
                return None
        
        return self.datasets[language]
    
    def get_all_examples(self, split: str = "validation") -> List[Dict[str, Any]]:
        """
        Get all examples from all languages combined.
        
        Args:
            split: Dataset split to use
            
        Returns:
            List of examples with language information added
        """
        all_examples = []
        
        for lang, dataset in self.datasets.items():
            if split in dataset:
                examples = dataset[split].to_list()
                # Add language information
                for example in examples:
                    example['language'] = lang
                all_examples.extend(examples)
        
        logger.info(f"Combined {len(all_examples)} examples from {len(self.datasets)} languages")
        return all_examples
    
    def get_language_examples(self, language: str, split: str = "validation") -> List[Dict[str, Any]]:
        """
        Get examples for a specific language.
        
        Args:
            language: Language code
            split: Dataset split to use
            
        Returns:
            List of examples for the language
        """
        dataset = self.get_language_dataset(language)
        if dataset is None or split not in dataset:
            return []
        
        examples = dataset[split].to_list()
        for example in examples:
            example['language'] = language
        
        return examples


class SQuADLoader:
    """Loader for SQuAD 2.0 dataset for English training."""
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize SQuAD loader.
        
        Args:
            cache_dir: Directory to cache datasets
        """
        self.cache_dir = cache_dir or "./cache"
        self.dataset = None
        
        logger.info("Initializing SQuAD 2.0 loader")
    
    def load_dataset(self) -> DatasetDict:
        """
        Load SQuAD 2.0 dataset.
        
        Returns:
            SQuAD 2.0 DatasetDict
        """
        if self.dataset is None:
            logger.info("Loading SQuAD 2.0 dataset")
            self.dataset = load_dataset("squad_v2", cache_dir=self.cache_dir)
            logger.info(f"Loaded SQuAD 2.0: {len(self.dataset['train'])} train, {len(self.dataset['validation'])} validation")
        
        return self.dataset
    
    def get_train_examples(self) -> List[Dict[str, Any]]:
        """Get training examples from SQuAD 2.0."""
        dataset = self.load_dataset()
        examples = dataset['train'].to_list()
        for example in examples:
            example['language'] = 'en'  # SQuAD is English
        return examples
    
    def get_validation_examples(self) -> List[Dict[str, Any]]:
        """Get validation examples from SQuAD 2.0."""
        dataset = self.load_dataset()
        examples = dataset['validation'].to_list()
        for example in examples:
            example['language'] = 'en'  # SQuAD is English
        return examples
    
    def filter_answerable_only(self, examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter to keep only answerable questions.
        
        Args:
            examples: List of examples
            
        Returns:
            Filtered list with only answerable questions
        """
        answerable = [ex for ex in examples if not ex.get('is_impossible', False)]
        logger.info(f"Filtered to {len(answerable)} answerable questions from {len(examples)} total")
        return answerable


class UnifiedDataLoader:
    """Unified loader that combines XQuAD and SQuAD datasets."""
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize unified data loader.
        
        Args:
            cache_dir: Directory to cache datasets
        """
        self.cache_dir = cache_dir or "./cache"
        self.xquad_loader = XQuADLoader(cache_dir=cache_dir)
        self.squad_loader = SQuADLoader(cache_dir=cache_dir)
        
        logger.info("Initialized unified data loader")
    
    def load_all_data(self) -> Dict[str, Any]:
        """
        Load all datasets (XQuAD and SQuAD).
        
        Returns:
            Dictionary containing all loaded datasets
        """
        logger.info("Loading all datasets...")
        
        # Load SQuAD 2.0 for training
        squad_data = {
            'train': self.squad_loader.get_train_examples(),
            'validation': self.squad_loader.get_validation_examples()
        }
        
        # Load XQuAD for evaluation
        xquad_data = self.xquad_loader.load_all_languages()
        
        return {
            'squad': squad_data,
            'xquad': xquad_data
        }
    
    def get_training_data(self, answerable_only: bool = True) -> List[Dict[str, Any]]:
        """
        Get training data from SQuAD 2.0.
        
        Args:
            answerable_only: Whether to filter to answerable questions only
            
        Returns:
            List of training examples
        """
        examples = self.squad_loader.get_train_examples()
        if answerable_only:
            examples = self.squad_loader.filter_answerable_only(examples)
        return examples
    
    def get_evaluation_data(self, languages: Optional[List[str]] = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get evaluation data from XQuAD.
        
        Args:
            languages: List of languages to get data for. If None, gets all languages.
            
        Returns:
            Dictionary mapping language codes to lists of examples
        """
        if languages is None:
            languages = self.xquad_loader.languages
        
        evaluation_data = {}
        for lang in languages:
            examples = self.xquad_loader.get_language_examples(lang)
            evaluation_data[lang] = examples
        
        return evaluation_data
    
    def get_cross_lingual_pairs(self, source_lang: str = 'en', target_languages: Optional[List[str]] = None) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
        """
        Get cross-lingual data pairs for evaluation.
        
        Args:
            source_lang: Source language (usually English)
            target_languages: List of target languages. If None, uses all XQuAD languages.
            
        Returns:
            Dictionary with source and target language data
        """
        if target_languages is None:
            target_languages = [lang for lang in self.xquad_loader.languages if lang != source_lang]
        
        # Get source language data (from SQuAD validation or XQuAD)
        if source_lang == 'en':
            source_data = self.squad_loader.get_validation_examples()
            source_data = self.squad_loader.filter_answerable_only(source_data)
        else:
            source_data = self.xquad_loader.get_language_examples(source_lang)
        
        # Get target language data
        target_data = {}
        for lang in target_languages:
            target_data[lang] = self.xquad_loader.get_language_examples(lang)
        
        return {
            'source': {source_lang: source_data},
            'target': target_data
        }


def load_datasets(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Load datasets based on configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary containing loaded datasets
    """
    cache_dir = config.get('cache_dir', './cache')
    languages = config.get('languages', None)
    
    loader = UnifiedDataLoader(cache_dir=cache_dir)
    
    if languages:
        loader.xquad_loader.languages = languages
    
    return loader.load_all_data()


def save_processed_data(data: Dict[str, Any], output_dir: str):
    """
    Save processed data to disk.
    
    Args:
        data: Processed data dictionary
        output_dir: Output directory
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for dataset_name, dataset_data in data.items():
        if isinstance(dataset_data, dict):
            for split_name, split_data in dataset_data.items():
                if isinstance(split_data, list):
                    # Save as JSON
                    import json
                    file_path = output_path / f"{dataset_name}_{split_name}.json"
                    with open(file_path, 'w', encoding='utf-8') as f:
                        json.dump(split_data, f, ensure_ascii=False, indent=2)
                    logger.info(f"Saved {len(split_data)} examples to {file_path}")


def load_processed_data(input_dir: str) -> Dict[str, Any]:
    """
    Load processed data from disk.
    
    Args:
        input_dir: Input directory
        
    Returns:
        Dictionary containing loaded data
    """
    import json
    from pathlib import Path
    
    input_path = Path(input_dir)
    data = {}
    
    for file_path in input_path.glob("*.json"):
        with open(file_path, 'r', encoding='utf-8') as f:
            file_data = json.load(f)
        
        # Parse filename to get dataset and split names
        filename = file_path.stem
        if '_' in filename:
            dataset_name, split_name = filename.split('_', 1)
            if dataset_name not in data:
                data[dataset_name] = {}
            data[dataset_name][split_name] = file_data
    
    logger.info(f"Loaded processed data from {input_dir}")
    return data

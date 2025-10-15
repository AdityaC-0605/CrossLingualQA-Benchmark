"""
Few-shot sampling strategies for cross-lingual QA.
Implements random, diverse, and stratified sampling approaches.
"""

import random
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict
import logging
from dataclasses import dataclass
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


@dataclass
class SamplingConfig:
    """Configuration for few-shot sampling."""
    num_shots: int
    strategy: str = "random"  # Options: random, diverse, stratified
    seed: int = 42
    max_samples_per_language: int = 50
    diversity_threshold: float = 0.3


class FewShotSampler:
    """Handles few-shot sampling for cross-lingual QA."""
    
    def __init__(self, config: SamplingConfig):
        """
        Initialize few-shot sampler.
        
        Args:
            config: Sampling configuration
        """
        self.config = config
        self.random_state = np.random.RandomState(config.seed)
        random.seed(config.seed)
        
        logger.info(f"Initialized few-shot sampler: {config.num_shots} shots, strategy={config.strategy}")
    
    def sample_examples(self, examples: List[Dict[str, Any]], language: str) -> List[Dict[str, Any]]:
        """
        Sample examples for a specific language.
        
        Args:
            examples: List of examples for the language
            language: Language code
            
        Returns:
            List of sampled examples
        """
        if len(examples) <= self.config.num_shots:
            logger.warning(f"Not enough examples for {language}: {len(examples)} <= {self.config.num_shots}")
            return examples
        
        # Limit the pool of examples to consider
        if len(examples) > self.config.max_samples_per_language:
            examples = self.random_state.choice(examples, self.config.max_samples_per_language, replace=False).tolist()
        
        if self.config.strategy == "random":
            return self._random_sampling(examples)
        elif self.config.strategy == "diverse":
            return self._diverse_sampling(examples)
        elif self.config.strategy == "stratified":
            return self._stratified_sampling(examples)
        else:
            raise ValueError(f"Unknown sampling strategy: {self.config.strategy}")
    
    def _random_sampling(self, examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Random sampling strategy."""
        return self.random_state.choice(examples, self.config.num_shots, replace=False).tolist()
    
    def _diverse_sampling(self, examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Diverse sampling strategy using TF-IDF similarity."""
        if len(examples) <= self.config.num_shots:
            return examples
        
        # Extract text features
        texts = []
        for example in examples:
            text = f"{example.get('question', '')} {example.get('context', '')}"
            texts.append(text)
        
        # Compute TF-IDF vectors
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        try:
            tfidf_matrix = vectorizer.fit_transform(texts)
        except ValueError:
            # Fallback to random sampling if TF-IDF fails
            logger.warning("TF-IDF vectorization failed, falling back to random sampling")
            return self._random_sampling(examples)
        
        # Start with a random example
        selected_indices = [self.random_state.randint(0, len(examples))]
        selected_examples = [examples[selected_indices[0]]]
        
        # Iteratively select examples that are most different from already selected ones
        while len(selected_examples) < self.config.num_shots:
            similarities = []
            
            for i in range(len(examples)):
                if i in selected_indices:
                    continue
                
                # Compute similarity with all selected examples
                max_similarity = 0
                for selected_idx in selected_indices:
                    similarity = cosine_similarity(
                        tfidf_matrix[i:i+1], 
                        tfidf_matrix[selected_idx:selected_idx+1]
                    )[0][0]
                    max_similarity = max(max_similarity, similarity)
                
                similarities.append((i, max_similarity))
            
            # Select example with minimum maximum similarity
            if similarities:
                similarities.sort(key=lambda x: x[1])
                selected_idx = similarities[0][0]
                selected_indices.append(selected_idx)
                selected_examples.append(examples[selected_idx])
            else:
                break
        
        return selected_examples
    
    def _stratified_sampling(self, examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Stratified sampling based on question types or answer lengths."""
        # Group examples by answer length (simple stratification)
        length_groups = defaultdict(list)
        
        for example in examples:
            answer = example.get('answers', {}).get('text', [''])[0] if 'answers' in example else example.get('answer', '')
            answer_length = len(answer.split())
            
            # Create length bins
            if answer_length <= 2:
                group = "short"
            elif answer_length <= 5:
                group = "medium"
            else:
                group = "long"
            
            length_groups[group].append(example)
        
        # Sample proportionally from each group
        selected_examples = []
        total_groups = len(length_groups)
        
        if total_groups == 0:
            return self._random_sampling(examples)
        
        examples_per_group = self.config.num_shots // total_groups
        remaining_examples = self.config.num_shots % total_groups
        
        for i, (group, group_examples) in enumerate(length_groups.items()):
            num_to_sample = examples_per_group
            if i < remaining_examples:
                num_to_sample += 1
            
            if len(group_examples) >= num_to_sample:
                sampled = self.random_state.choice(group_examples, num_to_sample, replace=False).tolist()
            else:
                sampled = group_examples
            
            selected_examples.extend(sampled)
        
        # If we don't have enough examples, fill with random sampling
        if len(selected_examples) < self.config.num_shots:
            remaining_needed = self.config.num_shots - len(selected_examples)
            remaining_examples = [ex for ex in examples if ex not in selected_examples]
            if remaining_examples:
                additional = self.random_state.choice(
                    remaining_examples, 
                    min(remaining_needed, len(remaining_examples)), 
                    replace=False
                ).tolist()
                selected_examples.extend(additional)
        
        return selected_examples[:self.config.num_shots]


class CrossLingualFewShotSampler:
    """Handles few-shot sampling across multiple languages."""
    
    def __init__(self, config: SamplingConfig):
        """
        Initialize cross-lingual few-shot sampler.
        
        Args:
            config: Sampling configuration
        """
        self.config = config
        self.sampler = FewShotSampler(config)
        
        logger.info(f"Initialized cross-lingual few-shot sampler")
    
    def sample_for_languages(self, 
                           language_data: Dict[str, List[Dict[str, Any]]], 
                           target_languages: Optional[List[str]] = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        Sample few-shot examples for multiple languages.
        
        Args:
            language_data: Dictionary mapping language codes to lists of examples
            target_languages: List of target languages. If None, uses all languages in language_data.
            
        Returns:
            Dictionary mapping language codes to sampled examples
        """
        if target_languages is None:
            target_languages = list(language_data.keys())
        
        sampled_data = {}
        
        for language in target_languages:
            if language not in language_data:
                logger.warning(f"No data found for language: {language}")
                continue
            
            examples = language_data[language]
            sampled_examples = self.sampler.sample_examples(examples, language)
            sampled_data[language] = sampled_examples
            
            logger.info(f"Sampled {len(sampled_examples)} examples for {language}")
        
        return sampled_data
    
    def create_few_shot_datasets(self, 
                               language_data: Dict[str, List[Dict[str, Any]]], 
                               shots: List[int], 
                               seeds: List[int]) -> Dict[str, Dict[str, Dict[str, List[Dict[str, Any]]]]]:
        """
        Create few-shot datasets for multiple shots and seeds.
        
        Args:
            language_data: Dictionary mapping language codes to lists of examples
            shots: List of shot numbers (e.g., [1, 5, 10])
            seeds: List of random seeds for reproducibility
            
        Returns:
            Nested dictionary: {shot: {seed: {language: examples}}}
        """
        few_shot_datasets = {}
        
        for shot in shots:
            few_shot_datasets[shot] = {}
            
            for seed in seeds:
                # Create new sampler with this seed
                config = SamplingConfig(
                    num_shots=shot,
                    strategy=self.config.strategy,
                    seed=seed,
                    max_samples_per_language=self.config.max_samples_per_language,
                    diversity_threshold=self.config.diversity_threshold
                )
                sampler = FewShotSampler(config)
                
                # Sample for all languages
                sampled_data = {}
                for language, examples in language_data.items():
                    sampled_examples = sampler.sample_examples(examples, language)
                    sampled_data[language] = sampled_examples
                
                few_shot_datasets[shot][seed] = sampled_data
                
                logger.info(f"Created {shot}-shot dataset with seed {seed} for {len(sampled_data)} languages")
        
        return few_shot_datasets


def create_few_shot_sampler(config: SamplingConfig) -> FewShotSampler:
    """
    Create a few-shot sampler instance.
    
    Args:
        config: Sampling configuration
        
    Returns:
        FewShotSampler instance
    """
    return FewShotSampler(config)


def create_cross_lingual_sampler(config: SamplingConfig) -> CrossLingualFewShotSampler:
    """
    Create a cross-lingual few-shot sampler instance.
    
    Args:
        config: Sampling configuration
        
    Returns:
        CrossLingualFewShotSampler instance
    """
    return CrossLingualFewShotSampler(config)


def sample_few_shot_examples(examples: List[Dict[str, Any]], 
                           num_shots: int, 
                           strategy: str = "random", 
                           seed: int = 42) -> List[Dict[str, Any]]:
    """
    Convenience function to sample few-shot examples.
    
    Args:
        examples: List of examples to sample from
        num_shots: Number of examples to sample
        strategy: Sampling strategy
        seed: Random seed
        
    Returns:
        List of sampled examples
    """
    config = SamplingConfig(
        num_shots=num_shots,
        strategy=strategy,
        seed=seed
    )
    sampler = FewShotSampler(config)
    return sampler.sample_examples(examples, "unknown")

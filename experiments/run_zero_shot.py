#!/usr/bin/env python3
"""
CLI script for running zero-shot cross-lingual QA experiments.
"""

import argparse
import yaml
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.training.zero_shot import run_zero_shot_comparison
from src.utils.device_utils import DeviceManager


def main():
    parser = argparse.ArgumentParser(description='Run zero-shot cross-lingual QA experiments')
    
    parser.add_argument('--config', type=str, default='config/experiment_configs.yaml',
                       help='Path to experiment configuration file')
    parser.add_argument('--model-config', type=str, default='config/model_configs.yaml',
                       help='Path to model configuration file')
    parser.add_argument('--training-config', type=str, default='config/training_configs.yaml',
                       help='Path to training configuration file')
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Output directory for results')
    parser.add_argument('--cache-dir', type=str, default='cache',
                       help='Cache directory for datasets')
    parser.add_argument('--languages', nargs='+', default=None,
                       help='List of languages to evaluate (default: all XQuAD languages)')
    parser.add_argument('--use-mps', action='store_true', default=True,
                       help='Use MPS (Apple Silicon GPU) if available')
    parser.add_argument('--use-cpu', action='store_true', default=False,
                       help='Force CPU usage')
    
    args = parser.parse_args()
    
    # Load configuration files
    with open(args.config, 'r') as f:
        experiment_config = yaml.safe_load(f)
    
    with open(args.model_config, 'r') as f:
        model_config = yaml.safe_load(f)
    
    with open(args.training_config, 'r') as f:
        training_config = yaml.safe_load(f)
    
    # Combine configurations
    config = {
        **model_config,
        **training_config['zero_shot'],
        **experiment_config,
        'output_dir': args.output_dir,
        'cache_dir': args.cache_dir,
        'use_mps': args.use_mps,
        'use_cpu': args.use_cpu
    }
    
    # Override languages if specified
    if args.languages:
        config['languages'] = args.languages
    
    # Initialize device manager
    device_manager = DeviceManager(config)
    print(f"Using device: {device_manager.get_device()}")
    
    # Run experiments
    print("Starting zero-shot experiments...")
    results = run_zero_shot_comparison(config, device_manager)
    
    print("Zero-shot experiments completed!")
    print(f"Results saved to: {args.output_dir}")


if __name__ == '__main__':
    main()

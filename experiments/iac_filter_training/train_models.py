#!/usr/bin/env python3
"""
Main training interface for Stage 2 - Model Training Approaches.

This script provides a unified interface to train both generative and encoder models
for Chef detection classification.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from iac_filter_training.generative_trainer import GenerativeTrainer
from iac_filter_training.encoder_trainer import EncoderTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_generative(args):
    """Train generative model (CodeLLaMA with LoRA)."""
    logger.info("Training generative model...")
    
    trainer = GenerativeTrainer(
        model_name=args.model_name,
        output_dir=args.output_dir
    )
    
    # Prepare datasets
    train_path = args.train_path
    val_path = args.val_path
    
    trainer.prepare_datasets(train_path, val_path)
    
    # Train the model
    trainer.train(
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        warmup_steps=args.warmup_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps
    )
    
    logger.info(f"Generative model training completed. Model saved to {args.output_dir}")

def train_encoder(args):
    """Train encoder model (CodeBERT/CodeT5)."""
    logger.info("Training encoder model...")
    
    trainer = EncoderTrainer(
        model_name=args.model_name,
        output_dir=args.output_dir
    )
    
    # Prepare datasets
    train_path = args.train_path
    val_path = args.val_path
    
    trainer.prepare_datasets(train_path, val_path)
    
    # Train the model
    trainer.train(
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        warmup_steps=args.warmup_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps
    )
    
    logger.info(f"Encoder model training completed. Model saved to {args.output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Train Chef detection models")
    parser.add_argument(
        "--approach", 
        choices=["generative", "encoder"], 
        required=True,
        help="Training approach: generative (CodeLLaMA) or encoder (CodeBERT)"
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="Model name (defaults to recommended models for each approach)"
    )
    parser.add_argument(
        "--train-path",
        default="experiments/iac_filter_training/data/formatted_dataset/chef_train.jsonl",
        help="Path to training data"
    )
    parser.add_argument(
        "--val-path", 
        default="experiments/iac_filter_training/data/formatted_dataset/chef_val.jsonl",
        help="Path to validation data"
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for trained model"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size for training"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Learning rate (defaults to recommended values)"
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=2,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=10,
        help="Number of warmup steps"
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=50,
        help="Save model every N steps"
    )
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=25,
        help="Evaluate model every N steps"
    )
    
    args = parser.parse_args()
    
    # Set defaults based on approach
    if args.model_name is None:
        if args.approach == "generative":
            args.model_name = "codellama/CodeLlama-7b-hf"
        else:
            args.model_name = "microsoft/codebert-base"
    
    if args.learning_rate is None:
        if args.approach == "generative":
            args.learning_rate = 5e-5
        else:
            args.learning_rate = 2e-5
    
    if args.output_dir is None:
        args.output_dir = f"experiments/iac_filter_training/models/{args.approach}"
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Train based on approach
    if args.approach == "generative":
        train_generative(args)
    else:
        train_encoder(args)

if __name__ == "__main__":
    main()

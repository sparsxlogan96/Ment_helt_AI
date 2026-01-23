#!/usr/bin/env python3
"""
Fine-tuning script for GPT-2 model on mental health data.
This script allows you to customize the GPT-2 model by training it on your mental health dataset.

Usage:
    python fine_tune_gpt2.py --data-dir mental_health_data --output-dir ./fine_tuned_model
"""

import argparse
import logging
import os
import glob
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    TextDataset,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_dataset(file_path, tokenizer, block_size=128):
    """Load and tokenize dataset for language modeling."""
    dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=block_size
    )
    return dataset


def prepare_training_data(data_dir, output_file="training_data.txt"):
    """
    Combine all text files from the data directory into a single training file.
    
    Args:
        data_dir: Directory containing .txt files with mental health data
        output_file: Path to the output combined file
    
    Returns:
        Path to the combined training file
    """
    if not os.path.exists(data_dir):
        raise ValueError(f"Data directory '{data_dir}' does not exist")
    
    filepaths = glob.glob(os.path.join(data_dir, "*.txt"))
    
    if not filepaths:
        raise ValueError(f"No .txt files found in '{data_dir}'")
    
    logger.info(f"Found {len(filepaths)} files in {data_dir}")
    
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for filepath in filepaths:
            logger.info(f"Adding content from {filepath}")
            with open(filepath, 'r', encoding='utf-8') as infile:
                content = infile.read()
                outfile.write(content + "\n\n")
    
    logger.info(f"Combined training data saved to {output_file}")
    return output_file


def fine_tune_gpt2(
    model_name="gpt2-medium",
    data_dir="mental_health_data",
    output_dir="./fine_tuned_model",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    save_steps=500,
    save_total_limit=2,
    block_size=128
):
    """
    Fine-tune GPT-2 model on custom mental health data.
    
    Args:
        model_name: Base model to fine-tune (gpt2, gpt2-medium, gpt2-large)
        data_dir: Directory containing training data
        output_dir: Directory to save the fine-tuned model
        num_train_epochs: Number of training epochs
        per_device_train_batch_size: Batch size per device
        save_steps: Save checkpoint every N steps
        save_total_limit: Maximum number of checkpoints to keep
        block_size: Maximum sequence length
    """
    
    logger.info(f"Loading tokenizer and model: {model_name}")
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    
    # GPT-2 doesn't have a pad token by default, set it to eos_token
    tokenizer.pad_token = tokenizer.eos_token
    
    # Prepare training data
    logger.info("Preparing training data...")
    training_file = prepare_training_data(data_dir, output_file="combined_training_data.txt")
    
    # Load dataset
    logger.info("Loading and tokenizing dataset...")
    train_dataset = load_dataset(training_file, tokenizer, block_size=block_size)
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # GPT-2 uses causal language modeling, not masked
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        logging_steps=100,
        logging_dir=f"{output_dir}/logs",
        learning_rate=5e-5,
        warmup_steps=100,
        weight_decay=0.01,
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )
    
    # Train
    logger.info("Starting fine-tuning...")
    trainer.train()
    
    # Save final model
    logger.info(f"Saving fine-tuned model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    logger.info("Fine-tuning completed successfully!")
    
    # Clean up temporary file
    if os.path.exists("combined_training_data.txt"):
        os.remove("combined_training_data.txt")


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune GPT-2 model on mental health data"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="gpt2-medium",
        help="Base model to fine-tune (gpt2, gpt2-medium, gpt2-large)"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="mental_health_data",
        help="Directory containing training .txt files"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./fine_tuned_model",
        help="Directory to save the fine-tuned model"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Training batch size per device"
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=128,
        help="Maximum sequence length"
    )
    
    args = parser.parse_args()
    
    fine_tune_gpt2(
        model_name=args.model_name,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        block_size=args.block_size
    )


if __name__ == "__main__":
    main()

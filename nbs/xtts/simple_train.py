#!/usr/bin/env python3
"""
Simple XTTS Training Script with PyTorch Lightning.

This is a simplified training script for XTTS that uses PyTorch Lightning.
It's designed to be easy to modify and extend for custom training scenarios.

Usage:
    python simple_train.py --data_path /path/to/data --output_path /path/to/output

Example:
    python simple_train.py \
        --data_path /data/ljspeech \
        --meta_file /data/ljspeech/metadata.csv \
        --output_path ./output \
        --batch_size 4 \
        --grad_accum 64 \
        --max_epochs 100
"""

import argparse
import os
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.layers.xtts.trainer.gpt_lightning import (
    XTTSDataModule,
    XTTSLightningConfig,
    XTTSLightningModule,
)
from TTS.utils.manage import ModelManager


def download_pretrained_models(output_dir: Path) -> dict:
    """Download pretrained XTTS v2.0 model files."""
    checkpoints_dir = output_dir / "pretrained"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    # URLs for pretrained models
    model_files = {
        "dvae": "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/dvae.pth",
        "mel_stats": "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/mel_stats.pth",
        "vocab": "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/vocab.json",
        "model": "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/model.pth",
    }

    paths = {}
    for name, url in model_files.items():
        local_path = checkpoints_dir / os.path.basename(url)
        paths[name] = str(local_path)

        if not local_path.exists():
            print(f" > Downloading {name}...")
            ModelManager._download_model_files([url], str(checkpoints_dir), progress_bar=True)

    return paths


def create_config(args, pretrained_paths: dict) -> XTTSLightningConfig:
    """Create training configuration from command line arguments."""
    return XTTSLightningConfig(
        # Model paths
        dvae_checkpoint=pretrained_paths["dvae"],
        xtts_checkpoint=pretrained_paths["model"] if not args.train_from_scratch else "",
        tokenizer_file=pretrained_paths["vocab"],
        mel_norm_file=pretrained_paths["mel_stats"],

        # Model architecture
        gpt_num_audio_tokens=1026,
        gpt_start_audio_token=1024,
        gpt_stop_audio_token=1025,
        gpt_use_masking_gt_prompt_approach=True,
        gpt_use_perceiver_resampler=True,

        # Training data limits
        max_conditioning_length=132300,  # 6 secs
        min_conditioning_length=66150,   # 3 secs
        max_wav_length=args.max_audio_length,
        max_text_length=args.max_text_length,

        # Audio config
        sample_rate=22050,
        dvae_sample_rate=22050,
        output_sample_rate=24000,

        # Training params
        lr=args.learning_rate,
        optimizer="AdamW",
        optimizer_params={"betas": [0.9, 0.96], "eps": 1e-8, "weight_decay": args.weight_decay},
        optimizer_wd_only_on_weights=True,
        lr_scheduler="MultiStepLR",
        lr_scheduler_params={
            "milestones": [int(m) for m in args.lr_milestones.split(",")],
            "gamma": 0.5,
        },

        # Data loading
        batch_size=args.batch_size,
        eval_batch_size=args.batch_size,
        num_loader_workers=args.num_workers,
        num_eval_loader_workers=args.num_workers // 2 or 1,

        # Test sentences (optional)
        test_sentences=[],
    )


def main():
    parser = argparse.ArgumentParser(description="Train XTTS with PyTorch Lightning")

    # Data arguments
    parser.add_argument("--data_path", type=str, required=True, help="Path to dataset directory")
    parser.add_argument("--meta_file", type=str, required=True, help="Path to metadata file")
    parser.add_argument("--output_path", type=str, default="./output", help="Output directory for checkpoints")
    parser.add_argument("--language", type=str, default="en", help="Dataset language code")
    parser.add_argument("--formatter", type=str, default="ljspeech", help="Dataset formatter name")

    # Training arguments
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per GPU")
    parser.add_argument("--grad_accum", type=int, default=64, help="Gradient accumulation steps")
    parser.add_argument("--max_epochs", type=int, default=100, help="Maximum training epochs")
    parser.add_argument("--learning_rate", type=float, default=5e-6, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--lr_milestones", type=str, default="900000,2700000,5400000", help="LR scheduler milestones (comma-separated)")
    parser.add_argument("--precision", type=str, default="16-mixed", choices=["32", "16-mixed", "bf16-mixed"], help="Training precision")
    parser.add_argument("--gradient_clip", type=float, default=1.0, help="Gradient clipping value")

    # Data arguments
    parser.add_argument("--max_audio_length", type=int, default=255995, help="Max audio length in samples (~11.6s at 22050)")
    parser.add_argument("--max_text_length", type=int, default=200, help="Max text length in tokens")
    parser.add_argument("--eval_split_size", type=float, default=0.01, help="Fraction of data for evaluation")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of data loading workers")

    # Hardware arguments
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs to use")
    parser.add_argument("--strategy", type=str, default="auto", help="Training strategy (ddp, fsdp, auto)")

    # Logging arguments
    parser.add_argument("--logger", type=str, default="tensorboard", choices=["tensorboard", "wandb"], help="Logger type")
    parser.add_argument("--project_name", type=str, default="xtts-training", help="Project name for logging")
    parser.add_argument("--run_name", type=str, default="xtts-run", help="Run name for logging")

    # Checkpoint arguments
    parser.add_argument("--resume_from", type=str, default=None, help="Resume from checkpoint path")
    parser.add_argument("--train_from_scratch", action="store_true", help="Train from scratch instead of fine-tuning")
    parser.add_argument("--save_every_n_steps", type=int, default=10000, help="Save checkpoint every N steps")
    parser.add_argument("--save_top_k", type=int, default=3, help="Keep top K checkpoints")

    # Early stopping
    parser.add_argument("--early_stopping", action="store_true", help="Enable early stopping")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience (epochs)")

    args = parser.parse_args()

    # Set seed for reproducibility
    pl.seed_everything(42)

    # Create output directory
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Download pretrained models
    print(" > Checking/downloading pretrained models...")
    pretrained_paths = download_pretrained_models(output_path)

    # Create configuration
    print(" > Creating configuration...")
    config = create_config(args, pretrained_paths)

    # Define dataset config
    dataset_config = BaseDatasetConfig(
        formatter=args.formatter,
        dataset_name="custom",
        path=args.data_path,
        meta_file_train=args.meta_file,
        language=args.language,
    )

    # Initialize the model
    print(" > Initializing XTTS Lightning module...")
    model = XTTSLightningModule(config)

    # Load training samples
    print(" > Loading training samples...")
    train_samples, eval_samples = load_tts_samples(
        [dataset_config],
        eval_split=True,
        eval_split_max_size=256,
        eval_split_size=args.eval_split_size,
    )
    print(f" > Found {len(train_samples)} training samples and {len(eval_samples)} eval samples")

    # Create data module
    data_module = XTTSDataModule(
        config=config,
        train_samples=train_samples,
        eval_samples=eval_samples,
        tokenizer=model.xtts.tokenizer,
    )

    # Setup logger
    if args.logger == "wandb":
        logger = WandbLogger(
            name=args.run_name,
            project=args.project_name,
            save_dir=str(output_path),
        )
    else:
        logger = TensorBoardLogger(
            save_dir=str(output_path),
            name=args.run_name,
        )

    # Setup callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=str(output_path / "checkpoints"),
            filename="xtts-{epoch:02d}-{step:06d}-{val/loss:.4f}",
            monitor="val/loss",
            mode="min",
            save_top_k=args.save_top_k,
            save_last=True,
            every_n_train_steps=args.save_every_n_steps,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    # Try to add rich progress bar if available
    try:
        callbacks.append(RichProgressBar())
    except ImportError:
        pass

    # Add early stopping if requested
    if args.early_stopping:
        callbacks.append(
            EarlyStopping(
                monitor="val/loss",
                mode="min",
                patience=args.patience,
                verbose=True,
            )
        )

    # Determine strategy
    strategy = args.strategy
    if args.gpus > 1 and strategy == "auto":
        strategy = "ddp"

    # Create trainer
    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=args.gpus if torch.cuda.is_available() else 1,
        strategy=strategy,
        precision=args.precision,
        max_epochs=args.max_epochs,
        accumulate_grad_batches=args.grad_accum,
        gradient_clip_val=args.gradient_clip,
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=50,
        val_check_interval=1.0,
        enable_progress_bar=True,
        enable_model_summary=True,
    )

    # Start training
    print(" > Starting training...")
    print(f"   - Batch size: {args.batch_size}")
    print(f"   - Gradient accumulation: {args.grad_accum}")
    print(f"   - Effective batch size: {args.batch_size * args.grad_accum * args.gpus}")
    print(f"   - Learning rate: {args.learning_rate}")
    print(f"   - Precision: {args.precision}")
    print(f"   - GPUs: {args.gpus}")
    print()

    trainer.fit(model, data_module, ckpt_path=args.resume_from)

    print(" > Training complete!")
    print(f" > Best model saved to: {trainer.checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    main()

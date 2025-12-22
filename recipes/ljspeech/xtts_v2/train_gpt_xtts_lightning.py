"""
XTTS GPT Training Script using PyTorch Lightning.

This script trains the XTTS GPT model using PyTorch Lightning instead of the
deprecated Coqui Trainer. It provides all the benefits of Lightning including:
- Automatic multi-GPU training
- Mixed precision training
- Checkpointing and logging
- Gradient accumulation
- Learning rate scheduling
"""

import os
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.layers.xtts.trainer.gpt_lightning import (
    XTTSDataModule,
    XTTSLightningConfig,
    XTTSLightningModule,
)
from TTS.utils.manage import ModelManager


# Logging parameters
RUN_NAME = "xtts-ljspeech-lightning"
PROJECT_NAME = "xtts-ljspeech"
LOGGER_TYPE = "tensorboard"  # "tensorboard" or "wandb"

# Set here the path that the checkpoints will be saved
OUT_PATH = Path("../../../../../models/xtts/ljspeech_lightning/")
OUT_PATH.mkdir(parents=True, exist_ok=True)

# Training Parameters
BATCH_SIZE = 3
GRAD_ACCUM_STEPS = 84  # BATCH_SIZE * GRAD_ACCUM_STEPS should be ~252
PRECISION = "16-mixed"  # "32", "16-mixed", or "bf16-mixed"
MAX_EPOCHS = 100
NUM_GPUS = 1  # Set to number of GPUs available

# Define the dataset
config_dataset = BaseDatasetConfig(
    formatter="ljspeech",
    dataset_name="ljspeech",
    path="/home/jjs/proj/tts-castalk/data/ljspeech/",
    meta_file_train="/home/jjs/proj/tts-castalk/data/ljspeech/metadata.csv",
    language="en",
)

DATASETS_CONFIG_LIST = [config_dataset]

# Define the path where XTTS v2.0.1 files will be downloaded
CHECKPOINTS_OUT_PATH = OUT_PATH / "XTTS_v2.0_original_model_files"
CHECKPOINTS_OUT_PATH.mkdir(parents=True, exist_ok=True)

# DVAE files
DVAE_CHECKPOINT_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/dvae.pth"
MEL_NORM_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/mel_stats.pth"

DVAE_CHECKPOINT = str(CHECKPOINTS_OUT_PATH / "dvae.pth")
MEL_NORM_FILE = str(CHECKPOINTS_OUT_PATH / "mel_stats.pth")

# Download DVAE files if needed
if not os.path.isfile(DVAE_CHECKPOINT) or not os.path.isfile(MEL_NORM_FILE):
    print(" > Downloading DVAE files!")
    ModelManager._download_model_files(
        [MEL_NORM_LINK, DVAE_CHECKPOINT_LINK],
        str(CHECKPOINTS_OUT_PATH),
        progress_bar=True,
    )

# XTTS v2.0 checkpoint files
TOKENIZER_FILE_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/vocab.json"
XTTS_CHECKPOINT_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/model.pth"

TOKENIZER_FILE = str(CHECKPOINTS_OUT_PATH / "vocab.json")
XTTS_CHECKPOINT = str(CHECKPOINTS_OUT_PATH / "model.pth")

# Download XTTS v2.0 files if needed
if not os.path.isfile(TOKENIZER_FILE) or not os.path.isfile(XTTS_CHECKPOINT):
    print(" > Downloading XTTS v2.0 files!")
    ModelManager._download_model_files(
        [TOKENIZER_FILE_LINK, XTTS_CHECKPOINT_LINK],
        str(CHECKPOINTS_OUT_PATH),
        progress_bar=True,
    )

# Speaker reference for test sentences
SPEAKER_REFERENCE = [
    "/home/jjs/proj/tts-castalk/data/ljspeech/wavs/LJ001-0002.wav"
]

LANGUAGE = config_dataset.language


def main():
    # Set seed for reproducibility
    pl.seed_everything(42)

    # Create the configuration
    config = XTTSLightningConfig(
        # Model paths
        dvae_checkpoint=DVAE_CHECKPOINT,
        xtts_checkpoint=XTTS_CHECKPOINT,
        tokenizer_file=TOKENIZER_FILE,
        mel_norm_file=MEL_NORM_FILE,

        # Model architecture (using XTTS v2.0 defaults)
        gpt_num_audio_tokens=1026,
        gpt_start_audio_token=1024,
        gpt_stop_audio_token=1025,
        gpt_use_masking_gt_prompt_approach=True,
        gpt_use_perceiver_resampler=True,

        # Training data limits
        max_conditioning_length=132300,  # 6 secs
        min_conditioning_length=66150,   # 3 secs
        max_wav_length=255995,           # ~11.6 seconds
        max_text_length=200,

        # Audio config
        sample_rate=22050,
        dvae_sample_rate=22050,
        output_sample_rate=24000,

        # Training params
        lr=5e-06,
        optimizer="AdamW",
        optimizer_params={"betas": [0.9, 0.96], "eps": 1e-8, "weight_decay": 1e-2},
        optimizer_wd_only_on_weights=True,
        lr_scheduler="MultiStepLR",
        lr_scheduler_params={
            "milestones": [50000 * 18, 150000 * 18, 300000 * 18],
            "gamma": 0.5,
        },

        # Data loading
        batch_size=BATCH_SIZE,
        eval_batch_size=BATCH_SIZE,
        num_loader_workers=8,
        num_eval_loader_workers=4,

        # Test sentences
        test_sentences=[
            {
                "text": "It took me quite a long time to develop a voice, and now that I have it I'm not going to be silent.",
                "speaker_wav": SPEAKER_REFERENCE,
                "language": LANGUAGE,
            },
            {
                "text": "This cake is great. It's so delicious and moist.",
                "speaker_wav": SPEAKER_REFERENCE,
                "language": LANGUAGE,
            },
        ],
    )

    # Initialize the model
    print(" > Initializing XTTS Lightning module...")
    model = XTTSLightningModule(config)

    # Load training samples
    print(" > Loading training samples...")
    train_samples, eval_samples = load_tts_samples(
        DATASETS_CONFIG_LIST,
        eval_split=True,
        eval_split_max_size=256,
        eval_split_size=0.01,
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
    if LOGGER_TYPE == "wandb":
        logger = WandbLogger(
            name=RUN_NAME,
            project=PROJECT_NAME,
            save_dir=str(OUT_PATH),
        )
    else:
        logger = TensorBoardLogger(
            save_dir=str(OUT_PATH),
            name=RUN_NAME,
        )

    # Setup callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=str(OUT_PATH / "checkpoints"),
            filename="xtts-{epoch:02d}-{val/loss:.4f}",
            monitor="val/loss",
            mode="min",
            save_top_k=3,
            save_last=True,
            every_n_train_steps=10000,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    # Create trainer
    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=NUM_GPUS if torch.cuda.is_available() else 1,
        strategy="ddp" if NUM_GPUS > 1 else "auto",
        precision=PRECISION,
        max_epochs=MAX_EPOCHS,
        accumulate_grad_batches=GRAD_ACCUM_STEPS,
        gradient_clip_val=1.0,
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=50,
        val_check_interval=1.0,  # Validate every epoch
        enable_progress_bar=True,
        enable_model_summary=True,
    )

    # Start training
    print(" > Starting training...")
    trainer.fit(model, data_module)

    print(" > Training complete!")


if __name__ == "__main__":
    main()

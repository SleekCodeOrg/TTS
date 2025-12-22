"""
XTTS Trainer module.

This module contains training utilities for XTTS models, including:
- GPTTrainer: Legacy Coqui Trainer-based implementation
- XTTSLightningModule: PyTorch Lightning-based implementation (recommended)
- XTTSDataModule: PyTorch Lightning DataModule for XTTS
- XTTSDataset: Dataset class for XTTS training
"""

from TTS.tts.layers.xtts.trainer.dataset import XTTSDataset
from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTArgs, GPTTrainer, GPTTrainerConfig

# PyTorch Lightning support (optional - requires pytorch-lightning)
try:
    from TTS.tts.layers.xtts.trainer.gpt_lightning import (
        XTTSDataModule,
        XTTSLightningConfig,
        XTTSLightningModule,
    )

    __all__ = [
        # Legacy trainer
        "GPTTrainer",
        "GPTTrainerConfig",
        "GPTArgs",
        # Dataset
        "XTTSDataset",
        # Lightning (recommended)
        "XTTSLightningModule",
        "XTTSLightningConfig",
        "XTTSDataModule",
    ]
except ImportError:
    # PyTorch Lightning not installed
    __all__ = [
        "GPTTrainer",
        "GPTTrainerConfig",
        "GPTArgs",
        "XTTSDataset",
    ]

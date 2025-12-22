"""
PyTorch Lightning module for XTTS GPT training.

This module provides a Lightning-based training implementation for the XTTS model,
replacing the previous Coqui Trainer-based approach.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import DataLoader

from TTS.tts.layers.tortoise.arch_utils import TorchMelSpectrogram
from TTS.tts.layers.xtts.dvae import DiscreteVAE
from TTS.tts.layers.xtts.tokenizer import VoiceBpeTokenizer
from TTS.tts.layers.xtts.trainer.dataset import XTTSDataset
from TTS.tts.models.xtts import Xtts, XttsArgs, XttsAudioConfig
from TTS.utils.io import load_fsspec


@dataclass
class XTTSLightningConfig:
    """Configuration for XTTS Lightning training."""

    # Model paths
    dvae_checkpoint: str = ""
    xtts_checkpoint: str = ""
    gpt_checkpoint: str = ""
    tokenizer_file: str = ""
    mel_norm_file: str = "https://coqui.gateway.scarf.sh/v0.14.0_models/mel_norms.pth"
    pretrained_vocoder_checkpoint: str = ""

    # Model architecture
    gpt_layers: int = 30
    gpt_n_model_channels: int = 1024
    gpt_n_heads: int = 16
    gpt_max_audio_tokens: int = 605
    gpt_max_text_tokens: int = 402
    gpt_max_prompt_tokens: int = 70
    gpt_num_audio_tokens: int = 8194
    gpt_start_audio_token: int = 8192
    gpt_stop_audio_token: int = 8193
    gpt_code_stride_len: int = 1024
    gpt_use_masking_gt_prompt_approach: bool = True
    gpt_use_perceiver_resampler: bool = False
    kv_cache: bool = True

    # Training data
    min_conditioning_length: int = 66150
    max_conditioning_length: int = 132300
    max_wav_length: int = 255995  # ~11.6 seconds
    max_text_length: int = 200
    debug_loading_failures: bool = False

    # Loss weights
    gpt_loss_text_ce_weight: float = 0.01
    gpt_loss_mel_ce_weight: float = 1.0

    # Audio config
    sample_rate: int = 22050
    dvae_sample_rate: int = 22050
    output_sample_rate: int = 24000

    # Training params
    lr: float = 5e-06
    optimizer: str = "AdamW"
    optimizer_params: dict = field(default_factory=lambda: {"betas": [0.9, 0.96], "eps": 1e-8, "weight_decay": 1e-2})
    optimizer_wd_only_on_weights: bool = False
    lr_scheduler: str = "MultiStepLR"
    lr_scheduler_params: dict = field(default_factory=lambda: {"milestones": [50000, 150000, 300000], "gamma": 0.5})

    # Data loading
    batch_size: int = 4
    eval_batch_size: int = 4
    num_loader_workers: int = 4
    num_eval_loader_workers: int = 4
    training_seed: int = 1

    # Test sentences for evaluation
    test_sentences: List[dict] = field(default_factory=lambda: [])


class XTTSLightningModule(pl.LightningModule):
    """
    PyTorch Lightning module for XTTS GPT training.

    This encapsulates the entire XTTS model (GPT, DVAE, HiFiGAN) for training
    with PyTorch Lightning, providing automatic distributed training,
    mixed precision, and logging support.
    """

    def __init__(self, config: XTTSLightningConfig):
        """
        Initialize the XTTS Lightning module.

        Args:
            config: XTTSLightningConfig with all training parameters
        """
        super().__init__()
        self.save_hyperparameters()
        self.config = config

        # Build XttsArgs for the underlying model
        self._build_xtts_config()

        # Initialize the XTTS model
        self.xtts = Xtts(self.xtts_config)

        # Create the tokenizer with the target vocabulary
        self.xtts.tokenizer = VoiceBpeTokenizer(config.tokenizer_file)

        # Initialize GPT and HiFiGAN
        self.xtts.init_models()

        # Load checkpoints
        self._load_checkpoints()

        # Set mel stats
        if config.mel_norm_file:
            self.xtts.mel_stats = load_fsspec(config.mel_norm_file)

        # Initialize mel spectrogram extractors
        self._init_mel_extractors()

        # Load and freeze DVAE
        self._init_dvae()

        # Freeze vocoder if using pretrained
        if config.pretrained_vocoder_checkpoint and not config.xtts_checkpoint:
            self._freeze_vocoder()

    def _build_xtts_config(self):
        """Build the XttsConfig from our lightning config."""
        from TTS.tts.configs.xtts_config import XttsConfig

        # Create model args
        model_args = XttsArgs(
            gpt_layers=self.config.gpt_layers,
            gpt_n_model_channels=self.config.gpt_n_model_channels,
            gpt_n_heads=self.config.gpt_n_heads,
            gpt_max_audio_tokens=self.config.gpt_max_audio_tokens,
            gpt_max_text_tokens=self.config.gpt_max_text_tokens,
            gpt_max_prompt_tokens=self.config.gpt_max_prompt_tokens,
            gpt_num_audio_tokens=self.config.gpt_num_audio_tokens,
            gpt_start_audio_token=self.config.gpt_start_audio_token,
            gpt_stop_audio_token=self.config.gpt_stop_audio_token,
            gpt_code_stride_len=self.config.gpt_code_stride_len,
            gpt_use_masking_gt_prompt_approach=self.config.gpt_use_masking_gt_prompt_approach,
            gpt_use_perceiver_resampler=self.config.gpt_use_perceiver_resampler,
            kv_cache=self.config.kv_cache,
        )

        # Create audio config
        audio_config = XttsAudioConfig(
            sample_rate=self.config.sample_rate,
            output_sample_rate=self.config.output_sample_rate,
        )

        # Create the main config
        self.xtts_config = XttsConfig(
            model_args=model_args,
            audio=audio_config,
        )

    def _load_checkpoints(self):
        """Load model checkpoints."""
        config = self.config

        if config.xtts_checkpoint:
            self._load_xtts_checkpoint(config.xtts_checkpoint)
        elif config.pretrained_vocoder_checkpoint:
            self._load_pretrained_vocoder(config.pretrained_vocoder_checkpoint)
            self._init_gpt_from_scratch()
            self._verify_gpt_initialization()

        if config.gpt_checkpoint:
            self._load_gpt_checkpoint(config.gpt_checkpoint)

    def _load_xtts_checkpoint(self, checkpoint_path: str):
        """Load a full XTTS checkpoint."""
        checkpoint = self.xtts.get_compatible_checkpoint_state_dict(checkpoint_path)
        self.xtts.load_state_dict(checkpoint, strict=False)
        print(f">> XTTS weights restored from: {checkpoint_path}")

    def _load_pretrained_vocoder(self, checkpoint_path: str):
        """Load only the pretrained vocoder (HiFiGAN decoder)."""
        print(f">> Loading pretrained vocoder from: {checkpoint_path}")
        pretrained_state = torch.load(checkpoint_path, map_location=torch.device("cpu"))
        if "model" in pretrained_state:
            pretrained_state = pretrained_state["model"]

        # Extract only vocoder-related weights
        vocoder_state = {}
        for key, value in pretrained_state.items():
            if "hifigan_decoder." in key:
                new_key = key.replace("xtts.", "") if key.startswith("xtts.") else key
                new_key = new_key.replace("hifigan_decoder.", "")
                vocoder_state[new_key] = value

        if vocoder_state:
            missing_keys, unexpected_keys = self.xtts.hifigan_decoder.load_state_dict(vocoder_state, strict=False)
            print(f">> Loaded {len(vocoder_state)} vocoder parameters")
            if missing_keys:
                print(f">> Missing keys: {missing_keys[:5]}..." if len(missing_keys) > 5 else f">> Missing keys: {missing_keys}")
        else:
            print(">> WARNING: No vocoder weights found in checkpoint!")

    def _load_gpt_checkpoint(self, checkpoint_path: str):
        """Load a GPT-only checkpoint."""
        gpt_checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))

        # Handle Coqui Trainer exported model
        if "model" in gpt_checkpoint.keys() and "config" in gpt_checkpoint.keys():
            print("Coqui Trainer checkpoint detected! Converting it!")
            gpt_checkpoint = gpt_checkpoint["model"]
            states_keys = list(gpt_checkpoint.keys())
            for key in states_keys:
                if "gpt." in key:
                    new_key = key.replace("gpt.", "")
                    gpt_checkpoint[new_key] = gpt_checkpoint[key]
                    del gpt_checkpoint[key]
                else:
                    del gpt_checkpoint[key]

        # Handle token count mismatch for transfer learning
        if (
            "text_embedding.weight" in gpt_checkpoint
            and gpt_checkpoint["text_embedding.weight"].shape != self.xtts.gpt.text_embedding.weight.shape
        ):
            self._adapt_checkpoint_tokens(gpt_checkpoint)

        self.xtts.gpt.load_state_dict(gpt_checkpoint, strict=True)
        print(f">> GPT weights restored from: {checkpoint_path}")

    def _adapt_checkpoint_tokens(self, gpt_checkpoint: dict):
        """Adapt checkpoint for different token counts."""
        num_new_tokens = (
            self.xtts.gpt.text_embedding.weight.shape[0] - gpt_checkpoint["text_embedding.weight"].shape[0]
        )
        print(f" > Loading checkpoint with {num_new_tokens} additional tokens.")

        # Adapt text_embedding
        emb_g = gpt_checkpoint["text_embedding.weight"]
        new_row = torch.randn(num_new_tokens, emb_g.shape[1])
        start_token_row = emb_g[-1, :]
        emb_g = torch.cat([emb_g, new_row], axis=0)
        emb_g[-1, :] = start_token_row
        gpt_checkpoint["text_embedding.weight"] = emb_g

        # Adapt text_head weight
        text_head_weight = gpt_checkpoint["text_head.weight"]
        start_token_row = text_head_weight[-1, :]
        new_entry = torch.randn(num_new_tokens, self.xtts.gpt.text_head.weight.shape[1])
        text_head_weight = torch.cat([text_head_weight, new_entry], axis=0)
        text_head_weight[-1, :] = start_token_row
        gpt_checkpoint["text_head.weight"] = text_head_weight

        # Adapt text_head bias
        text_head_bias = gpt_checkpoint["text_head.bias"]
        start_token_row = text_head_bias[-1]
        new_bias_entry = torch.zeros(num_new_tokens)
        text_head_bias = torch.cat([text_head_bias, new_bias_entry], axis=0)
        text_head_bias[-1] = start_token_row
        gpt_checkpoint["text_head.bias"] = text_head_bias

    def _init_gpt_from_scratch(self):
        """Initialize GPT model weights from scratch using GPT-2 style initialization."""
        print(">> Initializing GPT model from scratch...")
        std = 0.02

        for name, param in self.xtts.gpt.named_parameters():
            if param.dim() > 1:
                if 'ln' in name or 'norm' in name:
                    continue
                elif 'embedding' in name or 'wte' in name or 'wpe' in name:
                    nn.init.normal_(param, mean=0.0, std=std)
                elif 'latents' in name:
                    nn.init.normal_(param, mean=0.0, std=std)
                elif any(x in name for x in ['weight', 'w1', 'w2', 'w3', 'wi', 'wo', 'dense']):
                    nn.init.normal_(param, mean=0.0, std=std)
                    if any(x in name for x in ['out_proj', 'o_proj', 'wo', 'dense_4h_to_h', 'c_proj']):
                        with torch.no_grad():
                            param *= (2 * self.config.gpt_layers) ** -0.5
            elif param.dim() == 1:
                if 'ln' not in name and 'norm' not in name:
                    nn.init.zeros_(param)

        if hasattr(self.xtts.gpt, 'mel_solo_embedding'):
            nn.init.normal_(self.xtts.gpt.mel_solo_embedding, mean=0.0, std=std)
        if hasattr(self.xtts.gpt, 'text_solo_embedding'):
            nn.init.normal_(self.xtts.gpt.text_solo_embedding, mean=0.0, std=std)

        print(">> GPT model initialized with GPT-2 style initialization")

    def _verify_gpt_initialization(self):
        """Verify that GPT model is properly initialized."""
        critical_params = ['text_embedding.weight', 'mel_embedding.weight', 'text_head.weight', 'mel_head.weight']

        for param_name in critical_params:
            for name, param in self.xtts.gpt.named_parameters():
                if param_name in name:
                    if torch.isnan(param).any():
                        raise ValueError(f"NaN values found in {name}")
                    if param.abs().max() == 0:
                        raise ValueError(f"All zero values found in {name}")
                    break

    def _init_mel_extractors(self):
        """Initialize mel spectrogram extractors."""
        config = self.config

        if config.gpt_use_perceiver_resampler:
            self.torch_mel_spectrogram_style_encoder = TorchMelSpectrogram(
                filter_length=2048,
                hop_length=256,
                win_length=1024,
                normalize=False,
                sampling_rate=config.sample_rate,
                mel_fmin=0,
                mel_fmax=8000,
                n_mel_channels=80,
                mel_norm_file=config.mel_norm_file,
            )
        else:
            self.torch_mel_spectrogram_style_encoder = TorchMelSpectrogram(
                filter_length=4096,
                hop_length=1024,
                win_length=4096,
                normalize=False,
                sampling_rate=config.sample_rate,
                mel_fmin=0,
                mel_fmax=8000,
                n_mel_channels=80,
                mel_norm_file=config.mel_norm_file,
            )

        self.torch_mel_spectrogram_dvae = TorchMelSpectrogram(
            mel_norm_file=config.mel_norm_file,
            sampling_rate=config.dvae_sample_rate,
        )

    def _init_dvae(self):
        """Initialize and load DVAE."""
        config = self.config

        self.dvae = DiscreteVAE(
            channels=80,
            normalization=None,
            positional_dims=1,
            num_tokens=config.gpt_num_audio_tokens - 2,
            codebook_dim=512,
            hidden_dim=512,
            num_resnet_blocks=3,
            kernel_size=3,
            num_layers=2,
            use_transposed_convs=False,
        )

        if config.dvae_checkpoint:
            dvae_checkpoint = torch.load(config.dvae_checkpoint, map_location=torch.device("cpu"))
            self.dvae.load_state_dict(dvae_checkpoint, strict=False)
            print(f">> DVAE weights restored from: {config.dvae_checkpoint}")

            # Freeze DVAE
            for param in self.dvae.parameters():
                param.requires_grad = False
        else:
            raise RuntimeError("You need to specify dvae_checkpoint path!")

    def _freeze_vocoder(self):
        """Freeze the HiFiGAN vocoder to prevent training."""
        print(">> Freezing HiFiGAN vocoder...")

        for param in self.xtts.hifigan_decoder.parameters():
            param.requires_grad = False

        self.xtts.hifigan_decoder.eval()
        print(">> Vocoder frozen")
        self._print_trainable_parameters()

    def _print_trainable_parameters(self):
        """Print statistics about trainable parameters."""
        total_params = 0
        trainable_params = 0

        for name, param in self.named_parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()

        print(f">> Total parameters: {total_params:,}")
        print(f">> Trainable parameters: {trainable_params:,}")
        print(f">> Frozen parameters: {total_params - trainable_params:,}")
        print(f">> Trainable percentage: {100 * trainable_params / total_params:.2f}%")

    @torch.no_grad()
    def _format_batch(self, batch: Dict) -> Dict:
        """Compute spectrograms and audio codes on the device."""
        batch["text_lengths"] = batch["text_lengths"]
        batch["wav_lengths"] = batch["wav_lengths"]
        batch["text_inputs"] = batch["padded_text"]
        batch["cond_idxs"] = batch["cond_idxs"]

        # Compute conditioning mel specs
        B, num_cond_samples, C, T = batch["conditioning"].size()
        conditioning_reshaped = batch["conditioning"].view(B * num_cond_samples, C, T)
        paired_conditioning_mel = self.torch_mel_spectrogram_style_encoder(conditioning_reshaped)

        n_mel = self.torch_mel_spectrogram_style_encoder.n_mel_channels
        T_mel = paired_conditioning_mel.size(2)
        paired_conditioning_mel = paired_conditioning_mel.view(B, num_cond_samples, n_mel, T_mel)
        batch["cond_mels"] = paired_conditioning_mel

        # Compute codes using DVAE
        if self.config.sample_rate != self.config.dvae_sample_rate:
            dvae_wav = torchaudio.functional.resample(
                batch["wav"],
                orig_freq=self.config.sample_rate,
                new_freq=self.config.dvae_sample_rate,
                lowpass_filter_width=64,
                rolloff=0.9475937167399596,
                resampling_method="kaiser_window",
                beta=14.769656459379492,
            )
        else:
            dvae_wav = batch["wav"]

        dvae_mel_spec = self.torch_mel_spectrogram_dvae(dvae_wav)
        codes = self.dvae.get_codebook_indices(dvae_mel_spec)
        batch["audio_codes"] = codes

        # Clean up
        del batch["padded_text"]
        del batch["wav"]
        del batch["conditioning"]

        return batch

    def forward(
        self,
        text_inputs: torch.Tensor,
        text_lengths: torch.Tensor,
        audio_codes: torch.Tensor,
        wav_lengths: torch.Tensor,
        cond_mels: torch.Tensor,
        cond_idxs: Optional[torch.Tensor],
        cond_lens: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the GPT model.

        Args:
            text_inputs: Text token IDs (B, T)
            text_lengths: Text lengths (B,)
            audio_codes: Audio codes from DVAE (B, M)
            wav_lengths: Waveform lengths (B,)
            cond_mels: Conditioning mel spectrograms (B, num_samples, 80, T_mel)
            cond_idxs: Conditioning indices (B, 2)
            cond_lens: Conditioning lengths (B,)

        Returns:
            Tuple of (text_loss, mel_loss, additional_outputs)
        """
        return self.xtts.gpt(
            text_inputs,
            text_lengths,
            audio_codes,
            wav_lengths,
            cond_mels=cond_mels,
            cond_idxs=cond_idxs,
            cond_lens=cond_lens,
        )

    def _compute_loss(self, batch: Dict, ignore_masking: bool = False) -> Dict[str, torch.Tensor]:
        """Compute training loss from a batch."""
        batch = self._format_batch(batch)

        cond_mels = batch["cond_mels"]
        text_inputs = batch["text_inputs"]
        text_lengths = batch["text_lengths"]
        audio_codes = batch["audio_codes"]
        wav_lengths = batch["wav_lengths"]
        cond_idxs = None if ignore_masking else batch["cond_idxs"]
        cond_lens = batch["cond_lens"]

        loss_text, loss_mel, _ = self.forward(
            text_inputs, text_lengths, audio_codes, wav_lengths, cond_mels, cond_idxs, cond_lens
        )

        loss_text_weighted = loss_text * self.config.gpt_loss_text_ce_weight
        loss_mel_weighted = loss_mel * self.config.gpt_loss_mel_ce_weight
        total_loss = loss_text_weighted + loss_mel_weighted

        return {
            "loss": total_loss,
            "loss_text_ce": loss_text_weighted,
            "loss_mel_ce": loss_mel_weighted,
        }

    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        """Training step for PyTorch Lightning."""
        losses = self._compute_loss(batch)

        # Log all losses
        self.log("train/loss", losses["loss"], prog_bar=True, sync_dist=True)
        self.log("train/loss_text_ce", losses["loss_text_ce"], sync_dist=True)
        self.log("train/loss_mel_ce", losses["loss_mel_ce"], sync_dist=True)

        return losses["loss"]

    def validation_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        """Validation step for PyTorch Lightning."""
        # Ignore masking for more consistent evaluation
        losses = self._compute_loss(batch, ignore_masking=True)

        # Log all losses
        self.log("val/loss", losses["loss"], prog_bar=True, sync_dist=True)
        self.log("val/loss_text_ce", losses["loss_text_ce"], sync_dist=True)
        self.log("val/loss_mel_ce", losses["loss_mel_ce"], sync_dist=True)

        return losses["loss"]

    def on_train_epoch_start(self):
        """Called at the start of each training epoch."""
        # Keep GPT in train mode
        self.xtts.gpt.train()
        # Keep vocoder in eval mode when frozen
        self.xtts.hifigan_decoder.eval()

    def on_validation_epoch_end(self):
        """Run test sentence synthesis at the end of validation."""
        if self.config.test_sentences:
            self._synthesize_test_sentences()

    @torch.no_grad()
    def _synthesize_test_sentences(self):
        """Synthesize test sentences for evaluation."""
        try:
            # Initialize GPT for inference
            self.xtts.gpt.init_gpt_for_inference(kv_cache=self.config.kv_cache, use_deepspeed=False)
            self.xtts.gpt.eval()

            print(" | > Synthesizing test sentences.")
            for idx, s_info in enumerate(self.config.test_sentences):
                wav = self.xtts.synthesize(
                    s_info["text"],
                    self.xtts_config,
                    s_info["speaker_wav"],
                    s_info["language"],
                    gpt_cond_len=3,
                )["wav"]

                # Log audio to tensorboard/wandb
                if self.logger is not None:
                    self.logger.experiment.add_audio(
                        f"test_audio/{idx}",
                        torch.from_numpy(wav),
                        self.global_step,
                        sample_rate=self.config.output_sample_rate,
                    )

            # Clean up inference layers
            del self.xtts.gpt.gpt_inference
            del self.xtts.gpt.gpt.wte

        except Exception as e:
            print(f"Warning: Test sentence synthesis failed: {e}")
        finally:
            # Restore training mode
            self.xtts.gpt.train()

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizer and learning rate scheduler."""
        optimizer = self._get_optimizer()
        scheduler = self._get_scheduler(optimizer)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def _get_optimizer(self) -> torch.optim.Optimizer:
        """Create the optimizer."""
        config = self.config

        if config.optimizer_wd_only_on_weights:
            return self._get_optimizer_with_wd_on_weights()

        # Standard optimizer for GPT parameters only
        optimizer_class = getattr(torch.optim, config.optimizer)
        return optimizer_class(
            self.xtts.gpt.parameters(),
            lr=config.lr,
            **config.optimizer_params,
        )

    def _get_optimizer_with_wd_on_weights(self) -> torch.optim.Optimizer:
        """Create optimizer with weight decay only on weight parameters."""
        config = self.config
        net = self.xtts.gpt

        norm_modules = (
            nn.BatchNorm2d, nn.InstanceNorm2d, nn.BatchNorm1d,
            nn.InstanceNorm1d, nn.BatchNorm3d, nn.InstanceNorm3d,
            nn.GroupNorm, nn.LayerNorm,
        )
        emb_modules = (nn.Embedding, nn.EmbeddingBag)

        param_names_notweights = set()
        all_param_names = set()
        param_map = {}

        for mn, m in net.named_modules():
            for k, v in m.named_parameters():
                v.is_bias = k.endswith(".bias")
                v.is_weight = k.endswith(".weight")
                v.is_norm = isinstance(m, norm_modules)
                v.is_emb = isinstance(m, emb_modules)

                fpn = f"{mn}.{k}" if mn else k
                all_param_names.add(fpn)
                param_map[fpn] = v

                if v.is_bias or v.is_norm or v.is_emb:
                    param_names_notweights.add(fpn)

        params_names_notweights = sorted(list(param_names_notweights))
        params_notweights = [param_map[k] for k in params_names_notweights]
        params_names_weights = sorted(list(all_param_names ^ param_names_notweights))
        params_weights = [param_map[k] for k in params_names_weights]

        groups = [
            {"params": params_weights, "weight_decay": config.optimizer_params.get("weight_decay", 0.01)},
            {"params": params_notweights, "weight_decay": 0},
        ]

        optimizer_class = getattr(torch.optim, config.optimizer)
        optimizer_params = {k: v for k, v in config.optimizer_params.items() if k != "weight_decay"}

        return optimizer_class(groups, lr=config.lr, **optimizer_params)

    def _get_scheduler(self, optimizer: torch.optim.Optimizer):
        """Create the learning rate scheduler."""
        config = self.config
        scheduler_class = getattr(torch.optim.lr_scheduler, config.lr_scheduler)
        return scheduler_class(optimizer, **config.lr_scheduler_params)

    def load_checkpoint(
        self,
        checkpoint_path: str,
        strict: bool = True,
    ):
        """Load a full model checkpoint."""
        state = self.xtts.get_compatible_checkpoint_state_dict(checkpoint_path)
        self.xtts.load_state_dict(state, strict=strict)
        print(f">> Checkpoint loaded from: {checkpoint_path}")


class XTTSDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for XTTS training."""

    def __init__(
        self,
        config: XTTSLightningConfig,
        train_samples: List[Dict],
        eval_samples: Optional[List[Dict]] = None,
        tokenizer: Optional[VoiceBpeTokenizer] = None,
    ):
        """
        Initialize the data module.

        Args:
            config: XTTSLightningConfig with data parameters
            train_samples: List of training samples
            eval_samples: List of evaluation samples (optional)
            tokenizer: VoiceBpeTokenizer instance
        """
        super().__init__()
        self.config = config
        self.train_samples = train_samples
        self.eval_samples = eval_samples
        self.tokenizer = tokenizer or VoiceBpeTokenizer(config.tokenizer_file)

        self.train_dataset = None
        self.eval_dataset = None

    def setup(self, stage: Optional[str] = None):
        """Setup datasets for training and validation."""
        if stage == "fit" or stage is None:
            # Create a minimal config object for the dataset
            dataset_config = self._create_dataset_config()

            self.train_dataset = XTTSDataset(
                dataset_config,
                self.train_samples,
                self.tokenizer,
                self.config.sample_rate,
                is_eval=False,
            )

            if self.eval_samples:
                self.eval_dataset = XTTSDataset(
                    dataset_config,
                    self.eval_samples,
                    self.tokenizer,
                    self.config.sample_rate,
                    is_eval=True,
                )

    def _create_dataset_config(self):
        """Create a config object compatible with XTTSDataset."""
        from dataclasses import dataclass

        @dataclass
        class DatasetModelArgs:
            max_conditioning_length: int
            min_conditioning_length: int
            max_wav_length: int
            max_text_length: int
            debug_loading_failures: bool
            gpt_use_masking_gt_prompt_approach: bool

        @dataclass
        class DatasetConfig:
            model_args: DatasetModelArgs
            training_seed: int

        model_args = DatasetModelArgs(
            max_conditioning_length=self.config.max_conditioning_length,
            min_conditioning_length=self.config.min_conditioning_length,
            max_wav_length=self.config.max_wav_length,
            max_text_length=self.config.max_text_length,
            debug_loading_failures=self.config.debug_loading_failures,
            gpt_use_masking_gt_prompt_approach=self.config.gpt_use_masking_gt_prompt_approach,
        )

        return DatasetConfig(
            model_args=model_args,
            training_seed=self.config.training_seed,
        )

    def train_dataloader(self) -> DataLoader:
        """Create training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=self.train_dataset.collate_fn,
            num_workers=self.config.num_loader_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Create validation dataloader."""
        if self.eval_dataset is None:
            return None

        return DataLoader(
            self.eval_dataset,
            batch_size=self.config.eval_batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=self.eval_dataset.collate_fn,
            num_workers=self.config.num_eval_loader_workers,
            pin_memory=True,
        )

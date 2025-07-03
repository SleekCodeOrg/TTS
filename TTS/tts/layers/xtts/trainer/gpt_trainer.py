from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Union

import torch
import torch.nn as nn
import torchaudio
from coqpit import Coqpit
from torch.nn import functional as F
from torch.utils.data import DataLoader
from trainer.torch import DistributedSampler
from trainer.trainer_utils import get_optimizer, get_scheduler

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.datasets.dataset import TTSDataset
from TTS.tts.layers.tortoise.arch_utils import TorchMelSpectrogram
from TTS.tts.layers.xtts.dvae import DiscreteVAE
from TTS.tts.layers.xtts.tokenizer import VoiceBpeTokenizer
from TTS.tts.layers.xtts.trainer.dataset import XTTSDataset
from TTS.tts.models.base_tts import BaseTTS
from TTS.tts.models.xtts import Xtts, XttsArgs, XttsAudioConfig
from TTS.utils.io import load_fsspec


@dataclass
class GPTTrainerConfig(XttsConfig):
    lr: float = 5e-06
    training_seed: int = 1
    optimizer_wd_only_on_weights: bool = False
    weighted_loss_attrs: dict = field(default_factory=lambda: {})
    weighted_loss_multipliers: dict = field(default_factory=lambda: {})
    test_sentences: List[dict] = field(default_factory=lambda: [])


@dataclass
class XttsAudioConfig(XttsAudioConfig):
    dvae_sample_rate: int = 22050


@dataclass
class GPTArgs(XttsArgs):
    min_conditioning_length: int = 66150
    max_conditioning_length: int = 132300
    gpt_loss_text_ce_weight: float = 0.01
    gpt_loss_mel_ce_weight: float = 1.0
    gpt_num_audio_tokens: int = 8194
    debug_loading_failures: bool = False
    max_wav_length: int = 255995  # ~11.6 seconds
    max_text_length: int = 200
    tokenizer_file: str = ""
    mel_norm_file: str = "https://coqui.gateway.scarf.sh/v0.14.0_models/mel_norms.pth"
    dvae_checkpoint: str = ""
    xtts_checkpoint: str = ""
    gpt_checkpoint: str = ""  # if defined it will replace the gpt weights on xtts model
    vocoder: str = ""  # overide vocoder key on the config to avoid json write issues
    pretrained_vocoder_checkpoint: str = ""  # path to pretrained vocoder checkpoint


def callback_clearml_load_save(operation_type, model_info):
    # return None means skip the file upload/log, returning model_info will continue with the log/upload
    # you can also change the upload destination file name model_info.upload_filename or check the local file size with Path(model_info.local_model_path).stat().st_size
    assert operation_type in ("load", "save")
    # print(operation_type, model_info.__dict__)

    if "similarities.pth" in model_info.__dict__["local_model_path"]:
        return None

    return model_info


class GPTTrainer(BaseTTS):
    def update_training_dashboard_logger(self, batch=None, outputs=None):
        return 

    def __init__(self, config: Coqpit):
        """
        Tortoise GPT training class
        """
        super().__init__(config, ap=None, tokenizer=None)
        self.config = config
        # init XTTS model
        self.xtts = Xtts(self.config)
        # create the tokenizer with the target vocabulary
        self.xtts.tokenizer = VoiceBpeTokenizer(self.args.tokenizer_file)
        # init gpt encoder and hifigan decoder
        self.xtts.init_models()

        if self.args.xtts_checkpoint:
            self.load_checkpoint(self.config, self.args.xtts_checkpoint, eval=False, strict=False)
        elif self.args.pretrained_vocoder_checkpoint:
            # Load only the pretrained vocoder (HifiGAN decoder) when training from scratch
            print(">> Loading pretrained vocoder from:", self.args.pretrained_vocoder_checkpoint)
            pretrained_state = torch.load(self.args.pretrained_vocoder_checkpoint, map_location=torch.device("cpu"))
            if "model" in pretrained_state:
                pretrained_state = pretrained_state["model"]
            
            # Extract only vocoder-related weights
            vocoder_state = {}
            for key, value in pretrained_state.items():
                if "hifigan_decoder." in key:
                    # Remove the 'xtts.' prefix if present
                    new_key = key.replace("xtts.", "") if key.startswith("xtts.") else key
                    vocoder_state[new_key] = value
            
            # Load vocoder weights
            if vocoder_state:
                missing_keys, unexpected_keys = self.xtts.hifigan_decoder.load_state_dict(vocoder_state, strict=False)
                print(f">> Loaded {len(vocoder_state)} vocoder parameters from pretrained checkpoint")
                if missing_keys:
                    print(f">> Missing keys in vocoder: {missing_keys[:5]}..." if len(missing_keys) > 5 else f">> Missing keys in vocoder: {missing_keys}")
                if unexpected_keys:
                    print(f">> Unexpected keys in vocoder: {unexpected_keys[:5]}..." if len(unexpected_keys) > 5 else f">> Unexpected keys in vocoder: {unexpected_keys}")
            else:
                print(">> WARNING: No vocoder weights found in pretrained checkpoint!")
            
            # Initialize GPT model weights from scratch
            self._init_gpt_from_scratch()
            
            # Verify GPT initialization
            self._verify_gpt_initialization()

        # set mel stats
        if self.args.mel_norm_file:
            self.xtts.mel_stats = load_fsspec(self.args.mel_norm_file)

        # load GPT if available
        if self.args.gpt_checkpoint:
            gpt_checkpoint = torch.load(self.args.gpt_checkpoint, map_location=torch.device("cpu"))
            # deal with coqui Trainer exported model
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

            # edit checkpoint if the number of tokens is changed to ensures the better transfer learning possible
            if (
                "text_embedding.weight" in gpt_checkpoint
                and gpt_checkpoint["text_embedding.weight"].shape != self.xtts.gpt.text_embedding.weight.shape
            ):
                num_new_tokens = (
                    self.xtts.gpt.text_embedding.weight.shape[0] - gpt_checkpoint["text_embedding.weight"].shape[0]
                )
                print(f" > Loading checkpoint with {num_new_tokens} additional tokens.")

                # add new tokens to a linear layer (text_head)
                emb_g = gpt_checkpoint["text_embedding.weight"]
                new_row = torch.randn(num_new_tokens, emb_g.shape[1])
                start_token_row = emb_g[-1, :]
                emb_g = torch.cat([emb_g, new_row], axis=0)
                emb_g[-1, :] = start_token_row
                gpt_checkpoint["text_embedding.weight"] = emb_g

                # add new weights to the linear layer (text_head)
                text_head_weight = gpt_checkpoint["text_head.weight"]
                start_token_row = text_head_weight[-1, :]
                new_entry = torch.randn(num_new_tokens, self.xtts.gpt.text_head.weight.shape[1])
                text_head_weight = torch.cat([text_head_weight, new_entry], axis=0)
                text_head_weight[-1, :] = start_token_row
                gpt_checkpoint["text_head.weight"] = text_head_weight

                # add new biases to the linear layer (text_head)
                text_head_bias = gpt_checkpoint["text_head.bias"]
                start_token_row = text_head_bias[-1]
                new_bias_entry = torch.zeros(num_new_tokens)
                text_head_bias = torch.cat([text_head_bias, new_bias_entry], axis=0)
                text_head_bias[-1] = start_token_row
                gpt_checkpoint["text_head.bias"] = text_head_bias

            self.xtts.gpt.load_state_dict(gpt_checkpoint, strict=True)
            print(">> GPT weights restored from:", self.args.gpt_checkpoint)

        # Mel spectrogram extractor for conditioning
        if self.args.gpt_use_perceiver_resampler:
            self.torch_mel_spectrogram_style_encoder = TorchMelSpectrogram(
                filter_length=2048,
                hop_length=256,
                win_length=1024,
                normalize=False,
                sampling_rate=config.audio.sample_rate,
                mel_fmin=0,
                mel_fmax=8000,
                n_mel_channels=80,
                mel_norm_file=self.args.mel_norm_file,
            )
        else:
            self.torch_mel_spectrogram_style_encoder = TorchMelSpectrogram(
                filter_length=4096,
                hop_length=1024,
                win_length=4096,
                normalize=False,
                sampling_rate=config.audio.sample_rate,
                mel_fmin=0,
                mel_fmax=8000,
                n_mel_channels=80,
                mel_norm_file=self.args.mel_norm_file,
            )

        # Load DVAE
        self.dvae = DiscreteVAE(
            channels=80,
            normalization=None,
            positional_dims=1,
            num_tokens=self.args.gpt_num_audio_tokens - 2,
            codebook_dim=512,
            hidden_dim=512,
            num_resnet_blocks=3,
            kernel_size=3,
            num_layers=2,
            use_transposed_convs=False,
        )

        # Don't put DVAE in eval mode here - even though we freeze its parameters,
        # gradients still need to flow through it for GPT training
        if self.args.dvae_checkpoint:
            dvae_checkpoint = torch.load(self.args.dvae_checkpoint, map_location=torch.device("cpu"))
            self.dvae.load_state_dict(dvae_checkpoint, strict=False)
            print(">> DVAE weights restored from:", self.args.dvae_checkpoint)
            # Freeze DVAE - we don't train it
            for param in self.dvae.parameters():
                param.requires_grad = False
        else:
            raise RuntimeError(
                "You need to specify config.model_args.dvae_checkpoint path to be able to train the GPT decoder!!"
            )

        # Mel spectrogram extractor for DVAE
        self.torch_mel_spectrogram_dvae = TorchMelSpectrogram(
            mel_norm_file=self.args.mel_norm_file, sampling_rate=config.audio.dvae_sample_rate
        )
        
        # Now freeze vocoder if we loaded pretrained weights
        if self.args.pretrained_vocoder_checkpoint and not self.args.xtts_checkpoint:
            self._freeze_vocoder()

    def _init_gpt_from_scratch(self):
        """Initialize GPT model weights from scratch using GPT-2 style initialization."""
        print(">> Initializing GPT model from scratch...")
        
        # Standard deviation for initialization (GPT-2 uses 0.02)
        std = 0.02
        
        # Initialize all GPT parameters
        for name, param in self.xtts.gpt.named_parameters():
            if param.dim() > 1:  # Only initialize multi-dimensional parameters
                if 'ln' in name or 'norm' in name:
                    # Skip layer normalization - they have their own initialization
                    continue
                elif 'embedding' in name or 'wte' in name or 'wpe' in name:
                    # Embedding layers (including position embeddings)
                    nn.init.normal_(param, mean=0.0, std=std)
                elif 'latents' in name:
                    # Perceiver latents
                    nn.init.normal_(param, mean=0.0, std=std)
                elif any(x in name for x in ['weight', 'w1', 'w2', 'w3', 'wi', 'wo', 'dense']):
                    # Linear layers - use normal initialization like GPT-2
                    nn.init.normal_(param, mean=0.0, std=std)
                    
                    # Scale initialization by depth for output projections
                    if any(x in name for x in ['out_proj', 'o_proj', 'wo', 'dense_4h_to_h', 'c_proj']):
                        with torch.no_grad():
                            param *= (2 * self.config.model_args.gpt_layers) ** -0.5
            elif param.dim() == 1:
                # Biases - leave as initialized (usually zero)
                if 'ln' not in name and 'norm' not in name:
                    nn.init.zeros_(param)
        
        # Ensure solo embeddings are initialized if they exist
        if hasattr(self.xtts.gpt, 'mel_solo_embedding'):
            nn.init.normal_(self.xtts.gpt.mel_solo_embedding, mean=0.0, std=std)
        if hasattr(self.xtts.gpt, 'text_solo_embedding'):
            nn.init.normal_(self.xtts.gpt.text_solo_embedding, mean=0.0, std=std)
        
        print(">> GPT model initialized from scratch with GPT-2 style initialization")
        
        # Debug: Print weight statistics (uncomment for debugging)
        # self._debug_weight_stats()

    def _freeze_vocoder(self):
        """Freeze the HiFiGAN vocoder to prevent training."""
        print(">> Freezing HiFiGAN vocoder...")
        
        # Freeze all vocoder parameters
        for param in self.xtts.hifigan_decoder.parameters():
            param.requires_grad = False
        
        # Put vocoder in eval mode
        self.xtts.hifigan_decoder.eval()
        
        # Note: DVAE is already frozen after loading its checkpoint
        # We don't freeze it here as it's independent of vocoder freezing
        
        print(">> Vocoder frozen")
        
        # Verify what's trainable
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
    
    def _debug_weight_stats(self):
        """Print statistics about weight initialization for debugging."""
        print("\n>> Weight initialization statistics:")
        for name, param in self.xtts.gpt.named_parameters():
            if param.dim() > 1 and param.requires_grad:
                mean = param.data.mean().item()
                std = param.data.std().item()
                print(f"  {name}: mean={mean:.6f}, std={std:.6f}, shape={list(param.shape)}")
    
    def _verify_gpt_initialization(self):
        """Verify that GPT model is properly initialized."""
        # Check a few key parameters to ensure they're not zero or NaN
        critical_params = ['text_embedding.weight', 'mel_embedding.weight', 'text_head.weight', 'mel_head.weight']
        
        for param_name in critical_params:
            found = False
            for name, param in self.xtts.gpt.named_parameters():
                if param_name in name:
                    found = True
                    if torch.isnan(param).any():
                        raise ValueError(f"NaN values found in {name}")
                    if param.abs().max() == 0:
                        raise ValueError(f"All zero values found in {name}")
                    break
            
            if not found:
                print(f">> Warning: Could not find parameter {param_name} for verification")

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, text_inputs, text_lengths, audio_codes, wav_lengths, cond_mels, cond_idxs, cond_lens):
        """
        Forward pass that uses both text and voice in either text conditioning mode or voice conditioning mode
        (actuated by `text_first`).

        text_inputs: long tensor, (b,t)
        text_lengths: long tensor, (b,)
        mel_inputs:  long tensor, (b,m)
        wav_lengths: long tensor, (b,)
        cond_mels: MEL float tensor, (b, num_samples, 80,t_m)
        cond_idxs: cond start and end indexs, (b, 2)
        cond_lens: long tensor, (b,)
        """
        losses = self.xtts.gpt(
            text_inputs,
            text_lengths,
            audio_codes,
            wav_lengths,
            cond_mels=cond_mels,
            cond_idxs=cond_idxs,
            cond_lens=cond_lens,
        )
        return losses

    @torch.no_grad()
    def test_run(self, assets) -> Tuple[Dict, Dict]:  # pylint: disable=W0613
        test_audios = {}
        if self.config.test_sentences:
            # init gpt for inference mode
            self.xtts.gpt.init_gpt_for_inference(kv_cache=self.args.kv_cache, use_deepspeed=False)
            self.xtts.gpt.eval()
            print(" | > Synthesizing test sentences.")
            for idx, s_info in enumerate(self.config.test_sentences):
                wav = self.xtts.synthesize(
                    s_info["text"],
                    self.config,
                    s_info["speaker_wav"],
                    s_info["language"],
                    gpt_cond_len=3,
                )["wav"]
                test_audios["{}-audio".format(idx)] = wav

            # delete inference layers
            del self.xtts.gpt.gpt_inference
            del self.xtts.gpt.gpt.wte
        return {"audios": test_audios}

    def test_log(
        self, outputs: dict, logger: "Logger", assets: dict, steps: int  # pylint: disable=unused-argument
    ) -> None:
        logger.test_audios(steps, outputs["audios"], self.args.output_sample_rate)

    def format_batch(self, batch: Dict) -> Dict:
        return batch

    @torch.no_grad()  # torch no grad to avoid gradients from the pre-processing and DVAE codes extraction
    def format_batch_on_device(self, batch):
        """Compute spectrograms on the device."""
        batch["text_lengths"] = batch["text_lengths"]
        batch["wav_lengths"] = batch["wav_lengths"]
        batch["text_inputs"] = batch["padded_text"]
        batch["cond_idxs"] = batch["cond_idxs"]
        # compute conditioning mel specs
        # transform waves from torch.Size([B, num_cond_samples, 1, T] to torch.Size([B * num_cond_samples, 1, T] because if is faster than iterate the tensor
        B, num_cond_samples, C, T = batch["conditioning"].size()
        conditioning_reshaped = batch["conditioning"].view(B * num_cond_samples, C, T)
        paired_conditioning_mel = self.torch_mel_spectrogram_style_encoder(conditioning_reshaped)
        # transform torch.Size([B * num_cond_samples, n_mel, T_mel]) in torch.Size([B, num_cond_samples, n_mel, T_mel])
        n_mel = self.torch_mel_spectrogram_style_encoder.n_mel_channels  # paired_conditioning_mel.size(1)
        T_mel = paired_conditioning_mel.size(2)
        paired_conditioning_mel = paired_conditioning_mel.view(B, num_cond_samples, n_mel, T_mel)
        # get the conditioning embeddings
        batch["cond_mels"] = paired_conditioning_mel
        # compute codes using DVAE
        if self.config.audio.sample_rate != self.config.audio.dvae_sample_rate:
            dvae_wav = torchaudio.functional.resample(
                batch["wav"],
                orig_freq=self.config.audio.sample_rate,
                new_freq=self.config.audio.dvae_sample_rate,
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
        # delete useless batch tensors
        del batch["padded_text"]
        del batch["wav"]
        del batch["conditioning"]
        return batch

    def train_step(self, batch, criterion):
        loss_dict = {}
        cond_mels = batch["cond_mels"]
        text_inputs = batch["text_inputs"]
        text_lengths = batch["text_lengths"]
        audio_codes = batch["audio_codes"]
        wav_lengths = batch["wav_lengths"]
        cond_idxs = batch["cond_idxs"]
        cond_lens = batch["cond_lens"]

        loss_text, loss_mel, _ = self.forward(
            text_inputs, text_lengths, audio_codes, wav_lengths, cond_mels, cond_idxs, cond_lens
        )
        loss_dict["loss_text_ce"] = loss_text * self.args.gpt_loss_text_ce_weight
        loss_dict["loss_mel_ce"] = loss_mel * self.args.gpt_loss_mel_ce_weight
        loss_dict["loss"] = loss_dict["loss_text_ce"] + loss_dict["loss_mel_ce"]
        return {"model_outputs": None}, loss_dict

    def eval_step(self, batch, criterion):
        # ignore masking for more consistent evaluation
        batch["cond_idxs"] = None
        return self.train_step(batch, criterion)

    def on_train_epoch_start(self, trainer):
        # Only set specific components to their correct modes
        # Do NOT put the whole model in eval mode as it can interfere with gradient computation
        if hasattr(trainer.model, "module") and hasattr(trainer.model.module, "xtts"):
            # Multi-GPU setup
            trainer.model.module.xtts.gpt.train()
            # Keep vocoder in eval mode when frozen
            trainer.model.module.xtts.hifigan_decoder.eval()
            # Don't put DVAE in eval mode - gradients need to flow through it
        else:
            # Single GPU setup
            trainer.model.xtts.gpt.train()
            # Keep vocoder in eval mode when frozen
            trainer.model.xtts.hifigan_decoder.eval()
            # Don't put DVAE in eval mode - gradients need to flow through it

    def on_init_end(self, trainer):  # pylint: disable=W0613
        # ignore similarities.pth on clearml save/upload
        if self.config.dashboard_logger.lower() == "clearml":
            from clearml.binding.frameworks import WeightsFileHandler

            WeightsFileHandler.add_pre_callback(callback_clearml_load_save)

    @torch.no_grad()
    def inference(
        self,
        x,
        aux_input=None,
    ):  # pylint: disable=dangerous-default-value
        return None

    @staticmethod
    def get_criterion():
        return None

    def get_sampler(self, dataset: TTSDataset, num_gpus=1):
        # sampler for DDP
        batch_sampler = DistributedSampler(dataset) if num_gpus > 1 else None
        return batch_sampler

    def get_data_loader(
        self,
        config: Coqpit,
        assets: Dict,
        is_eval: bool,
        samples: Union[List[Dict], List[List]],
        verbose: bool,
        num_gpus: int,
        rank: int = None,
    ) -> "DataLoader":  # pylint: disable=W0613
        if is_eval and not config.run_eval:
            loader = None
        else:
            # init dataloader
            dataset = XTTSDataset(self.config, samples, self.xtts.tokenizer, config.audio.sample_rate, is_eval)

            # wait all the DDP process to be ready
            if num_gpus > 1:
                torch.distributed.barrier()

            # sort input sequences from short to long
            # dataset.preprocess_samples()

            # get samplers
            sampler = self.get_sampler(dataset, num_gpus)

            # ignore sampler when is eval because if we changed the sampler parameter we will not be able to compare previous runs
            if sampler is None or is_eval:
                loader = DataLoader(
                    dataset,
                    batch_size=config.eval_batch_size if is_eval else config.batch_size,
                    shuffle=False,
                    drop_last=False,
                    collate_fn=dataset.collate_fn,
                    num_workers=config.num_eval_loader_workers if is_eval else config.num_loader_workers,
                    pin_memory=False,
                )
            else:
                loader = DataLoader(
                    dataset,
                    sampler=sampler,
                    batch_size = config.eval_batch_size if is_eval else config.batch_size,
                    collate_fn=dataset.collate_fn,
                    num_workers=config.num_eval_loader_workers if is_eval else config.num_loader_workers,
                    pin_memory=False,
                )
        return loader

    def get_optimizer(self) -> List:
        """Initiate and return the optimizer based on the config parameters."""
        # ToDo: deal with multi GPU training
        if self.config.optimizer_wd_only_on_weights:
            # parameters to only GPT model
            net = self.xtts.gpt

            # normalizations
            norm_modules = (
                nn.BatchNorm2d,
                nn.InstanceNorm2d,
                nn.BatchNorm1d,
                nn.InstanceNorm1d,
                nn.BatchNorm3d,
                nn.InstanceNorm3d,
                nn.GroupNorm,
                nn.LayerNorm,
            )
            # nn.Embedding
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

                    fpn = "%s.%s" % (mn, k) if mn else k  # full param name
                    all_param_names.add(fpn)
                    param_map[fpn] = v
                    if v.is_bias or v.is_norm or v.is_emb:
                        param_names_notweights.add(fpn)

            params_names_notweights = sorted(list(param_names_notweights))
            params_notweights = [param_map[k] for k in params_names_notweights]
            params_names_weights = sorted(list(all_param_names ^ param_names_notweights))
            params_weights = [param_map[k] for k in params_names_weights]

            groups = [
                {"params": params_weights, "weight_decay": self.config.optimizer_params["weight_decay"]},
                {"params": params_notweights, "weight_decay": 0},
            ]
            # torch.optim.AdamW
            opt = get_optimizer(
                self.config.optimizer,
                self.config.optimizer_params,
                self.config.lr,
                parameters=groups,
            )
            opt._group_names = [params_names_weights, params_names_notweights]
            return opt

        return get_optimizer(
            self.config.optimizer,
            self.config.optimizer_params,
            self.config.lr,
            # optimize only for the GPT model
            parameters=self.xtts.gpt.parameters(),
        )

    def get_scheduler(self, optimizer) -> List:
        """Set the scheduler for the optimizer.

        Args:
            optimizer: `torch.optim.Optimizer`.
        """
        return get_scheduler(self.config.lr_scheduler, self.config.lr_scheduler_params, optimizer)

    def load_checkpoint(
        self,
        config,
        checkpoint_path,
        eval=False,
        strict=True,
        cache_storage="/tmp/tts_cache",
        target_protocol="s3",
        target_options={"anon": True},
    ):  # pylint: disable=unused-argument, disable=W0201, disable=W0102, redefined-builtin
        """Load the model checkpoint and setup for training or inference"""

        state = self.xtts.get_compatible_checkpoint_state_dict(checkpoint_path)

        # load the model weights
        self.xtts.load_state_dict(state, strict=strict)

        if eval:
            self.xtts.gpt.init_gpt_for_inference(kv_cache=self.args.kv_cache, use_deepspeed=False)
            self.eval()
            assert not self.training

    @staticmethod
    def init_from_config(config: "GPTTrainerConfig", samples: Union[List[List], List[Dict]] = None):
        """Initiate model from config

        Args:
            config (GPTTrainerConfig): Model config.
            samples (Union[List[List], List[Dict]]): Training samples to parse speaker ids for training.
                Defaults to None.
        """
        return GPTTrainer(config)

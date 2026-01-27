# Copyright 2026 Microsoft and the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Convert VibeVoice ASR checkpoints from microsoft/VibeVoice-ASR to Hugging Face format."""

from __future__ import annotations

import argparse
import gc
import json
import logging
import re
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

from transformers import (
    AutoTokenizer,
    Qwen2Config,
    VibeVoiceASRConfig,
    VibeVoiceASRForConditionalGeneration,
    VibeVoiceASRProcessor,
    VibeVoiceAcousticTokenizerConfig,
    VibeVoiceSemanticTokenizerConfig,
)


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


# fmt: off
STATE_DICT_MAPPING = {
    # Language model keys (from model.language_model.*)
    r"^model\.language_model\.embed_tokens\.weight":                                r"model.language_model.embed_tokens.weight",
    r"^model\.language_model\.layers\.(\d+)\.self_attn\.(q|k|v|o)_proj\.":         r"model.language_model.layers.\1.self_attn.\2_proj.",
    r"^model\.language_model\.layers\.(\d+)\.mlp\.(gate|up|down)_proj\.":          r"model.language_model.layers.\1.mlp.\2_proj.",
    r"^model\.language_model\.layers\.(\d+)\.input_layernorm\.":                   r"model.language_model.layers.\1.input_layernorm.",
    r"^model\.language_model\.layers\.(\d+)\.post_attention_layernorm\.":          r"model.language_model.layers.\1.post_attention_layernorm.",
    r"^model\.language_model\.norm\.":                                              r"model.language_model.norm.",

    # LM head
    r"^lm_head\.":                                                                  r"lm_head.",

    # Acoustic connector
    r"^model\.acoustic_connector\.":                                                r"model.acoustic_connector.",

    # Semantic connector
    r"^model\.semantic_connector\.":                                                r"model.semantic_connector.",

    # Acoustic tokenizer encoder - stem transformations
    r"^model\.acoustic_tokenizer\.encoder\.downsample_layers\.0\.0\.conv\.":       r"model.acoustic_tokenizer.encoder.stem.conv.conv.",
    r"^model\.acoustic_tokenizer\.encoder\.stages\.0\.":                           r"model.acoustic_tokenizer.encoder.stem.stage.",

    # Acoustic tokenizer encoder - conv_layers (shift indices by 1)
    r"^model\.acoustic_tokenizer\.encoder\.downsample_layers\.(\d+)\.0\.conv\.":   r"model.acoustic_tokenizer.encoder.conv_layers.PLACEHOLDER.conv.conv.",
    r"^model\.acoustic_tokenizer\.encoder\.stages\.(\d+)\.":                       r"model.acoustic_tokenizer.encoder.conv_layers.PLACEHOLDER.stage.",

    # Acoustic tokenizer encoder - mixer fixes
    r"mixer\.conv\.conv\.conv\.":                                                   r"mixer.conv.",
    r"\.conv\.conv\.conv\.":                                                        r".conv.conv.",

    # Acoustic tokenizer decoder - stem transformations
    r"^model\.acoustic_tokenizer\.decoder\.upsample_layers\.0\.0\.conv\.conv\.":   r"model.acoustic_tokenizer.decoder.stem.conv.conv.",
    r"^model\.acoustic_tokenizer\.decoder\.stages\.0\.":                           r"model.acoustic_tokenizer.decoder.stem.stage.",

    # Acoustic tokenizer decoder - conv_layers (shift indices by 1)
    r"^model\.acoustic_tokenizer\.decoder\.upsample_layers\.(\d+)\.0\.convtr\.convtr\.": r"model.acoustic_tokenizer.decoder.conv_layers.PLACEHOLDER.convtr.convtr.",
    r"^model\.acoustic_tokenizer\.decoder\.stages\.(\d+)\.":                       r"model.acoustic_tokenizer.decoder.conv_layers.PLACEHOLDER.stage.",

    # Acoustic tokenizer decoder - head fix
    r"^model\.acoustic_tokenizer\.decoder\.head\.conv\.":                          r"model.acoustic_tokenizer.decoder.head.",

    # Semantic tokenizer encoder - same pattern as acoustic (encoder-only, no decoder)
    r"^model\.semantic_tokenizer\.encoder\.downsample_layers\.0\.0\.conv\.":       r"model.semantic_tokenizer.encoder.stem.conv.conv.",
    r"^model\.semantic_tokenizer\.encoder\.stages\.0\.":                           r"model.semantic_tokenizer.encoder.stem.stage.",
    r"^model\.semantic_tokenizer\.encoder\.downsample_layers\.(\d+)\.0\.conv\.":   r"model.semantic_tokenizer.encoder.conv_layers.PLACEHOLDER.conv.conv.",
    r"^model\.semantic_tokenizer\.encoder\.stages\.(\d+)\.":                       r"model.semantic_tokenizer.encoder.conv_layers.PLACEHOLDER.stage.",
}
# fmt: on


def map_old_key_to_new(old_key: str) -> str:
    """
    Map a key from the original state dict to the equivalent key in HF format.

    Args:
        old_key: Original key from the checkpoint

    Returns:
        Mapped key for HF model
    """
    new_key = old_key

    # Apply all regex patterns
    for pattern, replacement in STATE_DICT_MAPPING.items():
        # Check if pattern matches
        match = re.search(pattern, new_key)
        if match:
            # Handle index shifts for conv_layers (downsample_layers/upsample_layers indexed from 1)
            if "PLACEHOLDER" in replacement and match.groups():
                # Extract the layer index from the match
                layer_idx = int(match.group(1))
                # Shift down by 1 since layer 0 becomes stem
                new_idx = layer_idx - 1
                # Replace PLACEHOLDER with the new index
                replacement = replacement.replace("PLACEHOLDER", str(new_idx))

            new_key = re.sub(pattern, replacement, new_key)
            # Don't break - continue applying patterns for nested fixes

    # Additional cleanup for conv layers that might not be caught by patterns
    # Handle cases where stem transformations already applied, but conv.conv needs fixing
    if "stem.conv.conv" not in new_key and "conv_layers." not in new_key:
        # Fix remaining .conv.conv. patterns (but not stem.conv.conv or conv_layers.*.conv.conv)
        new_key = re.sub(r"\.conv\.conv\.", r".conv.", new_key)

    return new_key


def convert_state_dict(original_state_dict: dict[str, Any]) -> dict[str, Any]:
    """
    Convert the original state dict keys to match the Hugging Face model structure.

    Args:
        original_state_dict: Original model state dict

    Returns:
        Converted state dict with HF-compatible keys
    """
    new_state_dict = {}

    for old_key, tensor in original_state_dict.items():
        new_key = map_old_key_to_new(old_key)
        new_state_dict[new_key] = tensor

        # Log conversions for debugging (optional)
        if old_key != new_key:
            logger.debug(f"Converted: {old_key} -> {new_key}")

    return new_state_dict


def load_original_checkpoint(checkpoint_path: str | Path) -> dict[str, Any]:
    """
    Load the original VibeVoice ASR checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint directory or file

    Returns:
        Dictionary containing the model state dict
    """
    checkpoint_path = Path(checkpoint_path)

    if checkpoint_path.is_file():
        # Single file checkpoint
        if checkpoint_path.suffix == ".safetensors":
            state_dict = load_file(str(checkpoint_path))
        else:
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            elif "model" in state_dict:
                state_dict = state_dict["model"]
    else:
        # Directory with sharded checkpoints
        # Look for model.safetensors or pytorch_model.bin
        safetensors_path = checkpoint_path / "model.safetensors"
        pytorch_path = checkpoint_path / "pytorch_model.bin"

        if safetensors_path.exists():
            state_dict = load_file(str(safetensors_path))
        elif pytorch_path.exists():
            state_dict = torch.load(pytorch_path, map_location="cpu")
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            elif "model" in state_dict:
                state_dict = state_dict["model"]
        else:
            raise FileNotFoundError(
                f"Could not find model checkpoint in {checkpoint_path}. "
                "Expected 'model.safetensors' or 'pytorch_model.bin'"
            )

    return state_dict


def create_config_from_checkpoint(checkpoint_path: str | Path) -> VibeVoiceASRConfig:
    """
    Create a VibeVoiceASRConfig from the checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint directory

    Returns:
        VibeVoiceASRConfig instance
    """
    checkpoint_path = Path(checkpoint_path)
    config_path = checkpoint_path / "config.json" if checkpoint_path.is_dir() else checkpoint_path.parent / "config.json"

    if config_path.exists():
        with open(config_path, "r") as f:
            original_config = json.load(f)

        # Process acoustic tokenizer config
        acoustic_config_dict = original_config.get("acoustic_tokenizer_config", {}).copy()

        # Parse encoder_depths string to list
        if "encoder_depths" in acoustic_config_dict and isinstance(acoustic_config_dict["encoder_depths"], str):
            acoustic_config_dict["encoder_depths"] = list(map(int, acoustic_config_dict["encoder_depths"].split("-")))

        # Rename and transform fields
        if "layernorm_eps" in acoustic_config_dict:
            acoustic_config_dict["rms_norm_eps"] = acoustic_config_dict.pop("layernorm_eps")
        if "encoder_ratios" in acoustic_config_dict:
            acoustic_config_dict["downsampling_ratios"] = list(reversed(acoustic_config_dict.pop("encoder_ratios")))
        if "encoder_n_filters" in acoustic_config_dict:
            acoustic_config_dict["n_filters"] = acoustic_config_dict.pop("encoder_n_filters")
        if "encoder_depths" in acoustic_config_dict:
            acoustic_config_dict["depths"] = acoustic_config_dict.pop("encoder_depths")
        if "vae_dim" in acoustic_config_dict:
            acoustic_config_dict["hidden_size"] = acoustic_config_dict.pop("vae_dim")
        if "conv_bias" in acoustic_config_dict:
            acoustic_config_dict["bias"] = acoustic_config_dict.pop("conv_bias")
        if "fix_std" in acoustic_config_dict:
            acoustic_config_dict["vae_std"] = acoustic_config_dict.pop("fix_std") / 0.8

        # Remove unused/decoder parameters
        for key in ["decoder_depths", "decoder_n_filters", "decoder_ratios", "std_dist_type",
                    "pad_mode", "causal", "mixer_layer", "layernorm", "disable_last_norm",
                    "conv_norm", "corpus_normalize", "layernorm_elementwise_affine"]:
            acoustic_config_dict.pop(key, None)

        acoustic_config = VibeVoiceAcousticTokenizerConfig(**acoustic_config_dict)

        # Process semantic tokenizer config (encoder-only, no decoder or VAE)
        semantic_config_dict = original_config.get("semantic_tokenizer_config", {}).copy()

        if "encoder_depths" in semantic_config_dict and isinstance(semantic_config_dict["encoder_depths"], str):
            semantic_config_dict["encoder_depths"] = list(map(int, semantic_config_dict["encoder_depths"].split("-")))

        if "layernorm_eps" in semantic_config_dict:
            semantic_config_dict["rms_norm_eps"] = semantic_config_dict.pop("layernorm_eps")
        if "encoder_ratios" in semantic_config_dict:
            semantic_config_dict["downsampling_ratios"] = list(reversed(semantic_config_dict.pop("encoder_ratios")))
        if "encoder_n_filters" in semantic_config_dict:
            semantic_config_dict["n_filters"] = semantic_config_dict.pop("encoder_n_filters")
        if "encoder_depths" in semantic_config_dict:
            semantic_config_dict["depths"] = semantic_config_dict.pop("encoder_depths")
        if "vae_dim" in semantic_config_dict:
            semantic_config_dict["hidden_size"] = semantic_config_dict.pop("vae_dim")
        if "conv_bias" in semantic_config_dict:
            semantic_config_dict["bias"] = semantic_config_dict.pop("conv_bias")

        # Remove unused parameters (including VAE and decoder parameters)
        for key in ["decoder_depths", "decoder_n_filters", "decoder_ratios",
                    "std_dist_type", "fix_std",  # No VAE component for semantic tokenizer
                    "pad_mode", "causal", "mixer_layer", "layernorm", "disable_last_norm",
                    "conv_norm", "corpus_normalize", "layernorm_elementwise_affine"]:
            semantic_config_dict.pop(key, None)

        semantic_config = VibeVoiceSemanticTokenizerConfig(**semantic_config_dict)

        # Process text/decoder config
        text_config = Qwen2Config(**original_config.get("decoder_config", {}))

        # Create main config
        config = VibeVoiceASRConfig(
            acoustic_tokenizer_config=acoustic_config,
            semantic_tokenizer_config=semantic_config,
            text_config=text_config,
            audio_token_id=original_config.get("audio_token_id", 151669),
            acoustic_vae_dim=original_config.get("acoustic_vae_dim", 64),
            semantic_vae_dim=original_config.get("semantic_vae_dim", 128),
        )
    else:
        # Use default config
        logger.warning("No config.json found, using default configuration")
        config = VibeVoiceASRConfig()

    return config


def create_processor(checkpoint_path: str | Path, output_dir: str | Path) -> VibeVoiceASRProcessor:
    """
    Create and save a VibeVoiceASRProcessor.

    Args:
        checkpoint_path: Path to the original checkpoint
        output_dir: Path to save the processor

    Returns:
        VibeVoiceASRProcessor instance
    """
    checkpoint_path = Path(checkpoint_path)

    # Load tokenizer from checkpoint or use default
    tokenizer_path = checkpoint_path if checkpoint_path.is_dir() else checkpoint_path.parent
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
    except Exception:
        logger.warning("Could not load tokenizer from checkpoint, using Qwen2 tokenizer")
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")

        # Add audio token if not present
        if "<audio>" not in tokenizer.get_vocab():
            tokenizer.add_special_tokens({"additional_special_tokens": ["<audio>"]})

    # Create processor (VibeVoice ASR processes audio at 24kHz directly)
    processor = VibeVoiceASRProcessor(
        audio_processor=None,
        tokenizer=tokenizer,
    )

    processor.save_pretrained(str(output_dir))
    logger.info(f"Saved processor to {output_dir}")

    return processor


def convert_checkpoint(
    checkpoint_path: str,
    output_dir: str,
    push_to_hub: str | None = None,
):
    """
    Convert a VibeVoice ASR checkpoint to Hugging Face format.

    Args:
        checkpoint_path: Path to the original checkpoint
        output_dir: Directory to save the converted checkpoint
        push_to_hub: Repository ID for pushing to Hub (e.g., 'username/vibevoice-asr-hf').
                     If None, only saves locally.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading checkpoint from {checkpoint_path}")
    original_state_dict = load_original_checkpoint(checkpoint_path)

    logger.info("Converting state dict")
    converted_state_dict = convert_state_dict(original_state_dict)

    logger.info("Creating config")
    config = create_config_from_checkpoint(checkpoint_path)
    config.save_pretrained(str(output_path))

    logger.info("Creating model")
    model = VibeVoiceASRForConditionalGeneration(config)

    logger.info("Loading weights into model")
    load_result = model.load_state_dict(converted_state_dict, strict=False)

    if load_result.missing_keys:
        logger.warning(f"Missing keys: {load_result.missing_keys}")
    if load_result.unexpected_keys:
        logger.warning(f"Unexpected keys: {load_result.unexpected_keys}")

    logger.info(f"Saving model to {output_path}")
    model.save_pretrained(str(output_path), safe_serialization=True)

    logger.info("Creating processor")
    create_processor(checkpoint_path, output_path)

    if push_to_hub:
        logger.info(f"Pushing to Hub: {push_to_hub}")
        model.push_to_hub(push_to_hub)
        processor = VibeVoiceASRProcessor.from_pretrained(str(output_path))
        processor.push_to_hub(push_to_hub)

    # Verify the conversion
    logger.info("Verifying conversion by reloading model")
    gc.collect()
    reloaded_model = VibeVoiceASRForConditionalGeneration.from_pretrained(str(output_path))
    logger.info("Model reloaded successfully!")

    logger.info("Conversion complete!")


"""
Conversion script to convert the original VibeVoice ASR model checkpoint to Hugging Face format.

The script handles:
1. Loading the original checkpoint (safetensors or pytorch format)
2. Converting state dict keys to match HF model structure
3. Processing acoustic and semantic tokenizer configs
4. Creating a VibeVoiceASRProcessor
5. Saving the converted model and processor
6. Optionally pushing to the Hugging Face Hub

Usage:

1) Download the VibeVoice ASR model checkpoint from microsoft/VibeVoice-ASR:
```bash
# Using huggingface-cli (recommended)
huggingface-cli download microsoft/VibeVoice-ASR --local-dir /path/to/vibevoice-asr

# Or using git-lfs
git lfs install
git clone https://huggingface.co/microsoft/VibeVoice-ASR /path/to/vibevoice-asr
```

2) Run the conversion script:
```bash
python src/transformers/models/vibevoice_asr/convert_vibevoice_asr_to_hf.py \
    --checkpoint_path /path/to/vibevoice-asr \
    --output_dir ./vibevoice_asr_hf
```

3) Test the converted model:
```python
from transformers import VibeVoiceASRForConditionalGeneration, AutoProcessor
import torch

processor = AutoProcessor.from_pretrained("./vibevoice_asr_hf")
model = VibeVoiceASRForConditionalGeneration.from_pretrained("./vibevoice_asr_hf")

# Load audio (24kHz)
audio = torch.randn(24000 * 5)  # 5 seconds of audio

inputs = processor.apply_transcription_request(audio=audio)
outputs = model.generate(**inputs, max_new_tokens=100)
transcription = processor.batch_decode(outputs[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print(transcription)
```

4) (Optional) Push to Hugging Face Hub:
```bash
python src/transformers/models/vibevoice_asr/convert_vibevoice_asr_to_hf.py \
    --checkpoint_path /path/to/vibevoice-asr \
    --output_dir ./vibevoice_asr_hf \
    --push_to_hub your-username/vibevoice-asr-hf
```

The converted checkpoint will be compatible with all Transformers features including:
- AutoModel/AutoProcessor for easy loading
- .generate() for inference
- Trainer for fine-tuning
- Integration with inference engines (vLLM, TGI, etc.)
"""


def main():
    parser = argparse.ArgumentParser(description="Convert VibeVoice ASR checkpoint to Hugging Face format")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to the original VibeVoice ASR checkpoint (directory or file)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the converted checkpoint",
    )
    parser.add_argument(
        "--push_to_hub",
        type=str,
        default=None,
        help="Repository ID for pushing to Hub (e.g., 'username/vibevoice-asr-hf'). If not provided, only saves locally.",
    )

    args = parser.parse_args()

    convert_checkpoint(
        checkpoint_path=args.checkpoint_path,
        output_dir=args.output_dir,
        push_to_hub=args.push_to_hub,
    )


if __name__ == "__main__":
    main()

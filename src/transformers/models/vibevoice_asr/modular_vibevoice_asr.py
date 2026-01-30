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

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from ...cache_utils import Cache
from ...generation import GenerationMixin
from ...modeling_outputs import CausalLMOutputWithPast, ModelOutput
from ...utils import auto_docstring, can_return_tuple
from ..auto import AutoModel, AutoModelForCausalLM
from ..mimi.modeling_mimi import MimiConv1dPaddingCache
from ..qwen2.modeling_qwen2 import Qwen2RMSNorm
from ..vibevoice_acoustic_tokenizer.modeling_vibevoice_acoustic_tokenizer import (
    VibeVoiceAcousticTokenizerModel,
    VibeVoiceAcousticTokenizerPreTrainedModel,
)
from .configuration_vibevoice_asr import VibeVoiceAsrEncoderConfig, VibeVoiceAsrConfig


class VibeVoiceAsrRMSNorm(Qwen2RMSNorm):
    pass


class VibeVoiceAsrConv1dPaddingCache(MimiConv1dPaddingCache):
    pass


class VibeVoiceAsrMultiModalProjector(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.norm = VibeVoiceAsrRMSNorm(output_dim, eps=1e-6)
        self.fc2 = nn.Linear(output_dim, output_dim)

    def forward(self, features):
        x = self.fc1(features)
        x = self.norm(x)
        x = self.fc2(x)
        return x
    

@auto_docstring
class VibeVoiceAsrPreTrainedModel(VibeVoiceAcousticTokenizerPreTrainedModel):
    config: VibeVoiceAsrConfig
    base_model_prefix = "model"
    main_input_name = "input_ids"
    _no_split_modules = ["VibeVoiceAsrEncoderLayer"]
    input_modalities = ("audio", "text")
    supports_gradient_checkpointing = True
    _skip_keys_device_placement = "past_key_values"
    _supports_cache_class = True
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_quantized_cache = True
    _supports_static_cache = True
    _supports_attention_backend = True



@dataclass
@auto_docstring
class VibeVoiceAsrEncoderOutput(ModelOutput):
    """
    latents (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
        Projected latents (continuous representations for semantic tokens) at the output of the encoder.
    padding_cache (`VibeVoiceAsrConv1dPaddingCache`, *optional*, returned when `use_cache=True` is passed):
        A [`VibeVoiceAsrConv1dPaddingCache`] instance containing cached convolution states for each layer that
        can be reused for streaming mode.
    """

    latents: torch.FloatTensor = None
    padding_cache: Optional["VibeVoiceAsrConv1dPaddingCache"] = None


@auto_docstring(
    custom_intro="""
    Tokenizer which only encodes audio into latent representations.
    """
)
class VibeVoiceAsrEncoderModel(VibeVoiceAcousticTokenizerModel):
    config: VibeVoiceAsrEncoderConfig

    def __init__(self, config):
        super().__init__(config)
        del self.decoder

    def encode(self, audio, padding_cache=None, use_cache=None):
        raise NotImplementedError("Encode method is not implemented.")

    def decode(self, latents, padding_cache=None, use_cache=False):
        raise NotImplementedError("Decode method is not implemented.")

    def forward(self, audio, padding_cache=None, use_cache=None, **kwargs):
        r"""
        audio (`torch.FloatTensor` of shape `(batch_size, channels, sequence_length)`):
            Input audio waveform to be encoded into latent representations.
        padding_cache (`VibeVoiceAsrConv1dPaddingCache`, *optional*):
            Cache object for streaming mode to maintain convolution states across layers.
        use_cache (`bool`, *optional*):
            Whether to use caching for convolution states.
        """
        if use_cache and padding_cache is None:
            padding_cache = VibeVoiceAsrConv1dPaddingCache(
                num_layers=self.encoder.num_conv_layers,
                per_layer_padding=self.encoder.per_conv_layer_padding,
                per_layer_padding_mode=self.encoder.per_conv_layer_padding_mode,
                per_layer_in_channels=self.encoder.per_conv_layer_in_channels,
            )
        latents = self.encoder(audio, padding_cache=padding_cache)

        return VibeVoiceAsrEncoderOutput(
            latents=latents,
            padding_cache=padding_cache if use_cache else None,
        )



@auto_docstring(
    custom_intro="""
    The VibeVoice ASR model with a language modeling head for conditional generation (ASR tasks).
    """
)
# TODO modular from Voxtral or AudioFlamingo3? for all the helper methods
class VibeVoiceAsrForConditionalGeneration(VibeVoiceAsrPreTrainedModel, GenerationMixin):

    def __init__(self, config: VibeVoiceAsrConfig):
        super().__init__(config)
        self.vocab_size = config.text_config.vocab_size
        self.acoustic_tokenizer = AutoModel.from_config(config.acoustic_tokenizer_config)
        self.semantic_tokenizer = AutoModel.from_config(config.semantic_tokenizer_config)
        self.acoustic_connector = VibeVoiceAsrMultiModalProjector(config.acoustic_tokenizer_config.hidden_size, config.text_config.hidden_size)
        self.semantic_connector = VibeVoiceAsrMultiModalProjector(config.semantic_tokenizer_config.hidden_size, config.text_config.hidden_size)
        self.language_model = AutoModelForCausalLM.from_config(config.text_config)
        self.post_init()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.language_model.set_output_embeddings(new_embeddings)

    def set_decoder(self, decoder):
        self.language_model.set_decoder(decoder)

    def get_decoder(self):
        return self.language_model.get_decoder()

    @can_return_tuple
    @auto_docstring(
        custom_intro="This method is used to encode audio input into embedding that can be used by the language model."
    )
    def get_audio_features(
        self,
        input_values: torch.FloatTensor,
        padding_mask: torch.BoolTensor | None = None,
    ) -> torch.FloatTensor:
        r"""
        input_values (`torch.FloatTensor` of shape `(batch_size, num_samples)`):
            Input audio tensor. Audio should be sampled at 24kHz.
        padding_mask (`torch.BoolTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing operations on padding feature indices.
        """

        total_samples = input_values.shape[-1]

        # Calculate segment size in samples
        # TODO rename as seems like from tokenizer rather than acoustic tokenizer
        segment_samples = self.config.tokenizer_chunk_size

        # adjust padding mask according to tokenizer compression
        num_audio_tokens = torch.ceil(padding_mask.sum(dim=-1) / self.acoustic_tokenizer.config.hop_length).to(
            torch.int64
        )
        padding_mask = torch.arange(max(num_audio_tokens)) < num_audio_tokens[:, None].cpu()

        # Using pre-trained tokenizers to extract acoustic and semantic features
        with torch.no_grad():
            acoustic_encoder_cache = None
            semantic_encoder_cache = None
            acoustic_latents = []
            semantic_latents = []

            def _iter_segments(total_length: int, segment_length: int):
                """Iterate over audio segments with a given segment length."""
                if segment_length <= 0:
                    raise ValueError("segment_length must be positive")
                for start in range(0, total_length, segment_length):
                    end = min(start + segment_length, total_length)
                    if end > start:
                        yield start, end

            # Process each segment
            segments = list(_iter_segments(total_samples, segment_samples))
            for seg_idx, (start, end) in enumerate(segments):
                chunk = input_values[:, start:end].contiguous()
                if chunk.numel() == 0:
                    continue

                # Encode chunk for acoustic tokenizer
                acoustic_encoder_output = self.acoustic_tokenizer(
                    chunk, padding_cache=acoustic_encoder_cache, use_cache=True,
                )
                acoustic_latents.append(acoustic_encoder_output.latents)
                acoustic_encoder_cache = acoustic_encoder_output.padding_cache

                # Encode chunk for semantic tokenizer
                semantic_encoder_output = self.semantic_tokenizer(
                    chunk, padding_cache=semantic_encoder_cache, use_cache=True,
                )
                semantic_latents.append(semantic_encoder_output.latents)
                semantic_encoder_cache = semantic_encoder_output.padding_cache

            # Concatenate all means and sample once
            acoustic_mean_full = torch.cat(acoustic_latents, dim=1).contiguous()

            # sample acoustic tokens
            noise_std = self.config.acoustic_vae_std * torch.randn(
                acoustic_mean_full.shape[0], device=acoustic_mean_full.device, dtype=acoustic_mean_full.dtype
            )
            acoustic_mean_full = acoustic_mean_full + noise_std[:, None, None] * torch.randn_like(acoustic_mean_full)

            # Project acoustic tokens
            acoustic_features = self.acoustic_connector(acoustic_mean_full)

            # Concatenate all semantic means
            semantic_tokens = torch.cat(semantic_latents, dim=1).contiguous()
            semantic_features = self.semantic_connector(semantic_tokens)

            # Combine acoustic and semantic features
            if padding_mask is not None:
                combined_features = acoustic_features[padding_mask] + semantic_features[padding_mask]
            else:
                combined_features = acoustic_features + semantic_features

        return combined_features

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        input_values: torch.FloatTensor | None = None,
        padding_mask: torch.BoolTensor | None = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        padding_mask (`torch.BoolTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing operations on padding feature indices.

        Example:

        ```python
        >>> from transformers import VibeVoiceAsrForConditionalGeneration, AutoProcessor
        >>> import torch

        >>> model_id = "microsoft/VibeVoice-ASR"
        >>> processor = AutoProcessor.from_pretrained(model_id)
        >>> model = VibeVoiceAsrForConditionalGeneration.from_pretrained(model_id, device_map="auto")

        >>> # Prepare audio input
        >>> audio = torch.randn(16000 * 5)  # 5 seconds of audio at 16kHz
        >>> text = "<audio>Transcribe the following audio."

        >>> inputs = processor(text=text, audio=audio, return_tensors="pt").to(model.device)
        >>> outputs = model.generate(**inputs, max_new_tokens=100)
        >>> transcription = processor.batch_decode(outputs[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        >>> print(transcription)
        ```"""

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        if input_values is not None and input_ids is not None:
            audio_embeds = self.get_audio_features(input_values=input_values, padding_mask=padding_mask)

            # replace text-audio token placeholders with audio embeddings
            audio_token_mask = (input_ids == self.config.audio_token_id).unsqueeze(-1)
            inputs_embeds = inputs_embeds.masked_scatter(
                audio_token_mask.to(inputs_embeds.device), audio_embeds.to(inputs_embeds.device)
            )

        outputs: CausalLMOutputWithPast = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            labels=labels,
            use_cache=use_cache,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            **kwargs,
        )
        return outputs

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        input_values=None,
        padding_mask=None,
        **kwargs,
    ):
        """
        Prepare inputs for generation step. Speech inputs are only forwarded on the first pass
        (when cache_position[0] == 0), and are excluded in subsequent generation steps.
        """
        # If we have past key values, we only need to process the new tokens
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                past_length = past_key_values.get_seq_length()
            else:
                past_length = past_key_values[0][0].shape[2]

            # Keep only the new tokens
            if input_ids is not None and input_ids.shape[1] > past_length:
                input_ids = input_ids[:, past_length:]

        # Prepare position ids
        if position_ids is None and attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values is not None and input_ids is not None:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # Prepare cache position
        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + (input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]),
                device=input_ids.device if input_ids is not None else inputs_embeds.device,
            )

        # Prepare model inputs
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
            }
        )

        if cache_position is not None and len(cache_position) > 0 and cache_position[0] == 0:
            if input_values is not None:
                model_inputs["input_values"] = input_values
            if padding_mask is not None:
                model_inputs["padding_mask"] = padding_mask

        return model_inputs


__all__ = ["VibeVoiceAsrForConditionalGeneration", "VibeVoiceAsrPreTrainedModel", "VibeVoiceAsrEncoderModel"]

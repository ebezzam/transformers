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

from ...activations import ACT2FN
from ...cache_utils import Cache
from ...generation import GenerationMixin
from ...modeling_outputs import CausalLMOutputWithPast, ModelOutput
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple
from ...utils.generic import check_model_inputs
from ..auto import AutoModel, AutoModelForCausalLM
from ..mimi.modeling_mimi import MimiConv1dPaddingCache
from ..qwen2.modeling_qwen2 import Qwen2Model, Qwen2RMSNorm
from ..vibevoice_acoustic_tokenizer.modeling_vibevoice_acoustic_tokenizer import (
    VibeVoiceAcousticTokenizerEncoder,
    VibeVoiceAcousticTokenizerModel,
    VibeVoiceAcousticTokenizerPreTrainedModel,
)
from .configuration_vibevoice_asr import VibeVoiceASRConfig, VibeVoiceSemanticTokenizerConfig


class VibeVoiceRMSNorm(Qwen2RMSNorm):
    pass


class VibeVoiceASRConv1dPaddingCache(MimiConv1dPaddingCache):
    pass


class VibeVoiceASRMultiModelProjector(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.norm = VibeVoiceRMSNorm(output_dim, eps=1e-6)
        self.fc2 = nn.Linear(output_dim, output_dim)

    def forward(self, features):
        x = self.fc1(features)
        x = self.norm(x)
        x = self.fc2(x)
        return x
    

@auto_docstring
class VibeVoiceASRPreTrainedModel(VibeVoiceAcousticTokenizerPreTrainedModel):
    config: VibeVoiceASRConfig
    base_model_prefix = "model"
    main_input_name = "input_ids"
    _no_split_modules = ["VibeVoiceEncoderLayer"]
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
class VibeVoiceASRSemanticTokenizerOutput(ModelOutput):
    """
    latents (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
        Projected latents (continuous representations for semantic tokens) at the output of the encoder.
    padding_cache (`VibeVoiceASRConv1dPaddingCache`, *optional*, returned when `use_cache=True` is passed):
        A [`VibeVoiceASRConv1dPaddingCache`] instance containing cached convolution states for each layer that
        can be reused for streaming mode.
    """

    latents: torch.FloatTensor = None
    padding_cache: Optional["VibeVoiceASRConv1dPaddingCache"] = None


@auto_docstring(
    custom_intro="""
    Semantic tokenizer which only encodes audio into semantic tokens, namely no decoding.
    """
)
class VibeVoiceASRSemanticTokenizerModel(VibeVoiceAcousticTokenizerModel):
    config: VibeVoiceSemanticTokenizerConfig
    base_model_prefix = "vibevoice_asr_semantic_tokenizer"
    main_input_name = "audio"
    _no_split_modules = ["VibeVoiceASRSemanticTokenizerEncoder"]

    def __init__(self, config):
        super().__init__(config)
        del self.decoder

    @can_return_tuple
    @auto_docstring
    def encode(self, audio, padding_cache=None, use_cache=None):
        r"""
        audio (`torch.FloatTensor` of shape `(batch_size, channels, sequence_length)`):
            Input audio waveform to be encoded into latent representations.
        padding_cache (`VibeVoiceASRConv1dPaddingCache`, *optional*):
            Cache object for streaming mode to maintain convolution states across layers.
        use_cache (`bool`, *optional*):
            Whether to use caching for convolution states.
        """
        if use_cache and padding_cache is None:
            padding_cache = VibeVoiceASRConv1dPaddingCache(
                num_layers=self.encoder.num_conv_layers,
                per_layer_padding=self.encoder.per_conv_layer_padding,
                per_layer_padding_mode=self.encoder.per_conv_layer_padding_mode,
                per_layer_in_channels=self.encoder.per_conv_layer_in_channels,
            )
        latents = self.encoder(audio, padding_cache=padding_cache)

        return VibeVoiceASRSemanticTokenizerOutput(
            latents=latents,
            padding_cache=padding_cache if use_cache else None,
        )

    def sample(self):
        raise NotImplementedError("Sample method is not implemented for VibeVoiceASRSemanticTokenizerModel.")

    def decode(self, latents, padding_cache=None, use_cache=False):
        raise NotImplementedError("Decode method is not implemented for VibeVoiceASRSemanticTokenizerModel.")

    def forward(self, audio, padding_cache=None, use_cache=None, **kwargs: Unpack[TransformersKwargs]):
        raise NotImplementedError("Forward method is not implemented for VibeVoiceASRSemanticTokenizerModel.")



@auto_docstring(
    custom_intro="""
    The VibeVoice ASR model with a language modeling head for conditional generation (ASR tasks).
    """
)
# TODO modular from Voxtral or AudioFlamingo3?
class VibeVoiceASRForConditionalGeneration(VibeVoiceASRPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: VibeVoiceASRConfig):
        super().__init__(config)
        self.vocab_size = config.text_config.vocab_size
        self.acoustic_tokenizer = AutoModel.from_config(config.acoustic_tokenizer_config)
        self.semantic_tokenizer = AutoModel.from_config(config.semantic_tokenizer_config)
        self.acoustic_connector = VibeVoiceASRMultiModelProjector(config.acoustic_tokenizer_config.hidden_size, config.text_config.hidden_size)
        self.semantic_connector = VibeVoiceASRMultiModelProjector(config.semantic_tokenizer_config.hidden_size, config.text_config.hidden_size)
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

    def tie_weights(self):
        """Tie the weights between the input embeddings and the output embeddings."""
        if getattr(self.config.text_config, "tie_word_embeddings", False):
            output_embeddings = self.get_output_embeddings()
            input_embeddings = self.get_input_embeddings()
            if hasattr(input_embeddings, "weight"):
                output_embeddings.weight = input_embeddings.weight
            else:
                output_embeddings.weight = input_embeddings

    @can_return_tuple
    @auto_docstring(
        custom_intro="This method is used to encode audio input into features that can be used by the language model."
    )
    def encode_speech(
        self,
        speech_tensors: torch.FloatTensor,
        speech_masks: torch.BoolTensor | None = None,
        speech_semantic_tensors: torch.FloatTensor | None = None,
        streaming_segment_duration: float = 60.0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.FloatTensor:
        r"""
        speech_tensors (`torch.FloatTensor` of shape `(batch_size, num_samples)`):
            Input audio tensor. Audio should be sampled at 24kHz.
        speech_masks (`torch.BoolTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing operations on padding feature indices.
        speech_semantic_tensors (`torch.FloatTensor`, *optional*):
            Pre-computed semantic tokens. If provided, semantic encoding is skipped.
        streaming_segment_duration (`float`, *optional*, defaults to 60.0):
            Segment duration in seconds for streaming processing of long audio.
        """
        # Determine dtype
        if hasattr(self.config, "torch_dtype") and self.config.torch_dtype is not None:
            if isinstance(self.config.torch_dtype, str):
                dtype = getattr(torch, self.config.torch_dtype)
            else:
                dtype = self.config.torch_dtype
        else:
            dtype = torch.float32

        speech_tensors = speech_tensors.to(dtype)

        # Ensure proper shape: (batch, samples)
        if speech_tensors.ndim == 1:
            speech_tensors = speech_tensors.unsqueeze(0)

        batch_size, total_samples = speech_tensors.shape
        sample_rate = 24000  # VibeVoice uses 24kHz

        # Calculate segment size in samples
        segment_samples = int(streaming_segment_duration * sample_rate)

        # Decide whether to use streaming based on audio length
        use_streaming = total_samples > segment_samples

        with torch.no_grad():
            if not use_streaming:
                # Short audio: direct processing
                encoder_output = self.model.acoustic_tokenizer.encode(speech_tensors.unsqueeze(1))
                audio_tokens = encoder_output.sample(dist_type=self.model.acoustic_tokenizer.std_dist_type)[0]
                acoustic_features = self.model.acoustic_connector(audio_tokens)

                # Encode semantic features
                if speech_semantic_tensors is not None:
                    semantic_features = self.model.semantic_connector(speech_semantic_tensors)
                else:
                    semantic_tokens = self.model.semantic_tokenizer.encode(speech_tensors.unsqueeze(1)).mean
                    semantic_features = self.model.semantic_connector(semantic_tokens)
            else:
                # Long audio: streaming processing
                # Import streaming cache from vibevoice_acoustic_tokenizer
                from ..vibevoice_acoustic_tokenizer.modeling_vibevoice_acoustic_tokenizer import (
                    VibeVoiceTokenizerStreamingCache,
                    VibeVoiceTokenizerEncoderOutput,
                )

                acoustic_encoder_cache = VibeVoiceTokenizerStreamingCache()
                semantic_encoder_cache = VibeVoiceTokenizerStreamingCache()
                acoustic_mean_segments = []
                semantic_mean_segments = []
                sample_indices = torch.arange(batch_size, device=speech_tensors.device)

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
                num_segments = len(segments)
                for seg_idx, (start, end) in enumerate(segments):
                    chunk = speech_tensors[:, start:end].contiguous()
                    if chunk.numel() == 0:
                        continue

                    is_final = seg_idx == num_segments - 1

                    # Encode chunk for acoustic tokenizer
                    acoustic_encoder_output = self.model.acoustic_tokenizer.encode(
                        chunk.unsqueeze(1),
                        cache=acoustic_encoder_cache,
                        sample_indices=sample_indices,
                        use_cache=True,
                        is_final_chunk=is_final,
                    )
                    acoustic_mean_segments.append(acoustic_encoder_output.mean)

                    # Encode chunk for semantic tokenizer
                    semantic_encoder_output = self.model.semantic_tokenizer.encode(
                        chunk.unsqueeze(1),
                        cache=semantic_encoder_cache,
                        sample_indices=sample_indices,
                        use_cache=True,
                        is_final_chunk=is_final,
                    )
                    semantic_mean_segments.append(semantic_encoder_output.mean)

                # Concatenate all means and sample once
                acoustic_mean_full = torch.cat(acoustic_mean_segments, dim=1).contiguous()
                acoustic_encoder_output = VibeVoiceTokenizerEncoderOutput(
                    mean=acoustic_mean_full, std=self.model.acoustic_tokenizer.fix_std
                )
                audio_tokens = acoustic_encoder_output.sample(dist_type=self.model.acoustic_tokenizer.std_dist_type)[
                    0
                ]
                acoustic_features = self.model.acoustic_connector(audio_tokens)

                # Concatenate all semantic means
                semantic_tokens = torch.cat(semantic_mean_segments, dim=1).contiguous()
                semantic_features = self.model.semantic_connector(semantic_tokens)

            # Combine acoustic and semantic features
            if speech_masks is not None:
                combined_features = acoustic_features[speech_masks] + semantic_features[speech_masks]
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
        # Speech-specific arguments
        speech_tensors: torch.FloatTensor | None = None,
        speech_masks: torch.BoolTensor | None = None,
        speech_semantic_tensors: torch.FloatTensor | None = None,
        acoustic_input_mask: torch.BoolTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        speech_tensors (`torch.FloatTensor` of shape `(batch_size, num_samples)`, *optional*):
            Input audio waveform tensor. Audio should be sampled at 24kHz.
        speech_masks (`torch.BoolTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing operations on padding feature indices.
        speech_semantic_tensors (`torch.FloatTensor`, *optional*):
            Pre-computed semantic tokens.
        acoustic_input_mask (`torch.BoolTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask indicating which positions in the input should be filled with acoustic features.

        Example:

        ```python
        >>> from transformers import VibeVoiceASRForConditionalGeneration, AutoProcessor
        >>> import torch

        >>> model_id = "microsoft/VibeVoice-ASR"
        >>> processor = AutoProcessor.from_pretrained(model_id)
        >>> model = VibeVoiceASRForConditionalGeneration.from_pretrained(model_id, device_map="auto")

        >>> # Prepare audio input
        >>> audio = torch.randn(16000 * 5)  # 5 seconds of audio at 16kHz
        >>> text = "<audio>Transcribe the following audio."

        >>> inputs = processor(text=text, audio=audio, return_tensors="pt").to(model.device)
        >>> outputs = model.generate(**inputs, max_new_tokens=100)
        >>> transcription = processor.batch_decode(outputs[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        >>> print(transcription)
        ```"""

        return_dict = kwargs.get("return_dict", True)
        use_cache = use_cache if use_cache is not None else getattr(self.config, "use_cache", False)

        # Process inputs
        if inputs_embeds is None and input_ids is not None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        # If we have speech input and acoustic_input_mask, encode and insert speech features
        if speech_tensors is not None and acoustic_input_mask is not None:
            speech_features = self.encode_speech(
                speech_tensors=speech_tensors,
                speech_masks=speech_masks,
                speech_semantic_tensors=speech_semantic_tensors,
            )
            # Clone to avoid in-place operation on leaf variable during training
            inputs_embeds = inputs_embeds.clone()
            inputs_embeds[acoustic_input_mask] = speech_features

        # Forward through the model
        outputs = self.model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs[0] if not return_dict else outputs.last_hidden_state

        # Compute logits
        if self.config.text_config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.text_config.pretraining_tp, dim=0)
            logits = [nn.functional.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.text_config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            if logits_to_keep:
                hidden_states = hidden_states[..., -logits_to_keep:, :]
            logits = self.lm_head(hidden_states)

        logits = logits.to(torch.float32)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            shift_logits = shift_logits.view(-1, self.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        speech_tensors=None,
        speech_masks=None,
        speech_semantic_tensors=None,
        acoustic_input_mask=None,
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

        # Only include speech inputs on the first forward pass
        if cache_position is not None and len(cache_position) > 0 and cache_position[0] == 0:
            # First forward pass - include speech inputs if provided
            model_inputs.update(
                {
                    "speech_tensors": speech_tensors,
                    "speech_masks": speech_masks,
                    "speech_semantic_tensors": speech_semantic_tensors,
                    "acoustic_input_mask": acoustic_input_mask,
                }
            )
        else:
            # Subsequent generation steps - exclude speech inputs
            model_inputs.update(
                {
                    "speech_tensors": None,
                    "speech_masks": None,
                    "speech_semantic_tensors": None,
                    "acoustic_input_mask": None,
                }
            )

        return model_inputs


__all__ = [
    "VibeVoiceASRForConditionalGeneration",
    "VibeVoiceASRPreTrainedModel",
    "VibeVoiceASRModel",
    "VibeVoiceASRSemanticTokenizerModel",
]

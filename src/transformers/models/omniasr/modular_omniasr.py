# coding=utf-8
# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

from typing import Optional, Union

import torch
from torch import nn

from ...activations import ACT2FN
from ...cache_utils import Cache
from ...generation import GenerationMixin
from ...integrations.deepspeed import is_deepspeed_zero3_enabled
from ...integrations.fsdp import is_fsdp_managed_module
from ...masking_utils import create_bidirectional_mask
from ...modeling_outputs import (
    BaseModelOutput,
    CausalLMOutput,
    CausalLMOutputWithPast,
    Wav2Vec2BaseModelOutput,
)
from ...modeling_utils import PreTrainedModel
from ...utils import (
    auto_docstring,
    can_return_tuple,
)
from ..auto import AutoModel, AutoModelForCausalLM
from ..wav2vec2.modeling_wav2vec2 import Wav2Vec2Encoder, Wav2Vec2EncoderLayer, Wav2Vec2Model, Wav2Vec2PreTrainedModel
from .configuration_omniasr import OmniASRCTCConfig, OmniASRLLMConfig


# Different from Wav2Vec2PositionalConvEmbedding: no weight norm, has residual, uses remove_pad instead of SamePadLayer
class OmniASRPositionalConvEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.conv = nn.Conv1d(
            config.hidden_size,
            config.hidden_size,
            kernel_size=config.num_conv_pos_embeddings,
            padding=config.num_conv_pos_embeddings // 2,
            groups=config.num_conv_pos_embedding_groups,
        )
        self.remove_pad = config.num_conv_pos_embeddings % 2 == 0
        self.activation = ACT2FN[config.feat_extract_activation]

    def forward(self, hidden_states):
        residual = hidden_states
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = self.conv(hidden_states)
        if self.remove_pad:
            hidden_states = hidden_states[:, :, :-1]
        hidden_states = self.activation(hidden_states)
        hidden_states = hidden_states.transpose(1, 2)
        return hidden_states + residual


class OmniASREncoderLayer(Wav2Vec2EncoderLayer):
    # NOTE: original: https://github.com/facebookresearch/fairseq2/blob/a1f0c565a99d3cd3e3157678b5c48653e3d439f4/src/fairseq2/models/transformer/encoder_layer.py#L141
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        output_attentions=False,
    ):
        # Self-attention block with pre-norm
        attn_residual = hidden_states
        hidden_states = self.layer_norm(hidden_states)  # Pre-norm: normalize BEFORE attention
        hidden_states, attn_weights, _ = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = self.dropout(hidden_states)
        hidden_states = attn_residual + hidden_states  # Add residual

        # FFN block with pre-norm
        ffn_residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)  # Pre-norm: normalize BEFORE FFN
        hidden_states = self.feed_forward(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = ffn_residual + hidden_states  # Add residual

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class OmniASREncoder(Wav2Vec2Encoder):
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        if attention_mask is not None:
            # make sure padded tokens output 0
            expand_attention_mask = attention_mask.unsqueeze(-1).repeat(1, 1, hidden_states.shape[2])
            hidden_states[~expand_attention_mask] = 0

        attention_mask = create_bidirectional_mask(
            config=self.config,
            input_embeds=hidden_states,
            attention_mask=attention_mask,
        )

        # NOTE (ebezzam): residual and layer norm removed here (wrt Wav2Vec2Encoder)
        hidden_states = self.pos_conv_embed(hidden_states)
        synced_gpus = is_deepspeed_zero3_enabled() or is_fsdp_managed_module(self)

        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # add LayerDrop (see https://huggingface.co/papers/1909.11556 for description)
            dropout_probability = torch.rand([])

            skip_the_layer = self.training and dropout_probability < self.config.layerdrop
            if not skip_the_layer or synced_gpus:
                # under fsdp or deepspeed zero3 all gpus must run in sync
                layer_outputs = layer(
                    hidden_states, attention_mask=attention_mask, output_attentions=output_attentions
                )
                hidden_states = layer_outputs[0]

            if skip_the_layer:
                layer_outputs = (None, None)

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # NOTE (ebezzam): layer norm shifted here (wrt Wav2Vec2Encoder)
        hidden_states = self.layer_norm(hidden_states)
        if self.training:
            hidden_states = self.dropout(hidden_states)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class OmniASRPreTrainedModel(Wav2Vec2PreTrainedModel):
    def _init_weights(self, module):
        # TODO upstream change to `Wav2Vec2PreTrainedModel` for standard layers?
        PreTrainedModel._init_weights(self, module)

    def apply_weight_norm(self, legacy=True):
        weight_norm = nn.utils.weight_norm
        if hasattr(nn.utils.parametrizations, "weight_norm") and not legacy:
            weight_norm = nn.utils.parametrizations.weight_norm

        if is_deepspeed_zero3_enabled():
            import deepspeed

            with deepspeed.zero.GatheredParameters(self.encoder.encoder.pos_conv_embed.conv.weight, modifier_rank=0):
                weight_norm(self.encoder.encoder.pos_conv_embed.conv, name="weight", dim=2)
            if hasattr(self.encoder.encoder.pos_conv_embed.conv, "parametrizations"):
                weight_g = self.encoder.encoder.pos_conv_embed.conv.parametrizations.weight.original0
                weight_v = self.encoder.encoder.pos_conv_embed.conv.parametrizations.weight.original1
            else:
                weight_g = self.encoder.encoder.pos_conv_embed.conv.weight_g
                weight_v = self.encoder.encoder.pos_conv_embed.conv.weight_v
            deepspeed.zero.register_external_parameter(self.encoder.encoder.pos_conv_embed, weight_v)
            deepspeed.zero.register_external_parameter(self.encoder.encoder.pos_conv_embed, weight_g)
        else:
            weight_norm(self.encoder.encoder.pos_conv_embed.conv, name="weight", dim=2)

    def remove_weight_norm(self, legacy=True):
        remove_weight_norm = nn.utils.remove_weight_norm
        if hasattr(nn.utils.parametrizations, "weight_norm") and not legacy:
            remove_weight_norm = torch.nn.utils.parametrize.remove_parametrizations

        # TODO deepspeed zero3 case
        remove_weight_norm(self.encoder.encoder.pos_conv_embed.conv, name="weight")


class OmniASRBaseModelOutput(Wav2Vec2BaseModelOutput):
    pass


class OmniASRModel(Wav2Vec2Model):
    pass


@auto_docstring(
    custom_intro="""
    OmniASR Model with a head for Connectionist Temporal Classification (CTC).
    """
)
class OmniASRForCTC(OmniASRPreTrainedModel):
    config: OmniASRCTCConfig

    def __init__(self, config: OmniASRCTCConfig):
        super().__init__(config)
        self.encoder = OmniASRModel(config.encoder_config)
        self.ctc_head = nn.Linear(config.hidden_size, config.vocab_size)
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_values: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[tuple, CausalLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, target_length)`, *optional*):
            Labels for connectionist temporal classification. Note that `target_length` has to be smaller or equal to
            the sequence length of the output logits. Indices are selected in `[-100, 0, ..., config.vocab_size - 1]`.
            All labels set to `-100` are ignored (masked), the loss is only computed for labels in `[0, ...,
            config.vocab_size - 1]`.
        """

        if labels is not None and labels.max() >= self.config.vocab_size:
            raise ValueError(f"Label values must be <= vocab_size: {self.config.vocab_size}")

        outputs = self.encoder(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        hidden_states = outputs[0]
        logits = self.ctc_head(hidden_states)

        loss = None
        if labels is not None:
            # TODO use attention mask from encoder (see updated Parakeet)
            # retrieve loss input_lengths from attention_mask
            attention_mask = (
                attention_mask if attention_mask is not None else torch.ones_like(input_values, dtype=torch.long)
            )
            input_lengths = self.encoder._get_feat_extract_output_lengths(attention_mask.sum(-1)).to(torch.long)

            # assuming that padded tokens are filled with -100 when not being attended to
            labels_mask = labels >= 0
            target_lengths = labels_mask.sum(-1)
            flattened_targets = labels.masked_select(labels_mask)

            # ctc_loss doesn't support fp16
            log_probs = nn.functional.log_softmax(logits, dim=-1, dtype=torch.float32).transpose(0, 1)

            with torch.backends.cudnn.flags(enabled=False):
                loss = nn.functional.ctc_loss(
                    log_probs,
                    flattened_targets,
                    input_lengths,
                    target_lengths,
                    blank=self.config.pad_token_id,
                    reduction=self.config.ctc_loss_reduction,
                    zero_infinity=self.config.ctc_zero_infinity,
                )

        return CausalLMOutput(
            loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions
        )


@auto_docstring(
    custom_intro="""
    OmniASR model, which consists of a Wav2Vec2, a multi-modal projector and a LLama language model.
    """
)
class OmniASRForConditionalGeneration(OmniASRPreTrainedModel, GenerationMixin):
    config: OmniASRLLMConfig
    main_input_name = "input_ids"

    def __init__(self, config):
        super().__init__(config)
        self.vocab_size = config.vocab_size
        self.language_token_id = config.language_token_id
        if config.num_special_tokens > 0:
            reserved_language_token_id = config.vocab_size - config.num_special_tokens
            if self.language_token_id < reserved_language_token_id:
                self.language_token_id = reserved_language_token_id
        # Streaming segment markers are allocated right after the (possibly bumped) language marker token, so they are
        # derived from the resolved `self.language_token_id` to stay consistent with the non-streaming forward path.
        self.last_segment_token_id = self.language_token_id + 1
        self.regular_segment_token_id = self.language_token_id + 2

        self.encoder = AutoModel.from_config(config.encoder_config)
        self.language_model = AutoModelForCausalLM.from_config(config.text_config)

        self.multi_modal_projector = nn.Linear(
            config.encoder_config.hidden_size * config.encoder_stacking,
            config.text_config.hidden_size,
            bias=True,
        )

        self.lang_embeddings = nn.Embedding(config.num_language_embeddings, config.text_config.hidden_size)

        self.post_init()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.language_model.set_output_embeddings(new_embeddings)

    def get_audio_features(self, input_features: torch.FloatTensor, attention_mask: Optional[torch.Tensor] = None):
        # Pass the audio attention mask through to the encoder: its CNN frontend + bidirectional pre-norm
        # Transformer must not attend over padded frames, otherwise batched unequal-length audio silently
        # contaminates the valid-frame embeddings (the CTC path already forwards the mask).
        audio_outputs = self.encoder(input_features, attention_mask=attention_mask)
        audio_hidden_states = audio_outputs.last_hidden_state
        # `encoder_stacking` stacks consecutive encoder frames along the feature dimension (e.g. the zero-shot model
        # uses 3); the multi-modal projector's input dimension already accounts for this. For the default stacking of
        # 1 this is a no-op.
        stacking = self.config.encoder_stacking
        if stacking > 1:
            batch_size, num_frames, hidden_size = audio_hidden_states.shape
            pad = (-num_frames) % stacking
            if pad:
                audio_hidden_states = nn.functional.pad(audio_hidden_states, (0, 0, 0, pad))
            audio_hidden_states = audio_hidden_states.reshape(
                batch_size, (num_frames + pad) // stacking, hidden_size * stacking
            )
        audio_embeds = self.multi_modal_projector(audio_hidden_states)
        return audio_embeds

    def _build_audio_context_attention_mask(self, audio_embeds, audio_attention_mask):
        # Build the decoder attention mask for the assembled `audio | lid_marker | lang_id | bos` context. The
        # incoming `audio_attention_mask` is at the raw-waveform resolution, so it is down-sampled to the audio-frame
        # resolution (otherwise padded audio in a batch would be attended to), then the 3 context tokens are marked
        # valid. The target/eos positions (training) are appended by the caller.
        batch_size, num_frames = audio_embeds.shape[:2]
        device = audio_embeds.device
        if audio_attention_mask is None:
            frame_mask = torch.ones(batch_size, num_frames, dtype=torch.long, device=device)
        else:
            valid_frames = self.encoder._get_feat_extract_output_lengths(audio_attention_mask.sum(-1)).to(torch.long)
            # `get_audio_features` stacks `encoder_stacking` frames together, so the number of audio embeddings is
            # `ceil(frames / encoder_stacking)`; scale the valid-frame counts to that same post-stacking resolution.
            stacking = self.config.encoder_stacking
            if stacking > 1:
                valid_frames = (valid_frames + stacking - 1) // stacking
            frame_mask = (torch.arange(num_frames, device=device)[None, :] < valid_frames[:, None]).long()
        context_mask = torch.ones(batch_size, 3, dtype=torch.long, device=device)
        return torch.cat([frame_mask, context_mask], dim=1)

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        input_values: Optional[torch.Tensor] = None,
        language_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        **kwargs,
    ) -> Union[tuple, CausalLMOutputWithPast]:
        r"""
        input_values (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Float values of the raw audio waveform, obtained from an audio file via [`OmniASRProcessor`] /
            [`OmniASRFeatureExtractor`]. The audio embeddings produced by the encoder are prepended to the text
            context before being passed to the language model.
        language_ids (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Integer language IDs that condition generation through the learned language-embedding table (a value of
            `0` selects the language-agnostic setting). These are produced by [`OmniASRProcessor`] from language codes
            such as `"eng_Latn"`.

        The forward builds the input embeddings following the original implementation's syntax
        `audio | lid_marker | lang_id | bos | [target_text | eos]`
        (see https://github.com/facebookresearch/omnilingual-asr/blob/main/src/omnilingual_asr/models/wav2vec2_llama/model.py#L141).
        """

        if inputs_embeds is None and input_ids is not None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        if input_values is not None:
            # First step: build full audio context (audio | lid_marker | lang_id | bos)
            batch_size = input_values.size(0)
            device = input_values.device

            audio_embeds = self.get_audio_features(input_values, attention_mask=attention_mask)
            dtype = audio_embeds.dtype

            language_id_token_batch = torch.full(
                (batch_size, 1), self.language_token_id, dtype=torch.long, device=device
            )
            bos_batch = torch.full((batch_size, 1), self.config.bos_token_id, dtype=torch.long, device=device)

            if language_ids is not None:
                language_id_batch = language_ids.to(device)
            else:
                language_id_batch = torch.zeros(batch_size, dtype=torch.long, device=device)

            if self.training and self.config.language_embedding_probability > 0.0:
                dropout_mask = torch.rand(batch_size, device=device) < (1 - self.config.language_embedding_probability)
                language_id_batch = language_id_batch.clone()
                language_id_batch[dropout_mask] = 0

            text_embed_fn = self.get_input_embeddings()
            lid_marker_embeds = text_embed_fn(language_id_token_batch).to(dtype)
            bos_embeds = text_embed_fn(bos_batch).to(dtype)
            lang_id_embeds = self.lang_embeddings(language_id_batch.unsqueeze(-1)).to(dtype)

            inputs_embeds = torch.cat([audio_embeds, lid_marker_embeds, lang_id_embeds, bos_embeds], dim=1)
            # The incoming `attention_mask` (if any) is the raw-waveform audio mask; rebuild it at the decoder
            # resolution so that padded audio frames are masked out for batched inputs.
            attention_mask = self._build_audio_context_attention_mask(audio_embeds, attention_mask)

        if labels is not None:
            # Training: append target_text + eos embeddings for teacher forcing. Labels follow the standard `-100`
            # ignore convention for padding, so embed a clamped copy and mask the padded positions out of attention.
            batch_size = input_values.size(0)
            device = input_values.device
            dtype = inputs_embeds.dtype
            text_embed_fn = self.get_input_embeddings()
            eos_batch = torch.full((batch_size, 1), self.config.eos_token_id, dtype=torch.long, device=device)
            safe_labels = labels.masked_fill(labels == -100, self.config.pad_token_id)
            target_embeds = text_embed_fn(safe_labels).to(dtype)
            eos_embeds = text_embed_fn(eos_batch).to(dtype)
            inputs_embeds = torch.cat([inputs_embeds, target_embeds, eos_embeds], dim=1)
            if attention_mask is not None:
                label_mask = (labels != -100).long()
                eos_mask = torch.ones(batch_size, 1, dtype=torch.long, device=device)
                attention_mask = torch.cat([attention_mask, label_mask, eos_mask], dim=1)

        # Build attention mask if not provided
        if attention_mask is None and past_key_values is None:
            batch_size = inputs_embeds.size(0)
            seq_len = inputs_embeds.size(1)
            attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long, device=inputs_embeds.device)

        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            **kwargs,
        )

        logits = outputs.logits
        loss = None
        if labels is not None:
            # Context length = audio_seq_len + 3 (lid_marker + lang_id + bos)
            context_seq_len = inputs_embeds.size(1) - labels.size(1) - 1  # subtract target + eos
            target_len = labels.size(1) + 1  # +1 for EOS
            target_logits = logits[:, context_seq_len - 1 : context_seq_len - 1 + target_len, :]

            batch_size = labels.size(0)
            device = labels.device
            eos_batch = torch.full((batch_size, 1), self.config.eos_token_id, dtype=torch.long, device=device)
            targets = torch.cat([labels, eos_batch], dim=1)

            loss = nn.functional.cross_entropy(
                input=target_logits.reshape(-1, target_logits.size(-1)),
                target=targets.reshape(-1),
                ignore_index=-100,
                reduction="mean",
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @torch.no_grad()
    def _streaming_generate(self, input_values, language_ids=None, **kwargs):
        """Long-audio ("unlimited") decoding.

        Splits each waveform into `config.segment_seconds` windows and decodes them segment-by-segment, keeping the
        last `config.num_context_segments` previous (audio + text) segments as context. Mirrors the original streaming
        syntax `[lang] ( audio_i <segment_marker> <bos> text_i <eos> ) x N`, with `<segment_marker>` being the
        last-segment token for the final window and the regular-segment token otherwise. Runs under `torch.no_grad`
        and retains only the bounded context window, to keep memory constant on arbitrarily long audio.
        """
        config = self.config
        device = input_values.device
        dtype = self.multi_modal_projector.weight.dtype
        text_embed = self.get_input_embeddings()
        eos_id = config.eos_token_id
        seg_size = int(config.audio_sample_rate * config.segment_seconds)
        min_seg = config.audio_sample_rate * config.min_audio_ms // 1000
        kwargs.pop("attention_mask", None)

        outputs = []
        for b in range(input_values.size(0)):
            audio = input_values[b : b + 1]
            total = audio.size(1)
            residue = total % seg_size
            if 0 < residue <= min_seg and total > seg_size:
                total -= residue
            num_segments = max(1, (total + seg_size - 1) // seg_size)

            lang_id = (
                language_ids[b : b + 1].to(device)
                if language_ids is not None
                else torch.zeros(1, dtype=torch.long, device=device)
            )
            lid = torch.full((1, 1), self.language_token_id, dtype=torch.long, device=device)
            lang_prefix = torch.cat(
                [self.lang_embeddings(lang_id.unsqueeze(-1)).to(dtype), text_embed(lid).to(dtype)], dim=1
            )

            blocks = []  # last `num_context_segments` context blocks: [audio_i, marker_i, bos, text_i, eos]
            segment_tokens = []
            for i in range(num_segments):
                segment = audio[:, i * seg_size : (i + 1) * seg_size]
                if segment.size(1) < min_seg:  # too short for the convolutional feature extractor
                    continue
                audio_embeds = self.get_audio_features(segment)
                marker_id = self.last_segment_token_id if i == num_segments - 1 else self.regular_segment_token_id
                marker = text_embed(torch.full((1, 1), marker_id, dtype=torch.long, device=device)).to(dtype)
                bos = text_embed(torch.full((1, 1), config.bos_token_id, dtype=torch.long, device=device)).to(dtype)
                current = torch.cat([audio_embeds, marker, bos], dim=1)
                prefix = torch.cat([lang_prefix, *blocks, current], dim=1)
                generated = super().generate(inputs_embeds=prefix, **kwargs)
                if not torch.is_tensor(generated):  # return_dict_in_generate=True
                    generated = generated.sequences
                generated = generated[:1]  # keep the top hypothesis (beam search / num_return_sequences)
                # the generated text (incl. its eos) extends this segment's context for the next window
                blocks.append(torch.cat([current, text_embed(generated).to(dtype)], dim=1))
                blocks = blocks[-config.num_context_segments :] if config.num_context_segments > 0 else []
                # strip a single trailing eos from the returned transcription tokens
                if generated.size(1) > 0 and generated[0, -1] == eos_id:
                    generated = generated[:, :-1]
                segment_tokens.append(generated)
            seq = (
                torch.cat(segment_tokens, dim=1)[0]
                if segment_tokens
                else torch.zeros(0, dtype=torch.long, device=device)
            )
            outputs.append(seq)

        max_len = max((o.size(0) for o in outputs), default=0)
        out_device = outputs[0].device if outputs else device
        padded = torch.full((len(outputs), max_len), config.pad_token_id, dtype=torch.long, device=out_device)
        for j, output in enumerate(outputs):
            padded[j, : output.size(0)] = output
        return padded

    def generate(self, input_values=None, language_ids=None, **kwargs):
        """Generate token sequences from audio input."""
        if input_values is None:
            input_values = kwargs.pop("input_values", None)
        if language_ids is None:
            language_ids = kwargs.pop("language_ids", None)

        if input_values is not None:
            if getattr(self.config, "is_streaming", False):
                return self._streaming_generate(input_values, language_ids=language_ids, **kwargs)
            batch_size = input_values.size(0)
            device = input_values.device

            audio_attention_mask = kwargs.pop("attention_mask", None)
            audio_embeds = self.get_audio_features(input_values, attention_mask=audio_attention_mask)
            dtype = audio_embeds.dtype
            text_embed_fn = self.get_input_embeddings()

            lid_marker_ids = torch.full((batch_size, 1), self.language_token_id, dtype=torch.long, device=device)
            bos_ids = torch.full((batch_size, 1), self.config.bos_token_id, dtype=torch.long, device=device)

            lid_marker_embeds = text_embed_fn(lid_marker_ids).to(dtype)
            bos_embeds = text_embed_fn(bos_ids).to(dtype)

            if language_ids is not None:
                language_id_batch = language_ids.to(device)
            else:
                language_id_batch = torch.zeros(batch_size, dtype=torch.long, device=device)

            lang_id_embeds = self.lang_embeddings(language_id_batch.unsqueeze(-1)).to(dtype)

            inputs_embeds = torch.cat([audio_embeds, lid_marker_embeds, lang_id_embeds, bos_embeds], dim=1)
            attention_mask = self._build_audio_context_attention_mask(audio_embeds, audio_attention_mask)

            return super().generate(inputs_embeds=inputs_embeds, attention_mask=attention_mask, **kwargs)

        return super().generate(**kwargs)


__all__ = [
    "OmniASRForCTC",
    "OmniASRForConditionalGeneration",
    "OmniASRModel",
    "OmniASRPreTrainedModel",
]

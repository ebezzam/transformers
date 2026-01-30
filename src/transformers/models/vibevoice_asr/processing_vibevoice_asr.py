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

import re

import numpy as np

from ...audio_utils import AudioInput, make_list_of_audio
from ...feature_extraction_utils import BatchFeature
from ...processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import TextInput


class VibeVoiceAsrProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "padding": True,
            "padding_side": "left",
            "add_special_tokens": False,
            "return_attention_mask": True,
            "return_tensors": "pt",
        },
        "audio_kwargs": {
            "sampling_rate": 24000,
            "padding": True,
            "return_attention_mask": True,
            "pad_to_multiple_of": 3200,  # acoustic_tokenizer.hop_length
        },
    }


class VibeVoiceAsrProcessor(ProcessorMixin):
    r"""
    Constructs a VibeVoice ASR processor which wraps [`VibeVoiceAcousticTokenizerFeatureExtractor`] and
    [`Qwen2TokenizerFast`] into a single processor that inherits both the audio feature extraction and
    tokenizer functionalities.

    See the [`~VibeVoiceAsrProcessor.__call__`] for more information.

    Args:
        feature_extractor (`VibeVoiceAcousticTokenizerFeatureExtractor`):
            The feature extractor for audio processing.
        tokenizer (`Qwen2TokenizerFast`):
            The tokenizer for text processing.
        chat_template (`str`, *optional*):
            A Jinja template which will be used to convert lists of messages in a chat into a tokenizable string.
    """

    feature_extractor_class = "VibeVoiceAcousticTokenizerFeatureExtractor"
    tokenizer_class = "Qwen2TokenizerFast"

    def __init__(
        self,
        feature_extractor,
        tokenizer,
        chat_template=None,
    ):
        super().__init__(feature_extractor, tokenizer, chat_template=chat_template)

        # NOTE (ebezzam) original: https://github.com/microsoft/VibeVoice/blob/b2aee8015c3c2d97c388346ebcfffdaf2f427f7d/vibevoice/processor/vibevoice_asr_processor.py#L71
        if not hasattr(tokenizer, "audio_bos_token"):
            self.audio_bos_token = "<|object_ref_start|>"
            self.audio_bos_token_id = tokenizer.convert_tokens_to_ids(self.audio_bos_token)
        else:
            self.audio_bos_token = tokenizer.audio_bos_token
            self.audio_bos_token_id = tokenizer.audio_bos_token_id

        if not hasattr(tokenizer, "audio_eos_token"):
            self.audio_eos_token = "<|object_ref_end|>"
            self.audio_eos_token_id = tokenizer.convert_tokens_to_ids(self.audio_eos_token)
        else:
            self.audio_eos_token = tokenizer.audio_eos_token
            self.audio_eos_token_id = tokenizer.audio_eos_token_id

        if not hasattr(tokenizer, "audio_token"):
            self.audio_token = "<|box_start|>"
            self.audio_token_id = tokenizer.convert_tokens_to_ids(self.audio_token)
        else:
            self.audio_token = tokenizer.audio_token
            self.audio_token_id = tokenizer.audio_token_id

        if not hasattr(tokenizer, "audio_duration_token"):
            self.audio_duration_token = "<|AUDIO_DURATION|>"
        else:
            self.audio_duration_token = tokenizer.audio_duration_token

        # # TODO (ebezzam) not sure about this
        # if not hasattr(tokenizer, "pad_token"):
        #     self.pad_token = "<|endoftext|>"
        #     self.pad_token_id = tokenizer.convert_tokens_to_ids(self.pad_token)
        # else:
        #     self.pad_token = tokenizer.pad_token
        #     self.pad_token_id = tokenizer.pad_token_id

    def __call__(
        self,
        text: TextInput | list[TextInput],
        audio: AudioInput | None = None,
        output_labels: bool | None = False,
        **kwargs: Unpack[VibeVoiceAsrProcessorKwargs],
    ) -> BatchFeature:
        """
        Main method to process text inputs with optional audio samples for ASR.

        This method processes text inputs (typically prepared by apply_chat_template) and optional audio samples
        for transcription. It replaces the audio duration placeholder and expands audio token placeholders based
        on the actual audio length.

        Args:
            text (`str`, `List[str]`):
                The input text(s) to process, typically prepared by apply_chat_template with audio token placeholders.
            audio (`List[Union[str, np.ndarray]]`, *optional*):
                Audio samples for transcription. Should match the number of audio token placeholders in text.
            output_labels (bool, *optional*, default=False):
                Whether to return labels for training.
            **kwargs:
                Additional keyword arguments passed to the tokenizer and feature extractor.

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:
            - **input_ids** -- List of token ids to be fed to the model.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
              `return_attention_mask=True`).
            - **speech_tensors** -- List of audio values to be fed to the model. Returned when `audio` is not `None`.
            - **acoustic_input_mask** -- Mask indicating which positions in input_ids correspond to audio tokens.
            - **labels** -- Labels for training language model. Returned when `output_labels=True`.
        """
        output_kwargs = self._merge_kwargs(
            VibeVoiceAsrProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        text_kwargs = output_kwargs["text_kwargs"]
        audio_kwargs = output_kwargs["audio_kwargs"]
        return_tensors = text_kwargs.get("return_tensors", None)
        if return_tensors != "pt":
            raise ValueError(f"{self.__class__.__name__} only supports `return_tensors='pt'`.")

        if isinstance(text, str):
            text = [text]
        elif not isinstance(text, (list, tuple)):
            raise ValueError("text input must be a string or list of strings")

        if audio is not None:
            audio = make_list_of_audio(audio)
            if len(text) != len(audio):
                raise ValueError(f"Got {len(text)} text but {len(audio)} audios; they must match 1:1.")

            data = self.feature_extractor(audio, **audio_kwargs)

            # Replace audio duration placeholders in text
            audio_lengths = data["padding_mask"].sum(dim=-1).cpu().numpy()
            audio_durations = audio_lengths / self.feature_extractor.sampling_rate
            for i in range(len(text)):
                text[i] = text[i].replace(self.audio_duration_token, f"{audio_durations[i]:.2f}")

            # Expand audio tokens in text
            num_audio_tokens = np.ceil(audio_lengths / audio_kwargs["pad_to_multiple_of"]).astype(int).tolist()
            for i, num_tokens in enumerate(num_audio_tokens):
                text[i] = re.sub(re.escape(self.audio_token), self.audio_token * num_tokens, text[i])

        # Tokenize
        text_inputs = self.tokenizer(text, **text_kwargs)
        data.update(text_inputs)

        if output_labels:
            labels = data["input_ids"].clone()
            labels[labels == self.audio_token_id] = -100
            labels[labels == self.audio_bos_token_id] = -100
            labels[labels == self.audio_eos_token_id] = -100
            labels[labels == self.tokenizer.pad_token_id] = -100
            data["labels"] = labels

        return BatchFeature(data=data, tensor_type=return_tensors)


__all__ = ["VibeVoiceAsrProcessor"]

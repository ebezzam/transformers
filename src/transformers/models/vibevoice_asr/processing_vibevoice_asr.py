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

import json
import re

import numpy as np

from ...audio_utils import AudioInput, make_list_of_audio
from ...feature_extraction_utils import BatchFeature
from ...processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import TextInput
from ...utils import is_torch_available, logging


if is_torch_available():
    import torch


logger = logging.get_logger(__name__)


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
            "pad_to_multiple_of": 3200,  # tokenizer hop length
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
            [`BatchFeature`]: A dictionary with tokenized text (`input_ids`, `attention_mask`) and
            audio features (`input_features`, `input_features_mask`).
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

    def apply_transcription_request(
        self,
        audio: str | list[str] | AudioInput,
        prompt: str | list[str] | None = None,
        **kwargs: Unpack[VibeVoiceAsrProcessorKwargs],
    ) -> BatchFeature:
        """
        Prepare inputs for automatic speech recognition without manually writing the chat template.

        Args:
            audio (`str`, `list[str]`, `np.ndarray`, `torch.Tensor`, `list[np.ndarray]`, `list[torch.Tensor]`):
                Audio to transcribe. Strings are interpreted as local paths or URLs and will be loaded automatically by
                the chat template loader; NumPy arrays and PyTorch tensors are forwarded directly.
            prompt (`str` or `list[str]`, *optional*):
                Custom prompt(s) to include in the user turn as extra context. A list must be the same length as the
                batch. When `None`, no additional context is provided.
            **kwargs:
                Additional keyword arguments forwarded to [`~VibeVoiceAsrProcessor.apply_chat_template`] (for example
                `text_kwargs`, `audio_kwargs`, ...).

        Returns:
            [`BatchFeature`]: Processor outputs ready to be passed to [`VibeVoiceAsrForConditionalGeneration.generate`].
        """

        if isinstance(audio, str):
            audio_items: list[str | np.ndarray] = [audio]
        elif isinstance(audio, (list, tuple)) and audio and all(isinstance(el, str) for el in audio):
            audio_items = list(audio)
        else:
            audio_items = list(make_list_of_audio(audio))
            if is_torch_available():
                audio_items = [el.detach().cpu().numpy() if isinstance(el, torch.Tensor) else el for el in audio_items]

        batch_size = len(audio_items)
        if batch_size == 0:
            raise ValueError("`audio` must contain at least one sample.")

        if prompt is None:
            prompts = [None] * batch_size
        elif isinstance(prompt, str):
            prompts = [prompt] * batch_size
        elif isinstance(prompt, (list, tuple)):
            if len(prompt) != batch_size:
                raise ValueError(
                    f"Received {len(prompt)} prompt(s) for {batch_size} audio sample(s); counts must match."
                )
            prompts = list(prompt)
        else:
            raise TypeError("`prompt` must be a string, a sequence of strings, or `None`.")

        conversations = []
        for prompt_text, audio_item in zip(prompts, audio_items):
            content = []
            if isinstance(audio_item, str):
                content.append({"type": "audio", "path": audio_item})
            else:
                content.append({"type": "audio", "audio": audio_item})

            if prompt_text is not None:
                content.append({"type": "text", "text": prompt_text})

            conversations.append([{"role": "user", "content": content}])

        return self.apply_chat_template(
            conversations,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            **kwargs,
        )

    def batch_decode(
        self, *args, return_as_dicts=False, extract_transcription=False, skip_special_tokens=True, **kwargs
    ):
        """
        Forward arguments to [`~PreTrainedTokenizer.batch_decode`] and optionally parse the dict-like output.

        VibeVoice ASR outputs transcriptions in a dictionary-like format, e.g.:
        ```
        [
            'assistant\n[{"Start":0.0,"End":7.56,"Speaker":0,"Content":"text"}]\n',
            'assistant\n[{"Start":0,"End":5.20,"Speaker":0,"Content":"text"}]\n'
        ]
        ```

        Args:
            return_as_dicts (`bool`, *optional*, defaults to `False`):
                Whether to reformat each decoded output as a list of dicts for each speaker.
            extract_transcription (`bool`, *optional*, defaults to `False`):
                Whether to extract only the transcription content from each decoded output, dropping the speaker tags
                and timestamps.

        Returns:
            `list`: If `return_as_dicts=True`, returns list of parsed dictionary objects.
                If `extract_transcription=True`, returns list of extracted transcription strings.
        """
        decoded = self.tokenizer.batch_decode(*args, skip_special_tokens=skip_special_tokens, **kwargs)

        if return_as_dicts or extract_transcription:
            decoded = [self._parse_dict_output(text) for text in decoded]

        if extract_transcription:
            return [self._extract_content_from_dict(dict_output) for dict_output in decoded]
        else:
            return decoded

    def _parse_dict_output(self, text: str) -> list[dict] | str:
        """Parse JSON output with validation, returning original text on failure."""
        text = text.strip()
        if text.startswith("assistant"):
            text = text[len("assistant") :].strip()

        if not text.startswith("["):
            logger.warning("Output doesn't start with '[', likely not JSON array.")
            return text

        segments = json.loads(text)
        if not isinstance(segments, list):
            logger.warning(f"Expected list, got {type(segments).__name__}.")
            return text

        if segments and not all(isinstance(seg, dict) and "Content" in seg for seg in segments):
            logger.warning("Not all segments have expected structure.")
            return text

        return segments

    def _extract_content_from_dict(self, dict_output: list[dict] | str) -> str:
        """Extract and concatenate 'Content' fields"""
        # If parsing failed, dict_output is the original string
        if isinstance(dict_output, str):
            return dict_output

        contents = [seg.get("Content", "") for seg in dict_output if isinstance(seg, dict)]
        return " ".join(contents).strip()


__all__ = ["VibeVoiceAsrProcessor"]

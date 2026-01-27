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
from ...utils import is_torch_available, logging


if is_torch_available():
    import torch


logger = logging.get_logger(__name__)


class VibeVoiceASRProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "padding": True,
        },
        "audio_kwargs": {
            "sampling_rate": 24000,
            "return_tensors": "pt",
        },
        "common_kwargs": {
            "return_tensors": "pt",
            "padding_side": "left",
        },
    }


class VibeVoiceASRProcessor(ProcessorMixin):
    r"""
    Constructs a VibeVoice ASR processor which wraps a feature extractor (or audio processor) and a tokenizer
    into a single processor.

    [`VibeVoiceASRProcessor`] offers all the functionalities of the audio processor and tokenizer. See the
    [`~VibeVoiceASRProcessor.__call__`] for more information.

    Args:
        audio_processor:
            The audio processor is a required input.
        tokenizer:
            The tokenizer is a required input.
        audio_token (`Optional[str]`, *optional*, defaults to `"<audio>"`):
            Special token used to represent audio inputs in the chat template.
        default_transcription_prompt (`str`, *optional*, defaults to `"Transcribe the following audio."`):
            Default prompt to use for transcription tasks when applying transcription requests.
    """

    def __init__(
        self,
        audio_processor,
        tokenizer,
        audio_token="<audio>",
        default_transcription_prompt="Transcribe the following audio.",
    ):
        self.audio_token = audio_token
        self.audio_token_id = tokenizer.convert_tokens_to_ids(audio_token)
        self.default_transcription_prompt = default_transcription_prompt
        super().__init__(audio_processor, tokenizer)

    def _get_audio_token_length(self, audio_lengths: "torch.Tensor") -> "torch.Tensor":
        """
        Calculate the number of audio tokens for given audio lengths.

        For VibeVoice ASR, the audio tokenizer downsamples by hop_length (default 6400).
        With 24kHz audio, each frame represents 6400/24000 ≈ 0.267 seconds.
        """
        # Assuming hop_length of 6400 for the acoustic tokenizer
        hop_length = 6400
        audio_tokens_lengths = (audio_lengths + hop_length - 1) // hop_length
        return audio_tokens_lengths

    def __call__(
        self,
        text: TextInput | list[TextInput],
        audio: AudioInput | None = None,
        output_labels: bool | None = False,
        **kwargs: Unpack[VibeVoiceASRProcessorKwargs],
    ) -> BatchFeature:
        r"""
        Main method to prepare one or several text sequence(s) and audio waveform(s) for the model. This
        method expands `<audio>` placeholders in the text based on the audio token counts, then tokenizes
        the provided strings and processes the audio.

        Args:
            text (`str` or `list[str]`):
                Input sequence or batch of sequences.
            audio (`np.ndarray` or `list[np.ndarray]` or `torch.Tensor` or `list[torch.Tensor]`, *optional*):
                Input audio or batch of audios as NumPy arrays or PyTorch tensors. If provided, there must be as many
                `text` inputs as `audio` inputs. Audio should be sampled at 24kHz.
            output_labels (bool, *optional*, default=False):
                Whether to return labels for training.

        Returns:
            [`BatchFeature`]: A dictionary with tokenized text (`input_ids`, `attention_mask`) and
            audio tensors (`speech_tensors`).
        """

        # Merge defaults with user kwargs
        call_kwargs = self._merge_kwargs(
            VibeVoiceASRProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        text_kwargs = call_kwargs["text_kwargs"]
        audio_kwargs = call_kwargs["audio_kwargs"]
        return_tensors = text_kwargs.get("return_tensors")
        if return_tensors != "pt":
            raise ValueError(f"{self.__class__.__name__} only supports `return_tensors='pt'`.")

        if isinstance(text, str):
            text = [text]
        elif not (isinstance(text, (list, tuple)) and all(isinstance(t, str) for t in text)):
            raise ValueError("Invalid input text. Please provide a string, or a list of strings")

        audio_inputs = {}
        if audio is not None:
            audio = make_list_of_audio(audio)
            if len(text) != len(audio):
                raise ValueError(f"Got {len(text)} text but {len(audio)} audios; they must match 1:1.")

            # Process audio
            audio_list = []
            audio_token_lengths = []

            for audio_el in audio:
                # Convert to numpy if needed
                if is_torch_available() and isinstance(audio_el, torch.Tensor):
                    audio_el = audio_el.detach().cpu().numpy()

                audio_list.append(audio_el)

                # Calculate number of tokens for this audio
                n_samples = len(audio_el)
                sampling_rate = audio_kwargs.get("sampling_rate", 24000)

                # Resample if needed (basic check)
                # Note: In practice, you'd want proper resampling here

                # Calculate audio token length based on hop_length
                audio_token_length = self._get_audio_token_length(torch.tensor([n_samples]))[0].item()
                audio_token_lengths.append(audio_token_length)

            # Expand audio tokens in text
            for i, audio_length in enumerate(audio_token_lengths):
                # Replace first occurrence of audio_token with repeated tokens
                text[i] = re.sub(re.escape(self.audio_token), self.audio_token * audio_length, text[i], count=1)

            # Stack audio tensors
            # For simplicity, we'll pass them as a list in the batch feature
            # The model will handle batching internally
            if is_torch_available():
                # Convert to torch tensors
                audio_tensors = []
                for aud in audio_list:
                    if isinstance(aud, np.ndarray):
                        audio_tensors.append(torch.from_numpy(aud).float())
                    else:
                        audio_tensors.append(aud)
                audio_inputs["speech_tensors"] = audio_tensors

        # Tokenize text
        text_inputs = self.tokenizer(text, **text_kwargs)

        # Combine text and audio inputs
        data = {**text_inputs, **audio_inputs}

        if output_labels:
            labels = data["input_ids"].clone()
            labels[labels == self.audio_token_id] = -100
            labels[labels == self.tokenizer.pad_token_id] = -100
            data["labels"] = labels

        # Create acoustic_input_mask indicating where audio tokens are
        if audio is not None and "input_ids" in data:
            acoustic_input_mask = data["input_ids"] == self.audio_token_id
            data["acoustic_input_mask"] = acoustic_input_mask

        return BatchFeature(data=data, tensor_type=return_tensors)

    @property
    def model_input_names(self) -> list[str]:
        tok_names = self.tokenizer.model_input_names
        audio_names = ["speech_tensors", "speech_masks", "acoustic_input_mask"]
        return list(dict.fromkeys(tok_names + audio_names))

    def apply_transcription_request(
        self,
        audio: str | list[str] | AudioInput,
        prompt: str | list[str] | None = None,
        **kwargs: Unpack[VibeVoiceASRProcessorKwargs],
    ) -> BatchFeature:
        """
        Prepare inputs for automatic speech recognition without manually writing the default transcription prompt.

        Args:
            audio (`str`, `list[str]`, `np.ndarray`, `torch.Tensor`, `list[np.ndarray]`, `list[torch.Tensor]`):
                Audio to transcribe. Strings are interpreted as local paths or URLs and will be loaded automatically by
                the chat template loader; NumPy arrays and PyTorch tensors are forwarded directly.
            prompt (`str` or `list[str]`, *optional*):
                Custom prompt(s) to include in the user turn. A list must be the same length as the batch. When `None`,
                each sample uses the default transcription prompt.
            **kwargs:
                Additional keyword arguments forwarded to [`~VibeVoiceASRProcessor.apply_chat_template`] (for example
                `text_kwargs`, `audio_kwargs`, ...).

        Returns:
            [`BatchFeature`]: Processor outputs ready to be passed to [`VibeVoiceASRForConditionalGeneration.generate`].
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
            prompts = [self.default_transcription_prompt] * batch_size
        elif isinstance(prompt, str):
            prompts = [prompt] * batch_size
        elif isinstance(prompt, (list, tuple)):
            if len(prompt) != batch_size:
                raise ValueError(
                    f"Received {len(prompt)} prompt(s) for {batch_size} audio sample(s); counts must match."
                )
            prompts = []
            for item in prompt:
                if item is None:
                    prompts.append(self.default_transcription_prompt)
                elif isinstance(item, str):
                    prompts.append(item)
                else:
                    raise TypeError("Each prompt must be a string or `None`.")
        else:
            raise TypeError("`prompt` must be a string, a sequence of strings, or `None`.")

        conversations = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {"type": "audio", "path": audio_item}
                        if isinstance(audio_item, str)
                        else {"type": "audio", "audio": audio_item},
                    ],
                }
            ]
            for prompt_text, audio_item in zip(prompts, audio_items)
        ]

        return self.apply_chat_template(
            conversations,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            **kwargs,
        )

    def batch_decode(self, *args, **kwargs):
        """
        Forward arguments to [`~PreTrainedTokenizer.batch_decode`].
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        Forward arguments to [`~PreTrainedTokenizer.decode`].
        """
        return self.tokenizer.decode(*args, **kwargs)


__all__ = ["VibeVoiceASRProcessor"]

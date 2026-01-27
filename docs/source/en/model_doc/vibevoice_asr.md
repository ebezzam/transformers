<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# VibeVoice ASR

<div class="flex flex-wrap space-x-1">
<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
<img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
<img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
</div>

## Overview

VibeVoice ASR is an automatic speech recognition model from Microsoft that combines acoustic and semantic audio tokenizers with a causal language model for robust speech-to-text transcription. The model uses VibeVoice's proprietary acoustic tokenizers that process audio at 24kHz, paired with a Qwen2-based language decoder for generating transcriptions.

The model checkpoint is available at: [microsoft/VibeVoice-ASR](https://huggingface.co/microsoft/VibeVoice-ASR)

Highlights:

- **Dual tokenizer architecture**: Combines acoustic and semantic features for improved transcription quality.
- **24kHz audio processing**: Handles higher quality audio inputs compared to typical 16kHz models.
- **Streaming support for long audio**: Processes very long audio files (>10 minutes) using automatic segmentation.
- **Replace-in-place audio fusion**: Audio tokens are replaced with audio embeddings without changing sequence length.

This model was contributed by [Eric Bezzam](https://huggingface.co/bezzam).

### Paper

[VibeVoice](https://arxiv.org/abs/2501.09891): A Unified Generative Streaming Audio Codec and LLM
Microsoft Research
Project: https://github.com/microsoft/VibeVoice

## Usage

### Basic Transcription

The model supports automatic speech recognition with simple text + audio instructions.

➡️ audio + text instruction

```python
from transformers import VibeVoiceASRForConditionalGeneration, AutoProcessor
import torch

model_id = "microsoft/VibeVoice-ASR"
processor = AutoProcessor.from_pretrained(model_id)
model = VibeVoiceASRForConditionalGeneration.from_pretrained(model_id, device_map="auto")

# Load audio (24kHz recommended)
# audio = load_your_audio_here()  # torch.Tensor of shape (num_samples,) at 24kHz

conversation = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Transcribe the following audio."},
            {"type": "audio", "audio": audio},
        ],
    }
]

inputs = processor.apply_chat_template(
    conversation,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=500)

decoded_outputs = processor.batch_decode(outputs[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print(decoded_outputs)
```

### Transcription Shortcut

For quick transcription without manually creating the conversation format:

```python
from transformers import VibeVoiceASRForConditionalGeneration, AutoProcessor
import torch

model_id = "microsoft/VibeVoice-ASR"
processor = AutoProcessor.from_pretrained(model_id)
model = VibeVoiceASRForConditionalGeneration.from_pretrained(model_id, device_map="auto")

# Load audio
# audio = load_your_audio_here()

inputs = processor.apply_transcription_request(audio=audio).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=500)
decoded_outputs = processor.batch_decode(outputs[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)

print(decoded_outputs)
```

### Batched Inference

Process multiple audio files simultaneously:

```python
from transformers import VibeVoiceASRForConditionalGeneration, AutoProcessor

model_id = "microsoft/VibeVoice-ASR"
processor = AutoProcessor.from_pretrained(model_id)
model = VibeVoiceASRForConditionalGeneration.from_pretrained(model_id, device_map="auto")

# Load multiple audios
# audio1 = load_audio_1()
# audio2 = load_audio_2()

conversations = [
    [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Transcribe the following audio."},
                {"type": "audio", "audio": audio1},
            ],
        }
    ],
    [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is said in this recording?"},
                {"type": "audio", "audio": audio2},
            ],
        }
    ],
]

inputs = processor.apply_chat_template(
    conversations,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=500)

decoded_outputs = processor.batch_decode(outputs[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print(decoded_outputs)
```

### Multi-turn Conversations

VibeVoice ASR supports multi-turn dialogues about audio content:

```python
from transformers import VibeVoiceASRForConditionalGeneration, AutoProcessor

model_id = "microsoft/VibeVoice-ASR"
processor = AutoProcessor.from_pretrained(model_id)
model = VibeVoiceASRForConditionalGeneration.from_pretrained(model_id, device_map="auto")

conversation = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "What is being discussed in this audio?"},
            {"type": "audio", "audio": audio},
        ],
    },
    {
        "role": "assistant",
        "content": [{"type": "text", "text": "The audio discusses machine learning techniques."}],
    },
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Can you provide more details about the specific techniques mentioned?"},
        ],
    },
]

inputs = processor.apply_chat_template(
    conversation,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=500)

decoded_outputs = processor.batch_decode(outputs[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print(decoded_outputs)
```

### Training

Fine-tune the model on your own ASR dataset:

```python
from transformers import VibeVoiceASRForConditionalGeneration, AutoProcessor

model_id = "microsoft/VibeVoice-ASR"
processor = AutoProcessor.from_pretrained(model_id)
model = VibeVoiceASRForConditionalGeneration.from_pretrained(model_id, device_map="auto")
model.train()

conversations = [
    [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Transcribe the following audio."},
                {"type": "audio", "audio": audio1},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "This is the correct transcription of audio one."}],
        }
    ],
    [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is said in this recording?"},
                {"type": "audio", "audio": audio2},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "This is the transcription of audio two."}],
        }
    ]
]

inputs = processor.apply_chat_template(
    conversations,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    output_labels=True,
).to(model.device)

loss = model(**inputs).loss
loss.backward()
```

## How the Model Works

### Architecture

VibeVoice ASR consists of three main components:

* **Acoustic Tokenizer**
  A VibeVoice acoustic tokenizer that processes raw audio waveforms at 24kHz. The tokenizer uses a ConvNeXt-style encoder with downsampling to produce acoustic latent representations.

* **Semantic Tokenizer**
  A second VibeVoice tokenizer that extracts semantic information from the same audio input. This provides complementary features focused on linguistic content rather than acoustic details.

* **Speech Connectors**
  Two-layer MLPs with RMSNorm that project both acoustic and semantic features to the language model's hidden size. The features are combined additively before being used.

* **Language Model**
  A Qwen2-based causal language model that processes text embeddings with audio frame embeddings replacing audio placeholder tokens in place.

### Audio Processing Pipeline

1. **Dual Encoding**: Raw audio is encoded by both the acoustic and semantic tokenizers independently, producing two sets of latent representations.

2. **Feature Projection**: Each set of latents is projected to the language model dimension using its respective speech connector.

3. **Feature Fusion**: Acoustic and semantic features are added together element-wise.

4. **Token Replacement**: The fused audio features replace audio placeholder tokens (`<audio>`) in the input embedding sequence.

5. **Generation**: The language model generates text tokens autoregressively, with audio information influencing predictions through the replaced embeddings.

### Streaming for Long Audio

For very long audio files (>60 seconds by default), VibeVoice ASR automatically uses streaming processing:

* Audio is split into 60-second segments (configurable via `streaming_segment_duration`).
* Each segment is encoded separately using streaming cache mechanisms.
* Encoded segments are concatenated and sampled together to maintain consistency.
* This prevents memory issues and computational overflow for very long recordings.

## Usage Patterns

### Custom Prompts

While transcription is the primary use case, you can customize prompts:

```python
inputs = processor.apply_transcription_request(
    audio=audio,
    prompt="Provide a detailed transcription with punctuation."
)
```

### Audio Token Expansion

The processor automatically:
1. Calculates the number of audio tokens based on the audio length and tokenizer hop length (6400 samples at 24kHz ≈ 0.267 seconds per frame).
2. Expands `<audio>` placeholder tokens in the text to match the number of audio frames.
3. Creates an `acoustic_input_mask` to indicate which tokens should be replaced with audio embeddings.

### Attention and Masking

* **Attention masks**: The processor returns standard attention masks for text tokens.
* **Acoustic input mask**: A boolean mask indicating positions where audio embeddings replace text embeddings.
* **Caching**: During generation, audio inputs are only processed on the first forward pass. Subsequent steps use cached key-value pairs.

## Troubleshooting

### Audio Format Issues

* Ensure audio is sampled at 24kHz. If your audio is at a different sample rate, resample it:
  ```python
  import torchaudio
  waveform, sample_rate = torchaudio.load("audio.wav")
  if sample_rate != 24000:
      resampler = torchaudio.transforms.Resample(sample_rate, 24000)
      waveform = resampler(waveform)
  ```

* Audio should be a 1D tensor (mono). If stereo, convert to mono:
  ```python
  if waveform.shape[0] > 1:
      waveform = waveform.mean(dim=0, keepdim=True)
  ```

### Empty or Truncated Outputs

* Use left padding for batched generation.
* Decode only the new tokens after the prompt length.
* Ensure `max_new_tokens` is sufficient for your expected transcription length.

### Memory Issues with Long Audio

* The model automatically switches to streaming mode for audio longer than the `streaming_segment_duration` (default 60s).
* You can adjust this parameter in the `encode_speech` method if needed.
* For extremely long audio, consider splitting it manually and processing in batches.

## VibeVoiceASRConfig

[[autodoc]] VibeVoiceASRConfig

## VibeVoiceASRProcessor

[[autodoc]] VibeVoiceASRProcessor
    - __call__
    - apply_transcription_request

## VibeVoiceASRModel

[[autodoc]] VibeVoiceASRModel
    - forward

## VibeVoiceASRForConditionalGeneration

[[autodoc]] VibeVoiceASRForConditionalGeneration
    - forward
    - encode_speech

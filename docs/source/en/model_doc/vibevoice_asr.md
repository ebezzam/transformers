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
*This model was released on 2026-01-26 and added to Hugging Face Transformers on 2026-02-04.*

# VibeVoice ASR

<div class="flex flex-wrap space-x-1">
<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
<img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
<img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
</div>

## Overview

VibeVoice ASR is an automatic speech recognition model from Microsoft that combines acoustic and semantic audio tokenizers with a causal language model for robust speech-to-text transcription. The model uses VibeVoice's proprietary acoustic tokenizers that process audio at 24kHz, paired with a Qwen2-based language decoder for generating transcriptions. See the [technical report](https://huggingface.co/papers/2601.18184) for more details.

The model checkpoint is available at: [microsoft/VibeVoice-ASR](https://huggingface.co/microsoft/VibeVoice-ASR)

Highlights:

- **🕒 60-minute Single-Pass Processing**:
  Unlike conventional ASR models that slice audio into short chunks (often losing global context), VibeVoice ASR accepts up to **60 minutes** of continuous audio input within 64K token length. This ensures consistent speaker tracking and semantic coherence across the entire hour.

- **👤 Customized Hotwords**:
  Users can provide customized hotwords (e.g., specific names, technical terms, or background info) to guide the recognition process, significantly improving accuracy on domain-specific content.

- **📝 Rich Transcription (Who, When, What)**:
  The model jointly performs ASR, diarization, and timestamping, producing a structured output that indicates *who* said *what* and *when*.
  
- **🌍 Multilingual & Code-Switching Support**:
  It supports over 50 languages, requires no explicit language setting, and natively handles code-switching within and across utterances. Language distribution can be found [here](#language-distribution).


This model was contributed by [Eric Bezzam](https://huggingface.co/bezzam).


## Usage

The model supports various automatic speech recognition functionalities.


### Speaker-timestamped transcription

A notable feature of VibeVoice ASR is its ability to transcribe multi-speaker content, denoting who spoke and when.

```python
from transformers import AutoProcessor, VibeVoiceAsrForConditionalGeneration


model_id = "bezzam/VibeVoice-ASR-7B"

# Load processor and model
processor = AutoProcessor.from_pretrained(model_id)
model = VibeVoiceAsrForConditionalGeneration.from_pretrained(model_id, device_map="auto")
print(f"Model loaded on {model.device} with dtype {model.dtype}")

# Prepare inputs using `apply_transcription_request`
inputs = processor.apply_transcription_request(
    audio="https://huggingface.co/datasets/bezzam/vibevoice_samples/resolve/main/example_output/VibeVoice-1.5B_output.wav",
).to(model.device, model.dtype)

# Apply model
output_ids = model.generate(**inputs)

# Print results
generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
transcription = processor.batch_decode(generated_ids)[0]
print("\n" + "=" * 60)
print("RAW OUTPUT")
print("=" * 60)
print(transcription)

transcription = processor.batch_decode(generated_ids, return_as_dicts=True)[0]
print("\n" + "=" * 60)
print("TRANSCRIPTION (list of dicts)")
print("=" * 60)
for speaker_transcription in transcription:
    print(speaker_transcription)

# Remove speaker labels, only get raw transcription
transcription = processor.batch_decode(generated_ids, extract_transcription=True)[0]
print("\n" + "=" * 60)
print("TRANSCRIPTION ONLY")
print("=" * 60)
print(transcription)

"""
============================================================
RAW OUTPUT
============================================================
assistant
[{"Start":0,"End":15.43,"Speaker":0,"Content":"Hello everyone and welcome to the Vibe Voice podcast. I'm your host, Alex, and today we're getting into one of the biggest debates in all of sports: who's the greatest basketball player of all time? I'm so excited to have Sam here to talk about it with me."},{"Start":15.43,"End":21.05,"Speaker":1,"Content":"Thanks so much for having me, Alex. And you're absolutely right. This question always brings out some seriously strong feelings."},{"Start":21.05,"End":31.66,"Speaker":0,"Content":"Okay, so let's get right into it. For me, it has to be Michael Jordan. Six trips to the finals, six championships. That kind of perfection is just incredible."},{"Start":31.66,"End":40.93,"Speaker":1,"Content":"Oh man, the first thing that always pops into my head is that shot against the Cleveland Cavaliers back in '89. Jordan just rises, hangs in the air forever, and just sinks it."}]

============================================================
TRANSCRIPTION (list of dicts)
============================================================
{'Start': 0, 'End': 15.43, 'Speaker': 0, 'Content': "Hello everyone and welcome to the Vibe Voice podcast. I'm your host, Alex, and today we're getting into one of the biggest debates in all of sports: who's the greatest basketball player of all time? I'm so excited to have Sam here to talk about it with me."}
{'Start': 15.43, 'End': 21.05, 'Speaker': 1, 'Content': "Thanks so much for having me, Alex. And you're absolutely right. This question always brings out some seriously strong feelings."}
{'Start': 21.05, 'End': 31.66, 'Speaker': 0, 'Content': "Okay, so let's get right into it. For me, it has to be Michael Jordan. Six trips to the finals, six championships. That kind of perfection is just incredible."}
{'Start': 31.66, 'End': 40.93, 'Speaker': 1, 'Content': "Oh man, the first thing that always pops into my head is that shot against the Cleveland Cavaliers back in '89. Jordan just rises, hangs in the air forever, and just sinks it."}

============================================================
TRANSCRIPTION ONLY
============================================================
Hello everyone and welcome to the Vibe Voice podcast. I'm your host, Alex, and today we're getting into one of the biggest debates in all of sports: who's the greatest basketball player of all time? I'm so excited to have Sam here to talk about it with me. Thanks so much for having me, Alex. And you're absolutely right. This question always brings out some seriously strong feelings. Okay, so let's get right into it. For me, it has to be Michael Jordan. Six trips to the finals, six championships. That kind of perfection is just incredible. Oh man, the first thing that always pops into my head is that shot against the Cleveland Cavaliers back in '89. Jordan just rises, hangs in the air forever, and just sinks it.
"""
```

The VibeVoice ASR model is trained to generate a string that resembles a JSON structure. The flag `return_as_dicts=True` tries to return the generated output as a list of dicts, while `extract_transcription=True` tries to extract only the transcribed audio. If they fail, the generated output is returned as-is.

### Providing context

It is also possible to provide context. This can be useful if certain words cannot be transcribed correctly, such as proper nouns.

Below we transcribe an audio where the speaker (with a German accent) talks about VibeVoice, comparing with and without the context "About VibeVoice".

```python
from transformers import AutoProcessor, VibeVoiceAsrForConditionalGeneration


model_id = "bezzam/VibeVoice-ASR-7B"


# Load processor and model
processor = AutoProcessor.from_pretrained(model_id)
model = VibeVoiceAsrForConditionalGeneration.from_pretrained(model_id, device_map="auto")
print(f"Model loaded on {model.device} with dtype {model.dtype}")

# Without context
inputs = processor.apply_transcription_request(
    audio="https://huggingface.co/datasets/bezzam/vibevoice_samples/resolve/main/realtime_model/vibevoice_tts_german.wav",
).to(model.device, model.dtype)
output_ids = model.generate(**inputs)
generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
transcription = processor.batch_decode(generated_ids, extract_transcription=True)[0]
print(f"WITHOUT CONTEXT: {transcription}")

# Without context
inputs = processor.apply_transcription_request(
    audio="https://huggingface.co/datasets/bezzam/vibevoice_samples/resolve/main/realtime_model/vibevoice_tts_german.wav",
    prompt="About VibeVoice",
).to(model.device, model.dtype)
output_ids = model.generate(**inputs)
generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
transcription = processor.batch_decode(generated_ids, extract_transcription=True)[0]
print(f"WITH CONTEXT   : {transcription}")

"""
WITHOUT CONTEXT: Revevoices is a novel framework designed for generating expressive, long-form, multi-speaker conversational audio.
WITH CONTEXT   : VibeVoice is this novel framework designed for generating expressive, long-form, multi-speaker, conversational audio.
"""
```


### Batch inference

Batch inference is possible by passing a list of audio and (if provided) a list of prompts of equal length.

```python
from transformers import AutoProcessor, VibeVoiceAsrForConditionalGeneration


model_id = "bezzam/VibeVoice-ASR-7B"
audio = [
    "https://huggingface.co/datasets/bezzam/vibevoice_samples/resolve/main/realtime_model/vibevoice_tts_german.wav",
    "https://huggingface.co/datasets/bezzam/vibevoice_samples/resolve/main/example_output/VibeVoice-1.5B_output.wav"
]
prompts = ["About VibeVoice", None]

# Load processor and model
processor = AutoProcessor.from_pretrained(model_id)
model = VibeVoiceAsrForConditionalGeneration.from_pretrained(model_id, device_map="auto")
print(f"Model loaded on {model.device} with dtype {model.dtype}")

# Apply model with batch inputs
inputs = processor.apply_transcription_request(audio, prompt=prompts).to(model.device, model.dtype)
output_ids = model.generate(**inputs)
generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
transcription = processor.batch_decode(generated_ids, extract_transcription=True)

print(transcription)
```


### Adjusting tokenizer chunk (e.g. if out-of-memory)

A key feature of VibeVoice ASR is that it can transcribe up to 60 minutes of continuous audio. This is done by chunking audio into 60-second segments (1440000 samples at 24kHz) and caching the convolution states between each segment.

However, if chunks of 60 seconds are too large for your device, the `tokenizer_chunk_size` argument passed to `generate` can be adjusted. *Note it should be a multiple of the hop length (3200 for the original acoustic tokenizer).*

```python
from transformers import AutoProcessor, VibeVoiceAsrForConditionalGeneration


model_id = "bezzam/VibeVoice-ASR-7B"
tokenizer_chunk_size = 64000    # default is 1440000 (60s @ 24kHz)
audio = [
    "https://huggingface.co/datasets/bezzam/vibevoice_samples/resolve/main/realtime_model/vibevoice_tts_german.wav",
    "https://huggingface.co/datasets/bezzam/vibevoice_samples/resolve/main/example_output/VibeVoice-1.5B_output.wav"
]
prompts = ["About VibeVoice", None]


# Load processor and model
processor = AutoProcessor.from_pretrained(model_id)
model = VibeVoiceAsrForConditionalGeneration.from_pretrained(model_id, device_map="auto")
print(f"Model loaded on {model.device} with dtype {model.dtype}")

# Apply model with batch inputs
inputs = processor.apply_transcription_request(audio, prompt=prompts).to(model.device, model.dtype)
output_ids = model.generate(**inputs, tokenizer_chunk_size=tokenizer_chunk_size)
generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
transcription = processor.batch_decode(generated_ids, extract_transcription=True)
```



### Chat template

VibeVoice ASR also accepts chat template inputs (`apply_transcription_request` is actually a wrapper for convenience):
```python
from transformers import AutoProcessor, VibeVoiceAsrForConditionalGeneration


model_id = "bezzam/VibeVoice-ASR-7B"

# Load processor and model
processor = AutoProcessor.from_pretrained(model_id)
model = VibeVoiceAsrForConditionalGeneration.from_pretrained(model_id, device_map="auto")

# Prepare chat template
chat_template = [
    [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "About VibeVoice"},
                {
                    "type": "audio",
                    "path": "https://huggingface.co/datasets/bezzam/vibevoice_samples/resolve/main/realtime_model/vibevoice_tts_german.wav",
                },
            ],
        }
    ],
    [
        {
            "role": "user",
            "content": [
                {
                    "type": "audio",
                    "path": "https://huggingface.co/datasets/bezzam/vibevoice_samples/resolve/main/example_output/VibeVoice-1.5B_output.wav",
                },
            ],
        }
    ],
]

# Prepare inputs
inputs = processor.apply_chat_template(
    chat_template,
    tokenize=True,
    return_dict=True,
).to(model.device, model.dtype)

# Apply model
output_ids = model.generate(**inputs)
generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
transcription = processor.batch_decode(generated_ids, extract_transcription=True)

print(transcription)
```

### Training

VibeVoice ASR can be trained, the model outputs a loss


```python
from transformers import AutoProcessor, VibeVoiceAsrForConditionalGeneration


model_id = "bezzam/VibeVoice-ASR-7B"

# Load processor and model
processor = AutoProcessor.from_pretrained(model_id)
model = VibeVoiceAsrForConditionalGeneration.from_pretrained(model_id, device_map="auto")
model.train()

# Prepare inputs (batch of 2)
# -- NOTE: original model outputs content, speaker ID, and timestamps in JSON-like format. Below we are only using the transcription text.
chat_template = [
    [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "VibeVoice is this novel framework designed for generating expressive, long-form, multi-speaker, conversational audio."},
                {
                    "type": "audio",
                    "path": "https://huggingface.co/datasets/bezzam/vibevoice_samples/resolve/main/realtime_model/vibevoice_tts_german.wav",
                },
            ],
        }
    ],
    [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Hello everyone and welcome to the VibeVoice podcast. I'm your host, Alex, and today we're getting into one of the biggest debates in all of sports: who's the greatest basketball player of all time? I'm so excited to have Sam here to talk about it with me. Thanks so much for having me, Alex. And you're absolutely right. This question always brings out some seriously strong feelings. Okay, so let's get right into it. For me, it has to be Michael Jordan. Six trips to the finals, six championships. That kind of perfection is just incredible. Oh man, the first thing that always pops into my head is that shot against the Cleveland Cavaliers back in '89. Jordan just rises, hangs in the air forever, and just sinks it."},
                {
                    "type": "audio",
                    "path": "https://huggingface.co/datasets/bezzam/vibevoice_samples/resolve/main/example_output/VibeVoice-1.5B_output.wav",
                },
            ],
        }
    ],
]
inputs = processor.apply_chat_template(
    chat_template,
    tokenize=True,
    return_dict=True,
    output_labels=True,
).to(model.device, model.dtype)

# Apply model and backpropagate loss
loss = model(**inputs).loss
print("Loss:", loss.item())
loss.backward()
```


## VibeVoiceAsrEncoderConfig

[[autodoc]] VibeVoiceAsrEncoderConfig

## VibeVoiceAsrConfig

[[autodoc]] VibeVoiceAsrConfig

## VibeVoiceAsrProcessor

[[autodoc]] VibeVoiceAsrProcessor
    - __call__
    - apply_transcription_request
    - batch_decode

## VibeVoiceAsrEncoderModel

[[autodoc]] VibeVoiceAsrEncoderModel
    - forward

## VibeVoiceAsrForConditionalGeneration

[[autodoc]] VibeVoiceAsrForConditionalGeneration
    - forward
    - get_audio_features

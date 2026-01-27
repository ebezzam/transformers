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
"""Testing suite for the VibeVoice ASR processor."""

import unittest

import numpy as np

from transformers import AutoTokenizer, VibeVoiceASRProcessor
from transformers.testing_utils import require_torch


@require_torch
class VibeVoiceASRProcessorTest(unittest.TestCase):
    def setUp(self):
        # Use a simple tokenizer for testing
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")
        # Add audio token if not present
        if "<audio>" not in self.tokenizer.get_vocab():
            self.tokenizer.add_special_tokens({"additional_special_tokens": ["<audio>"]})

        self.processor = VibeVoiceASRProcessor(
            audio_processor=None,
            tokenizer=self.tokenizer,
            audio_token="<audio>",
        )

    def test_processor_tokenizer_only(self):
        """Test that processor works with text-only inputs."""
        text = "Hello, how are you?"
        encoded = self.processor(text=text, audio=None)

        self.assertIn("input_ids", encoded)
        self.assertIn("attention_mask", encoded)
        self.assertNotIn("speech_tensors", encoded)

    def test_processor_with_audio(self):
        """Test that processor works with text and audio inputs."""
        text = "<audio>Transcribe this audio."
        # Create dummy audio (1 second at 24kHz)
        audio = np.random.randn(24000).astype(np.float32)

        encoded = self.processor(text=text, audio=audio)

        self.assertIn("input_ids", encoded)
        self.assertIn("attention_mask", encoded)
        self.assertIn("speech_tensors", encoded)
        self.assertIn("acoustic_input_mask", encoded)

        # Check that audio token was expanded
        audio_token_id = self.processor.audio_token_id
        num_audio_tokens = (encoded["input_ids"] == audio_token_id).sum().item()
        self.assertGreater(num_audio_tokens, 0)

    def test_processor_batch(self):
        """Test that processor works with batched inputs."""
        texts = ["<audio>First audio.", "<audio>Second audio."]
        # Create dummy audios
        audios = [
            np.random.randn(24000).astype(np.float32),
            np.random.randn(48000).astype(np.float32),  # Different length
        ]

        encoded = self.processor(text=texts, audio=audios)

        self.assertIn("input_ids", encoded)
        self.assertIn("attention_mask", encoded)
        self.assertIn("speech_tensors", encoded)
        self.assertEqual(len(encoded["speech_tensors"]), 2)

    def test_apply_transcription_request(self):
        """Test the apply_transcription_request convenience method."""
        audio = np.random.randn(24000).astype(np.float32)

        encoded = self.processor.apply_transcription_request(audio=audio)

        self.assertIn("input_ids", encoded)
        self.assertIn("speech_tensors", encoded)

        # Check that default prompt was added
        decoded_text = self.processor.tokenizer.decode(encoded["input_ids"][0], skip_special_tokens=False)
        self.assertIn(self.processor.default_transcription_prompt, decoded_text)

    def test_audio_token_expansion(self):
        """Test that audio tokens are correctly expanded based on audio length."""
        text = "<audio>Test"
        audio_short = np.random.randn(24000).astype(np.float32)  # 1 second
        audio_long = np.random.randn(240000).astype(np.float32)  # 10 seconds

        encoded_short = self.processor(text=text, audio=audio_short)
        encoded_long = self.processor(text=text, audio=audio_long)

        # Count audio tokens
        audio_token_id = self.processor.audio_token_id
        num_tokens_short = (encoded_short["input_ids"] == audio_token_id).sum().item()
        num_tokens_long = (encoded_long["input_ids"] == audio_token_id).sum().item()

        # Longer audio should have more tokens
        self.assertGreater(num_tokens_long, num_tokens_short)


if __name__ == "__main__":
    unittest.main()

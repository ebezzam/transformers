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

import shutil
import tempfile
import unittest

from parameterized import parameterized

from transformers import (
    AutoProcessor,
    AutoTokenizer,
    VibeVoiceAcousticTokenizerFeatureExtractor,
    VibeVoiceAsrProcessor,
)
from transformers.testing_utils import require_torch

from ...test_processing_common import ProcessorTesterMixin


class VibeVoiceAsrProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = VibeVoiceAsrProcessor

    @classmethod
    @require_torch
    def setUpClass(cls):
        cls.checkpoint = "bezzam/VibeVoice-ASR-7B"
        cls.tmpdirname = tempfile.mkdtemp()

        processor = VibeVoiceAsrProcessor.from_pretrained(cls.checkpoint)
        processor.save_pretrained(cls.tmpdirname)

    @require_torch
    def get_tokenizer(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdirname, **kwargs).tokenizer

    @require_torch
    def get_feature_extractor(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdirname, **kwargs).feature_extractor

    @require_torch
    def get_processor(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdirname, **kwargs)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdirname, ignore_errors=True)

    @require_torch
    def test_can_load_various_tokenizers(self):
        processor = VibeVoiceAsrProcessor.from_pretrained(self.checkpoint)
        tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)
        self.assertEqual(processor.tokenizer.__class__, tokenizer.__class__)

    @require_torch
    def test_save_load_pretrained_default(self):
        tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)
        processor = VibeVoiceAsrProcessor.from_pretrained(self.checkpoint)
        feature_extractor = processor.feature_extractor

        processor = VibeVoiceAsrProcessor(tokenizer=tokenizer, feature_extractor=feature_extractor)

        with tempfile.TemporaryDirectory() as tmpdir:
            processor.save_pretrained(tmpdir)
            reloaded = VibeVoiceAsrProcessor.from_pretrained(tmpdir)

        self.assertEqual(reloaded.tokenizer.get_vocab(), tokenizer.get_vocab())
        self.assertEqual(reloaded.feature_extractor.to_json_string(), feature_extractor.to_json_string())
        self.assertIsInstance(reloaded.feature_extractor, VibeVoiceAcousticTokenizerFeatureExtractor)

    @require_torch
    def test_apply_transcription_request_single(self):
        processor = AutoProcessor.from_pretrained(self.checkpoint)

        audio_url = "https://huggingface.co/datasets/bezzam/vibevoice_samples/resolve/main/realtime_model/vibevoice_tts_german.wav"
        helper_outputs = processor.apply_transcription_request(audio=audio_url, prompt="About VibeVoice")

        conversation = [
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
        ]
        manual_outputs = processor.apply_chat_template(
            conversation,
            tokenize=True,
            return_dict=True,
        )

        for key in ("input_ids", "attention_mask", "input_values", "padding_mask"):
            self.assertIn(key, helper_outputs)
            self.assertTrue(helper_outputs[key].equal(manual_outputs[key]))

    @parameterized.expand([(1, "np"), (1, "pt"), (2, "np"), (2, "pt")])
    def test_apply_chat_template_audio(self, batch_size: int, return_tensors: str):
        self.skipTest("VibeVoiceAsrProcessor does not support chat templates with text-only inputs.")

    def test_apply_chat_template_assistant_mask(self):
        self.skipTest("VibeVoiceAsrProcessor does not support chat templates with text-only inputs.")

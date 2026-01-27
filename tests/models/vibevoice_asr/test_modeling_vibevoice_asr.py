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
"""Testing suite for the PyTorch VibeVoice ASR model."""

import unittest

import pytest

from transformers import (
    VibeVoiceASRConfig,
    VibeVoiceASRForConditionalGeneration,
    VibeVoiceAcousticTokenizerConfig,
    is_torch_available,
)
from transformers.testing_utils import (
    require_torch,
    slow,
    torch_device,
)

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor


if is_torch_available():
    import torch


class VibeVoiceASRModelTester:
    """
    Builds a tiny VibeVoice ASR config and synthetic inputs for testing.
    """

    def __init__(
        self,
        parent,
        audio_token_id=0,
        seq_length=25,
        audio_samples=24000,  # 1 second at 24kHz
        text_config=None,
        acoustic_tokenizer_config=None,
        semantic_tokenizer_config=None,
        is_training=True,
    ):
        self.parent = parent
        self.audio_token_id = audio_token_id
        self.seq_length = seq_length
        self.audio_samples = audio_samples
        self.is_training = is_training

        # Small text backbone (Qwen2-ish)
        if text_config is None:
            text_config = {
                "model_type": "qwen2",
                "intermediate_size": 36,
                "initializer_range": 0.02,
                "hidden_size": 32,
                "max_position_embeddings": 128,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "vocab_size": 100,
                "pad_token_id": 1,
            }

        # Small acoustic tokenizer config
        if acoustic_tokenizer_config is None:
            acoustic_tokenizer_config = {
                "model_type": "vibevoice_acoustic_tokenizer",
                "hidden_size": 16,
                "n_filters": 8,
                "downsampling_ratios": [2, 2, 4],
                "depths": [1, 1, 1, 2],
            }

        # Small semantic tokenizer config
        if semantic_tokenizer_config is None:
            semantic_tokenizer_config = {
                "model_type": "vibevoice_acoustic_tokenizer",
                "hidden_size": 32,
                "n_filters": 8,
                "downsampling_ratios": [2, 2, 4],
                "depths": [1, 1, 1, 2],
            }

        self.text_config = text_config
        self.acoustic_tokenizer_config = acoustic_tokenizer_config
        self.semantic_tokenizer_config = semantic_tokenizer_config

        self.batch_size = 2
        self.num_channels = 1

    def prepare_config_and_inputs(self):
        # Text input
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.text_config["vocab_size"])

        # Create attention mask
        attention_mask = torch.ones([self.batch_size, self.seq_length], dtype=torch.long)

        # Audio input (batch_size, samples)
        speech_tensors = floats_tensor([self.batch_size, self.audio_samples])

        # Mask for where to insert audio features in input_ids
        acoustic_input_mask = input_ids == self.audio_token_id

        config = self.get_config()

        return config, input_ids, attention_mask, speech_tensors, acoustic_input_mask

    def get_config(self):
        return VibeVoiceASRConfig(
            acoustic_tokenizer_config=self.acoustic_tokenizer_config,
            semantic_tokenizer_config=self.semantic_tokenizer_config,
            text_config=self.text_config,
            audio_token_id=self.audio_token_id,
            acoustic_vae_dim=16,
            semantic_vae_dim=32,
        )

    def create_and_check_model(self, config, input_ids, attention_mask, speech_tensors, acoustic_input_mask):
        model = VibeVoiceASRForConditionalGeneration(config=config)
        model.to(torch_device)
        model.eval()

        with torch.no_grad():
            result = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                speech_tensors=speech_tensors,
                acoustic_input_mask=acoustic_input_mask,
            )

        self.parent.assertIsNotNone(result.logits)
        self.parent.assertEqual(
            result.logits.shape,
            (self.batch_size, self.seq_length, self.text_config["vocab_size"]),
        )

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, input_ids, attention_mask, speech_tensors, acoustic_input_mask = config_and_inputs

        inputs_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "speech_tensors": speech_tensors,
            "acoustic_input_mask": acoustic_input_mask,
        }

        return config, inputs_dict


@require_torch
class VibeVoiceASRModelTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):
    all_model_classes = (VibeVoiceASRForConditionalGeneration,) if is_torch_available() else ()
    test_head_masking = False
    test_pruning = False
    test_resize_embeddings = False
    test_attention_outputs = False
    fx_compatible = False

    def setUp(self):
        self.model_tester = VibeVoiceASRModelTester(self)
        self.config_tester = ConfigTester(self, config_class=VibeVoiceASRConfig, has_text_modality=False)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    @unittest.skip("VibeVoice ASR does not support inputs_embeds in generation")
    def test_inputs_embeds(self):
        pass

    @unittest.skip("VibeVoice ASR does not support inputs_embeds in generation")
    def test_inputs_embeds_matches_input_ids(self):
        pass

    @unittest.skip("Not applicable for VibeVoice ASR")
    def test_model_common_attributes(self):
        pass

    @unittest.skip("Not applicable for VibeVoice ASR")
    def test_initialization(self):
        pass

    @slow
    @unittest.skip("Model not yet available on Hub")
    def test_model_from_pretrained(self):
        pass


@require_torch
class VibeVoiceASRModelIntegrationTest(unittest.TestCase):
    @slow
    @unittest.skip("Model not yet available on Hub")
    def test_inference(self):
        # TODO: Add integration test once model is available on Hub
        pass

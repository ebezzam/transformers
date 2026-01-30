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

import json
import unittest
from pathlib import Path

from transformers import (
    VibeVoiceAsrConfig,
    VibeVoiceAsrForConditionalGeneration,
    is_datasets_available,
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


if is_datasets_available():
    from datasets import Audio, load_dataset

if is_torch_available():
    import torch


class VibeVoiceAsrModelTester:
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
        return VibeVoiceAsrConfig(
            acoustic_tokenizer_config=self.acoustic_tokenizer_config,
            semantic_tokenizer_config=self.semantic_tokenizer_config,
            text_config=self.text_config,
            audio_token_id=self.audio_token_id,
            acoustic_vae_dim=16,
            semantic_vae_dim=32,
        )

    def create_and_check_model(self, config, input_ids, attention_mask, speech_tensors, acoustic_input_mask):
        model = VibeVoiceAsrForConditionalGeneration(config=config)
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
class VibeVoiceAsrModelTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):
    all_model_classes = (VibeVoiceAsrForConditionalGeneration,) if is_torch_available() else ()
    test_head_masking = False
    test_pruning = False
    test_resize_embeddings = False
    test_attention_outputs = False
    fx_compatible = False

    def setUp(self):
        self.model_tester = VibeVoiceAsrModelTester(self)
        self.config_tester = ConfigTester(self, config_class=VibeVoiceAsrConfig, has_text_modality=False)

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
class VibeVoiceAsrForConditionalGenerationIntegrationTest(unittest.TestCase):
    _dataset = None

    @classmethod
    def setUp(cls):
        from transformers import AutoProcessor
        from transformers.testing_utils import cleanup

        cleanup(torch_device, gc_collect=True)
        cls.checkpoint = "bezzam/VibeVoice-ASR-7B"
        cls.processor = AutoProcessor.from_pretrained(cls.checkpoint)

    def tearDown(self):
        from transformers.testing_utils import cleanup

        cleanup(torch_device, gc_collect=True)

    @classmethod
    def _load_dataset(cls):
        # Lazy loading of the dataset. Because it is a class method, it will only be loaded once per pytest process.
        if cls._dataset is None:
            cls._dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
            cls._dataset = cls._dataset.cast_column(
                "audio", Audio(sampling_rate=cls.processor.feature_extractor.sampling_rate)
            )

    def _load_datasamples(self, num_samples):
        self._load_dataset()
        ds = self._dataset
        speech_samples = ds.sort("id")[:num_samples]["audio"]
        return [x["array"] for x in speech_samples]

    @slow
    def test_integration_single(self):
        """
        reproducer: https://gist.github.com/ebezzam/e1200bcecdc29e87dadd9d8423ae7ecb#file-reproducer_vibevoice_asr-py
        """

        path = Path(__file__).parent.parent.parent / "fixtures/vibevoice_asr/expected_results_single.json"
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        exp_inp_ids = torch.tensor(raw["input_ids"])
        exp_gen_ids = torch.tensor(raw["generated_ids"])
        exp_txt = raw["transcriptions"]

        samples = self._load_datasamples(1)

        conversation = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "audio",
                        "audio": samples[0],
                    },
                ],
            }
        ]

        model = VibeVoiceAsrForConditionalGeneration.from_pretrained(
            self.checkpoint, device_map=torch_device, torch_dtype=torch.bfloat16
        ).eval()

        batch = self.processor.apply_chat_template(conversation, tokenize=True, return_dict=True).to(
            model.device, dtype=model.dtype
        )

        torch.testing.assert_close(batch["input_ids"].cpu(), exp_inp_ids)

        seq = model.generate(**batch, max_new_tokens=512)
        inp_len = batch["input_ids"].shape[1]
        gen_ids = seq[:, inp_len:] if seq.shape[1] >= inp_len else seq

        torch.testing.assert_close(gen_ids.cpu(), exp_gen_ids)
        txt = self.processor.batch_decode(gen_ids, skip_special_tokens=True)
        self.assertListEqual(txt, exp_txt)

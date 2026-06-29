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
"""Testing suite for the PyTorch OmniASR model."""

import json
import unittest
from pathlib import Path

from parameterized import parameterized

from transformers import is_datasets_available, is_torch_available
from transformers.testing_utils import cleanup, require_torch, slow, torch_device

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import (
    TEST_EAGER_MATCHES_SDPA_INFERENCE_PARAMETERIZATION,
    ModelTesterMixin,
    floats_tensor,
    ids_tensor,
)


if is_datasets_available():
    from datasets import Audio, load_dataset

if is_torch_available():
    import torch

    from transformers import (
        AutoProcessor,
        OmniASRForConditionalGeneration,
        OmniASRForCTC,
        OmniASRLLMConfig,
    )


class OmniASRModelTester:
    """
    Builds a tiny OmniASR LLM config (Wav2Vec2-style audio encoder + Llama decoder) and synthetic inputs.

    Unlike most audio-LMs, OmniASR does not splice audio embeddings into placeholder `input_ids`; the encoder output
    is projected and concatenated to the text context inside the model, so the synthetic inputs only contain raw
    `input_values` (waveform) and optional `language_ids`.
    """

    def __init__(
        self,
        parent,
        batch_size=3,
        audio_seq_length=1600,
        is_training=True,
        encoder_config=None,
        text_config=None,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.audio_seq_length = audio_seq_length
        self.is_training = is_training

        if encoder_config is None:
            encoder_config = {
                "hidden_size": 32,
                "conv_dim": [32, 32, 32, 32, 32, 32, 32],
                "conv_stride": [5, 2, 2, 2, 2, 2, 2],
                "conv_kernel": [10, 3, 3, 3, 3, 2, 2],
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "intermediate_size": 37,
                "num_conv_pos_embeddings": 16,
                "num_conv_pos_embedding_groups": 2,
            }
        if text_config is None:
            text_config = {
                "model_type": "llama",
                "hidden_size": 32,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "intermediate_size": 37,
                "vocab_size": 99,
                "pad_token_id": 1,
            }

        self.encoder_config = encoder_config
        self.text_config = text_config

        # Attributes consumed by the common mixins.
        self.vocab_size = text_config["vocab_size"]
        self.hidden_size = text_config["hidden_size"]
        self.num_attention_heads = text_config["num_attention_heads"]
        self.num_hidden_layers = text_config["num_hidden_layers"]
        self.pad_token_id = text_config["pad_token_id"]
        self.seq_length = 7

    def get_config(self):
        return OmniASRLLMConfig(
            encoder_config=self.encoder_config,
            text_config=self.text_config,
            language_mapping={"eng_latn": 1},
            language_token_id=90,
            num_special_tokens=1,
        )

    def prepare_config_and_inputs(self):
        input_values = floats_tensor([self.batch_size, self.audio_seq_length])
        config = self.get_config()
        return config, input_values

    def prepare_config_and_inputs_for_common(self):
        config, input_values = self.prepare_config_and_inputs()
        language_ids = ids_tensor([self.batch_size], config.num_language_embeddings)
        inputs_dict = {
            "input_values": input_values,
            "language_ids": language_ids,
        }
        return config, inputs_dict


@require_torch
class OmniASRForConditionalGenerationModelTest(ModelTesterMixin, unittest.TestCase):
    # OmniASR uses a custom audio-prompted `generate` that assembles the decoder context
    # (`audio | lid_marker | lang_id | bos`) from audio embeddings, so it does not fit the `input_ids`-based
    # `GenerationTesterMixin` contract. Per `test_generation_tester_mixin_inheritance`, the sanctioned way to opt out
    # of the standard generation battery is to clear `all_generative_model_classes`; generation is covered by
    # `test_generate` here and by the slow integration tests below.
    all_model_classes = (OmniASRForConditionalGeneration,) if is_torch_available() else ()
    all_generative_model_classes = ()
    # NOTE: OmniASR is not (yet) wired into the `automatic-speech-recognition` pipeline (its audio-prompted
    # `generate` does not match the pipeline's input contract), so no `pipeline_model_mapping` is declared.
    _is_composite = True
    test_pruning = False
    test_headmasking = False
    test_resize_embeddings = False

    def setUp(self):
        self.model_tester = OmniASRModelTester(self)
        self.config_tester = ConfigTester(self, config_class=OmniASRLLMConfig, has_text_modality=False)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_forward(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            model = model_class(config).to(torch_device).eval()
            with torch.no_grad():
                logits = model(**inputs_dict).logits
            # context = audio_frames + lid_marker + lang_id + bos
            self.assertEqual(logits.shape[0], self.model_tester.batch_size)
            self.assertEqual(logits.shape[-1], config.vocab_size)

    def test_training_loss_and_backward(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        labels = ids_tensor([self.model_tester.batch_size, 5], config.vocab_size - 1) + 1
        for model_class in self.all_model_classes:
            model = model_class(config).to(torch_device).train()
            out = model(**inputs_dict, labels=labels)
            self.assertIsNotNone(out.loss)
            out.loss.backward()
            # gradients must reach all three trainable sub-modules, not merely the projector
            self.assertTrue(any(p.grad is not None for p in model.encoder.parameters()))
            self.assertTrue(any(p.grad is not None for p in model.language_model.parameters()))
            self.assertIsNotNone(model.multi_modal_projector.weight.grad)

    def test_generate(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            model = model_class(config).to(torch_device).eval()
            with torch.no_grad():
                generated = model.generate(**inputs_dict, max_new_tokens=4, do_sample=False)
            self.assertEqual(generated.shape[0], self.model_tester.batch_size)
            self.assertTrue(1 <= generated.shape[1] <= 4, f"unexpected generated length {generated.shape[1]}")

    @unittest.skip(
        reason="OmniASR builds its own input embeddings from audio; it has no input_ids/embeds equivalence."
    )
    def test_inputs_embeds_matches_input_ids(self):
        pass

    @unittest.skip(reason="OmniASR is prompted by audio embeddings, not input_ids; inputs_embeds path is internal.")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="OmniASR has no separate base model without a head.")
    def test_model_base_model_prefix(self):
        pass

    @parameterized.expand(TEST_EAGER_MATCHES_SDPA_INFERENCE_PARAMETERIZATION)
    @unittest.skip(
        reason="OmniASR assembles the decoder context from audio embeddings; eager/SDPA equivalence with input_ids "
        "padding does not apply."
    )
    def test_eager_matches_sdpa_inference(self, *args):
        pass

    @unittest.skip(reason="OmniASR's unified output does not expose encoder attentions in the common format.")
    def test_attention_outputs(self):
        pass

    @unittest.skip(reason="OmniASR's unified output does not expose encoder hidden states in the common format.")
    def test_hidden_states_output(self):
        pass

    @unittest.skip(reason="OmniASR's unified output does not expose encoder hidden states/attentions to retain grad.")
    def test_retain_grad_hidden_states_attentions(self):
        pass

    @unittest.skip(reason="Flex attention is not supported by the OmniASR audio encoder path.")
    def test_flex_attention_with_grads(self, *args):
        pass

    def test_streaming_generate(self):
        # Streaming ("unlimited"-length) decode: a 0.05s segment is 800 samples, so a 1500-sample clip spans 2
        # segments and exercises the segment loop + marker tokens + context carry-over.
        tester = self.model_tester
        config = OmniASRLLMConfig(
            encoder_config=tester.encoder_config,
            text_config=tester.text_config,
            language_mapping={"eng_latn": 1},
            language_token_id=90,
            num_special_tokens=3,
            is_streaming=True,
            segment_seconds=0.05,
            num_context_segments=1,
        )
        model = OmniASRForConditionalGeneration(config).to(torch_device).eval()
        input_values = floats_tensor([1, 1500])
        language_ids = ids_tensor([1], config.num_language_embeddings)
        # no outer torch.no_grad(): the streaming path must manage that itself
        generated = model.generate(input_values=input_values, language_ids=language_ids, max_new_tokens=3)
        self.assertEqual(generated.shape[0], 1)
        self.assertGreaterEqual(generated.shape[1], 1)


@require_torch
class OmniASRForCTCIntegrationTest(unittest.TestCase):
    _dataset = None

    @classmethod
    def setUpClass(cls):
        cls.checkpoint_name = "bezzam/omniasr-ctc-300m-v2"
        cls.dtype = torch.float32
        cls.processor = AutoProcessor.from_pretrained("bezzam/omniasr-ctc-300m-v2")

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    @classmethod
    def _load_dataset(cls):
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
    def test_ctc_300m_v2_model_integration(self):
        """
        reproducer (creates JSON directly in repo): https://gist.github.com/ebezzam/26af2bd40fa207af322de39701179650#file-reproducer_ctc-py
        """
        RESULTS_PATH = Path(__file__).parent.parent.parent / "fixtures/omniasr/expected_results_single.json"
        with open(RESULTS_PATH, "r") as f:
            raw_data = json.load(f)
        EXPECTED_TOKEN_IDS = torch.tensor(raw_data["pred_ids"])
        EXPECTED_TRANSCRIPTIONS = raw_data["transcriptions"]

        samples = self._load_datasamples(1)
        model = OmniASRForCTC.from_pretrained(self.checkpoint_name, torch_dtype=self.dtype, device_map="auto")
        model.eval()
        model.to(torch_device)

        inputs = self.processor(
            samples, return_tensors="pt", sampling_rate=self.processor.feature_extractor.sampling_rate
        )
        inputs.to(torch_device, dtype=self.dtype)
        with torch.no_grad():
            logits = model(**inputs).logits
        predicted_ids = torch.argmax(logits, dim=-1)

        torch.testing.assert_close(predicted_ids.cpu(), EXPECTED_TOKEN_IDS)
        predicted_transcripts = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)
        self.assertListEqual(predicted_transcripts, EXPECTED_TRANSCRIPTIONS)

    @slow
    def test_ctc_300m_v2_model_integration_batched(self):
        """
        reproducer (creates JSON directly in repo): https://gist.github.com/ebezzam/26af2bd40fa207af322de39701179650#file-reproducer_ctc_batch-py
        NOTE: only compare transcripts because of differences in batch padding
        """
        RESULTS_PATH = Path(__file__).parent.parent.parent / "fixtures/omniasr/expected_results_batch.json"
        with open(RESULTS_PATH, "r") as f:
            raw_data = json.load(f)
        EXPECTED_TRANSCRIPTIONS = raw_data["transcriptions"]

        samples = self._load_datasamples(3)
        model = OmniASRForCTC.from_pretrained(self.checkpoint_name, torch_dtype=self.dtype, device_map="auto")
        model.eval()
        model.to(torch_device)

        inputs = self.processor(
            samples, return_tensors="pt", sampling_rate=self.processor.feature_extractor.sampling_rate, padding=True
        )
        inputs.to(torch_device, dtype=self.dtype)
        with torch.no_grad():
            logits = model(**inputs).logits
        predicted_ids = torch.argmax(logits, dim=-1)

        predicted_transcripts = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)
        self.assertListEqual(predicted_transcripts, EXPECTED_TRANSCRIPTIONS)


@require_torch
class OmniASRForConditionalGenerationIntegrationTest(unittest.TestCase):
    _dataset = None

    @classmethod
    def setUpClass(cls):
        cls.checkpoint_name = "bezzam/omniasr-llm-300m-v2"
        cls.dtype = torch.float32
        cls.processor = AutoProcessor.from_pretrained("bezzam/omniasr-llm-300m-v2")

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    @classmethod
    def _load_dataset(cls):
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
    def test_llm_300m_v2_model_integration(self):
        """
        reproducer (creates JSON directly in repo): https://gist.github.com/ebezzam/26af2bd40fa207af322de39701179650#file-reproducer_llm-py
        """
        RESULTS_PATH = Path(__file__).parent.parent.parent / "fixtures/omniasr/expected_results_single_llm.json"
        with open(RESULTS_PATH, "r") as f:
            raw_data = json.load(f)
        EXPECTED_TOKEN_IDS = torch.tensor(raw_data["pred_ids"])
        EXPECTED_TRANSCRIPTIONS = raw_data["transcriptions"]

        samples = self._load_datasamples(1)
        model = OmniASRForConditionalGeneration.from_pretrained(
            self.checkpoint_name, torch_dtype=self.dtype, device_map="auto"
        )
        model.eval()
        model.to(torch_device)

        inputs = self.processor(
            samples,
            return_tensors="pt",
            sampling_rate=self.processor.feature_extractor.sampling_rate,
            language=["eng_Latn"],
        )
        inputs.to(torch_device, dtype=self.dtype)
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=256,
            )

        torch.testing.assert_close(generated_ids.cpu(), EXPECTED_TOKEN_IDS)
        predicted_transcripts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        self.assertListEqual(predicted_transcripts, EXPECTED_TRANSCRIPTIONS)

    @slow
    def test_llm_300m_v2_model_integration_batched(self):
        """
        reproducer (creates JSON directly in repo): https://gist.github.com/ebezzam/26af2bd40fa207af322de39701179650#file-reproducer_llm_batch-py
        """
        RESULTS_PATH = Path(__file__).parent.parent.parent / "fixtures/omniasr/expected_results_batch_llm.json"
        with open(RESULTS_PATH, "r") as f:
            raw_data = json.load(f)
        EXPECTED_TRANSCRIPTIONS = raw_data["transcriptions"]

        samples = self._load_datasamples(3)
        model = OmniASRForConditionalGeneration.from_pretrained(
            self.checkpoint_name, torch_dtype=self.dtype, device_map="auto"
        )
        model.eval()
        model.to(torch_device)

        inputs = self.processor(
            samples,
            return_tensors="pt",
            sampling_rate=self.processor.feature_extractor.sampling_rate,
            padding=True,
            language=["eng_Latn"] * len(samples),
        )
        inputs.to(torch_device, dtype=self.dtype)
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=256,
            )

        predicted_transcripts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        self.assertListEqual(predicted_transcripts, EXPECTED_TRANSCRIPTIONS)

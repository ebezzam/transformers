# OmniASR port — progress & handoff

Continues **PR #43265** (`ebezzam:omnilingual`, head `96f6b65` "Start modular switch") in this fresh clone
(`/mnt/raid_drive/drive_data/drive_2/omnilingual_asr_hf_port`, local branch `omnilingual`). Focus: the **LLM
variant** (`OmniASRForConditionalGeneration`); CTC kept as-is. Goal: working + finetune-ready first, then polished
toward a human-owned, mergeable PR.

## TL;DR — state is green
- Both `@slow` integration suites pass in float32: `OmniASRForCTCIntegrationTest` (2) and
  `OmniASRForConditionalGenerationIntegrationTest` (2), against `bezzam/omniasr-{ctc,llm}-300m-v2`.
- New fast unit suite: **51 passed, 65 skipped** (`OmniASRForConditionalGenerationModelTest`).
- Repo-consistency (omniasr): `ruff`, `check_modular_conversion` (in sync), `check_repo`, `check_config_attributes`,
  `check_config_docstrings`, `check_copies`, `check_dummies`, `check_inits` — all rc=0.
- Finetune path verified: tiny-config forward → loss (≈ln(vocab)) → backward → generate all work.

## Pre-PR audit — bugs found & fixed
A deliberate edge-case / silent-error sweep (not just re-running the checks) found and **fixed**:
- **Batched-audio attention masking (silent correctness).** A padded batch's raw-waveform `attention_mask` was
  mis-applied (Llama sliced its first entries as the decoder mask), so unequal-length batched audio was not masked in
  `forward`/`generate`. Added `_build_audio_context_attention_mask` (down-samples the audio mask to audio frames, marks
  the context/target tokens valid); used by both `forward` and `generate`. All-ones / full-length inputs are unchanged
  (the single-sample integration test stays byte-identical) while padded batches now mask padding. Verified.
- **CTC training crashed** (`OmniASRCTCConfig` lacked `add_adapter`, then `ctc_loss_reduction`/`ctc_zero_infinity`).
  Fixed: `_get_feat_extract_output_lengths` is now called on `self.encoder`, and the two CTC-loss fields were added to
  `OmniASRCTCConfig`. CTC train+backward now works (CTC inference already did).
- **Converter passed the 15 pruned config fields** to `OmniASRConfig` (absorbed by `**kwargs` but messy, and referenced
  a removed param). Cleaned the `OmniASRConfig(...)` call and removed the now-orphaned `TransformerNormOrder` import.
- **Misleading `pipeline_model_mapping`** (claimed ASR-pipeline support that isn't wired up) — removed with a note.
- **bf16** forward+generate verified (integration tests only cover float32).
- Re-verified after all fixes: full suite **57 passed / 63 skipped** with `RUN_SLOW=1` (all 4 integration tests pass);
  all omniasr consistency checks rc=0.

## Environment (already provisioned; reused)
- conda env **`omniasr`** (py3.12, torch 2.8+cu128, fairseq2 0.6, omnilingual-asr 0.1.0, sentencepiece).
- Re-pointed its editable `transformers` to this clone: `pip install -e . --no-deps`.
- Added to the env this session: `accelerate` (needed by `device_map="auto"` integration tests), `ruff==0.13.1`,
  `libcst`, `parameterized`, `sentencepiece`, `torchvision==0.23.0+cu128` (needed only so `check_repo` can import all models).
- Run tests/checks with the env python, e.g. `/home/user/anaconda3/envs/omniasr/bin/python`. GPUs are shared/busy;
  use `CUDA_VISIBLE_DEVICES=0`.

## What changed vs `96f6b65`
- **New:** `src/transformers/models/omniasr/__init__.py` (was missing — lazy module), `docs/source/en/model_doc/omniasr.md`
  (+ `_toctree.yml` entry, alphabetical in audio models).
- **2 real finetune-path bug fixes** in `modular_omniasr.py` → regenerated `modeling_omniasr.py`
  (`OmniASRForConditionalGeneration.forward`); only the direct/`labels` path was affected, generation untouched
  (integration tests still pass):
  1. `inputs_embeds = embed(input_ids)` crashed when called with audio only (`input_ids=None`). Guarded with
     `and input_ids is not None`.
  2. `attention_mask`/`cache_position` were built at the **audio-context** length, before `labels` were appended →
     RoPE length mismatch. Removed the premature build; the existing fallback now sizes the mask from the final
     `inputs_embeds`, and Llama computes `cache_position`.
- **auto_docstring fixes** (modular): documented `input_values`/`language_ids` on the LLM forward; added a checkpoint
  markdown link to `OmniASRConfig` docstring (resolves the CTC `forward` "no checkpoint" error). Checkpoint links also
  added to `OmniASRCTCConfig`/`OmniASRLLMConfig` docstrings.
- **Config prune** (`OmniASRConfig`): removed 15 unused fairseq2-legacy fields (`max_seq_len`, `feature_dim`,
  `use_fbank`, `num_fbank_channels`, `fbank_stride`, `sample_fbank_every_k`, `pos_encoder_depth`, `use_conformer`,
  `depthwise_conv_kernel_size`, `first_pass_dropout_p`, `layer_norm_features`, `feature_grad_scale`,
  `position_embeddings_type`, `layer_norm_pre`, `use_intermediate_ffn_before_adapter`). Old checkpoints still load
  (extra json fields are ignored). `OmniASRLLMConfig.language_mapping` allow-listed in `utils/check_config_attributes.py`
  (used by the processor, not the model).
- **Auto-map fix:** removed a duplicate `omniasr` key in `MODEL_NAMES_MAPPING` (`configuration_auto.py`).
- **Tests:** added `OmniASRModelTester` + `OmniASRForConditionalGenerationModelTest` (`ModelTesterMixin`); kept the 4
  integration tests. `OmniASRForCTC`/`OmniASRModel` added to `IGNORE_NON_TESTED` (CTC covered by integration; encoder
  exercised internally).
- Style/whitespace cleanup (ruff) across the omniasr files.

## Key decisions
- **CTC kept, not extended.** Per the LLM-only focus, no new CTC unit work; its integration tests already pass.
- **Generation tests opted out the sanctioned way.** OmniASR's `generate` is audio-prompted (no `input_ids` prompt;
  returns only new tokens), so it can't satisfy `GenerationTesterMixin`. Per `test_generation_tester_mixin_inheritance`,
  set `all_generative_model_classes = ()` and do **not** inherit `GenerationTesterMixin`. Generation is covered by
  `test_generate` (tiny) + the integration tests. SDPA-eager equivalence and encoder attn/hidden-state output tests are
  skipped (audio-prompt architecture) using the blip_2 parametrized-skip pattern.
- **Tokenizer:** current approach (Wav2Vec2CTCTokenizer from the checkpoint + processor `language_mapping` →
  `language_ids` → learned `lang_embeddings`) is functional and passes the language-conditioned integration tests. A
  dedicated SentencePiece-backed `OmniASRTokenizer` remains an optional refinement.

## Remaining risks / recommended follow-ups
1. **Variable-length labels in batched training.** Audio padding is now masked correctly (see audit), but the loss
   appends a single EOS at the batch-max label length, so for shorter sequences in a mixed-length batch the EOS lands
   after the label padding. Pad-label positions are ignored via `ignore_index`, so this is a minor alignment nuance, not
   a crash — single-sample / equal-length training is exact. Per-sample EOS placement is a recommended refinement.
2. **Dedicated CTC/encoder unit testers.** For full parity with the sibling LASR, add `OmniASRModelTest` (encoder) and
   `OmniASRForCTCModelTest` and remove the two `IGNORE_NON_TESTED` entries. (Deferred per LLM-only focus.)
3. **Deeper modeling prune.** The generated `modeling_omniasr.py` still carries unused Wav2Vec2 machinery (adapter,
   stable-layer-norm, group-norm, feature-projection variants) pulled in via inheritance. Trimming the inheritance would
   shrink it further but needs care.
4. **`make repo-consistency` full run.** Done per-check with the env python (all omniasr checks green). `make` uses the
   base PATH python (wrong transformers) — run checks via the env python, or `conda activate omniasr` first.
   `check_docstrings` still reports unrelated **non-omniasr** objects (branch is ~months behind `main`); these clear on
   a rebase.
5. **Rebase onto current `main`** before opening the PR (this branch sits on ebezzam's older base).
6. **Conversion parity** is validated via the integration fixtures (dumped from the original model). A standalone
   reproducer vs the `omnilingual-asr` package is a nice extra, especially if regenerating checkpoints.

## How to verify
```bash
PY=/home/user/anaconda3/envs/omniasr/bin/python
cd /mnt/raid_drive/drive_data/drive_2/omnilingual_asr_hf_port
# fast unit tests
CUDA_VISIBLE_DEVICES=0 $PY -m pytest tests/models/omniasr/test_modeling_omniasr.py::OmniASRForConditionalGenerationModelTest -q
# integration (downloads bezzam/omniasr-*-300m-v2)
CUDA_VISIBLE_DEVICES=0 RUN_SLOW=1 $PY -m pytest tests/models/omniasr/test_modeling_omniasr.py -q
# consistency
$PY utils/check_modular_conversion.py --files src/transformers/models/omniasr/modular_omniasr.py
for c in check_repo check_config_attributes check_config_docstrings check_copies check_dummies check_inits; do $PY utils/$c.py; done
/home/user/anaconda3/envs/omniasr/bin/ruff check src/transformers/models/omniasr/ tests/models/omniasr/
```

## Handoff
- **Do not auto-open a PR.** This must be human-owned (HF anti-agent-PR policy). StephennFernandes owns it and should
  coordinate with `ebezzam`/`ArthurZucker`, ideally pushing on top of / crediting ebezzam's branch. Apache-2.0 throughout.
- Edit the model only via `modular_omniasr.py` then regenerate (`utils/modular_model_converter.py`); never hand-edit
  generated `modeling_omniasr.py`.

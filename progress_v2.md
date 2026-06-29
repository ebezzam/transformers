# OmniASR v2 + new-capabilities port — plan & progress

Extends the v1 port to the **v2 release** of Omnilingual ASR and its **new capabilities** (Unlimited/streaming
long-audio + Zero-Shot in-context), HF- and ebezzam-style compliant.

## Environment (isolated, set up this session)
- **`omniasr_v2`** conda env = clone of `omniasr` (keeps working fairseq2 0.6 / torch 2.8) with omnilingual-asr
  **0.2.0** editable from a fresh upstream clone at `/mnt/raid_drive/drive_data/drive_2/omnilingual-asr-upstream`.
  Your original `omniasr` env + v0.1.0 clone are **untouched**.
- Verified: `Wav2Vec2LlamaStreamingConfig`, `ModelType.{LLM_ASR,LLM_ASR_LID,ZERO_SHOT}`, and the v2/Unlimited/ZS
  cards all resolve; my transformers port imports.
- `fairseq2 0.6` satisfies omnilingual-asr 0.2.0 (`>=0.5.2,<=0.6.0`) — no reinstall needed.

## The v2 matrix (from `rc_models_v2.yaml`; tokenizer `omniASR_tokenizer_written_v2` everywhere)
- **CTC v2** 300M/1B/3B/7B, **LLM v2** 300M/1B/3B/7B — identical architecture to v1 → conversion only.
- **LLM Unlimited v2** 300M/1B/3B/7B — LLM + `is_streaming` → conversion **+ new streaming generate**.
- **Zero-Shot** `omniASR_LLM_7B_ZS` (v1 card) — `encoder_stacking=3`, 10 in-context pairs → **new modeling**.

## Phase 1 — v2 CTC+LLM weights (same architecture) — IN PROGRESS
Converting `omniASR_{CTC,LLM}_{300M,1B,3B,7B}_v2` with the (already v2-aware) converter: `"v2"` cards route to the
`written_v2` tokenizer; the size-detection + local-save fixes from the v1 work apply. Reload-verify + transcription
check each. No modeling changes.

## Phase 2 — Unlimited / streaming (new capability)
Upstream `create_streaming_syntax` (model.py): decoder sequence is
`[lang] ( audio_i · <segment_marker_i> · <bos> · text_i · <eos> ) × N`, audio split into `segment_secs`=15 s
windows; `regular_segment`/`last_segment` marker tokens distinguish the final window; during inference the text+eos of
the in-progress segment are generated and previous segments stay in context (`n_context_segments`=1).
**HF design (modular, ebezzam style):**
- Add a streaming sub-config (segment_secs, n_context_segments, sample_rate, min_audio_ms, segment marker token ids)
  to `OmniASRLLMConfig`; carried from `Wav2Vec2LlamaStreamingConfig` by the converter.
- Implement a long-audio streaming path in `OmniASRForConditionalGeneration.generate`: window `input_values`, loop
  segments building `inputs_embeds` incrementally (audio_seg | marker | bos), `super().generate` per segment to `<eos>`,
  carry prior (audio+text) embeddings as KV context, concatenate transcriptions. Falls back to the normal path for
  short audio / non-streaming configs.
- Convert + verify the 4 Unlimited checkpoints (weights are LLM-shaped; `is_streaming` is a config flag).

## Phase 3 — Zero-Shot (new capability)
Upstream: `model_type=ZERO_SHOT`, `encoder_stacking=3`, `n_special_tokens=6`, `n_context_examples=10`; forward uses
`create_text_context_syntax`/`create_audio_context_syntax` + `remove_context_logits`.
**HF design:**
- Implement the **encoder_stacking reshape** in `get_audio_features` (stack `encoder_stacking` consecutive frames →
  feature dim ×N, pad to multiple) — currently missing (only stacking=1 works); the projector width already accounts
  for it in config.
- Add an in-context forward/generate path accepting 10 `(context_audio, context_text)` pairs + query audio, building
  the demonstration prompt; a processor method to assemble examples; strip context logits for decode.
- Convert + verify `omniASR_LLM_7B_ZS`.

## Phase 4 — tests / docs / repo-consistency
Unit tests for the encoder_stacking reshape, the streaming segment loop (tiny config, 2 segments), and the ZS prompt
assembly; doc updates; modular kept in sync; `ruff`/`check_modular_conversion`/`check_repo`/`check_config_*` green.

## Disk / compute
Outputs → raid (2.6 TB). v2 originals download to `~/.cache/fairseq2` (home, ~215 GB free) — ~180 GB for the full v2
set; monitor and redirect/clean if tight. 7B variants convert in bf16 (f32 original+HF > 48 GB).

## Status
- [x] Isolated `omniasr_v2` env + newer source + v2 cards verified.
- [~] Phase 1: v2 CTC+LLM converting (8 models).
- [ ] Phase 2: streaming modeling + Unlimited conversion.
- [ ] Phase 3: ZS modeling + conversion.
- [ ] Phase 4: tests/docs/consistency.

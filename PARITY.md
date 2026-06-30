# OmniASR HF Port — Tier-1 Layer-by-Layer Parity

Evidence that the HuggingFace port reproduces Meta's original `omnilingual-asr` (fairseq2) models
**exactly**, validated activation-by-activation through the full forward pass for **every model, every size,
and every variant** — v1 and v2, CTC and LLM (300M→7B), plus the **Unlimited (streaming)** and **Zero-Shot**
models.

This is the numerical-equivalence check the `add_new_model` workflow expects (per-layer hidden states
agreeing to `≤1e-3`), run at maximum strength: a same-framework float32 conversion makes the bound
**bit-exact (`max_abs_diff == 0.0`)** for the shared encoder, the CTC head, and the audio projector — i.e.
not merely within tolerance but identical to the bit.

## Results — encoder + CTC head / audio projector (float32)

Each run feeds the **same** normalized `input_values` to both models and compares the conv frontend, **every**
transformer layer, the final encoder norm, and the head/projector output.

| Model | release | enc. layers | max abs diff | verdict |
|-------|:-------:|:-----------:|:------------:|---------|
| omniASR_CTC_300M / _v2 | v1 / v2 | 24 | **0.0** | ✅ bit-exact |
| omniASR_CTC_1B / _v2 | v1 / v2 | 48 | **0.0** | ✅ bit-exact |
| omniASR_CTC_3B / _v2 | v1 / v2 | 60 | **0.0** | ✅ bit-exact |
| omniASR_CTC_7B / _v2 | v1 / v2 | 128 | **0.0** | ✅ bit-exact |
| omniASR_LLM_300M / _v2 | v1 / v2 | 24 | **0.0** | ✅ bit-exact |
| omniASR_LLM_1B / _v2 | v1 / v2 | 48 | **0.0** | ✅ bit-exact |
| omniASR_LLM_3B / _v2 | v1 / v2 | 60 | **0.0** | ✅ bit-exact |
| omniASR_LLM_7B / _v2 | v1 / v2 | 128 | **0.0** | ✅ bit-exact |
| omniASR_LLM_Unlimited_300M_v2 (streaming) | v2 | 24 | **0.0** | ✅ bit-exact |
| omniASR_LLM_7B_ZS (zero-shot) | v1 | 128 | **0.0** | ✅ bit-exact |

Every single stage (e.g. all 128 layers of a 7B encoder, plus the CTC logits / the audio projector) reports
`max_abs_diff = 0.000e+00`.

## Results — Llama decoder (LLM)

OmniASR attaches a **shared 12-layer / 4096-d Llama decoder** to every LLM variant (the size scales the audio
encoder, not the decoder). Parity feeds one identical `inputs_embeds` (projected audio ++ text embeds) to
both decoders and compares each decoder layer + the LM logits.

| Model | max **rel** diff | argmax match | verdict |
|-------|:----------------:|:------------:|---------|
| omniASR_LLM_300M (v1) | 7.9e-7 | ✅ every position | ✅ |
| omniASR_LLM_1B (v1) | 8.6e-7 | ✅ every position | ✅ |
| omniASR_LLM_3B (v1) | 7.9e-7 | ✅ every position | ✅ |
| omniASR_LLM_7B (v1) | 9.2e-7 | ✅ every position | ✅ |
| omniASR_LLM_300M_v2 | 1.3e-6 | ✅ every position | ✅ |
| omniASR_LLM_1B_v2 | 1.1e-6 | ✅ every position | ✅ |
| omniASR_LLM_3B_v2 | 1.1e-6 | ✅ every position | ✅ |
| omniASR_LLM_7B_v2 | 9.8e-7 | ✅ every position | ✅ |
| omniASR_LLM_Unlimited_300M_v2 (streaming) | 8.1e-7 | ✅ every position | ✅ |
| omniASR_LLM_7B_ZS (zero-shot) | 4.3e-7 | ✅ every position | ✅ |

The decoder shows a tiny **relative** diff (~1e-6) rather than exactly 0 because HF and fairseq2 apply
attention/RoPE in a slightly different op order, so float32 rounding differs in the last digit. It looks
larger in absolute terms (2.4e-3) only because Llama hidden states contain "massive activations" — a few
dimensions reach magnitude **~8000** — so `2.4e-3 / 8000 ≈ 3e-7`. The LM logits differ by ~1e-5 and the
**predicted token is identical at every position**.

## Methodology

- **Identical-input isolation.** Both models receive the *same* normalized `input_values` from the HF
  processor (the original's collater does not normalize — normalization lives in its audio pipeline, which we
  bypass). Feeding one tensor to both forwards isolates the *weight/architecture conversion*: any drift
  localizes to a specific layer rather than to preprocessing.
- **Forward hooks** capture every aligned stage:
  - original `encoder_frontend.feature_extractor` ↔ HF `encoder.feature_extractor` (conv frontend)
  - original `encoder.layers[i]` ↔ HF `encoder.encoder.layers[i]` (each transformer layer)
  - original `encoder.layer_norm` ↔ HF `encoder.encoder.layer_norm` (final encoder norm)
  - CTC: original `final_proj` ↔ HF `ctc_head`; LLM: original `encoder_proj` ↔ HF `multi_modal_projector`
  - LLM decoder: identical `inputs_embeds` fed to original `llama_decoder`+`final_proj` and to HF
    `language_model`; compares each decoder layer + the LM logits.
- **float32** throughout (the strongest setting; bf16 would only ever match to ~1e-2).
- **Memory-safe**: original and HF run sequentially (original → capture to CPU → free GPU → HF → capture →
  compare), so even 7B float32 fits on a single 48 GB GPU.
- **Streaming / Zero-Shot** models gate their full forward on streaming/context inputs, so their **encoder**
  is driven directly (`encoder_frontend` → `encoder`) on the shared `input_values`; the streaming
  segmentation loop and zero-shot syntax are weightless and covered by the integration/streaming tests.

## Notes

- **7B precision / storage.** The 7B checkpoints **ship in bfloat16** — exactly how the original infers them
  (`omnilingual-asr`'s pipeline defaults to bf16). The bit-exact rows use a float32 re-conversion of the 7B.
  Getting them bit-exact required removing weight-norm on the **GPU** (CPU vs GPU float32 rounding in baking
  `pos_conv` otherwise leaves a relative ~1e-6 residual that appears only *after* the weight-normed
  `pos_conv`, while the conv frontend before it stays bit-exact). The converter now loads the original on CPU
  and keeps the HF model on GPU, so weight-norm bakes on GPU and the 7B float32 conversion also fits a single
  GPU. (This same fix unblocked the **Zero-Shot** conversion, which previously OOM'd.)
- **Vocab.** The HF LM head carries a few extra rows vs the original `final_proj` (`+num_special_tokens`: the
  language/segment markers the original keeps in separate embeddings). The shared text-token logits align
  exactly; parity compares the common vocab and confirms argmax agreement.

## Reproduction

```bash
# encoder + CTC-logits / audio-projector parity  (variant = ctc | llm)
python scripts/parity/parity_harness.py   <model_card> <hf_path> <variant>  out.json
# Llama decoder layer + logit parity (LLM)
python scripts/parity/decoder_parity.py   <model_card> <hf_path>            out.json
# encoder-only parity for streaming / zero-shot models (full forward needs streaming/context inputs)
python scripts/parity/zs_encoder_parity.py <model_card> <hf_path>
```

Run with the `omnilingual-asr` package installed (provides the original fairseq2 models) against the converted
HF checkpoints. For 7B / Zero-Shot float32 parity, re-convert on a GPU host (original loads on CPU, HF on GPU).

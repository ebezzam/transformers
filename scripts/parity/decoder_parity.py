"""Tier-1 decoder parity for the LLM variant: isolates the Llama (RoPE-permuted) conversion.

Builds ONE inputs_embeds sequence = [projected audio embeds] ++ [text token embeds] on the HF side, then feeds
the IDENTICAL tensor to BOTH decoders and compares per-layer hidden states + final logits:
  original  m.llama_decoder(embeds, layout) -> m.final_proj
  HF        hf.language_model(inputs_embeds=embeds) -> .logits

Same input embeds => any drift localizes to the decoder weights / RoPE handling. Memory-safe (sequential).

Usage:  decoder_parity.py <model_card> <hf_path> [out_json]
"""
import gc
import json
import sys

import torch
from datasets import Audio, load_dataset

model_card, hf_path = sys.argv[1], sys.argv[2]
out_json = sys.argv[3] if len(sys.argv) > 3 else None
dev, dtype = torch.device("cuda"), torch.float32
TOL = 1e-3

from transformers import AutoProcessor, OmniASRForConditionalGeneration  # noqa: E402

ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation").cast_column(
    "audio", Audio(sampling_rate=16000)
)
audio = ds.sort("id")[0]["audio"]["array"]
proc = AutoProcessor.from_pretrained(hf_path)
iv = proc(audio, sampling_rate=16000, return_tensors="pt")["input_values"].to(dev, dtype)


def mk(store, name):
    def fn(_m, _i, out):
        t = out[0] if isinstance(out, (tuple, list)) else out
        if torch.is_tensor(t):
            store[name] = t.detach().float().cpu()
    return fn


# ===== HF: construct inputs_embeds (audio ++ text) and capture decoder layers + logits =====
hf_acts = {}
hf = OmniASRForConditionalGeneration.from_pretrained(hf_path, dtype=dtype).to(dev).eval()
n_dec = len(hf.language_model.model.layers)
for i, layer in enumerate(hf.language_model.model.layers):
    layer.register_forward_hook(mk(hf_acts, f"dec_layer_{i:02d}"))
with torch.no_grad():
    audio_embeds = hf.get_audio_features(iv)  # [1, Ta, D]
    # a fixed, arbitrary text continuation (ids within vocab); same tensor goes to both decoders
    text_ids = torch.tensor([[1, 100, 1000, 5000, 2000, 42, 7, 2]], device=dev)
    text_embeds = hf.language_model.model.embed_tokens(text_ids)
    inputs_embeds = torch.cat([audio_embeds, text_embeds], dim=1).to(dtype)  # [1, S, D]
    logits_hf = hf.language_model(inputs_embeds=inputs_embeds, use_cache=False).logits.float().cpu()
S = inputs_embeds.shape[1]
ie_cpu = inputs_embeds.detach().cpu()
del hf
gc.collect()
torch.cuda.empty_cache()

# ===== ORIGINAL: feed the IDENTICAL inputs_embeds through llama_decoder + final_proj =====
orig_acts = {}
from fairseq2.nn.batch_layout import BatchLayout  # noqa: E402

from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline  # noqa: E402

pipe = ASRInferencePipeline(model_card=model_card, device=dev, dtype=dtype)
m = pipe.model.eval()
for i, layer in enumerate(m.llama_decoder.layers):
    layer.register_forward_hook(mk(orig_acts, f"dec_layer_{i:02d}"))
ie = ie_cpu.to(dev, dtype)
with torch.no_grad():
    layout = BatchLayout((1, S), seq_lens=[S], device=dev)
    dec_out = m.llama_decoder(ie, layout)
    logits_orig = m.final_proj(dec_out).float().cpu()
del pipe, m
gc.collect()
torch.cuda.empty_cache()

# ===== compare (report absolute AND relative; relative is the meaningful metric for the decoder,
# whose Llama "massive activations" make a few dims huge => abs float32 ulp noise looks large) =====
REL_TOL = 1e-4
print(f"=== DECODER PARITY {model_card}  vs  {hf_path.split('/')[-1]}  (S={S}, layers={n_dec}) ===")
overall_rel = 0.0
rows = {}
for i in range(n_dec):
    k = f"dec_layer_{i:02d}"
    if k not in orig_acts or k not in hf_acts:
        continue
    a, b = orig_acts[k], hf_acts[k]
    if a.shape != b.shape:
        print(f"  {k:14s} SHAPE MISMATCH {tuple(a.shape)} vs {tuple(b.shape)}")
        continue
    d = (a - b).abs().max().item()
    mag = a.abs().max().item()
    rel = d / (mag + 1e-12)
    overall_rel = max(overall_rel, rel)
    rows[k] = {"max_abs_diff": d, "max_abs_val": mag, "rel": rel, "shape": list(a.shape)}
    print(f"  {k:14s} abs={d:.3e}  |val|max={mag:8.1f}  rel={rel:.2e}  shape={tuple(a.shape)}")
# logits: compare on the common vocab slice (HF lm_head may carry 1 extra padded row)
V = min(logits_orig.shape[-1], logits_hf.shape[-1])
lo, lh = logits_orig[..., :V], logits_hf[..., :V]
dl = (lo - lh).abs().max().item()
lmag = lo.abs().max().item()
rell = dl / (lmag + 1e-12)
overall_rel = max(overall_rel, rell)
# argmax agreement (does the predicted token match at every position?)
argmax_match = bool((lo.argmax(-1) == lh.argmax(-1)).all())
rows["logits"] = {"max_abs_diff": dl, "max_abs_val": lmag, "rel": rell,
                  "vocab_orig": logits_orig.shape[-1], "vocab_hf": logits_hf.shape[-1],
                  "common_vocab": V, "argmax_match": argmax_match}
print(f"  {'logits':14s} abs={dl:.3e}  |val|max={lmag:8.1f}  rel={rell:.2e}  "
      f"vocab(orig={logits_orig.shape[-1]},hf={logits_hf.shape[-1]})  argmax_match={argmax_match}")
verdict = "PASS" if (overall_rel < REL_TOL and argmax_match) else "FAIL"
print(f"OVERALL decoder max_REL_diff = {overall_rel:.3e}  ->  {verdict}  (rel_tol {REL_TOL}, argmax must match)")

if out_json:
    with open(out_json, "w") as f:
        json.dump({"model_card": model_card, "kind": "decoder", "overall_max_rel_diff": overall_rel,
                   "verdict": verdict, "rel_tol": REL_TOL, "stages": rows}, f, indent=2)

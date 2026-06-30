"""Tier-1 layer-by-layer parity: original fairseq2 omnilingual-asr model vs the HF port.

Feeds the SAME normalized `input_values` to both models (isolating the weight/arch conversion), runs them
SEQUENTIALLY (original -> capture activations to CPU -> free GPU -> HF -> capture -> compare) so even the 7B
float32 models fit on one 48GB GPU. Reports max abs diff at every stage:

  feature extractor (conv) -> each encoder layer -> final encoder norm -> CTC logits | audio projector

Usage:  parity_harness.py <model_card> <hf_path> <ctc|llm>  [out_json]
"""

import gc
import json
import sys

import torch
from datasets import Audio, load_dataset


model_card, hf_path, variant = sys.argv[1], sys.argv[2], sys.argv[3]
out_json = sys.argv[4] if len(sys.argv) > 4 else None
dev, dtype = torch.device("cuda"), torch.float32
TOL = 1e-3

# --- shared normalized input (HF processor); kept on CPU, moved to GPU per-model ---
from transformers import AutoProcessor  # noqa: E402


ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation").cast_column(
    "audio", Audio(sampling_rate=16000)
)
audio = ds.sort("id")[0]["audio"]["array"]
proc = AutoProcessor.from_pretrained(hf_path)
input_values = proc(audio, sampling_rate=16000, return_tensors="pt")["input_values"].to(dtype)  # [1, T] CPU


def mk(store, name):
    def fn(_m, _i, out):
        t = out[0] if isinstance(out, (tuple, list)) else out
        if torch.is_tensor(t):
            store[name] = t.detach().float().cpu()

    return fn


# ============================ ORIGINAL (fairseq2) ============================
orig_acts = {}
from fairseq2.datasets.batch import Seq2SeqBatch  # noqa: E402
from fairseq2.nn.batch_layout import BatchLayout  # noqa: E402
from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline  # noqa: E402


pipe = ASRInferencePipeline(model_card=model_card, device=dev, dtype=dtype)
om = pipe.model.eval()
n_layers = len(om.encoder.layers)
for i, layer in enumerate(om.encoder.layers):
    layer.register_forward_hook(mk(orig_acts, f"enc_layer_{i:02d}"))
om.encoder.layer_norm.register_forward_hook(mk(orig_acts, "enc_final_norm"))
try:
    om.encoder_frontend.feature_extractor.register_forward_hook(mk(orig_acts, "feat_extract"))
except AttributeError:
    pass
if variant == "ctc":
    om.final_proj.register_forward_hook(mk(orig_acts, "head"))
else:
    om.encoder_proj.register_forward_hook(mk(orig_acts, "audio_proj"))

iv = input_values.to(dev)
with torch.no_grad():
    if variant == "ctc":
        bl = BatchLayout(iv.shape, seq_lens=[iv.shape[1]], device=iv.device)
        om(iv, bl)
    else:
        batch = Seq2SeqBatch(
            source_seqs=iv,
            source_seq_lens=torch.tensor([iv.shape[1]], device=dev),
            target_seqs=torch.tensor([[0]], device=dev, dtype=torch.int64),
            target_seq_lens=torch.tensor([1], device=dev),
            example={"lang": ["eng_Latn"]},
        )
        om(batch, return_decoder_inputs=True)

del pipe, om
gc.collect()
torch.cuda.empty_cache()

# ============================ HF PORT ============================
hf_acts = {}
from transformers import OmniASRForConditionalGeneration, OmniASRForCTC  # noqa: E402


Cls = OmniASRForCTC if variant == "ctc" else OmniASRForConditionalGeneration
hf = Cls.from_pretrained(hf_path, dtype=dtype).to(dev).eval()
enc = hf.encoder  # OmniASRModel wrapper
for i, layer in enumerate(enc.encoder.layers):
    layer.register_forward_hook(mk(hf_acts, f"enc_layer_{i:02d}"))
enc.encoder.layer_norm.register_forward_hook(mk(hf_acts, "enc_final_norm"))
try:
    enc.feature_extractor.register_forward_hook(mk(hf_acts, "feat_extract"))
except AttributeError:
    pass

iv = input_values.to(dev)
with torch.no_grad():
    if variant == "ctc":
        hf.ctc_head.register_forward_hook(mk(hf_acts, "head"))
        hf(iv)
    else:
        hf.multi_modal_projector.register_forward_hook(mk(hf_acts, "audio_proj"))
        hf.get_audio_features(iv)

del hf
gc.collect()
torch.cuda.empty_cache()

# ============================ COMPARE ============================
order = ["feat_extract"] + [f"enc_layer_{i:02d}" for i in range(n_layers)] + ["enc_final_norm", "head", "audio_proj"]
print(f"=== PARITY {model_card}  ({variant})  vs  {hf_path.split('/')[-1]} ===")
overall = 0.0
rows = {}
for k in order:
    if k not in orig_acts or k not in hf_acts:
        continue
    a, b = orig_acts[k], hf_acts[k]
    if a.shape != b.shape and a.transpose(-1, -2).shape == b.shape:
        a = a.transpose(-1, -2)
    if a.shape != b.shape:
        print(f"  {k:16s} SHAPE MISMATCH {tuple(a.shape)} vs {tuple(b.shape)}")
        rows[k] = {"shape_mismatch": [tuple(a.shape), tuple(b.shape)]}
        continue
    d = (a - b).abs().max().item()
    rel = d / (a.abs().max().item() + 1e-12)
    overall = max(overall, d)
    flag = "" if d < TOL else "  <-- EXCEEDS TOL"
    print(f"  {k:16s} max_abs_diff={d:.3e}  rel={rel:.2e}  shape={tuple(a.shape)}{flag}")
    rows[k] = {"max_abs_diff": d, "rel": rel, "shape": list(a.shape)}
verdict = "PASS" if overall < TOL else "FAIL"
print(f"OVERALL max_abs_diff = {overall:.3e}  ->  {verdict}  (tol {TOL})")

if out_json:
    with open(out_json, "w") as f:
        json.dump(
            {
                "model_card": model_card,
                "variant": variant,
                "overall_max_abs_diff": overall,
                "verdict": verdict,
                "tol": TOL,
                "stages": rows,
            },
            f,
            indent=2,
        )

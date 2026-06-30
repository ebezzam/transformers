"""ZS (zero-shot) encoder parity. The ZS model's full forward needs context examples (zero-shot syntax),
so we validate the audio path directly: run the original encoder_frontend+encoder (+ encoder_proj) on the
same input_values as the HF encoder, hooking each layer. The decoder is the standard Llama (validated
elsewhere); encoder_stacking=3 is a weightless reshape exercised by HF `get_audio_features`.

Usage:  zs_encoder_parity.py <model_card> <hf_path>
"""
import gc
import sys

import torch
from datasets import Audio, load_dataset

model_card, hf_path = sys.argv[1], sys.argv[2]
dev, dtype = torch.device("cuda"), torch.float32
TOL = 1e-3

from transformers import AutoProcessor, OmniASRForConditionalGeneration  # noqa: E402

ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation").cast_column(
    "audio", Audio(sampling_rate=16000)
)
audio = ds.sort("id")[0]["audio"]["array"]
proc = AutoProcessor.from_pretrained(hf_path)
input_values = proc(audio, sampling_rate=16000, return_tensors="pt")["input_values"].to(dtype)


def mk(store, name):
    def fn(_m, _i, out):
        t = out[0] if isinstance(out, (tuple, list)) else out
        if torch.is_tensor(t):
            store[name] = t.detach().float().cpu()
    return fn


# ---- HF encoder (hook layers) ----
hf_acts = {}
hf = OmniASRForConditionalGeneration.from_pretrained(hf_path, dtype=dtype).to(dev).eval()
n_layers = len(hf.encoder.encoder.layers)
for i, l in enumerate(hf.encoder.encoder.layers):
    l.register_forward_hook(mk(hf_acts, f"enc_layer_{i:02d}"))
hf.encoder.encoder.layer_norm.register_forward_hook(mk(hf_acts, "enc_final_norm"))
with torch.no_grad():
    hf.get_audio_features(input_values.to(dev))  # runs encoder (+ stacking + projector)
del hf
gc.collect()
torch.cuda.empty_cache()

# ---- ORIGINAL encoder (hook layers); drive it directly via the frontend+encoder ----
orig_acts = {}
from fairseq2.nn.batch_layout import BatchLayout  # noqa: E402

from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline  # noqa: E402

pipe = ASRInferencePipeline(model_card=model_card, device=dev, dtype=dtype)
om = pipe.model.eval()
for i, l in enumerate(om.encoder.layers):
    l.register_forward_hook(mk(orig_acts, f"enc_layer_{i:02d}"))
om.encoder.layer_norm.register_forward_hook(mk(orig_acts, "enc_final_norm"))
iv = input_values.to(dev)
with torch.no_grad():
    bl = BatchLayout(iv.shape, seq_lens=[iv.shape[1]], device=iv.device)
    seqs, enc_layout = om.encoder_frontend(iv, bl)
    om.encoder(seqs, enc_layout)
del pipe, om
gc.collect()
torch.cuda.empty_cache()

# ---- compare ----
print(f"=== ZS ENCODER PARITY {model_card} ({n_layers} layers) ===")
overall = 0.0
for k in [f"enc_layer_{i:02d}" for i in range(n_layers)] + ["enc_final_norm"]:
    if k in orig_acts and k in hf_acts:
        a, b = orig_acts[k], hf_acts[k]
        if a.shape != b.shape:
            print(f"  {k}: SHAPE {tuple(a.shape)} vs {tuple(b.shape)}")
            continue
        d = (a - b).abs().max().item()
        overall = max(overall, d)
        if k in (f"enc_layer_00", f"enc_layer_{n_layers-1:02d}", "enc_final_norm"):
            print(f"  {k}: max_abs_diff={d:.3e} shape={tuple(a.shape)}")
print(f"OVERALL encoder max_abs_diff = {overall:.3e} -> {'PASS' if overall < TOL else 'FAIL'} (tol {TOL})")

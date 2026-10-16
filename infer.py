"""
micro-chat inference for Raspberry Pi. NumPy + tokenizers only, O(1) state/token.

    python infer.py --prompt "Once upon a time" --n 100

Loads the pickle-free int8 bundle written by `main.py --action export`
(.npz, opened with allow_pickle=False, so a downloaded model cannot run code).
"""
import argparse
import json
import os
import time

import numpy as np
from tokenizers import Tokenizer

BUNDLE = os.environ.get("PI_MODEL", "./pi_model.npz")
TOK_PATH = os.environ.get("PI_TOK", "./tokenizer.json")

EPS_RMS = 1e-6
DT_MAX = 100.0


# ----------------------------------------------------------------
# Primitives
# ----------------------------------------------------------------
def silu(x):
    return x * (0.5 * (1.0 + np.tanh(0.5 * x)))  # x * sigmoid(x), overflow-free


def softplus(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def rms_norm(x, scale):
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + EPS_RMS) * scale


# ----------------------------------------------------------------
# Bundle
# ----------------------------------------------------------------
class Bundle:
    """Reads the .npz, dequantises int8 tensors to float32."""

    def __init__(self, path=BUNDLE):
        with np.load(path, allow_pickle=False) as z:
            self.meta = json.loads(bytes(z["meta"]).decode())
            if self.meta.get("format") != "micro-chat-int8":
                raise ValueError(f"{path}: not a micro-chat int8 bundle")
            self.params, self.awq = {}, {}
            for key in z.files:
                if key.startswith("q/") and key.endswith("/idx"):
                    name = key[2:-4]
                    self.params[name] = (z[key].astype(np.float32)
                                         * z[f"q/{name}/scale"].astype(np.float32))
                elif key.startswith("f/"):
                    self.params[key[2:]] = z[key].astype(np.float32)
                elif key.startswith("awq/"):
                    self.awq[key[4:]] = z[key].astype(np.float32)
        m = self.meta
        self.vocab_size, self.d_model, self.n_layers = m["vocab_size"], m["d_model"], m["n_layers"]
        self.d_state, self.d_conv = m["d_state"], m["d_conv"]
        self.d_inner = m["expand"] * m["d_model"]
        self.head_dim = m["head_dim"]
        self.n_heads = self.d_inner // self.head_dim

    def summary(self):
        n = sum(v.size for v in self.params.values())
        return n, sum(v.nbytes for v in self.params.values())


# ----------------------------------------------------------------
# One Mamba-2 layer (mirrors main.Mamba2Block step by step)
# ----------------------------------------------------------------
class Mamba2Layer:
    def __init__(self, b: Bundle, idx: int):
        self.b = b
        pre = f"Block_{idx}/Mamba2Block_0/"
        P = b.params
        self.W_in = P[pre + "QDense_0/kernel"]
        self.W_out = P[pre + "QDense_1/kernel"]
        self.rms_s = P[pre + "RMSNorm_0/scale"]
        self.conv_w = P[pre + "Conv_0/kernel"].squeeze(1)   # (d_conv, C)
        self.conv_b = P[pre + "Conv_0/bias"]
        self.dt_bias = P[pre + "dt_bias"]
        self.A = -np.exp(P[pre + "A_log"])
        self.D = P[pre + "D"]
        # AWQ: matmul used W*s, so activations are multiplied by 1/s
        s_in = b.awq.get(pre + "QDense_0/kernel")
        s_out = b.awq.get(pre + "QDense_1/kernel")
        self.inv_in = None if s_in is None else 1.0 / s_in
        self.inv_out = None if s_out is None else 1.0 / s_out
        self.reset()

    def reset(self):
        b = self.b
        self.conv_buf = np.zeros((b.d_conv - 1, b.d_inner + 2 * b.d_state), np.float32)
        self.S = np.zeros((b.n_heads, b.head_dim, b.d_state), np.float32)

    def __call__(self, x):
        b = self.b
        di, N, H, Pd = b.d_inner, b.d_state, b.n_heads, b.head_dim

        h = rms_norm(x, self.rms_s)
        if self.inv_in is not None:
            h = h * self.inv_in
        proj = h @ self.W_in
        z = proj[:di]
        xBC = proj[di:2 * di + 2 * N]
        dt = proj[2 * di + 2 * N:]

        full = np.concatenate([self.conv_buf, xBC[None, :]], axis=0)
        self.conv_buf = full[1:]
        xBC = silu((full * self.conv_w).sum(axis=0) + self.conv_b)

        xi = xBC[:di].reshape(H, Pd)
        Bt, Ct = xBC[di:di + N], xBC[di + N:]
        dt = np.clip(softplus(dt + self.dt_bias), 0.0, DT_MAX)

        self.S *= np.exp(self.A * dt)[:, None, None]
        self.S += (dt[:, None] * xi)[:, :, None] * Bt[None, None, :]
        y = (self.S @ Ct + self.D[:, None] * xi).reshape(di) * silu(z)
        if self.inv_out is not None:
            y = y * self.inv_out
        return x + y @ self.W_out


# ----------------------------------------------------------------
# Full model
# ----------------------------------------------------------------
class PiModel:
    def __init__(self, bundle_path=BUNDLE, tok_path=TOK_PATH):
        self.bundle = Bundle(bundle_path)
        self.layers = [Mamba2Layer(self.bundle, i) for i in range(self.bundle.n_layers)]
        P = self.bundle.params
        self.embed = P["Embed_0/embedding"]
        self.final_s = P["RMSNorm_0/scale"]
        # tied: head is the embedding table; untied: separate (d_model, vocab) kernel
        self.head = P["lm_head/kernel"] if "lm_head/kernel" in P else self.embed.T
        self.tok = Tokenizer.from_file(tok_path)
        eos = self.tok.token_to_id("</s>")
        self.eos_id = 2 if eos is None else eos

    def reset(self):
        for layer in self.layers:
            layer.reset()

    def step(self, token_id: int, logits: bool = True):
        x = self.embed[token_id]
        for layer in self.layers:
            x = layer(x)
        if not logits:          # prompt tokens only need to update the state
            return None
        return rms_norm(x, self.final_s) @ self.head

    def generate(self, prompt: str, n_new=100, temp=0.8, top_k=40, seed=None):
        rng = np.random.default_rng(seed)
        self.reset()
        # Training text is "story </s> story </s> ...": start after a </s>.
        ids = [self.eos_id] + self.tok.encode(prompt).ids

        t0 = time.time()
        for t in ids[:-1]:
            self.step(t, logits=False)
        t_prompt = time.time() - t0

        t0, n_gen = time.time(), 0
        for _ in range(n_new):
            lg = self.step(ids[-1]) / max(temp, 1e-6)
            if 0 < top_k < lg.size:
                lg = np.where(lg < np.partition(lg, -top_k)[-top_k], -np.inf, lg)
            p = np.exp(lg - lg.max())
            ids.append(int(rng.choice(p.size, p=p / p.sum())))
            n_gen += 1
            if ids[-1] == self.eos_id:
                break
        return ids, n_gen, t_prompt, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="Once upon a time")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--model", default=BUNDLE)
    ap.add_argument("--tok", default=TOK_PATH)
    a = ap.parse_args()

    model = PiModel(a.model, a.tok)
    n, nbytes = model.bundle.summary()
    print(f"[model] {n / 1e6:.1f}M params ({nbytes / 1e6:.1f} MB fp32) "
          f"awq={len(model.bundle.awq)} kernels")
    ids, n_gen, t_prompt, t_gen = model.generate(
        a.prompt, a.n, a.temp, a.top_k, a.seed)
    print(f"[prompt] {len(ids) - n_gen} tok in {t_prompt:.2f}s")
    if n_gen:
        print(f"[gen] {n_gen} tok in {t_gen:.2f}s ({n_gen / max(t_gen, 1e-6):.1f} tok/s)")
    print("\n--- output ---")
    print(model.tok.decode(ids, skip_special_tokens=True))


if __name__ == "__main__":
    main()

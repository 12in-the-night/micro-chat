"""
Pi inference for micro-chat Mamba-2. NumPy only. RAM ~100 MB.

Run: python infer.py --prompt "The sun was" --n 100
"""
import os, time, pickle, argparse
import numpy as np
from tokenizers import Tokenizer


BUNDLE = os.environ.get("PI_MODEL", "./pi_model.pkl")
TOK_PATH = os.environ.get("PI_TOK", "./tokenizer.json")

EPS_LN = 1e-6          # must match Flax nn.LayerNorm default
DT_MIN, DT_MAX = 0.0, 100.0


# ----------------------------------------------------------------
# Numerically-stable primitives
# ----------------------------------------------------------------
def silu(x):
    # Avoid overflow on large negative x.
    return x * np.where(x >= 0,
                        1.0 / (1.0 + np.exp(-x)),
                        np.exp(x) / (1.0 + np.exp(x)))


def softplus(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)


def layer_norm(x, scale, bias, eps=EPS_LN):
    m = x.mean(-1, keepdims=True)
    v = x.var(-1, keepdims=True)
    return (x - m) / np.sqrt(v + eps) * scale + bias


def dequant(entry):
    return (entry["idx"].astype(np.float32) * entry["scale"]).astype(np.float32)


# ----------------------------------------------------------------
# Bundle
# ----------------------------------------------------------------
class Bundle:
    """Loads pi_model.pkl, dequantizes int8 tensors, exposes metadata."""

    def __init__(self, path=BUNDLE):
        with open(path, "rb") as f:
            self.raw = pickle.load(f)
        self.vocab_size = self.raw["vocab_size"]
        self.d_model = self.raw["d_model"]
        self.n_layers = self.raw["n_layers"]
        self.max_len = self.raw.get("max_len", 512)
        self.tie = self.raw.get("tie", True)
        self.mamba = self.raw["mamba"]
        self.params = self._load()

    def _load(self):
        p = {k: dequant(v) for k, v in self.raw["quant"].items()}
        p.update(self.raw["fp32"])
        return p

    @property
    def d_inner(self):
        return self.mamba["expand"] * self.d_model

    @property
    def head_dim(self):
        return self.mamba["head_dim"]

    @property
    def n_heads(self):
        return self.d_inner // self.head_dim

    @property
    def d_state(self):
        return self.mamba["d_state"]

    @property
    def d_conv(self):
        return self.mamba["d_conv"]

    def summary(self):
        n = sum(v.size for v in self.params.values())
        nbytes = sum(v.nbytes for v in self.params.values())
        return n, nbytes


# ----------------------------------------------------------------
# One Mamba-2 layer
# ----------------------------------------------------------------
class Mamba2Layer:
    """A single Mamba-2 block with persistent recurrent state."""

    def __init__(self, bundle: Bundle, idx: int):
        self.b = bundle
        self.idx = idx
        pref = f"Block_{idx}/Mamba2Block_0/"
        P = bundle.params
        self.W_in = P[pref + "QDense_0/kernel"]
        self.W_out = P[pref + "QDense_1/kernel"]
        self.ln_s = P[pref + "LayerNorm_0/scale"]
        self.ln_b = P[pref + "LayerNorm_0/bias"]
        self.conv_w = P[pref + "Conv_0/kernel"].squeeze(1)
        self.conv_b = P[pref + "Conv_0/bias"]
        self.dt_bias = P[pref + "dt_bias"]
        self.A = -np.exp(P[pref + "A_log"])
        self.D = P[pref + "D"]
        self.reset()

    def reset(self):
        d_inner = self.b.d_inner
        N = self.b.d_state
        self.conv_buf = np.zeros((self.b.d_conv - 1, d_inner + 2 * N), np.float32)
        self.S = np.zeros((self.b.n_heads, self.b.head_dim, N), np.float32)

    def __call__(self, x):
        b = self.b
        d_inner = b.d_inner
        H = b.n_heads
        P = b.head_dim
        N = b.d_state

        h = layer_norm(x, self.ln_s, self.ln_b)
        proj = h @ self.W_in
        z = proj[:d_inner]
        xBC = proj[d_inner:d_inner + d_inner + 2 * N]
        dt = proj[d_inner + d_inner + 2 * N:]

        full = np.concatenate([self.conv_buf, xBC[None, :]], axis=0)
        conv_out = (full * self.conv_w).sum(axis=0) + self.conv_b
        self.conv_buf = full[1:]
        xBC = silu(conv_out)

        xi = xBC[:d_inner].reshape(H, P)
        B_t = xBC[d_inner:d_inner + N]
        C_t = xBC[d_inner + N:]
        dt_ = np.clip(softplus(dt + self.dt_bias), DT_MIN, DT_MAX)

        dec = np.exp(self.A * dt_)
        self.S = (dec[:, None, None] * self.S
                  + (dt_[:, None, None] * xi[..., None]) * B_t[None, None, :])
        y = np.einsum("hpn,n->hp", self.S, C_t) + self.D[:, None] * xi

        y = y.reshape(d_inner) * silu(z)
        return x + y @ self.W_out


# ----------------------------------------------------------------
# Full model
# ----------------------------------------------------------------
class PiModel:
    """micro-chat with O(1) per-token inference state."""

    def __init__(self, bundle_path=BUNDLE, tok_path=TOK_PATH):
        self.bundle = Bundle(bundle_path)
        self.layers = [Mamba2Layer(self.bundle, i)
                       for i in range(self.bundle.n_layers)]
        P = self.bundle.params
        self.embed = P["Embed_0/embedding"]
        self.final_ln_s = P["LayerNorm_0/scale"]
        self.final_ln_b = P["LayerNorm_0/bias"]
        self.tok = Tokenizer.from_file(tok_path)
        eos = self.tok.token_to_id("</s>")
        self.eos_id = 2 if eos is None else eos

    def reset(self):
        for layer in self.layers:
            layer.reset()

    def step(self, token_id: int):
        x = self.embed[token_id]
        for layer in self.layers:
            x = layer(x)
        x = layer_norm(x, self.final_ln_s, self.final_ln_b)
        return x @ self.embed.T

    def generate(self, prompt: str, n_new=100, temp=0.8, top_k=40):
        self.reset()
        ids = self.tok.encode(prompt).ids
        if not ids:
            bos = self.tok.token_to_id("<s>")
            ids = [1 if bos is None else bos]

        t0 = time.time()
        for i in range(len(ids) - 1):
            self.step(ids[i])
        t_prompt = time.time() - t0

        t0 = time.time()
        n_gen = 0
        for _ in range(n_new):
            logits = self.step(ids[-1]) / temp
            if top_k:
                kth = np.partition(logits, -top_k)[-top_k]
                logits = np.where(logits < kth, -1e9, logits)
            p = np.exp(logits - logits.max())
            p /= p.sum()
            ids.append(int(np.random.choice(len(p), p=p)))
            n_gen += 1
            if ids[-1] == self.eos_id:
                break
        t_gen = time.time() - t0
        return ids, n_gen, t_prompt, t_gen


# ----------------------------------------------------------------
# Entry
# ----------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="The sun was")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--model", default=BUNDLE)
    ap.add_argument("--tok", default=TOK_PATH)
    args = ap.parse_args()

    model = PiModel(args.model, args.tok)
    n, nbytes = model.bundle.summary()
    print(f"[model] {n/1e6:.1f}M params  ({nbytes/1e6:.1f} MB fp32)")

    ids, n_gen, t_prompt, t_gen = model.generate(
        args.prompt, n_new=args.n, temp=args.temp, top_k=args.top_k,
    )
    print(f"[prompt] {len(ids) - n_gen} tok in {t_prompt:.2f}s")
    if n_gen:
        print(f"[gen] {n_gen} tok in {t_gen:.2f}s ({n_gen/max(t_gen,1e-6):.1f} tok/s)")
    print("\n--- output ---")
    print(model.tok.decode(ids))


if __name__ == "__main__":
    main()

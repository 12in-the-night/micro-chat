"""
micro-chat: a small dense Mamba-2 causal LM trained on TinyStories.

QAT int8 (per-output-channel) -> layer pruning -> AWQ -> int8 export for a
Raspberry Pi (NumPy-only inference, see infer.py).

Single- and multi-GPU data parallelism through jit + sharding. fp16 (with
dynamic loss scaling) on Turing/Volta/Pascal, bf16 on Ampere+, fp32 on CPU.

Actions:
  python main.py --action tokenizer      train the BPE tokenizer
  python main.py                         train the model
  python main.py --action sample --prompt "Once upon a time"
  python main.py --action eval           val NLL: QAT vs simulated int8 export
  python main.py --action prune_layers
  python main.py --action awq
  python main.py --action export
  python main.py --action full           prune -> AWQ -> pick best -> export
"""
import argparse
import functools
import glob
import itertools
import json
import math
import os
import pickle
import queue
import re
import sys
import threading
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, Iterator, Optional

import numpy as np
import yaml

# Must be set before JAX initialises its backend. Preallocation off so two
# processes (or a notebook + script) can share one GPU.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from flax.traverse_util import flatten_dict, unflatten_dict
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

CKPT_VERSION = 2
QMAX = 127
Q_EPS = 1e-8
DTYPES = {"fp32": jnp.float32, "fp16": jnp.float16, "bf16": jnp.bfloat16}


# ================================================================
# CONFIG
# ================================================================
class Config:
    """Thin wrapper around the YAML dict with dotted-path lookup."""

    def __init__(self, raw: Optional[dict] = None):
        self.raw = raw or {}

    @classmethod
    def load(cls, path: str) -> "Config":
        with open(path) as f:
            return cls(yaml.safe_load(f))

    def get(self, path: str, default=None):
        v: Any = self.raw
        for k in path.split("."):
            if not isinstance(v, dict) or k not in v:
                return default
            v = v[k]
        return default if v is None else v


@dataclass(frozen=True)
class ModelCfg:
    """Everything that defines the network. Stored inside every checkpoint so
    post-processing never depends on whatever config.yaml says today."""

    vocab_size: int
    d_model: int = 256
    n_layers: int = 24
    d_state: int = 32
    d_conv: int = 4
    chunk: int = 128
    expand: int = 2
    head_dim: int = 64
    tie_embeddings: bool = True

    def __post_init__(self):
        if (self.expand * self.d_model) % self.head_dim:
            raise ValueError("expand * d_model must be divisible by head_dim")

    @property
    def d_inner(self) -> int:
        return self.expand * self.d_model

    @property
    def n_heads(self) -> int:
        return self.d_inner // self.head_dim

    @classmethod
    def from_config(cls, cfg: Config, vocab_size: int) -> "ModelCfg":
        return cls(
            tie_embeddings=bool(cfg.get("model.tie_embeddings", True)),
            vocab_size=int(vocab_size),
            d_model=int(cfg.get("model.d_model", 256)),
            n_layers=int(cfg.get("model.n_layers", 24)),
            d_state=int(cfg.get("mamba2.d_state", 32)),
            d_conv=int(cfg.get("mamba2.d_conv", 4)),
            chunk=int(cfg.get("mamba2.chunk", 128)),
            expand=int(cfg.get("mamba2.expand", 2)),
            head_dim=int(cfg.get("mamba2.head_dim", 64)),
        )


def model_cfg_from_ckpt(ck: dict) -> ModelCfg:
    if "model" in ck:
        return ModelCfg(**ck["model"])
    # v1 checkpoints: rebuild from the stored yaml + the embedding shape.
    vocab = ck["params"]["Embed_0"]["embedding"].shape[0]
    return ModelCfg.from_config(Config(ck["config"]), vocab)


def pick_precision(requested: str) -> str:
    requested = str(requested).lower()
    if requested in DTYPES:
        return requested
    dev = jax.devices()[0]
    if dev.platform == "tpu":
        return "bf16"
    if dev.platform not in ("gpu", "cuda", "rocm"):
        return "fp32"
    kind = str(dev.device_kind).lower()
    if any(s in kind for s in
           ("a100", "h100", "a10", "l4", "h200", "rtx 40", "rtx 50")):
        return "bf16"
    return "fp16"


# ================================================================
# INT8 QUANTISATION
# ================================================================
def _q_scale(absmax, xp):
    return xp.maximum(absmax, Q_EPS) / QMAX


def fake_quant_int8(x, axis):
    """Straight-through int8 fake quantisation, scales along `axis` reduced."""
    x32 = x.astype(jnp.float32)
    s = _q_scale(jnp.max(jnp.abs(x32), axis=axis, keepdims=True), jnp)
    xq = jnp.clip(jnp.round(x32 / s), -QMAX, QMAX) * s
    return x32 + jax.lax.stop_gradient(xq - x32)


def quant_i8_np(w, axis):
    """NumPy twin of fake_quant_int8 -> (int8 values, float32 scales)."""
    w32 = np.asarray(w, np.float32)
    s = _q_scale(np.max(np.abs(w32), axis=axis, keepdims=True), np)
    idx = np.clip(np.round(w32 / s), -QMAX, QMAX).astype(np.int8)
    return idx, s.astype(np.float32)


# ================================================================
# TOKENIZER
# ================================================================
def make_text_source(cfg: Config, split: str, seed: int = 0,
                     shuffle: bool = True) -> Callable[[int], Iterator[str]]:
    """Returns texts(epoch) -> iterator of documents (streaming from the Hub)."""
    name = cfg.get("data.name", "roneneldan/TinyStories")
    conf = cfg.get("data.config")
    min_chars = int(cfg.get("data.min_chars", 32))
    buf = int(cfg.get("data.shuffle_buffer", 50000))

    def texts(epoch: int = 0) -> Iterator[str]:
        from datasets import load_dataset
        ds = load_dataset(name, conf, split=split, streaming=True)
        if shuffle and buf:
            ds = ds.shuffle(seed=seed + epoch, buffer_size=buf)
        for ex in ds:
            t = ex.get("text") or ""
            if len(t) >= min_chars:
                yield t

    return texts


class TokenizerManager:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.path = cfg.get("tokenizer.path", "./tokenizer.json")
        self._tok = None

    def load(self):
        if self._tok is None:
            if not os.path.exists(self.path):
                raise FileNotFoundError(self.path)
            from tokenizers import Tokenizer
            self._tok = Tokenizer.from_file(self.path)
        return self._tok

    def get_or_train(self):
        try:
            return self.load()
        except FileNotFoundError:
            return self.train()

    def train(self):
        from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
        c = self.cfg
        vocab = int(c.get("tokenizer.vocab_size", 8192))
        target = int(c.get("tokenizer.train_corpus_mb", 50)) * 1024 * 1024
        specials = c.get("tokenizer.special_tokens", ["<pad>", "<s>", "</s>", "<unk>"])
        print(f"[tok] training {vocab}-BPE on ~{target // 2**20} MB")
        texts = make_text_source(c, c.get("data.split_train", "train"),
                                 shuffle=False)(0)

        def corpus():
            n = 0
            for t in texts:
                yield t
                n += len(t)
                if n >= target:
                    return

        tok = Tokenizer(models.BPE(unk_token="<unk>"))
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tok.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=vocab, special_tokens=specials,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=True,
        )
        tok.train_from_iterator(corpus(), trainer=trainer)
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        tok.save(self.path)
        self._tok = tok
        print(f"[tok] saved {self.path}  vocab={tok.get_vocab_size()}")
        return tok

    def encode_batch(self, texts):
        return [e.ids for e in self.load().encode_batch(texts)]

    @property
    def eos_id(self) -> int:
        i = self.load().token_to_id("</s>")
        return 2 if i is None else i

    @property
    def vocab_size(self) -> int:
        return self.load().get_vocab_size()


# ================================================================
# MODEL
# ================================================================
class QDense(nn.Module):
    """Bias-free dense layer with int8 fake-quantised weights (per out channel)."""
    features: int
    dtype: Any = jnp.float32
    quantize: bool = True
    capture: bool = False

    @nn.compact
    def __call__(self, x):
        if self.capture:  # per-input-channel mean |activation| for AWQ
            self.sow("intermediates", "act_mag",
                     jnp.mean(jnp.abs(x.astype(jnp.float32)),
                              axis=tuple(range(x.ndim - 1))))
        k = self.param("kernel", nn.initializers.lecun_normal(),
                       (x.shape[-1], self.features))
        if self.quantize:
            k = fake_quant_int8(k, axis=0)
        y = jnp.matmul(x.astype(self.dtype), k.astype(self.dtype),
                       preferred_element_type=jnp.float32)
        return y.astype(self.dtype)


def segsum(x):
    """Stable segment sum: out[..., i, j] = sum_{k=j+1..i} x[..., k] (i >= j)."""
    T = x.shape[-1]
    cs = jnp.cumsum(x, axis=-1)
    ss = cs[..., :, None] - cs[..., None, :]
    return jnp.where(jnp.tril(jnp.ones((T, T), bool)), ss, -jnp.inf)


def ssd(x, dt, A, B, C, D, chunk):
    """
    Chunked state-space-duality scan (Mamba-2), all in float32.

    x (b,L,H,P)  dt (b,L,H) post-softplus  A (H,) negative
    B, C (b,L,N) shared by all heads       D (H,)
    Per head:  S_t = exp(A dt_t) S_{t-1} + dt_t x_t (x) B_t ;  y_t = S_t C_t + D x_t
    """
    b, L, H, Pd = x.shape
    N = B.shape[-1]
    pad = (-L) % chunk
    if pad:  # dt = 0 on padding -> decay 1, no contribution
        x = jnp.pad(x, ((0, 0), (0, pad), (0, 0), (0, 0)))
        dt = jnp.pad(dt, ((0, 0), (0, pad), (0, 0)))
        B = jnp.pad(B, ((0, 0), (0, pad), (0, 0)))
        C = jnp.pad(C, ((0, 0), (0, pad), (0, 0)))
    nC = (L + pad) // chunk
    xc = x.reshape(b, nC, chunk, H, Pd)
    dtc = dt.reshape(b, nC, chunk, H)
    Bc = B.reshape(b, nC, chunk, N)
    Cc = C.reshape(b, nC, chunk, N)

    At = jnp.transpose(A * dtc, (0, 1, 3, 2))            # (b,c,H,l)
    Acs = jnp.cumsum(At, axis=-1)
    xd = xc * dtc[..., None]

    # intra-chunk (diagonal blocks); C.B is head-independent
    CB = jnp.einsum("bcln,bcsn->bcls", Cc, Bc)
    y_diag = jnp.einsum("bchls,bcshp->bclhp",
                        CB[:, :, None] * jnp.exp(segsum(At)), xd)

    # state contributed by each chunk, then carried across chunks
    decay_to_end = jnp.exp(Acs[..., -1:] - Acs)           # (b,c,H,l)
    chunk_state = jnp.einsum("bcln,bchl,bclhp->bchpn", Bc, decay_to_end, xd)
    chunk_decay = jnp.exp(Acs[..., -1])                   # (b,c,H)

    def carry(st, inp):
        cs, cd = inp
        return cd[..., None, None] * st + cs, st          # emit state at chunk start

    _, st_in = jax.lax.scan(
        carry, jnp.zeros((b, H, Pd, N), jnp.float32),
        (jnp.swapaxes(chunk_state, 0, 1), jnp.swapaxes(chunk_decay, 0, 1)),
    )
    st_in = jnp.swapaxes(st_in, 0, 1)                     # (b,c,H,P,N)
    y_off = jnp.einsum("bcln,bchpn->bclhp", Cc, st_in) \
        * jnp.exp(jnp.transpose(Acs, (0, 1, 3, 2)))[..., None]

    y = y_diag + y_off + D[None, None, None, :, None] * xc
    return y.reshape(b, nC * chunk, H, Pd)[:, :L]


class Mamba2Block(nn.Module):
    cfg: ModelCfg
    dtype: Any = jnp.float32
    quantize: bool = True
    capture: bool = False

    @nn.compact
    def __call__(self, x):
        c = self.cfg
        B_, T, C_ = x.shape
        d_inner, H, Pd, N = c.d_inner, c.n_heads, c.head_dim, c.d_state

        h = nn.RMSNorm(epsilon=1e-6, dtype=jnp.float32)(x.astype(jnp.float32))
        proj = QDense(2 * d_inner + 2 * N + H, self.dtype, self.quantize,
                      self.capture)(h.astype(self.dtype))
        z, xBC, dt = jnp.split(proj, [d_inner, 2 * d_inner + 2 * N], axis=-1)

        xBC = jnp.pad(xBC, [(0, 0), (c.d_conv - 1, 0), (0, 0)])
        xBC = nn.Conv(features=xBC.shape[-1], kernel_size=(c.d_conv,),
                      padding="VALID", feature_group_count=xBC.shape[-1],
                      use_bias=True, dtype=self.dtype, param_dtype=jnp.float32)(xBC)
        xBC = nn.silu(xBC).astype(jnp.float32)
        xi, Bt, Ct = jnp.split(xBC, [d_inner, d_inner + N], axis=-1)

        A_log = self.param(
            "A_log",
            lambda k, s: jnp.log(jax.random.uniform(k, s, minval=1.0, maxval=16.0)),
            (H,))
        D = self.param("D", nn.initializers.ones, (H,))

        def dt_bias_init(k, s):  # inverse-softplus of dt ~ logU[1e-3, 1e-1]
            u = jax.random.uniform(k, s)
            v = jnp.exp(u * (math.log(0.1) - math.log(1e-3)) + math.log(1e-3))
            return v + jnp.log(-jnp.expm1(-v))

        dt_bias = self.param("dt_bias", dt_bias_init, (H,))
        dt = jnp.clip(jax.nn.softplus(dt.astype(jnp.float32) + dt_bias), 0.0, 100.0)

        y = ssd(xi.reshape(B_, T, H, Pd), dt, -jnp.exp(A_log), Bt, Ct, D, c.chunk)
        y = y.reshape(B_, T, d_inner) * nn.silu(z.astype(jnp.float32))
        return QDense(C_, self.dtype, self.quantize, self.capture)(y.astype(self.dtype))


class Block(nn.Module):
    """Residual Mamba-2 mixer. `remat` recomputes activations in backward.
    (Disabled during AWQ capture: `sow` inside remat is fragile.)"""
    cfg: ModelCfg
    dtype: Any = jnp.float32
    quantize: bool = True
    capture: bool = False
    remat: bool = False

    @nn.compact
    def __call__(self, x):
        cls = nn.remat(Mamba2Block) if (self.remat and not self.capture) else Mamba2Block
        out = x + cls(self.cfg, self.dtype, self.quantize, self.capture,
                      name="Mamba2Block_0")(x)
        if self.capture:  # ShortGPT-style block influence = 1 - cos(in, out)
            cos = jnp.sum(x * out, -1) / (
                jnp.linalg.norm(x, axis=-1) * jnp.linalg.norm(out, axis=-1) + 1e-6)
            self.sow("intermediates", "cos_sim", jnp.mean(cos))
        return out


class LM(nn.Module):
    cfg: ModelCfg
    dtype: Any = jnp.float32
    quantize: bool = True
    capture: bool = False
    remat: bool = False

    @nn.compact
    def __call__(self, tokens):
        c = self.cfg
        emb = nn.Embed(c.vocab_size, c.d_model, param_dtype=jnp.float32,
                       embedding_init=nn.initializers.normal(0.02))
        table = emb.embedding
        if self.quantize:  # same table is used for lookup and the tied head
            table = fake_quant_int8(table, axis=1)
        x = jnp.take(table, tokens, axis=0).astype(jnp.float32)
        for i in range(c.n_layers):
            x = Block(c, self.dtype, self.quantize, self.capture, self.remat,
                      name=f"Block_{i}")(x)
        x = nn.RMSNorm(epsilon=1e-6, dtype=jnp.float32)(x).astype(self.dtype)
        if c.tie_embeddings:
            return jnp.matmul(x, table.astype(self.dtype).T,
                              preferred_element_type=jnp.float32)
        head = QDense(c.vocab_size, self.dtype, self.quantize, name="lm_head")
        return head(x).astype(jnp.float32)


def lm_loss(apply_fn, params, batch):
    logits = apply_fn({"params": params}, batch[:, :-1])
    return optax.softmax_cross_entropy_with_integer_labels(logits, batch[:, 1:]).mean()


# ================================================================
# DATA
# ================================================================
class PackedStream:
    """Tokenises documents, joins them with </s>, yields (batch, seq+1) int32."""

    def __init__(self, tok: TokenizerManager, texts: Callable[[int], Iterator[str]],
                 batch: int, seq: int, repeat: bool = False, doc_batch: int = 512):
        self.tok, self.texts = tok, texts
        self.batch, self.seq, self.repeat, self.doc_batch = batch, seq, repeat, doc_batch

    def __iter__(self):
        eos = np.asarray([self.tok.eos_id], np.int32)
        need = self.batch * (self.seq + 1)
        buf = np.empty(0, np.int32)
        for epoch in itertools.count():
            produced = False
            it = self.texts(epoch)
            while True:
                docs = list(itertools.islice(it, self.doc_batch))
                if not docs:
                    break
                parts = [buf]
                for ids in self.tok.encode_batch(docs):
                    parts.append(np.asarray(ids, np.int32))
                    parts.append(eos)
                buf = np.concatenate(parts)
                while buf.size >= need:
                    produced = True
                    yield buf[:need].reshape(self.batch, self.seq + 1).copy()
                    buf = buf[need:]
            if not self.repeat:
                return
            if not produced:
                raise RuntimeError("dataset yielded no full batch in an epoch")


def grouped(it, n: int):
    """Yield np.stack of n consecutive items (leading axis = micro-batch)."""
    it = iter(it)
    while True:
        g = list(itertools.islice(it, n))
        if len(g) < n:
            return
        yield np.stack(g)


class Prefetcher:
    """Runs an iterator in a background thread, re-raising its exceptions."""
    _END = object()

    def __init__(self, it, depth: int = 4):
        self._q: "queue.Queue" = queue.Queue(maxsize=depth)
        threading.Thread(target=self._run, args=(it,), daemon=True).start()

    def _run(self, it):
        try:
            for x in it:
                self._q.put(x)
            self._q.put(self._END)
        except BaseException as e:  # noqa: BLE001 - forwarded to the consumer
            self._q.put(e)

    def __iter__(self):
        while True:
            x = self._q.get()
            if x is self._END:
                return
            if isinstance(x, BaseException):
                raise x
            yield x


def fixed_tokens(tok, cfg, split, n_seq, seq, seed=0, shuffle=False) -> np.ndarray:
    """First n_seq packed sequences of a split: (n_seq, seq+1)."""
    stream = PackedStream(tok, make_text_source(cfg, split, seed, shuffle), n_seq, seq)
    try:
        return next(iter(stream))
    except StopIteration:
        raise RuntimeError(f"split '{split}' is too small for {n_seq} x {seq} tokens")


# ================================================================
# CHECKPOINTS
# ================================================================
def _atomic_pickle(payload, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(payload, f, protocol=4)
    os.replace(tmp, path)
    print(f"[saved] {path} ({os.path.getsize(path) / 1e6:.1f} MB)")


def load_ckpt(path):
    # NOTE: pickle executes code on load. Training checkpoints are local and
    # trusted; the *distributed* artifact is the pickle-free int8 .npz.
    with open(path, "rb") as f:
        return pickle.load(f)


def save_train_ckpt(state, step, path, mcfg: ModelCfg, cfg: Config, precision):
    payload = {
        "version": CKPT_VERSION, "step": int(step), "precision": precision,
        "config": cfg.raw, "model": asdict(mcfg),
        "params": jax.tree_util.tree_map(np.asarray, jax.device_get(state.params)),
        "opt_leaves": [np.asarray(x) for x in
                       jax.tree_util.tree_leaves(jax.device_get(state.opt_state))],
    }
    _atomic_pickle(payload, path)


# ================================================================
# OPTIMISER / TRAINER
# ================================================================
def _wd_mask(params):
    def keep(path, _):
        keys = [str(getattr(k, "key", k)) for k in path]
        if any("Norm" in k for k in keys):
            return False
        return keys[-1] not in ("bias", "A_log", "D", "dt_bias", "embedding")
    return jax.tree_util.tree_map_with_path(keep, params)


def build_optimizer(cfg: Config, params):
    """clip -> (Adafactor-style) scaling -> decoupled weight decay -> -lr.

    Built from optax primitives rather than `optax.adafactor`, because the
    latter already flips the sign of its update, and its weight decay is not
    scaled by the learning rate.
    """
    steps = int(cfg.get("training.steps", 20000))
    lr = float(cfg.get("training.lr", 6e-4))
    sched = optax.warmup_cosine_decay_schedule(
        init_value=0.0, peak_value=lr,
        warmup_steps=int(cfg.get("training.warmup", 300)), decay_steps=steps,
        end_value=lr * float(cfg.get("training.min_lr_frac", 0.1)))

    af = cfg.get("optim.adafactor", {}) or {}
    if af.get("enabled", True):
        core = [optax.scale_by_factored_rms(
            factored=True, decay_rate=float(af.get("decay_rate", 0.8)),
            step_offset=int(af.get("decay_offset", 0)),
            min_dim_size_to_factor=int(af.get("min_dim_size_to_factor", 128)))]
        if af.get("clipping_threshold", 1.0) is not None:
            core.append(optax.clip_by_block_rms(float(af.get("clipping_threshold", 1.0))))
        if af.get("multiply_by_parameter_scale", False):
            core.append(optax.scale_by_param_block_rms())
        if af.get("momentum", 0.9) is not None:
            core.append(optax.ema(float(af.get("momentum", 0.9)), debias=False,
                                  accumulator_dtype=jnp.float32))
    else:
        core = [optax.scale_by_adam(b1=0.9, b2=0.95)]

    return optax.chain(
        optax.clip_by_global_norm(float(cfg.get("optim.clip_norm", 1.0))),
        *core,
        optax.add_decayed_weights(float(cfg.get("training.weight_decay", 0.1)),
                                  _wd_mask(params)),
        optax.scale_by_learning_rate(sched),  # flip_sign=True -> descent
    )


class TrainState(train_state.TrainState):
    loss_scale: jax.Array
    good_steps: jax.Array
    skipped: jax.Array


def make_text_source(cfg: Config, split: str, seed: int = 0,
                     shuffle: bool = True) -> Callable[[int], Iterator[str]]:
    """texts(epoch) -> iterator of documents, weighted-interleaved."""
    sources = cfg.get("data.sources") or [
        {"name": cfg.get("data.name", "roneneldan/TinyStories"),
         "config": cfg.get("data.config"),
         "split": split,
         "weight": 1.0},
    ]
    min_chars = int(cfg.get("data.min_chars", 32))
    buf = int(cfg.get("data.shuffle_buffer", 50000))
    total_w = sum(float(s.get("weight", 1.0)) for s in sources)
    cum, acc = [], 0.0
    for s in sources:
        acc += float(s.get("weight", 1.0)) / total_w
        cum.append(acc)

    def texts(epoch: int = 0) -> Iterator[str]:
        import random
        from datasets import load_dataset
        rng = random.Random(seed + epoch)
        iters = {}
        for s in sources:
            ds = load_dataset(s["name"], s.get("config"),
                              split=s.get("split", split), streaming=True)
            if shuffle and buf:
                ds = ds.shuffle(seed=seed + epoch, buffer_size=buf)
            iters[s["name"]] = iter(ds)
        dead = set()
        while len(dead) < len(sources):
            r = rng.random()
            pick = next(i for i, c in enumerate(cum) if r < c)
            s = sources[pick]
            if s["name"] in dead:
                continue
            try:
                ex = next(iters[s["name"]])
            except StopIteration:
                dead.add(s["name"])
                continue
            t = ex.get("text") or ""
            if len(t) >= min_chars:
                yield t

    return texts
    class TrainState(train_state.TrainState):
        loss_scale: jax.Array
        good_steps: jax.Array
        skipped: jax.Array


    def make_train_step(accum: int, dynamic_scale: bool, state_sh, batch_sh):
        """One optimiser step over `accum` micro-batches, fully inside one jit."""
        def step(state: TrainState, batches):
            scale = state.loss_scale

            def micro(carry, batch):
                g_acc, l_acc = carry
                loss, g = jax.value_and_grad(
                    lambda p: lm_loss(state.apply_fn, p, batch) * scale)(state.params)
                return (jax.tree_util.tree_map(jnp.add, g_acc, g), l_acc + loss), None

            zeros = jax.tree_util.tree_map(jnp.zeros_like, state.params)
            (g, l), _ = jax.lax.scan(micro, (zeros, jnp.zeros((), jnp.float32)), batches)
            inv = 1.0 / (scale * accum)
            grads = jax.tree_util.tree_map(lambda x: x * inv, g)
            gnorm = jnp.sqrt(sum(jnp.sum(jnp.square(x))
                                 for x in jax.tree_util.tree_leaves(grads)))
            finite = jnp.isfinite(gnorm)

            new = state.apply_gradients(grads=grads)
            pick = lambda a, b: jnp.where(finite, a, b)
            good = jnp.where(finite, state.good_steps + 1, 0)
            if dynamic_scale:
                grow = good >= 2000
                new_scale = jnp.where(
                    finite, jnp.where(grow, jnp.minimum(scale * 2, 2.0 ** 24), scale),
                    jnp.maximum(scale / 2, 1.0))
                good = jnp.where(grow, 0, good)
            else:
                new_scale = scale
            out = state.replace(
                step=pick(new.step, state.step),
                params=jax.tree_util.tree_map(pick, new.params, state.params),
                opt_state=jax.tree_util.tree_map(pick, new.opt_state, state.opt_state),
                loss_scale=new_scale, good_steps=good,
                skipped=state.skipped + jnp.where(finite, 0, 1),
            )
            return out, {"loss": l * inv, "gnorm": gnorm, "scale": scale,
                         "skipped": out.skipped}

        return jax.jit(step, donate_argnums=0, in_shardings=(state_sh, batch_sh),
                       out_shardings=(state_sh, None))
class Trainer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.tok = TokenizerManager(cfg)
        self.tok.get_or_train()
        self.mcfg = ModelCfg.from_config(cfg, self.tok.vocab_size)
        self.prec = pick_precision(cfg.get("training.precision", "auto"))
        self.seq = int(cfg.get("training.seq_len", 512))
        self.batch = int(cfg.get("training.batch", 64))
        self.accum = max(1, int(cfg.get("training.grad_accum", 1)))
        self.steps = int(cfg.get("training.steps", 20000))
        self.eval_batch = int(cfg.get("training.eval_batch", 128))
        self.model = LM(self.mcfg, dtype=DTYPES[self.prec], quantize=True,
                        remat=bool(cfg.get("training.remat", True)))

        devs = jax.devices()
        self.mesh = Mesh(np.array(devs), ("data",))
        self.repl = NamedSharding(self.mesh, P())
        self.batch_sh = NamedSharding(self.mesh, P(None, "data", None))
        for name, v in (("batch", self.batch), ("eval_batch", self.eval_batch)):
            if v % len(devs):
                raise ValueError(f"training.{name}={v} not divisible by "
                                 f"{len(devs)} devices")
        print(f"[env] {len(devs)} x {devs[0].device_kind} | precision={self.prec} "
              f"remat={self.model.remat} accum={self.accum}")

    def _init_state(self) -> TrainState:
        params = self.model.init(
            jax.random.PRNGKey(int(self.cfg.get("training.seed", 0))),
            jnp.zeros((1, self.seq), jnp.int32))["params"]
        return TrainState.create(
            apply_fn=self.model.apply, params=params,
            tx=build_optimizer(self.cfg, params),
            loss_scale=jnp.float32(2.0 ** 15 if self.prec == "fp16" else 1.0),
            good_steps=jnp.int32(0), skipped=jnp.int32(0))

    def _resume(self, state: TrainState, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"resume checkpoint not found: {path}")
        ck = load_ckpt(path)
        if model_cfg_from_ckpt(ck) != self.mcfg:
            raise ValueError("checkpoint model config differs from config.yaml")
        step = int(ck["step"])
        fresh, treedef = jax.tree_util.tree_flatten(state.opt_state)
        saved = ck.get("opt_leaves")
        if saved is not None and len(saved) == len(fresh) and all(
                np.shape(a) == np.shape(b) for a, b in zip(saved, fresh)):
            opt_state = jax.tree_util.tree_unflatten(
                treedef, [jnp.asarray(x) for x in saved])
        else:
            print("[resume] optimiser state incompatible -> re-initialised")
            opt_state = state.opt_state
        state = state.replace(
            params=jax.tree_util.tree_map(jnp.asarray, ck["params"]),
            opt_state=opt_state, step=jnp.int32(step))
        print(f"[resume] {path} @ step {step}")
        return state, step

    def run(self):
        cfg = self.cfg
        state, start = self._init_state(), 0
        if cfg.get("resume"):
            state, start = self._resume(state, cfg.get("resume"))
        n = sum(int(x.size) for x in jax.tree_util.tree_leaves(state.params))
        print(f"[model] {n / 1e6:.2f}M params | vocab={self.mcfg.vocab_size} "
              f"layers={self.mcfg.n_layers} | tokens/step="
              f"{self.batch * self.accum * self.seq:,}")

        state = jax.device_put(state, self.repl)
        train_step = make_train_step(self.accum, self.prec == "fp16",
                                     self.repl, self.batch_sh)
        eval_sh = NamedSharding(self.mesh, P("data", None))
        eval_step = jax.jit(
            lambda params, b: lm_loss(self.model.apply, params, b),
            in_shardings=(self.repl, eval_sh))
        eval_tokens = jax.device_put(
            fixed_tokens(self.tok, cfg, cfg.get("data.split_val", "validation"),
                         self.eval_batch, self.seq), eval_sh)

        # Different shuffle seed after a resume so we do not replay the same data.
        stream = PackedStream(
            self.tok,
            make_text_source(cfg, cfg.get("data.split_train", "train"),
                             seed=42 + start),
            self.batch, self.seq, repeat=True)
        batches = iter(Prefetcher(grouped(stream, self.accum), depth=4))

        log_every = int(cfg.get("training.log_every", 50))
        eval_every = int(cfg.get("training.eval_every", 1000))
        ckpt_every = int(cfg.get("training.ckpt_every", 1000))
        ckpt_dir = cfg.get("training.ckpt_dir", "./ckpts")
        os.makedirs(ckpt_dir, exist_ok=True)
        save = lambda s, p: save_train_ckpt(  # noqa: E731
            state, s, p, self.mcfg, cfg, self.prec)

        t0 = time.time()
        tok_seen = start * self.batch * self.accum * self.seq
        step = start
        try:
            for step in range(start, self.steps):
                state, m = train_step(
                    state, jax.device_put(next(batches), self.batch_sh))
                tok_seen += self.batch * self.accum * self.seq
                done = step + 1
                if step % log_every == 0 or done == self.steps:
                    m = jax.device_get(m)
                    dt = time.time() - t0
                    nll = float(m["loss"])
                    print(f"step {step:6d} | nll {nll:.4f} | ppl "
                          f"{math.exp(min(nll, 20)):8.2f} | gnorm {float(m['gnorm']):.2f}"
                          f" | scale {float(m['scale']):g} | skipped {int(m['skipped'])}"
                          f" | {tok_seen / max(dt, 1e-6) / 1e3:.1f}k tok/s")
                if done % eval_every == 0:
                    vl = float(eval_step(state.params, eval_tokens))
                    print(f"[eval] step {done:6d} | nll {vl:.4f} | "
                          f"ppl {math.exp(min(vl, 20)):.2f}")
                if done % ckpt_every == 0 and done != self.steps:
                    save(done, os.path.join(ckpt_dir, f"ckpt_{done:07d}.pkl"))
                    for old in sorted(glob.glob(
                            os.path.join(ckpt_dir, "ckpt_[0-9]*.pkl")))[:-2]:
                        os.remove(old)
        except KeyboardInterrupt:
            save(int(jax.device_get(state.step)),
                 os.path.join(ckpt_dir, "ckpt_interrupted.pkl"))
            print("[interrupted] checkpoint saved; set `resume:` to continue")
            return
        save(self.steps, os.path.join(ckpt_dir, "ckpt_final.pkl"))
        print("[done]")


# ================================================================
# EXPORT HELPERS (shared by eval + export so they cannot drift apart)
# ================================================================
def quantize_params(params, scales: Dict[str, np.ndarray]):
    """-> (quantised: name -> (int8, scale), fp32: name -> array).

    AWQ: kernels with an entry in `scales` are multiplied row-wise by s before
    quantisation; at inference the activation is divided by s.
    """
    q, fp = {}, {}
    for name, leaf in flatten_dict(params, sep="/").items():
        arr = np.asarray(leaf, np.float32)
        last = name.rsplit("/", 1)[-1]
        if last == "kernel" and arr.ndim == 2:
            s = scales.get(name)
            q[name] = quant_i8_np(arr * s[:, None] if s is not None else arr, axis=0)
        elif last == "embedding":
            q[name] = quant_i8_np(arr, axis=1)
        else:
            fp[name] = arr
    return q, fp


def simulate_export(params, scales):
    """Float params exactly equal to what the Pi computes (int8 round trip)."""
    q, fp = quantize_params(params, scales)
    out = dict(fp)
    for name, (idx, sc) in q.items():
        w = idx.astype(np.float32) * sc
        if name in scales:
            w = w / scales[name][:, None]
        out[name] = w
    return unflatten_dict(out, sep="/")


# ================================================================
# POST-PROCESSING: eval / prune / AWQ / export
# ================================================================
class PostProcessor:
    def __init__(self, cfg: Config, tok: TokenizerManager):
        self.cfg, self.tok = cfg, tok
        self.seq = int(cfg.get("training.seq_len", 512))
        self.alpha = float(cfg.get("awq.alpha", 0.5))
        self.n_calib = int(cfg.get("awq.calib_batches", 32)) * \
            int(cfg.get("awq.calib_batch_size", 8))
        self._calib = None
        self._val = None

    # ---- data ----
    def calib_tokens(self):
        if self._calib is None:  # packed, no padding; different seed than training
            self._calib = fixed_tokens(
                self.tok, self.cfg, self.cfg.get("data.split_train", "train"),
                self.n_calib, self.seq, seed=1234, shuffle=True)[:, :self.seq]
        return self._calib

    def val_tokens(self):
        if self._val is None:
            self._val = fixed_tokens(
                self.tok, self.cfg, self.cfg.get("data.split_val", "validation"),
                int(self.cfg.get("training.eval_batch", 128)), self.seq)
        return self._val

    # ---- evaluation (float32 compute, to isolate quantisation effects) ----
    def nll(self, ckpt, params=None, quantize=True, bs=16) -> float:
        model = LM(model_cfg_from_ckpt(ckpt), quantize=quantize)
        params = jax.tree_util.tree_map(
            jnp.asarray, ckpt["params"] if params is None else params)
        fn = jax.jit(lambda p, b: lm_loss(model.apply, p, b))
        toks, tot, n = self.val_tokens(), 0.0, 0
        for i in range(0, len(toks), bs):
            b = jnp.asarray(toks[i:i + bs])
            tot += float(fn(params, b)) * len(b)
            n += len(b)
        return tot / n

    def nll_export(self, ckpt, scales) -> float:
        return self.nll(ckpt, simulate_export(ckpt["params"], scales or {}),
                        quantize=False)

    # ---- calibration statistics ----
    def stats(self, ckpt, bs=8):
        model = LM(model_cfg_from_ckpt(ckpt), capture=True)
        params = jax.tree_util.tree_map(jnp.asarray, ckpt["params"])
        fn = jax.jit(lambda p, t: model.apply(
            {"params": p}, t, mutable=["intermediates"])[1]["intermediates"])
        toks, agg, n = self.calib_tokens(), {}, 0
        for i in range(0, len(toks), bs):
            flat = flatten_dict(fn(params, jnp.asarray(toks[i:i + bs])), sep="/")
            for k, v in flat.items():
                v = np.asarray(v[0] if isinstance(v, tuple) else v, np.float64)
                agg[k] = agg.get(k, 0.0) + v
            n += 1
        return {k: (v / n).astype(np.float32) for k, v in agg.items()}

    # ---- layer pruning ----
    def prune_layers(self, ckpt):
        pc = self.cfg
        mc = model_cfg_from_ckpt(ckpt)
        keep_n = max(1, int(round(mc.n_layers * float(pc.get("postprocess.layer_keep_ratio", 0.75)))))
        method = pc.get("postprocess.prune_method", "influence")
        if method == "tail":
            keep = list(range(keep_n))
        elif method == "influence":  # drop the blocks that change the stream least
            st = self.stats(ckpt)
            infl = np.array([1.0 - float(st[f"Block_{i}/cos_sim"])
                             for i in range(mc.n_layers)])
            keep = sorted(np.argsort(-infl)[:keep_n].tolist())
            print("[prune] block influence:", np.round(infl, 4).tolist())
        else:
            raise ValueError(f"unknown prune_method: {method}")
        print(f"[prune] {method}: {mc.n_layers} -> {keep_n} layers, keep {keep}")
        remap = {old: new for new, old in enumerate(keep)}
        params = {}
        for k, v in ckpt["params"].items():
            m = re.fullmatch(r"Block_(\d+)", k)
            if m is None:
                params[k] = v
            elif int(m.group(1)) in remap:
                params[f"Block_{remap[int(m.group(1))]}"] = v
        out = {k: v for k, v in ckpt.items() if k not in ("opt_leaves", "opt_state")}
        out.update(params=params, model={**asdict(mc), "n_layers": keep_n},
                   pruned_from=mc.n_layers)
        out.pop("awq_scales", None)
        return out

    # ---- AWQ ----
    def awq_scales(self, ckpt):
        lo, hi = 0.1, 10.0
        scales = {}
        for key, mag in self.stats(ckpt).items():
            if not key.endswith("/act_mag"):
                continue
            s = np.maximum(mag, 1e-6) ** self.alpha
            s = np.clip(s / s.mean(), lo, hi)
            scales[key[:-len("/act_mag")] + "/kernel"] = s.astype(np.float32)
        print(f"[awq] {len(scales)} kernels, alpha={self.alpha}")
        return scales

    # ---- export ----
    @staticmethod
    def export_int8(ckpt, scales, out_path):
        mc = model_cfg_from_ckpt(ckpt)
        q, fp = quantize_params(ckpt["params"], scales or {})
        arrays = {}
        for name, (idx, sc) in q.items():
            arrays[f"q/{name}/idx"] = idx
            arrays[f"q/{name}/scale"] = sc
        for name, arr in fp.items():
            arrays[f"f/{name}"] = arr
        for name, s in (scales or {}).items():
            arrays[f"awq/{name}"] = s
        meta = {"format": "micro-chat-int8", "version": 2, **asdict(mc),
                "max_len": int(Config(ckpt.get("config")).get("training.seq_len", 512)),
                "tie": mc.tie_embeddings}
        arrays["meta"] = np.frombuffer(json.dumps(meta).encode(), np.uint8)
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        np.savez_compressed(out_path, **arrays)
        print(f"[export] {out_path} ({os.path.getsize(out_path) / 1e6:.1f} MB) "
              f"awq={len(scales or {})} kernels")


def _ppl(x):
    return math.exp(min(x, 20))


def postprocess_main(cfg: Config, action: str, inp: Optional[str], out: Optional[str]):
    pp_in = inp or cfg.get("postprocess.input_ckpt", "./ckpts/ckpt_final.pkl")
    out_dir = cfg.get("postprocess.output_dir", "./ckpts_compressed")
    export_path = cfg.get("postprocess.export_path", "./pi_model.npz")
    if not os.path.exists(pp_in):
        sys.exit(f"[error] not found: {pp_in}")
    ckpt = load_ckpt(pp_in)
    pp = PostProcessor(cfg, TokenizerManager(cfg))
    pp.tok.get_or_train()
    awq_on = bool(cfg.get("awq.enabled", True))

    if action == "eval":
        sc = ckpt.get("awq_scales") or {}
        a = pp.nll(ckpt)
        print(f"[eval] QAT fake-quant       nll {a:.4f}  ppl {_ppl(a):.2f}")
        b = pp.nll_export(ckpt, {})
        print(f"[eval] int8 export          nll {b:.4f}  ppl {_ppl(b):.2f}")
        if sc:
            c = pp.nll_export(ckpt, sc)
            print(f"[eval] int8 export + AWQ    nll {c:.4f}  ppl {_ppl(c):.2f}")
    elif action == "prune_layers":
        _atomic_pickle(pp.prune_layers(ckpt),
                       out or os.path.join(out_dir, "pruned.pkl"))
    elif action == "awq":
        ckpt = {**ckpt, "awq_scales": pp.awq_scales(ckpt)}
        _atomic_pickle(ckpt, out or os.path.join(out_dir, "awq.pkl"))
    elif action == "export":
        pp.export_int8(ckpt, ckpt.get("awq_scales") or {}, out or export_path)
    elif action == "full":
        base = pp.nll(ckpt)
        print(f"[full] trained model     nll {base:.4f}  ppl {_ppl(base):.2f}")
        ckpt = pp.prune_layers(ckpt)
        _atomic_pickle(ckpt, os.path.join(out_dir, "pruned.pkl"))
        pr = pp.nll(ckpt)
        print(f"[full] after pruning     nll {pr:.4f}  ppl {_ppl(pr):.2f}")
        plain = pp.nll_export(ckpt, {})
        print(f"[full] int8, no AWQ      nll {plain:.4f}  ppl {_ppl(plain):.2f}")
        scales = {}
        if awq_on:
            cand = pp.awq_scales(ckpt)
            with_awq = pp.nll_export(ckpt, cand)
            print(f"[full] int8 + AWQ        nll {with_awq:.4f}  ppl {_ppl(with_awq):.2f}")
            if with_awq < plain:
                scales = cand
            else:
                print("[full] AWQ did not help on validation -> not applied")
        ckpt = {**ckpt, "awq_scales": scales}
        _atomic_pickle(ckpt, os.path.join(out_dir, "awq.pkl"))
        pp.export_int8(ckpt, scales, export_path)
        print("[full] done")


# ================================================================
# SAMPLING (JAX, uses the checkpoint's own model config)
# ================================================================
def sample_main(cfg: Config, path: Optional[str], prompt: str, n_new: int,
                temp: float, top_k: int, seed: int):
    path = path or cfg.get("postprocess.input_ckpt", "./ckpts/ckpt_final.pkl")
    ck = load_ckpt(path)
    tok = TokenizerManager(cfg)
    t = tok.load()
    model = LM(model_cfg_from_ckpt(ck), quantize=True)
    params = jax.tree_util.tree_map(jnp.asarray, ck["params"])
    seq = int(Config(ck.get("config")).get("training.seq_len", 512))
    fwd = jax.jit(lambda p, x: model.apply({"params": p}, x))
    rng = np.random.default_rng(seed)
    # training data is "story </s> story </s> ...", so start after a </s>
    ids = [tok.eos_id] + t.encode(prompt).ids
    for _ in range(n_new):
        window = ids[-seq:]
        arr = np.zeros((1, seq), np.int32)
        arr[0, :len(window)] = window
        logits = np.asarray(fwd(params, jnp.asarray(arr))[0, len(window) - 1],
                            np.float64) / max(temp, 1e-6)
        if 0 < top_k < logits.size:
            logits = np.where(logits < np.partition(logits, -top_k)[-top_k],
                              -np.inf, logits)
        p = np.exp(logits - logits.max())
        ids.append(int(rng.choice(p.size, p=p / p.sum())))
        if ids[-1] == tok.eos_id:
            break
    print(t.decode(ids, skip_special_tokens=True))


# ================================================================
# ENTRY
# ================================================================
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=os.environ.get("LM_CONFIG", "config.yaml"))
    ap.add_argument("--action", default="train",
                    choices=["train", "tokenizer", "sample", "eval",
                             "prune_layers", "awq", "export", "full"])
    ap.add_argument("--prompt", default="Once upon a time")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--input", default=None)
    ap.add_argument("--output", default=None)
    args = ap.parse_args(argv)

    cfg = Config.load(args.config)
    if args.action == "tokenizer":
        TokenizerManager(cfg).train()
    elif args.action == "train":
        Trainer(cfg).run()
    elif args.action == "sample":
        sample_main(cfg, args.input, args.prompt, args.n, args.temp,
                    args.top_k, args.seed)
    else:
        postprocess_main(cfg, args.action, args.input, args.output)


if __name__ == "__main__":
    main()

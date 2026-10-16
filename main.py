"""
micro-chat — ~7M dense Mamba-2 causal LM trained on TinyStories.
QAT int8 (per-output-channel) + layer pruning. Exports int8 for Raspberry Pi.

Actions:
  python main.py                    train
  python main.py --action tokenizer train 8k BPE
  python main.py --action prune_layers
  python main.py --action export
  python main.py --action full      prune -> export
  python main.py --action sample --prompt "Once upon a time"
"""
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.9")

try:
    from kaggle_secrets import UserSecretsClient
    os.environ.setdefault("HF_TOKEN", UserSecretsClient().get_secret("hg"))
except Exception:
    pass

import sys, glob, math, time, queue, pickle, threading, functools, argparse, re
import numpy as np
import jax, jax.numpy as jnp
import flax.linen as nn
import optax
import yaml
from flax.training import train_state
from flax.jax_utils import replicate, unreplicate


# ================================================================
# CONFIG
# ================================================================
CFG_PATH = os.environ.get("LM_CONFIG", "config.yaml")
with open(CFG_PATH) as f:
    CFG = yaml.safe_load(f)


def _get(path, default=None):
    v = CFG
    for k in path.split("."):
        if not isinstance(v, dict) or k not in v:
            return default
        v = v[k]
    return v


SEQ_LEN   = _get("training.seq_len", 512)
BATCH     = _get("training.batch", 64)
STEPS     = _get("training.steps", 20000)
LR        = float(_get("training.lr", 6e-4))
WARMUP    = _get("training.warmup", 300)
MIN_LR    = float(_get("training.min_lr_frac", 0.1))
LOG_EVERY = _get("training.log_every", 50)
EVAL_EVERY= _get("training.eval_every", 1000)
EVAL_BATCH= _get("training.eval_batch", 128)
CKPT_EVERY= _get("training.ckpt_every", 1000)
CKPT_DIR  = _get("training.ckpt_dir", "./ckpts")
REMAT     = _get("training.remat", False)
PREC      = _get("training.precision", "bf16").lower()
WD        = float(_get("training.weight_decay", 0.1))
RESUME    = _get("resume", None)

D_MODEL  = _get("model.d_model", 256)
N_LAYERS = _get("model.n_layers", 12)
TIE_EMB  = _get("model.tie_embeddings", True)

D_STATE  = _get("mamba2.d_state", 32)
D_CONV   = _get("mamba2.d_conv", 4)
CHUNK    = _get("mamba2.chunk", 128)
EXPAND   = _get("mamba2.expand", 2)
HEAD_DIM = _get("mamba2.head_dim", 64)

VOCAB_SIZE  = _get("tokenizer.vocab_size", 8192)
TOK_PATH    = _get("tokenizer.path", "./tokenizer.json")
TOK_MB      = _get("tokenizer.train_corpus_mb", 50)
TOK_SPECIAL = _get("tokenizer.special_tokens", ["<pad>", "<s>", "</s>", "<unk>"])

DATA_NAME    = _get("data.name", "roneneldan/TinyStories")
DATA_CFG     = _get("data.config", None)
DATA_SPLIT_TR= _get("data.split_train", "train")
DATA_SPLIT_VA= _get("data.split_val", "validation")
DATA_MIN     = _get("data.min_chars", 32)
DATA_SHUFFLE = _get("data.shuffle_buffer", 50000)

CLIP = float(_get("optim.clip_norm", 1.0))

PP            = _get("postprocess", {})
PP_INPUT      = PP.get("input_ckpt", "./ckpts/ckpt_final.pkl")
PP_OUTPUT_DIR = PP.get("output_dir", "./ckpts_compressed")
PP_L_KEEP     = float(PP.get("layer_keep_ratio", 0.75))
PP_EXPORT_PATH= PP.get("export_path", "./pi_model.pkl")

if PREC == "bf16":
    CT = jnp.bfloat16
elif PREC == "fp16":
    CT = jnp.float16
else:
    CT = jnp.float32
MT = jnp.float32

os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(PP_OUTPUT_DIR, exist_ok=True)


# ================================================================
# 8-BIT QAT
# ================================================================
QMAX = 127  # symmetric int8


def fake_quant_int8(x, axis=-1, eps=1e-8):
    """Symmetric per-axis int8 fake quantization (straight-through)."""
    x32 = x.astype(jnp.float32)
    s = jnp.max(jnp.abs(x32), axis=axis, keepdims=True) + eps
    idx = jnp.clip(jnp.round(x32 / s * QMAX), -QMAX, QMAX)
    xq = (idx * (s / QMAX)).astype(x.dtype)
    return x + jax.lax.stop_gradient(xq - x)


def quant_i8_np(w, axis=0):
    """Symmetric int8 quantization on numpy arrays, matching QAT."""
    w32 = w.astype(np.float32)
    s = np.max(np.abs(w32), axis=axis, keepdims=True) / QMAX
    s = np.maximum(s, 1e-8)
    idx = np.clip(np.round(w32 / s), -QMAX, QMAX).astype(np.int8)
    return idx, s.astype(np.float32)


# ================================================================
# TOKENIZER
# ================================================================
class TokenizerManager:
    """Trains / loads the BPE tokenizer and provides encode/decode."""

    def __init__(self, path=TOK_PATH, vocab_size=VOCAB_SIZE,
                 specials=TOK_SPECIAL, corpus_mb=TOK_MB,
                 dataset_name=DATA_NAME, dataset_config=DATA_CFG,
                 split=DATA_SPLIT_TR, min_chars=DATA_MIN):
        self.path = path
        self.vocab_size = vocab_size
        self.specials = specials
        self.corpus_mb = corpus_mb
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.split = split
        self.min_chars = min_chars
        self._tok = None

    def load(self):
        if self._tok is not None:
            return self._tok
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
        print(f"[tok] training {self.vocab_size}-BPE on {self.corpus_mb} MB "
              f"of {self.dataset_name}")
        from datasets import load_dataset
        from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

        ds = load_dataset(
            self.dataset_name, self.dataset_config,
            split=self.split, streaming=True,
        )

        def corpus():
            n = 0
            target = self.corpus_mb * 1024 * 1024
            for ex in ds:
                t = ex.get("text", "")
                if len(t) < self.min_chars:
                    continue
                yield t
                n += len(t)
                if n >= target:
                    return

        tok = Tokenizer(models.BPE(unk_token="<unk>"))
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tok.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=self.specials,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=True,
        )
        tok.train_from_iterator(corpus(), trainer=trainer)
        tok.save(self.path)
        self._tok = tok
        print(f"[tok] saved {self.path}  vocab={tok.get_vocab_size()}")
        return tok

    def encode(self, text):
        return self.load().encode(text).ids

    @property
    def eos_id(self):
        tok = self.load()
        i = tok.token_to_id("</s>")
        return 2 if i is None else i

    @property
    def actual_vocab_size(self):
        return self.load().get_vocab_size()


# ================================================================
# FLAX LAYERS
# ================================================================
class QDense(nn.Module):
    features: int
    capture: bool = False
    dtype: jnp.dtype = CT
    param_dtype: jnp.dtype = MT

    @nn.compact
    def __call__(self, x):
        if self.capture:
            self.sow("intermediates", "act_mag",
                     jnp.mean(jnp.abs(x.astype(jnp.float32)),
                              axis=tuple(range(x.ndim - 1))))
        k = self.param("kernel", nn.initializers.lecun_normal(),
                       (x.shape[-1], self.features))
        kq = fake_quant_int8(k, axis=0).astype(self.dtype)   # per-output-channel
        y = jnp.matmul(x.astype(self.dtype), kq,
                       preferred_element_type=jnp.float32)
        return y.astype(self.dtype)


def _pad_seq(x, p):
    if p == 0:
        return x
    pw = [(0, 0)] * x.ndim
    pw[1] = (0, p)
    return jnp.pad(x, pw, constant_values=0.0)


def segsum(x):
    T = x.shape[-1]
    cs = jnp.cumsum(x, axis=-1)
    ss = cs[..., :, None] - cs[..., None, :]
    mask = jnp.tril(jnp.ones((T, T), dtype=bool), k=0)
    return jnp.where(mask, ss, -jnp.inf)


def ssd_fwd(x, dt, A, Bm, Cm, chunk, D, dtb, dtmin, dtmax):
    Bs, L, H, P = x.shape
    dt = jnp.clip(jax.nn.softplus(dt + dtb), dtmin, dtmax)
    pad = (chunk - L % chunk) % chunk
    xp, dtp = _pad_seq(x, pad), _pad_seq(dt, pad)
    Bp, Cp = _pad_seq(Bm, pad), _pad_seq(Cm, pad)
    Lp = L + pad
    nC = Lp // chunk
    N = Bm.shape[-1]
    xc = xp.reshape(Bs, nC, chunk, H, P)
    dtc = dtp.reshape(Bs, nC, chunk, H)
    Bc = Bp.reshape(Bs, nC, chunk, H, N)
    Cc = Cp.reshape(Bs, nC, chunk, H, N)
    Ad = A.astype(jnp.float32) * dtc.astype(jnp.float32)
    At = jnp.transpose(Ad, (0, 1, 3, 2))
    Acs = jnp.cumsum(At, axis=-1)
    dec_end = jnp.exp(Acs[..., -1:])
    Lmat = jnp.exp(segsum(At))
    xd = xc.astype(jnp.float32) * dtc[..., None].astype(jnp.float32)
    C32 = Cc.astype(jnp.float32)
    B32 = Bc.astype(jnp.float32)
    sc = jnp.einsum("bclhn,bcshn->bchls", C32, B32) * Lmat
    yc = jnp.einsum("bchls,bcshp->bclhp", sc, xd)
    st = jnp.zeros((Bs, H, P, N), jnp.float32)
    dec_all = dec_end.squeeze(-1)
    yis = []
    for ci in range(nC):
        Bci, Cci = B32[:, ci], C32[:, ci]
        xdi, Aci = xd[:, ci], Acs[:, ci]
        dci = dec_all[:, ci]
        yi = jnp.einsum("blhn,bhpn->blhp", Cci, st)
        yi = yi * jnp.exp(Aci).transpose(0, 2, 1)[..., None]
        decay = jnp.exp(Aci[..., -1:] - Aci)
        cs = jnp.einsum("blhn,bhl,blhp->bhpn", Bci, decay, xdi)
        st = dci[..., None, None] * st + cs
        yis.append(yi)
    yf = yc + jnp.stack(yis, axis=1)
    Dres = D.astype(jnp.float32).reshape(1, 1, 1, H, 1) * xc.astype(jnp.float32)
    yf = (yf + Dres).reshape(Bs, Lp, H, P)
    if pad:
        yf = yf[:, :L]
    return yf.astype(x.dtype)


class Mamba2Block(nn.Module):
    d_model: int
    d_state: int = D_STATE
    d_conv: int = D_CONV
    chunk: int = CHUNK
    expand: int = EXPAND
    head_dim: int = HEAD_DIM
    capture: bool = False

    @nn.compact
    def __call__(self, x):
        B, T, C = x.shape
        d_inner = self.expand * C
        H = d_inner // self.head_dim
        P = self.head_dim
        N = self.d_state

        h = nn.LayerNorm(epsilon=1e-6, dtype=jnp.float32)(x.astype(jnp.float32))
        h = h.astype(CT)
        proj = QDense(2 * d_inner + 2 * N + H, capture=self.capture)(h)
        z, xBC, dt = jnp.split(proj, [d_inner, 2 * d_inner + 2 * N], axis=-1)

        xBC = jnp.pad(xBC, [(0, 0), (self.d_conv - 1, 0), (0, 0)])
        xBC = nn.Conv(
            features=xBC.shape[-1], kernel_size=(self.d_conv,),
            padding="VALID", feature_group_count=xBC.shape[-1],
            use_bias=True, dtype=CT, param_dtype=MT,
        )(xBC)
        xBC = nn.silu(xBC)

        xi, Bt, Ct = jnp.split(xBC, [d_inner, d_inner + N], axis=-1)
        xi = xi.reshape(B, T, H, P)
        Bt = Bt.reshape(B, T, N)
        Ct = Ct.reshape(B, T, N)
        Be = jnp.broadcast_to(Bt[:, :, None, :], (B, T, H, N))
        Ce = jnp.broadcast_to(Ct[:, :, None, :], (B, T, H, N))

        A_log = self.param(
            "A_log",
            lambda k, s: jnp.log(jax.random.uniform(k, s, 1.0, 16.0)),
            (H,),
        )
        A = -jnp.exp(A_log.astype(jnp.float32))
        D = self.param("D", nn.initializers.ones, (H,))

        def _dtb(k, s):
            u = jax.random.uniform(k, s)
            v = jnp.exp(u * (math.log(0.1) - math.log(1e-3)) + math.log(1e-3))
            return v + jnp.log(-jnp.expm1(-v))

        dtb = self.param("dt_bias", _dtb, (H,))

        y = ssd_fwd(xi, dt, A, Be, Ce, self.chunk, D, dtb, 0.0, 100.0)
        y = y.reshape(B, T, d_inner) * nn.silu(z)
        return QDense(C, capture=self.capture)(y)


class Block(nn.Module):
    d_model: int
    remat: bool = False
    capture: bool = False

    @nn.compact
    def __call__(self, x):
        Mb = nn.remat(Mamba2Block) if self.remat else Mamba2Block
        return x + Mb(self.d_model, capture=self.capture)(x)


class LM(nn.Module):
    vocab: int
    d_model: int = D_MODEL
    n_layers: int = N_LAYERS
    max_len: int = SEQ_LEN
    tie_embeddings: bool = TIE_EMB
    remat: bool = False
    capture: bool = False

    @nn.compact
    def __call__(self, tokens):
        emb = nn.Embed(self.vocab, self.d_model,
                       dtype=CT, param_dtype=MT,
                       embedding_init=nn.initializers.normal(0.02))
        x = emb(tokens).astype(jnp.float32)
        for li in range(self.n_layers):
            x = Block(self.d_model, remat=self.remat,
                      capture=self.capture)(x)
        x = nn.LayerNorm(epsilon=1e-6, dtype=jnp.float32)(x.astype(jnp.float32))
        x = x.astype(CT)
        w = emb.variables["params"]["embedding"]
        wq = fake_quant_int8(w, axis=1).astype(CT)   # per-token-row
        logits = jnp.matmul(x, wq.T, preferred_element_type=jnp.float32)
        return logits.astype(jnp.float32)


# ================================================================
# DATA
# ================================================================
class TokenizedStream:
    """Yields packed (batch, seq+1) int32 arrays from a HF text stream."""

    def __init__(self, tokenizer: TokenizerManager,
                 dataset_name=DATA_NAME, dataset_config=DATA_CFG,
                 split=DATA_SPLIT_TR,
                 batch=BATCH, seq=SEQ_LEN, seed=42,
                 shuffle_buffer=DATA_SHUFFLE, min_chars=DATA_MIN,
                 skip_docs=0):
        self.tok = tokenizer
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.split = split
        self.batch = batch
        self.seq = seq
        self.seed = seed
        self.shuffle_buffer = shuffle_buffer
        self.min_chars = min_chars
        self.skip_docs = skip_docs
        self.eos = tokenizer.eos_id

    def _texts(self):
        from datasets import load_dataset
        # config=None is valid and means "no sub-config".
        ds = load_dataset(
            self.dataset_name,
            self.dataset_config,
            split=self.split,
            streaming=True,
        )
        if self.skip_docs:
            ds = ds.skip(self.skip_docs)
        if self.shuffle_buffer:
            ds = ds.shuffle(seed=self.seed, buffer_size=self.shuffle_buffer)
        for ex in ds:
            t = ex.get("text", "")
            if len(t) >= self.min_chars:
                yield t

    def __iter__(self):
        buf = []
        need = self.batch * (self.seq + 1)
        for text in self._texts():
            buf.extend(self.tok.encode(text))
            buf.append(self.eos)
            while len(buf) >= need:
                chunk = np.asarray(buf[:need], np.int32).reshape(
                    self.batch, self.seq + 1,
                )
                del buf[:need]
                yield chunk


class DataPipeline:
    """Background-prefetched train batches + a deterministic val batch."""

    def __init__(self, tokenizer: TokenizerManager,
                 batch=BATCH, seq=SEQ_LEN, n_steps=STEPS, prefetch=4):
        self.tokenizer = tokenizer
        self.batch = batch
        self.seq = seq
        self.n_steps = n_steps
        self.prefetch = prefetch
        self._q = None
        self._thread = None

    def start(self):
        self._q = queue.Queue(maxsize=self.prefetch)
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _worker(self):
        try:
            stream = TokenizedStream(
                self.tokenizer,
                batch=self.batch, seq=self.seq,
                split=DATA_SPLIT_TR,
            )
            for i, chunk in enumerate(stream):
                if i >= self.n_steps:
                    break
                self._q.put(chunk)
        except Exception as e:            # noqa: BLE001
            self._q.put(e)
        finally:
            self._q.put(None)

    def __iter__(self):
        while True:
            item = self._q.get()
            if item is None:
                return
            if isinstance(item, Exception):
                raise item
            yield item

    def validation_batch(self):
        """Deterministic held-out batch from the validation split."""
        stream = TokenizedStream(
            self.tokenizer, batch=1, seq=self.seq,
            split=DATA_SPLIT_VA,
            shuffle_buffer=0,             # val must be deterministic
        )
        need = EVAL_BATCH * (self.seq + 1)
        buf = []
        for chunk in stream:
            buf.extend(chunk.reshape(-1).tolist())
            if len(buf) >= need:
                break
        arr = np.asarray(buf[:need], np.int32).reshape(EVAL_BATCH, self.seq + 1)
        return jnp.asarray(arr)


# ================================================================
# CHECKPOINTS
# ================================================================
def save_ckpt(state, step, path):
    host = unreplicate(state)
    payload = {
        "step": step,
        "params": jax.tree_util.tree_map(np.asarray, host.params),
        "opt_state": jax.tree_util.tree_map(np.asarray, host.opt_state),
        "config": CFG,
    }
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(payload, f, protocol=4)
    os.replace(tmp, path)
    print(f"[ckpt] {path} ({os.path.getsize(path)/1e6:.1f} MB)")


def load_ckpt(path):
    with open(path, "rb") as f:
        return pickle.load(f)


# ================================================================
# TRAINER
# ================================================================
def _wd_mask(params):
    def _k(path, leaf):
        keys = [str(getattr(k, "key", k)) for k in path]
        if any("LayerNorm" in k for k in keys):
            return False
        return keys[-1] not in ("bias", "A_log", "D", "dt_bias", "embedding")
    return jax.tree_util.tree_map_with_path(_k, params)


class Trainer:
    """Model init, optim, training loop, checkpointing."""

    def __init__(self):
        self.tok = TokenizerManager()
        self.tok.get_or_train()
        self.vocab = self.tok.actual_vocab_size
        self.model = LM(vocab=self.vocab, d_model=D_MODEL, n_layers=N_LAYERS,
                        max_len=SEQ_LEN, remat=REMAT)
        self.devices = jax.devices()
        self.n_dev = len(self.devices)
        assert BATCH % self.n_dev == 0, "batch must divide #devices"
        self.per_dev = BATCH // self.n_dev

    def _make_state(self):
        d = jnp.zeros((1, SEQ_LEN), jnp.int32)
        params = self.model.init(jax.random.PRNGKey(0), d)["params"]
        sched = optax.warmup_cosine_decay_schedule(
            init_value=0.0, peak_value=LR, warmup_steps=WARMUP,
            decay_steps=STEPS, end_value=LR * MIN_LR,
        )
        tx = optax.chain(
            optax.clip_by_global_norm(CLIP),
            optax.adafactor(learning_rate=None,
                            multiply_by_parameter_scale=False),
            optax.add_decayed_weights(WD, _wd_mask(params)),
            optax.scale_by_learning_rate(sched),
        )
        return train_state.TrainState.create(
            apply_fn=self.model.apply, params=params, tx=tx,
        )

    @staticmethod
    @functools.partial(jax.pmap, axis_name="dev", donate_argnums=(0,))
    def train_step(state, batch):
        def loss_fn(params):
            inp, tgt = batch[:, :-1], batch[:, 1:]
            logits = state.apply_fn({"params": params}, inp)
            return optax.softmax_cross_entropy_with_integer_labels(
                logits, tgt,
            ).mean()
        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        grads = jax.lax.pmean(grads, axis_name="dev")
        state = state.apply_gradients(grads=grads)
        return state, jax.lax.pmean(loss, "dev")

    @staticmethod
    @functools.partial(jax.pmap, axis_name="dev")
    def eval_step(state, batch):
        inp, tgt = batch[:, :-1], batch[:, 1:]
        logits = state.apply_fn({"params": state.params}, inp)
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits, tgt,
        ).mean()
        return jax.lax.pmean(loss, "dev")

    def run(self):
        state = self._make_state()
        start = 0
        if RESUME and os.path.exists(RESUME):
            ck = load_ckpt(RESUME)
            start = int(ck["step"])
            state = state.replace(
                params=jax.tree_util.tree_map(jnp.asarray, ck["params"]),
                opt_state=jax.tree_util.tree_map(jnp.asarray, ck["opt_state"]),
                step=start,
            )
            print(f"[resume] {RESUME} @ {start}")

        n = sum(int(x.size) for x in jax.tree_util.tree_leaves(state.params))
        print(f"devices: {self.n_dev}  vocab={self.vocab}  "
              f"seq={SEQ_LEN}  batch={BATCH}  params={n/1e6:.2f}M")

        data = DataPipeline(self.tok, n_steps=STEPS - start)
        data.start()
        eb = data.validation_batch().reshape(
            self.n_dev, EVAL_BATCH // self.n_dev, SEQ_LEN + 1,
        )

        state = replicate(state)
        t0 = time.time()
        for ls, chunk in enumerate(data):
            step = start + ls
            batch = jnp.asarray(chunk).reshape(
                self.n_dev, self.per_dev, SEQ_LEN + 1,
            )
            state, loss = self.train_step(state, batch)

            if step % LOG_EVERY == 0 or step == STEPS - 1:
                dt = time.time() - t0
                lf = float(unreplicate(loss).mean())
                tokc = (ls + 1) * BATCH * SEQ_LEN
                print(f"step {step:6d} | nll {lf:.4f} | "
                      f"ppl {math.exp(min(lf, 20)):7.2f} | "
                      f"{dt:6.1f}s | {tokc/1e6:.1f}M tok")

            if step > 0 and step % EVAL_EVERY == 0:
                vl = float(unreplicate(self.eval_step(state, eb)).mean())
                print(f"[eval] step {step:6d} | nll {vl:.4f} | "
                      f"ppl {math.exp(min(vl, 20)):7.2f}")

            if step > 0 and step % CKPT_EVERY == 0:
                save_ckpt(state, step + 1,
                          os.path.join(CKPT_DIR, f"ckpt_{step:06d}.pkl"))
                for old in sorted(
                    glob.glob(os.path.join(CKPT_DIR, "ckpt_0*.pkl"))
                )[:-2]:
                    os.remove(old)

        save_ckpt(state, STEPS, os.path.join(CKPT_DIR, "ckpt_final.pkl"))
        print("[done]")


# ================================================================
# SAMPLING (full-seq, JAX — for GPU-side sanity checks)
# ================================================================
def sample_main(prompt, n_new=200, temp=0.8, top_k=40):
    tok = TokenizerManager().load()
    model = LM(vocab=tok.get_vocab_size(), d_model=D_MODEL,
               n_layers=N_LAYERS, max_len=SEQ_LEN, remat=False)
    ck = load_ckpt(PP_INPUT)
    params = jax.tree_util.tree_map(jnp.asarray, ck["params"])
    ids = tok.encode(prompt).ids
    eos = tok.token_to_id("</s>")
    if eos is None:
        eos = 2
    fwd = jax.jit(lambda p, t: model.apply({"params": p}, t))
    for _ in range(n_new):
        window = ids[-SEQ_LEN:]
        arr = np.zeros((1, SEQ_LEN), np.int32)
        arr[0, :len(window)] = window
        logits = np.asarray(
            fwd(params, jnp.asarray(arr))[0, len(window) - 1], np.float64,
        ) / temp
        if top_k:
            kth = np.partition(logits, -top_k)[-top_k]
            logits = np.where(logits < kth, -1e9, logits)
        p = np.exp(logits - logits.max())
        p /= p.sum()
        ids.append(int(np.random.choice(len(p), p=p)))
        if ids[-1] == eos:
            break
    print(tok.decode(ids))


# ================================================================
# POSTPROCESS (prune + int8 export)
# ================================================================
class PostProcessor:
    """Layer pruning and int8 export for Raspberry Pi deployment."""

    def __init__(self, keep_ratio=PP_L_KEEP):
        self.keep_ratio = keep_ratio

    def prune_layers(self, ckpt):
        n_old = ckpt["config"]["model"]["n_layers"]
        n_new = max(1, int(round(n_old * self.keep_ratio)))
        print(f"[prune] layers {n_old} -> {n_new}")
        new_params = {}
        for k, v in ckpt["params"].items():
            m = re.match(r"Block_(\d+)", k)
            if m and int(m.group(1)) >= n_new:
                continue
            new_params[k] = v
        out = dict(ckpt)
        out["params"] = new_params
        cfg = dict(ckpt["config"])
        cfg["model"] = dict(cfg["model"])
        cfg["model"]["n_layers"] = n_new
        out["config"] = cfg
        return out

    def export_int8(self, ckpt, out_path):
        params = ckpt["params"]
        c = ckpt["config"]
        bundle = {
            "vocab_size": c["tokenizer"]["vocab_size"],
            "d_model": c["model"]["d_model"],
            "n_layers": c["model"]["n_layers"],
            "max_len": c["training"]["seq_len"],
            "tie": c["model"].get("tie_embeddings", True),
            "mamba": {
                "d_state": c["mamba2"]["d_state"],
                "d_conv": c["mamba2"]["d_conv"],
                "expand": c["mamba2"]["expand"],
                "head_dim": c["mamba2"]["head_dim"],
            },
            "quant": {},
            "fp32": {},
        }

        def _walk(path, leaf):
            keys = [str(getattr(k, "key", k)) for k in path]
            name = keys[-1]
            ps = "/".join(keys)
            arr = np.asarray(leaf, dtype=np.float32)
            if name == "kernel" and arr.ndim == 2:
                idx, s = quant_i8_np(arr, axis=0)
                bundle["quant"][ps] = {
                    "idx": idx, "scale": s, "shape": arr.shape,
                }
            elif name == "embedding":
                idx, s = quant_i8_np(arr, axis=1)
                bundle["quant"][ps] = {
                    "idx": idx, "scale": s, "shape": arr.shape,
                }
            else:
                bundle["fp32"][ps] = arr

        jax.tree_util.tree_map_with_path(_walk, params)
        with open(out_path, "wb") as f:
            pickle.dump(bundle, f, protocol=4)
        print(f"[export] {out_path}  ({os.path.getsize(out_path)/1e6:.1f} MB)")


def _save(payload, path):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(payload, f, protocol=4)
    os.replace(tmp, path)
    print(f"[saved] {path}")


def postprocess_main(action, input_path, output_path):
    p = input_path or PP_INPUT
    if not os.path.exists(p):
        print(f"[error] not found: {p}")
        sys.exit(1)
    ckpt = load_ckpt(p)
    pp = PostProcessor(PP_L_KEEP)

    if action == "prune_layers":
        ckpt = pp.prune_layers(ckpt)
        _save(ckpt, output_path or os.path.join(PP_OUTPUT_DIR, "pruned.pkl"))
    elif action == "export":
        pp.export_int8(ckpt, output_path or PP_EXPORT_PATH)
    elif action == "full":
        ckpt = pp.prune_layers(ckpt)
        _save(ckpt, os.path.join(PP_OUTPUT_DIR, "pruned.pkl"))
        pp.export_int8(ckpt, PP_EXPORT_PATH)
        print("[full] done")


# ================================================================
# ENTRY
# ================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--action", default="train",
                    choices=["train", "tokenizer", "sample",
                             "prune_layers", "export", "full"])
    ap.add_argument("--prompt", default="Once upon a time")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--input", default=None)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    if args.action == "tokenizer":
        TokenizerManager().train()
    elif args.action == "train":
        Trainer().run()
    elif args.action == "sample":
        sample_main(args.prompt, n_new=args.n)
    else:
        postprocess_main(args.action, args.input, args.output)


if __name__ == "__main__":
    main()

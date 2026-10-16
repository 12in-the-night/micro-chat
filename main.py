"""
micro-chat — ~7M dense Mamba-2 causal LM trained on TinyStories.
QAT int8 (per-output-channel) + AWQ + layer pruning. Exports int8 for Pi.

Supports single-GPU and multi-GPU (pmap). fp16 on Turing (T4), bf16 on
Ampere+, fp32 on CPU — auto-detected.

Actions:
  python main.py                    train
  python main.py --action tokenizer train 8k BPE
  python main.py --action prune_layers
  python main.py --action awq
  python main.py --action export
  python main.py --action full      prune -> AWQ -> export
  python main.py --action sample --prompt "Once upon a time"
"""
import os

# ---- XLA flags: multi-GPU latency hiding + PCIe P2P (no NVLink on T4) ----
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.9")
os.environ.setdefault(
    "XLA_FLAGS",
    "--xla_gpu_enable_latency_hiding_scheduler=true "
    "--xla_gpu_enable_pipelined_p2p=true "
    "--xla_gpu_memory_limit_slop_factor=5",
)

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


SEQ_LEN    = _get("training.seq_len", 512)
BATCH      = _get("training.batch", 64)
GRAD_ACCUM = max(1, int(_get("training.grad_accum", 1)))
STEPS      = _get("training.steps", 20000)
LR         = float(_get("training.lr", 6e-4))
WARMUP     = _get("training.warmup", 300)
MIN_LR     = float(_get("training.min_lr_frac", 0.1))
LOG_EVERY  = _get("training.log_every", 50)
EVAL_EVERY = _get("training.eval_every", 1000)
EVAL_BATCH = _get("training.eval_batch", 128)
CKPT_EVERY = _get("training.ckpt_every", 1000)
CKPT_DIR   = _get("training.ckpt_dir", "./ckpts")
REMAT      = bool(_get("training.remat", True))
PREC_CFG   = str(_get("training.precision", "auto")).lower()
WD         = float(_get("training.weight_decay", 0.1))
RESUME     = _get("resume", None)

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

DATA_NAME     = _get("data.name", "roneneldan/TinyStories")
DATA_CFG      = _get("data.config", None)
DATA_SPLIT_TR = _get("data.split_train", "train")
DATA_SPLIT_VA = _get("data.split_val", "validation")
DATA_MIN      = _get("data.min_chars", 32)
DATA_SHUFFLE  = _get("data.shuffle_buffer", 50000)

AWQ_ENABLED = bool(_get("awq.enabled", True))
AWQ_ALPHA   = float(_get("awq.alpha", 0.5))
AWQ_BATCHES = int(_get("awq.calib_batches", 32))
AWQ_BS      = int(_get("awq.calib_batch_size", 8))

CLIP = float(_get("optim.clip_norm", 1.0))
AF = _get("optim.adafactor", {}) or {}
AF_ENABLED   = bool(AF.get("enabled", True))
AF_MIN_DIM   = int(AF.get("min_dim_size_to_factor", 128))
AF_DECAY     = float(AF.get("decay_rate", 0.8))
AF_OFFSET    = int(AF.get("decay_offset", 0))
AF_PARAM_SCL = bool(AF.get("multiply_by_parameter_scale", False))
AF_CLIP      = float(AF.get("clipping_threshold", 1.0))
AF_MOM       = AF.get("momentum", 0.9)
AF_MOM       = None if AF_MOM is None else float(AF_MOM)
AF_EPS_OUT   = float(AF.get("eps_outer", 1e-3))

PP            = _get("postprocess", {})
PP_INPUT      = PP.get("input_ckpt", "./ckpts/ckpt_final.pkl")
PP_OUTPUT_DIR = PP.get("output_dir", "./ckpts_compressed")
PP_L_KEEP     = float(PP.get("layer_keep_ratio", 0.75))
PP_EXPORT_PATH= PP.get("export_path", "./pi_model.pkl")


# ================================================================
# AUTO PRECISION (T4 = Turing/sm_75: no bf16 -> fp16)
# ================================================================
def _pick_precision(cfg_prec: str) -> str:
    if cfg_prec in ("bf16", "fp16", "fp32"):
        return cfg_prec
    try:
        kind = jax.devices()[0].device_kind.lower()
    except Exception:
        kind = ""
    if "t4" in kind or "v100" in kind or "p100" in kind or "turing" in kind:
        return "fp16"
    if "a100" in kind or "h100" in kind or "a10" in kind or "l4" in kind:
        return "bf16"
    if "cuda" in kind or "gpu" in kind:
        return "fp16"
    return "fp32"


PREC = _pick_precision(PREC_CFG)
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
QMAX = 127


def fake_quant_int8(x, axis=-1, eps=1e-8):
    x32 = x.astype(jnp.float32)
    s = jnp.max(jnp.abs(x32), axis=axis, keepdims=True) + eps
    idx = jnp.clip(jnp.round(x32 / s * QMAX), -QMAX, QMAX)
    xq = (idx * (s / QMAX)).astype(x.dtype)
    return x + jax.lax.stop_gradient(xq - x)


def quant_i8_np(w, axis=0):
    w32 = w.astype(np.float32)
    s = np.max(np.abs(w32), axis=axis, keepdims=True) / QMAX
    s = np.maximum(s, 1e-8)
    idx = np.clip(np.round(w32 / s), -QMAX, QMAX).astype(np.int8)
    return idx, s.astype(np.float32)


# ================================================================
# PATH HELPERS
# ================================================================
def _key(k):
    return str(getattr(k, "key", k))


def _pn(path):
    return [_key(k) for k in path]


def _flat(tree, prefix=""):
    out = {}
    for k, v in tree.items():
        p = f"{prefix}/{k}" if prefix else str(k)
        out.update(_flat(v, p) if isinstance(v, dict) else {p: v})
    return out


# ================================================================
# TOKENIZER
# ================================================================
class TokenizerManager:
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

        ds = load_dataset(self.dataset_name, self.dataset_config,
                          split=self.split, streaming=True)

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
        kq = fake_quant_int8(k, axis=0).astype(self.dtype)
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
    """
    Mamba-2 block with residual.

    remat=True wraps the whole block (residual included) in jax.remat,
    which is where checkpointing actually saves memory. It's disabled
    automatically when capture=True (AWQ calibration), because `sow`
    inside `nn.remat` is fragile.
    """
    d_model: int
    remat: bool = False
    capture: bool = False

    @nn.compact
    def __call__(self, x):
        def body(x):
            return x + Mamba2Block(self.d_model, capture=self.capture)(x)

        if self.remat and not self.capture:
            return nn.remat(body, prevent_cse=False)(x)
        return body(x)


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
        wq = fake_quant_int8(w, axis=1).astype(CT)
        logits = jnp.matmul(x, wq.T, preferred_element_type=jnp.float32)
        return logits.astype(jnp.float32)


# ================================================================
# DATA
# ================================================================
class TokenizedStream:
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
        ds = load_dataset(self.dataset_name, self.dataset_config,
                          split=self.split, streaming=True)
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
            # Number of micro-batches = optimizer_steps * grad_accum
            stream = TokenizedStream(
                self.tokenizer,
                batch=self.batch, seq=self.seq, split=DATA_SPLIT_TR,
            )
            for i, chunk in enumerate(stream):
                if i >= self.n_steps * GRAD_ACCUM:
                    break
                self._q.put(chunk)
        except Exception as e:
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
        stream = TokenizedStream(
            self.tokenizer, batch=1, seq=self.seq,
            split=DATA_SPLIT_VA, shuffle_buffer=0,
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
def save_ckpt(state, step, path, extra=None):
    host = unreplicate(state)
    payload = {
        "step": step,
        "params": jax.tree_util.tree_map(np.asarray, host.params),
        "opt_state": jax.tree_util.tree_map(np.asarray, host.opt_state),
        "config": CFG,
        "precision": PREC,
    }
    if extra:
        payload.update(extra)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(payload, f, protocol=4)
    os.replace(tmp, path)
    print(f"[ckpt] {path} ({os.path.getsize(path)/1e6:.1f} MB)")


def load_ckpt(path):
    with open(path, "rb") as f:
        return pickle.load(f)


# ================================================================
# OPTIM / TRAINER
# ================================================================
def _wd_mask(params):
    def _k(path, leaf):
        keys = [str(getattr(k, "key", k)) for k in path]
        if any("LayerNorm" in k for k in keys):
            return False
        return keys[-1] not in ("bias", "A_log", "D", "dt_bias", "embedding")
    return jax.tree_util.tree_map_with_path(_k, params)


def _build_optimizer(params):
    """Clip -> decay (gradient) -> Adafactor -> schedule (learning rate)."""
    sched = optax.warmup_cosine_decay_schedule(
        init_value=0.0, peak_value=LR, warmup_steps=WARMUP,
        decay_steps=STEPS, end_value=LR * MIN_LR,
    )

    if AF_ENABLED:
        grad_tx = optax.adafactor(
            learning_rate=None,                 # scale_by_learning_rate applies LR
            min_dim_size_to_factor=AF_MIN_DIM,
            decay_rate=AF_DECAY,
            decay_offset=AF_OFFSET,
            multiply_by_parameter_scale=AF_PARAM_SCL,
            clipping_threshold=AF_CLIP,
            momentum=AF_MOM,
            dtype_momentum=jnp.float32,
            eps=(1e-30, AF_EPS_OUT),
        )
    else:
        grad_tx = optax.adam(learning_rate=None, b1=0.9, b2=0.95)

    return optax.chain(
        optax.clip_by_global_norm(CLIP),
        optax.add_decayed_weights(WD, _wd_mask(params)),  # BEFORE adafactor
        grad_tx,
        optax.scale_by_learning_rate(sched),              # -lr * update
    )


class Trainer:
    def __init__(self):
        self.tok = TokenizerManager()
        self.tok.get_or_train()
        self.vocab = self.tok.actual_vocab_size
        self.model = LM(vocab=self.vocab, d_model=D_MODEL, n_layers=N_LAYERS,
                        max_len=SEQ_LEN, remat=REMAT)
        self.devices = jax.devices()
        self.n_dev = len(self.devices)
        assert BATCH % self.n_dev == 0, \
            f"global batch {BATCH} must divide #devices {self.n_dev}"
        self.per_dev = BATCH // self.n_dev

        kind = self.devices[0].device_kind if self.n_dev else "?"
        print(f"[env] devices={self.n_dev} kind={kind} precision={PREC} "
              f"remat={REMAT} grad_accum={GRAD_ACCUM}")

    def _make_state(self):
        d = jnp.zeros((1, SEQ_LEN), jnp.int32)
        params = self.model.init(jax.random.PRNGKey(0), d)["params"]
        tx = _build_optimizer(params)
        return train_state.TrainState.create(
            apply_fn=self.model.apply, params=params, tx=tx,
        )

    # ----- gradient accumulation primitives -----
    @staticmethod
    @functools.partial(jax.pmap, axis_name="dev")
    def _loss_and_grads(state, batch):
        def loss_fn(params):
            inp, tgt = batch[:, :-1], batch[:, 1:]
            logits = state.apply_fn({"params": params}, inp)
            return optax.softmax_cross_entropy_with_integer_labels(
                logits, tgt,
            ).mean()
        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        grads = jax.lax.pmean(grads, axis_name="dev")
        loss = jax.lax.pmean(loss, axis_name="dev")
        return grads, loss

    @staticmethod
    @functools.partial(jax.pmap, axis_name="dev", donate_argnums=(0,))
    def _apply_grads(state, grads):
        return state.apply_gradients(grads=grads)

    @staticmethod
    @functools.partial(jax.pmap, axis_name="dev")
    def eval_step(state, batch):
        inp, tgt = batch[:, :-1], batch[:, 1:]
        logits = state.apply_fn({"params": state.params}, inp)
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits, tgt,
        ).mean()
        return jax.lax.pmean(loss, "dev")

    def _to_dev(self, chunk):
        return jnp.asarray(chunk).reshape(
            self.n_dev, self.per_dev, SEQ_LEN + 1,
        )

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
        print(f"[model] {n/1e6:.2f}M params  vocab={self.vocab}  "
              f"seq={SEQ_LEN}  batch={BATCH}  accum={GRAD_ACCUM}  "
              f"effective_batch={BATCH * GRAD_ACCUM}")

        data = DataPipeline(self.tok, n_steps=STEPS - start)
        data.start()
        eb = data.validation_batch().reshape(
            self.n_dev, EVAL_BATCH // self.n_dev, SEQ_LEN + 1,
        )

        state = replicate(state)
        stream = iter(data)
        t0 = time.time()
        seen_tokens = 0

        for opt_step in range(start, STEPS):
            accum = None
            loss_sum = None
            for _ in range(GRAD_ACCUM):
                chunk = next(stream)
                batch = self._to_dev(chunk)
                grads, loss = self._loss_and_grads(state, batch)
                accum = grads if accum is None else \
                    jax.tree_util.tree_map(jnp.add, accum, grads)
                loss_sum = loss if loss_sum is None else loss_sum + loss
            accum = jax.tree_util.tree_map(
                lambda g: g / GRAD_ACCUM, accum,
            )
            state = self._apply_grads(state, accum)
            avg_loss = loss_sum / GRAD_ACCUM
            seen_tokens += BATCH * GRAD_ACCUM * SEQ_LEN

            if opt_step % LOG_EVERY == 0 or opt_step == STEPS - 1:
                dt = time.time() - t0
                lf = float(unreplicate(avg_loss).mean())
                print(f"step {opt_step:6d} | nll {lf:.4f} | "
                      f"ppl {math.exp(min(lf, 20)):7.2f} | "
                      f"{dt:6.1f}s | {seen_tokens/1e6:.1f}M tok | "
                      f"{seen_tokens/max(dt,1e-6)/1e3:.1f}k tok/s")

            if opt_step > 0 and opt_step % EVAL_EVERY == 0:
                vl = float(unreplicate(self.eval_step(state, eb)).mean())
                print(f"[eval] step {opt_step:6d} | nll {vl:.4f} | "
                      f"ppl {math.exp(min(vl, 20)):7.2f}")

            if opt_step > 0 and opt_step % CKPT_EVERY == 0:
                save_ckpt(state, opt_step,
                          os.path.join(CKPT_DIR, f"ckpt_{opt_step:06d}.pkl"))
                for old in sorted(
                    glob.glob(os.path.join(CKPT_DIR, "ckpt_0*.pkl"))
                )[:-2]:
                    os.remove(old)

        save_ckpt(state, STEPS, os.path.join(CKPT_DIR, "ckpt_final.pkl"))
        print("[done]")


# ================================================================
# AWQ
# ================================================================
class AWQCalibrator:
    """
    Activation-aware weight scaling (input-channel).

    For each QDense kernel W (in, out):
      - collect per-input-channel activation magnitude m
      - set s = m**alpha, normalized to mean 1
      - scale W' = W * s[:, None]
    At inference, divide the activation by s before the matmul.
    """

    def __init__(self, alpha=AWQ_ALPHA):
        self.alpha = alpha

    def calibrate(self, ckpt, tokenizer, n_batches=AWQ_BATCHES,
                  batch_size=AWQ_BS, seq_len=None):
        cfg = ckpt["config"]
        seq_len = seq_len or cfg["training"]["seq_len"]
        m = cfg["model"]
        # capture=True -> remat auto-disabled inside Block
        model = LM(vocab=cfg["tokenizer"]["vocab_size"],
                   d_model=m["d_model"], n_layers=m["n_layers"],
                   max_len=seq_len, remat=False, capture=True)
        params = jax.tree_util.tree_map(jnp.asarray, ckpt["params"])

        print(f"[awq] calibrating on {n_batches} x {batch_size} "
              f"(seq={seq_len}, alpha={self.alpha})")
        mag = self._aggregate(model, params, tokenizer,
                              n_batches, batch_size, seq_len)
        scales = self._scales_from_mag(mag)
        print(f"[awq] {len(scales)} kernels scaled")
        return scales

    def _sample_texts(self, tokenizer, n):
        from datasets import load_dataset
        ds = load_dataset(DATA_NAME, DATA_CFG, split=DATA_SPLIT_TR,
                          streaming=True)
        ds = ds.shuffle(seed=1234, buffer_size=DATA_SHUFFLE)
        texts = []
        for ex in ds:
            t = ex.get("text", "")
            if len(t) >= DATA_MIN:
                texts.append(t[:2000])
                if len(texts) >= n:
                    break
        return texts

    def _aggregate(self, model, params, tokenizer,
                   n_batches, batch_size, seq_len):
        texts = self._sample_texts(tokenizer, n_batches * batch_size)
        agg = {}
        for b in range(n_batches):
            chunk = texts[b * batch_size:(b + 1) * batch_size]
            toks_np = np.zeros((len(chunk), seq_len), np.int32)
            for i, t in enumerate(chunk):
                ids = tokenizer.encode(t)[:seq_len]
                toks_np[i, :len(ids)] = ids
            toks = jnp.asarray(toks_np)
            _, inter = model.apply({"params": params}, toks,
                                   mutable=["intermediates"])
            for k, v in _flat(inter["intermediates"]).items():
                arr = np.asarray(v)
                if arr.ndim > 1:
                    arr = arr.mean(axis=0)
                agg.setdefault(k, np.zeros_like(arr))
                agg[k] = agg[k] + arr
        return {k: v / n_batches for k, v in agg.items()}

    def _scales_from_mag(self, mag):
        scales = {}
        for key, val in mag.items():
            if not key.endswith("/act_mag"):
                continue
            base = key[:-len("/act_mag")]
            kernel_key = base + "/kernel"
            m = np.maximum(val.astype(np.float32), 1e-6)
            s = np.power(m, self.alpha)
            s = s / s.mean()
            scales[kernel_key] = s.astype(np.float32)
        return scales

    @staticmethod
    def apply(params, scales):
        if not scales:
            return params

        def _scale(path, leaf):
            ps = "/".join(_pn(path))
            if ps in scales:
                s = jnp.asarray(scales[ps])
                if leaf.ndim == 2 and leaf.shape[0] == s.shape[0]:
                    return leaf * s[:, None]
            return leaf

        return jax.tree_util.tree_map_with_path(_scale, params)


# ================================================================
# SAMPLING
# ================================================================
def sample_main(prompt, n_new=200, temp=0.8, top_k=40):
    tok = TokenizerManager().load()
    model = LM(vocab=tok.get_vocab_size(), d_model=D_MODEL,
               n_layers=N_LAYERS, max_len=SEQ_LEN, remat=False)
    ck = load_ckpt(PP_INPUT)
    params = jax.tree_util.tree_map(jnp.asarray, ck["params"])
    awq = ck.get("awq_scales") or {}
    ids = tok.encode(prompt).ids
    eos = tok.token_to_id("</s>")
    if eos is None:
        eos = 2

    def _unscale(path, leaf):
        ps = "/".join(_pn(path))
        if ps in awq:
            s = jnp.asarray(awq[ps])
            if leaf.ndim == 2 and leaf.shape[0] == s.shape[0]:
                return leaf / s[:, None]
        return leaf

    fwd_params = jax.tree_util.tree_map_with_path(_unscale, params)
    fwd = jax.jit(lambda p, t: model.apply({"params": p}, t))
    for _ in range(n_new):
        window = ids[-SEQ_LEN:]
        arr = np.zeros((1, SEQ_LEN), np.int32)
        arr[0, :len(window)] = window
        logits = np.asarray(
            fwd(fwd_params, jnp.asarray(arr))[0, len(window) - 1], np.float64,
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
# POSTPROCESS
# ================================================================
class PostProcessor:
    def __init__(self, keep_ratio=PP_L_KEEP, awq_alpha=AWQ_ALPHA):
        self.keep_ratio = keep_ratio
        self.awq = AWQCalibrator(alpha=awq_alpha)

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

    def apply_awq(self, ckpt, tokenizer):
        if not AWQ_ENABLED:
            print("[awq] disabled in config; skipping")
            return ckpt
        scales = self.awq.calibrate(
            ckpt, tokenizer, n_batches=AWQ_BATCHES, batch_size=AWQ_BS,
        )
        out = dict(ckpt)
        out["params"] = self.awq.apply(ckpt["params"], scales)
        out["awq_scales"] = scales
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
            "awq_scales": ckpt.get("awq_scales", {}),
        }

        def _walk(path, leaf):
            keys = _pn(path)
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
        print(f"[export] {out_path}  ({os.path.getsize(out_path)/1e6:.1f} MB)  "
              f"awq={len(bundle['awq_scales'])} kernels")


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
    tok = TokenizerManager()
    tok.get_or_train()

    if action == "prune_layers":
        ckpt = pp.prune_layers(ckpt)
        _save(ckpt, output_path or os.path.join(PP_OUTPUT_DIR, "pruned.pkl"))
    elif action == "awq":
        ckpt = pp.apply_awq(ckpt, tok)
        _save(ckpt, output_path or os.path.join(PP_OUTPUT_DIR, "awq.pkl"))
    elif action == "export":
        pp.export_int8(ckpt, output_path or PP_EXPORT_PATH)
    elif action == "full":
        ckpt = pp.prune_layers(ckpt)
        _save(ckpt, os.path.join(PP_OUTPUT_DIR, "pruned.pkl"))
        ckpt = pp.apply_awq(ckpt, tok)
        _save(ckpt, os.path.join(PP_OUTPUT_DIR, "awq.pkl"))
        pp.export_int8(ckpt, PP_EXPORT_PATH)
        print("[full] done")


# ================================================================
# ENTRY
# ================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--action", default="train",
                    choices=["train", "tokenizer", "sample",
                             "prune_layers", "awq", "export", "full"])
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

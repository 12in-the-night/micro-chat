"""Convert micro-chat training checkpoint (pickle) directly to GGUF.

Key shape rules for llama.cpp Mamba-2 loader:
  blk.N.ssm_in.weight       (d_inner+2N,  d_model)          <- QDense_0 kernel .T
  blk.N.ssm_conv1d.weight   (d_conv,       d_inner+2N)      <- Conv_0 kernel squeeze(1), NO .T
  blk.N.ssm_out.weight      (d_model,      d_inner)         <- QDense_1 kernel .T
  blk.N.ssm_a               (1,             n_head)
  blk.N.ssm_d               (1,             n_head)
  blk.N.ssm_dt.bias         (n_head,)
  blk.N.ssm_norm.weight     (d_inner,)
  blk.N.attn_norm.weight    (d_model,)
  output_norm.weight        (d_model,)
  token_embd.weight         (vocab,         d_model)
  output.weight             (vocab,         d_model)
"""
import argparse, json, os, pickle, sys
import numpy as np
import gguf
from tokenizers import Tokenizer


# ---------------------------------------------------------------- helpers
def flatten_dict(d, parent_key='', sep='/'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def _expect(name, arr, expected, verbose=True):
    """Raise if arr.shape != expected. Also returns arr for chaining."""
    if tuple(arr.shape) != tuple(expected):
        raise ValueError(
            f"tensor {name}: expected shape {tuple(expected)}, "
            f"got {tuple(arr.shape)}")
    if verbose:
        print(f"  {name:32s} {tuple(arr.shape)}")
    return arr


def _try_add(w, method, *args):
    """Call w.<method>(*args) but tolerate it being absent in older gguf-py."""
    fn = getattr(w, method, None)
    if fn is None:
        print(f"  [warn] GGUFWriter has no {method}; skipping")
        return
    fn(*args)


# ---------------------------------------------------------------- main
def convert(ckpt_path, tok_path, out_path, verbose=True):
    with open(ckpt_path, "rb") as f:
        ck = pickle.load(f)
    params = flatten_dict(ck["params"], sep="/")
    cfg    = ck["config"]
    m, mb  = cfg["model"], cfg["mamba2"]

    d_model  = int(m["d_model"])
    n_layers = int(m["n_layers"])
    d_state  = int(mb["d_state"])
    d_conv   = int(mb["d_conv"])
    expand   = int(mb["expand"])
    head_dim = int(mb["head_dim"])
    d_inner  = expand * d_model
    n_head   = d_inner // head_dim
    n_group  = 1                       # matches Mamba2Block: one SSM group
    conv_ch  = d_inner + 2 * n_group * d_state

    vocab_size_ckpt = int(cfg["tokenizer"]["vocab_size"])

    if verbose:
        print(f"[cfg] d_model={d_model}  d_inner={d_inner}  n_head={n_head} "
              f" head_dim={head_dim}  d_state={d_state}  d_conv={d_conv}")
        print(f"[cfg] conv_ch={conv_ch}  n_layers={n_layers}  "
              f"vocab(ckpt)={vocab_size_ckpt}")

    # ---- tokenizer (needed before writer for vocab size / ids) -----------
    tok = Tokenizer.from_file(tok_path)
    with open(tok_path, encoding="utf-8") as f:
        tok_json = json.load(f)

    vocab = tok.get_vocab()                     # dict[str, int]
    tokens = [None] * len(vocab)
    for t, i in vocab.items():
        tokens[i] = t
    if any(t is None for t in tokens):
        raise RuntimeError("tokenizer vocab has gaps")
    if len(tokens) != vocab_size_ckpt:
        print(f"[warn] tokenizer size {len(tokens)} != ckpt vocab_size "
              f"{vocab_size_ckpt}  (using tokenizer)")

    # ---- writer ---------------------------------------------------------
    w = gguf.GGUFWriter(out_path, "mamba2")
    w.add_name("micro-chat")
    w.add_context_length(int(cfg["training"]["seq_len"]))
    w.add_embedding_length(d_model)
    w.add_block_count(n_layers)
    _try_add(w, "add_vocab_size", len(tokens))

    # hyperparams that llama.cpp's mamba2 loader reads
    w.add_uint32("mamba2.ssm.state_size", d_state)
    w.add_uint32("mamba2.ssm.conv_kernel", d_conv)
    w.add_uint32("mamba2.ssm.inner_size", d_inner)
    w.add_uint32("mamba2.ssm.group_count", n_group)
    w.add_uint32("mamba2.ssm.time_step_rank", n_head)
    w.add_uint32("mamba2.ssm.head_dim", head_dim)
    w.add_float32("mamba2.attention.layer_norm_rms_epsilon", 1e-6)

    # ---- tokenizer ------------------------------------------------------
    w.add_tokenizer_model("gpt2")
    w.add_string("tokenizer.ggml.pre", "default")
    w.add_token_list(tokens)

    special_types = {"<pad>": 3, "<s>": 3, "</s>": 3, "<unk>": 2}
    w.add_token_types([special_types.get(t, 1) for t in tokens])
    _try_add(w, "add_token_scores", [0.0] * len(tokens))

    merges = [" ".join(pair) for pair in tok_json["model"]["merges"]]
    w.add_token_merges(merges)

    bos_id = tok.token_to_id("<s>")
    eos_id = tok.token_to_id("</s>")
    w.add_bos_token_id(1 if bos_id is None else bos_id)
    w.add_eos_token_id(2 if eos_id is None else eos_id)

    # ---- tensors --------------------------------------------------------
    if verbose:
        print("[tensors]")

    emb = np.asarray(params["embedding"], np.float32)
    _expect("token_embd.weight", emb, (len(tokens), d_model), verbose)
    w.add_tensor("token_embd.weight", emb)

    for i in range(n_layers):
        p = f"Block_{i}/Mamba2Block_0/"

        # RMSNorm before the block (d_model)
        ln = np.asarray(params[p + "RMSNorm_0/scale"], np.float32).reshape(-1)
        _expect(f"blk.{i}.attn_norm.weight", ln, (d_model,), verbose)
        w.add_tensor(f"blk.{i}.attn_norm.weight", ln)

        # QDense_0: (d_model, d_inner+2N+n_head?) -- in this block it's
        # (d_model, 2*d_inner + 2N + n_head) in Flax, split into
        # z (d_inner), xBC (d_inner+2N), dt (n_head).  But to_gguf only
        # exports the *concat* kernel as ssm_in per the ggml mamba2 layout:
        #   ssm_in : (d_inner + 2*n_group*d_state,  d_model)
        # The Flax kernel is (d_model, 2*d_inner + 2N + n_head), so we can't
        # simply transpose.  In practice this project's QDense_0 emits
        # only d_inner + 2N columns for xBC and separate head for dt...  The
        # training code shows QDense(2*d_inner + 2N + n_head).  llama.cpp
        # mamba2 ssm_in expects `d_inner + 2*n_group*d_state` rows, so we
        # must slice the first (d_inner + 2N) columns of the kernel.
        k0 = np.asarray(params[p + "QDense_0/kernel"], np.float32)   # (d_model, cols)
        cols = k0.shape[1]
        if cols >= conv_ch:
            # take xBC slice, then transpose -> (conv_ch, d_model)
            ssm_in = k0[:, :conv_ch].T
        else:
            ssm_in = k0.T                                  # best effort
        _expect(f"blk.{i}.ssm_in.weight", ssm_in, (conv_ch, d_model), verbose)
        w.add_tensor(f"blk.{i}.ssm_in.weight", np.ascontiguousarray(ssm_in))

        # ---- conv1d: Flax depthwise Conv kernel is (d_conv, C, 1) ------
        conv = np.asarray(params[p + "Conv_0/kernel"], np.float32).squeeze(1)
        _expect(f"blk.{i}.ssm_conv1d.weight", conv, (d_conv, conv_ch), verbose)
        w.add_tensor(f"blk.{i}.ssm_conv1d.weight", np.ascontiguousarray(conv))

        cb = np.asarray(params[p + "Conv_0/bias"], np.float32).reshape(-1)
        _expect(f"blk.{i}.ssm_conv1d.bias", cb, (conv_ch,), verbose)
        w.add_tensor(f"blk.{i}.ssm_conv1d.bias", cb)

        # ---- dt bias: (n_head,) ---------------------------------------
        dtb = np.asarray(params[p + "dt_bias"], np.float32).reshape(-1)
        _expect(f"blk.{i}.ssm_dt.bias", dtb, (n_head,), verbose)
        w.add_tensor(f"blk.{i}.ssm_dt.bias", dtb)

        # ---- ssm_a : -exp(A_log), shape (1, n_head) -------------------
        a_log = np.asarray(params[p + "A_log"], np.float32).reshape(-1)
        ssm_a = (-np.exp(a_log)).reshape(1, -1)
        _expect(f"blk.{i}.ssm_a", ssm_a, (1, n_head), verbose)
        w.add_tensor(f"blk.{i}.ssm_a", ssm_a)

        # ---- ssm_d : (1, n_head) --------------------------------------
        ssm_d = np.asarray(params[p + "D"], np.float32).reshape(1, -1)
        _expect(f"blk.{i}.ssm_d", ssm_d, (1, n_head), verbose)
        w.add_tensor(f"blk.{i}.ssm_d", ssm_d)

        # ---- ssm_norm: RMSNorm over d_inner ---------------------------
        sn = np.asarray(params[p + "SsmRMSNorm/scale"], np.float32).reshape(-1)
        _expect(f"blk.{i}.ssm_norm.weight", sn, (d_inner,), verbose)
        w.add_tensor(f"blk.{i}.ssm_norm.weight", sn)

        # ---- ssm_out: QDense_1 kernel -> (d_model, d_inner) -----------
        so = np.asarray(params[p + "QDense_1/kernel"], np.float32).T
        _expect(f"blk.{i}.ssm_out.weight", so, (d_model, d_inner), verbose)
        w.add_tensor(f"blk.{i}.ssm_out.weight", np.ascontiguousarray(so))

    # ---- final norm + tied output --------------------------------------
    fn = np.asarray(params["RMSNorm_0/scale"], np.float32).reshape(-1)
    _expect("output_norm.weight", fn, (d_model,), verbose)
    w.add_tensor("output_norm.weight", fn)
    w.add_tensor("output.weight", emb)

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"[gguf] {out_path} ({os.path.getsize(out_path)/1e6:.1f} MB)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="./ckpts/ckpt_final.pkl")
    ap.add_argument("--tok",  default="./tokenizer.json")
    ap.add_argument("--out",  default="./micro-chat-f16.gguf")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    try:
        convert(a.ckpt, a.tok, a.out, verbose=not a.quiet)
    except Exception as e:
        print(f"[error] {e}", file=sys.stderr)
        sys.exit(1)

"""Convert a micro-chat training checkpoint (pickle) to a llama.cpp Mamba-2 GGUF (f16 matrices).

Convention: gguf-py reverses numpy shapes, so numpy shape = (ne1, ne0).
Expected llama.cpp tensors (example: d_model=192, d_inner=384, n_head=6, d_state=32, d_conv=4):
  token_embd.weight        ne=(d_model, vocab)          -> np (vocab, d_model)      f16
  blk.N.attn_norm.weight   ne=(d_model,)                -> np (d_model,)            f32
  blk.N.ssm_in.weight      ne=(d_model, 2*d_inner+2*d_state+n_head) -> np (838, d_model)  f16
  blk.N.ssm_conv1d.weight  ne=(d_conv, d_inner+2*d_state)           -> np (448, d_conv)   f32
  blk.N.ssm_conv1d.bias    ne=(conv_ch,)                -> np (conv_ch,)            f32
  blk.N.ssm_dt.bias        ne=(n_head,)                 -> np (n_head,)             f32
  blk.N.ssm_a              ne=(1, n_head)               -> np (n_head, 1)           f32
  blk.N.ssm_d              ne=(1, n_head)               -> np (n_head, 1)           f32
  blk.N.ssm_norm.weight    ne=(d_inner, n_group)        -> np (n_group, d_inner)    f32
  blk.N.ssm_out.weight     ne=(d_inner, d_model)        -> np (d_model, d_inner)    f16
  output_norm.weight       ne=(d_model,)                -> np (d_model,)            f32
  output.weight            only if untied (else llama.cpp reuses token_embd.weight)
"""
import argparse, json, os, pickle, sys
import numpy as np
import gguf


def flatten_dict(d, parent_key="", sep="/"):
    items = []
    for k, v in d.items():
        key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, key, sep).items())
        else:
            items.append((key, v))
    return dict(items)


def f32(a):
    return np.ascontiguousarray(np.asarray(a, np.float32))


def f16(a):
    return np.ascontiguousarray(np.asarray(a, np.float32).astype(np.float16))


def put(w, name, arr, expected_np_shape, verbose):
    if tuple(arr.shape) != tuple(expected_np_shape):
        raise ValueError(f"{name}: expected numpy shape {tuple(expected_np_shape)}, got {tuple(arr.shape)}")
    if verbose:
        print(f"  {name:30s} {str(arr.dtype):8s} np{tuple(arr.shape)}")
    w.add_tensor(name, arr)


def load_tokenizer(tok_path, n_rows):
    """tokens, token_types, merges, special-token ids from a HF `tokenizers` BPE json."""
    with open(tok_path, encoding="utf-8") as f:
        tj = json.load(f)
    vocab = dict(tj["model"]["vocab"])
    specials = {t["content"]: t["id"] for t in tj.get("added_tokens", [])}
    size = max([n_rows, *(i + 1 for i in vocab.values()), *(i + 1 for i in specials.values())])
    if size > n_rows:
        raise ValueError(f"tokenizer has {size} ids but embedding has only {n_rows} rows")
    tokens = [f"[PAD{i}]" for i in range(n_rows)]           # pad (unused) if BPE didn't fill the vocab
    types = [gguf.TokenType.UNUSED] * n_rows
    for tok, i in vocab.items():
        tokens[i], types[i] = tok, gguf.TokenType.NORMAL
    for tok, i in specials.items():
        tokens[i] = tok
        types[i] = gguf.TokenType.UNKNOWN if tok == "<unk>" else gguf.TokenType.CONTROL
    # merges are "a b" strings (tokenizers < 0.20) or [a, b] pairs (>= 0.20)
    merges = [m if isinstance(m, str) else " ".join(m) for m in tj["model"]["merges"]]
    return tokens, types, merges, specials


def convert(ckpt_path, tok_path, out_path, verbose=True):
    with open(ckpt_path, "rb") as f:
        ck = pickle.load(f)
    params = flatten_dict(ck["params"])
    cfg = ck["config"]
    m, mb = cfg["model"], cfg["mamba2"]

    d_model, n_layers = int(m["d_model"]), int(m["n_layers"])
    d_state, d_conv = int(mb["d_state"]), int(mb["d_conv"])
    head_dim, n_group = int(mb["head_dim"]), 1
    d_inner = int(mb["expand"]) * d_model
    n_head = d_inner // head_dim
    conv_ch = d_inner + 2 * n_group * d_state
    in_rows = 2 * d_inner + 2 * n_group * d_state + n_head
    tie = bool(m.get("tie_embeddings", True))
    n_rows = int(np.asarray(params["embedding"]).shape[0])

    if verbose:
        print(f"[cfg] d_model={d_model} d_inner={d_inner} n_head={n_head} d_state={d_state} "
              f"d_conv={d_conv} conv_ch={conv_ch} in_rows={in_rows} layers={n_layers} vocab={n_rows} tied={tie}")

    tokens, types, merges, sp = load_tokenizer(tok_path, n_rows)

    w = gguf.GGUFWriter(out_path, "mamba2")
    w.add_name("micro-chat")
    w.add_context_length(int(cfg["training"]["seq_len"]))
    w.add_embedding_length(d_model)
    w.add_feed_forward_length(0)
    w.add_head_count(0)
    w.add_block_count(n_layers)
    w.add_layer_norm_rms_eps(1e-6)
    w.add_ssm_conv_kernel(d_conv)
    w.add_ssm_inner_size(d_inner)
    w.add_ssm_state_size(d_state)
    w.add_ssm_time_step_rank(n_head)
    w.add_ssm_group_count(n_group)
    w.add_file_type(gguf.LlamaFileType.MOSTLY_F16)

    # ByteLevel BPE + GPT-2 regex (see TokenizerManager in main.py) == llama.cpp "gpt2" / pre "gpt-2"
    w.add_tokenizer_model("gpt2")
    w.add_tokenizer_pre("gpt-2")
    w.add_token_list(tokens)
    w.add_token_types(types)
    w.add_token_merges(merges)
    for name, setter in (("<s>", w.add_bos_token_id), ("</s>", w.add_eos_token_id),
                         ("<unk>", w.add_unk_token_id), ("<pad>", w.add_pad_token_id)):
        if name in sp:
            setter(sp[name])
    w.add_add_bos_token(False)   # training never prepends BOS; documents are separated by </s>
    w.add_add_eos_token(False)

    if verbose:
        print("[tensors]")
    put(w, "token_embd.weight", f16(params["embedding"]), (n_rows, d_model), verbose)

    for i in range(n_layers):
        p = f"Block_{i}/Mamba2Block_0/"
        put(w, f"blk.{i}.attn_norm.weight", f32(params[p + "RMSNorm_0/scale"]).reshape(-1), (d_model,), verbose)
        put(w, f"blk.{i}.ssm_in.weight", f16(np.asarray(params[p + "QDense_0/kernel"]).T), (in_rows, d_model), verbose)
        conv = np.asarray(params[p + "Conv_0/kernel"], np.float32).squeeze(1)        # (d_conv, C)
        put(w, f"blk.{i}.ssm_conv1d.weight", f32(conv.T), (conv_ch, d_conv), verbose)
        put(w, f"blk.{i}.ssm_conv1d.bias", f32(params[p + "Conv_0/bias"]).reshape(-1), (conv_ch,), verbose)
        put(w, f"blk.{i}.ssm_dt.bias", f32(params[p + "dt_bias"]).reshape(-1), (n_head,), verbose)
        a = -np.exp(np.asarray(params[p + "A_log"], np.float32))
        put(w, f"blk.{i}.ssm_a", f32(a).reshape(n_head, 1), (n_head, 1), verbose)    # ne = (1, n_head)
        put(w, f"blk.{i}.ssm_d", f32(params[p + "D"]).reshape(n_head, 1), (n_head, 1), verbose)
        put(w, f"blk.{i}.ssm_norm.weight", f32(params[p + "SsmRMSNorm/scale"]).reshape(n_group, d_inner),
            (n_group, d_inner), verbose)
        put(w, f"blk.{i}.ssm_out.weight", f16(np.asarray(params[p + "QDense_1/kernel"]).T), (d_model, d_inner), verbose)

    put(w, "output_norm.weight", f32(params["RMSNorm_0/scale"]).reshape(-1), (d_model,), verbose)
    if not tie:
        put(w, "output.weight", f16(params["lm_head"]), (n_rows, d_model), verbose)

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"[gguf] {out_path} ({os.path.getsize(out_path)/1e6:.1f} MB)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="./ckpts/ckpt_final.pkl")
    ap.add_argument("--tok", default="./tokenizer.json")
    ap.add_argument("--out", default="./micro-chat-f16.gguf")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    try:
        convert(a.ckpt, a.tok, a.out, verbose=not a.quiet)
    except Exception as e:
        print(f"[error] {e}", file=sys.stderr)
        sys.exit(1)

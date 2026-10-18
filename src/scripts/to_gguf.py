"""Convert micro-chat training checkpoint (pickle) directly to GGUF."""
import argparse, json, os, pickle
import numpy as np
import gguf
from tokenizers import Tokenizer


def flatten_dict(d, parent_key='', sep='/'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def convert(ckpt_path, tok_path, out_path):
    with open(ckpt_path, "rb") as f:
        ck = pickle.load(f)
    params = flatten_dict(ck["params"], sep="/")
    cfg = ck["config"]
    m, mb = cfg["model"], cfg["mamba2"]

    w = gguf.GGUFWriter(out_path, "mamba2")
    w.add_name("micro-chat")
    w.add_context_length(cfg["training"]["seq_len"])
    w.add_embedding_length(m["d_model"])
    w.add_block_count(m["n_layers"])
    w.add_vocab_size(cfg["tokenizer"]["vocab_size"])

    # --- HYPERPARAMETERS ---
    w.add_uint32("mamba2.ssm.state_size", mb["d_state"])
    w.add_uint32("mamba2.ssm.conv_kernel", mb["d_conv"])
    w.add_uint32("mamba2.ssm.inner_size", mb["expand"] * m["d_model"])
    w.add_uint32("mamba2.ssm.group_count", 1)
    w.add_uint32("mamba2.ssm.time_step_rank",
                 mb["expand"] * m["d_model"] // mb["head_dim"])
    w.add_uint32("mamba2.ssm.head_dim", mb["head_dim"])
    w.add_float32("mamba2.attention.layer_norm_rms_epsilon", 1e-6)

    # --- TOKENIZER ---
    tok = Tokenizer.from_file(tok_path)
    with open(tok_path, encoding="utf-8") as f:
        tok_json = json.load(f)

    w.add_tokenizer_model("gpt2")
    w.add_string("tokenizer.ggml.pre", "default")

    vocab = tok.get_vocab()
    tokens = [None] * len(vocab)
    for k, i in vocab.items():
        tokens[i] = k
    w.add_token_list(tokens)

    special_types = {"<pad>": 3, "<s>": 3, "</s>": 3, "<unk>": 2}
    token_types = [special_types.get(t, 1) for t in tokens]
    w.add_token_types(token_types)

    merges = [" ".join(pair) for pair in tok_json["model"]["merges"]]
    w.add_token_merges(merges)

    w.add_bos_token_id(tok.token_to_id("<s>") or 1)
    w.add_eos_token_id(tok.token_to_id("</s>") or 2)

    # --- TENSORS ---
    emb = np.asarray(params["embedding"], np.float32)
    w.add_tensor("token_embd.weight", emb)

    for i in range(m["n_layers"]):
        p = f"Block_{i}/Mamba2Block_0/"

        w.add_tensor(f"blk.{i}.attn_norm.weight",
                     np.asarray(params[p + "RMSNorm_0/scale"], np.float32))

        w.add_tensor(f"blk.{i}.ssm_in.weight",
                     np.asarray(params[p + "QDense_0/kernel"], np.float32).T)

        conv = np.asarray(params[p + "Conv_0/kernel"], np.float32).squeeze(1)
        w.add_tensor(f"blk.{i}.ssm_conv1d.weight", conv.T)
        w.add_tensor(f"blk.{i}.ssm_conv1d.bias",
                     np.asarray(params[p + "Conv_0/bias"], np.float32))

        w.add_tensor(f"blk.{i}.ssm_dt.bias",
                     np.asarray(params[p + "dt_bias"], np.float32))

        # FIX: ssm_a must be 2D (1, n_heads)
        a_log = np.asarray(params[p + "A_log"], np.float32).reshape(-1)
        ssm_a = (-np.exp(a_log)).reshape(1, -1)
        if i == 0: print(f"  DEBUG ssma_shape={ssm_a.shape}")
        w.add_tensor(f"blk.{i}.ssm_a", ssm_a)

        # FIX: ssm_d must be 2D (1, n_heads)
        ssm_d = np.asarray(params[p + "D"], np.float32).reshape(-1).reshape(1, -1)
        w.add_tensor(f"blk.{i}.ssm_d", ssm_d)

        w.add_tensor(f"blk.{i}.ssm_norm.weight",
                     np.asarray(params[p + "SsmRMSNorm/scale"], np.float32))

        w.add_tensor(f"blk.{i}.ssm_out.weight",
                     np.asarray(params[p + "QDense_1/kernel"], np.float32).T)

    w.add_tensor("output_norm.weight",
                 np.asarray(params["RMSNorm_0/scale"], np.float32))
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
    a = ap.parse_args()
    convert(a.ckpt, a.tok, a.out)

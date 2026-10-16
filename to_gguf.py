"""Convert micro-chat training checkpoint (pickle) directly to GGUF f16."""
import argparse, json, os, pickle
import numpy as np
import gguf
from flax.traverse_util import flatten_dict
from tokenizers import Tokenizer


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
    w.add_uint32("mamba2.ssm.state_size", mb["d_state"])
    w.add_uint32("mamba2.ssm.conv_kernel", mb["d_conv"])
    w.add_uint32("mamba2.ssm.inner_size", mb["expand"] * m["d_model"])
    w.add_uint32("mamba2.ssm.n_groups", 1)
    w.add_uint32("mamba2.ssm.time_step_rank",
                 mb["expand"] * m["d_model"] // mb["head_dim"])

    tok = Tokenizer.from_file(tok_path)
    vocab = tok.get_vocab()
    tokens = [None] * len(vocab)
    for k, i in vocab.items():
        tokens[i] = k
    w.add_token_list(tokens)
    w.add_bos_token_id(tok.token_to_id("<s>") or 1)
    w.add_eos_token_id(tok.token_to_id("</s>") or 2)

    emb = np.asarray(params["Embed_0/embedding"], np.float32)
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
        w.add_tensor(f"blk.{i}.ssm_a",
                     -np.exp(np.asarray(params[p + "A_log"], np.float32)))
        w.add_tensor(f"blk.{i}.ssm_d",
                     np.asarray(params[p + "D"], np.float32))
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

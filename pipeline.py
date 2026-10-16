#!/usr/bin/env python3
"""
micro-chat end-to-end pipeline.

Clone repo -> install deps -> train tokenizer -> train model ->
export GGUF -> quantize -> upload to HF.

Usage:
    python pipeline.py                 # full pipeline
    python pipeline.py --skip-clone    # repo already present
    python pipeline.py --skip-install  # deps already installed
    python pipeline.py --no-upload     # skip HF push
    python pipeline.py --no-quantize   # only f16 GGUF
    python pipeline.py --resume PATH   # resume training from checkpoint

Env:
    HF_TOKEN   HuggingFace write token (for --upload)
    HF_REPO    repo id, default "rgdfbd/micro-chat-pi"
    WORKDIR    working dir, default "/kaggle/working"
"""
import argparse
import os
import shutil
import subprocess
import sys
import time

# ----------------------------------------------------------------
# Config
# ----------------------------------------------------------------
REPO_URL = "https://github.com/12in-the-night/micro-chat.git"
REPO_NAME = "micro-chat"
DEFAULT_HF_REPO = os.environ.get("HF_REPO", "rgdfbd/micro-chat-pi")
WORKDIR = os.environ.get("WORKDIR", "/kaggle/working")
ROOT = os.path.join(WORKDIR, REPO_NAME)

os.environ["PYTHONUNBUFFERED"] = "1"
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.9")

T0 = time.time()


def log(msg):
    print(f"[{time.time()-T0:7.1f}s] {msg}", flush=True)


def run(args, cwd=None, check=True):
    """Run subprocess, stream stdout+stderr line by line."""
    log("run: " + " ".join(str(a) for a in args))
    p = subprocess.Popen(
        [str(a) for a in args], cwd=cwd,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    for line in p.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
    rc = p.wait()
    if check and rc != 0:
        raise SystemExit(f"exit {rc}: {' '.join(str(a) for a in args)}")
    return rc


# ----------------------------------------------------------------
# Steps
# ----------------------------------------------------------------
def step_clone(force=True):
    os.chdir(WORKDIR)
    if os.path.isdir(ROOT):
        if force:
            log(f"removing {ROOT}")
            shutil.rmtree(ROOT)
        else:
            log(f"{ROOT} exists, skipping clone")
            os.chdir(ROOT)
            return
    run(["git", "clone", REPO_URL])
    os.chdir(ROOT)
    log("clone done")


def step_install():
    os.chdir(ROOT)
    run([sys.executable, "-m", "pip", "install", "-U", "jax[cuda12]"])
    run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    import jax
    devs = jax.devices()
    log(f"jax devices: {devs}")
    assert any(d.platform in ("gpu", "cuda") for d in devs), \
        f"no GPU found: {devs}"


def step_tokenizer():
    os.chdir(ROOT)
    run([sys.executable, "-u", "main.py", "--action", "tokenizer"])


def step_train(resume=None):
    os.chdir(ROOT)
    args = [sys.executable, "-u", "main.py"]
    if resume:
        args += ["--resume", resume]
    run(args)


def step_gguf(out_name="micro-chat-f16.gguf"):
    os.chdir(ROOT)
    run([sys.executable, "-u", "to_gguf.py",
         "--ckpt", "./ckpts/ckpt_final.pkl",
         "--tok",  "./tokenizer.json",
         "--out",  f"./{out_name}"])
    return os.path.join(ROOT, out_name)


def step_quantize(f16_path, out_name="micro-chat-q8_0.gguf",
                  quant="Q8_0"):
    """Build llama.cpp and quantize. CUDA build if available."""
    llama_dir = os.path.join(WORKDIR, "llama.cpp")
    if not os.path.isdir(llama_dir):
        os.chdir(WORKDIR)
        run(["git", "clone", "--depth", "1",
             "https://github.com/ggml-org/llama.cpp", llama_dir])
    build = os.path.join(llama_dir, "build")
    if not os.path.isdir(build):
        run(["cmake", "-S", llama_dir, "-B", build,
             "-DGGML_CUDA=ON", "-DCMAKE_BUILD_TYPE=Release"])
        run(["cmake", "--build", build, "-j", "4"])
    quant_bin = os.path.join(build, "bin", "llama-quantize")
    out_path = os.path.join(ROOT, out_name)
    run([quant_bin, f16_path, out_path, quant])
    return out_path


def step_upload(hf_repo, names):
    token = os.environ.get("HF_TOKEN")
    if not token:
        try:
            from kaggle_secrets import UserSecretsClient
            token = UserSecretsClient().get_secret("HF_TOKEN")
        except Exception:
            log("no HF_TOKEN, skipping upload")
            return
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    api.create_repo(hf_repo, repo_type="model", exist_ok=True)
    for name in names:
        fp = os.path.join(ROOT, name)
        if not os.path.exists(fp):
            log(f"missing: {name}")
            continue
        size = os.path.getsize(fp) / 1e6
        log(f"upload {name} ({size:.1f} MB)")
        api.upload_file(fp, name, hf_repo, repo_type="model")
    log(f"https://huggingface.co/{hf_repo}")


# ----------------------------------------------------------------
# Main
# ----------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-clone",   action="store_true")
    ap.add_argument("--skip-install", action="store_true")
    ap.add_argument("--skip-train",   action="store_true")
    ap.add_argument("--no-quantize",  action="store_true")
    ap.add_argument("--no-upload",    action="store_true")
    ap.add_argument("--resume", default=None,
                    help="path to ckpt_*.pkl to resume from")
    ap.add_argument("--hf-repo", default=DEFAULT_HF_REPO)
    ap.add_argument("--quant", default="Q8_0")
    args = ap.parse_args()

    if not args.skip_clone:
        step_clone()
    else:
        os.chdir(ROOT)

    if not args.skip_install:
        step_install()

    if not args.skip_train:
        if not os.path.exists(os.path.join(ROOT, "tokenizer.json")):
            step_tokenizer()
        step_train(resume=args.resume)

    f16 = step_gguf()
    uploads = ["micro-chat-f16.gguf", "tokenizer.json", "config.yaml"]

    if not args.no_quantize:
        q = step_quantize(f16, quant=args.quant)
        uploads.append(os.path.basename(q))

    if not args.no_upload:
        step_upload(args.hf_repo, uploads)

    log("done")


if __name__ == "__main__":
    main()

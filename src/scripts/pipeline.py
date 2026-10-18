#!/usr/bin/env python3
"""micro-chat end-to-end: train -> GGUF -> quantize -> upload."""
import argparse, os, shutil, subprocess, sys, time

REPO_URL = "https://github.com/12in-the-night/micro-chat.git"
REPO_NAME = "micro-chat"
WORKDIR = os.environ.get("WORKDIR", "/kaggle/working")
DEFAULT_HF_REPO = os.environ.get("HF_REPO", "rgdfbd/micro-chat-pi")

os.environ["PYTHONUNBUFFERED"] = "1"
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.9")
os.environ.setdefault("NCCL_P2P_DISABLE", "1")

T0 = time.time()

def log(m): print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)

def run(args, cwd=None, check=True):
    log("$ " + " ".join(map(str, args)))
    p = subprocess.Popen([str(a) for a in args], cwd=cwd,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, bufsize=1)
    for line in p.stdout:
        sys.stdout.write(line); sys.stdout.flush()
    rc = p.wait()
    if check and rc != 0:
        raise SystemExit(f"exit {rc}: {' '.join(map(str, args))}")
    return rc

def locate_repo():
    here = os.path.dirname(os.path.abspath(__file__))
    # If running from src/scripts/
    if os.path.isfile(os.path.join(here, "..", "main.py")):
        root = os.path.abspath(os.path.join(here, "..", ".."))
        log(f"using repo at {root}")
        return root
    # Fallback to working dir
    root = os.path.join(WORKDIR, REPO_NAME)
    if not os.path.isdir(root):
        os.chdir(WORKDIR)
        run(["git", "clone", REPO_URL])
    else:
        log(f"using existing {root}")
    return root

def get_hf_token():
    for key in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HF_HUB_TOKEN"):
        v = os.environ.get(key)
        if v:
            log(f"token from env {key}")
            return v
    try:
        from kaggle_secrets import UserSecretsClient
        v = UserSecretsClient().get_secret("HF_TOKEN")
        log("token from kaggle secret HF_TOKEN")
        return v
    except Exception as e:
        log(f"kaggle secret failed: {e}")
    return None

def step_install(root):
    os.chdir(root)
    run([sys.executable, "-m", "pip", "install","-v", "-U", "jax[cuda12]"])
    run([sys.executable, "-m", "pip", "install","-v", "-r", "requirements.txt"])
    import jax
    devs = jax.devices()
    log(f"jax: {devs}")
    assert any(d.platform in ("gpu", "cuda") for d in devs), f"no GPU: {devs}"

def step_tokenizer(root):
    os.chdir(os.path.join(root, "src"))
    run([sys.executable, "-u", "main.py", "--action", "tokenizer"])

def step_train(root, resume=None):
    os.chdir(os.path.join(root, "src"))
    a = [sys.executable, "-u", "main.py"]
    if resume:
        a += ["--resume", resume]
    run(a)

def step_gguf(root):
    os.chdir(os.path.join(root, "src"))
    out_path = os.path.join(root, "micro-chat-f16.gguf")
    run([sys.executable, "-u", "scripts/to_gguf.py", "--out", out_path])
    return out_path

def step_quantize(root, f16, quant="Q8_0"):
    llama = os.path.join(WORKDIR, "llama.cpp")
    if not os.path.isdir(llama):
        os.chdir(WORKDIR)
        run(["git", "clone", "--depth", "1", "https://github.com/ggml-org/llama.cpp", llama])
    build = os.path.join(llama, "build")
    if not os.path.isdir(build):
        run(["cmake", "-S", llama, "-B", build, "-DGGML_CUDA=OFF", "-DCMAKE_BUILD_TYPE=Release"])
        run(["cmake", "--build", build, "-j", "4", "--target", "llama-quantize"])
    out = os.path.join(root, f"micro-chat-{quant.lower()}.gguf")
    run([os.path.join(build, "bin", "llama-quantize"), f16, out, quant])
    return out

def step_upload(root, hf_repo, names):
    token = get_hf_token()
    if not token:
        log("no HF token, skip upload")
        return
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    who = api.whoami()
    log(f"hf user: {who.get('name')}")
    api.create_repo(hf_repo, repo_type="model", exist_ok=True)
    for name in names:
        fp = os.path.join(root, name)
        if not os.path.exists(fp):
            log(f"missing {name}"); continue
        log(f"upload {name} ({os.path.getsize(fp)/1e6:.1f} MB)")
        api.upload_file(path_or_fileobj=fp, path_in_repo=name, repo_id=hf_repo, repo_type="model")
    log(f"https://huggingface.co/{hf_repo}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-install", action="store_true")
    ap.add_argument("--skip-train",   action="store_true")
    ap.add_argument("--no-quantize",  action="store_true")
    ap.add_argument("--no-upload",    action="store_true")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--hf-repo", default=DEFAULT_HF_REPO)
    ap.add_argument("--quant", default="Q8_0")
    ap.add_argument("--both-gpus", action="store_true")
    args = ap.parse_args()
    if not args.both_gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    root = locate_repo()
    os.chdir(root)

    if not args.skip_install:
        step_install(root)

    if not args.skip_train:
        if not os.path.exists(os.path.join(root, "src", "tokenizer.json")):
            step_tokenizer(root)
        step_train(root, args.resume)

    f16 = step_gguf(root)
    uploads = ["micro-chat-f16.gguf", "src/tokenizer.json", "src/config.yaml"]

    if not args.no_quantize:
        q = step_quantize(root, f16, args.quant)
        uploads.append(os.path.basename(q))

    if not args.no_upload:
        step_upload(root, args.hf_repo, uploads)

    log("done")

if __name__ == "__main__":
    main()

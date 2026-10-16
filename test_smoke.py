"""
Offline tests (no network, CPU). Run:  python -m pytest -q tests

Uses 2 virtual CPU devices so the data-parallel (sharded) path is exercised.
"""
import os
import sys

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"
os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dataclasses import asdict, replace  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import optax  # noqa: E402
import pytest  # noqa: E402
import yaml  # noqa: E402

import infer  # noqa: E402
import main as M  # noqa: E402

WORDS = ("the cat sat on a mat . a dog ran to the park . tom had a red ball . "
         "she was happy and he was sad . once upon a time there was a bird .").split()


def fake_source(cfg, split, seed=0, shuffle=True):
    def texts(epoch=0):
        rng = np.random.default_rng(seed * 1000 + epoch)
        for _ in range(400):
            n = int(rng.integers(20, 60))
            yield " ".join(rng.choice(WORDS, n))
    return texts


@pytest.fixture()
def tiny(tmp_path, monkeypatch):
    monkeypatch.setattr(M, "make_text_source", fake_source)
    raw = {
        "tokenizer": {"vocab_size": 200, "path": str(tmp_path / "tok.json"),
                      "train_corpus_mb": 1},
        "data": {"split_train": "train", "split_val": "validation"},
        "training": {"seq_len": 64, "batch": 4, "grad_accum": 2, "steps": 40,
                     "lr": 1e-2, "warmup": 2, "log_every": 10, "eval_every": 20,
                     "eval_batch": 4, "ckpt_every": 20, "precision": "fp32",
                     "ckpt_dir": str(tmp_path / "ck"), "remat": True},
        "model": {"d_model": 32, "n_layers": 4},
        "mamba2": {"d_state": 8, "d_conv": 4, "chunk": 16, "expand": 2, "head_dim": 16},
        "awq": {"enabled": True, "alpha": 0.5, "calib_batches": 2, "calib_batch_size": 4},
        "postprocess": {"input_ckpt": str(tmp_path / "ck" / "ckpt_final.pkl"),
                        "output_dir": str(tmp_path / "cc"), "layer_keep_ratio": 0.5,
                        "export_path": str(tmp_path / "pi.npz")},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw))
    return M.Config(raw), raw, path


def test_two_devices_visible():
    assert len(jax.devices()) == 2


def test_ssd_matches_sequential_recurrence():
    rng = np.random.default_rng(0)
    b, L, H, Pd, N, chunk = 2, 37, 3, 5, 4, 16        # L not a multiple of chunk
    x = rng.normal(size=(b, L, H, Pd)).astype(np.float32)
    dt = rng.uniform(0.01, 0.5, (b, L, H)).astype(np.float32)
    A = -rng.uniform(1, 8, H).astype(np.float32)
    Bm = rng.normal(size=(b, L, N)).astype(np.float32)
    Cm = rng.normal(size=(b, L, N)).astype(np.float32)
    D = rng.normal(size=H).astype(np.float32)

    ref = np.zeros((b, L, H, Pd), np.float32)
    S = np.zeros((b, H, Pd, N), np.float32)
    for t in range(L):
        S = (np.exp(A * dt[:, t])[..., None, None] * S
             + (dt[:, t][..., None] * x[:, t])[..., None] * Bm[:, t][:, None, None, :])
        ref[:, t] = np.einsum("bhpn,bn->bhp", S, Cm[:, t]) + D[None, :, None] * x[:, t]

    out = M.ssd(*map(jnp.asarray, (x, dt, A, Bm, Cm, D)), chunk)
    np.testing.assert_allclose(np.asarray(out), ref, rtol=1e-4, atol=1e-4)


def test_optimizer_descends_and_params_use_correct_sign():
    cfg = M.Config({"training": {"steps": 100, "lr": 1e-2, "warmup": 1,
                                 "weight_decay": 0.1}})
    p = {"w": jnp.full((8,), 0.5)}
    tx = M.build_optimizer(cfg, p)
    s = tx.init(p)
    f = lambda q: jnp.sum((q["w"] - 1.0) ** 2)  # noqa: E731
    start = float(f(p))
    for _ in range(100):
        u, s = tx.update(jax.grad(f)(p), s, p)
        p = optax.apply_updates(p, u)
    assert float(f(p)) < 0.1 * start


def test_causality_and_pad_invariance(tiny):
    cfg, *_ = tiny
    mc = M.ModelCfg.from_config(cfg, 200)
    model = M.LM(mc)
    toks = jnp.asarray(np.random.default_rng(1).integers(0, 200, (1, 40)), jnp.int32)
    params = model.init(jax.random.PRNGKey(0), toks)["params"]
    full = model.apply({"params": params}, toks)
    pref = model.apply({"params": params}, toks[:, :23])
    np.testing.assert_allclose(np.asarray(full[:, :23]), np.asarray(pref),
                               rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("tie", [True, False])
@pytest.mark.parametrize("use_awq", [False, True])
def test_numpy_inference_matches_jax(tiny, tmp_path, use_awq, tie):
    """Exported int8 (+AWQ) bundle run by infer.py == JAX model on simulated weights."""
    cfg, raw, _ = tiny
    tok = M.TokenizerManager(cfg)
    tok.train()
    mc = M.ModelCfg.from_config(cfg, tok.vocab_size)
    mc = replace(mc, tie_embeddings=tie)
    model = M.LM(mc, quantize=False)
    ids = np.random.default_rng(2).integers(4, tok.vocab_size, 24).astype(np.int32)
    params = jax.tree_util.tree_map(
        np.asarray, model.init(jax.random.PRNGKey(0), jnp.asarray(ids[None]))["params"])
    # larger embeddings so logits are not ~0 and the comparison is meaningful
    params["Embed_0"]["embedding"] = params["Embed_0"]["embedding"] * 20

    scales = {}
    if use_awq:
        rng = np.random.default_rng(3)
        for name, w in M.flatten_dict(params, sep="/").items():
            if name.endswith("QDense_0/kernel") or name.endswith("QDense_1/kernel"):
                scales[name] = rng.uniform(0.5, 2.0, w.shape[0]).astype(np.float32)

    ck = {"params": params, "model": asdict(mc), "config": raw}
    out = str(tmp_path / "m.npz")
    M.PostProcessor.export_int8(ck, scales, out)

    sim = M.simulate_export(params, scales)
    ref = np.asarray(model.apply({"params": sim}, jnp.asarray(ids[None]))[0])

    pi = infer.PiModel(out, raw["tokenizer"]["path"])
    got = np.stack([pi.step(int(t)) for t in ids])
    assert len(pi.bundle.awq) == len(scales)
    assert ("lm_head/kernel" in pi.bundle.params) == (not tie)
    np.testing.assert_allclose(got, ref, rtol=2e-3, atol=2e-3)


@pytest.mark.parametrize("precision", ["fp32", "fp16"])
def test_train_resume_postprocess_infer(tiny, tmp_path, precision, capsys):
    cfg, raw, path = tiny
    cfg.raw["training"]["precision"] = precision

    tr = M.Trainer(cfg)
    tr.run()
    ck_dir = tmp_path / "ck"
    final = ck_dir / "ckpt_final.pkl"
    assert final.exists()
    out = capsys.readouterr().out
    assert "[eval] step     20" in out and "skipped" in out
    ck = M.load_ckpt(str(final))
    assert ck["step"] == 40 and ck["version"] == M.CKPT_VERSION
    # loss must have dropped on the toy language
    nlls = [float(l.split("nll")[1].split("|")[0]) for l in out.splitlines()
            if l.startswith("step")]
    assert nlls[-1] < nlls[0] - 0.8, nlls

    # resume from the mid checkpoint
    mid = sorted(ck_dir.glob("ckpt_0*.pkl"))
    assert mid, "periodic checkpoint missing"
    cfg.raw["resume"] = str(mid[-1])
    cfg.raw["training"]["steps"] = 44
    M.Trainer(cfg).run()
    assert M.load_ckpt(str(final))["step"] == 44

    # full post-processing pipeline
    M.postprocess_main(cfg, "full", None, None)
    npz = tmp_path / "pi.npz"
    assert npz.exists()
    pi = infer.PiModel(str(npz), raw["tokenizer"]["path"])
    assert pi.bundle.n_layers == 2                     # 4 * 0.5
    ids, n_gen, _, _ = pi.generate("the cat", n_new=8, seed=0)
    assert n_gen > 0 and ids[0] == pi.eos_id

    # pruned ckpt + --input path works for sample/eval
    M.postprocess_main(cfg, "eval", str(tmp_path / "cc" / "awq.pkl"), None)
    M.sample_main(cfg, str(tmp_path / "cc" / "pruned.pkl"), "the cat", 5, 0.8, 10, 0)


def test_npz_has_no_pickle(tiny, tmp_path):
    cfg, raw, _ = tiny
    tok = M.TokenizerManager(cfg)
    tok.train()
    mc = M.ModelCfg.from_config(cfg, tok.vocab_size)
    model = M.LM(mc)
    p = jax.tree_util.tree_map(np.asarray, model.init(
        jax.random.PRNGKey(0), jnp.zeros((1, 16), jnp.int32))["params"])
    M.PostProcessor.export_int8({"params": p, "model": asdict(mc), "config": raw},
                                {}, str(tmp_path / "x.npz"))
    with np.load(tmp_path / "x.npz", allow_pickle=False) as z:  # raises if pickled
        assert all(z[k].dtype != object for k in z.files)

# micro-chat

A 12M-parameter dense **Mamba-2** language model trained on
[TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories).
Trains on a GPU with int8 quantisation-aware training, is pruned and AWQ-scaled,
then runs on a **Raspberry Pi** with nothing but NumPy (~40 MB of weights in RAM).

| File | Purpose |
|---|---|
| `config.yaml` | All hyper-parameters |
| `main.py` | Tokenizer, training, eval, prune / AWQ / export, JAX sampling |
| `infer.py` | Pi inference, NumPy + `tokenizers` only |
| `requirements.txt` | Training deps (GPU) |
| `pi-need.txt` | Inference deps (Pi) |
| `tests/` | Offline tests (CPU, no network) |

## Training (GPU, CUDA 12)

```bash
python -m venv venv && source venv/bin/activate
pip install -U "jax[cuda12]"
pip install -r requirements.txt
python -c "import jax; print(jax.devices())"     # should list your GPU(s)

python main.py --action tokenizer                # ~5 min, 8k byte-level BPE
python main.py                                   # train (resume: set `resume:` in config.yaml)
python main.py --action sample --prompt "Once upon a time"
python main.py --action full                     # prune -> AWQ -> export
```

* Multi-GPU is automatic (data parallel via `jit` + sharding). `training.batch`
  and `training.eval_batch` must be divisible by the number of GPUs.
* Precision is chosen from the GPU: fp16 **with dynamic loss scaling** on T4/V100,
  bf16 on A100/H100, fp32 on CPU. Override with `training.precision`.
* `Ctrl-C` writes `ckpts/ckpt_interrupted.pkl`; point `resume:` at it to continue.

### What `--action full` does

1. Evaluates the trained model on the validation set.
2. **Prunes** 25 % of the layers (`prune_method: influence` removes the blocks that
   change the residual stream least; `tail` drops the last ones).
3. Calibrates **AWQ** scales and compares validation NLL of the int8 model with and
   without them. AWQ is exported only if it helps.
4. Writes `pi_model.npz` (int8 weights + float32 norms/SSM params + AWQ scales).

Every step prints validation NLL/perplexity, so you can see what each costs.
Pruning without any re-training does lose quality; if the drop is too large, lower
`layer_keep_ratio` or fine-tune the pruned checkpoint.

`python main.py --action eval --input ckpts_compressed/awq.pkl` re-runs the comparison.

## Inference on the Pi

```bash
sudo apt-get install -y python3-pip python3-venv libopenblas-dev
python3 -m venv ~/lm_env && source ~/lm_env/bin/activate
pip install -r requirements-pi.txt

# copy pi_model.npz, tokenizer.json and infer.py to the Pi, then:
python infer.py --prompt "Once upon a time" --n 100 --temp 0.8 --top-k 40
```

The bundle is a plain `.npz` loaded with `allow_pickle=False`, so a downloaded
model cannot execute code. (`ckpts/*.pkl` training checkpoints *are* pickles:
only load ones you created yourself.)

## Tests

```bash
python -m pytest -q tests
```

Covers: chunked SSD scan vs. the step-by-step recurrence, causality, optimiser
direction, JAX vs. NumPy logits for the exported int8 (+AWQ) bundle, and an
end-to-end train -> resume -> prune -> AWQ -> export -> generate run on two
virtual devices (fp32 and fp16).

## License

GPL-3.0, see `LICENSE`.

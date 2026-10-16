# micro-chat

15M-parameter Mamba-2 language model.
Trains on GPU, runs on Raspberry Pi (200 MB RAM).

## Layout

| File | Purpose |
|---|---|
| `config.yaml` | All hyperparameters |
| `main.py` | Training + postprocess (prune / AWQ / export) |
| `pi_infer.py` | Pi inference, NumPy only |
| `requirements.txt` | GPU training deps |
| `pi_requirements.txt` | Pi inference deps |

## Training (GPU, CUDA 12)

```bash
python -m venv venv
source venv/bin/activate
pip install -U "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install -r requirements.txt

# verify GPU
python -c "import jax; print(jax.devices())"
# [CudaDevice(id=0)]

# 1. Train tokenizer (~5 min)
python main.py --action tokenizer

# 2. Train model
python main.py

# 3. Prune → AWQ → export int8 (~5 min)
python main.py --action full

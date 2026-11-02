# Environment restore

Reference: Python 3.10.16, CUDA 11.8, PyTorch 2.4.0+cu118, PyTorch Lightning 1.9.5.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For the exact original server environment, `requirements-lock-server.txt` is the cleaned `pip freeze` (267 packages). It includes unrelated base-env packages such as vLLM/DeepSpeed/FAISS/FlashAttention; use `requirements.txt` for a normal TTS restore.

Piper training source is not a PyPI dependency used by this repo:

```bash
git clone https://github.com/rhasspy/piper.git
cd piper
git checkout v1.0.0
git apply /path/to/piper_num_workers.patch
export PYTHONPATH="$PWD/src/python:$PYTHONPATH"
```

`piper_num_workers.patch` only adds the `--num-workers` argument used by the training commands.

Kokoro/misaki teacher source: install `kokoro==0.9.4` and `misaki[zh]==0.9.4` from requirements; teacher weights `hexgrad/Kokoro-82M-v1.1-zh` and the `zf_001` voice are not in this repo. The third-party app lexicon/tokens are also not committed; use the app model pack.

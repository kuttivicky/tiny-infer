# tiny-infer

A hand-written Qwen2 inference stack, built from scratch to match HuggingFace
`transformers` numerically.

## Status

**Milestone 1 — done.** `tinyinfer.model.Qwen2Model` reproduces HF logits for
Qwen2.5-0.5B-Instruct to within `rtol=1e-4, atol=1e-4` in fp32 on CPU.

```
python test_milestone1.py
```

## Model

Weights are *not* in this repo (942 MB, over GitHub's 100 MB file limit). The
config and tokenizer are vendored under [reference/](reference/) for offline
inspection; the weights come from the Hub:

| | |
|---|---|
| repo | `Qwen/Qwen2.5-0.5B-Instruct` |
| revision | `7ae557604adf67be50417f59c2c2f167def9a775` |
| license | Apache-2.0 |

```bash
huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct \
  --revision 7ae557604adf67be50417f59c2c2f167def9a775
```

`test_milestone1.py` resolves the snapshot by globbing `HF_HOME` (set to
`F:\hf_cache` on the machine this was developed on) — adjust `MODEL_PATH` if
your cache lives elsewhere.

## Environment

Pinned in [requirements.txt](requirements.txt). Developed against torch
2.12.1+cu126 (CUDA 12.6) and transformers 5.13.0 on Python 3.12.

```bash
python -m venv venv
venv/Scripts/activate       # source venv/bin/activate on POSIX
pip install -r requirements.txt
pip install -e . --no-deps  # puts `tinyinfer` on the path from any cwd
```

The editable install matters: without it, `python scripts/generate.py` fails
with `ModuleNotFoundError: No module named 'tinyinfer'`, because Python seeds
`sys.path[0]` with the *script's* directory (`scripts/`) rather than the repo
root. `--no-deps` keeps pip from replacing the CUDA torch wheel with the CPU
build from PyPI.

## Layout

| path | |
|---|---|
| [tinyinfer/config.py](tinyinfer/config.py) | reads HF `config.json` |
| [tinyinfer/model.py](tinyinfer/model.py) | the Qwen2 forward pass |
| [tinyinfer/loader.py](tinyinfer/loader.py) | safetensors → module tree, strips the `model.` prefix |
| [tinyinfer/sampler.py](tinyinfer/sampler.py) | token sampling |
| [scripts/generate.py](scripts/generate.py) | generation CLI (empty — milestone 2) |

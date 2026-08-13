# tiny-infer

A hand-written Qwen2 inference stack, built from scratch to match HuggingFace
`transformers` numerically.

## Status

| | milestone | check |
|---|---|---|
| M1 | hand-written Qwen2 matches HF logits | `python test_milestone1.py` → `0.0` |
| M3 | contiguous KV cache, TTFT/ITL metrics | `python scripts/generate.py "..."` |
| M4 | PagedAttention: block manager + paged pool | `python test_paged.py` |
| M5 | continuous batching with preemption | `python test_engine.py` |

Every test is differential — each milestone is checked against the previous,
simpler thing that is already known correct, rather than against a golden
file. M4 compares paged decode to cacheless recompute; M5 compares batched
generation to running each request alone.

### Numbers (GTX 1650, 4 GB, fp16)

Single-stream decode is launch-bound rather than bandwidth-bound: 0.99 GB of
weights at ~128 GB/s implies ~8 ms/token, and we see ~130 ms. That gap is why
batching scales nearly linearly here — it amortizes a fixed per-step cost.

| 8 requests x 48 tokens | wall | tok/s | TTFT p50 |
|---|---|---|---|
| sequential | 49.5 s | 7.8 | 21662 ms |
| continuous, max_batch=4 | 14.0 s | 27.4 | 3653 ms |
| continuous, max_batch=8 | **7.1 s** | **53.9 (6.9x)** | **612 ms (35x)** |

```
python scripts/bench_batching.py
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
| [tinyinfer/model.py](tinyinfer/model.py) | the Qwen2 forward pass + `PagedKVCache` |
| [tinyinfer/loader.py](tinyinfer/loader.py) | safetensors → module tree, strips the `model.` prefix |
| [tinyinfer/sampler.py](tinyinfer/sampler.py) | token sampling |
| [tinyinfer/block_manager.py](tinyinfer/block_manager.py) | logical position → physical slot, OS-page-table style |
| [tinyinfer/engine.py](tinyinfer/engine.py) | continuous batching: admit / decode / retire / preempt |
| [scripts/generate.py](scripts/generate.py) | single-stream generation with TTFT + ITL |
| [scripts/bench_batching.py](scripts/bench_batching.py) | continuous batching vs sequential |

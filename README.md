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
| M6 | measurement + figures | `python scripts/plot_bench.py` |

Every test is differential — each milestone is checked against the previous,
simpler thing that is already known correct, rather than against a golden
file. M4 compares paged decode to cacheless recompute; M5 compares batched
generation to running each request alone.

## Results

All figures: Qwen2.5-0.5B-Instruct, GTX 1650 (4 GB), fp16. Regenerate with
`python scripts/collect_bench.py && python scripts/plot_bench.py`.

### 1. Why a KV cache

![cost per token vs context length](docs/01_kv_cache.png)

Cost of producing the *next* token at a given context length. Without a cache
every step re-attends over the whole context, so it climbs linearly — 2356 ms
at 1024 tokens against a flat 110 ms.

Measured this way on purpose: below ~128 tokens this GPU is launch-bound and
the attention term is invisible, so timing a short generation loop produces a
flat step function that hides the effect entirely.

### 2. Batching trades nothing away

![throughput and TTFT vs batch size](docs/02_batch_sweep.png)

Throughput rises and time-to-first-token falls *together* — 8.6 → 58 tok/s
while TTFT p50 drops 19.4 s → 0.5 s. Scaling is near-linear because decode
here is launch-bound, not bandwidth-bound: 0.99 GB of weights at ~128 GB/s
implies ~8 ms/token and we measure ~110 ms, so the fixed per-step cost is what
batching amortizes. On a GPU that was already bandwidth-saturated the same
code would show a smaller multiple.

*(Two panels rather than one twin-axis chart: two y-scales on a single plot
let the author choose where the curves cross, which invents a relationship the
data doesn't contain.)*

### 3. Static vs continuous, uneven output lengths

![static vs continuous batching](docs/03_static_vs_continuous.png)

The one that matters. Eight requests asking for 8–96 tokens each. Static
batching admits a wave and holds every slot until the *longest* member
finishes, so occupancy decays to 25% — the GPU is computing rows whose output
is discarded. Continuous batching retires finishers immediately and admits
from the queue, holding 100%.

**19.9 s → 12.7 s (1.57x)** on identical work. The speedup here is bounded by
mean occupancy (53% → 100%), which is why this gap widens as output lengths
get more variable — and real traffic is far more variable than this.

### 4. KV memory: contiguous vs paged

![KV memory, contiguous vs paged](docs/04_kv_memory.png)

A contiguous per-sequence cache reserves `max_seq` slots whatever the sequence
actually does: 201 MB for 8 sequences that use 5 MB. Paged allocates in
16-token blocks, so waste is bounded by the block size rather than by the
worst case — **97% wasted → 13%**.

That 13% is the aggregate over these eight short sequences; internal
fragmentation is per-sequence at most `block_size - 1` slots, so the figure
falls as sequences get longer (a single 103-token sequence wastes 8%).

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
| [scripts/collect_bench.py](scripts/collect_bench.py) | all measurements → `bench_data.json` |
| [scripts/plot_bench.py](scripts/plot_bench.py) | `bench_data.json` → the four figures |

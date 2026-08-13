"""Collect every measurement the M6 plots need into one bench_data.json.

Separated from plotting so the (slow) GPU work runs once and the figures can
be restyled without re-benchmarking.
"""
import glob, json, math, statistics, time
import torch
from transformers import AutoTokenizer

from tinyinfer.loader import load_model
from tinyinfer.model import PagedKVCache
from tinyinfer.block_manager import BlockManager
from tinyinfer.engine import Engine, Request
from tinyinfer.sampler import sample

MODEL_PATH = glob.glob(r"F:\hf_cache\hub\models--Qwen--Qwen2.5-0.5B-Instruct\snapshots\*")[0]
DEV, DTYPE = "cuda", torch.float16
BLOCK, NBLOCKS = 16, 1024
OUT = "bench_data.json"

tok = AutoTokenizer.from_pretrained(MODEL_PATH)
model = load_model(MODEL_PATH, device=DEV, dtype=DTYPE)
EOS = {tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")}
cfg = model.cfg
data = {"meta": {
    "gpu": torch.cuda.get_device_name(0),
    "model": "Qwen2.5-0.5B-Instruct",
    "dtype": "float16",
    "block_size": BLOCK,
}}


class StaticEngine(Engine):
    """Static batching: a wave is admitted together and holds its slots until
    the LAST sequence in it finishes.

    Two costs continuous batching removes, both modelled here: a finished
    sequence keeps its row in the decode batch (the GPU computes it, the token
    is thrown away) and keeps its KV blocks, and no waiting request may join
    until the entire wave drains.
    """
    def _admit(self):
        if self.running:
            return                      # wave in flight — nobody joins
        super()._admit()

    def _retire(self):
        for r in self.running:
            if r.finished:
                continue
            if r.output_ids[-1] in EOS:
                r.output_ids.pop()
                r.finished = True
            elif len(r.output_ids) >= r.max_new_tokens:
                r.finished = True
        if self.running and all(r.finished for r in self.running):
            for r in self.running:      # whole wave leaves together
                self.bm.free(r.req_id)
                self.done.append(r)
            self.running = []


# ---------------------------------------------------------------- plot 1
# Cost of producing ONE more token, as a function of how long the context
# already is. Measuring it this way (rather than timing a generation loop)
# isolates the scaling: below ~128 tokens this GPU is launch-bound and the
# attention term is invisible, so a short run shows a flat step function and
# hides the very effect the cache exists to fix.
print("[1/4] cost of next token vs context length ...")
CTX = [32, 64, 128, 256, 384, 512, 768, 1024]
REPS = 7


@torch.no_grad()
def time_median(fn, reps=REPS):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    ms = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ms.append((time.perf_counter() - t0) * 1000)
    return statistics.median(ms)


@torch.no_grad()
def naive_at(n):
    """No cache: attend over the whole n-token context from scratch."""
    ids = torch.randint(0, 10000, (1, n), device=DEV)
    return time_median(lambda: model(ids))


@torch.no_grad()
def cached_at(n):
    """KV cache warm with n tokens: one new token attends to the cache."""
    bm = BlockManager(num_blocks=NBLOCKS, block_size=BLOCK)
    cache = PagedKVCache(cfg, NBLOCKS, BLOCK, DEV, DTYPE)
    w = torch.tensor(bm.allocate(0, n), device=DEV, dtype=torch.long)
    r = torch.tensor(bm.slots_for(0), device=DEV, dtype=torch.long)
    model(torch.randint(0, 10000, (1, n), device=DEV), paged=(cache, w, r),
          start_pos=0)

    one = torch.randint(0, 10000, (1, 1), device=DEV)

    def step():
        pos = bm.seq_lens[0]
        w1 = torch.tensor(bm.allocate(0, 1), device=DEV, dtype=torch.long)
        r1 = torch.tensor(bm.slots_for(0), device=DEV, dtype=torch.long)
        model(one, paged=(cache, w1, r1), start_pos=pos)

    return time_median(step)


data["next_token_cost"] = {
    "context_lens": CTX,
    "naive_ms": [naive_at(n) for n in CTX],
    "cached_ms": [cached_at(n) for n in CTX],
}
for n, a, b in zip(CTX, data["next_token_cost"]["naive_ms"],
                   data["next_token_cost"]["cached_ms"]):
    print(f"      ctx {n:5d}: no-cache {a:7.1f} ms   cached {b:6.1f} ms")

# ---------------------------------------------------------------- plot 2
# Throughput and TTFT vs batch size (uniform output lengths).
print("[2/4] throughput + TTFT vs batch size ...")
SWEEP_PROMPTS = [
    "The capital of France is",
    "Once upon a time in a small village at the edge of a cold forest, there",
    "Write a haiku about the sea:",
    "Explain gravity briefly:",
    "The three primary colors are",
    "List two reasons why the sky appears blue:",
    "def fibonacci(n):",
    "Translate to French: good morning",
]
sweep_ids = [tok(p).input_ids for p in SWEEP_PROMPTS]
UNIFORM = 48

sweep = []
for mb in (1, 2, 4, 8):
    eng = Engine(model, EOS, num_blocks=NBLOCKS, block_size=BLOCK,
                 device=DEV, dtype=DTYPE, max_batch=mb)
    t0 = time.perf_counter()
    for i, p in enumerate(sweep_ids):
        eng.add_request(Request(req_id=i, prompt_ids=p,
                                max_new_tokens=UNIFORM, arrival_t=t0))
    eng.run()
    wall = time.perf_counter() - t0
    gen = sum(len(r.output_ids) for r in eng.done)
    ttfts = sorted((r.first_token_t - r.arrival_t) * 1000 for r in eng.done)
    sweep.append({"max_batch": mb, "wall_s": wall, "tokens": gen,
                  "tok_per_s": gen / wall,
                  "ttft_p50_ms": statistics.median(ttfts),
                  "ttft_p95_ms": ttfts[int(0.95 * (len(ttfts) - 1))]})
    print(f"      max_batch={mb}: {gen/wall:.1f} tok/s")
data["batch_sweep"] = sweep

# ---------------------------------------------------------------- plot 3
# Static vs continuous when output lengths differ wildly. The money plot.
print("[3/4] static vs continuous, heterogeneous lengths ...")
HETERO = [8, 96, 12, 80, 16, 64, 10, 72]        # deliberately lopsided
assert len(HETERO) == len(sweep_ids)

hetero = {}
for label, cls in (("static", StaticEngine), ("continuous", Engine)):
    eng = cls(model, EOS, num_blocks=NBLOCKS, block_size=BLOCK,
              device=DEV, dtype=DTYPE, max_batch=4)
    t0 = time.perf_counter()
    for i, p in enumerate(sweep_ids):
        eng.add_request(Request(req_id=i, prompt_ids=p,
                                max_new_tokens=HETERO[i], arrival_t=t0))
    eng.run()
    wall = time.perf_counter() - t0
    gen = sum(len(r.output_ids) for r in eng.done)
    ttfts = sorted((r.first_token_t - r.arrival_t) * 1000 for r in eng.done)
    hetero[label] = {
        "wall_s": wall, "tokens": gen, "tok_per_s": gen / wall,
        "steps": eng.steps,
        "ttft_p50_ms": statistics.median(ttfts),
        "batch_sizes": eng.batch_sizes,
        "useful_sizes": eng.useful_sizes,
        "per_request": sorted(
            [{"req_id": r.req_id, "asked": HETERO[r.req_id],
              "got": len(r.output_ids)} for r in eng.done],
            key=lambda d: d["req_id"]),
    }
    occ = statistics.fmean(u / b for u, b in
                           zip(eng.useful_sizes, eng.batch_sizes))
    print(f"      {label:11s} wall {wall:5.2f}s  {gen/wall:5.1f} tok/s  "
          f"occupancy {occ:.1%}")
data["hetero"] = hetero
data["hetero"]["requested"] = HETERO

# ---------------------------------------------------------------- plot 4
# KV memory: what a contiguous per-sequence cache reserves vs what paged uses.
print("[4/4] KV memory model ...")
BYTES_PER_SLOT = (cfg.num_hidden_layers * cfg.num_key_value_heads
                  * cfg.head_dim * 2 * 2)        # k and v, 2 bytes each (fp16)
CONTIG_MAX_SEQ = 2048                            # what the M3 cache reserved

# Real sequence lengths from the heterogeneous run above.
lens = [len(sweep_ids[i]) + HETERO[i] for i in range(len(HETERO))]
mem = []
for n in range(1, len(lens) + 1):
    live = lens[:n]
    used = sum(live)
    paged_slots = sum(math.ceil(L / BLOCK) * BLOCK for L in live)
    mem.append({
        "concurrency": n,
        "used_tokens": used,
        "contiguous_slots": n * CONTIG_MAX_SEQ,
        "paged_slots": paged_slots,
        "contiguous_mb": n * CONTIG_MAX_SEQ * BYTES_PER_SLOT / 1e6,
        "paged_mb": paged_slots * BYTES_PER_SLOT / 1e6,
        "used_mb": used * BYTES_PER_SLOT / 1e6,
    })
data["kv_memory"] = {
    "bytes_per_slot": BYTES_PER_SLOT,
    "contiguous_max_seq": CONTIG_MAX_SEQ,
    "seq_lens": lens,
    "rows": mem,
}

with open(OUT, "w") as f:
    json.dump(data, f, indent=1)
print(f"\nwrote {OUT}")

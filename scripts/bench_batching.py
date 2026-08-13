"""Continuous batching vs running requests one at a time.

Same requests, same tokens out, same pool. The only difference is whether the
scheduler keeps one sequence on the GPU or many, so any delta is the batching.
"""
import glob, statistics, sys, time, torch
from transformers import AutoTokenizer

from tinyinfer.loader import load_model
from tinyinfer.engine import Engine, Request

MODEL_PATH = glob.glob(r"F:\hf_cache\hub\models--Qwen--Qwen2.5-0.5B-Instruct\snapshots\*")[0]
DEV, DTYPE = "cuda", torch.float16
BLOCK, NBLOCKS, MAXNEW = 16, 512, 48

PROMPTS = [
    "The capital of France is",
    "Once upon a time in a small village at the edge of a cold forest, there",
    "Write a haiku about the sea:",
    "Explain gravity briefly:",
    "The three primary colors are",
    "List two reasons why the sky appears blue:",
    "def fibonacci(n):",
    "Translate to French: good morning",
]

tok = AutoTokenizer.from_pretrained(MODEL_PATH)
model = load_model(MODEL_PATH, device=DEV, dtype=DTYPE)
EOS = {tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")}
prompt_ids = [tok(p).input_ids for p in PROMPTS]


def make_engine(max_batch):
    return Engine(model, EOS, num_blocks=NBLOCKS, block_size=BLOCK,
                  device=DEV, dtype=DTYPE, max_batch=max_batch)


def run(max_batch, label):
    eng = make_engine(max_batch)
    t0 = time.perf_counter()
    for i, pids in enumerate(prompt_ids):
        eng.add_request(Request(req_id=i, prompt_ids=pids,
                                max_new_tokens=MAXNEW, arrival_t=t0))
    eng.run()
    wall = time.perf_counter() - t0

    gen = sum(len(r.output_ids) for r in eng.done)
    ttfts = sorted((r.first_token_t - r.arrival_t) * 1000 for r in eng.done)
    mean_bs = statistics.fmean(eng.batch_sizes)
    print(f"{label:22s} wall {wall:6.2f}s  {gen:4d} tok  "
          f"{gen/wall:6.1f} tok/s  steps {eng.steps:4d}  "
          f"mean batch {mean_bs:4.2f}  TTFT p50 {statistics.median(ttfts):7.1f}ms "
          f"p95 {ttfts[int(0.95*(len(ttfts)-1))]:7.1f}ms")
    return gen / wall, eng


# warmup so neither arm eats CUDA init
warm = make_engine(2)
warm.add_request(Request(req_id=0, prompt_ids=prompt_ids[0], max_new_tokens=4))
warm.run()
torch.cuda.synchronize()

print(f"{len(PROMPTS)} requests x {MAXNEW} max new tokens\n")
seq_tps, _ = run(1, "sequential (batch=1)")
results = [(1, seq_tps)]
for mb in (2, 4, 8):
    tps, _ = run(mb, f"continuous (max={mb})")
    results.append((mb, tps))

print()
for mb, tps in results[1:]:
    print(f"  max_batch={mb}: {tps/seq_tps:.2f}x throughput vs sequential")

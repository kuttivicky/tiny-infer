"""Continuous batching must not change what a request generates.

A sequence sharing a decode batch with others has different padding, a
different attention mask, and neighbours at unrelated positions in the RoPE
table. None of that may perturb its tokens. So: generate each prompt alone,
then generate all of them interleaved, and require the token streams to match
exactly. This is the test that catches a wrong per-sequence position or a
padding slot leaking into attention.
"""
import glob, sys, torch
from transformers import AutoTokenizer

from tinyinfer.loader import load_model
from tinyinfer.model import PagedKVCache
from tinyinfer.block_manager import BlockManager
from tinyinfer.engine import Engine, Request
from tinyinfer.sampler import sample

MODEL_PATH = glob.glob(r"F:\hf_cache\hub\models--Qwen--Qwen2.5-0.5B-Instruct\snapshots\*")[0]
DEV = "cuda" if torch.cuda.is_available() else "cpu"
# fp32 by default: this asserts EXACT token equality, and in fp16 the batched
# kernels reduce in a different order, so a near-tied argmax can legitimately
# flip. Run with --fp16 to see that (it is precision, not a scheduling bug).
DTYPE = torch.float16 if "--fp16" in sys.argv else torch.float32
BLOCK, NBLOCKS, MAXNEW = 16, 256, 24

tok = AutoTokenizer.from_pretrained(MODEL_PATH)
model = load_model(MODEL_PATH, device=DEV, dtype=DTYPE)
EOS = {tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")}

# Deliberately uneven lengths so the batch is ragged and padding is exercised.
PROMPTS = [
    "The capital of France is",
    "Once upon a time, in a very small village at the edge of a cold forest, there",
    "2 + 2 =",
    "Explain gravity briefly:",
]


@torch.no_grad()
def generate_alone(prompt_ids, max_new):
    """Reference: one sequence, its own pool, no padding, no neighbours."""
    bm = BlockManager(num_blocks=NBLOCKS, block_size=BLOCK)
    cache = PagedKVCache(model.cfg, NBLOCKS, BLOCK, DEV, DTYPE)

    def slots(n):
        w = bm.allocate(0, n)
        r = bm.slots_for(0)
        return (torch.tensor(w, device=DEV, dtype=torch.long),
                torch.tensor(r, device=DEV, dtype=torch.long))

    ids = torch.tensor([prompt_ids], device=DEV)
    logits = model(ids, paged=(cache,) + slots(len(prompt_ids)), start_pos=0)
    nxt = sample(logits[0, -1], temperature=0.0)
    out = [nxt]

    while len(out) < max_new:
        if nxt in EOS:
            out.pop()
            break
        pos = bm.seq_lens[0]
        logits = model(torch.tensor([[nxt]], device=DEV),
                       paged=(cache,) + slots(1), start_pos=pos)
        nxt = sample(logits[0, -1], temperature=0.0)
        out.append(nxt)
    else:
        if out and out[-1] in EOS:
            out.pop()
    return out


prompt_ids = [tok(p).input_ids for p in PROMPTS]

print("=== reference: each request generated alone ===")
alone = {}
for i, pids in enumerate(prompt_ids):
    alone[i] = generate_alone(pids, MAXNEW)
    print(f"  req {i} ({len(pids):3d} prompt tok): {tok.decode(alone[i])!r}")

print("\n=== continuous batching ===")
eng = Engine(model, EOS, num_blocks=NBLOCKS, block_size=BLOCK,
             device=DEV, dtype=DTYPE, max_batch=4)
for i, pids in enumerate(prompt_ids):
    eng.add_request(Request(req_id=i, prompt_ids=pids, max_new_tokens=MAXNEW))
eng.run()

batched = {r.req_id: r.output_ids for r in eng.done}
for i in sorted(batched):
    print(f"  req {i}: {tok.decode(batched[i])!r}")

mean_bs = sum(eng.batch_sizes) / len(eng.batch_sizes)
print(f"\nsteps={eng.steps}  mean batch={mean_bs:.2f}  preemptions={eng.preemptions}")
print(f"pool fully reclaimed: {len(eng.bm.free_blocks) == NBLOCKS} "
      f"({len(eng.bm.free_blocks)}/{NBLOCKS} free, tables={eng.bm.block_tables})")

fails = [i for i in sorted(alone) if alone[i] != batched.get(i)]
for i in fails:
    print(f"\nMISMATCH req {i}\n  alone:   {alone[i]}\n  batched: {batched.get(i)}")
assert not fails, f"continuous batching changed output for requests {fails}"
assert len(eng.bm.free_blocks) == NBLOCKS, "block leak: pool not fully reclaimed"


# ---- preemption: same check, but a pool too small to hold all four at once ----
# Peak demand is 9 blocks (2+3+2+2); 7 admits all four on their prompts alone,
# then runs dry as they grow, which is exactly the eviction path. It stays
# above the 3 blocks the longest request needs on its own, so forward progress
# is always possible and the scheduler cannot livelock.
TIGHT = 7
print("\n=== continuous batching under memory pressure (forces preemption) ===")
tight = Engine(model, EOS, num_blocks=TIGHT, block_size=BLOCK,
               device=DEV, dtype=DTYPE, max_batch=4)
for i, pids in enumerate(prompt_ids):
    tight.add_request(Request(req_id=i, prompt_ids=pids, max_new_tokens=MAXNEW))
tight.run()

pre = {r.req_id: r.output_ids for r in tight.done}
print(f"steps={tight.steps}  preemptions={tight.preemptions}  "
      f"evicted={[r.req_id for r in tight.done if r.preempted]}")
print(f"pool fully reclaimed: {len(tight.bm.free_blocks) == TIGHT}")

fails2 = [i for i in sorted(alone) if alone[i] != pre.get(i)]
for i in fails2:
    print(f"\nMISMATCH req {i}\n  alone:     {alone[i]}\n  preempted: {pre.get(i)}")
assert not fails2, f"preemption changed output for requests {fails2}"
assert tight.preemptions > 0, "pool was not tight enough to exercise preemption"
print("\nCONTINUOUS BATCHING PASSED")

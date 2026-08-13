"""Paged KV must be bit-comparable to recomputing the whole sequence.

The block table deliberately scatters a sequence across discontiguous physical
blocks, so this is the test that catches address-translation bugs: if `slot()`
or the gather order were wrong, attention would read the right *tokens* in the
wrong *order* and the logits would drift.
"""
import glob, torch
from transformers import AutoTokenizer
from tinyinfer.loader import load_model
from tinyinfer.model import PagedKVCache
from tinyinfer.block_manager import BlockManager

MODEL_PATH = glob.glob(r"F:\hf_cache\hub\models--Qwen--Qwen2.5-0.5B-Instruct\snapshots\*")[0]
BLOCK_SIZE, NUM_BLOCKS = 16, 64
DEV = "cuda" if torch.cuda.is_available() else "cpu"

tok = AutoTokenizer.from_pretrained(MODEL_PATH)
model = load_model(MODEL_PATH, device=DEV, dtype=torch.float32)

ids = tok("The capital of France is", return_tensors="pt").input_ids.to(DEV)
cache = PagedKVCache(model.cfg, NUM_BLOCKS, BLOCK_SIZE, DEV, torch.float32)
bm = BlockManager(num_blocks=NUM_BLOCKS, block_size=BLOCK_SIZE)


def slots(seq_id, n_new):
    w = bm.allocate(seq_id, n_new)
    r = bm.slots_for(seq_id)
    return (torch.tensor(w, device=DEV, dtype=torch.long),
            torch.tensor(r, device=DEV, dtype=torch.long))


# Fragment the pool first: hold a decoy sequence so seq 0 cannot land on a
# contiguous run of blocks. A bug that assumes contiguity survives a clean pool.
bm.allocate(seq_id=77, n_new=BLOCK_SIZE * 3)

full = ids.clone()
worst = 0.0

with torch.no_grad():
    # prefill
    logits = model(ids, paged=(cache,) + slots(0, ids.shape[1]), start_pos=0)
    ref = model(full, return_all_logits=True)
    d = (logits[0, -1] - ref[0, -1]).abs().max().item()
    worst = max(worst, d)
    print(f"prefill      paged-vs-recompute max abs diff: {d}")

    nxt = int(logits[0, -1].argmax())
    for step in range(12):
        full = torch.cat([full, torch.tensor([[nxt]], device=DEV)], dim=1)
        pos = full.shape[1] - 1

        got = model(torch.tensor([[nxt]], device=DEV),
                    paged=(cache,) + slots(0, 1), start_pos=pos)
        want = model(full, return_all_logits=True)      # no cache at all

        d = (got[0, -1] - want[0, -1]).abs().max().item()
        worst = max(worst, d)
        print(f"decode {step:2d}    paged-vs-recompute max abs diff: {d:.3e}"
              f"   token={tok.decode(nxt)!r}")
        nxt = int(got[0, -1].argmax())

print(f"\nblock table for seq 0: {bm.block_tables[0]}")
print(f"physically contiguous? "
      f"{bm.block_tables[0] == list(range(min(bm.block_tables[0]), max(bm.block_tables[0]) + 1))}")
print(f"worst diff across all steps: {worst:.3e}")
assert worst < 1e-3, f"paged attention diverged: {worst}"
print("PAGED KV CACHE PASSED")

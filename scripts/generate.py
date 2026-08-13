import glob, statistics, sys, time, torch
from transformers import AutoTokenizer
from tinyinfer.loader import load_model
from tinyinfer.sampler import sample
from tinyinfer.model import PagedKVCache
from tinyinfer.block_manager import BlockManager

MODEL_PATH = glob.glob(r"F:\hf_cache\hub\models--Qwen--Qwen2.5-0.5B-Instruct\snapshots\*")[0]

BLOCK_SIZE = 16
NUM_BLOCKS = 128          # 2048 token slots, same capacity as the old flat cache


def step_slots(bm, seq_id, n_new, device):
    """Reserve n_new tokens and return (write_slots, read_slots) as CUDA long
    tensors. Built ONCE per step — passing lists down would cost 24 separate
    host-to-device copies, one per layer."""
    write = bm.allocate(seq_id, n_new)
    read = bm.slots_for(seq_id)
    return (torch.tensor(write, device=device, dtype=torch.long),
            torch.tensor(read, device=device, dtype=torch.long))


def main(prompt: str, max_new_tokens: int = 64):
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = load_model(MODEL_PATH, device="cuda", dtype=torch.float16)

    text = tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False, add_generation_prompt=True,
    )
    ids = tok(text, return_tensors="pt").input_ids.cuda()
    eos_ids = {tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")}   # define BEFORE use

    cache = PagedKVCache(model.cfg, NUM_BLOCKS, BLOCK_SIZE, "cuda", torch.float16)
    bm = BlockManager(num_blocks=NUM_BLOCKS, block_size=BLOCK_SIZE)

    with torch.no_grad():
        # ---- warmup so we dont time CUDA init ----
        # Decode (s=1) dispatches to different SDPA kernels than prefill (s=N),
        # so warming only the prefill shape leaves the first few ITL samples
        # carrying autotune cost. Warm both, under a throwaway seq_id we free
        # afterwards so the real run starts against a clean block table.
        model(ids, paged=(cache,) + step_slots(bm, 99, ids.shape[1], "cuda"), start_pos=0)
        for i in range(5):
            model(ids[:, :1], paged=(cache,) + step_slots(bm, 99, 1, "cuda"),
                  start_pos=ids.shape[1] + i)
        torch.cuda.synchronize()
        bm.free(99)

        # ---- PREFILL: whole prompt in one pass -> TTFT ----
        t0 = time.perf_counter()
        logits = model(ids, paged=(cache,) + step_slots(bm, 0, ids.shape[1], "cuda"),
                       start_pos=0)
        next_id = sample(logits[0, -1], temperature=0.0)
        torch.cuda.synchronize()
        ttft_ms = (time.perf_counter() - t0) * 1000

        pos = ids.shape[1]
        out_ids, itl_ms = [next_id], []
        print(tok.decode(next_id), end="", flush=True)   # the prefill token counts too

        # ---- DECODE: one token in, one token out ---- 
        for _ in range(max_new_tokens - 1):
            t0 = time.perf_counter()
            cur = torch.tensor([[next_id]], device="cuda")
            logits = model(cur, paged=(cache,) + step_slots(bm, 0, 1, "cuda"),
                           start_pos=pos)                   # <- only ONE token
            next_id = sample(logits[0, -1], temperature=0.00)
            torch.cuda.synchronize()
            itl_ms.append((time.perf_counter() - t0) * 1000)

            pos += 1
            if next_id in eos_ids:
                break
            out_ids.append(next_id)
            print(tok.decode(next_id), end="", flush=True)

    print(f"\n\nTTFT: {ttft_ms:.1f} ms ({ids.shape[1]} prompt tokens)")
    # Median is the honest headline for ITL: the mean is skewed by the odd
    # scheduling hiccup, and these are per-token samples, not a total.
    print(f"ITL: median {statistics.median(itl_ms):.2f} ms, "
          f"mean {statistics.fmean(itl_ms):.2f}, "
          f"first {itl_ms[0]:.2f}, last {itl_ms[-1]:.2f}")
    print(f"decode throughput: {1000*len(itl_ms)/sum(itl_ms):.1f} tok/s")
    print(f"pool utilization: {bm.utilization():.3f} "
          f"({len(bm.free_blocks)}/{NUM_BLOCKS} blocks free)")
    torch.save(itl_ms, "bench_paged.pt")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "What is the capital of France?")
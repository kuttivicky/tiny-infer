import glob, statistics, sys, time, torch
from transformers import AutoTokenizer
from tinyinfer.loader import load_model
from tinyinfer.sampler import sample
from tinyinfer.model import KVCache

MODEL_PATH = glob.glob(r"F:\hf_cache\hub\models--Qwen--Qwen2.5-0.5B-Instruct\snapshots\*")[0]

def main(prompt: str, max_new_tokens: int = 64):
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = load_model(MODEL_PATH, device="cuda", dtype=torch.float16)

    text = tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False, add_generation_prompt=True,
    )
    ids = tok(text, return_tensors="pt").input_ids.cuda()
    eos_ids = {tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")}   # define BEFORE use

    cache = KVCache(model.cfg, max_seq=2048, device="cuda", dtype=torch.float16)

    with torch.no_grad():
        # ---- warmup so we dont time CUDA init ----
        # Decode (s=1) dispatches to different SDPA kernels than prefill (s=N),
        # so warming only the prefill shape leaves the first few ITL samples
        # carrying autotune cost. Warm both, on a throwaway cache so we don't
        # poison the positions the real run is about to read.
        warm = KVCache(model.cfg, 2048, "cuda", torch.float16)
        model(ids, cache=warm, start_pos=0)
        for i in range(5):
            model(ids[:, :1], cache=warm, start_pos=ids.shape[1] + i)
        torch.cuda.synchronize()
        del warm

        # ---- PREFILL: whole prompt in one pass -> TTFT ----
        t0 = time.perf_counter()
        logits = model(ids, cache=cache, start_pos=0)
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
            logits = model(cur, cache=cache, start_pos=pos) # <- only ONE token
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
    torch.save(itl_ms, "bench_kvcache.pt")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "What is the capital of France?")
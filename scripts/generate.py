import glob, sys, time, torch
from transformers import AutoTokenizer
from tinyinfer.loader import load_model
from tinyinfer.sampler import sample

MODEL_PATH = glob.glob(r"F:\hf_cache\hub\models--Qwen--Qwen2.5-0.5B-Instruct\snapshots\*")[0]

def main(prompt: str, max_new_tokens: int = 64):
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = load_model(MODEL_PATH, device="cuda", dtype=torch.float16)

    text = tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False, add_generation_prompt=True,
    )
    ids = tok(text, return_tensors="pt").input_ids.cuda()

    eos_ids = {tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")}
    per_token_ms = []

    with torch.no_grad():
        for i in range(max_new_tokens):
            t0 = time.perf_counter()

            logits = model(ids)              # NAIVE: recompute the WHOLE sequence
            next_id = sample(logits[0, -1], temperature=0.0)

            torch.cuda.synchronize()         # GPU is async; sync before timing
            per_token_ms.append((time.perf_counter() - t0) * 1000)

            if next_id in eos_ids:
                break
            ids = torch.cat([ids, torch.tensor([[next_id]], device=ids.device)], dim=1)
            print(tok.decode(next_id), end="", flush=True)

    print(f"\n\nprompt tokens: {ids.shape[1] - len(per_token_ms)}")
    print(f"generated: {len(per_token_ms)} tokens")
    print(f"first token: {per_token_ms[0]:.1f} ms   last token: {per_token_ms[-1]:.1f} ms")
    print(f"mean: {sum(per_token_ms)/len(per_token_ms):.1f} ms  "
          f"({1000*len(per_token_ms)/sum(per_token_ms):.1f} tok/s)")
    torch.save(per_token_ms, "bench_naive.pt")   # baseline for later comparison

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "What is the capital of France?")
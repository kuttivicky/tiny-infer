import glob, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tinyinfer.loader import load_model

MODEL_PATH = glob.glob(r"F:\hf_cache\hub\models--Qwen--Qwen2.5-0.5B-Instruct\snapshots\*")[0]

tok = AutoTokenizer.from_pretrained(MODEL_PATH)
ids = tok("The capital of France is", return_tensors="pt").input_ids

mine = load_model(MODEL_PATH, device="cpu", dtype=torch.float32)
ref  = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float32).eval()

with torch.no_grad():
    # every position, not just the last — this is a teacher-forced comparison
    a = mine(ids, return_all_logits=True)
    b = ref(ids).logits

print("max abs diff:", (a - b).abs().max().item())
print("my top token:  ", tok.decode(a[0, -1].argmax()))
print("ref top token: ", tok.decode(b[0, -1].argmax()))
torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-4)
print("MILESTONE 1 PASSED")
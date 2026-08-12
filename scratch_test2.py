# scratch_test2.py
import torch
from tinyinfer.config import load_config
from tinyinfer.model import Qwen2Attention, build_rope_cache

cfg = load_config(r"<MODEL_PATH>")
attn = Qwen2Attention(cfg)
x = torch.randn(1, 5, cfg.hidden_size)
cos, sin = build_rope_cache(cfg.head_dim, 32, cfg.rope_theta, "cpu", torch.float32)
out = attn(x, cos[:5], sin[:5])
print(out.shape)   # expect (1, 5, 896)
import torch
from tinyinfer.config import load_config
from tinyinfer.model import RMSNorm, Qwen2MLP

cfg = load_config(r"F:\hf_cache\hub\models--Qwen--Qwen2.5-0.5B-Instruct\snapshots\7ae557604adf67be50417f59c2c2f167def9a775")
x = torch.randn(1, 5, cfg.hidden_size) # batch=1, seq=5 tokens
print(RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)(x).shape)
print(Qwen2MLP(cfg)(x).shape)
import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import ModelConfig

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim)) # learned per-channel scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., dim). Compute in fp32 for stability, return in input dtype.
        dtype = x.dtype
        x = x.float()
        variance = x.pow(2).mean(-1, keepdim=True) # mean of squares
        x = x * torch.rsqrt(variance + self.eps)    # divide by RMS
        return (x * self.weight.to(torch.float32)).to(dtype)


class Qwen2MLP(nn.Module): # MLP SwiGlU
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


def rotate_half(x):
    # split last dim in two halves [x1, x2] -> [-x2, x1]
    x1, x2 = x.chunk(2, dim= -1)
    return torch.cat((-x2, x1), dim= -1)

def apply_rope(q, k, cos, sin):
    # q,k: (batch, num_heads, seq, head_dim); cos,sin: (seq, head_dim)
    q_rot = q * cos + rotate_half(q) * sin
    k_rot = k * cos + rotate_half(k) * sin
    return q_rot, k_rot

def build_rope_cache(head_dim, max_pos, theta, device, dtype):
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    positions = torch.arange(max_pos, device=device).float()
    freqs = torch.outer(positions, inv_freq)          # (max_pos, head_dim/2)
    emb = torch.cat((freqs, freqs), dim=-1)           # (max_pos, head_dim)
    return emb.cos().to(dtype), emb.sin().to(dtype)
    

class Qwen2Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.num_heads = cfg.num_attention_heads    #14
        self.num_kv_heads = cfg.num_key_value_heads #2
        self.head_dim = cfg.head_dim                #64
        self.num_kv_groups = self.num_heads // self.num_kv_heads  #7

        # NOTE: Qwen2 quirk — q/k/v have bias=True, o_proj has bias=False
        self.q_proj = nn.Linear(cfg.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(cfg.hidden_size, self.num_kv_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(cfg.hidden_size, self.num_kv_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, cfg.hidden_size, bias=False)

    def forward(self, x, cos, sin):
        # x: (batch, seq, 896)
        b, s, _ = x.shape

        q = self.q_proj(x)   # (b, s, 14*64 = 896)
        k = self.k_proj(x)   # (b, s,  2*64 = 128)
        v = self.v_proj(x)   # (b, s,  2*64 = 128)

        # reshape into heads: (b, seq, heads, head_dim) -> (b, heads, seq, head_dim)
        q = q.view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, s, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, s, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # rotate q and k by position
        q, k = apply_rope(q, k, cos, sin)

        # GQA: expand 2 kv heads to 14 so each group of 7 q-heads sees its kv head
        k = k.repeat_interleave(self.num_kv_groups, dim=1)  # (b, 14, s, 64)
        v = v.repeat_interleave(self.num_kv_groups, dim=1)

        # scaled dot-product attention with causal mask
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        # merge heads back: (b, 14, s, 64) -> (b, s, 896)
        out = out.transpose(1, 2).contiguous().view(b, s, -1)
        return self.o_proj(out)


class Qwen2DecoderLayer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.self_attn = Qwen2Attention(cfg)
        self.mlp = Qwen2MLP(cfg)
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)

    def forward(self, x, cos, sin):
        #  pre-norm residual: normalize the INPUT to each sub-block, add the output back
        x = x + self.self_attn(self.input_layernorm(x), cos, sin)
        x = x + self.mlp(self.post_attention_layernorm(x))

        return x

class Qwen2Model(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(
            [Qwen2DecoderLayer(cfg) for _ in range(cfg.num_hidden_layers)]
        )
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

        # tied embeddings: input and output matrices are the SAME tensor
        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        # RoPE tables built once, reused by every layer
        cos, sin = build_rope_cache(
            cfg.head_dim, cfg.max_position_embeddings, cfg.rope_theta,
            device="cpu", dtype=torch.float32,
        )
        # persistent=False -> not saved in state_dict, so strict loading stays clean
        self.register_buffer("cos_cache", cos, persistent=False)
        self.register_buffer("sin_cache", sin, persistent=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # input_ids: (batch, seq) of int token ids
        b, s = input_ids.shape
        x = self.embed_tokens(input_ids)          # (b, s, 896)
        cos, sin = self.cos_cache[:s], self.sin_cache[:s]
        for layer in self.layers:
            x = layer(x, cos, sin)
        x = self.norm(x)
        return self.lm_head(x)                    # (b, s, vocab_size)



# def apply_rope(q, k, positions, inv_freq):
#     return
    
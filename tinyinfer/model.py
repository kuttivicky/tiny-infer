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
    
class PagedKVCache:
    """Flat block-structured KV pool shared by ALL sequences.

    Layout per layer: (num_blocks * block_size, kv_heads, head_dim)
    We keep the slot dimension flat rather than (num_blocks, block_size, ...)
    so writes and gathers are a single index_select on dim 0 — no reshape
    arithmetic in the hot path.
    """
    def __init__(self, cfg, num_blocks, block_size, device, dtype):
        n_slots = num_blocks * block_size
        shape = (cfg.num_hidden_layers, n_slots, cfg.num_key_value_heads, cfg.head_dim)
        self.k = torch.zeros(shape, device=device, dtype=dtype)
        self.v = torch.zeros(shape, device=device, dtype=dtype)

    def write(self, layer_idx, k, v, slots):
        # k, v: (b, kv_heads, s, head_dim) -> (b*s, kv_heads, head_dim), so this
        # serves both prefill (b=1, s=n) and batched decode (b=n, s=1) unchanged.
        b, h, s, d = k.shape
        self.k[layer_idx].index_copy_(0, slots, k.transpose(1, 2).reshape(b * s, h, d))
        self.v[layer_idx].index_copy_(0, slots, v.transpose(1, 2).reshape(b * s, h, d))

    def gather(self, layer_idx, slots):
        """slots (n,)   -> (1, kv_heads, n, head_dim)      single sequence
           slots (b, n) -> (b, kv_heads, n, head_dim)      ragged batch, padded

        Either way it is one index_select on the flat slot dimension; the 2D
        case just flattens first and folds the batch back out afterwards.
        """
        if slots.dim() == 1:
            k = self.k[layer_idx].index_select(0, slots)   # (n, kv_heads, head_dim)
            v = self.v[layer_idx].index_select(0, slots)
            return k.transpose(0, 1).unsqueeze(0), v.transpose(0, 1).unsqueeze(0)

        b, n = slots.shape
        flat = slots.reshape(-1)
        k = self.k[layer_idx].index_select(0, flat)        # (b*n, kv_heads, head_dim)
        v = self.v[layer_idx].index_select(0, flat)
        h, d = k.shape[1], k.shape[2]
        return (k.view(b, n, h, d).transpose(1, 2),
                v.view(b, n, h, d).transpose(1, 2))


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

    def forward(self, x, cos, sin, paged=None, layer_idx=0, start_pos=0):
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

        attn_mask = None
        if paged is not None:
            cache, write_slots, read_slots, attn_mask = paged
            cache.write(layer_idx, k, v, write_slots)      # scatter new K/V
            k, v = cache.gather(layer_idx, read_slots)     # gather ALL of this seq's K/V

        # GQA: SDPA broadcasts the 2 kv heads across the 14 q heads internally,
        # so we hand it the narrow k/v directly. repeat_interleave used to
        # materialize a 7x-larger k/v here — pure memory traffic for no reason.
        #
        # CRITICAL: is_causal only when q_len == kv_len (prefill).
        # In decode, one query legitimately attends to ALL cached keys.
        # Ragged batched decode is the third case: q_len == 1 so nothing is
        # causal, but the shorter sequences' padding MUST be masked or they
        # attend to whatever tenant last held those slots.
        if attn_mask is not None:
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, enable_gqa=True
            )
        else:
            out = F.scaled_dot_product_attention(
                q, k, v, is_causal=(s > 1), enable_gqa=True
            )


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

    def forward(self, x, cos, sin, paged=None, layer_idx=0, start_pos=0):
        #  pre-norm residual: normalize the INPUT to each sub-block, add the output back
        x = x + self.self_attn(self.input_layernorm(x), cos, sin, paged, layer_idx, start_pos)
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

    def forward(self, input_ids: torch.Tensor, paged=None, start_pos=0,
                return_all_logits: bool = False, positions=None,
                pad_mask=None) -> torch.Tensor:
        # input_ids: (batch, seq) of int token ids
        # paged:     None, or (PagedKVCache, write_slots, read_slots) with both
        #            slot tensors already on-device — built once per step, not
        #            per layer. read_slots may be (n,) or (b, n) for a batch.
        # positions: (b, s) absolute positions, REQUIRED when the rows of the
        #            batch sit at different points in their own sequences.
        # pad_mask:  (b, n_kv) bool, True = real token. Only for ragged batches.
        b, s = input_ids.shape
        x = self.embed_tokens(input_ids)          # (b, s, 896)

        if positions is None:
            # single sequence (or a batch that happens to be aligned)
            positions = torch.arange(start_pos, start_pos + s,
                                     device=input_ids.device).unsqueeze(0)

        # (b, s, head_dim) -> (b, 1, s, head_dim) to broadcast over heads.
        # Under continuous batching each row is at its own position, so this
        # cannot be a single contiguous slice of the table.
        cos = self.cos_cache[positions].unsqueeze(1)
        sin = self.sin_cache[positions].unsqueeze(1)

        if paged is not None and pad_mask is not None:
            # Built once here, not 24 times inside attention.
            add_mask = torch.where(pad_mask[:, None, None, :],
                                   0.0, float("-inf")).to(x.dtype)
            paged = (paged[0], paged[1], paged[2], add_mask)
        elif paged is not None:
            paged = (paged[0], paged[1], paged[2], None)

        for i, layer in enumerate(self.layers):
            x = layer(x, cos, sin, paged, i, start_pos)
        x = self.norm(x)

        # Generation only ever samples from the last position, so projecting all
        # s positions through a 151936-wide lm_head is wasted work — it dominates
        # prefill for long prompts. Teacher-forced comparisons (the Milestone 1
        # oracle) need every position, so they pass return_all_logits=True.
        if not return_all_logits:
            return self.lm_head(x[:, -1:, :])     # (b, 1, vocab_size)
        return self.lm_head(x)                    # (b, s, vocab_size)

    
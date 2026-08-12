import json
from dataclasses import dataclass
from pathlib import Path

@dataclass
class ModelConfig:
    hidden_size: int        # width of the residual stream (896)
    num_hidden_layers: int  # transformer blocks stacked (24)
    num_attention_heads: int # query heads (14)
    num_key_value_heads: int # KV heads (2) <- GQA
    intermediate_size: int   # MLP inner width (4864)
    vocab_size: int         # number of tokens the model knows (151936)
    rms_norm_eps: float     # numerical-stability epsilon
    rope_theta: float       # RoPE base frequency(1,000,000)
    max_position_embeddings: int
    tie_word_embeddings: bool

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads # 896/14 = 64       

def load_config(model_path: str) -> ModelConfig:
    with open(Path(model_path) / "config.json") as f:
        raw = json.load(f)
    return ModelConfig(
        hidden_size=raw["hidden_size"],
        num_hidden_layers=raw["num_hidden_layers"],
        num_attention_heads=raw["num_attention_heads"],
        num_key_value_heads=raw["num_key_value_heads"],
        intermediate_size=raw["intermediate_size"],
        vocab_size=raw["vocab_size"],
        rms_norm_eps=raw["rms_norm_eps"],
        rope_theta=raw["rope_theta"],
        max_position_embeddings=raw["max_position_embeddings"],
        tie_word_embeddings=raw["tie_word_embeddings"],
    )
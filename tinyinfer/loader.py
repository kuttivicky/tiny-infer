import glob
import torch
from pathlib import Path
from safetensors.torch import load_file
from .config import load_config
from .model import Qwen2Model


def load_model(model_path: str, device: str = "cuda", dtype=torch.float16) -> Qwen2Model:
    cfg = load_config(model_path)
    model = Qwen2Model(cfg)

    # merge all shards (0.5B is a single file, but this generalizes)
    raw = {}
    for shard in glob.glob(str(Path(model_path) / "*.safetensors")):
        raw.update(load_file(shard))

    # HF keys are prefixed "model." (e.g. model.layers.0.self_attn.q_proj.weight);
    # our module tree has no such prefix, so strip it.
    state = {}
    for k, v in raw.items():
        state[k[len("model."):] if k.startswith("model.") else k] = v

    missing, unexpected = model.load_state_dict(state, strict=False)
    # lm_head.weight is expected to be missing when embeddings are tied
    assert not unexpected, f"unexpected keys: {unexpected[:5]}"
    assert all(m == "lm_head.weight" for m in missing), f"missing keys: {missing[:5]}"

    return model.to(device=device, dtype=dtype).eval()
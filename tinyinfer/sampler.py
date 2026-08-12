import torch

def sample(logits: torch.Tensor, temperature: float = 0.0, top_p: float = 1.0) -> int:
    """logits: (vocab,) for ONE position. Returns a token id."""
    if temperature <= 0.0:                       # greedy / deterministic
        return int(logits.argmax())

    probs = torch.softmax(logits.float() / temperature, dim=-1)

    if top_p < 1.0:                              # nucleus sampling
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cumsum = sorted_probs.cumsum(dim=-1)
        # keep the smallest set whose cumulative mass exceeds top_p
        cutoff = (cumsum - sorted_probs) > top_p   # shift so the crossing token is kept
        sorted_probs[cutoff] = 0.0
        sorted_probs /= sorted_probs.sum()
        choice = torch.multinomial(sorted_probs, 1)
        return int(sorted_idx[choice])

    return int(torch.multinomial(probs, 1))
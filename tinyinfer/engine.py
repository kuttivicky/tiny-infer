"""Continuous-batching scheduler over the paged KV pool.

Static batching makes every sequence in a batch wait for the longest one to
finish. Continuous batching instead re-forms the batch every step: finished
sequences leave immediately and waiting ones are admitted into the hole, so
the GPU never idles on padding.
"""
import time
from collections import deque
from dataclasses import dataclass, field

import torch

from .block_manager import BlockManager
from .model import PagedKVCache
from .sampler import sample


@dataclass
class Request:
    req_id: int
    prompt_ids: list[int]
    max_new_tokens: int
    output_ids: list[int] = field(default_factory=list)
    arrival_t: float = 0.0
    first_token_t: float | None = None   # for TTFT
    finished: bool = False

    # Set when the scheduler evicted this request mid-flight; purely for stats.
    preempted: int = 0

    @property
    def context_ids(self) -> list[int]:
        """Everything that must be in KV for the next decode step.

        Also exactly what a preempted request re-prefills: recompute restores
        the evicted blocks by replaying prompt + tokens already emitted, so
        eviction costs time but never loses progress.
        """
        return self.prompt_ids + self.output_ids


def batched_read_slots(bm, req_ids, device):
    seqs = [bm.slots_for(r) for r in req_ids]
    max_len = max(len(s) for s in seqs)
    slots = torch.zeros(len(seqs), max_len, dtype=torch.long, device=device)
    mask = torch.zeros(len(seqs), max_len, dtype=torch.bool, device=device)
    for i, s in enumerate(seqs):
        slots[i, :len(s)] = torch.tensor(s, device=device)
        mask[i, :len(s)] = True          # True = real token, False = padding
    return slots, mask


class Engine:
    def __init__(self, model, eos_ids, num_blocks=128, block_size=16,
                 device="cuda", dtype=torch.float16, max_batch=8):
        self.model = model
        self.eos_ids = set(eos_ids)
        self.device = device
        self.max_batch = max_batch

        self.bm = BlockManager(num_blocks=num_blocks, block_size=block_size)
        self.cache = PagedKVCache(model.cfg, num_blocks, block_size, device, dtype)

        self.waiting: deque[Request] = deque()
        self.running: list[Request] = []
        self.done: list[Request] = []

        self.steps = 0
        self.preemptions = 0
        self.batch_sizes: list[int] = []    # rows in the decode batch
        self.useful_sizes: list[int] = []   # rows still producing wanted tokens

    # ---------------------------------------------------------------- intake

    def add_request(self, req: Request) -> None:
        if not req.arrival_t:
            req.arrival_t = time.perf_counter()
        self.waiting.append(req)

    @property
    def pending(self) -> bool:
        return bool(self.waiting or self.running)

    # ------------------------------------------------------------ admission

    def _admit(self) -> None:
        """Prefill waiting requests into free capacity, head of queue first."""
        while self.waiting and len(self.running) < self.max_batch:
            req = self.waiting[0]
            n = len(req.context_ids)
            if not self.bm.can_allocate(req.req_id, n):
                break                    # head-of-line blocks; don't reorder
            self.waiting.popleft()
            self._prefill(req)
            self.running.append(req)

    @torch.no_grad()
    def _prefill(self, req: Request) -> None:
        ctx = req.context_ids
        ids = torch.tensor([ctx], device=self.device)
        write = torch.tensor(self.bm.allocate(req.req_id, len(ctx)),
                             device=self.device, dtype=torch.long)
        read = torch.tensor(self.bm.slots_for(req.req_id),
                            device=self.device, dtype=torch.long)

        logits = self.model(ids, paged=(self.cache, write, read), start_pos=0)
        tok = sample(logits[0, -1], temperature=0.0)
        req.output_ids.append(tok)
        if req.first_token_t is None:
            req.first_token_t = time.perf_counter()

    # --------------------------------------------------------------- decode

    def _blocks_for_one_more(self) -> int:
        """Blocks needed to extend EVERY running sequence by one token.

        Per-sequence can_allocate is not enough here: four sequences each
        needing one fresh block need four blocks collectively, and checking
        them one at a time would happily approve the batch with three free.
        """
        need = 0
        for r in self.running:
            filled = self.bm.seq_lens[r.req_id]
            have = len(self.bm.block_tables[r.req_id])
            need += max(0, -(-(filled + 1) // self.bm.block_size) - have)
        return need

    def _preempt_newest(self) -> None:
        """Evict the most recently admitted request and recompute it later.

        Newest-first is deliberate: it has the least accumulated work to
        redo, and it protects the older requests' latency. The victim goes to
        the FRONT of the queue so eviction can't starve it.
        """
        victim = self.running.pop()
        self.bm.free(victim.req_id)
        victim.preempted += 1
        self.waiting.appendleft(victim)
        self.preemptions += 1

    @torch.no_grad()
    def _decode(self) -> None:
        if not self.running:
            return

        while self._blocks_for_one_more() > len(self.bm.free_blocks):
            if len(self.running) == 1:
                raise MemoryError(
                    "KV pool cannot hold even one sequence; raise num_blocks")
            self._preempt_newest()

        # Position of the token about to be written, per sequence — captured
        # BEFORE allocate() bumps seq_lens.
        positions = torch.tensor([[self.bm.seq_lens[r.req_id]] for r in self.running],
                                 device=self.device, dtype=torch.long)
        write = torch.tensor([self.bm.allocate(r.req_id, 1)[0] for r in self.running],
                             device=self.device, dtype=torch.long)
        read, mask = batched_read_slots(
            self.bm, [r.req_id for r in self.running], self.device)

        ids = torch.tensor([[r.output_ids[-1]] for r in self.running],
                           device=self.device)

        logits = self.model(ids, paged=(self.cache, write, read),
                            positions=positions, pad_mask=mask)

        for i, r in enumerate(self.running):
            # Guard on `finished` for the benefit of a static scheduler, which
            # holds done sequences in the batch until the whole wave completes:
            # the GPU still computes their row, but the token is discarded.
            if not r.finished:
                r.output_ids.append(sample(logits[i, -1], temperature=0.0))

        self.batch_sizes.append(len(self.running))
        self.useful_sizes.append(sum(1 for r in self.running if not r.finished))

    # --------------------------------------------------------------- retire

    def _retire(self) -> None:
        still = []
        for r in self.running:
            hit_eos = r.output_ids[-1] in self.eos_ids
            hit_cap = len(r.output_ids) >= r.max_new_tokens
            if hit_eos or hit_cap:
                if hit_eos:
                    r.output_ids.pop()          # don't surface the stop token
                r.finished = True
                self.bm.free(r.req_id)
                self.done.append(r)
            else:
                still.append(r)
        self.running = still

    # ----------------------------------------------------------------- step

    def step(self) -> None:
        """One scheduler tick: admit, decode the whole batch, retire finishers."""
        self._admit()
        self._retire()      # a prefill can already be terminal (EOS, or cap of 1)
        self._decode()
        self._retire()
        self.steps += 1

    def run(self, max_steps: int = 10_000) -> list[Request]:
        while self.pending and self.steps < max_steps:
            self.step()
        return self.done

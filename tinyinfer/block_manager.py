class BlockManager:
    """Allocates fixed-size KV blocks to sequences, OS-page-table style.

    Physical storage is a flat pool of `num_blocks * block_size` token slots.
    A sequence's KV is scattered across any free blocks; its block_table maps
    logical block index -> physical block id. Sequences therefore need
    contiguity only in the TABLE, never in memory.
    """

    def __init__(self, num_blocks: int, block_size: int = 16):
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.free_blocks = list(range(num_blocks))
        self.block_tables: dict[int, list[int]] = {}   # seq_id -> physical block ids
        self.seq_lens: dict[int, int] = {}

    def _blocks_needed(self, seq_id: int, n_new: int) -> int:
        table = self.block_tables.get(seq_id, [])
        used = len(table) * self.block_size
        # tokens already written into the last (partially filled) block
        filled = self.seq_lens.get(seq_id, 0)
        return max(0, -(-(filled + n_new) // self.block_size) - len(table))

    def can_allocate(self, seq_id: int, n_new: int) -> bool:
        return self._blocks_needed(seq_id, n_new) <= len(self.free_blocks)

    def allocate(self, seq_id: int, n_new: int) -> list[int]:
        """Reserve blocks for n_new tokens; return the flat slot ids to write into."""
        self.block_tables.setdefault(seq_id, [])
        self.seq_lens.setdefault(seq_id, 0)
        need = self._blocks_needed(seq_id, n_new)
        if need > len(self.free_blocks):
            raise MemoryError(f"KV pool exhausted: need {need}, free {len(self.free_blocks)}")
        for _ in range(need):
            self.block_tables[seq_id].append(self.free_blocks.pop())

        start = self.seq_lens[seq_id]
        slots = [self.slot(seq_id, start + i) for i in range(n_new)]
        self.seq_lens[seq_id] = start + n_new
        return slots

    def slot(self, seq_id: int, logical_pos: int) -> int:
        """logical token position -> physical flat slot id. The core address translation."""
        block = self.block_tables[seq_id][logical_pos // self.block_size]
        return block * self.block_size + (logical_pos % self.block_size)

    def slots_for(self, seq_id: int) -> list[int]:
        """All physical slots holding this sequence's KV, in logical order (for the gather)."""
        return [self.slot(seq_id, i) for i in range(self.seq_lens[seq_id])]

    def free(self, seq_id: int) -> None:
        for b in self.block_tables.pop(seq_id, []):
            self.free_blocks.append(b)
        self.seq_lens.pop(seq_id, None)

    def utilization(self) -> float:
        used_tokens = sum(self.seq_lens.values())
        capacity = self.num_blocks * self.block_size
        return used_tokens / capacity
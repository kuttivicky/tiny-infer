from tinyinfer.block_manager import BlockManager

bm = BlockManager(num_blocks=8, block_size=4)   # tiny, so you can trace by hand
print(bm.allocate(seq_id=0, n_new=6))   # 6 tokens -> 2 blocks
print(bm.block_tables, bm.free_blocks)
print(bm.allocate(seq_id=1, n_new=3))   # interleaved second sequence
print(bm.allocate(seq_id=0, n_new=1))   # seq 0 grows by one decode token
print("util:", bm.utilization())
bm.free(0)
print("after free:", bm.block_tables, bm.free_blocks)
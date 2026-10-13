# GPU Miner

CUDA implementation of the proof-of-work miner. Produces the same results as `../cpu_miner`, only faster.

## Implementation (`src/utils.cu`)

- **Merkle root** – `compute_hashes` hashes every transaction in parallel, then `reduce_tree` is launched once per tree level (one thread per pair), alternating between two device buffers.
- **Nonce search** – `find_nonce_kernel` uses a grid-stride loop (1024 blocks x 256 threads). Each thread builds `block_content + nonce` locally and hashes it. Valid nonces are reduced with `atomicMin` so the smallest one wins, and threads exit early once one is found.
- **Housekeeping** – device buffers are allocated lazily once and reused; `warm_up_gpu()` initializes the CUDA context before timing.
- `src/sha256.cu` provides `__host__ __device__` SHA-256.

## Usage

```bash
mkdir -p output
make                 # compile with nvcc (via sbatch, requires a GPU node)
make run TEST=test4  # run a test from ../tests
make clean
```

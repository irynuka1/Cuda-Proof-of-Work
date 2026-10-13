# CUDA Proof-of-Work Miner

A simplified blockchain miner that builds blocks from batches of transactions and finds a valid proof-of-work nonce for each one. It ships two interchangeable implementations of the same pipeline: a sequential **CPU** reference and a parallel **CUDA GPU** version.

## How it works

For every block, the miner runs two stages:

1. **Merkle root** – each transaction is hashed with SHA-256, then hashes are combined pairwise (duplicating the last one when the count is odd) until a single root remains.
2. **Nonce search** – the content `previous_block_hash + merkle_root + nonce` is hashed repeatedly with an increasing nonce until the hex digest is lexicographically `<=` the difficulty target (a string starting with N `0`s followed by `f`s).

The resulting block hash becomes the `previous_block_hash` of the next block, forming the chain.

### GPU parallelization

| Stage | CPU | GPU (`gpu_miner/src/utils.cu`) |
|-------|-----|--------------------------------|
| Transaction hashing | Sequential loop | One thread per transaction |
| Merkle tree | Nested loops | One kernel launch per tree level, one thread per pair, ping-pong buffers |
| Nonce search | Single `for` loop | Grid-stride loop over the nonce space; `atomicMin` keeps the smallest valid nonce, and an early-exit flag stops other threads |

Device memory is allocated once and reused across blocks, and a warm-up kernel runs before timing starts.


## Project structure

```
.
├── common/miner.cpp      # Shared driver: input parsing, block loop, timing, output
├── include/              # sha256.h (SHA-256 API), utils.h (constants + miner API)
├── cpu_miner/            # Sequential C implementation (gcc)
├── gpu_miner/            # CUDA implementation (nvcc)
├── tests/                # Input files (test1 – test4)
└── Makefile              # Shared build rules (Slurm and local targets)
```

## Input format

The first line of a test file is comma separated:

```
<num_transactions>,<prev_block_hash>,<difficulty_zeros>,<max_nonce>,<max_tx_per_block>,<tx_size>
```

Each following line is one transaction. A `max_nonce` of `0` means `UINT32_MAX`.

| Test | Transactions | Difficulty (leading zeros) | Max tx / block |
|------|-------------:|---------------------------:|---------------:|
| test1 | 35 | 5 | 10 |
| test2 | 55 | 5 | 10 |
| test3 | 10 500 | 7 | 1 000 |
| test4 | 100 000 | 4 | 10 000 |

## Build and run

The Makefiles submit jobs to a Slurm cluster (partition `ucsx`; the GPU build loads `libraries/cuda-13.1`). Run from inside `cpu_miner/` or `gpu_miner/`:

```bash
mkdir -p output          # results are appended here
make                     # compile (sbatch)
make run TEST=test4      # run one test
make clean               # remove binaries and outputs
```

Overridable variables: `PARTITION` (default `ucsx`) and `TIME` (default `00:01:00`).

### Local build (no Slurm)

On a local machine, skip `sbatch` and use the `local` targets. They compile and run directly with the toolchain on your `PATH` (`gcc` for the CPU miner, `nvcc` plus an NVIDIA GPU for the GPU miner):

```bash
cd cpu_miner                # or gpu_miner
make local                  # compile only
make local-run TEST=test1   # compile, create output/ and run
make clean
```

## Output

Results go to `output/<TEST>.out`, one line per block:

```
<block_no>,<block_hash>,<merkle_root>,<nonce>,<merkle_time>,<nonce_time>,<block_time>
```

The last line holds the totals: `<total_merkle_time>,<total_nonce_time>,<total_time>`. Comparing this line between the CPU and GPU builds shows the speed-up.

## Credits

SHA-256 implementation by Brad Conte (public domain).

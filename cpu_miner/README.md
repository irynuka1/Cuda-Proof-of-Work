# CPU Miner

Sequential reference implementation of the proof-of-work miner, written in C.

- `src/utils.c` – Merkle root construction and linear nonce search (`nonce = 0, 1, 2, ...`).
- `src/sha256.c` – SHA-256 implementation.
- Shared driver: `../common/miner.cpp` (compiled as C with `-x c`).

## Usage

```bash
mkdir -p output
make                 # compile with gcc (via sbatch)
make run TEST=test1  # run a test from ../tests
make clean
```

Use its output as the correctness and performance baseline for `../gpu_miner`.

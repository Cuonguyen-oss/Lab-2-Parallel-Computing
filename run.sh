#!/bin/bash
# usage: ./run.sh <matrix.csv> [NUM_PROCESSES=4] [extra options, e.g. --iters 2]
set -e
cd "$(dirname "$0")"
CSV=${1:-radioactive_matrix.csv}; NP=${2:-4}; shift 2 || true
./seq_sim "$CSV" "$@" --out result_seq.bin
mpirun --oversubscribe -np "$NP" ./mpi_naive "$CSV" "$@" --out result_naive.bin
mpirun --oversubscribe -np "$NP" ./mpi_sim   "$CSV" "$@" --out result_opt.bin
cmp result_seq.bin result_naive.bin && echo "VERIFY naive: identical to sequential"
cmp result_seq.bin result_opt.bin   && echo "VERIFY optimized: identical to sequential"
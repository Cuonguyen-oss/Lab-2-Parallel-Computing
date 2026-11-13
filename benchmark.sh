#!/bin/bash
# usage: ./benchmark.sh <matrix.csv> "1 2 4 8" [extra options, e.g. --iters 10]
cd "$(dirname "$0")"
CSV=${1:-radioactive_matrix.csv}; PROCS=${2:-"1 2 4 8"}; shift 2 2>/dev/null
TS=$(./seq_sim "$CSV" "$@" | awk '/computation time/{print $(NF-1)}')
echo "version,procs,seq_time_s,par_compute_s,par_total_s,speedup_compute,speedup_total" > results.csv
for V in mpi_naive mpi_sim; do
  for P in $PROCS; do
    OUT=$(mpirun --oversubscribe -np "$P" ./$V "$CSV" "$@")
    TC=$(echo "$OUT" | awk '/computation\+halo/{print $(NF-1)}')
    TT=$(echo "$OUT" | awk '/total parallel/{print $(NF-1)}')
    echo "$V,$P,$TS,$TC,$TT,$(awk -v a=$TS -v b=$TC 'BEGIN{printf "%.4f",a/b}'),$(awk -v a=$TS -v b=$TT 'BEGIN{printf "%.4f",a/b}')" | tee -a results.csv
  done
done
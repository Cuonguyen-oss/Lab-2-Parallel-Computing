Lab 2 - Open MPI: radioactive contamination (advection-diffusion-decay)

FILES
  common.h      : parameters, stencil kernel, CSV reader (shared by all versions)
  seq.cpp       : sequential version
  mpi_naive.cpp : parallel (basic): blocking Send/Recv halo exchange, no overlap
  mpi_par.cpp   : parallel (optimized): non-blocking Isend/Irecv, overlaps
                  interior computation with halo communication
  compile.sh    : build all versions -> ./seq_sim ./mpi_naive ./mpi_sim
  run.sh        : run seq + both parallel versions, verify results are identical
  benchmark.sh  : time seq + both parallel versions for several process counts

REQUIREMENTS
  g++ and Open MPI (mpicxx, mpirun).
  Ubuntu/WSL: sudo apt install openmpi-bin libopenmpi-dev
  The input file radioactive_matrix.csv (4000x4000) is provided by the course
  and is not included in the zip; put it in this folder or pass its full path.

BUILD
  ./compile.sh

RUN (full problem: 4000x4000 map, 100 iterations, 50 substeps/iteration)
  Single version:
    ./seq_sim radioactive_matrix.csv
    mpirun --oversubscribe -np 4 ./mpi_naive radioactive_matrix.csv
    mpirun --oversubscribe -np 4 ./mpi_sim   radioactive_matrix.csv
  Run all three and verify the parallel results are bit-identical to seq:
    ./run.sh radioactive_matrix.csv 4
  Benchmark (writes results.csv, used for the report table):
    ./benchmark.sh radioactive_matrix.csv "1 2 4 8"
  Do NOT run "mpirun ... ./benchmark.sh": the script calls mpirun itself.
  Multi-node: add --hostfile hosts.txt to mpirun.

QUICK TEST (about a minute instead of tens of minutes)
    ./run.sh radioactive_matrix.csv 4 --iters 2
    ./benchmark.sh radioactive_matrix.csv "1 2 4 8" --iters 5
  Expected: "VERIFY naive: identical to sequential" and
            "VERIFY optimized: identical to sequential"

OUTPUT
  Every program prints the number of uncontaminated cells (C < eps) at
  iteration 0 and every --print-every iterations. seq prints the computation
  time; the MPI programs print the computation+halo time (max over ranks) and
  the total parallel time (scatter..gather).
  benchmark.sh writes results.csv:
    version,procs,seq_time_s,par_compute_s,par_total_s,speedup_compute,speedup_total

OPTIONS (all programs)
  --iters N         number of iterations, 1 iteration = 1 simulated second (default 100)
  --substeps S      explicit steps per iteration, dt = 1/S (default 50, dt = 0.02 s)
  --eps E           threshold for "uncontaminated" cells (default 1e-6)
  --print-every K   print the uncontaminated count every K iterations (default 10)
  --out file.bin    write the final map as raw doubles (row-major, 4000x4000)

NOTES
  - The full benchmark takes about 25-30 minutes; use --iters for quick runs.
  - dt = 0.02 s is needed for stability of the explicit scheme
    (D*dt/dx^2 = 0.2 < 0.25).
  - The number of rows need not be divisible by the number of processes:
    the map is padded with zero rows (zero boundary condition is preserved).
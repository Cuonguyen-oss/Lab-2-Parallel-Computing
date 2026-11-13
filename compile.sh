#!/bin/bash
# Compile the sequential and the Open MPI versions.
set -e
cd "$(dirname "$0")"
FLAGS="-O3 -march=native -ffp-contract=off -std=c++11"
g++    $FLAGS -o seq_sim seq.cpp
mpicxx $FLAGS -o mpi_sim mpi_par.cpp
mpicxx $FLAGS -o mpi_naive mpi_naive.cpp
echo "Built: ./seq_sim  ./mpi_naive  ./mpi_sim"
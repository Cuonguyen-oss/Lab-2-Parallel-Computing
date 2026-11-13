// mpi_naive.cpp - dummy parallel: blocking Send/Recv, no overlap
#include <mpi.h>
#include <algorithm>
#include "common.h"

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    Params p;
    std::vector<double> m;
    int cfg[5] = {0, 0, 0, 0, 0};
    if (rank == 0) {
        bool ok = parse_args(argc, argv, p) && read_csv(p.csv, m, cfg[0], cfg[1]) && cfg[1] >= 2;
        if (!ok) { fprintf(stderr, "cannot read input\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
        cfg[2] = p.iters; cfg[3] = p.substeps; cfg[4] = p.print_every;
    }
    MPI_Bcast(cfg, 5, MPI_INT, 0, MPI_COMM_WORLD);
    const int R = cfg[0], W = cfg[1], iters = cfg[2], substeps = cfg[3];
    if (rank != 0) { p.iters = iters; p.substeps = substeps; p.print_every = cfg[4]; }
    double eps = p.eps;
    MPI_Bcast(&eps, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    const int chunk = (R + size - 1) / size;
    const int valid = std::max(0, std::min(chunk, R - rank * chunk));
    if (rank == 0)
        printf("[NAIVE] map %d x %d, processes=%d, rows/process=%d, iterations=%d, substeps=%d\n",
               R, W, size, chunk, iters, substeps);

    std::vector<double> full;
    if (rank == 0) {
        full.assign((size_t)chunk * size * W, 0.0);
        memcpy(full.data(), m.data(), sizeof(double) * (size_t)R * W);
        std::vector<double>().swap(m);
    }
    std::vector<double> A((size_t)(chunk + 2) * W, 0.0), B((size_t)(chunk + 2) * W, 0.0);
    Coef k = make_coef(p, 1.0 / substeps);
    const int up = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
    const int down = (rank < size - 1) ? rank + 1 : MPI_PROC_NULL;
    const int TAG_UP = 1, TAG_DOWN = 2;

    MPI_Barrier(MPI_COMM_WORLD);
    double t_total0 = MPI_Wtime();

    MPI_Scatter(rank == 0 ? full.data() : nullptr, chunk * W, MPI_DOUBLE,
                &A[W], chunk * W, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    auto report = [&](int it, const double* cur, bool print) {
        long long local = count_clean(cur + W, (size_t)valid * W, eps), total = 0;
        MPI_Reduce(&local, &total, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
        if (rank == 0 && print) printf("iter %3d: uncontaminated cells = %lld\n", it, total);
    };
    report(0, A.data(), true);

    double* cur = A.data(); double* nxt = B.data();
    double t_comp = 0.0;
    for (int it = 1; it <= iters; ++it) {
        double tc0 = MPI_Wtime();
        for (int s = 0; s < substeps; ++s) {
            // blocking halo exchange; even/odd ordering avoids deadlock
            if (rank % 2 == 0) {
                MPI_Send(cur + W,                       W, MPI_DOUBLE, up,   TAG_UP,   MPI_COMM_WORLD);
                MPI_Send(cur + (size_t)chunk * W,       W, MPI_DOUBLE, down, TAG_DOWN, MPI_COMM_WORLD);
                MPI_Recv(cur,                           W, MPI_DOUBLE, up,   TAG_DOWN, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(cur + (size_t)(chunk + 1) * W, W, MPI_DOUBLE, down, TAG_UP,   MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            } else {
                MPI_Recv(cur,                           W, MPI_DOUBLE, up,   TAG_DOWN, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(cur + (size_t)(chunk + 1) * W, W, MPI_DOUBLE, down, TAG_UP,   MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Send(cur + W,                       W, MPI_DOUBLE, up,   TAG_UP,   MPI_COMM_WORLD);
                MPI_Send(cur + (size_t)chunk * W,       W, MPI_DOUBLE, down, TAG_DOWN, MPI_COMM_WORLD);
            }
            step_rows(cur, nxt, W, 1, chunk + 1, k);   // whole block after halo arrives
            if (valid < chunk) memset(nxt + (size_t)(valid + 1) * W, 0, sizeof(double) * (size_t)(chunk - valid) * W);
            std::swap(cur, nxt);
        }
        t_comp += MPI_Wtime() - tc0;
        report(it, cur, it % p.print_every == 0 || it == iters);
        MPI_Barrier(MPI_COMM_WORLD);
    }

    int stop = (rank == 0);
    MPI_Bcast(&stop, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (stop) MPI_Gather(cur + W, chunk * W, MPI_DOUBLE,
                         rank == 0 ? full.data() : nullptr, chunk * W, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    double t_total = MPI_Wtime() - t_total0;

    double tc_max = 0.0;
    MPI_Reduce(&t_comp, &tc_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("[NAIVE] computation+halo time (max over ranks) = %.3f s\n", tc_max);
        printf("[NAIVE] total parallel time (scatter..gather)  = %.3f s\n", t_total);
        if (!p.out.empty()) {
            FILE* f = fopen(p.out.c_str(), "wb");
            fwrite(full.data(), sizeof(double), (size_t)R * W, f); fclose(f);
        }
    }
    MPI_Finalize();
    return 0;
}
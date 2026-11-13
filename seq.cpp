// seq.cpp - sequential reference implementation
#include <chrono>
#include "common.h"

int main(int argc, char** argv) {
    Params p;
    if (!parse_args(argc, argv, p)) return 1;
    std::vector<double> m; int R, W;
    if (!read_csv(p.csv, m, R, W) || W < 2) { fprintf(stderr, "cannot read %s\n", p.csv.c_str()); return 1; }
    printf("[SEQ] map %d x %d, iterations=%d, substeps=%d (dt=%g s)\n", R, W, p.iters, p.substeps, 1.0 / p.substeps);

    std::vector<double> A((size_t)(R + 2) * W, 0.0), B((size_t)(R + 2) * W, 0.0);
    memcpy(&A[W], m.data(), sizeof(double) * (size_t)R * W);
    std::vector<double>().swap(m);
    Coef k = make_coef(p, 1.0 / p.substeps);

    printf("iter %3d: uncontaminated cells = %lld\n", 0, count_clean(&A[W], (size_t)R * W, p.eps));
    auto t0 = std::chrono::steady_clock::now();
    for (int it = 1; it <= p.iters; ++it) {
        for (int s = 0; s < p.substeps; ++s) { step_rows(A.data(), B.data(), W, 1, R + 1, k); A.swap(B); }
        if (it % p.print_every == 0 || it == p.iters)
            printf("iter %3d: uncontaminated cells = %lld\n", it, count_clean(&A[W], (size_t)R * W, p.eps));
    }
    double t = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    printf("[SEQ] computation time = %.3f s\n", t);
    if (!p.out.empty()) {
        FILE* f = fopen(p.out.c_str(), "wb");
        fwrite(&A[W], sizeof(double), (size_t)R * W, f); fclose(f);
    }
    return 0;
}

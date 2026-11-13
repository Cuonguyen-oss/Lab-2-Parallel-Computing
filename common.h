// common.h - shared by the sequential and MPI versions (Lab 2, CO3068)
// Radioactive contamination: advection (upwind) - diffusion - decay/deposition
//   dC/dt = -u.grad(C) + D*laplace(C) - (lambda+k)*C
// Conventions: x = column index (West->East), y = row index (North->South).
// With u = (3.3, 1.4) > 0 the upwind neighbours are the cell to the West
// (column-1) and the cell to the North (row-1). Outside the map C = 0.
#pragma once
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

struct Params {
    double D = 1000.0, k = 1e-4, lambda = 3e-5;   // coefficients from the lab
    double ux = 3.3, uy = 1.4;                    // wind
    double dx = 10.0, dy = 10.0;                  // cell size [m]
    int    iters = 100;                           // 1 iteration = 1 simulated second
    int    substeps = 50;                         // explicit steps per iteration, dt = 1/substeps
    double eps = 1e-6;                            // C < eps  => cell is "uncontaminated"
    int    print_every = 10;
    std::string csv, out;
};

struct Coef { double cC, cW, cN, cE, cS; };

// Cnew = C + dt*[ -ux(C-Cw)/dx - uy(C-Cn)/dy + D(Ce-2C+Cw)/dx^2 + D(Cs-2C+Cn)/dy^2 - (lambda+k)C ]
// collected per neighbour (algebraically identical to the formulas in the handout).
inline Coef make_coef(const Params& p, double dt) {
    Coef c;
    c.cW = dt * (p.ux / p.dx + p.D / (p.dx * p.dx));
    c.cE = dt * (p.D / (p.dx * p.dx));
    c.cN = dt * (p.uy / p.dy + p.D / (p.dy * p.dy));
    c.cS = dt * (p.D / (p.dy * p.dy));
    c.cC = 1.0 - dt * (p.ux / p.dx + p.uy / p.dy + 2 * p.D / (p.dx * p.dx)
                       + 2 * p.D / (p.dy * p.dy) + p.lambda + p.k);
    return c;
}

// Block layout: (L+2) rows x W cols, row 0 and row L+1 are halo rows.
// Updates local rows [r0, r1) of `in` into `out`. Left/right map edges use 0.
inline void step_rows(const double* in, double* out, int W, int r0, int r1, const Coef& k) {
    for (int r = r0; r < r1; ++r) {
        const double* c = in + (size_t)r * W;
        const double* n = c - W;
        const double* s = c + W;
        double* o = out + (size_t)r * W;
        o[0] = k.cC * c[0] + k.cE * c[1] + k.cN * n[0] + k.cS * s[0];
        for (int j = 1; j < W - 1; ++j)
            o[j] = k.cC * c[j] + k.cW * c[j - 1] + k.cE * c[j + 1] + k.cN * n[j] + k.cS * s[j];
        o[W - 1] = k.cC * c[W - 1] + k.cW * c[W - 2] + k.cN * n[W - 1] + k.cS * s[W - 1];
    }
}

inline long long count_clean(const double* rows, size_t n, double eps) {
    long long cnt = 0;
    for (size_t i = 0; i < n; ++i) cnt += (rows[i] < eps);
    return cnt;
}

// Fast CSV reader (rows = lines, cols = comma separated values).
inline bool read_csv(const std::string& path, std::vector<double>& data, int& R, int& W) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return false;
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    std::string buf(sz + 1, '\0');
    if (fread(&buf[0], 1, sz, f) != (size_t)sz) { fclose(f); return false; }
    fclose(f);
    data.clear(); data.reserve(sz / 5);
    R = 0; W = 0; int cur = 0;
    const char* p = buf.c_str(); const char* end = p + sz;
    while (p < end) {
        if (*p == ',' || *p == ' ' || *p == '\r') { ++p; continue; }
        if (*p == '\n') {
            if (cur > 0) { if (R == 0) W = cur; ++R; cur = 0; }
            ++p; continue;
        }
        char* e; data.push_back(strtod(p, &e));
        if (e == p) return false;
        p = e; ++cur;
    }
    if (cur > 0) { if (R == 0) W = cur; ++R; }
    return (size_t)R * W == data.size();
}

inline bool parse_args(int argc, char** argv, Params& p) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s matrix.csv [--iters N] [--substeps S] [--eps E] [--print-every K] [--out result.bin]\n", argv[0]);
        return false;
    }
    p.csv = argv[1];
    for (int i = 2; i + 1 < argc; i += 2) {
        std::string a = argv[i];
        if (a == "--iters") p.iters = atoi(argv[i + 1]);
        else if (a == "--substeps") p.substeps = atoi(argv[i + 1]);
        else if (a == "--eps") p.eps = atof(argv[i + 1]);
        else if (a == "--print-every") p.print_every = atoi(argv[i + 1]);
        else if (a == "--out") p.out = argv[i + 1];
        else { fprintf(stderr, "unknown option %s\n", a.c_str()); return false; }
    }
    return true;
}

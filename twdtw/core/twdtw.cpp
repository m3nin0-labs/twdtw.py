#include "twdtw.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <algorithm>
#include <cmath>
#include <limits>

namespace twdtw {

// Cyclic distance between two in-cycle times (e.g. days of year)
static inline double cyclic_dt(double a, double b, double cycle_length) {
    double d = std::fabs(a - b);

    return std::min(d, cycle_length - d);
}

// Euclidean distance over the bands plus the logistic time penalty
static inline double local_cost(const double *y, const double *x, int D, double dt, double alpha,
                                double beta) {
    double sq = 0.0;

    for (int k = 1; k < D; ++k) {
        double diff = y[k] - x[k];

        sq += diff * diff;
    }

    return std::sqrt(sq) + 1.0 / (1.0 + std::exp(-alpha * (dt - beta)));
}

void compute(const double *XM, const double *YM, double *CM, std::int32_t *DM, std::int32_t *VM,
             std::int32_t *JB, int N, int M, int D, double alpha, double beta, double max_elapsed,
             double cycle_length) {
    const double INF = std::numeric_limits<double>::infinity();
    const std::int32_t NONE = -1;

    auto cm = [&](int r, int j) -> double & { return CM[r * M + j]; };
    auto dm = [&](int r, int j) -> std::int32_t & { return DM[r * M + j]; };
    auto vm = [&](int r, int j) -> std::int32_t & { return VM[r * M + j]; };
    auto y = [&](int i) { return &YM[i * D]; };
    auto x = [&](int j) { return &XM[j * D]; };

    // open-begin boundary row: zero cost, each column is its own origin
    for (int j = 0; j < M; ++j) {
        cm(0, j) = 0.0;
        dm(0, j) = 0;
        vm(0, j) = j;
    }

    // first column: the pattern is consumed vertically from the series start
    for (int i = 0; i < N; ++i) {
        double dt = cyclic_dt(y(i)[0], x(0)[0], cycle_length);

        cm(i + 1, 0) = cm(i, 0) + local_cost(y(i), x(0), D, dt, alpha, beta);
        dm(i + 1, 0) = 3;
        vm(i + 1, 0) = 0;
    }

    // fill the cost matrix column by column
    for (int j = 1; j < M; ++j) {
        for (int i = 0; i < N; ++i) {
            // row index
            int r = i + 1;

            // cyclic elapsed time between the pattern and series
            double dt = cyclic_dt(y(i)[0], x(j)[0], cycle_length);

            // if the elapsed time is greater than the maximum allowed,
            // set the cost to infinity
            if (dt > max_elapsed) {
                cm(r, j) = INF;
                dm(r, j) = NONE;
                vm(r, j) = NONE;

                continue;
            }

            // local cost of the match
            double cp = local_cost(y(i), x(j), D, dt, alpha, beta);

            // accumulated cost of the match
            cm(r, j) = cp + cm(r - 1, j - 1);  // diagonal
            dm(r, j) = 1;
            vm(r, j) = vm(r - 1, j - 1);

            // accumulated cost of the match from the left
            double left = cp + cm(r, j - 1);
            if (left < cm(r, j)) {
                cm(r, j) = left;
                dm(r, j) = 2;
                vm(r, j) = vm(r, j - 1);
            }

            // accumulated cost of the match from the bottom
            double down = cp + cm(r - 1, j);
            if (down < cm(r, j)) {
                cm(r, j) = down;
                dm(r, j) = 3;
                vm(r, j) = vm(r - 1, j);
            }
        }
    }

    // best matches: walk the last row, group consecutive columns by origin, and
    // keep the lowest-cost column within each group. JB is sized M, the largest
    // possible number of groups.
    for (int j = 0; j < M; ++j) {
        JB[j] = NONE;
    }

    // best matches: walk the last row, group consecutive columns by origin, and
    int k = -1;
    std::int32_t origin = NONE;

    for (int j = 0; j < M; ++j) {
        std::int32_t v = vm(N, j);

        // if the origin is none, skip
        if (v == NONE) {
            continue;
        }

        // if the origin is different from the previous, add a new group
        if (k == -1 || v != origin) {
            JB[++k] = j;
            origin = v;
        } else if (cm(N, j) < cm(N, JB[k])) {
            JB[k] = j;
        }
    }
}

}  // namespace twdtw

namespace nb = nanobind;

using f64_2d = nb::ndarray<double, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
using i32_2d = nb::ndarray<std::int32_t, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
using i32_1d = nb::ndarray<std::int32_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

static void twdtw_core(f64_2d XM, f64_2d YM, f64_2d CM, i32_2d DM, i32_2d VM, i32_1d JB,
                       double alpha, double beta, double max_elapsed, double cycle_length) {
    int M = static_cast<int>(XM.shape(0));
    int N = static_cast<int>(YM.shape(0));
    int D = static_cast<int>(XM.shape(1));

    // run the TWDTW algorithm
    twdtw::compute(XM.data(), YM.data(), CM.data(), DM.data(), VM.data(), JB.data(), N, M, D, alpha,
                   beta, max_elapsed, cycle_length);
}

NB_MODULE(corecpp, m) {
    m.def("twdtw_core", &twdtw_core, nb::arg("XM"), nb::arg("YM"), nb::arg("CM"), nb::arg("DM"),
          nb::arg("VM"), nb::arg("JB"), nb::arg("alpha"), nb::arg("beta"), nb::arg("max_elapsed"),
          nb::arg("cycle_length"),
          "Run the TWDTW dynamic program, writing into preallocated CM/DM/VM/JB.");
}

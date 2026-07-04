#pragma once
#include <cstdint>

namespace twdtw {

// TWDTW algorithm.
//
// XM is the long time series (M rows), YM the temporal pattern (N rows). Both
// have D columns where column 0 is the numeric in-cycle time and columns 1...D-1
// are the band values. The caller preallocates the outputs, all stored
// row-major: CM/DM/VM are (N+1, M) and JB is (M,). Row 0 of CM/DM/VM is the
// open-begin boundary, so a match may start at any series column.
//
//   CM  accumulated cost          DM  step direction (1 diag, 2 left, 3 down)
//   VM  origin column of the path JB  end column of each best match (-1 = none)
void compute(const double *XM, const double *YM, double *CM, std::int32_t *DM, std::int32_t *VM,
             std::int32_t *JB, int N, int M, int D, double alpha, double beta, double max_elapsed,
             double cycle_length);

}  // namespace twdtw

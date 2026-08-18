#pragma once

/**
 * Batched lgamma over a contiguous buffer.
 *
 * Computing lgamma for a run of independent inputs in one tight loop lets an
 * out-of-order core overlap many lgamma pipelines, which is markedly faster than
 * interleaving each lgamma with the dependent reduction that consumes it. The
 * omp-simd hint additionally lets glibc's libmvec supply a vectorized lgamma on
 * Linux/x86; where no vector lgamma exists (e.g. Apple libm) it stays scalar.
 *
 * Portable and dependency-free: results equal std::lgamma on every platform.
 */

#include <cmath>

static inline void lgamma_batch(const double* in, double* out, int n)
{
#if defined(_OPENMP)
#pragma omp simd
#endif
    for (int i = 0; i < n; ++i) out[i] = std::lgamma(in[i]);
}

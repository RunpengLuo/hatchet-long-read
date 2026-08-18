/**
 * Log-likelihood kernel for the 2-mixture BAF+RDR HMM.
 *
 * BAF: Beta-Binomial (lgamma-based log betaln).
 * RDR: Gaussian (closed-form log-normal constant).
 * Parallelised over bins with OpenMP; per-bin (k, m) loop is sequential.
 *
 * Data-invariant terms (log_bc_nm) and the share_tau lgamma(total+tau) buffer
 * are precomputed by the caller (run_hmm_cpp) and passed in, so they are not
 * rebuilt every EM iteration.
 */

#include "loglik.h"
#include "lgamma_batch.h"

#include <cmath>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

static const double LOG2PI = std::log(2.0 * M_PI);

static inline double betaln_cpp(double a, double b) {
    return std::lgamma(a) + std::lgamma(b) - std::lgamma(a + b);
}


void compute_loglik_cpp(
    const double* X_rdrs,
    const double* X_alphas,
    const double* X_betas,
    const double* X_totals,
    const double* rdr_means,
    const double* rdr_vars,
    const double* baf_means,
    const double* baf_taus,
    const double* log_bc_nm,           // (N, M) data-invariant, precomputed
    const double* lgamma_tot_tau_nm,   // (N, M) share_tau only, precomputed; else nullptr
    double*       lls0,
    double*       lls1,
    int N, int K, int M, bool share_tau)
{
    // Per-(k,m) constants: bb_alpha, bb_beta, bb_delta, and per-k Gaussian norm.
    // Small (K*M); recomputed each call because emission params change.
    std::vector<double> bb_alpha(K * M);
    std::vector<double> bb_beta(K * M);
    std::vector<double> bb_delta(K * M);
    std::vector<double> log_norm_const(K, 0.0);

    for (int k = 0; k < K; ++k) {
        for (int m = 0; m < M; ++m) {
            double tau  = baf_taus[(long)k * M + m];
            double mean = baf_means[(long)k * M + m];
            double a    = tau * mean;
            double b    = tau * (1.0 - mean);
            bb_alpha[(long)k * M + m] = a;
            bb_beta[(long)k * M + m]  = b;
            bb_delta[(long)k * M + m] = betaln_cpp(a, b);
            log_norm_const[k] += 0.5 * (LOG2PI + std::log(rdr_vars[(long)k * M + m]));
        }
    }

    // Main loop: parallelise over bins. Per-thread scratch batches the BAF
    // lgamma arguments for all (k, m) at a fixed bin into one tight lgamma pass,
    // which overlaps lgamma pipelines far better than interleaving each call
    // with the reduction that consumes it.
#ifdef _OPENMP
#pragma omp parallel
#endif
    {
        const long KM = (long)K * M;
        std::vector<double> args(4 * KM);   // per (k,m): alpha+a, beta+b, beta+a, alpha+b
        std::vector<double> res(4 * KM);
        std::vector<double> tot_args, tot_res;
        if (!share_tau) { tot_args.resize(KM); tot_res.resize(KM); }

#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
        for (int n = 0; n < N; ++n) {
            for (int k = 0; k < K; ++k) {
                for (int m = 0; m < M; ++m) {
                    double alpha = X_alphas[(long)n * M + m];
                    double beta  = X_betas[(long)n * M + m];
                    double a     = bb_alpha[(long)k * M + m];
                    double b     = bb_beta[(long)k * M + m];
                    long   base  = 4 * ((long)k * M + m);
                    args[base + 0] = alpha + a;
                    args[base + 1] = beta + b;
                    args[base + 2] = beta + a;
                    args[base + 3] = alpha + b;
                    if (!share_tau)
                        tot_args[(long)k * M + m] =
                            X_totals[(long)n * M + m] + baf_taus[(long)k * M + m];
                }
            }

            lgamma_batch(args.data(), res.data(), (int)(4 * KM));
            if (!share_tau)
                lgamma_batch(tot_args.data(), tot_res.data(), (int)KM);

            for (int k = 0; k < K; ++k) {
                double ll_rdr    = -log_norm_const[k];
                double ll_baf_h0 = 0.0;
                double ll_baf_h1 = 0.0;

                for (int m = 0; m < M; ++m) {
                    double diff = X_rdrs[(long)n * M + m] - rdr_means[(long)k * M + m];
                    double var  = rdr_vars[(long)k * M + m];
                    ll_rdr -= 0.5 * diff * diff / var;

                    double log_bc = log_bc_nm[(long)n * M + m];
                    double delta  = bb_delta[(long)k * M + m];
                    double lg_tot_tau = share_tau
                        ? lgamma_tot_tau_nm[(long)n * M + m]
                        : tot_res[(long)k * M + m];
                    long base = 4 * ((long)k * M + m);

                    ll_baf_h0 += log_bc + res[base + 0] + res[base + 1] - lg_tot_tau - delta;
                    ll_baf_h1 += log_bc + res[base + 2] + res[base + 3] - lg_tot_tau - delta;
                }

                lls0[(long)n * K + k] = ll_rdr + ll_baf_h0;
                lls1[(long)n * K + k] = ll_rdr + ll_baf_h1;
            }
        }
    }
}

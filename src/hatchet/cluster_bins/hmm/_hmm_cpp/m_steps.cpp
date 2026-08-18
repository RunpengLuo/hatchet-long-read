/**
 * BAF M-step: MLE for Beta-Binomial means.
 *
 * For each (k, m) pair, maximizes the posterior-weighted Beta-Binomial
 * log-likelihood over p in (eps, 1-eps), with a = τp, b = τ(1-p):
 *
 *   Q(p) = C + Σ_n [ w0[n]*(lgamma(α_n+a) + lgamma(β_n+b))
 *                   + w1[n]*(lgamma(β_n+a) + lgamma(α_n+b)) ]
 *             - Wk * (lgamma(a) + lgamma(b))
 *
 * With τ fixed, a+b=τ is constant, so the normalizers lgamma(total+τ)/lgamma(τ)
 * drop out and Q'' = τ² Σ w [trigamma(count+a) - trigamma(a)] ≤ 0: Q is concave
 * and unimodal. The maximizer is found by a warm-started, boost-safeguarded
 * Newton iteration on Q'(p)=0 (derivatives in digamma/trigamma; no lgamma), with
 * closed-form boundary handling and a Brent fallback for numerical pathologies.
 *
 * The outer (k, m) loop is parallelised with OpenMP collapse(2).
 */

#include "m_steps.h"
#include "lgamma_batch.h"

#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>
#include <boost/math/special_functions/digamma.hpp>
#include <boost/math/special_functions/trigamma.hpp>
#include <boost/math/tools/minima.hpp>
#include <boost/math/tools/roots.hpp>

#ifdef _OPENMP
#include <omp.h>
#endif

// Root/optimizer precision (bits of the bracket). 22 bits ~= 1e-7 on the search
// interval, far tighter than any downstream use.
static const int SOLVE_BITS = 22;

// Batched digamma/trigamma: a tight loop overlaps many independent special-fn
// pipelines (same instruction-level-parallelism win as lgamma_batch).
static inline void digamma_batch(const double* in, double* out, int n) {
    for (int i = 0; i < n; ++i) out[i] = boost::math::digamma(in[i]);
}
static inline void trigamma_batch(const double* in, double* out, int n) {
    for (int i = 0; i < n; ++i) out[i] = boost::math::trigamma(in[i]);
}


void update_baf_means_cpp(
    const double* alphas_mn,   // (M, N)
    const double* betas_mn,    // (M, N)
    const double* posts_kn2,   // (K, N, 2)
    const double* baf_taus,    // (K, M)
    double*       p_km,        // (K, M)
    int N, int K, int M, double eps,
    int k_start)
{
    const double lo = eps, hi = 1.0 - eps;
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(dynamic)
#endif
    for (int k = k_start; k < K; ++k) {
        for (int m = 0; m < M; ++m) {
            const double* alpha_m = alphas_mn + (long)m * N;
            const double* beta_m  = betas_mn  + (long)m * N;
            // posts_kn2[k, n, h] = posts_kn2[k*N*2 + n*2 + h]
            const double* pkn2    = posts_kn2 + (long)k * N * 2;
            double tau = baf_taus[(long)k * M + m];

            // Wk = Σ_n (w0[n] + w1[n]); Newton needs no lgamma precompute.
            double Wk = 0.0;
            for (int n = 0; n < N; ++n) Wk += pkn2[n * 2 + 0] + pkn2[n * 2 + 1];

            // Thread-local scratch reused across evals and (k,m) pairs.
            static thread_local std::vector<double> qarg, qpsi, qpsi1;
            qarg.resize(4 * (size_t)N);
            qpsi.resize(4 * (size_t)N);
            qpsi1.resize(4 * (size_t)N);

            // Gradient/Hessian of Q at p, batched over the 4 shared points
            // {α_n+a, β_n+b, β_n+a, α_n+b}.
            auto grad_hess = [&](double p) -> std::pair<double, double> {
                double a = tau * p;
                double b = tau * (1.0 - p);
                for (int n = 0; n < N; ++n) {
                    double al = alpha_m[n];
                    double be = beta_m[n];
                    long base = 4 * (long)n;
                    qarg[base + 0] = al + a;
                    qarg[base + 1] = be + b;
                    qarg[base + 2] = be + a;
                    qarg[base + 3] = al + b;
                }
                digamma_batch(qarg.data(), qpsi.data(), 4 * N);
                trigamma_batch(qarg.data(), qpsi1.data(), 4 * N);
                double G = 0.0, H = 0.0;
                for (int n = 0; n < N; ++n) {
                    double w0 = pkn2[n * 2 + 0];
                    double w1 = pkn2[n * 2 + 1];
                    long base = 4 * (long)n;
                    G += w0 * (qpsi[base + 0] - qpsi[base + 1])
                       + w1 * (qpsi[base + 2] - qpsi[base + 3]);
                    H += w0 * (qpsi1[base + 0] + qpsi1[base + 1])
                       + w1 * (qpsi1[base + 2] + qpsi1[base + 3]);
                }
                double psia = boost::math::digamma(a), psib = boost::math::digamma(b);
                double t1a  = boost::math::trigamma(a), t1b = boost::math::trigamma(b);
                double Qp  = tau * (G - Wk * (psia - psib));
                double Qpp = tau * tau * (H - Wk * (t1a + t1b));
                return {Qp, Qpp};
            };

            // Concavity => the maximizer is pinned by the endpoint gradients.
            double p_star;
            if (grad_hess(lo).first <= 0.0) {
                p_star = lo;                       // Q' already <= 0 at lo
            } else if (grad_hess(hi).first >= 0.0) {
                p_star = hi;                       // Q' still >= 0 at hi
            } else {
                double guess = std::min(std::max(p_km[(long)k * M + m], lo), hi);
                std::uintmax_t max_iter = 60;
                p_star = boost::math::tools::newton_raphson_iterate(
                    grad_hess, guess, lo, hi, SOLVE_BITS, max_iter);
                if (max_iter >= 60) {
                    // Did not converge (should not happen for concave Q): fall
                    // back to Brent on the exact objective for this (k,m).
                    static thread_local std::vector<double> lgt, lgt_arg, qa, qr;
                    lgt.resize(N); lgt_arg.resize(N);
                    qa.resize(4 * (size_t)N); qr.resize(4 * (size_t)N);
                    double lgamma_tau = std::lgamma(tau);
                    double C = 0.0;
                    for (int n = 0; n < N; ++n) lgt_arg[n] = alpha_m[n] + beta_m[n] + tau;
                    lgamma_batch(lgt_arg.data(), lgt.data(), N);
                    for (int n = 0; n < N; ++n)
                        C -= (pkn2[n * 2 + 0] + pkn2[n * 2 + 1]) * lgt[n];
                    C += Wk * lgamma_tau;
                    auto neg_Q = [&](double p) -> double {
                        double a = tau * p, b = tau * (1.0 - p);
                        double norm_term = Wk * (std::lgamma(a) + std::lgamma(b));
                        for (int n = 0; n < N; ++n) {
                            double al = alpha_m[n], be = beta_m[n];
                            long base = 4 * (long)n;
                            qa[base + 0] = al + a; qa[base + 1] = be + b;
                            qa[base + 2] = be + a; qa[base + 3] = al + b;
                        }
                        lgamma_batch(qa.data(), qr.data(), 4 * N);
                        double Q = C - norm_term;
                        for (int n = 0; n < N; ++n) {
                            double w0 = pkn2[n * 2 + 0], w1 = pkn2[n * 2 + 1];
                            long base = 4 * (long)n;
                            Q += w0 * (qr[base + 0] + qr[base + 1])
                               + w1 * (qr[base + 2] + qr[base + 3]);
                        }
                        return -Q;
                    };
                    p_star = boost::math::tools::brent_find_minima(
                        neg_Q, lo, hi, SOLVE_BITS).first;
                }
            }
            p_km[(long)k * M + m] = p_star;
        }
    }
}


// ---------- RDR M-step ----------

void update_rdr_params_cpp(
    const double* X_rdrs,
    const double* posts_nk,
    double*       rdr_means,
    double*       rdr_vars,
    int N, int K, int M, double min_covar,
    double ig_alpha,
    const double* ig_beta)
{
    // n-outer loop: posts_nk[n*K + k] reads are sequential across k,
    // avoiding the strided-by-K cache misses of the old k-outer layout.
    // Thread-local (K*M) accumulators are reduced at the end.
    std::vector<double> global_sum1(K * M, 0.0);
    std::vector<double> global_sum2(K * M, 0.0);
    std::vector<double> global_Nk(K, 0.0);

#ifdef _OPENMP
#pragma omp parallel
    {
        std::vector<double> loc1(K * M, 0.0);
        std::vector<double> loc2(K * M, 0.0);
        std::vector<double> locNk(K, 0.0);

#pragma omp for nowait schedule(static)
        for (int n = 0; n < N; ++n) {
            for (int k = 0; k < K; ++k) {
                double p = posts_nk[(long)n * K + k];
                locNk[k] += p;
                for (int m = 0; m < M; ++m) {
                    double r = X_rdrs[(long)n * M + m];
                    loc1[(long)k * M + m] += p * r;
                    loc2[(long)k * M + m] += p * r * r;
                }
            }
        }

#pragma omp critical
        {
            for (int km = 0; km < K * M; ++km) {
                global_sum1[km] += loc1[km];
                global_sum2[km] += loc2[km];
            }
            for (int k = 0; k < K; ++k)
                global_Nk[k] += locNk[k];
        }
    }
#else
    for (int n = 0; n < N; ++n) {
        for (int k = 0; k < K; ++k) {
            double p = posts_nk[(long)n * K + k];
            global_Nk[k] += p;
            for (int m = 0; m < M; ++m) {
                double r = X_rdrs[(long)n * M + m];
                global_sum1[(long)k * M + m] += p * r;
                global_sum2[(long)k * M + m] += p * r * r;
            }
        }
    }
#endif

    bool use_ig = (ig_alpha > 0 && ig_beta != nullptr);
    for (int k = 0; k < K; ++k) {
        double nk = std::max(global_Nk[k], 1e-10);
        for (int m = 0; m < M; ++m) {
            double mean = global_sum1[(long)k * M + m] / nk;
            double weighted_var = global_sum2[(long)k * M + m] / nk - mean * mean;
            rdr_means[(long)k * M + m] = mean;
            if (use_ig) {
                double raw_SS = weighted_var * nk;
                double beta_m = ig_beta[m];
                rdr_vars[(long)k * M + m] = std::max(
                    (raw_SS + 2.0 * beta_m) / (nk + 2.0 * (ig_alpha + 1.0)),
                    min_covar);
            } else {
                rdr_vars[(long)k * M + m] = std::max(weighted_var, min_covar);
            }
        }
    }
}


// ---------- BAF tau M-step ----------

void update_baf_tau_cpp(
    const double* alphas_nm,
    const double* betas_nm,
    const double* posts_nk2,
    const double* baf_means,
    double*       baf_taus,
    int N, int K, int M,
    double min_tau, double max_tau, bool share_tau)
{
    const double lo = std::log(min_tau), hi = std::log(max_tau);

    if (share_tau) {
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
        for (int m = 0; m < M; ++m) {
            const double* alpha_m_base = alphas_nm + m;   // stride M
            const double* beta_m_base  = betas_nm  + m;   // stride M
            const double* p_m_base     = baf_means + m;   // stride M

            auto neg_Q_logtau = [&](double log_tau) -> double {
                double tau = std::exp(log_tau);
                double total = 0.0;
                for (int k = 0; k < K; ++k) {
                    double p = p_m_base[(long)k * M];
                    double a = tau * p;
                    double b = tau * (1.0 - p);
                    double norm = std::lgamma(a) + std::lgamma(b)
                                - std::lgamma(a + b);
                    for (int n = 0; n < N; ++n) {
                        double w0 = posts_nk2[(long)n * K * 2 + (long)k * 2 + 0];
                        double w1 = posts_nk2[(long)n * K * 2 + (long)k * 2 + 1];
                        if (w0 + w1 < 1e-12) continue;
                        double alpha_n = alpha_m_base[(long)n * M];
                        double beta_n  = beta_m_base[(long)n * M];
                        double lg_tot = std::lgamma(alpha_n + beta_n + tau);
                        double ll0 = std::lgamma(alpha_n + a)
                                   + std::lgamma(beta_n  + b) - lg_tot - norm;
                        double ll1 = std::lgamma(beta_n  + a)
                                   + std::lgamma(alpha_n + b) - lg_tot - norm;
                        total += w0 * ll0 + w1 * ll1;
                    }
                }
                return -total;
            };

            auto result = boost::math::tools::brent_find_minima(
                neg_Q_logtau, lo, hi, SOLVE_BITS);
            double tau_m = std::exp(result.first);
            for (int k = 0; k < K; ++k) baf_taus[(long)k * M + m] = tau_m;
        }
    } else {
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(dynamic)
#endif
        for (int k = 0; k < K; ++k) {
            for (int m = 0; m < M; ++m) {
                const double* alpha_m_base = alphas_nm + m;   // stride M
                const double* beta_m_base  = betas_nm  + m;   // stride M
                double p = baf_means[(long)k * M + m];

                auto neg_Q_logtau = [&](double log_tau) -> double {
                    double tau = std::exp(log_tau);
                    double a = tau * p;
                    double b = tau * (1.0 - p);
                    double norm = std::lgamma(a) + std::lgamma(b)
                                - std::lgamma(a + b);
                    double total = 0.0;
                    for (int n = 0; n < N; ++n) {
                        double w0 = posts_nk2[(long)n * K * 2 + (long)k * 2 + 0];
                        double w1 = posts_nk2[(long)n * K * 2 + (long)k * 2 + 1];
                        if (w0 + w1 < 1e-12) continue;
                        double alpha_n = alpha_m_base[(long)n * M];
                        double beta_n  = beta_m_base[(long)n * M];
                        double lg_tot = std::lgamma(alpha_n + beta_n + tau);
                        double ll0 = std::lgamma(alpha_n + a)
                                   + std::lgamma(beta_n  + b) - lg_tot - norm;
                        double ll1 = std::lgamma(beta_n  + a)
                                   + std::lgamma(alpha_n + b) - lg_tot - norm;
                        total += w0 * ll0 + w1 * ll1;
                    }
                    return -total;
                };

                auto result = boost::math::tools::brent_find_minima(
                    neg_Q_logtau, lo, hi, SOLVE_BITS);
                baf_taus[(long)k * M + m] = std::exp(result.first);
            }
        }
    }
}

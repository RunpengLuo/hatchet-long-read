"""Numpy log-likelihood kernels for the 2-mixture BAF+RDR HMM.

Computes per-bin per-cluster emission log-likelihoods under both haplotype
orientations (h=0 and h=1).  The BAF and RDR emissions are pluggable: BAF
supports "betabinom"; RDR supports "gaussian" and "negbinom".  The private
``_loglik_*`` cores return the per-sample (unsummed) (N, K, M) tensors and are
shared by the two public entry points, which differ only in whether they
reduce over samples: ``compute_loglik`` sums to (N, K) for the EM E-step,
``compute_loglik_unroll`` keeps the sample axis (N, K, M) for cna++ seeding.
All inputs are C-contiguous float64 numpy arrays.
"""

import numpy as np
from scipy.special import betaln, gammaln, xlogy


def _loglik_baf_betabinom(X_alphas, X_betas, log_binom_const, baf_means, baf_taus):
    """Beta-Binomial BAF emission log-likelihoods for both phase orientations.

    Per-sample (unsummed) over the M axis.

    Args:
        X_alphas:        (N, M) A-allele counts.
        X_betas:         (N, M) B-allele counts.
        log_binom_const: (N, M) gammaln(T+1)-gammaln(B+1)-gammaln(A+1).
        baf_means:       (K, M) per-cluster per-sample BAF means.
        baf_taus:        (K, M) or (M,) Beta-Binomial dispersion parameters.

    Returns:
        ll_bafs_h0: (N, K, M) BAF log-likelihoods under orientation h=0.
        ll_bafs_h1: (N, K, M) BAF log-likelihoods under orientation h=1.
    """
    bt = baf_taus[None, :, :] if baf_taus.ndim == 2 else baf_taus[None, None, :]
    bb_alpha = bt * baf_means[None, :, :]  # (1, K, M)
    bb_beta = bt * (1 - baf_means[None, :, :])
    bb_delta = betaln(bb_alpha, bb_beta)
    lnB_h0 = betaln(X_alphas[:, None, :] + bb_alpha, X_betas[:, None, :] + bb_beta)
    lnB_h1 = betaln(X_betas[:, None, :] + bb_alpha, X_alphas[:, None, :] + bb_beta)
    ll_bafs_h0 = log_binom_const[:, None, :] + lnB_h0 - bb_delta  # (N, K, M)
    ll_bafs_h1 = log_binom_const[:, None, :] + lnB_h1 - bb_delta
    return ll_bafs_h0, ll_bafs_h1


def _loglik_rdr_gaussian(X_rdrs, rdr_means, rdr_vars):
    """Gaussian RDR emission log-likelihood, per-sample (unsummed).

    Phase-independent (returned once and shared across both orientations).

    Args:
        X_rdrs:    (N, M) observed (log-)RDR values.
        rdr_means: (K, M) per-cluster per-sample Gaussian means.
        rdr_vars:  (K, M) per-cluster per-sample Gaussian variances.

    Returns:
        ll_rdrs: (N, K, M) RDR log-likelihoods.
    """
    diff2 = (X_rdrs[:, None, :] - rdr_means[None, :, :]) ** 2  # (N, K, M)
    return (
        -0.5 * diff2 / rdr_vars[None, :, :]
        - 0.5 * np.log(2 * np.pi * rdr_vars)[None, :, :]
    )


def _loglik_rdr_negbinom(X_counts, X_nb_offsets, rho, phi):
    """Negative-binomial count emission log-likelihood, per-sample (unsummed).

    Phase-independent (returned once and shared across both orientations).
    NB2 mean/dispersion parameterization with
        - mean mu[i, s, k] = X_nb_offsets[i, s] * rho[k, s],
        - variance mu + phi * mu**2,
        - size r = 1/phi,
        - and success probability r/(r+mu).

    Args:
        X_counts:      (N, M) per-bin per-sample counts.
        X_nb_offsets:  (N, M) per-bin per-sample NB offset (lambda_i * T_s).
        rho:           (K, M) per-cluster per-sample relative-copy state.
        phi:           (K, M) per-sample NB dispersion (rows tied across k).

    Returns:
        ll_rdrs: (N, K, M) phase-independent log-likelihoods.
    """
    mu = X_nb_offsets[:, None, :] * rho[None, :, :]  # (N, K, M)
    r = 1.0 / phi[None, :, :]  # (1, K, M)
    y = X_counts[:, None, :]  # (N, 1, M)
    frac = mu / (r + mu)  # (N, K, M) in [0, 1)
    return (
        gammaln(y + r)
        - gammaln(r)
        - gammaln(X_counts + 1.0)[:, None, :]
        + r * np.log1p(-frac)
        + xlogy(y, frac)
    )  # (N, K, M)


def _emission_loglik_terms(
    X_rdrs,
    X_alphas,
    X_betas,
    log_binom_const,
    rdr_means,
    rdr_vars,
    baf_means,
    baf_taus,
    rdr_emission,
    baf_emission,
    X_counts,
    X_nb_offsets,
):
    """Dispatch the RDR and BAF emissions to per-sample (unsummed) terms.

    Args:
        X_rdrs:          (N, M) observed RDR values (gaussian RDR emission).
        X_alphas:        (N, M) A-allele counts.
        X_betas:         (N, M) B-allele counts.
        log_binom_const: (N, M) BAF binomial-coefficient log-constant.
        rdr_means:       (K, M) RDR mean parameters (rho for negbinom).
        rdr_vars:        (K, M) RDR variance/dispersion (phi for negbinom).
        baf_means:       (K, M) per-cluster per-sample BAF means.
        baf_taus:        (K, M) or (M,) BAF dispersion parameters.
        rdr_emission:    "gaussian" or "negbinom".
        baf_emission:    "betabinom".
        X_counts:        (N, M) per-bin per-sample counts (negbinom emission).
        X_nb_offsets:    (N, M) per-bin per-sample NB offset (negbinom emission).

    Returns:
        ll_rdr:    (N, K, M) phase-independent RDR term.
        ll_baf_h0: (N, K, M) BAF term under orientation h=0.
        ll_baf_h1: (N, K, M) BAF term under orientation h=1.

    Raises:
        ValueError: on an unknown emission option string, or when a required
            negbinom input is missing.
    """
    if baf_emission == "betabinom":
        ll_baf_h0, ll_baf_h1 = _loglik_baf_betabinom(
            X_alphas, X_betas, log_binom_const, baf_means, baf_taus
        )
    else:
        raise ValueError(f"unknown baf_emission: {baf_emission!r}")

    if rdr_emission == "gaussian":
        ll_rdr = _loglik_rdr_gaussian(X_rdrs, rdr_means, rdr_vars)
    elif rdr_emission == "negbinom":
        if X_counts is None or X_nb_offsets is None:
            raise ValueError("negbinom emission requires X_counts and X_nb_offsets")
        ll_rdr = _loglik_rdr_negbinom(X_counts, X_nb_offsets, rdr_means, rdr_vars)
    else:
        raise ValueError(f"unknown rdr_emission: {rdr_emission!r}")

    return ll_rdr, ll_baf_h0, ll_baf_h1


def compute_loglik(
    X_rdrs,
    X_alphas,
    X_betas,
    X_totals,
    rdr_means,
    rdr_vars,
    baf_means,
    baf_taus,
    rdr_emission="gaussian",
    baf_emission="betabinom",
    X_counts=None,
    X_nb_offsets=None,
):
    """Per-bin per-cluster emission log-likelihoods summed over samples.

    EM E-step kernel. Dispatches BAF and RDR emissions on the option strings,
    adds the (phase-independent) RDR term to each BAF orientation, and reduces
    over samples to (N, K).

    Args:
        X_rdrs:       (N, M) observed RDR values (gaussian RDR emission).
        X_alphas:     (N, M) A-allele counts.
        X_betas:      (N, M) B-allele counts.
        X_totals:     (N, M) total allele counts.
        rdr_means:    (K, M) per-cluster per-sample RDR mean parameters.
        rdr_vars:     (K, M) per-cluster per-sample RDR variance/dispersion.
        baf_means:    (K, M) per-cluster per-sample BAF means.
        baf_taus:     (K, M) or (M,) BAF dispersion parameters.
        rdr_emission: "gaussian" or "negbinom".
        baf_emission: "betabinom".
        X_counts:     (N, M) per-bin per-sample counts (negbinom emission).
        X_nb_offsets: (N, M) per-bin per-sample NB offset lambda_i*T_s
                      (negbinom emission).

    Returns:
        lls0: (N, K) log-likelihoods under haplotype orientation h=0.
        lls1: (N, K) log-likelihoods under haplotype orientation h=1.

    Raises:
        ValueError: on an unknown emission option string, or when a required
            negbinom input is missing.
    """
    log_binom_const = (
        gammaln(X_totals + 1) - gammaln(X_betas + 1) - gammaln(X_alphas + 1)
    )  # (N, M)
    ll_rdr, ll_baf_h0, ll_baf_h1 = _emission_loglik_terms(
        X_rdrs,
        X_alphas,
        X_betas,
        log_binom_const,
        rdr_means,
        rdr_vars,
        baf_means,
        baf_taus,
        rdr_emission,
        baf_emission,
        X_counts,
        X_nb_offsets,
    )
    lls0 = np.sum(ll_rdr + ll_baf_h0, axis=2)  # (N, K)
    lls1 = np.sum(ll_rdr + ll_baf_h1, axis=2)  # (N, K)
    return lls0, lls1


def compute_loglik_unroll(
    X_rdrs,
    X_alphas,
    X_betas,
    rdr_means,
    rdr_vars,
    baf_means,
    baf_taus,
    log_binom_const,
    rdr_emission="gaussian",
    baf_emission="betabinom",
    X_counts=None,
    X_nb_offsets=None,
):
    """Per-bin per-cluster per-sample log-likelihoods, unsummed over samples.

    Same emission model and dispatch as compute_loglik but keeps the sample
    axis, returning (N, K, M) for a batch of K candidate clusters. Takes a
    precomputed log_binom_const to avoid redundant gammaln evaluations across
    the many candidate-batch calls in cna++ seeding.

    Args:
        X_rdrs:          (N, M) observed RDR values (gaussian RDR emission).
        X_alphas:        (N, M) A-allele counts.
        X_betas:         (N, M) B-allele counts.
        rdr_means:       (K, M) RDR means per candidate (rho for negbinom).
        rdr_vars:        (K, M) RDR variances per candidate (phi for negbinom).
        baf_means:       (K, M) BAF means per candidate.
        baf_taus:        (K, M) or (M,) BAF dispersion parameters.
        log_binom_const: (N, M) precomputed gammaln(T+1)-gammaln(B+1)-gammaln(A+1).
        rdr_emission:    "gaussian" or "negbinom".
        baf_emission:    "betabinom".
        X_counts:        (N, M) per-bin per-sample counts (negbinom emission).
        X_nb_offsets:    (N, M) per-bin per-sample NB offset lambda_i*T_s
                         (negbinom emission).

    Returns:
        lls0: (N, K, M) log-likelihoods under haplotype orientation h=0.
        lls1: (N, K, M) log-likelihoods under haplotype orientation h=1.

    Raises:
        ValueError: on an unknown emission option string, or when a required
            negbinom input is missing.
    """
    ll_rdr, ll_baf_h0, ll_baf_h1 = _emission_loglik_terms(
        X_rdrs,
        X_alphas,
        X_betas,
        log_binom_const,
        rdr_means,
        rdr_vars,
        baf_means,
        baf_taus,
        rdr_emission,
        baf_emission,
        X_counts,
        X_nb_offsets,
    )
    return ll_rdr + ll_baf_h0, ll_rdr + ll_baf_h1  # (N, K, M)

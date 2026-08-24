"""Numpy log-likelihood kernels for the 2-mixture BAF+RDR HMM.

Computes per-bin per-cluster emission log-likelihoods under both haplotype
orientations (h=0 and h=1).  The BAF and RDR emissions are pluggable: BAF
supports "betabinom"; RDR supports "gaussian" (and "negbinom", TODO).
``compute_loglik`` dispatches on the two option strings.  All inputs are
C-contiguous float64 numpy arrays.
"""

import numpy as np
from scipy.special import betaln, gammaln


def _loglik_baf_betabinom(X_alphas, X_betas, X_totals, baf_means, baf_taus):
    """Beta-Binomial BAF emission log-likelihoods for both phase orientations.

    Args:
        X_alphas:  (N, M) A-allele counts.
        X_betas:   (N, M) B-allele counts.
        X_totals:  (N, M) total allele counts.
        baf_means: (K, M) per-cluster per-sample BAF means.
        baf_taus:  (K, M) or (M,) Beta-Binomial dispersion parameters.

    Returns:
        ll_bafs_h0: (N, K) BAF log-likelihoods under haplotype orientation h=0.
        ll_bafs_h1: (N, K) BAF log-likelihoods under haplotype orientation h=1.
    """
    log_binom_const = (
        gammaln(X_totals + 1) - gammaln(X_betas + 1) - gammaln(X_alphas + 1)
    )  # (N, M)
    bt = baf_taus[None, :, :] if baf_taus.ndim == 2 else baf_taus[None, None, :]
    bb_alpha = bt * baf_means[None, :, :]  # (1, K, M)
    bb_beta = bt * (1 - baf_means[None, :, :])
    bb_delta = betaln(bb_alpha, bb_beta)
    lnB_h0 = betaln(X_alphas[:, None, :] + bb_alpha, X_betas[:, None, :] + bb_beta)
    lnB_h1 = betaln(X_betas[:, None, :] + bb_alpha, X_alphas[:, None, :] + bb_beta)
    ll_bafs_h0 = np.sum(
        log_binom_const[:, None, :] + lnB_h0 - bb_delta, axis=2
    )  # (N, K)
    ll_bafs_h1 = np.sum(log_binom_const[:, None, :] + lnB_h1 - bb_delta, axis=2)
    return ll_bafs_h0, ll_bafs_h1


def _loglik_rdr_gaussian(X_rdrs, rdr_means, rdr_vars):
    """Gaussian RDR emission log-likelihood summed over samples.

    Phase-independent (returned once and shared across both orientations).

    Args:
        X_rdrs:    (N, M) observed (log-)RDR values.
        rdr_means: (K, M) per-cluster per-sample Gaussian means.
        rdr_vars:  (K, M) per-cluster per-sample Gaussian variances.

    Returns:
        ll_rdrs: (N, K) RDR log-likelihoods.
    """
    log_norm_const = 0.5 * np.sum(np.log(2 * np.pi * rdr_vars), axis=1)  # (K,)
    quad = 0.5 * np.einsum(
        "nkm,km->nk",
        (X_rdrs[:, None, :] - rdr_means[None, :, :]) ** 2,
        1.0 / rdr_vars,
    )  # (N, K)
    return -quad - log_norm_const


def _loglik_rdr_negbinom(X_counts, X_props, X_libsizes, rho, phi):
    """TODO: negative-binomial count emission log-likelihood.

    For pseudobulk single-cell data with per-bin counts (docs/TODO.md
    section 1):
        X_counts[i, s] ~ NB(mu = X_props[i] * X_libsizes[s] * rho[k, s], phi[s]),
    where rho (state relative copy, carried in the rdr_means slot) and phi
    (per-sample dispersion, carried in the rdr_vars slot) are the emission
    parameters.

    Args:
        X_counts:   (N, M) per-bin per-sample integer-like counts.
        X_props:    (N,)   per-bin baseline proportion.
        X_libsizes: (M,)   per-sample library size.
        rho:        (K, M) per-cluster per-sample relative-copy state.
        phi:        (K, M) per-sample NB dispersion.

    Returns:
        ll_rdrs: (N, K) phase-independent log-likelihoods.
    """
    raise NotImplementedError("negbinom count emission not implemented")


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
    X_props=None,
    X_libsizes=None,
):
    """Per-bin per-cluster emission log-likelihoods summed over samples.

    Dispatches BAF and RDR emissions on the option strings, then adds the
    (phase-independent) RDR term to each BAF orientation.

    Args:
        X_rdrs:       (N, M) observed RDR values (gaussian RDR emission).
        X_alphas:     (N, M) A-allele counts.
        X_betas:      (N, M) B-allele counts.
        X_totals:     (N, M) total allele counts.
        rdr_means:    (K, M) per-cluster per-sample RDR mean parameters.
        rdr_vars:     (K, M) per-cluster per-sample RDR variance/dispersion.
        baf_means:    (K, M) per-cluster per-sample BAF means.
        baf_taus:     (K, M) or (M,) BAF dispersion parameters.
        rdr_emission: "gaussian" or "negbinom" (TODO).
        baf_emission: "betabinom".
        X_counts:     (N, M) per-bin per-sample counts (negbinom emission).
        X_props:      (N,)   per-bin baseline proportion (negbinom emission).
        X_libsizes:   (M,)   per-sample library size (negbinom emission).

    Returns:
        lls0: (N, K) log-likelihoods under haplotype orientation h=0.
        lls1: (N, K) log-likelihoods under haplotype orientation h=1.

    Raises:
        ValueError: on an unknown emission option string, or when a required
            negbinom input is missing.
    """
    if baf_emission == "betabinom":
        ll_bafs_h0, ll_bafs_h1 = _loglik_baf_betabinom(
            X_alphas, X_betas, X_totals, baf_means, baf_taus
        )
    else:
        raise ValueError(f"unknown baf_emission: {baf_emission!r}")

    if rdr_emission == "gaussian":
        ll_rdrs = _loglik_rdr_gaussian(X_rdrs, rdr_means, rdr_vars)
    elif rdr_emission == "negbinom":
        if X_counts is None or X_props is None or X_libsizes is None:
            raise ValueError(
                "negbinom emission requires X_counts, X_props, and X_libsizes"
            )
        ll_rdrs = _loglik_rdr_negbinom(
            X_counts, X_props, X_libsizes, rdr_means, rdr_vars
        )
    else:
        raise ValueError(f"unknown rdr_emission: {rdr_emission!r}")

    return ll_rdrs + ll_bafs_h0, ll_rdrs + ll_bafs_h1


def compute_loglik_single_cluster_batch(
    X_rdrs,
    X_alphas,
    X_betas,
    rdr_means_batch,
    rdr_vars_batch,
    baf_means_batch,
    baf_taus,
    log_binom_const,
):
    """Per-bin per-sample log-likelihoods for a batch of candidate clusters.

    Same emission model as compute_loglik but for C candidate clusters at
    once, reusing a precomputed log_binom_const to avoid redundant gammaln
    evaluations across calls.

    Args:
        X_rdrs:           (N, M) observed RDR values.
        X_alphas:         (N, M) A-allele counts.
        X_betas:          (N, M) B-allele counts.
        rdr_means_batch:  (C, M) RDR means for each candidate cluster.
        rdr_vars_batch:   (C, M) RDR variances for each candidate cluster.
        baf_means_batch:  (C, M) BAF means for each candidate cluster.
        baf_taus:         (M,)   per-sample Beta-Binomial dispersion.
        log_binom_const:  (N, M) precomputed gammaln(T+1)-gammaln(B+1)-gammaln(A+1).

    Returns:
        lls0: (N, C, M) log-likelihoods under haplotype orientation h=0.
        lls1: (N, C, M) log-likelihoods under haplotype orientation h=1.
    """
    bb_alpha = baf_taus[None, None, :] * baf_means_batch[None, :, :]  # (1, C, M)
    bb_beta = baf_taus[None, None, :] * (1 - baf_means_batch[None, :, :])  # (1, C, M)
    bb_delta = betaln(bb_alpha, bb_beta)  # (1, C, M)
    lnB_h0 = betaln(X_alphas[:, None, :] + bb_alpha, X_betas[:, None, :] + bb_beta)
    lnB_h1 = betaln(X_betas[:, None, :] + bb_alpha, X_alphas[:, None, :] + bb_beta)
    ll_baf_h0 = log_binom_const[:, None, :] + lnB_h0 - bb_delta  # (N, C, M)
    ll_baf_h1 = log_binom_const[:, None, :] + lnB_h1 - bb_delta  # (N, C, M)

    diff2 = (X_rdrs[:, None, :] - rdr_means_batch[None, :, :]) ** 2  # (N, C, M)
    ll_rdr = (
        -0.5 * diff2 / rdr_vars_batch[None, :, :]
        - 0.5 * np.log(2 * np.pi * rdr_vars_batch)[None, :, :]
    )  # (N, C, M)

    return ll_rdr + ll_baf_h0, ll_rdr + ll_baf_h1

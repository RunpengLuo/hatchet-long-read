"""EM M-step routines for the 2-mixture BAF+RDR HMM.

RDR and BAF emission updates are pluggable and dispatched by ``do_mstep`` on
the option strings:
- RDR "gaussian": closed-form posterior-weighted mean & variance.
- RDR "negbinom": TODO.
- BAF "betabinom": scipy Brent means + optional posterior-weighted tau MLE.
- Start probabilities (emission-independent): posterior counts at segment starts.
"""

import logging

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import betaln


def do_mstep(
    X_rdrs,  # (N, M)
    X_alphas,  # (N, M)
    X_betas,  # (N, M)
    posts,  # (N, K, 2) — posteriors from E-step
    baf_taus,  # (K, M)
    baf_means_init,  # (K, M) — warm start
    X_lengths,  # (S,) segment lengths
    update_tau=False,
    share_tau=True,
    min_covar=1e-3,
    tol=1e-6,
    min_tau=50,
    max_tau=100,
    baf_eps=1e-6,
    ig_alpha=10.0,
    ig_beta=0.01,
    baf_k_start=0,
    rdr_emission="gaussian",
    baf_emission="betabinom",
    X_counts=None,
    X_props=None,
    X_libsizes=None,
):
    """EM M-step: emission parameters + start probabilities.

    Dispatches the RDR and BAF emission updates on the option strings and
    computes start probabilities from posterior counts at segment starts.
    BAF means are NOT folded here; the mhBAF fold is applied after decoding
    in cluster_bins.py to preserve EM monotonicity.

    Args:
        rdr_emission: "gaussian" or "negbinom" (TODO).
        baf_emission: "betabinom".
        X_counts:     (N, M) per-bin per-sample counts (negbinom emission).
        X_props:      (N,)   per-bin baseline proportion (negbinom emission).
        X_libsizes:   (M,)   per-sample library size (negbinom emission).

    Returns:
        rdr_means:      (K, M) numpy array.
        rdr_vars:       (K, M) numpy array.
        baf_means:      (K, M) numpy array.
        baf_taus:       (K, M) numpy array.
        log_startprobs: (K, 2) numpy array.

    Raises:
        ValueError: on an unknown emission option string.
    """
    N, K, _ = posts.shape

    # ---- start probabilities ----
    seg_starts = np.concatenate([[0], np.cumsum(X_lengths[:-1])])
    gamma0 = np.sum(np.maximum(posts[seg_starts], tol), axis=0)  # (K, 2)
    log_startprobs = np.log(gamma0 / np.sum(gamma0))  # (K, 2)

    # ---- RDR emission parameters ----
    posts_marg = np.sum(posts, axis=-1)  # (N, K)
    Nk = np.sum(posts_marg, axis=0)  # (K,)
    if np.any(Nk < tol):
        logging.warning(f"Some clusters have effective Nk < {tol}")

    if rdr_emission == "gaussian":
        rdr_means, rdr_vars = _update_rdr_gaussian(
            X_rdrs, posts_marg, Nk, min_covar, ig_alpha, ig_beta, tol
        )
    elif rdr_emission == "negbinom":
        if X_counts is None or X_props is None or X_libsizes is None:
            raise ValueError(
                "negbinom emission requires X_counts, X_props, and X_libsizes"
            )
        rdr_means, rdr_vars = _update_rdr_negbinom(
            X_counts, X_props, X_libsizes, posts_marg, Nk, min_covar, tol
        )
    else:
        raise ValueError(f"unknown rdr_emission: {rdr_emission!r}")

    # ---- BAF emission parameters ----
    if baf_emission == "betabinom":
        # tau first (optional), before means so p is optimal for the new tau
        if update_tau:
            baf_taus = _update_baf_tau(
                X_alphas,
                X_betas,
                posts,
                baf_means_init,
                baf_taus,
                min_tau=min_tau,
                max_tau=max_tau,
                share_tau=share_tau,
            )
        posts_kn2 = np.ascontiguousarray(posts.transpose(1, 0, 2))  # (K, N, 2)
        baf_means = _update_baf_means(
            baf_means_init,
            X_alphas.T,
            X_betas.T,
            baf_taus,
            posts_kn2,
            baf_eps,
            k_start=baf_k_start,
        )
    else:
        raise ValueError(f"unknown baf_emission: {baf_emission!r}")

    return rdr_means, rdr_vars, baf_means, baf_taus, log_startprobs


def _update_rdr_gaussian(X_rdrs, posts_marg, Nk, min_covar, ig_alpha, ig_beta, tol):
    """Closed-form posterior-weighted Gaussian RDR mean/variance update.

    The variance uses an inverse-gamma MAP shrinkage when ig_alpha > 0, else
    the plain weighted variance; both are floored at min_covar.

    Args:
        X_rdrs:     (N, M) (log-)RDR observations.
        posts_marg: (N, K) cluster-marginal posteriors.
        Nk:         (K,)   effective per-cluster counts.
        min_covar:  variance floor.
        ig_alpha:   inverse-gamma shape (<=0 disables the prior).
        ig_beta:    inverse-gamma scale.
        tol:        lower clamp for Nk in denominators.

    Returns:
        rdr_means: (K, M) numpy array.
        rdr_vars:  (K, M) numpy array.
    """
    safe_Nk = np.maximum(Nk, tol)[:, None]  # (K, 1)
    rdr_means = np.einsum("nk,nm->km", posts_marg, X_rdrs) / safe_Nk
    weighted_var = (
        np.einsum("nk,nm->km", posts_marg, X_rdrs**2) / safe_Nk - rdr_means**2
    )
    if ig_alpha > 0:
        raw_SS = weighted_var * safe_Nk  # (K, M)
        rdr_vars = np.maximum(
            (raw_SS + 2 * ig_beta) / (Nk[:, None] + 2 * (ig_alpha + 1)),
            min_covar,
        )
    else:
        rdr_vars = np.maximum(weighted_var, min_covar)
    return rdr_means, rdr_vars


def _update_rdr_negbinom(X_counts, X_props, X_libsizes, posts_marg, Nk, min_covar, tol):
    """TODO: negative-binomial count M-step.

    Update the state relative-copy rho (returned in the rdr_means slot) and
    per-sample dispersion phi (returned in the rdr_vars slot) for the model
    X_counts[i, s] ~ NB(X_props[i] * X_libsizes[s] * rho[k, s], phi[s]).
    rho has a posterior-weighted closed form given the NB offset
    X_props[i] * X_libsizes[s]; phi has none and needs a per-sample solve.

    Args:
        X_counts:   (N, M) per-bin per-sample counts.
        X_props:    (N,)   per-bin baseline proportion.
        X_libsizes: (M,)   per-sample library size.
        posts_marg: (N, K) cluster-marginal posteriors.
        Nk:         (K,)   effective per-cluster counts.
        min_covar:  dispersion floor.
        tol:        lower clamp for Nk in denominators.

    Returns:
        rho: (K, M) numpy array.
        phi: (K, M) numpy array.
    """
    raise NotImplementedError("negbinom count M-step not implemented")


def _update_baf_tau(
    X_alphas,
    X_betas,
    posts,
    baf_means,
    baf_taus,
    min_tau=50,
    max_tau=500,
    share_tau=True,
):
    """MLE for BAF tau via Brent in log-tau space, maximising Q_BAF.

    With share_tau=True a single tau per sample (pooling all clusters) is fit
    and broadcast to all K rows; with share_tau=False tau is fit independently
    per (cluster, sample).

    Args:
        X_alphas:  (N, M) A-allele counts.
        X_betas:   (N, M) B-allele counts.
        posts:     (N, K, 2) full posteriors.
        baf_means: (K, M) current BAF means.
        baf_taus:  (K, M) current tau values.
        min_tau, max_tau: search bounds.
        share_tau: tie tau across clusters within a sample.

    Returns:
        (K, M) updated tau values.
    """
    N, K, _ = posts.shape
    M = X_alphas.shape[1]
    taus_new = baf_taus.copy()
    lo, hi = np.log(min_tau), np.log(max_tau)

    def neg_Q_km(log_tau, alpha, beta, w0, w1, p):
        tau = np.exp(log_tau)
        a, b = tau * p, tau * (1 - p)
        norm = betaln(a, b)
        ll0 = betaln(alpha + a, beta + b) - norm
        ll1 = betaln(beta + a, alpha + b) - norm
        return -(w0 @ ll0 + w1 @ ll1)

    for m in range(M):
        alpha_m = X_alphas[:, m]
        beta_m = X_betas[:, m]

        if share_tau:

            def neg_Q(
                log_tau, _a=alpha_m, _b=beta_m, _posts=posts, _baf=baf_means[:, m]
            ):
                tau = np.exp(log_tau)
                total = 0.0
                for k in range(K):
                    p = _baf[k]
                    a, b = tau * p, tau * (1 - p)
                    norm = betaln(a, b)
                    ll0 = betaln(_a + a, _b + b) - norm
                    ll1 = betaln(_b + a, _a + b) - norm
                    total += _posts[:, k, 0] @ ll0 + _posts[:, k, 1] @ ll1
                return -total

            res = minimize_scalar(neg_Q, bounds=(lo, hi), method="bounded")
            taus_new[:, m] = np.exp(res.x)
        else:
            for k in range(K):
                res = minimize_scalar(
                    neg_Q_km,
                    bounds=(lo, hi),
                    method="bounded",
                    args=(
                        alpha_m,
                        beta_m,
                        posts[:, k, 0],
                        posts[:, k, 1],
                        baf_means[k, m],
                    ),
                )
                taus_new[k, m] = np.exp(res.x)

    return taus_new


def _update_baf_means(
    p0_km, alphas_mn, betas_mn, baf_taus, posts_kn2, baf_eps=1e-6, k_start=0
):
    """MLE for BAF means via scipy bounded scalar optimization.

    For each (k, m) with k >= k_start, minimizes the posterior-weighted
    negative BB log-likelihood over p in (baf_eps, 1-baf_eps) using Brent.
    Clusters k < k_start retain their initial BAF means.

    Args:
        p0_km:     (K, M) — initial BAF means (preserved for k < k_start).
        alphas_mn: (M, N) — A-allele counts.
        betas_mn:  (M, N) — B-allele counts.
        baf_taus:  (K, M) — dispersion params.
        posts_kn2: (K, N, 2) — posteriors.
        baf_eps:   float  — Brent search bounds [baf_eps, 1-baf_eps].
        k_start:   int    — first cluster index to update (default 0 = all).

    Returns:
        (K, M) BAF means.
    """
    K, M = p0_km.shape
    p_km = p0_km.copy()
    posts0 = posts_kn2[:, :, 0]  # (K, N)
    posts1 = posts_kn2[:, :, 1]  # (K, N)
    EPS = baf_eps

    for m in range(M):
        alpha_m = alphas_mn[m]  # (N,)
        beta_m = betas_mn[m]  # (N,)
        for k in range(k_start, K):
            tau = baf_taus[k, m]
            w0 = posts0[k]  # (N,)
            w1 = posts1[k]  # (N,)

            def neg_Q(p, _tau=tau, _a=alpha_m, _b=beta_m, _w0=w0, _w1=w1):
                a = _tau * p
                b = _tau * (1.0 - p)
                # h=0: BB(alpha, beta | p, tau)
                ll0 = betaln(_a + a, _b + b) - betaln(a, b)
                # h=1: BB(beta, alpha | p, tau)
                ll1 = betaln(_b + a, _a + b) - betaln(a, b)
                return -(_w0 @ ll0 + _w1 @ ll1)

            res = minimize_scalar(neg_Q, bounds=(EPS, 1.0 - EPS), method="bounded")
            p_km[k, m] = res.x

    return p_km

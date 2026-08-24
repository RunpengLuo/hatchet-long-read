"""EM M-step routines for the 2-mixture BAF+RDR HMM.

RDR and BAF emission updates are pluggable and dispatched by ``do_mstep`` on
the option strings:
- RDR "gaussian": closed-form posterior-weighted mean & variance.
- RDR "negbinom": ECM (Meng & Rubin 1993) alternating rho|phi (Newton) and
  phi|rho (Brent) to their common fixed point.
- BAF "betabinom": scipy Brent means + optional posterior-weighted tau MLE.
- Start probabilities (emission-independent): posterior counts at segment starts.
"""

import logging

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import betaln, gammaln, xlogy


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
    share_phi=False,
    X_counts=None,
    X_nb_offsets=None,
):
    """EM M-step: emission parameters + start probabilities.

    Dispatches the RDR and BAF emission updates on the option strings and
    computes start probabilities from posterior counts at segment starts.
    BAF means are NOT folded here; the mhBAF fold is applied after decoding
    in cluster_bins.py to preserve EM monotonicity.

    Args:
        rdr_emission: "gaussian" or "negbinom".
        baf_emission: "betabinom".
        share_phi:    tie NB phi across clusters within a sample (negbinom).
        X_counts:     (N, M) per-bin per-sample counts (negbinom emission).
        X_nb_offsets: (N, M) per-bin per-sample NB offset lambda_i*T_s
                      (negbinom emission).

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
        if X_counts is None or X_nb_offsets is None:
            raise ValueError("negbinom emission requires X_counts and X_nb_offsets")
        rdr_means, rdr_vars = _update_rdr_negbinom(
            X_counts, X_nb_offsets, posts_marg, Nk, min_covar, tol, share_phi=share_phi
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


def _update_rdr_negbinom(
    X_counts,
    X_nb_offsets,
    posts_marg,
    Nk,
    min_covar,
    tol,
    share_phi=False,
    max_ecm=50,
    ecm_tol=1e-7,
    max_newton=50,
    newton_tol=1e-8,
    max_phi=1e3,
):
    """Negative-binomial count M-step via ECM (Meng & Rubin 1993).

    Alternates two conditional-maximization steps to their common fixed
    point, which equals the joint maximizer of the posterior-weighted NB
    log-likelihood for X_counts[i, s] ~ NB(mu, phi), Var = mu + phi * mu**2,
    mu = X_nb_offsets[i, s] * rho[k, s]:
    - rho | phi: per-(k, s) 1D concave solve by Newton, warm-started from the
      method-of-moments estimate sum_i w y / sum_i w offset (exact rho MLE in
      the Poisson phi->0 limit).
    - phi | rho: 1D solve (Brent in log-phi); per (cluster, sample) when
      share_phi is False, else one per sample pooled across clusters.
    Iterating to convergence makes ECM equal joint (rho, phi) maximization
    while preserving EM monotone ascent (doi:10.1093/biomet/80.2.267). rho is
    returned in the rdr_means slot, phi in the rdr_vars slot.

    Args:
        X_counts:     (N, M) per-bin per-sample counts.
        X_nb_offsets: (N, M) per-bin per-sample NB offset lambda_i*T_s.
        posts_marg:   (N, K) cluster-marginal posteriors.
        Nk:           (K,)   effective per-cluster counts.
        min_covar:    dispersion (phi) floor.
        tol:          lower clamp for rho and weight sums in denominators.
        share_phi:    tie phi across clusters within a sample.
        max_ecm:      max ECM sweeps.
        ecm_tol:      ECM stop tolerance on the weighted log-likelihood.
        max_newton:   max Newton iters per rho conditional-maximization.
        newton_tol:   Newton stop tolerance on max |step / rho|.
        max_phi:      upper bound on phi in the Brent search.

    Returns:
        rho: (K, M) numpy array.
        phi: (K, M) numpy array (rows tied across clusters when share_phi).
    """
    K = posts_marg.shape[1]
    M = X_counts.shape[1]
    Wy = posts_marg.T @ X_counts  # (K, M) = sum_i w_ik y_is
    Woff = posts_marg.T @ X_nb_offsets  # (K, M) = sum_i w_ik offset

    rho = np.maximum(Wy / np.maximum(Woff, tol), tol)  # (K, M) MoM init
    phi = np.full((K, M), max(min_covar, 1e-2))  # (K, M)

    prev_ll = -np.inf
    for _ in range(max_ecm):
        phi = _cm_phi_negbinom(
            X_counts, X_nb_offsets, posts_marg, rho, phi, min_covar, max_phi, share_phi
        )
        rho = _cm_rho_negbinom(
            X_counts,
            X_nb_offsets,
            posts_marg,
            Wy,
            rho,
            phi,
            tol,
            max_newton,
            newton_tol,
        )
        ll = _weighted_ll_negbinom(X_counts, X_nb_offsets, posts_marg, rho, phi)
        if abs(ll - prev_ll) <= ecm_tol * (abs(prev_ll) + ecm_tol):
            break
        prev_ll = ll
    return rho, phi


def _cm_rho_negbinom(
    X_counts, offset, posts_marg, Wy, rho, phi, tol, max_newton, newton_tol
):
    """CM-step for rho: per-(k, s) NB mean-MLE by Newton at fixed phi.

    The posterior-weighted NB log-likelihood is concave in rho[k, s] (log link,
    linear offset), so Newton from the MoM warm start converges quadratically.

    Args:
        X_counts:   (N, M) counts.
        offset:     (N, M) lambda_i * T_s.
        posts_marg: (N, K) cluster-marginal posteriors.
        Wy:         (K, M) sum_i w_ik y_is (phi-independent, precomputed).
        rho:        (K, M) warm start.
        phi:        (K, M) fixed dispersion.
        tol:        lower clamp on rho.
        max_newton: max iterations.
        newton_tol: stop tolerance on max |step / rho|.

    Returns:
        (K, M) updated rho.
    """
    r = 1.0 / phi  # (K, M) NB size
    y = X_counts[:, None, :]  # (N, 1, M)
    c = offset[:, None, :]  # (N, 1, M)
    rho = np.maximum(rho.copy(), tol)
    for _ in range(max_newton):
        denom = r[None, :, :] + c * rho[None, :, :]  # (N, K, M)
        yr = y + r[None, :, :]  # (N, K, M)
        g = Wy / rho - np.einsum("nk,nkm->km", posts_marg, c * yr / denom)
        gp = -Wy / rho**2 + np.einsum("nk,nkm->km", posts_marg, c**2 * yr / denom**2)
        rho_new = np.maximum(rho - g / gp, tol)
        if np.max(np.abs(rho_new - rho) / rho) < newton_tol:
            return rho_new
        rho = rho_new
    return rho


def _neg_Q_phi(log_phi, mu, y, w):
    """Negative posterior-weighted NB log-likelihood as a function of log-phi.

    Args:
        log_phi: scalar log-dispersion.
        mu:      NB means, broadcastable with y and w.
        y:       observed counts, broadcastable with mu.
        w:       posterior weights, same shape as the mu*y grid.

    Returns:
        scalar negative weighted log-likelihood.
    """
    r = 1.0 / np.exp(log_phi)
    frac = mu / (r + mu)
    ll = gammaln(y + r) - gammaln(r) + r * np.log1p(-frac) + xlogy(y, frac)
    return -np.sum(w * ll)


def _cm_phi_negbinom(
    X_counts, offset, posts_marg, rho, phi, min_covar, max_phi, share_phi
):
    """CM-step for phi: NB dispersion MLE by Brent in log-phi at fixed rho.

    With share_phi=True a single phi per sample (pooling all clusters via the
    posterior weights) is fit and broadcast to all K rows; with share_phi=False
    phi is fit independently per (cluster, sample). The search runs over
    [log(min_covar), log(max_phi)].

    Args:
        X_counts:   (N, M) counts.
        offset:     (N, M) lambda_i * T_s.
        posts_marg: (N, K) cluster-marginal posteriors.
        rho:        (K, M) fixed relative-copy state.
        phi:        (K, M) current dispersion (returned shape).
        min_covar:  phi floor (lower Brent bound).
        max_phi:    phi ceiling (upper Brent bound).
        share_phi:  tie phi across clusters within a sample.

    Returns:
        (K, M) updated phi.
    """
    M = X_counts.shape[1]
    K = posts_marg.shape[1]
    phi_new = phi.copy()
    lo, hi = np.log(min_covar), np.log(max_phi)

    for s in range(M):
        if share_phi:
            mu = offset[:, s][:, None] * rho[:, s][None, :]  # (N, K)
            y = X_counts[:, s][:, None]  # (N, 1)
            res = minimize_scalar(
                _neg_Q_phi,
                bounds=(lo, hi),
                method="bounded",
                args=(mu, y, posts_marg),
            )
            phi_new[:, s] = np.exp(res.x)
        else:
            for k in range(K):
                mu = offset[:, s] * rho[k, s]  # (N,)
                res = minimize_scalar(
                    _neg_Q_phi,
                    bounds=(lo, hi),
                    method="bounded",
                    args=(mu, X_counts[:, s], posts_marg[:, k]),
                )
                phi_new[k, s] = np.exp(res.x)
    return phi_new


def _weighted_ll_negbinom(X_counts, offset, posts_marg, rho, phi):
    """Posterior-weighted NB log-likelihood (drops the y! constant).

    Used as the ECM convergence monitor; the constant gammaln(y+1) is omitted
    since it does not affect the stopping test.

    Args:
        X_counts:   (N, M) counts.
        offset:     (N, M) lambda_i * T_s.
        posts_marg: (N, K) cluster-marginal posteriors.
        rho:        (K, M) relative-copy state.
        phi:        (K, M) dispersion.

    Returns:
        float scalar.
    """
    mu = offset[:, None, :] * rho[None, :, :]  # (N, K, M)
    r = 1.0 / phi[None, :, :]  # (1, K, M)
    y = X_counts[:, None, :]  # (N, 1, M)
    frac = mu / (r + mu)  # (N, K, M)
    ll = gammaln(y + r) - gammaln(r) + r * np.log1p(-frac) + xlogy(y, frac)
    return float(np.sum(posts_marg * np.sum(ll, axis=2)))


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

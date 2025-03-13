import os
import sys

import pandas as pd
import numpy as np

import hatchet.utils.Supporting as sp


def get_scaling_factor_no_WGD(
    seg: pd.DataFrame, samples: list, cluster_sizes: dict, tol_baf: float, v=1
):
    """
    find the netural clonal cluster s with BAF~0.5
    estimate rd scaling factors when no WGD
    """
    s = None
    for cid, _ in sorted(cluster_sizes.items(), key=lambda tp: tp[1], reverse=True):
        if all(abs(seg.loc[seg["#ID"] == cid, "BAF"] - 0.5) <= tol_baf):
            s = cid
            break
    
    if s == None:
        sp.log(msg=f"ERROR! unable to locate netural cluster with td={tol_baf}\n", level="ERROR")
        assert s != None

    gammas = {}
    for p in samples:
        rd = seg.loc[(seg["SAMPLE"] == p) & (seg["#ID"] == s), "RD"].iloc[0]
        assert rd > 0
        gammas[p] = 2 / rd
    return s, gammas


def get_scaling_factor_WGD(
    seg: pd.DataFrame,
    sid: int,
    cluster_sizes: dict,
    max_cn: int,
    lb_purity: float,
    rd_tol: float,
    baf_tol: float,
    v=1,
):
    """
    under WGD, compute RD scaling factor for each sample

    assumptions:
    1. neutral clonal cluster s has copynumber cs=4
    2. another cluster z is candidate clonal for sample p if the following holds:
        1. z is not neutral, i.e., |RD(z, p)-RD(s, p)| > <rd_tol>
        2. cz=0...3 if RD(z, p) < RD(s, p), and
        3. cz=5..<max_cn> if RD(z, p) > RD(s, p)
        4. for any cz, inferred tumor purity must above <lb_purity>,
        5. and have minimum BAF-error < <baf_tol>
        6. z has lowest copy-number cz that satisfies (4) and (5).
    3. among all candidate clonal cluster, the most weighted 
       cluster z that appears in all samples are chosen.
    4. cz is selected as the lowest value among all samples.
    """

    def get_gamma(rds: float, rdz: float, cz: int):
        dom = (cz - 2) * rds - 2 * rdz
        if dom == 0.0:
            return -1
        return (2 * cz - 8) / dom

    def get_purity(rds: float, gamma: float):
        if gamma <= 0:
            return -1
        return (rds * gamma / 2) - 1

    def get_mBAF(b: int, c: int, purity: float):
        num = 1 + (b - 1) * purity
        dom = 2 + (c - 2) * purity
        if num <= 0 or dom <= 0:
            return -1
        return num / dom

    samples = seg["SAMPLE"].unique()
    clonals = {}
    for p in samples:
        clonals[p] = {}  # stores a list of candidate clonal clusters
        seg_p = seg[seg["SAMPLE"] == p]
        s = seg_p[seg["#ID"] == sid]
        rds = float(s.iloc[0]["RD"])
        cs = 4  # fixed constant for neutral cluster clonal copy-number

        for _, z in seg_p.iterrows():
            zid = z["#ID"]
            rdz = float(z["RD"])
            bafz = float(z["BAF"])
            if zid == sid:
                continue
            # potential total copy-numbers for clonal cluster
            czs = None
            if rdz < (rds - rd_tol):
                czs = [c for c in range(1, cs)]
            elif rdz > (rds + rd_tol):
                czs = [c for c in range(5, max_cn + 1)]
            if czs == None:
                # similar RD as netural cluster is not informative
                continue

            # all possible cz that passes baf-tol threshold
            cz_stats = []
            for cz in czs:
                # for fixed cz, we can compute gamma and purity.
                gamma = get_gamma(rds, rdz, cz)
                purity = get_purity(rds, gamma)
                if gamma <= 0 or purity < lb_purity or purity > 1:
                    continue

                bzs = {}
                for bz in range(0, cz + 1):
                    bafz_ = get_mBAF(bz, cz, purity)
                    bzs[bz] = abs(bafz - bafz_)

                bz, bz_err = min(bzs.items(), key=lambda tp: tp[1])
                if bz_err < baf_tol:
                    cz_stats.append([zid, cz, gamma, purity, (cz - bz, bz), bz_err])
            if len(cz_stats) == 0:
                continue

            # pick the lowest cz among all candidate czs
            pz_result = min(cz_stats, key=lambda elem: elem[1])
            clonals[p][pz_result[0]] = pz_result

    # find maximum-weighted clonal cluster z that appears as candidate to all samples
    for zid in sorted(cluster_sizes.keys(), key=lambda z_: cluster_sizes[z_], reverse=True):
        if all(zid in clonals[p] for p in samples):
            final_clonals = {}  # may also be useful.
            gammas = {}
            for p in samples:
                final_clonals[p] = clonals[p][zid]
                gammas[p] = final_clonals[p][2]
            # pick the lowest cz=(az, bz) among all samples
            cz = min(final_clonals.values(), key=lambda val: val[1])[4]  # (az, bz)
            return zid, cz, gammas
    return None, None, None

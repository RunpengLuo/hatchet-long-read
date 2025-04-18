import os
import sys

import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde
from collections import OrderedDict

import hatchet.utils.Supporting as sp


def ovl_from_samples(rd1, rd2, grid_points=1000):
    kde1 = gaussian_kde(rd1)
    kde2 = gaussian_kde(rd2)

    xmin = min(np.min(rd1), np.min(rd2))
    xmax = max(np.max(rd1), np.max(rd2))
    x = np.linspace(xmin, xmax, grid_points)

    pdf1 = kde1(x)
    pdf2 = kde2(x)

    overlap = np.trapz(np.minimum(pdf1, pdf2), x)

    return overlap, x, pdf1, pdf2


def locate_balanced_clusters(seg: pd.DataFrame, tol_baf: float):
    """
    detect balanced clusters via <tol_baf> threshold
    """
    baf = seg.pivot(index="#ID", columns="SAMPLE", values="BAF")
    clusters = baf.index.tolist()
    balanced_s = []
    unbalanced_z = []
    for cid in clusters:
        if all(abs(baf.loc[cid, :] - 0.5) <= tol_baf):
            balanced_s.append(cid)
        else:
            unbalanced_z.append(cid)
    return balanced_s, unbalanced_z


def pairwise_merge(
    samples: list,
    seg: pd.DataFrame,
    bbc: pd.DataFrame,
    balanced_s: list,
    tol_area: float,
    v=1,
):
    """
    compute overlap area (OVL) for adjacent balanced clusters, merge if OVL >= <tol_area>
    """
    if len(balanced_s) <= 1:
        return bbc, seg, balanced_s
    rdr = seg.pivot(index="#ID", columns="SAMPLE", values="RD")
    balanced_s = sorted(balanced_s, key=lambda s: rdr.loc[s, :].mean())

    merged_ss = [balanced_s[0]]
    for curr_i in range(1, len(balanced_s)):
        s1, s2 = merged_ss[-1], balanced_s[curr_i]
        ovls = np.zeros(len(samples), dtype=np.float64)
        for j, sample in enumerate(samples):
            _bbc = bbc[(bbc["SAMPLE"] == sample)]
            x1 = _bbc.loc[_bbc["CLUSTER"] == s1, "RD"].to_numpy()
            x2 = _bbc.loc[_bbc["CLUSTER"] == s2, "RD"].to_numpy()
            ovl, x, p1, p2 = ovl_from_samples(x1, x2)
            ovls[j] = ovl
        if v >= 1:
            sp.log(msg=f"{s1}-{s2}\tOVL={ovls}\n", level="INFO")
        if np.any(ovls <= tol_area):
            merged_ss.append(s2)
            continue
        # merge s1 and s2
        sp.log(msg=f"merge {s1} and {s2}, drop {s2}\n", level="INFO")
        bbc.loc[bbc["CLUSTER"] == s2, "CLUSTER"] = s1
        bbc.loc[bbc["#ID"] == s2, "#ID"] = s1
        for sample in samples:
            # BINS	RD	#SNPS	COV	ALPHA	BETA	BAF
            _seg = seg.loc[
                (seg["#ID"] == s1) & (seg["SAMPLE"] == sample), :
            ]
            _bins = _seg["#BINS"].sum()
            _rd = (
                _seg.apply(func=lambda r: r["RD"] * r["#BINS"], axis=1).sum()
                / _bins
            )
            _snps = _seg["#SNPS"].sum()
            _cov = (
                _seg.apply(func=lambda r: r["COV"] * r["#BINS"], axis=1).sum()
                / _bins
            )
            _alpha = _seg["ALPHA"].sum()
            _beta = _seg["BETA"].sum()
            _baf = (
                _seg.apply(func=lambda r: r["BAF"] * r["#BINS"], axis=1).sum()
                / _bins
            )
            print(_seg)
            print(s1, _bins, _rd, _snps, _cov, _alpha, _beta, _baf)
            seg.loc[(seg["#ID"] == s1) & (seg["SAMPLE"] == sample), :] = [
                s1,
                _bins,
                _rd,
                _snps,
                _cov,
                _alpha,
                _beta,
                _baf,
            ]
    seg = seg.drop_duplicates(["#ID", "SAMPLE"], keep="first")
    sp.log(msg=f"balanced clusters after merge: {merged_ss}\n", level="INFO")
    return bbc, seg, merged_ss

# clonal cluster z (a,b)
def get_purity_by_baf(bafz: float, a: int, b: int):
    """
    compute tumor purity by BAF and copy-number (a,b)
    """
    num = (2 * bafz - 1)
    dom = (b - 1) - bafz * (a + b - 2)
    return -1 if dom == 0 else num / dom

def get_purity_by_rrd(rrdz: float, a: int, b: int, is_wgd=True):
    """
    compute tumor purity by RRD and copy-number (a,b)
    assumes a + b > 2 when no WGD
    """
    num = 2 * rrdz - 2
    if is_wgd:
        dom = a + b - 2 - 2 * rrdz
    else:
        dom = a + b - 2
    return -1 if dom == 0 else num / dom

def purity_est_err(bafz: float, rrdz: float, a: int, b: int, is_wgd: bool):
    """
    estimate tumor purity by 1) BAF and 2) RRD with given cn (a,b)
    return: estimation error
    """
    pbaf = get_purity_by_baf(bafz, a, b)
    prrd = get_purity_by_rrd(rrdz, a, b, is_wgd)
    if pbaf <= 0.0 or pbaf > 1.0:
        return np.inf, pbaf, prrd
    if prrd <= 0.0 or prrd > 1.0:
        return np.inf, pbaf, prrd
    return abs(pbaf - prrd), pbaf, prrd

def get_gamma_WGD(rds: float, rdz: float, cz: int):
    dom = (cz - 2) * rds - 2 * rdz
    if dom == 0.0:
        return -1
    return (2 * cz - 8) / dom

def get_scaling_factor(samples: list,
    seg: pd.DataFrame,
    bbc: pd.DataFrame,
    balanced_s: list,
    unbalanced_z: list,
    tol_rd_ratio: float,
    tol_err: float,
    maxcn: int,
    maxcn_wgd: int,
    v=1
    ):
    """
    Compute scaling factors
    """
    gammas_noWGD = {}
    purities_noWGD = {}
    pair_noWGD = None

    gammas_WGD = {}
    purities_WGD = {}
    pair_WGD = None

    rdr = seg.pivot(index="#ID", columns="SAMPLE", values="RD")
    baf = seg.pivot(index="#ID", columns="SAMPLE", values="BAF")
    balanced_s = sorted(balanced_s, key=lambda s: rdr.loc[s, :].mean())

    if len(balanced_s) >= 2:
        # TODO also reason about (0,0) or (2,2) base?
        s0, s1 = balanced_s[0], balanced_s[1]
        pair_noWGD = (s0, s1, (1,1), (2,2))
        sp.log(msg=f"found >1 balanced clusters, assign (1,1) and (2,2) to {s0} and {s1}\n", level="INFO")
        for sample in samples:
            gamma = rdr.loc[s0, sample] / 2
            gammas_noWGD[sample] = gamma
            purity = (0.5 * rdr.loc[s1, sample] / gamma) - 1
            purities_noWGD[sample] = purity
        return s0, pair_noWGD, gammas_noWGD, pair_WGD, gammas_WGD

    # found one unbalanced cluster z that pairs with s0 across all samples
    s0 = balanced_s[0]
    assert np.all(rdr.loc[s0, :] > 0), f"balanced cluster {s0} has RD<=0, invalid value"
    rrdr = rdr / rdr.loc[s0, :]

    is_cand_loh = {z: True for z in unbalanced_z}
    # LOH cluster cannot have any lower-RD cluster also has lower BAF.
    for z in unbalanced_z:
        for _z in unbalanced_z:
            if np.all(rdr.loc[_z] < rdr.loc[z]):
                if np.any(baf.loc[_z] < baf.loc[z]):
                    is_cand_loh[z] = False
                    break
    
    for z in sorted(unbalanced_z, key=lambda z: baf.loc[z, :].mean()):
        if not is_cand_loh[z]:
            sp.log(msg=f"\t({z},{s0}) cannot be LOH pair, skip.\n")
            continue
        rd_ratio_zs = rdr.loc[z] / rdr.loc[s0]
        rd_dist_zs = rdr.loc[z] - rdr.loc[s0]
        if np.all(np.abs(rd_ratio_zs - 1) <= tol_rd_ratio):
            sp.log(msg=f"-----------z={z} pair {s0}\n", level="STEP")
            # case 1, (1,1) and (2,0)
            if pair_noWGD == None:
                for sample in samples:
                    pbaf = get_purity_by_baf(baf.loc[z, sample], 2, 0)
                    if pbaf > 0.0 and pbaf <= 1.0:
                        purities_noWGD[sample] = pbaf
                    else:
                        purities_noWGD[sample] = None
                        break
                if all(p != None for p in purities_noWGD.values()):
                    pair_noWGD = (s0, z, (1,1), (2,0))
            
            # case 2, (2,2) and (4,0)
            if pair_WGD == None:
                for sample in samples:
                    perr, pbaf, prrd = purity_est_err(baf.loc[z, sample], rrdr.loc[z, sample], 4, 0, True)
                    if perr <= tol_err:
                        purities_WGD[sample] = (pbaf + prrd)/2
                    else:
                        purities_noWGD[sample] = None
                        break
                if all(p != None for p in purities_WGD.values()):
                    pair_WGD = (s0, z, (2,2), (4,0))
        elif np.all(rd_dist_zs > 0):
            sp.log(msg=f"-----------z={z} above {s0}\n", level="STEP")
            if pair_noWGD == None:
                lohs_nowgd = [(a, 0) for a in range(3, maxcn + 1)]
                for (a, b) in lohs_nowgd:
                    for sample in samples:
                        perr, pbaf, prrd = purity_est_err(baf.loc[z, sample], rrdr.loc[z, sample], a, b, False)
                        if perr <= tol_err:
                            purities_noWGD[sample] = (pbaf + prrd)/2
                        else:
                            purities_noWGD[sample] = None
                            break
                    if all(p != None for p in purities_noWGD.values()):
                        pair_noWGD = (s0, z, (1,1), (a,b))
                        break

            if pair_WGD == None:
                lohs_wgd = [(a, 0) for a in range(4, maxcn_wgd + 1)]
                for (a, b) in lohs_wgd:
                    for sample in samples:
                        perr, pbaf, prrd = purity_est_err(baf.loc[z, sample], rrdr.loc[z, sample], a, b, True)
                        if perr <= tol_err:
                            purities_WGD[sample] = (pbaf + prrd)/2
                        else:
                            purities_WGD[sample] = None
                            break
                    if all(p != None for p in purities_WGD.values()):
                        pair_WGD = (s0, z, (2,2), (a,b))
                        break
        elif np.all(rd_dist_zs < 0):
            sp.log(msg=f"-----------z={z} below {s0}\n", level="STEP")
            if pair_noWGD == None:
                lohs_nowgd = [(1,0)]
                for (a, b) in lohs_nowgd:
                    for sample in samples:
                        perr, pbaf, prrd = purity_est_err(baf.loc[z, sample], rrdr.loc[z, sample], a, b, False)
                        if perr <= tol_err:
                            purities_noWGD[sample] = (pbaf + prrd)/2
                        else:
                            purities_noWGD[sample] = None
                            break
                    if all(p != None for p in purities_noWGD.values()):
                        pair_noWGD = (s0, z, (1,1), (a,b))
                        break
                    
            if pair_WGD == None:
                lohs_wgd = [(1,0), (2,0), (3,0)]
                for (a, b) in lohs_wgd:
                    for sample in samples:
                        perr, pbaf, prrd = purity_est_err(baf.loc[z, sample], rrdr.loc[z, sample], a, b, True)
                        if perr <= tol_err:
                            purities_WGD[sample] = (pbaf + prrd)/2
                        else:
                            purities_WGD[sample] = None
                            break
                    if all(p != None for p in purities_WGD.values()):
                        pair_WGD = (s0, z, (2,2), (a,b))
                        break
        else:
            sp.log(msg=f"cluster {z} has inconsistent relative position to {s0} across samples\n", level="WARN")
            sp.log(msg=f"RD-distance(z,s)={rd_dist_zs}\n", level="WARN")
            sp.log(msg=f"RD-ratio(z,s)={rd_ratio_zs}\n", level="WARN")

        if pair_noWGD != None and pair_WGD != None:
            break
    
    # if pair_noWGD != None:
    #     (_, z, (sa, sb), (za, zb)) = pair_noWGD
    # in noWGD case, pair is not required.
    for sample in samples:
        gamma = rdr.loc[s0, sample] / 2
        gammas_noWGD[sample] = gamma
    
    if pair_WGD != None:
        (_, z, (_, _), (za, zb)) = pair_WGD
        for sample in samples:
            gamma = get_gamma_WGD(rdr.loc[s0, sample], rdr.loc[z, sample], za + zb)
            gammas_WGD[sample] = gamma
    
    return s0, pair_noWGD, gammas_noWGD, pair_WGD, gammas_WGD


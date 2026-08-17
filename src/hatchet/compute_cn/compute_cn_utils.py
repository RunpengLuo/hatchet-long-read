import os
import re
import glob
import logging
import pandas as pd
import numpy as np

from hatchet.utils import build_seg_from_bbc
from hatchet.io_utils import read_region_bed, write_ucn_wide
from hatchet import const


def compute_fractional_cn(input_data, gammas, alpha=0.05, min_ci_margin=0.1):
    """Compute fractional copy numbers and CI.

    Returns a new dict containing all fields from input_data plus
    fcn, fa, fb, fa_lo, fa_hi, fb_lo, fb_hi. input_data is not modified.

    Args:
        input_data: solver input dict with rdr, baf, rdr_se.
        gammas: dict or Series of per-sample gamma values.
        alpha: significance level (default 0.05 → 95% CI).
        min_ci_margin: hard minimum CI half-width in FCN space.
    """
    from scipy.stats import norm

    rdr = input_data["rdr"]
    baf = input_data["baf"]
    rdr_se = input_data["rdr_se"]

    gammas = pd.Series(gammas).sort_index()
    fcn = rdr * gammas
    fb = fcn * baf
    fa = fcn - fb

    z = norm.ppf(1 - alpha / 2)
    margin_fa = np.maximum(z * gammas * (1 - baf) * rdr_se, min_ci_margin)
    margin_fb = np.maximum(z * gammas * baf * rdr_se, min_ci_margin)

    return {
        **input_data,
        "fcn": fcn,
        "fa": fa,
        "fb": fb,
        "fa_lo": fa - margin_fa,
        "fa_hi": fa + margin_fa,
        "fb_lo": fb - margin_fb,
        "fb_hi": fb + margin_fb,
    }


def store_gammas(out_file, scaling, samples):
    """Write per-sample gamma values for all ploidies.

    Args:
        out_file: output TSV path.
        scaling: dict from get_scaling_factor with 'diploid', 'tetraploid' keys.
        samples: ordered sample list.
    """
    with open(out_file, "w") as fd:
        for sample in samples:
            g_dip = scaling["diploid"]["gammas"].get(sample, 0)
            g_tet = (
                scaling["tetraploid"]["gammas"].get(sample, 0)
                if scaling["tetraploid"]
                else 0
            )
            fd.write(f"{sample}\t{g_dip}\t{g_tet}\n")


def store_solve_input(out_file, input_data):
    """Write solver input (FCN + weights) to a TSV."""
    weights = input_data["weights"]
    nbins = input_data["nbins"]
    fcn_cols = ["fcn", "fa", "fb", "fa_lo", "fa_hi", "fb_lo", "fb_hi"]
    header = "CLUSTER\tSAMPLE\t#BINS\t" + "\t".join(fcn_cols) + "\tweight"
    fa = input_data["fa"]
    with open(out_file, "w") as fd:
        fd.write(header + "\n")
        for sample in fa.columns:
            for cid in fa.index:
                nb = int(nbins.loc[cid, sample])
                vals = [str(input_data[c].loc[cid, sample]) for c in fcn_cols]
                fd.write(
                    f"{cid}\t{sample}\t{nb}\t" + "\t".join(vals) + f"\t{weights[cid]}\n"
                )


# === Solution persistence ===


def _write_solution_tsv(fd, input_data, cA, cB, u, n, cluster_ids, sample_ids, header):
    """Write a single solution's per-cluster/sample details to an open file."""
    cA_ = np.array(cA)
    cB_ = np.array(cB)
    u_ = np.array(u)
    exp_a = cA_ @ u_
    exp_b = cB_ @ u_

    fd.write(header + "\n")
    for ci, cid in enumerate(cluster_ids):
        for si, sample in enumerate(sample_ids):
            fa_lo = input_data["fa_lo"].iloc[ci, si]
            fa_hi = input_data["fa_hi"].iloc[ci, si]
            fb_lo = input_data["fb_lo"].iloc[ci, si]
            fb_hi = input_data["fb_hi"].iloc[ci, si]
            ea, eb = exp_a[ci, si], exp_b[ci, si]
            accepted = ea >= fa_lo and ea <= fa_hi and eb >= fb_lo and eb <= fb_hi
            n_bins = int(input_data["nbins"].loc[cid, sample])
            fields = [
                cid,
                sample,
                n_bins,
                input_data["fa"].loc[cid, sample],
                input_data["fb"].loc[cid, sample],
                ea,
                eb,
                fa_lo,
                fa_hi,
                fb_lo,
                fb_hi,
            ]
            for oi in range(n):
                fields.extend([f"{cA[ci][oi]}|{cB[ci][oi]}", u[oi][si]])
            fields.append(accepted)
            fd.write("\t".join(str(v) for v in fields) + "\n")


def store_instance_tofile(pool_instances, input_data, sol_dir, solve_mode):
    """Store all solution detail TSVs."""
    n = len(pool_instances[next(iter(pool_instances))]["cA"][0])
    cluster_ids = input_data["cluster_ids"]
    sample_ids = input_data["sample_ids"]
    clone_cols = ["cn_normal\tu_normal"] + [
        f"cn_clone{i}\tu_clone{i}" for i in range(1, n)
    ]
    cols = (
        [
            "CLUSTER",
            "SAMPLE",
            "#BINS",
            "f_a",
            "f_b",
            "exp_f_a",
            "exp_f_b",
            "fa_lo",
            "fa_hi",
            "fb_lo",
            "fb_hi",
        ]
        + clone_cols
        + ["ci_accepted"]
    )
    header = "\t".join(cols)

    for sol_id, sol in pool_instances.items():
        path = const.SOLUTION_TSV(sol_dir, solve_mode, sol_id)
        with open(path, "w") as fd:
            _write_solution_tsv(
                fd,
                input_data,
                sol["cA"],
                sol["cB"],
                sol["u"],
                n,
                cluster_ids,
                sample_ids,
                header,
            )


def update_objectives_tsv(sols_dir, new_df):
    """Merge per-restart objectives into a single sols/objectives.tsv.

    new_df carries columns ploidy, n, sol_id, restart_id, imf_obj, reg_obj. Existing
    rows for any (ploidy, n) present in new_df are replaced; rows for other (ploidy, n)
    are kept so partial reruns (some n skipped) don't lose their objectives. Read back
    by load_pool_from_disk to reconstruct pool objectives on rerun.
    """
    cols = ["ploidy", "n", "sol_id", "restart_id", "imf_obj", "reg_obj"]
    new_df = new_df[cols]
    path = const.OBJECTIVES_TSV(sols_dir)
    if os.path.exists(path):
        old = pd.read_csv(path, sep="\t")
        keys = set(map(tuple, new_df[["ploidy", "n"]].itertuples(index=False)))
        keep = ~old[["ploidy", "n"]].apply(tuple, axis=1).isin(keys)
        new_df = pd.concat([old[keep], new_df], ignore_index=True)
    new_df.to_csv(path, sep="\t", index=False)


def load_pool_from_disk(sol_dir, cluster_ids, sample_ids):
    """Read pool solution TSVs from sol_dir into {sol_id: {"imf_obj": ..., "cA": ..., ...}}.

    Per-solution objectives are reconstructed from the consolidated
    sols/objectives.tsv: for this (ploidy, n) the best (min imf_obj) restart per
    sol_id is taken, matching how the pool selects its representative at solve time.
    """
    ploidy, _, n = os.path.basename(sol_dir.rstrip("/")).rpartition("_n")
    obj_path = const.OBJECTIVES_TSV(os.path.dirname(sol_dir.rstrip("/")))
    obj_map = {}
    if os.path.exists(obj_path) and ploidy:
        odf = pd.read_csv(obj_path, sep="\t")
        odf = odf[(odf["ploidy"] == ploidy) & (odf["n"] == int(n))]
        best = odf.loc[odf.groupby("sol_id")["imf_obj"].idxmin()]
        obj_map = {
            str(r["sol_id"]): (float(r["imf_obj"]), float(r["reg_obj"]))
            for _, r in best.iterrows()
        }

    pool = {}
    for path in sorted(glob.glob(os.path.join(sol_dir, "*.tsv"))):
        basename = os.path.basename(path)
        # Match old format (sol*_pool*) or new format (mode_solid)
        m = re.match(r".*_sol([\d.]+)_pool(\d+)\.tsv", basename)
        if m:
            sol_id = f"p{m.group(1)}_s{m.group(2)}"
        else:
            m2 = re.match(r"(?:cd|ilp)_(.+)\.tsv", basename)
            if m2:
                sol_id = m2.group(1)
            else:
                continue

        sol = pd.read_csv(path, sep="\t")
        cn_cols = sorted(
            [c for c in sol.columns if c.startswith("cn_")],
            key=lambda c: (0 if c == "cn_normal" else 1, c),
        )
        u_cols = sorted(
            [c for c in sol.columns if c.startswith("u_")],
            key=lambda c: (0 if c == "u_normal" else 1, c),
        )

        sol_s = (
            sol[sol["SAMPLE"] == sample_ids[0]]
            .sort_values("CLUSTER")
            .reset_index(drop=True)
        )
        cA, cB = [], []
        for _, row in sol_s.iterrows():
            ca, cb = zip(
                *(
                    (int(a), int(b))
                    for a, b in (str(row[c]).split("|") for c in cn_cols)
                )
            )
            cA.append(list(ca))
            cB.append(list(cb))

        u = [
            [float(sol[sol["SAMPLE"] == sid].iloc[0][uc]) for sid in sample_ids]
            for uc in u_cols
        ]
        imf_obj, reg_obj = obj_map.get(sol_id, (0.0, 0.0))
        pool[sol_id] = {
            "imf_obj": imf_obj,
            "reg_obj": reg_obj,
            "cA": cA,
            "cB": cB,
            "u": u,
        }

    if pool:
        logging.info(f"loaded {len(pool)} pool solutions from {sol_dir}")
    return pool


# === Segmentation ===


def segmentation(
    cA,
    cB,
    u,
    input_data: dict,
    bins: pd.DataFrame,
    samples: list,
    rd_mat,
    cov_mat,
    baf_mat,
    alpha_mat,
    beta_mat,
    region_file: str,
    bbc_out_file=None,
    seg_out_file=None,
    is_wide_format: bool = False,
):
    """Annotate bins with inferred CN states and build a segment-level DataFrame.

    Args:
        cA: (num_clusters, num_clones) allele-A CN.
        cB: (num_clusters, num_clones) allele-B CN.
        u: (num_clones, num_samples) clone proportions.
        input_data: solver input dict with cluster_ids, sample_ids.
        bins: per-bin frame (#CHR, START, END, #SNPS, CLUSTER).
        samples: ordered tumor sample names (column order of the *_mat arrays).
        rd_mat, cov_mat, baf_mat, alpha_mat, beta_mat: (N, M) per-(bin, sample) fields.
        region_file: Path to region BED file.
        bbc_out_file: If provided, write annotated bin-level TSV.
        seg_out_file: If provided, write segment-level TSV.

    Returns:
        Segment-level DataFrame.
    """
    cluster_ids = input_data["cluster_ids"]
    sample_ids = input_data["sample_ids"]
    m = len(samples)
    df = pd.DataFrame(
        {
            "#CHR": np.repeat(bins["#CHR"].to_numpy(), m),
            "START": np.repeat(bins["START"].to_numpy(), m),
            "END": np.repeat(bins["END"].to_numpy(), m),
            "SAMPLE": np.tile(samples, len(bins)),
            "#SNPS": np.repeat(bins["#SNPS"].to_numpy(), m),
            "CLUSTER": np.repeat(bins["CLUSTER"].to_numpy(), m),
            "RD": rd_mat.ravel(),
            "COV": cov_mat.ravel(),
            "BAF": baf_mat.ravel(),
            "ALPHA": alpha_mat.ravel(),
            "BETA": beta_mat.ravel(),
        }
    )

    n_clone = len(cA[0])
    cN = pd.DataFrame(
        np.array(cA).astype(str) + "|" + np.array(cB).astype(str),
        index=cluster_ids,
        columns=["cn_normal"] + [f"cn_clone{i}" for i in range(1, n_clone)],
    )
    u_df = pd.DataFrame(u, index=range(n_clone), columns=sample_ids).T
    u_df.columns = ["u_normal"] + [f"u_clone{i}" for i in range(1, n_clone)]
    extra_columns = [col for pair in zip(cN.columns, u_df.columns) for col in pair]

    df = df.merge(cN, left_on="CLUSTER", right_index=True)
    df = df.merge(u_df, left_on="SAMPLE", right_index=True)
    df = df.sort_values(["#CHR", "START", "END", "SAMPLE"]).reset_index(drop=True)

    cn_cols = list(cN.columns)
    u_cols = list(u_df.columns)
    if bbc_out_file is not None:
        orig_cols = df.columns[: -2 * n_clone].tolist()
        bbc_df = df[orig_cols + extra_columns]
        if is_wide_format:
            write_ucn_wide(
                bbc_out_file,
                bbc_df,
                sample_ids,
                fixed_cols=["#CHR", "START", "END", "#SNPS", "CLUSTER"] + cn_cols,
                fmt_fields=["RD", "COV", "BAF", "ALPHA", "BETA"] + u_cols,
            )
        else:
            bbc_df.to_csv(bbc_out_file, sep="\t", index=False)

    regions = read_region_bed(region_file)
    seg_df = build_seg_from_bbc(df, regions)
    if seg_out_file is not None:
        if is_wide_format:
            write_ucn_wide(
                seg_out_file,
                seg_df,
                sample_ids,
                fixed_cols=["#CHR", "START", "END", "CLUSTER"] + cn_cols,
                fmt_fields=u_cols,
            )
        else:
            seg_df.to_csv(seg_out_file, sep="\t", index=False)

    return seg_df

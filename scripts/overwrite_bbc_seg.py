import os
import sys

import numpy as np
import pandas as pd

from hatchet.compute_cn.compute_cn_utils import segmentation

# Overwrite HATCHet BBC with a pool solution file, produce bbc.ucn.tsv + seg.ucn.tsv
if __name__ == "__main__":
    if len(sys.argv) != 5:
        print(
            "usage: overwrite_bbc_seg.py <bbc> <solfile> <region_bed> <outprefix>",
            file=sys.stderr,
        )
        sys.exit(1)
    _, fbbc, solfile, region_file, outprefix = sys.argv
    solID = os.path.splitext(os.path.basename(solfile))[0]
    print(f"overwrite BBC with solution {solID}")

    bbc = pd.read_table(fbbc, sep="\t")
    sol = pd.read_table(solfile, sep="\t")

    # Ordered clone columns (normal first) from the solution file
    cn_cols = sorted(
        [c for c in sol.columns if c.startswith("cn_")],
        key=lambda c: (0 if c == "cn_normal" else 1, c),
    )
    u_cols = sorted(
        [c for c in sol.columns if c.startswith("u_")],
        key=lambda c: (0 if c == "u_normal" else 1, c),
    )

    sample_ids = sol["SAMPLE"].drop_duplicates().tolist()
    samples = sample_ids

    # Reconstruct cA, cB (num_clusters, num_clones) from one sample slice; cluster order
    # defines cluster_ids used to index CN back onto bins.
    sol_s = (
        sol[sol["SAMPLE"] == sample_ids[0]].sort_values("CLUSTER").reset_index(drop=True)
    )
    cluster_ids = sol_s["CLUSTER"].tolist()
    cA, cB = [], []
    for _, row in sol_s.iterrows():
        ca, cb = zip(
            *((int(a), int(b)) for a, b in (str(row[c]).split("|") for c in cn_cols))
        )
        cA.append(list(ca))
        cB.append(list(cb))

    # u (num_clones, num_samples) indexed by sample_ids
    u = [
        [float(sol[sol["SAMPLE"] == sid].iloc[0][uc]) for sid in sample_ids]
        for uc in u_cols
    ]

    # Pivot per-bin fields into (n_bins, n_samples) matrices aligned to bins rows
    bin_cols = ["#CHR", "START", "END", "#SNPS", "CLUSTER"]

    def mat(field):
        p = bbc.pivot_table(
            index=bin_cols, columns="SAMPLE", values=field, sort=False
        )
        return p.reindex(columns=samples)

    piv_rd = mat("RD")
    bins = piv_rd.index.to_frame(index=False)
    rd_mat = piv_rd.to_numpy()
    cov_mat = mat("COV").to_numpy()
    baf_mat = mat("BAF").to_numpy()
    alpha_mat = mat("ALPHA").to_numpy()
    beta_mat = mat("BETA").to_numpy()

    input_data = {"cluster_ids": cluster_ids, "sample_ids": sample_ids}

    seg_df = segmentation(
        cA,
        cB,
        u,
        input_data,
        bins=bins,
        samples=samples,
        rd_mat=rd_mat,
        cov_mat=cov_mat,
        baf_mat=baf_mat,
        alpha_mat=alpha_mat,
        beta_mat=beta_mat,
        region_file=region_file,
        bbc_out_file=f"{outprefix}.bbc.ucn.tsv",
        seg_out_file=f"{outprefix}.seg.ucn.tsv",
        is_wide_format=False,
    )
    print(
        f"wrote {outprefix}.bbc.ucn.tsv and {outprefix}.seg.ucn.tsv "
        f"({len(seg_df)} segments)"
    )

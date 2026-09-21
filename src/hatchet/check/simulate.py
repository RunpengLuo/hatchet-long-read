"""Simulate a small synthetic bb_dir for `hatchet check`.

One matched normal plus one tumor sample over four arm-level regions carrying
distinct copy-number states. Per bin, RDR is the mixed-clone expectation plus
Gaussian noise, and B-allele counts are Beta-Binomial around the mixed-clone
BAF. Everything is driven by a fixed seed, so a run is reproducible.
"""

import os

import numpy as np
import pandas as pd

from hatchet import const

# NB: hg38 sizes and approximate arm boundaries (UCSC cytoBand)
GENOME_SIZES = {"chr1": 248956422, "chr22": 50818468}
REGIONS = [
    ("chr1", 0, 121700000, "chr1_p-arm"),
    ("chr1", 121700000, 248956422, "chr1_q-arm"),
    ("chr22", 0, 13000000, "chr22_p-arm"),
    ("chr22", 13000000, 50818468, "chr22_q-arm"),
]
# Tumor clone (cn_a, cn_b) per region; the normal clone is 1|1 everywhere.
CN_PROFILE = {
    "chr1_p-arm": (1, 1),
    "chr1_q-arm": (2, 1),
    "chr22_p-arm": (1, 1),
    "chr22_q-arm": (1, 0),
}
PROP_NORMAL = 0.2
PROP_TUMOR = 0.8

BIN_SIZE = 1_000_000
SIGMA_RDR = 0.03
TAU_BAF = 100.0
DEPTH_LAMBDA = 100
NORMAL_DEPTH = 30.0
SWITCH_PROB = 1e-6


def simulate_bb_dir(out_dir, seed=42):
    """Write a synthetic bb_dir plus its genome.sizes and regions.bed.

    Args:
        out_dir: Directory to create; receives ``bb_dir/``, ``genome.sizes``
            and ``regions.bed``.
        seed: RandomState seed driving every draw.

    Returns:
        Tuple ``(bb_dir, genome_size, region_bed, ground_truth)`` of paths under
        ``out_dir`` plus the simulated truth (cn_profile, purity, gamma, regions).
    """
    rng = np.random.RandomState(seed)
    bb_dir = os.path.join(out_dir, "bb_dir")
    genome_size = os.path.join(out_dir, "genome.sizes")
    region_bed = os.path.join(out_dir, "regions.bed")
    os.makedirs(bb_dir, exist_ok=True)

    bins = [
        (chrom, start + i * BIN_SIZE, start + (i + 1) * BIN_SIZE, name)
        for chrom, start, end, name in REGIONS
        for i in range((end - start) // BIN_SIZE)
    ]
    n = len(bins)

    # NB: column 0 is the matched normal, column 1 the tumor
    rdr_mat = np.zeros((n, 1), dtype=np.float32)
    depth_mat = np.zeros((n, 2), dtype=np.float32)
    a_mat = np.zeros((n, 2), dtype=np.int32)
    b_mat = np.zeros((n, 2), dtype=np.int32)
    t_mat = np.zeros((n, 2), dtype=np.int32)
    n_snps = np.zeros(n, dtype=np.int32)

    for i, (_chrom, _start, _end, name) in enumerate(bins):
        cn_a, cn_b = CN_PROFILE[name]
        rdr, baf = _mixed_rdr_baf(cn_a, cn_b)

        obs_rdr = max(rdr + rng.normal(0, SIGMA_RDR), 0.01)
        rdr_mat[i, 0] = obs_rdr
        depth_mat[i] = (NORMAL_DEPTH, obs_rdr * NORMAL_DEPTH)

        total = max(rng.poisson(DEPTH_LAMBDA), 10)
        p = rng.beta(max(TAU_BAF * baf, 0.01), max(TAU_BAF * (1 - baf), 0.01))
        b_count = rng.binomial(total, p)
        a_mat[i, 1], b_mat[i, 1], t_mat[i, 1] = total - b_count, b_count, total

        normal_total = max(rng.poisson(DEPTH_LAMBDA), 10)
        normal_b = rng.binomial(normal_total, 0.5)
        a_mat[i, 0] = normal_total - normal_b
        b_mat[i, 0] = normal_b
        t_mat[i, 0] = normal_total
        n_snps[i] = total

    chroms, starts, ends, names = zip(*bins)
    pd.DataFrame(
        {
            "#CHR": chroms,
            "START": starts,
            "END": ends,
            "region_id": names,
            "switchprobs": SWITCH_PROB,
            "#SNPS": n_snps,
        }
    ).to_csv(
        os.path.join(bb_dir, const.BB_TSV_GZ), sep="\t", index=False, compression="gzip"
    )
    pd.DataFrame(
        {
            "SAMPLE": ["normal", "tumor1"],
            "sample_type": ["normal", "tumor"],
            "assay_type": ["wgs", "wgs"],
        }
    ).to_csv(os.path.join(bb_dir, const.SAMPLE_IDS), sep="\t", index=False)

    for fname, mat in (
        (const.BB_RDR_NPZ, rdr_mat),
        (const.BB_DEPTH_NPZ, depth_mat),
        (const.BB_A_ALLELE_NPZ, a_mat),
        (const.BB_B_ALLELE_NPZ, b_mat),
        (const.BB_T_ALLELE_NPZ, t_mat),
    ):
        np.savez(os.path.join(bb_dir, fname), mat=mat)

    with open(genome_size, "w") as f:
        for chrom, size in GENOME_SIZES.items():
            f.write(f"{chrom}\t{size}\n")
    with open(region_bed, "w") as f:
        for chrom, start, end, name in REGIONS:
            f.write(f"{chrom}\t{start}\t{end}\t{name}\n")

    ground_truth = {
        "cn_profile": CN_PROFILE,
        "purity": PROP_TUMOR,
        # NB: balanced-cluster RDR is 1.0 by construction, so gamma = 2/1
        "gamma": 2.0,
        "regions": REGIONS,
    }
    return bb_dir, genome_size, region_bed, ground_truth


def _mixed_rdr_baf(cn_a, cn_b):
    """Expected (RDR, BAF) of a normal/tumor mixture for one tumor CN state."""
    rdr = PROP_NORMAL * 1.0 + PROP_TUMOR * (cn_a + cn_b) / 2.0
    total = PROP_NORMAL * 2 + PROP_TUMOR * (cn_a + cn_b)
    return rdr, (PROP_NORMAL * 1 + PROP_TUMOR * cn_b) / total

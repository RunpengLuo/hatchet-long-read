"""Integration tests for the experimental wide-format BBC output (--wide_format).

Runs cluster-bins in long and wide mode on the synthetic fixture and checks the
single wide TSV round-trips to the same long BBC DataFrame, that only the intended
files are emitted, and that compute-cn produces identical results from either.
"""

import gzip
import os

import numpy as np
import pandas as pd
import pytest

from hatchet import filenames as fn
from hatchet.io_utils import read_bbc_file, read_wide_bbc
from tests.conftest import CBC_AVAILABLE

MAXK = 5
MINK = 3
EXACT_COLS = ["#CHR", "START", "END", "SAMPLE", "#SNPS", "CLUSTER", "BETA", "ALPHA"]
CLOSE_COLS = ["RD", "COV", "BAF"]


def _cluster_bins_args(bb_dir, bbc_dir, genome_sizes, regions_bed, wide):
    args = {
        "bb_dir": bb_dir,
        "bbc_dir": bbc_dir,
        "genome_size": genome_sizes,
        "region_bed": regions_bed,
        "maxK": MAXK,
        "restarts": 3,
        "top_restarts": 2,
        "n_local_trials": 2,
        "niters": 5,
        "tau_iters": 1,
        "force": True,
        "verbosity": 1,
    }
    if wide:
        args["wide_format"] = True
    return args


@pytest.fixture(scope="module")
def long_and_wide_dirs(synthetic_data, tmp_path_factory):
    """Run cluster-bins long and wide with identical (deterministic) settings."""
    bb_dir, genome_sizes, regions_bed, _ = synthetic_data
    from hatchet.cluster_bins.cluster_bins import run as run_cluster_bins

    long_dir = str(tmp_path_factory.mktemp("wide_long"))
    wide_dir = str(tmp_path_factory.mktemp("wide_wide"))
    run_cluster_bins(
        _cluster_bins_args(bb_dir, long_dir, genome_sizes, regions_bed, False)
    )
    run_cluster_bins(
        _cluster_bins_args(bb_dir, wide_dir, genome_sizes, regions_bed, True)
    )
    return long_dir, wide_dir


def _sorted(df):
    return df.sort_values(["#CHR", "START", "SAMPLE"]).reset_index(drop=True)


def test_wide_roundtrip_matches_long(long_and_wide_dirs):
    long_dir, wide_dir = long_and_wide_dirs
    df_long = _sorted(read_bbc_file(fn.BULK_BBC(long_dir, False)))
    df_wide = _sorted(read_wide_bbc(fn.BULK_BBC(wide_dir, True)))

    assert list(df_long.columns) == list(df_wide.columns)
    assert len(df_long) == len(df_wide)
    assert "COV-N" not in df_wide.columns
    for col in EXACT_COLS:
        assert (df_long[col].to_numpy() == df_wide[col].to_numpy()).all(), col
    for col in CLOSE_COLS:
        assert np.allclose(
            df_long[col].to_numpy(dtype=float), df_wide[col].to_numpy(dtype=float)
        ), col


def test_wide_header_and_files(long_and_wide_dirs):
    _long_dir, wide_dir = long_and_wide_dirs
    for path in (fn.BULK_BBC(wide_dir, True), fn.BULK_SEG(wide_dir, True)):
        assert os.path.exists(path), path
    for path in (
        fn.BULK_BBC(wide_dir, False),
        os.path.join(wide_dir, fn.BB_PHASED_TSV_GZ),
    ):
        assert not os.path.exists(path), path

    with gzip.open(fn.BULK_BBC(wide_dir, True), "rt") as fh:
        header = fh.readline().rstrip("\n").split("\t")
        row = fh.readline().rstrip("\n").split("\t")
    assert header == [
        "#CHR",
        "START",
        "END",
        "#SNPS",
        "CLUSTER",
        "PHASE",
        "PHASE_POSTS",
        "FORMAT",
        "tumor1",
    ], header
    assert "region_id" not in header and "switchprobs" not in header
    fmt = row[header.index("FORMAT")]
    assert fmt == "RD:COV:BAF:ALPHA:BETA", fmt
    assert len(row[header.index("tumor1")].split(":")) == len(fmt.split(":"))

    labels = fn.LABELS_DIR(wide_dir)
    for k in range(MINK, MAXK + 1):
        assert os.path.exists(fn.BULK_BBC_k(wide_dir, True, k)), k
        assert not os.path.exists(fn.BULK_BBC_k(wide_dir, False, k)), k
        assert not os.path.exists(fn.BULK_K_PHASED(wide_dir, k)), k
    assert not any(f.endswith(".npz") for f in os.listdir(labels))


def test_wide_seg_matches_long(long_and_wide_dirs):
    long_dir, wide_dir = long_and_wide_dirs
    seg_long = pd.read_table(fn.BULK_SEG(long_dir, False), sep="\t")
    seg_wide = pd.read_table(fn.BULK_SEG(wide_dir, True), sep="\t")
    pd.testing.assert_frame_equal(seg_long, seg_wide)


@pytest.mark.skipif(not CBC_AVAILABLE, reason="CBC solver not available")
def test_compute_cn_parity(long_and_wide_dirs, synthetic_data, tmp_path_factory):
    long_dir, wide_dir = long_and_wide_dirs
    _bb_dir, genome_sizes, regions_bed, _ = synthetic_data
    from hatchet.compute_cn.compute_cn import run as run_compute_cn

    def _run(bbc_file, seg_file, extra):
        result_dir = str(tmp_path_factory.mktemp("wide_cn"))
        args = {
            "bbc": bbc_file,
            "seg": seg_file,
            "result_dir": result_dir,
            "genome_size": genome_sizes,
            "region_bed": regions_bed,
            "solver": "cbc",
            "timelimit": 60,
            "maxClone": 2,
            "diploid": True,
            "force": True,
            "reg_steps": 3,
            "diploidcmax": 6,
            "cd_njobs": 1,
            "zero_cn_thres": 0.005,
            "verbosity": 1,
        }
        args.update(extra)
        run_compute_cn(args)
        return result_dir

    long_res = _run(fn.BULK_BBC(long_dir, False), fn.BULK_SEG(long_dir, False), {})
    wide_res = _run(
        fn.BULK_BBC(wide_dir, True),
        fn.BULK_SEG(wide_dir, True),
        {"wide_format": True},
    )

    ucn_long = pd.read_table(os.path.join(long_res, fn.BEST_BBC_UCN), sep="\t")
    ucn_wide = pd.read_table(os.path.join(wide_res, fn.BEST_BBC_UCN), sep="\t")
    cn_cols = [c for c in ucn_long.columns if c.startswith("cn_")]
    assert cn_cols
    left = ucn_long.sort_values(["#CHR", "START", "SAMPLE"]).reset_index(drop=True)
    right = ucn_wide.sort_values(["#CHR", "START", "SAMPLE"]).reset_index(drop=True)
    for col in cn_cols:
        assert (left[col].to_numpy() == right[col].to_numpy()).all(), col

    gam_long = pd.read_table(os.path.join(long_res, fn.GAMMAS), sep="\t", header=None)
    gam_wide = pd.read_table(os.path.join(wide_res, fn.GAMMAS), sep="\t", header=None)
    pd.testing.assert_frame_equal(gam_long, gam_wide)

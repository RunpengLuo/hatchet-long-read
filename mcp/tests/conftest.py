"""Shared fixtures: an isolated config file and a synthetic bb_dir."""

import csv
import gzip

import numpy as np
import pytest

HG38_SUBSET = {"chr1": 248956422, "chr22": 50818468}


@pytest.fixture(autouse=True)
def isolated_config(tmp_path, monkeypatch):
    """Point hatchet-mcp at a throwaway config file for every test."""
    monkeypatch.setenv("HATCHET_MCP_CONFIG", str(tmp_path / "config.json"))


def write_bb_dir(root, n_bins=10, tumors=("T1", "T2"), normal="N", chrom="chr22", rdr_cols=None):
    """Write a minimal bb_dir; ``rdr_cols`` overrides the tumor-matrix width."""
    root.mkdir(parents=True, exist_ok=True)
    samples = ([normal] if normal else []) + list(tumors)
    with open(root / "sample_ids.tsv", "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["SAMPLE", "sample_type", "assay_type"])
        for s in samples:
            w.writerow([s, "normal" if s == normal else "tumor", "bulkWES"])
    with gzip.open(root / "bb.tsv.gz", "wt", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["#CHR", "START", "END", "#SNPS", "region_id", "switchprobs"])
        for i in range(n_bins):
            w.writerow([chrom, i * 1000, (i + 1) * 1000, 0 if i == 0 else 5, f"{chrom}q", 0.01])
    n_t = len(tumors) if rdr_cols is None else rdr_cols
    np.savez(root / "bb.rdr.npz", mat=np.ones((n_bins, n_t), dtype=np.float32))
    for name in ("bb.depth.npz", "bb.Aallele.npz", "bb.Ballele.npz", "bb.Tallele.npz"):
        np.savez(root / name, mat=np.ones((n_bins, len(samples)), dtype=np.int32))
    return root


@pytest.fixture
def bb_dir(tmp_path):
    """A valid two-tumor, one-normal bb_dir on chr22."""
    return write_bb_dir(tmp_path / "bb")

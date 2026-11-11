"""bb_dir validation and summary statistics."""

from pathlib import Path

import pytest
from conftest import HG38_SUBSET, write_bb_dir

from hatchet_mcp.inputs import inspect_bb_dir, read_npz_shape

REPO = Path(__file__).resolve().parents[2]
SU006 = REPO / "tmp/su006/run_SU006_0907/bb_lowess_inv-var/MSR100/bulk"
REPO_REFERENCE = REPO / "data/reference"
SIZES = {"hg38": HG38_SUBSET, "tiny": {"chr22": 5000}}


def _repo_sizes():
    sizes = {}
    for genome, name in (
        ("hg38", "hg38.chrom.sizes"),
        ("hg19", "hg19.chrom.sizes"),
        ("chm13", "T2T-CHM13v2.0.sizes"),
    ):
        rows = (REPO_REFERENCE / name).read_text().splitlines()
        sizes[genome] = {c: int(s) for c, s in (r.split("\t")[:2] for r in rows if r)}
    return sizes


def test_npz_shape_read_without_loading(bb_dir):
    assert read_npz_shape(bb_dir / "bb.rdr.npz") == (10, 2)
    assert read_npz_shape(bb_dir / "bb.depth.npz") == (10, 3)


def test_valid_bb_dir_stats(bb_dir):
    report = inspect_bb_dir(str(bb_dir), "hg38", SIZES)
    assert report["valid"], report["errors"]
    assert report["samples"]["n_tumor"] == 2 and report["samples"]["has_normal"]
    assert report["bins"]["n_bins"] == 10
    assert report["bins"]["bins_without_snps"] == 1
    assert report["genome"]["compatible"] == ["hg38"]


def test_missing_file_is_reported(bb_dir):
    (bb_dir / "bb.Tallele.npz").unlink()
    report = inspect_bb_dir(str(bb_dir), None, SIZES)
    assert not report["valid"]
    assert "bb.Tallele.npz" in report["errors"][0]


def test_tumor_matrix_width_must_match(tmp_path):
    root = write_bb_dir(tmp_path / "bb", rdr_cols=3)
    report = inspect_bb_dir(str(root), None, SIZES)
    assert any("bb.rdr.npz" in e for e in report["errors"])


def test_coordinates_outside_genome_rejected(tmp_path):
    root = write_bb_dir(tmp_path / "bb", n_bins=10)
    report = inspect_bb_dir(str(root), "tiny", SIZES)
    assert any("outside tiny" in e for e in report["errors"])


def test_missing_normal_warns(tmp_path):
    root = write_bb_dir(tmp_path / "bb", normal=None)
    report = inspect_bb_dir(str(root), None, SIZES)
    assert report["valid"]
    assert any("matched-normal" in w for w in report["warnings"])


def test_unavailable_references_skip_genome_check(bb_dir):
    report = inspect_bb_dir(str(bb_dir), "hg38", None)
    assert report["valid"] and report["genome"]["compatible"] is None


@pytest.mark.skipif(not SU006.is_dir(), reason="SU006 bb_dir not present")
def test_real_wes_bb_dir_against_hatchet_references():
    report = inspect_bb_dir(str(SU006), "hg38", _repo_sizes())
    assert report["valid"], report["errors"]
    assert report["samples"]["n_tumor"] == 2 and report["samples"]["n_normal"] == 1
    assert report["matrices"]["bb.rdr.npz"] == [report["bins"]["n_bins"], 2]
    assert report["genome"]["compatible"] == ["hg38"]

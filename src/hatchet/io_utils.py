"""File readers and table reshaping for HATCHet inputs/outputs.

Readers for sample sheets, genome sizes, region BEDs, BBC and seg.ucn tables and
gamma files, plus :func:`override_solution` which loads an alternative
copy-number solution onto BBC/seg tables. Chromosome ordering and segment
merging live in :mod:`hatchet.utils`.
"""

import logging
from collections import OrderedDict

import numpy as np
import pandas as pd

from hatchet import filenames as fn
from hatchet.utils import build_seg_from_bbc, sort_df_chr


# =============================================================================
# File readers
# =============================================================================


def read_sample_file(sample_file: str):
    sample_df = pd.read_table(sample_file, sep="\t")
    sample_types = sample_df["sample_type"].tolist()
    if "normal" in sample_types:
        normal_idx = [i for i, t in enumerate(sample_types) if t == "normal"]
        tumor_idx = [i for i, t in enumerate(sample_types) if t == "tumor"]
    else:
        normal_idx = []
        tumor_idx = list(range(len(sample_types)))

    assert "assay_type" in sample_df.columns, "missing `assay_type` column."
    assays = sample_df["assay_type"].tolist()
    normal_set, tumor_set = set(normal_idx), set(tumor_idx)
    assay2samples = {}
    for i, a in enumerate(assays):
        grp = assay2samples.setdefault(a, {"normal": [], "tumor": []})
        if i in normal_set:
            grp["normal"].append(i)
        if i in tumor_set:
            grp["tumor"].append(i)
    return sample_df, normal_idx, tumor_idx, assay2samples


def read_genome_sizes(sz_file: str):
    chr_sizes = OrderedDict()
    with open(sz_file, "r") as rfd:
        for line in rfd.readlines():
            ch, sizes = line.strip().split()
            chr_sizes[ch] = int(sizes)
        rfd.close()
    return chr_sizes


def read_region_bed(bed_file: str, names=["#CHR", "START", "END", "NAME"]):
    regions = pd.read_table(
        bed_file,
        sep="\t",
        header=None,
        names=names,
    )
    return regions


def read_bbc_file(bbc_file: str, is_wide_format: bool = False):
    """Read a BBC table as the long DataFrame, from either layout.

    Args:
        bbc_file: Path to the BBC table (long ``bulk.bbc`` or, when
            ``is_wide_format``, the VCF-like wide ``bulk.bbc.tsv.gz``).
        is_wide_format: Parse the wide layout via :func:`read_wide_bbc`.

    Returns:
        Chromosome-sorted long BBC DataFrame.
    """
    if is_wide_format:
        return read_wide_bbc(bbc_file)
    df = pd.read_table(bbc_file, sep="\t")
    df = sort_df_chr(df, pos="START")
    return df


def read_wide_bbc(bbc_file: str):
    """Read a wide-format BBC (single gzip TSV) and return the long BBC frame.

    Parses the VCF-like layout (fixed bin columns + ``FORMAT`` key list + one
    colon-joined column per sample) back into the same long DataFrame
    :func:`read_bbc_file` returns. The wide file is self-contained; no ``bb_dir``
    is needed.

    Args:
        bbc_file: Path to the wide BBC (``bulk.bbc.tsv.gz`` or
            ``labels/bulk<K>.bbc.tsv.gz``).

    Returns:
        Chromosome-sorted long BBC DataFrame with columns #CHR, START, END,
        SAMPLE, #SNPS, CLUSTER, RD, COV, BAF, ALPHA, BETA.
    """
    fixed = ["#CHR", "START", "END", "#SNPS", "CLUSTER", "PHASE", "PHASE_POSTS"]
    wide = pd.read_table(bbc_file, sep="\t")
    fmt = wide["FORMAT"].iloc[0].split(":")
    samples = [c for c in wide.columns if c not in fixed and c != "FORMAT"]
    if not samples:
        raise ValueError(f"no sample columns found in wide BBC {bbc_file}")
    required = set(fn.WIDE_BBC_FIELDS)
    if not required.issubset(fmt):
        raise ValueError(
            f"wide BBC {bbc_file} FORMAT {fmt} missing required fields "
            f"{sorted(required - set(fmt))}"
        )
    n_bins, m = len(wide), len(samples)

    # split each sample's colon-joined cells into a (n_bins, len(fmt)) frame
    parsed = {}
    for s in samples:
        parts = wide[s].str.split(":", expand=True)
        if parts.shape[1] != len(fmt) or parts.isna().any().any():
            raise ValueError(
                f"wide BBC {bbc_file} sample {s!r} has cells not matching FORMAT {fmt}"
            )
        parts.columns = fmt
        parsed[s] = parts

    def _stack(field):
        return np.column_stack([parsed[s][field].to_numpy() for s in samples]).ravel()

    df = pd.DataFrame(
        {
            "#CHR": np.repeat(wide["#CHR"].to_numpy(), m),
            "START": np.repeat(wide["START"].to_numpy(), m),
            "END": np.repeat(wide["END"].to_numpy(), m),
            "SAMPLE": np.tile(samples, n_bins),
            "#SNPS": np.repeat(wide["#SNPS"].to_numpy(), m),
            "CLUSTER": np.repeat(wide["CLUSTER"].to_numpy(), m),
        }
    )
    # canonical long-frame field order (ALPHA before BETA), matching read_bbc_file
    for field in ("RD", "COV", "BAF", "ALPHA", "BETA"):
        df[field] = _stack(field).astype(fn.FORMAT_DTYPE[field])

    df = sort_df_chr(df, pos="START")
    return df


def read_seg_file(seg_file: str):
    """Read a cluster-bins SEG table (per-cluster, per-sample summary).

    The SEG rows are aggregated per cluster and carry no genomic coordinates, so
    the table is returned as-is without chromosome sorting.

    Args:
        seg_file: Path to a ``bulk.seg`` table with CLUSTER, SAMPLE, the per-cluster
            BAF/RD summary columns, and the ``is_filtered`` flag.

    Returns:
        The SEG DataFrame.
    """
    return pd.read_table(seg_file, sep="\t")


# =============================================================================
# File writers
# =============================================================================


def _fmt_field(field, v):
    """Format a per-sample field vector into VCF cell strings."""
    if fn.FORMAT_DTYPE[field].startswith("int"):
        return np.char.mod("%d", v.astype(np.int64))
    return np.char.mod("%.6g", v.astype(np.float64))


def write_bbc_file(
    bbc_file: str,
    bbs,
    k_labels,
    field_mats,
    tumor_samples,
    is_wide_format: bool = False,
):
    """Write a clustered BBC table in either the long or wide layout.

    The long N*M frame is materialized only for the non-wide path (inside
    :func:`_build_long_bbc`); the wide writer builds its cells directly from the
    ``(N, M)`` field matrices, so wide mode never allocates the long frame.

    Args:
        bbc_file: Output path (long ``bulk<K>.bbc`` or wide ``bulk<K>.bbc.tsv.gz``).
        bbs: Per-bin frame with #CHR, START, END, #SNPS, PHASE, PHASE_POSTS.
        k_labels: ``(N,)`` cluster label per bin.
        field_mats: dict of ``(N, M)`` per-(bin, sample) arrays keyed by FORMAT
            field (RD, COV, BAF, ALPHA, BETA).
        tumor_samples: Ordered tumor sample names = per-sample column order.
        is_wide_format: Write the VCF-like wide layout via :func:`write_wide_bbc`.
    """
    if is_wide_format:
        write_wide_bbc(bbc_file, bbs, k_labels, field_mats, tumor_samples)
    else:
        long_bbc = _build_long_bbc(bbs, k_labels, field_mats, tumor_samples)
        long_bbc.to_csv(bbc_file, sep="\t", header=True, index=False)


def _build_long_bbc(bbs, k_labels, field_mats, tumor_samples):
    """Materialize the long BBC frame (one row per bin x tumor sample).

    Columns follow the read contract order (#CHR, START, END, SAMPLE, #SNPS,
    CLUSTER, RD, COV, BAF, ALPHA, BETA) in bin-major/sample-minor rows.
    """
    m = len(tumor_samples)
    df = pd.DataFrame(
        {
            "#CHR": np.repeat(bbs["#CHR"].to_numpy(), m),
            "START": np.repeat(bbs["START"].to_numpy(), m),
            "END": np.repeat(bbs["END"].to_numpy(), m),
            "SAMPLE": np.tile(tumor_samples, len(bbs)),
            "#SNPS": np.repeat(bbs["#SNPS"].to_numpy(), m),
            "CLUSTER": np.repeat(k_labels, m),
        }
    )
    for field in ("RD", "COV", "BAF", "ALPHA", "BETA"):
        if field in field_mats:
            df[field] = field_mats[field].ravel()
    return df


def write_wide_bbc(bbc_gz_path, bbs, k_labels, field_mats, tumor_samples):
    """Serialize a clustered BBC as a single self-contained VCF-like wide TSV.

    One row per bin: fixed bin-level columns (#CHR, START, END, #SNPS, CLUSTER,
    PHASE, PHASE_POSTS), a constant ``FORMAT`` key list, and one column per tumor
    sample holding the fields colon-joined in FORMAT order. The file carries every
    per-sample value, so no ``bb_dir`` is needed to read it back.

    Args:
        bbc_gz_path: Output path for the gzip wide TSV.
        bbs: Per-bin frame carrying #CHR, START, END, #SNPS, and the current K's
            PHASE, PHASE_POSTS.
        k_labels: ``(N,)`` cluster label per bin.
        field_mats: dict of ``(N, M)`` per-(bin, sample) arrays keyed by FORMAT field.
        tumor_samples: Ordered tumor sample names = per-sample column order.
    """
    cols = {
        "#CHR": bbs["#CHR"].to_numpy(),
        "START": bbs["START"].to_numpy(),
        "END": bbs["END"].to_numpy(),
        "#SNPS": bbs["#SNPS"].to_numpy(),
        "CLUSTER": k_labels,
        "PHASE": bbs["PHASE"].to_numpy(),
        "PHASE_POSTS": bbs["PHASE_POSTS"].to_numpy(),
    }
    fields = [f for f in fn.WIDE_BBC_FIELDS if f in field_mats]
    cols["FORMAT"] = ":".join(fields)
    for si, sample in enumerate(tumor_samples):
        cell = _fmt_field(fields[0], field_mats[fields[0]][:, si])
        for f in fields[1:]:
            cell = np.char.add(
                np.char.add(cell, ":"), _fmt_field(f, field_mats[f][:, si])
            )
        cols[sample] = cell
    wide = pd.DataFrame(cols)
    wide.to_csv(bbc_gz_path, sep="\t", header=True, index=False, compression="gzip")


def read_seg_ucn_file(seg_ucn_file: str):
    """Read a seg.ucn table, sort by chromosome, and derive the clone list.

    Per-clone proportions are per sample (``u_<clone>`` columns), so read them at
    the call site from the relevant sample's rows rather than here.

    Args:
        seg_ucn_file: Path to a seg.ucn table with ``cn_<clone>`` / ``u_<clone>``
            columns.

    Returns:
        (df, clones): the chromosome-sorted table and the clone-name list
        ("normal", "clone1", ...).
    """
    segs_df = pd.read_table(seg_ucn_file, sep="\t")
    segs_df = sort_df_chr(segs_df, pos="START")
    n_clones = len([c for c in segs_df.columns if c.startswith("cn_")])
    clones = ["normal"] + [f"clone{c}" for c in range(1, n_clones)]
    return segs_df, clones


def read_gamma_file(gamma_file: str, is_diploid=True):
    """Read per-sample RDR scaling factors from a gammas.tsv file.

    Each line is ``sample\\tgamma_diploid\\tgamma_tetraploid``.

    Args:
        gamma_file: Path to the gammas TSV.
        is_diploid: Return the diploid gamma when True, else the tetraploid one.

    Returns:
        {sample: gamma} mapping.
    """
    gammas = {}
    with open(gamma_file, "r") as fd:
        for line in fd.readlines():
            sample, gamma_dip, gamma_tet = line.strip().split("\t")
            gammas[sample] = float(gamma_dip) if is_diploid else float(gamma_tet)
    return gammas


# =============================================================================
# Solution loading
# =============================================================================


def override_solution(
    bbcs: pd.DataFrame,
    samples: list,
    clusters: list,
    n_clones: int,
    solfile: str,
    regions: pd.DataFrame,
):
    """Overwrite BBC copy-number fields with an alternative solution TSV.

    Merges a per-cluster solution onto the bins and re-segments via
    :func:`hatchet.utils.build_seg_from_bbc`.

    Returns:
        (bbcs, segs, n_clones, n_tumors, solID).
    """
    solID = solfile[str.rindex(solfile, "/") + 1 : -len(".tsv")]
    logging.info(f"overwrite BBC fields with solution {solID}!")
    sol = pd.read_table(solfile)
    assert sorted(sol.CLUSTER.unique().tolist()) == clusters
    assert sorted(sol.SAMPLE.unique().tolist()) == samples

    clones = ["normal"] + [f"clone{i}" for i in range(1, n_clones)]
    for clone in clones:
        bbcs.drop(columns=[f"u_{clone}", f"cn_{clone}"], inplace=True)

    bbcs = pd.merge(
        left=bbcs,
        right=sol,
        on=["SAMPLE", "CLUSTER"],
        how="left",
        validate="m:1",
        sort=False,
    )

    n_tumors = len([c for c in bbcs.columns.tolist() if str.startswith(c, "cn_clone")])
    n_clones = n_tumors + 1
    segs = build_seg_from_bbc(bbcs, regions)
    return bbcs, segs, n_clones, n_tumors, solID

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

from hatchet import const
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
    """Read a BBC table into flat wide arrays (no long N*M frame).

    Args:
        bbc_file: Path to the BBC table (long ``bulk.bbc`` or, when
            ``is_wide_format``, the VCF-like wide ``bulk.bbc.tsv.gz``).
        is_wide_format: Parse the wide layout directly into the arrays.

    Returns:
        ``(bins, samples, rd_mat, cov_mat, baf_mat, alpha_mat, beta_mat)``.
        ``bins`` is a chromosome-sorted DataFrame with one row per bin
        (#CHR, START, END, #SNPS, CLUSTER); ``samples`` is the sorted tumor
        sample list; each ``*_mat`` is an ``(N, M)`` array aligned to ``bins``
        rows and ``samples`` columns, dtype per ``const.FORMAT_DTYPE``.
    """
    if is_wide_format:
        return _read_wide_bbc_mats(bbc_file)
    df = pd.read_table(bbc_file, sep="\t")
    df = sort_df_chr(df, pos="START")
    return _long_bbc_to_mats(df)


def _long_bbc_to_mats(df: pd.DataFrame):
    """Pivot a long BBC DataFrame into ``(bins, samples, *_mat)``."""
    samples = sorted(df["SAMPLE"].unique().tolist())
    df = df.copy()
    df["_bin"] = (
        df["#CHR"].astype(str)
        + ":"
        + df["START"].astype(str)
        + ":"
        + df["END"].astype(str)
    )
    bins = df.drop_duplicates("_bin")[
        ["#CHR", "START", "END", "#SNPS", "CLUSTER", "_bin"]
    ].reset_index(drop=True)
    bin_order = bins["_bin"].tolist()

    def _mat(field):
        p = df.pivot(index="_bin", columns="SAMPLE", values=field)
        p = p.reindex(index=bin_order, columns=samples)
        return const.cast_field(field, p.to_numpy())

    rd_mat, cov_mat, baf_mat, alpha_mat, beta_mat = (
        _mat(f) for f in ("RD", "COV", "BAF", "ALPHA", "BETA")
    )
    bins = bins.drop(columns=["_bin"])
    return bins, samples, rd_mat, cov_mat, baf_mat, alpha_mat, beta_mat


def _read_wide_bbc_mats(bbc_file: str):
    """Parse a wide-format BBC straight into ``(bins, samples, *_mat)`` (no melt)."""
    wide = pd.read_table(bbc_file, sep="\t")
    fmt = wide["FORMAT"].iloc[0].split(":")
    file_samples = [
        c for c in wide.columns if c not in const.WIDE_BBC_FIXED and c != "FORMAT"
    ]
    if not file_samples:
        raise ValueError(f"no sample columns found in wide BBC {bbc_file}")
    required = set(const.WIDE_BBC_FIELDS)
    if not required.issubset(fmt):
        raise ValueError(
            f"wide BBC {bbc_file} FORMAT {fmt} missing required fields "
            f"{sorted(required - set(fmt))}"
        )
    samples = sorted(file_samples)
    parsed = {}
    for s in samples:
        parts = wide[s].str.split(":", expand=True)
        if parts.shape[1] != len(fmt) or parts.isna().any().any():
            raise ValueError(
                f"wide BBC {bbc_file} sample {s!r} has cells not matching FORMAT {fmt}"
            )
        parts.columns = fmt
        parsed[s] = parts

    bins = wide[["#CHR", "START", "END", "#SNPS", "CLUSTER"]].copy()
    bins["_row"] = np.arange(len(bins))
    bins = sort_df_chr(bins, pos="START").reset_index(drop=True)
    order = bins["_row"].to_numpy()
    bins = bins.drop(columns=["_row"])

    def _mat(field):
        stacked = np.column_stack([parsed[s][field].to_numpy() for s in samples])
        return const.cast_field(field, stacked[order])

    rd_mat, cov_mat, baf_mat, alpha_mat, beta_mat = (
        _mat(f) for f in ("RD", "COV", "BAF", "ALPHA", "BETA")
    )
    return bins, samples, rd_mat, cov_mat, baf_mat, alpha_mat, beta_mat


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
    wide = pd.read_table(bbc_file, sep="\t")
    fmt = wide["FORMAT"].iloc[0].split(":")
    samples = [
        c for c in wide.columns if c not in const.WIDE_BBC_FIXED and c != "FORMAT"
    ]
    if not samples:
        raise ValueError(f"no sample columns found in wide BBC {bbc_file}")
    required = set(const.WIDE_BBC_FIELDS)
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
        df[field] = const.cast_field(field, _stack(field))

    df = sort_df_chr(df, pos="START")
    return df


def read_seg_file(seg_file: str, is_wide_format: bool | None = None):
    """Read a cluster-bins SEG table into per-(cluster, sample) DataFrames.

    Handles both the long layout (one row per cluster x sample) and the wide
    VCF-like layout (one row per cluster; FORMAT key list + one colon-joined
    column per sample). Auto-detects wide via the presence of a ``FORMAT`` column
    when ``is_wide_format`` is None. The seg summary carries no genomic
    coordinates, so no chromosome sorting is applied; clusters and samples are
    returned in sorted order and every field frame is aligned to them.

    Args:
        seg_file: Path to ``bulk.seg`` (or ``labels/bulk<K>.seg``).
        is_wide_format: Force wide (True) / long (False) parsing; None auto-detects.

    Returns:
        Tuple ``(clusters, samples, rdr_df, baf_df, rdr_se_df, baf_se_df, rdr_var_df,
        baf_tau_df, nbins_df, weights, is_balanced, is_filtered)``. ``clusters`` is
        the sorted cluster-id list; ``samples`` the sorted sample list; each
        ``*_df`` is a ``(K, M)`` DataFrame indexed by ``clusters`` with ``samples``
        columns (``nbins_df`` int, the rest float); ``weights`` a ``(K,)``
        length-weight array (``100 * LENGTH / sum(LENGTH)``); ``is_balanced`` /
        ``is_filtered`` ``(K,)`` bool arrays. COV/ALPHA/BETA are not returned
        (unused by compute-cn).
    """
    df = pd.read_table(seg_file, sep="\t")
    if is_wide_format or (is_wide_format is None and "FORMAT" in df.columns):
        return _read_wide_seg_dfs(df)
    return _long_seg_to_dfs(df)


def _long_seg_to_dfs(df):
    """Pivot a long SEG frame into per-(cluster, sample) field DataFrames."""
    clusters = sorted(df["CLUSTER"].unique().tolist())
    samples = sorted(df["SAMPLE"].unique().tolist())

    def _field(field):
        p = df.pivot(index="CLUSTER", columns="SAMPLE", values=field)
        p = p.reindex(index=clusters, columns=samples)
        return const.cast_field(field, p)

    rdr_df, baf_df, rdr_se_df, baf_se_df, rdr_var_df, baf_tau_df, nbins_df = (
        _field(f)
        for f in ("RD", "BAF", "RD-se", "BAF-se", "RD-var", "BAF-tau", "#BINS")
    )

    per_cluster = df.drop_duplicates("CLUSTER").set_index("CLUSTER").reindex(clusters)
    lengths = per_cluster["LENGTH"].to_numpy(dtype=np.float64)
    weights = 100.0 * lengths / lengths.sum()
    is_balanced = per_cluster["is_balanced"].to_numpy(dtype=bool)
    is_filtered = per_cluster["is_filtered"].to_numpy(dtype=bool)
    return (
        clusters,
        samples,
        rdr_df,
        baf_df,
        rdr_se_df,
        baf_se_df,
        rdr_var_df,
        baf_tau_df,
        nbins_df,
        weights,
        is_balanced,
        is_filtered,
    )


def _read_wide_seg_dfs(df):
    """Expand a wide SEG frame to long, then reuse the long parser for parity."""
    fmt = df["FORMAT"].iloc[0].split(":")
    file_samples = [
        c for c in df.columns if c not in const.WIDE_SEG_FIXED and c != "FORMAT"
    ]
    if not file_samples:
        raise ValueError("no sample columns found in wide SEG")
    k, m = len(df), len(file_samples)
    parsed = {}
    for s in file_samples:
        parts = df[s].str.split(":", expand=True)
        if parts.shape[1] != len(fmt) or parts.isna().any().any():
            raise ValueError(
                f"wide SEG sample {s!r} has cells not matching FORMAT {fmt}"
            )
        parts.columns = fmt
        parsed[s] = parts

    long = pd.DataFrame(
        {
            "CLUSTER": np.repeat(df["CLUSTER"].to_numpy(), m),
            "SAMPLE": np.tile(file_samples, k),
            "#BINS": np.repeat(df["#BINS"].to_numpy(), m),
            "#SNPS": np.repeat(df["#SNPS"].to_numpy(), m),
            "LENGTH": np.repeat(df["LENGTH"].to_numpy(), m),
            "is_balanced": np.repeat(df["is_balanced"].to_numpy(), m),
            "is_filtered": np.repeat(df["is_filtered"].to_numpy(), m),
        }
    )
    for f in fmt:
        stacked = np.column_stack(
            [parsed[s][f].to_numpy() for s in file_samples]
        ).ravel()
        long[f] = const.cast_field(f, stacked)
    return _long_seg_to_dfs(long)


# =============================================================================
# File writers
# =============================================================================


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
    fields = [f for f in const.WIDE_BBC_FIELDS if f in field_mats]
    cols["FORMAT"] = ":".join(fields)
    for si, sample in enumerate(tumor_samples):
        cell = const.fmt_field(fields[0], field_mats[fields[0]][:, si])
        for f in fields[1:]:
            cell = np.char.add(
                np.char.add(cell, ":"), const.fmt_field(f, field_mats[f][:, si])
            )
        cols[sample] = cell
    wide = pd.DataFrame(cols)
    wide.to_csv(bbc_gz_path, sep="\t", header=True, index=False, compression="gzip")


def write_seg_file(seg_path, segs_long, tumor_samples, is_wide_format: bool = False):
    """Write a cluster-bins SEG table in either the long or wide layout.

    Args:
        seg_path: Output path (long ``bulk<K>.seg`` or its wide TSV).
        segs_long: Long SEG frame from :func:`mat2segs` with the per-cluster
            fixed columns, per-(cluster, sample) fields, and is_balanced/is_filtered.
        tumor_samples: Ordered tumor sample names = per-sample column order.
        is_wide_format: Write the VCF-like wide layout via :func:`write_wide_seg`.
    """
    if is_wide_format:
        write_wide_seg(seg_path, segs_long, tumor_samples)
    else:
        segs_long.to_csv(seg_path, sep="\t", header=True, index=False)


def write_wide_seg(seg_path, segs_long, tumor_samples):
    """Serialize a SEG table as a single VCF-like wide TSV.

    One row per cluster: cluster-level fixed columns (CLUSTER, #BINS, #SNPS,
    LENGTH, is_balanced, is_filtered), a constant ``FORMAT`` key list, and one
    column per tumor sample holding the per-(cluster, sample) fields colon-joined
    in FORMAT order.
    """
    fixed = segs_long.drop_duplicates("CLUSTER")[const.WIDE_SEG_FIXED].reset_index(
        drop=True
    )
    clusters = fixed["CLUSTER"].tolist()
    cols = {c: fixed[c].to_numpy() for c in const.WIDE_SEG_FIXED}
    cols["FORMAT"] = ":".join(const.WIDE_SEG_FIELDS)
    piv = {
        f: segs_long.pivot(index="CLUSTER", columns="SAMPLE", values=f).reindex(
            index=clusters, columns=tumor_samples
        )
        for f in const.WIDE_SEG_FIELDS
    }
    for sample in tumor_samples:
        cell = const.fmt_field(
            const.WIDE_SEG_FIELDS[0], piv[const.WIDE_SEG_FIELDS[0]][sample].to_numpy()
        )
        for f in const.WIDE_SEG_FIELDS[1:]:
            cell = np.char.add(
                np.char.add(cell, ":"), const.fmt_field(f, piv[f][sample].to_numpy())
            )
        cols[sample] = cell
    pd.DataFrame(cols).to_csv(seg_path, sep="\t", header=True, index=False)


# =============================================================================
# Wide .ucn output: cn_* as fixed bin columns, u_* + fields in per-sample cells
# =============================================================================


def write_ucn_wide(out_path, df, samples, fixed_cols, fmt_fields):
    """Serialize a long .ucn frame (row per bin x sample) as a wide VCF-like TSV.

    ``fixed_cols`` are bin-level (sample-independent) columns kept verbatim;
    ``fmt_fields`` are per-(bin, sample) columns colon-joined per sample in
    FORMAT order.
    """
    df = df.copy()
    df["_key"] = (
        df["#CHR"].astype(str)
        + ":"
        + df["START"].astype(str)
        + ":"
        + df["END"].astype(str)
    )
    base = df.drop_duplicates("_key").reset_index(drop=True)
    keys = base["_key"].tolist()
    cols = {c: base[c].to_numpy() for c in fixed_cols}
    cols["FORMAT"] = ":".join(fmt_fields)
    formatted = {
        f: const.fmt_field(
            f,
            df.pivot(index="_key", columns="SAMPLE", values=f)
            .reindex(index=keys, columns=samples)
            .to_numpy(),
        )
        for f in fmt_fields
    }
    for si, sample in enumerate(samples):
        cell = formatted[fmt_fields[0]][:, si]
        for f in fmt_fields[1:]:
            cell = np.char.add(np.char.add(cell, ":"), formatted[f][:, si])
        cols[sample] = cell
    pd.DataFrame(cols).to_csv(out_path, sep="\t", index=False)


def _expand_wide_ucn(wide, geom):
    """Expand a wide .ucn frame back to the long (row per bin x sample) layout."""
    fmt = wide["FORMAT"].iloc[0].split(":")
    cn_cols = [c for c in wide.columns if c.startswith("cn_")]
    present_geom = [c for c in geom if c in wide.columns]
    fixed = present_geom + cn_cols
    samples = [c for c in wide.columns if c not in fixed and c != "FORMAT"]
    m = len(samples)
    parsed = {}
    for s in samples:
        parts = wide[s].str.split(":", expand=True)
        parts.columns = fmt
        parsed[s] = parts
    out = {c: np.repeat(wide[c].to_numpy(), m) for c in present_geom}
    out["SAMPLE"] = np.tile(samples, len(wide))
    for c in cn_cols:
        out[c] = np.repeat(wide[c].to_numpy(), m)
    for f in fmt:
        stacked = np.column_stack([parsed[s][f].to_numpy() for s in samples]).ravel()
        out[f] = const.cast_field(f, stacked)
    return sort_df_chr(pd.DataFrame(out), pos="START")


def read_bbc_ucn(ucn_file: str, is_wide_format=None):
    """Read a bin-level .ucn table as a long DataFrame (row per bin x sample).

    Accepts either the long layout or the wide VCF-like layout; when
    ``is_wide_format`` is None the layout is auto-detected by a ``FORMAT`` column.
    """
    df = pd.read_table(ucn_file, sep="\t")
    if is_wide_format is None:
        is_wide_format = "FORMAT" in df.columns
    if not is_wide_format:
        return sort_df_chr(df, pos="START")
    return _expand_wide_ucn(df, geom=["#CHR", "START", "END", "#SNPS", "CLUSTER"])


def read_seg_ucn(seg_ucn_file: str, is_wide_format=None):
    """Read a seg.ucn table (long or wide) and derive the clone list.

    Per-clone proportions are per sample (``u_<clone>``), so read them at the call
    site from the relevant sample's rows rather than here.

    Returns:
        (df, clones): the chromosome-sorted long table and the clone-name list
        ("normal", "clone1", ...).
    """
    df = pd.read_table(seg_ucn_file, sep="\t")
    if is_wide_format is None:
        is_wide_format = "FORMAT" in df.columns
    if is_wide_format:
        df = _expand_wide_ucn(df, geom=["#CHR", "START", "END", "CLUSTER"])
    else:
        df = sort_df_chr(df, pos="START")
    n_clones = len([c for c in df.columns if c.startswith("cn_")])
    clones = ["normal"] + [f"clone{c}" for c in range(1, n_clones)]
    return df, clones


def read_seg_ucn_file(seg_ucn_file: str):
    """Backward-compatible wrapper for :func:`read_seg_ucn`."""
    return read_seg_ucn(seg_ucn_file)


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

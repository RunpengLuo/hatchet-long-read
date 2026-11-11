"""Validate a HATCHet ``bb_dir`` and summarise it using only the standard library.

Matrix shapes are read from the ``.npy`` headers inside each ``.npz`` archive, so no
numpy is needed and nothing is loaded into memory beyond the headers. The expected
layout follows ``docs/modules/cluster-bins.md``.
"""

import ast
import csv
import gzip
import statistics
import struct
import zipfile
from pathlib import Path

from hatchet_mcp.config import GENOMES

BB_TSV = "bb.tsv.gz"
SAMPLE_TSV = "sample_ids.tsv"
TUMOR_MATRICES = ("bb.rdr.npz",)
ALL_SAMPLE_MATRICES = ("bb.depth.npz", "bb.Aallele.npz", "bb.Ballele.npz", "bb.Tallele.npz")
REQUIRED_FILES = (BB_TSV, SAMPLE_TSV) + TUMOR_MATRICES + ALL_SAMPLE_MATRICES
BB_COLUMNS = ("#CHR", "START", "END", "region_id", "switchprobs", "#SNPS")
SAMPLE_COLUMNS = ("SAMPLE", "sample_type")


def read_npz_shape(path: Path) -> tuple[int, ...]:
    """Return the shape of the single array stored in a ``.npz`` archive.

    Args:
        path: ``.npz`` file written by ``numpy.savez``.

    Returns:
        Array shape parsed from the ``.npy`` header.

    Raises:
        ValueError: If the archive does not hold exactly one ``.npy`` member.
    """
    with zipfile.ZipFile(path) as archive:
        members = [m for m in archive.namelist() if m.endswith(".npy")]
        if len(members) != 1:
            raise ValueError(f"{path.name}: expected one array, found {len(members)}")
        with archive.open(members[0]) as fh:
            if fh.read(6) != b"\x93NUMPY":
                raise ValueError(f"{path.name}: not a numpy array")
            major = fh.read(2)[0]
            size_fmt, size_len = ("<H", 2) if major == 1 else ("<I", 4)
            header_len = struct.unpack(size_fmt, fh.read(size_len))[0]
            header = ast.literal_eval(fh.read(header_len).decode("latin1"))
    return tuple(header["shape"])


def inspect_bb_dir(
    bb_dir: str,
    genome: str | None = None,
    chrom_sizes: dict[str, dict[str, int]] | None = None,
) -> dict:
    """Check a ``bb_dir`` against HATCHet's input contract and summarise it.

    Args:
        bb_dir: Directory produced by upstream preprocessing.
        genome: Genome key to validate coordinates against; when None every genome in
            ``chrom_sizes`` is tested and the compatible ones are reported.
        chrom_sizes: ``{genome: {chromosome: size}}`` from HATCHet's reference
            directory; when None the coordinate check is skipped with a warning.

    Returns:
        Dict with ``valid``, ``errors``, ``warnings``, ``samples``, ``bins``,
        ``matrices`` and ``genome`` sections.
    """
    root = Path(bb_dir).expanduser().resolve()
    errors, warnings = [], []
    report = {"bb_dir": str(root), "valid": False, "errors": errors, "warnings": warnings}
    if not root.is_dir():
        errors.append(f"not a directory: {root}")
        return report

    missing = [f for f in REQUIRED_FILES if not (root / f).is_file()]
    if missing:
        errors.append(f"missing required files: {', '.join(missing)}")
        return report

    with open(root / SAMPLE_TSV, newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        sample_cols = reader.fieldnames or []
        samples = list(reader)
    absent = [c for c in SAMPLE_COLUMNS if c not in sample_cols]
    if absent:
        errors.append(f"{SAMPLE_TSV} lacks columns: {', '.join(absent)}")
        return report
    bad_types = sorted({s["sample_type"] for s in samples} - {"normal", "tumor"})
    if bad_types:
        errors.append(f"{SAMPLE_TSV}: sample_type must be normal or tumor, got {bad_types}")
    tumors = [s["SAMPLE"] for s in samples if s["sample_type"] == "tumor"]
    normals = [s["SAMPLE"] for s in samples if s["sample_type"] == "normal"]
    if not tumors:
        errors.append("no tumor samples in sample_ids.tsv")
    if not normals:
        warnings.append("no matched-normal sample; RDR must already be normalised upstream")
    report["samples"] = {
        "n_samples": len(samples),
        "n_tumor": len(tumors),
        "n_normal": len(normals),
        "has_normal": bool(normals),
        "tumor": tumors,
        "normal": normals,
        "assay_types": sorted({s.get("assay_type") or "unspecified" for s in samples}),
    }

    with gzip.open(root / BB_TSV, "rt", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        bb_cols = reader.fieldnames or []
        absent = [c for c in BB_COLUMNS if c not in bb_cols]
        if absent:
            errors.append(f"{BB_TSV} lacks columns: {', '.join(absent)}")
            return report
        n_bins, total_snps, zero_snp_bins = 0, 0, 0
        chrom_order, regions, lengths, chrom_end = [], set(), [], {}
        for row in reader:
            n_bins += 1
            chrom, start, end = row["#CHR"], int(row["START"]), int(row["END"])
            snps = int(float(row["#SNPS"]))
            if chrom not in chrom_end:
                chrom_order.append(chrom)
            chrom_end[chrom] = max(chrom_end.get(chrom, 0), end)
            regions.add(row["region_id"])
            lengths.append(end - start)
            total_snps += snps
            zero_snp_bins += snps == 0
    report["bins"] = {
        "n_bins": n_bins,
        "n_chromosomes": len(chrom_order),
        "chromosomes": chrom_order,
        "n_regions": len(regions),
        "total_snps": total_snps,
        "bins_without_snps": zero_snp_bins,
        "median_bin_length_bp": int(statistics.median(lengths)) if lengths else 0,
    }
    if zero_snp_bins:
        warnings.append(f"{zero_snp_bins} bins carry no heterozygous SNPs (no BAF signal)")

    shapes = {}
    for name in TUMOR_MATRICES + ALL_SAMPLE_MATRICES:
        try:
            shapes[name] = read_npz_shape(root / name)
        except (ValueError, zipfile.BadZipFile, KeyError) as exc:
            errors.append(f"{name}: {exc}")
    report["matrices"] = {k: list(v) for k, v in shapes.items()}
    for name, shape in shapes.items():
        expect_cols = len(tumors) if name in TUMOR_MATRICES else len(samples)
        if len(shape) != 2 or shape[0] != n_bins or shape[1] != expect_cols:
            errors.append(
                f"{name} has shape {tuple(shape)}, expected ({n_bins}, {expect_cols}) "
                f"= (bins, {'tumor samples' if name in TUMOR_MATRICES else 'samples'})"
            )

    if genome is not None and genome not in GENOMES:
        errors.append(f"unknown genome '{genome}'; choose one of {sorted(GENOMES)}")
    if chrom_sizes is None:
        warnings.append("genome coordinates not checked: HATCHet reference files unavailable")
        report["genome"] = {"requested": genome, "compatible": None}
    else:
        fits = {
            key: all(c in sizes and chrom_end[c] <= sizes[c] for c in chrom_end)
            for key, sizes in chrom_sizes.items()
        }
        compatible = sorted(k for k, ok in fits.items() if ok)
        report["genome"] = {"requested": genome, "compatible": compatible}
        if genome in fits and not fits[genome]:
            errors.append(
                f"bins fall outside {genome} chromosome sizes; compatible builds: "
                f"{compatible or 'none'}"
            )
        elif genome is None and not compatible:
            errors.append("bin coordinates fit none of HATCHet's reference genome builds")

    report["valid"] = not errors
    return report

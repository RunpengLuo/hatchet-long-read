"""Summarise stage outputs and list the figures to show the user.

HATCHet's PDFs are the figures the user sees: every one is listed by absolute path with
its page count. For the agent, the stage scripts render each PDF to PNG previews in the
scratch directory, mirroring the PDF's path under ``<tmp>/previews/<run_id>/`` so the run
directory holds only HATCHet's outputs. Only headline figures (model selection, the
selected result, one QC overview) carry previews into the tool result. The selected
number of clusters and the selected ploidy are read from files HATCHet itself writes for
its chosen model, never re-derived.
"""

import csv
import re
from dataclasses import dataclass
from pathlib import Path

MAX_PREVIEW_PAGES = 4
SEG_DISPLAY = (
    "CLUSTER",
    "SAMPLE",
    "#BINS",
    "#SNPS",
    "LENGTH",
    "RD",
    "BAF",
    "is_balanced",
    "is_filtered",
)
_PAGE_OBJECT = re.compile(rb"/Type\s*/Page(?![a-zA-Z])")


def _read_tsv(path: Path) -> list[dict]:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def _num(value: str):
    try:
        f = float(value)
    except (TypeError, ValueError):
        return value
    return int(f) if f.is_integer() and "." not in str(value) else round(f, 4)


def pdf_page_count(pdf: Path) -> int:
    """Number of pages in a PDF, counted from its page objects."""
    return len(_PAGE_OBJECT.findall(pdf.read_bytes()))


def _page_number(png: Path) -> int:
    try:
        return int(png.stem.rsplit("-", 1)[1])
    except (IndexError, ValueError):
        return 0


@dataclass(frozen=True)
class Previews:
    """Locates the PNG previews the stage scripts rendered from a run's PDFs.

    Attributes:
        run_dir: Run directory the PDFs live in.
        root: ``<tmp>/previews/<run_id>``, mirroring ``run_dir``.
    """

    run_dir: Path
    root: Path

    def pages(self, pdf: Path) -> list[Path]:
        """Preview PNGs of ``pdf`` in page order; empty if none were rendered."""
        try:
            rel = pdf.relative_to(self.run_dir)
        except ValueError:
            return []
        return sorted((self.root / rel.parent).glob(f"{rel.stem}-*.png"), key=_page_number)


def _figure(
    previews: Previews,
    description: str,
    pdf: Path | None,
    inline: bool,
    order=None,
    max_pages=MAX_PREVIEW_PAGES,
):
    """Describe one PDF, attaching previews for the agent when it is a headline figure.

    Args:
        previews: Preview locator for the run.
        description: What the figure shows.
        pdf: The PDF HATCHet wrote, or None.
        inline: Attach preview images to the tool result.
        order: Page indices to preview, in display order; None keeps page order.
        max_pages: Most pages to preview.

    Returns:
        Figure dict, or None when the PDF does not exist.
    """
    if pdf is None or not pdf.exists():
        return None
    entry = {
        "description": description,
        "pdf": str(pdf),
        "pages": pdf_page_count(pdf),
        "inline": inline,
    }
    pages = previews.pages(pdf)
    if inline and pages:
        chosen = [pages[i] for i in order] if order and len(pages) == entry["pages"] else pages
        entry["preview_png"] = [str(page) for page in chosen[:max_pages]]
    return entry


def _other_figures(roots: list[Path], listed: list[dict]) -> list[dict]:
    """Every other PDF under ``roots``, listed by path for on-demand viewing."""
    seen = {f["pdf"] for f in listed}
    extra = sorted({pdf for root in roots if root.exists() for pdf in root.rglob("*.pdf")})
    return [
        {"description": pdf.stem, "pdf": str(pdf), "pages": pdf_page_count(pdf), "inline": False}
        for pdf in extra
        if str(pdf) not in seen
    ]


def selected_k(bbc_dir: Path) -> int | None:
    """K chosen by cluster-bins model selection, from its ``bulk.K<k>.pdf`` output."""
    for pdf in bbc_dir.glob("bulk.K*.pdf"):
        match = re.fullmatch(r"bulk\.K(\d+)", pdf.stem)
        if match:
            return int(match.group(1))
    return None


def cluster_bins_results(run_dir: str, preview_root: str) -> tuple[dict, list[dict]]:
    """Summarise the cluster-bins stage of a run.

    Args:
        run_dir: Run directory; cluster-bins output is its ``cluster-bins`` subdirectory.
        preview_root: ``<tmp>/previews/<run_id>``.

    Returns:
        Summary dict and a list of figures: headline figures first (model selection,
        selected clustering, QC), then every other PDF.
    """
    previews = Previews(Path(run_dir), Path(preview_root))
    root = Path(run_dir) / "cluster-bins"
    summary = {"bbc_dir": str(root), "selected_k": selected_k(root)}

    scores_path = root / "model_scores.tsv"
    if scores_path.exists():
        rows = _read_tsv(scores_path)
        metrics = [m for m in ("bic", "icl") if rows and m in rows[0]]
        best = {}
        for row in rows:
            k = int(row["K"])
            entry = best.setdefault(k, {"K": k, "ll": float("-inf")})
            entry["ll"] = max(entry["ll"], float(row["ll"]))
            for m in metrics:
                entry[m] = min(entry.get(m, float("inf")), float(row[m]))
        summary["model_scores_best_per_k"] = [
            {key: round(val, 3) if isinstance(val, float) else val for key, val in best[k].items()}
            for k in sorted(best)
        ]

    seg_path = root / "bulk.seg"
    if seg_path.exists():
        seg = _read_tsv(seg_path)
        summary["n_clusters_after_filtering"] = len({r["CLUSTER"] for r in seg})
        summary["clusters"] = [{c: _num(r[c]) for c in SEG_DISPLAY if c in r} for r in seg]

    bbc_path = root / "bulk.bbc"
    if bbc_path.exists():
        with open(bbc_path, newline="") as fh:
            reader = csv.DictReader(fh, delimiter="\t")
            summary["n_bins_clustered"] = len({(r["#CHR"], r["START"]) for r in reader})

    plots = root / "plots"
    k = summary["selected_k"]
    headline = [
        _figure(
            previews, "Model selection across K (cluster-bins)", plots / "model_scores.pdf", True
        ),
        _figure(previews, f"Model-selected clustering, K={k}", root / f"bulk.K{k}.pdf", True)
        if k is not None
        else None,
        _figure(previews, "QC: EM convergence per K", plots / "elbo_traces.pdf", True, max_pages=1),
    ]
    figures = [f for f in headline if f]
    return summary, figures + _other_figures([root], figures)


def _clone_columns(header: list[str]) -> list[str]:
    return sorted(
        (c for c in header if c.startswith("cn_clone")),
        key=lambda c: int(c.removeprefix("cn_clone") or 0),
    )


def compute_cn_results(run_dir: str, preview_root: str) -> tuple[dict, list[dict]]:
    """Summarise the compute-cn and plot-cn stages of a run.

    Args:
        run_dir: Run directory holding ``compute-cn`` and ``plot-cn``.
        preview_root: ``<tmp>/previews/<run_id>``.

    Returns:
        Summary dict and a list of figures: headline figures first (model selection, the
        1D and 2D profiles of the selected solution, QC), then every other PDF.
    """
    previews = Previews(Path(run_dir), Path(preview_root))
    root, plots_out = Path(run_dir) / "compute-cn", Path(run_dir) / "plot-cn"
    summary = {"result_dir": str(root), "plot_dir": str(plots_out)}

    ploidy_file = root / "selected_ploidy.txt"
    summary["selected_ploidy"] = ploidy_file.read_text().strip() if ploidy_file.exists() else None
    summary["wgd"] = {"diploid": False, "tetraploid": True}.get(summary["selected_ploidy"])

    summary_path = root / "summary.tsv"
    if summary_path.exists():
        summary["model_selection"] = [
            {k: _num(v) for k, v in row.items()} for row in _read_tsv(summary_path)
        ]

    seg_path = root / "best.seg.ucn"
    if seg_path.exists():
        seg = _read_tsv(seg_path)
        header = list(seg[0].keys()) if seg else []
        clones = _clone_columns(header)
        summary["n_tumor_clones"] = len(clones)
        summary["hatchet_n_clones"] = len(clones) + 1
        summary["clone_count_note"] = (
            "HATCHet's n counts the normal clone, so figure titles showing n="
            f"{len(clones) + 1} mean {len(clones)} tumor clone(s)."
        )
        per_sample = {}
        for row in seg:
            if row["SAMPLE"] in per_sample:
                continue
            u_normal = float(row["u_normal"])
            per_sample[row["SAMPLE"]] = {
                "purity": round(1 - u_normal, 4),
                "u_normal": round(u_normal, 4),
                **{
                    f"u_{c.removeprefix('cn_')}": round(float(row[f"u_{c.removeprefix('cn_')}"]), 4)
                    for c in clones
                },
            }
        summary["samples"] = per_sample
        first = next(iter(per_sample), None)
        if first is not None:
            rows = [r for r in seg if r["SAMPLE"] == first]
            lengths = [int(r["END"]) - int(r["START"]) for r in rows]
            total = sum(lengths) or 1
            summary["clone_ploidy_length_weighted"] = {
                c.removeprefix("cn_"): round(
                    sum(n * sum(int(x) for x in r[c].split("|")) for n, r in zip(lengths, rows))
                    / total,
                    3,
                )
                for c in clones
            }

    gamma_path = root / "gammas.tsv"
    if gamma_path.exists():
        summary["gammas_tsv"] = [
            line.split("\t") for line in gamma_path.read_text().splitlines() if line
        ]

    selection_pdf = root / "plots" / "model_selection.pdf"
    order = None
    n_ploidy = len({row["ploidy"] for row in summary.get("model_selection", [])})
    n_pages = pdf_page_count(selection_pdf) if selection_pdf.exists() else 0
    if n_ploidy and n_pages == n_ploidy + 1:
        order = [n_pages - 1, *range(n_ploidy)]
        summary["model_selection_figure_note"] = (
            f"model_selection.pdf page {n_pages} is the decision across ploidy and clone "
            f"count; pages 1-{n_ploidy} are per-ploidy regularisation paths."
        )

    def plot_cn(suffix):
        hits = sorted(plots_out.glob(f"*{suffix}"))
        return hits[0] if hits else None

    ploidy = summary["selected_ploidy"]
    scaling = root / "plots" / f"scaling_2d.{ploidy}.pdf"
    headline = [
        _figure(previews, "Model selection (compute-cn)", selection_pdf, True, order=order),
        _figure(previews, "1D copy-number profile along the genome", plot_cn(".1D.pdf"), True),
        _figure(
            previews, "1D allele-specific fractional copy number", plot_cn(".1D.FCN_AB.pdf"), True
        ),
        _figure(previews, "2D mhBAF vs fractional copy number", plot_cn(".2D.pdf"), True),
        _figure(previews, f"QC: RDR scaling ({ploidy})", scaling, True) if ploidy else None,
    ]
    figures = [f for f in headline if f]
    return summary, figures + _other_figures([root, plots_out], figures)


def check_results(log: str) -> dict:
    """Parse ``hatchet check`` assertion lines out of a job log.

    Args:
        log: Text of the job log.

    Returns:
        Dict with ``passed``, ``n_pass``, ``n_fail`` and the assertion lines.
    """
    lines = [ln.strip() for ln in log.splitlines() if "[PASS]" in ln or "[FAIL]" in ln]
    return {
        "passed": "hatchet check PASSED" in log,
        "n_pass": sum("[PASS]" in ln for ln in lines),
        "n_fail": sum("[FAIL]" in ln for ln in lines),
        "assertions": lines,
    }

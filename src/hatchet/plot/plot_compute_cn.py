"""Plotting for the compute-cn command.

Solution-pool CNP panels, the scaling-factor 2D diagnostic, and clone-tree
rendering — everything the compute-cn pipeline draws.
"""

from __future__ import annotations

import os
import re
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from cnplot import annotate_landmarks, plot_cnv_profile, plot_scatter_2d, set_palette
from hatchet.utils import sort_df_chr
from hatchet import const
from hatchet.plot import plot_cn as _plot_cn
from hatchet.plot.plot_utils import build_genome_axis, use_editable_fonts


def _clones_from_cn(df):
    """Ordered clone names ("normal", "clone1", ...) from a df's cn_ columns."""
    n = len([c for c in df.columns if c.startswith("cn_")])
    return ["normal"] + [f"clone{i}" for i in range(1, n)]


def _fmt_pool_label(tag):
    """Convert 'pool_p0.05_s1' to 'p=0.05,s=1'."""
    m = re.match(r"pool_p([^_]+)_s(\d+)", tag)
    if m:
        return f"p={m.group(1)},s={m.group(2)}"
    return tag


def _fmt_prop(v):
    """Round proportion to 2 decimals as percent; literal '0' if zero."""
    pct = round(v * 100, 2)
    return "0" if pct == 0 else f"{pct}%"


def plot_pool_cnp(
    pool_instances,
    genome_size,
    region_bed,
    out_dir,
    sel_df=None,
    segs=None,
    title=None,
    width=20,
    height=1,
    dpi=150,
    out_name=const.POOL_PDF,
):
    """Plot a single multi-row CNV profile panel of Pareto solutions into out_dir.

    Args:
        pool_instances: {sol_id: {"imf_obj", "reg_obj", "cA", "cB", "u", ...}}.
        region_bed: Path to the whitelist region BED file.
        out_dir: Output directory for pool plots.
        sel_df: Selection DataFrame from model_select_elbow_from_regularization.
        segs: {sol_id: seg_df} pre-computed segmentation DataFrames.
        title: Optional figure title.
        width: Figure width in inches.
        height: Height in inches per profile row.
        dpi: Output resolution.
        genome_size: Chromosome-sizes path for the genome axis.
    """
    os.makedirs(out_dir, exist_ok=True)

    use_editable_fonts()
    # Restrict the axis to chromosomes present in the solutions' segments.
    keep = set()
    for sdf in (segs or {}).values():
        keep.update(sdf["#CHR"].unique())
    genome_axis = build_genome_axis(region_bed, genome_size, keep_chroms=keep or None)

    selected_ids = set()
    if sel_df is not None:
        selected_ids = set(
            sel_df.loc[sel_df["selected"] == "*", "instance_id"].tolist()
        )

    pareto_ids = []
    if sel_df is not None:
        pareto_ids = sel_df.loc[sel_df["is_pareto"], "instance_id"].tolist()
    else:
        pareto_ids = sorted(pool_instances.keys())

    entries = []
    for sol_id in pareto_ids:
        sol = pool_instances[sol_id]
        is_selected = sol_id in selected_ids
        entries.append((sol_id, segs[sol_id], sol["imf_obj"], is_selected))
    entries.sort(key=lambda x: x[2])

    if not entries:
        logging.warning(f"plot_pool_cnp: no solutions to plot, skipping {out_dir}")
        return

    nrows = len(entries)
    first_seg_df = entries[0][1]
    n_clones = len([c for c in first_seg_df.columns if c.startswith("cn_")])
    n_samples = first_seg_df["SAMPLE"].nunique()
    row_h = height * max(1, n_clones - 1) + 0.2 * max(0, n_samples - 1)
    fig, axes = plt.subplots(
        nrows=nrows + 1,
        ncols=1,
        figsize=(width, row_h * nrows),
        gridspec_kw={"height_ratios": [row_h] * nrows + [2 * height]},
    )
    fig.subplots_adjust(hspace=0.6 + 0.1 * max(0, n_samples - 1))
    main_axes = axes[:-1] if nrows > 1 else [axes[0]]
    ax_leg = axes[-1]

    for i, (label, seg_df, obj, is_selected) in enumerate(entries):
        seg_info_all = sort_df_chr(seg_df.copy(), pos="START")
        clones = _clones_from_cn(seg_info_all)
        samples = seg_info_all["SAMPLE"].unique().tolist()
        seg_info = seg_info_all.loc[
            seg_info_all["SAMPLE"] == samples[0], :
        ].reset_index(drop=True)

        plot_cnv_profile(
            main_axes[i],
            seg_info,
            genome_axis,
            ax_leg=(ax_leg if i == nrows - 1 else None),
            plot_chrname=True,
            show_prop=False,
        )

        short_label = _fmt_pool_label(str(label))
        if is_selected:
            short_label += " *"
        prop_lines = []
        for sid in samples:
            sp = seg_info_all.loc[seg_info_all["SAMPLE"] == sid, :].reset_index(
                drop=True
            )
            cps = sp[[f"u_{c}" for c in clones]].iloc[0].tolist()
            prop_lines.append(f"{sid}:" + "|".join(_fmt_prop(c) for c in cps))
        ylabel = f"{short_label}\nimf {round(obj, 2)}\n" + "\n".join(prop_lines)
        color = "red" if is_selected else "black"
        main_axes[i].set_ylabel(
            ylabel, rotation=0, ha="right", va="center", color=color
        )

    if title:
        main_axes[0].set_title(title)
    out_file = os.path.join(out_dir, out_name)
    plt.savefig(out_file, dpi=dpi, bbox_inches="tight")
    plt.close()
    logging.info(f"pool CNP panel saved to {out_file}")


def plot_scaling_2d(
    samples: list,
    clusters: list,
    bins: pd.DataFrame,
    rd_mat,
    baf_mat,
    seg_rdr: pd.DataFrame,
    seg_baf: pd.DataFrame,
    scaling: dict,
    plot_dir: str,
    markersize: float = 3.0,
    markersize_centroid: float = 14,
    dpi: int = 300,
    transparent: bool = False,
    maxlim_rdr: int = 10,
):
    """2D RDR-vs-BAF scatter anchoring the scaling inference from get_scaling_factor.

    One PDF per WGD mode (scaling_2d.diploid.pdf for noWGD, scaling_2d.tetraploid.pdf
    for WGD, written only when a WGD scaling was inferred), with one page per sample.
    Bins are colored per cluster via cnplot's plot_scatter_2d; anchor clusters are
    circled and annotated with their inferred (a, b) clonal states via
    annotate_landmarks. Cluster centroids come from the per-(cluster, sample) seg
    RD/BAF frames, indexed by cluster label.
    """
    use_editable_fonts()

    with plt.rc_context():
        palette = set_palette(num_colors=len(clusters))
    pal = {str(c): palette[i] for i, c in enumerate(clusters)}

    panels = [("diploid", "noWGD", scaling["diploid"])]
    if scaling.get("tetraploid") is not None:
        panels.append(("tetraploid", "WGD", scaling["tetraploid"]))

    cluster_set = set(clusters)
    cluster_str = bins["CLUSTER"].astype(str).to_numpy()

    for ploidy, label, info in panels:
        out_file = const.SCALING_2D_PDF(plot_dir, ploidy)
        pdf = PdfPages(out_file)
        for si, sample in enumerate(samples):
            obs = pd.DataFrame({"BAF": baf_mat[:, si], "RD": rd_mat[:, si]})
            obs["CLUSTER"] = cluster_str

            lim_baf = (0, 1) if obs["BAF"].max() > 0.5 else (0, 0.55)
            lim_rdr = (0, min(max(2, int(np.ceil(obs["RD"].max()))), maxlim_rdr))

            landmarks = []
            for c, (a, b) in info["clonal"].items():
                if c not in cluster_set:
                    continue
                landmarks.append(
                    {
                        "x": seg_baf.loc[c, sample],
                        "y": seg_rdr.loc[c, sample],
                        "label": f"({a},{b})",
                        "clonal": True,
                    }
                )

            p = (info.get("purities") or {}).get(sample)
            ptxt = f"  purity={p:.3f}" if p is not None else ""
            grid = plot_scatter_2d(
                obs,
                xcol="BAF",
                ycol="RD",
                hue="CLUSTER",
                palette=pal,
                xlim=lim_baf,
                ylim=lim_rdr,
                xlabel="BAF",
                ylabel="RDR",
                title=f"sample={sample}  {label}{ptxt}",
                refline_x=0.5,
                markersize=markersize,
            )
            annotate_landmarks(grid.ax_joint, landmarks, markersize=markersize_centroid)
            pdf.savefig(
                grid.figure, dpi=dpi, bbox_inches="tight", transparent=transparent
            )
            plt.close(grid.figure)
        pdf.close()
        logging.info(f"scaling 2D scatter saved to {out_file}")


def run_plot_cn(args, bbc, seg, gamma_file, plot_dir, ploidy, name=None):
    """Auto-run plot-cn on a compute-cn solution; styling falls through to hatchet.yaml."""
    if not os.path.exists(bbc) or not os.path.exists(seg):
        return
    _plot_cn.run(
        {
            "bbc": bbc,
            "seg": seg,
            "genome_size": args["genome_size"],
            "region_bed": args["region_bed"],
            "gamma_file": gamma_file,
            "solfile": None,
            "patient_id": name,
            "plot_dir": plot_dir,
            "ploidy": ploidy,
        }
    )


def plot_pareto_curve(summary_df, plot_dir, reg_term, elbow_fig=None):
    """Plot REG vs IMF Pareto curves + elbow/BIC page as a multi-page PDF."""
    outfile = const.MODEL_SELECTION_PDF(plot_dir)
    reg_col = reg_term if reg_term in summary_df.columns else "REG"
    ploidies = sorted(summary_df["ploidy"].unique())
    cmap = plt.get_cmap("tab10")

    n_pages = 0
    with PdfPages(outfile) as pdf:
        # One page per ploidy; overlay all n-clone solutions, each n a distinct color.
        for ploidy in ploidies:
            pdf_grp = summary_df[summary_df["ploidy"] == ploidy]
            ns = sorted(pdf_grp["n_clones"].unique())
            fig, ax = plt.subplots(figsize=(7, 5))

            # Non-pareto across all n: shared light-gray backdrop
            non_pareto = pdf_grp[~pdf_grp["is_pareto"]]
            if len(non_pareto) > 0:
                ax.scatter(
                    non_pareto[reg_col],
                    non_pareto["IMF"],
                    c="0.8",
                    s=15,
                    zorder=2,
                    alpha=0.4,
                    linewidths=0,
                )

            for ni, n_clones in enumerate(ns):
                grp = pdf_grp[pdf_grp["n_clones"] == n_clones]
                color = cmap(ni % 10)
                pareto = grp[grp["is_pareto"]].sort_values(reg_col)
                if len(pareto) > 0:
                    ax.plot(
                        pareto[reg_col],
                        pareto["IMF"],
                        "-o",
                        color=color,
                        markersize=5,
                        linewidth=1.3,
                        zorder=4,
                        label=f"n={n_clones}",
                    )
                sel = grp[grp["selected"] == "*"]
                if len(sel) > 0:
                    ax.scatter(
                        sel[reg_col],
                        sel["IMF"],
                        facecolors=color,
                        marker="*",
                        s=250,
                        zorder=5,
                        edgecolors="black",
                        linewidths=1,
                    )

            ax.set_xlabel(reg_col, fontsize=11)
            ax.set_ylabel("IMF", fontsize=11)
            ax.set_title(
                f"{ploidy} ({len(pdf_grp)} solutions)",
                fontsize=13,
                fontweight="bold",
            )
            ax.legend(fontsize=9, title="clones")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
            n_pages += 1

        # Append elbow/BIC figure as last page
        if elbow_fig is not None:
            pdf.savefig(elbow_fig)
            plt.close(elbow_fig)
            n_pages += 1

    logging.info(f"wrote {outfile} ({n_pages} pages)")

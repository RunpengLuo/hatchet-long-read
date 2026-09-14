"""HATCHet input/output constants: filenames, subdirectories, and table formats.

Producers and consumers of a file must import the SAME name from here, so a
rename cannot silently break a cross-stage contract. Bare leaf names used as
``out_name`` are module constants; fully-resolved paths and names that embed run
parameters (output dir, ploidy, clone count, K, sample) are helper functions.
The BBC/SEG/.ucn column schema and per-field dtype/cell-formatting helpers also
live here, so readers and writers share one definition.

Notes:
    Purely user-named outputs (e.g. plot out_prefix) are not centralized here.
    The legacy ``evaluate_pool_solutions`` reader references filenames with no
    current producer and is intentionally not represented here.
"""

import os

import numpy as np


# =============================================================================
# Table column schema and per-field dtypes (BBC / SEG / .ucn)
# =============================================================================
# wide-format BBC per-sample FORMAT keys in on-disk order.
WIDE_BBC_FIELDS = ["RD", "COV", "BAF", "ALPHA", "BETA"]
# Fixed (non-sample, non-FORMAT) columns of a wide-format BBC file.
WIDE_BBC_FIXED = ["#CHR", "START", "END", "#SNPS", "CLUSTER", "PHASE", "PHASE_POSTS"]
# Per-(cluster, sample) SEG fields carried as wide FORMAT cells, in write order.
WIDE_SEG_FIELDS = [
    "ALPHA",
    "BETA",
    "COV",
    "BAF",
    "BAF-se",
    "BAF-tau",
    "RD",
    "RD-se",
    "RD-var",
]
# Cluster-level SEG columns (constant across samples).
WIDE_SEG_FIXED = ["CLUSTER", "#BINS", "#SNPS", "LENGTH", "is_balanced", "is_filtered"]
# Column -> dtype for every BBC and SEG field (FORMAT cells and numeric fixed
# columns), consulted on read and to pick int vs float cell formatting on write.
FORMAT_DTYPE = {
    "RD": "float64",
    "COV": "float64",
    "BAF": "float64",
    "ALPHA": "int64",
    "BETA": "int64",
    "RD-se": "float64",
    "RD-var": "float64",
    "BAF-se": "float64",
    "BAF-tau": "float64",
    "#BINS": "int64",
    "#SNPS": "int64",
    "LENGTH": "float64",
}


def _field_dtype(field):
    """Storage dtype for a BBC/SEG/.ucn field; ``u_<clone>`` proportions are float."""
    return "float64" if field.startswith("u_") else FORMAT_DTYPE[field]


def cast_field(field, arr):
    """Cast a parsed field (numpy array or pandas frame) to its storage dtype."""
    return arr.astype(_field_dtype(field))


def fmt_field(field, v):
    """Format a field vector into VCF cell strings (``%d`` int, ``%.6g`` float)."""
    v = np.asarray(v)
    if _field_dtype(field).startswith("int"):
        return np.char.mod("%d", v.astype(np.int64))
    return np.char.mod("%.6g", v.astype(np.float64))


# =============================================================================
# Subdirectories
# =============================================================================
LABELS_DIR = lambda out_dir: os.path.join(
    out_dir, "labels"
)  # cluster-bins per-K BBC/SEG
PLOTS_DIR = lambda out_dir: os.path.join(
    out_dir, "plots"
)  # cluster-bins + compute-cn plots
TRACES_DIR = "traces"  # cluster-bins EM traces
INIT_DIAG_DIR = "init_diag"  # cluster-bins init diagnostics
SOLS_DIR = "sols"  # compute-cn per-(ploidy, n) solutions


# =============================================================================
# cluster-bins inputs (bb_dir)
# =============================================================================
BB_TSV_GZ = "bb.tsv.gz"
SAMPLE_IDS = "sample_ids.tsv"
BB_RDR_NPZ = "bb.rdr.npz"
BB_DEPTH_NPZ = "bb.depth.npz"
BB_A_ALLELE_NPZ = "bb.Aallele.npz"
BB_B_ALLELE_NPZ = "bb.Ballele.npz"
BB_T_ALLELE_NPZ = "bb.Tallele.npz"


# =============================================================================
# cluster-bins outputs (bbc_dir)
# =============================================================================
# Top-level BBC/SEG: full path resolved here; BBC extension picked by wide_format
# (wide_format is unused for SEG, kept for a uniform call signature).
BULK_BBC = lambda out_dir, wide_format: os.path.join(
    out_dir, "bulk.bbc.tsv.gz" if wide_format else "bulk.bbc"
)
BULK_SEG = lambda out_dir, wide_format: os.path.join(out_dir, "bulk.seg")
BB_PHASED_TSV_GZ = "bb.phased.tsv.gz"
MODEL_SCORES_TSV = "model_scores.tsv"
MODEL_SCORES_PDF = "model_scores.pdf"
ELBO_TRACES_PDF = "elbo_traces.pdf"
HMM_INIT_PDF = "hmm_init.pdf"


# Per-K BBC/SEG under labels/: full path resolved here; BBC extension picked by
# wide_format (unused for SEG, kept for a uniform call signature).
BULK_BBC_k = lambda out_dir, wide_format, k: os.path.join(
    LABELS_DIR(out_dir), f"bulk{k}.bbc.tsv.gz" if wide_format else f"bulk{k}.bbc"
)
BULK_SEG_k = lambda out_dir, wide_format, k: os.path.join(
    LABELS_DIR(out_dir), f"bulk{k}.seg"
)
BULK_K_PHASED = lambda out_dir, k: os.path.join(
    LABELS_DIR(out_dir), f"bulk{k}.bb.phased.tsv.gz"
)
K_EM_TRACE = lambda out_dir, k: os.path.join(out_dir, TRACES_DIR, f"K{k}.em_trace.npz")
K_PLOT = lambda out_dir, k: os.path.join(PLOTS_DIR(out_dir), f"K{k}.pdf")
BULK_K_PLOT = lambda out_dir, k: os.path.join(out_dir, f"bulk.K{k}.pdf")
INIT_PDF = lambda plot_dir, name: os.path.join(plot_dir, f"{name}_init.pdf")


# =============================================================================
# compute-cn outputs (result_dir)
# =============================================================================
# Fully-resolved paths under the compute-cn result dir (or its plots/ or sols/).
GAMMA_FILE = lambda out_dir: os.path.join(out_dir, "gammas.tsv")
SCALING_2D_PDF = lambda plot_dir, ploidy: os.path.join(
    plot_dir, f"scaling_2d.{ploidy}.pdf"
)
SUMMARY_TSV = lambda out_dir: os.path.join(out_dir, "summary.tsv")
BEST_BBC_UCN = lambda out_dir: os.path.join(out_dir, "best.bbc.ucn")
BEST_SEG_UCN = lambda out_dir: os.path.join(out_dir, "best.seg.ucn")
MODEL_SELECTION_PDF = lambda plot_dir: os.path.join(plot_dir, "model_selection.pdf")
POOL_PDF = "pool.pdf"  # out_name (joined with its plot dir by the caller)
# sols/
OBJECTIVES_TSV = lambda sols_dir: os.path.join(sols_dir, "objectives.tsv")
U0_SEEDS_TSV = lambda sol_dir: os.path.join(sol_dir, "u0_seeds.tsv")


SOLVER_INPUT = lambda out_dir, ploidy: os.path.join(
    out_dir, SOLS_DIR, f"solver_input.{ploidy}.tsv"
)
RESULTS_BBC_UCN = lambda out_dir, ploidy, n: os.path.join(
    out_dir, f"results.{ploidy}.n{n}.bbc.ucn.tsv"
)
RESULTS_SEG_UCN = lambda out_dir, ploidy, n: os.path.join(
    out_dir, f"results.{ploidy}.n{n}.seg.ucn.tsv"
)
CHOSEN_BBC_UCN = lambda out_dir, ploidy: os.path.join(
    out_dir, f"chosen.{ploidy}.bbc.ucn"
)
CHOSEN_SEG_UCN = lambda out_dir, ploidy: os.path.join(
    out_dir, f"chosen.{ploidy}.seg.ucn"
)
# parent_dir is the sols/ or plots/ dir this leaf subdir lives under.
PLOIDY_N_SUBDIR = lambda parent_dir, ploidy, n: os.path.join(
    parent_dir, f"{ploidy}_n{n}"
)
SOLUTION_TSV = lambda sol_dir, solve_mode, sol_id: os.path.join(
    sol_dir, f"{solve_mode}_{sol_id}.tsv"
)
# Per-(ploidy, n) pool CNP panel; out_name only (joined with its plot dir by the caller).
POOL_CNP_PDF = lambda pid, ploidy, n: f"{pid}.pool_{ploidy}_n{n}.pdf"


# =============================================================================
# evaluate outputs (out_dir)
# =============================================================================
SOMATIC_SNVS_TSV = "somatic_snvs.tsv"
POOL_EVAL_TSV = "pool_eval.tsv"
EVAL_SUMMARY_TSV = "eval_summary.tsv"


VAF_1D_PDF = lambda out_dir, sample: os.path.join(out_dir, f"{sample}.vaf_1d.pdf")


# =============================================================================
# logs
# =============================================================================
COMMAND_LOG = lambda out_dir, command: os.path.join(out_dir, f"{command}.log")

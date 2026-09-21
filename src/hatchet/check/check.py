"""End-to-end self-check of the HATCHet pipeline.

Simulates a small dataset from a fixed seed (see :mod:`hatchet.check.simulate`),
runs cluster-bins -> compute-cn -> plot-cn on it, and verifies that each stage
wrote its expected outputs and recovered the simulated copy-number profile.

Nothing is shipped on disk and nothing is left behind: the input is regenerated
per run under the working directory, which defaults to a system temporary
directory that is removed once every assertion passes.
"""

import logging
import os
import shutil
import tempfile
import traceback

import matplotlib
import numpy as np
import pandas as pd

from hatchet import const
from hatchet.hatchet_parser import solver_available
from hatchet.check.simulate import simulate_bb_dir
from hatchet.utils import log_step_start, normalize_args, setup_logging

# NB: smallest stage settings that still exercise every code path
CLUSTER_BINS_PARAMS = {
    "maxK": 5,
    "restarts": 3,
    "top_restarts": 2,
    "n_local_trials": 2,
    "niters": 5,
    "tau_iters": 1,
    "force": True,
}
COMPUTE_CN_PARAMS = {
    "timelimit": 60,
    "maxClone": 2,
    "diploid": True,
    "force": True,
    "reg_steps": 3,
    "diploidcmax": 6,
    "cd_njobs": 1,
    "zero_cn_thres": 0.005,
}

# NB: assertion thresholds
MIN_CLUSTERS = 3
MIN_CLUSTER_RDR_RANGE = 0.3
PURITY_TOL = 0.15
GAMMA_TOL = 0.5
REGION_CN_FRAC = 0.5


def resolve_solver(solver):
    """Return the solver to use, auto-detecting gurobi then cbc when unset."""
    if solver is not None:
        if not solver_available(solver):
            raise RuntimeError(f"requested solver '{solver}' is not available")
        return solver
    for candidate in ("gurobi", "cbc"):
        if solver_available(candidate):
            return candidate
    raise RuntimeError(
        "no MILP solver found; install gurobi or cbc (conda install -c conda-forge coincbc)"
    )


def _major_minor(a, b):
    """Order a CN pair as (major, minor); the A/B labelling is not identifiable."""
    return (max(int(a), int(b)), min(int(a), int(b)))


def _parse_cn(cn_str):
    """Parse an ``a|b`` CN cell into an order-independent (major, minor) pair."""
    return _major_minor(*str(cn_str).split("|"))


def run_cluster_bins_stage(bb_dir, bbc_dir, genome_size, region_bed, verbosity):
    from hatchet.cluster_bins.cluster_bins import run as run_cluster_bins

    run_cluster_bins(
        {
            **CLUSTER_BINS_PARAMS,
            "bb_dir": bb_dir,
            "bbc_dir": bbc_dir,
            "genome_size": genome_size,
            "region_bed": region_bed,
            "verbosity": verbosity,
        }
    )


def run_compute_cn_stage(
    bbc_file, seg_file, result_dir, genome_size, region_bed, solver, verbosity
):
    from hatchet.compute_cn.compute_cn import run as run_compute_cn

    run_compute_cn(
        {
            **COMPUTE_CN_PARAMS,
            "bbc": bbc_file,
            "seg": seg_file,
            "result_dir": result_dir,
            "genome_size": genome_size,
            "region_bed": region_bed,
            "solver": solver,
            "verbosity": verbosity,
        }
    )


def run_plot_cn_stage(result_dir, plot_dir, genome_size, region_bed, verbosity):
    from hatchet.plot.plot_cn import run as run_plot_cn

    run_plot_cn(
        {
            "bbc": const.BEST_BBC_UCN(result_dir),
            "seg": const.BEST_SEG_UCN(result_dir),
            "gamma_file": const.GAMMA_FILE(result_dir),
            "solfile": None,
            "plot_dir": plot_dir,
            "genome_size": genome_size,
            "region_bed": region_bed,
            "ploidy": "diploid",
            "verbosity": verbosity,
        }
    )


def check_cluster_bins(bbc_dir):
    """Assertions on cluster-bins output; returns a list of (name, passed, detail)."""
    bbc_file = const.BULK_BBC(bbc_dir, False)
    seg_file = const.BULK_SEG(bbc_dir, False)
    scores_file = os.path.join(bbc_dir, const.MODEL_SCORES_TSV)

    results = [
        (f"cluster-bins wrote {os.path.basename(p)}", os.path.isfile(p), p)
        for p in (bbc_file, seg_file, scores_file)
    ]
    if not all(ok for _, ok, _ in results):
        return results

    bbc = pd.read_table(bbc_file, sep="\t")
    n_clusters = bbc["CLUSTER"].nunique()
    results.append(
        (
            "cluster-bins found distinct clusters",
            n_clusters >= MIN_CLUSTERS,
            f"{n_clusters} clusters (expected >= {MIN_CLUSTERS})",
        )
    )

    medians = bbc.groupby("CLUSTER")["RD"].median()
    rdr_range = float(medians.max() - medians.min())
    results.append(
        (
            "cluster-bins separated RDR levels",
            rdr_range > MIN_CLUSTER_RDR_RANGE,
            f"cluster RDR range {rdr_range:.3f} (expected > {MIN_CLUSTER_RDR_RANGE})",
        )
    )
    return results


def check_compute_cn(result_dir, ground_truth):
    """Assertions on compute-cn output against the simulated truth."""
    bbc_ucn = const.BEST_BBC_UCN(result_dir)
    seg_ucn = const.BEST_SEG_UCN(result_dir)
    gamma_file = const.GAMMA_FILE(result_dir)

    results = [
        (f"compute-cn wrote {os.path.basename(p)}", os.path.isfile(p), p)
        for p in (bbc_ucn, seg_ucn, gamma_file)
    ]
    if not all(ok for _, ok, _ in results):
        return results

    bbc = pd.read_table(bbc_ucn, sep="\t")
    results.append(
        (
            "normal clone is 1|1 everywhere",
            bool((bbc["cn_normal"] == "1|1").all()),
            f"{int((bbc['cn_normal'] != '1|1').sum())} bins deviate",
        )
    )

    u_cols = [c for c in bbc.columns if c.startswith("u_")]
    prop_sums = bbc[u_cols].to_numpy().sum(axis=1)
    results.append(
        (
            "clone proportions sum to 1",
            bool(np.allclose(prop_sums, 1.0, atol=0.05)),
            f"range [{prop_sums.min():.3f}, {prop_sums.max():.3f}]",
        )
    )

    purity = float(bbc[[c for c in u_cols if c != "u_normal"]].iloc[0].sum())
    expected_purity = ground_truth["purity"]
    results.append(
        (
            "tumor purity recovered",
            abs(purity - expected_purity) < PURITY_TOL,
            f"{purity:.3f} (truth {expected_purity:.3f}, tol {PURITY_TOL})",
        )
    )

    gammas = pd.read_table(
        gamma_file, sep="\t", header=None, names=["sample", "diploid", "tetraploid"]
    )
    gamma_dip = float(gammas["diploid"].iloc[0])
    expected_gamma = ground_truth["gamma"]
    results.append(
        (
            "RDR scaling (gamma) recovered",
            abs(gamma_dip - expected_gamma) < GAMMA_TOL,
            f"{gamma_dip:.3f} (truth {expected_gamma:.3f}, tol {GAMMA_TOL})",
        )
    )

    results.extend(_check_region_cn(bbc, ground_truth))
    return results


def _check_region_cn(bbc, ground_truth):
    """Per-region modal tumor CN against the simulated profile (allele order ignored)."""
    results = []
    for chrom, start, end, name in ground_truth["regions"]:
        expected = _major_minor(*ground_truth["cn_profile"][name])
        bins = bbc[
            (bbc["#CHR"] == chrom) & (bbc["START"] >= start) & (bbc["END"] <= end)
        ]
        if bins.empty:
            results.append((f"{name} CN recovered", False, "no bins in region"))
            continue
        states = bins["cn_clone1"].map(_parse_cn)
        frac = float((states == expected).mean())
        results.append(
            (
                f"{name} CN recovered",
                frac > REGION_CN_FRAC,
                f"{frac:.0%} of {len(bins)} bins are {expected[0]}|{expected[1]}",
            )
        )
    return results


def check_plot_cn(plot_dir, sample_id, img_type):
    """Assertions that plot-cn wrote its three figures."""
    names = [
        f"{sample_id}.1D.{img_type}",
        f"{sample_id}.1D.FCN_AB.{img_type}",
        f"{sample_id}.2D.{img_type}",
    ]
    return [
        (
            f"plot-cn wrote {name}",
            os.path.isfile(os.path.join(plot_dir, name)),
            os.path.join(plot_dir, name),
        )
        for name in names
    ]


def report(results):
    """Print one line per assertion to stdout; returns the number of failures.

    Printed rather than logged: the table is the command's output, so it must
    appear at the default verbosity, which silences INFO logging.
    """
    width = max(len(name) for name, _, _ in results)
    print(f"\n{len(results)} assertions:")
    for name, ok, detail in results:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name.ljust(width)}  {detail}")
    return sum(1 for _, ok, _ in results if not ok)


def run(args=None):
    """Main entry point for the check step.

    Simulates the input, runs the three pipeline stages on it, and reports
    per-assertion PASS/FAIL. The simulated input and every output go to
    ``--out_dir`` when given, otherwise to a temporary directory under
    ``--tmpdir`` that is removed on success and kept on failure.

    Args:
        args: dict or Namespace of CLI arguments (see hatchet_parser.py).

    Raises:
        SystemExit: if any stage raises or any assertion fails.
    """
    args = normalize_args(args)
    setup_logging(args)
    logging.info("run hatchet check")
    _log_done = log_step_start()

    matplotlib.use("Agg")

    verbosity = args["verbosity"]
    try:
        solver = resolve_solver(args["solver"])
    except RuntimeError as e:
        raise SystemExit(f"hatchet check FAILED ({e})")
    logging.info(f"solver: {solver}")

    out_dir = args["out_dir"]
    is_temp = out_dir is None
    tmpdir = args["tmpdir"]
    if tmpdir is not None and not os.path.isdir(tmpdir):
        raise SystemExit(
            f"hatchet check FAILED (--tmpdir is not a directory: {tmpdir})"
        )
    if tmpdir is not None and not is_temp:
        logging.warning("--tmpdir ignored because --out_dir was given")
    out_dir = os.path.abspath(
        tempfile.mkdtemp(prefix="hatchet-check-", dir=tmpdir) if is_temp else out_dir
    )
    bbc_dir = os.path.join(out_dir, "bbc")
    result_dir = os.path.join(out_dir, "results")
    plot_dir = os.path.join(out_dir, "plots")
    logging.info(f"working directory: {out_dir}")

    failures = 0
    stage = "simulate"
    results = []
    try:
        bb_dir, genome_size, region_bed, ground_truth = simulate_bb_dir(
            os.path.join(out_dir, "input")
        )
        logging.info(f"simulated input: {bb_dir}")

        stage = "cluster-bins"
        logging.info("[1/3] cluster-bins")
        run_cluster_bins_stage(bb_dir, bbc_dir, genome_size, region_bed, verbosity)
        results = check_cluster_bins(bbc_dir)

        stage = "compute-cn"
        logging.info("[2/3] compute-cn")
        run_compute_cn_stage(
            const.BULK_BBC(bbc_dir, False),
            const.BULK_SEG(bbc_dir, False),
            result_dir,
            genome_size,
            region_bed,
            solver,
            verbosity,
        )
        results += check_compute_cn(result_dir, ground_truth)

        stage = "plot-cn"
        logging.info("[3/3] plot-cn")
        run_plot_cn_stage(result_dir, plot_dir, genome_size, region_bed, verbosity)
        results += check_plot_cn(plot_dir, args["sample_id"], args["plot_img_type"])

        failures = report(results)
        summary = f"{failures} failing assertions"
    except Exception:
        traceback.print_exc()
        if results:
            report(results)
        failures = 1
        summary = f"{stage} raised an exception"
    finally:
        if is_temp and failures == 0:
            shutil.rmtree(out_dir, ignore_errors=True)
        elif is_temp:
            print(f"outputs kept for inspection: {out_dir}")

    _log_done("check")
    if failures:
        raise SystemExit(f"\nhatchet check FAILED ({summary})")
    print(f"\nhatchet check PASSED{'' if is_temp else f' (outputs: {out_dir})'}")

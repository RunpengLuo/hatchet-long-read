import os
import logging
import shutil

import numpy as np
import pandas as pd

from hatchet.utils import (
    add_file_logging,
    log_arguments,
    log_step_start,
    normalize_args,
    setup_logging,
)
from hatchet.io_utils import read_bbc_file, read_seg_file
from hatchet import const
from hatchet.compute_cn.compute_cn_utils import (
    store_gammas,
    store_solve_input,
    store_instance_tofile,
    update_objectives_tsv,
    compute_fractional_cn,
    load_pool_from_disk,
    segmentation,
)
from hatchet.compute_cn.model_select import (
    model_selection_ploidy,
    model_select_elbow_from_regularization,
)
from hatchet.compute_cn.scaling import get_scaling_factor
from hatchet.hatchet_parser import parse_arguments_compute_cn
from hatchet.compute_cn.solve.datatypes import SolverParams, SolverInputs
from hatchet.compute_cn.solve.inference import (
    run_full_ilp,
    run_coordinate_descent,
)
from hatchet.plot.plot_compute_cn import (
    plot_pareto_curve,
    plot_pool_cnp,
    plot_scaling_2d,
    run_plot_cn,
)


def run(args=None):
    args = parse_arguments_compute_cn(normalize_args(args))
    setup_logging(args)
    logging.info("run hatchet compute cn")
    _log_done = log_step_start()

    bbc_file = args["bbc"]
    seg_file = args["seg"]
    out_dir = args["result_dir"]
    os.makedirs(out_dir, exist_ok=True)
    add_file_logging(out_dir, "compute-cn")
    log_arguments(args)
    plot_dir = const.PLOTS_DIR(out_dir)
    sols_dir = os.path.join(out_dir, const.SOLS_DIR)
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(sols_dir, exist_ok=True)

    bins, samples, rd_mat, cov_mat, baf_mat, alpha_mat, beta_mat = read_bbc_file(
        bbc_file, is_wide_format=args["wide_format"]
    )
    (
        clusters,
        seg_samples,
        seg_rdr,
        seg_baf,
        seg_rdr_se,
        seg_baf_se,
        seg_rdr_var,
        seg_baf_tau,
        seg_nbins,
        weights,
        is_balanced,
        is_filtered,
    ) = read_seg_file(seg_file)
    assert seg_samples == samples, "BBC and SEG sample order mismatch"

    # Remove clusters marked as filtered by cluster-bins
    if is_filtered.any():
        filtered_ids = [c for c, f in zip(clusters, is_filtered) if f]
        logging.info(f"Excluding filtered clusters from seg: {filtered_ids}")
        keep_seg = ~is_filtered
        clusters = [c for c, k in zip(clusters, keep_seg) if k]
        (
            seg_rdr,
            seg_baf,
            seg_rdr_se,
            seg_baf_se,
            seg_rdr_var,
            seg_baf_tau,
            seg_nbins,
        ) = (
            seg_rdr[keep_seg],
            seg_baf[keep_seg],
            seg_rdr_se[keep_seg],
            seg_baf_se[keep_seg],
            seg_rdr_var[keep_seg],
            seg_baf_tau[keep_seg],
            seg_nbins[keep_seg],
        )
        is_balanced = is_balanced[keep_seg]
        # Renormalize length weights over the retained clusters.
        w = weights[keep_seg]
        weights = 100.0 * w / w.sum()
        keep_bin = ~np.isin(bins["CLUSTER"].to_numpy(), filtered_ids)
        bins = bins[keep_bin].reset_index(drop=True)
        rd_mat, cov_mat, baf_mat, alpha_mat, beta_mat = (
            rd_mat[keep_bin],
            cov_mat[keep_bin],
            baf_mat[keep_bin],
            alpha_mat[keep_bin],
            beta_mat[keep_bin],
        )

    scaling, _balanced_clusters = get_scaling_factor(
        samples,
        clusters,
        bins,
        rd_mat,
        seg_rdr,
        seg_baf,
        seg_baf_se,
        seg_rdr_var,
        seg_nbins,
        is_balanced,
        fix_cn_dip=args["fix_cn_dip"],
        fix_cn_tet=args["fix_cn_tet"],
        maxcn=args["diploidcmax"],
        maxcn_wgd=args["tetraploidcmax"],
    )
    gamma_outfile = const.GAMMA_FILE(out_dir)
    store_gammas(gamma_outfile, scaling, samples)

    plot_scaling_2d(
        samples,
        clusters,
        bins,
        rd_mat,
        baf_mat,
        seg_rdr,
        seg_baf,
        scaling,
        plot_dir,
    )

    solve_mode = args["mode"]
    input_data = {
        "rdr": seg_rdr,
        "baf": seg_baf,
        "rdr_se": seg_rdr_se,
        "baf_se": seg_baf_se,
        "nbins": seg_nbins,
        "weights": pd.Series(weights, index=clusters),
        "cluster_ids": clusters,
        "sample_ids": samples,
    }
    minClone = args["minClone"]
    maxClone = args["maxClone"] + 1

    run_ploidy = {"diploid": args["diploid"], "tetraploid": args["tetraploid"]}
    if not run_ploidy["diploid"] and not run_ploidy["tetraploid"]:
        run_ploidy["diploid"] = True
        run_ploidy["tetraploid"] = True

    # 3. Run solves
    whole_pool = {}
    chosen_sols = {}
    model_selection_df = []
    obj_dfs = []
    for ploidy, run_it in run_ploidy.items():
        if not run_it or scaling[ploidy] is None:
            continue
        logging.info(f"running {ploidy} with n={minClone}..{maxClone}")
        whole_pool[ploidy] = {}
        chosen_sols[ploidy] = {}
        gammas = scaling[ploidy]["gammas"]
        clonals = scaling[ploidy]["clonal"]
        purities = scaling[ploidy]["purities"]
        logging.info(f"{ploidy} clonal CN: {clonals}")
        for sample, gamma in gammas.items():
            logging.info(f"  {sample}\tgamma={gamma}")

        fcn_data = compute_fractional_cn(
            input_data,
            gammas,
            alpha=args["fcn_ci_alpha"],
            min_ci_margin=args["min_ci_margin"],
        )
        store_solve_input(
            const.SOLVER_INPUT(out_dir, ploidy),
            fcn_data,
        )

        for n in range(minClone, maxClone):
            out_bbc = const.RESULTS_BBC_UCN(out_dir, ploidy, n)
            out_seg = const.RESULTS_SEG_UCN(out_dir, ploidy, n)
            sol_dir = const.PLOIDY_N_SUBDIR(
                os.path.join(out_dir, const.SOLS_DIR), ploidy, n
            )
            if (
                not args["force"]
                and os.path.exists(out_bbc)
                and os.path.exists(out_seg)
                and os.path.isdir(sol_dir)
            ):
                logging.info(
                    f"skip {ploidy} n={n}: results already exist (use --force to re-solve)"
                )
                pool_instances = load_pool_from_disk(
                    sol_dir, fcn_data["cluster_ids"], fcn_data["sample_ids"]
                )
            else:
                os.makedirs(sol_dir, exist_ok=True)
                pool_instances, obj_df = solve(
                    n,
                    clonals,
                    args,
                    ploidy,
                    fcn_data,
                    purities,
                    sol_dir,
                    solve_mode=solve_mode,
                )
                obj_dfs.append(obj_df.assign(ploidy=ploidy, n=n))
            whole_pool[ploidy][n] = pool_instances

            selected_id, sel_df = model_select_elbow_from_regularization(pool_instances)
            best_sol = pool_instances[selected_id]
            chosen_sols[ploidy][n] = best_sol
            logging.info(
                f"{ploidy} n={n} selected={selected_id} "
                f"imf={best_sol['imf_obj']:.4f} reg={best_sol['reg_obj']:.1f}"
            )
            sel_df["ploidy"] = ploidy
            sel_df["n_clones"] = n
            model_selection_df.append(sel_df)

            cn_segs = {}
            for sol_id, sol in pool_instances.items():
                bbc_out = out_bbc if sol_id == selected_id else None
                seg_out = out_seg if sol_id == selected_id else None
                cn_segs[sol_id] = segmentation(
                    sol["cA"],
                    sol["cB"],
                    sol["u"],
                    fcn_data,
                    bins=bins,
                    samples=samples,
                    rd_mat=rd_mat,
                    cov_mat=cov_mat,
                    baf_mat=baf_mat,
                    alpha_mat=alpha_mat,
                    beta_mat=beta_mat,
                    region_file=args["region_bed"],
                    bbc_out_file=bbc_out,
                    seg_out_file=seg_out,
                    is_wide_format=args["wide_format"],
                )

            pid = args["patient_id"] or "panel"
            nplot_dir = const.PLOIDY_N_SUBDIR(plot_dir, ploidy, n)
            run_plot_cn(
                args,
                out_bbc,
                out_seg,
                gamma_outfile,
                nplot_dir,
                ploidy,
                name=f"{pid}.{ploidy}_n{n}",
            )

            plot_pool_cnp(
                pool_instances,
                args["genome_size"],
                args["region_bed"],
                nplot_dir,
                sel_df=sel_df,
                segs=cn_segs,
                title=f"{ploidy} n={n}",
                out_name=const.POOL_CNP_PDF(pid, ploidy, n),
            )

    if obj_dfs:
        update_objectives_tsv(sols_dir, pd.concat(obj_dfs, ignore_index=True))

    summary_df = (
        pd.concat(model_selection_df, ignore_index=True)
        if model_selection_df
        else pd.DataFrame()
    )
    summary_path = const.SUMMARY_TSV(out_dir)
    summary_df.to_csv(summary_path, sep="\t", index=False)
    logging.info(f"wrote {summary_path} ({len(summary_df)} solutions)")

    best_ploidy, best_n, chosen_n, elbow_fig = model_selection_ploidy(
        chosen_sols,
        out_dir,
        scaling,
        clusters,
        samples,
        seg_rdr_var,
        seg_baf_tau,
        method=args["model_select"],
    )
    plot_pareto_curve(summary_df, plot_dir, args["reg_term"], elbow_fig)

    # Write chosen per-ploidy
    for ploidy, n in chosen_n.items():
        shutil.copy2(
            const.RESULTS_BBC_UCN(out_dir, ploidy, n),
            const.CHOSEN_BBC_UCN(out_dir, ploidy),
        )
        shutil.copy2(
            const.RESULTS_SEG_UCN(out_dir, ploidy, n),
            const.CHOSEN_SEG_UCN(out_dir, ploidy),
        )
        logging.info(f"chosen {ploidy} n={n}: {const.CHOSEN_BBC_UCN(out_dir, ploidy)}")

    # Write best (across ploidies)
    shutil.copy2(
        const.CHOSEN_BBC_UCN(out_dir, best_ploidy),
        const.BEST_BBC_UCN(out_dir),
    )
    shutil.copy2(
        const.CHOSEN_SEG_UCN(out_dir, best_ploidy),
        const.BEST_SEG_UCN(out_dir),
    )
    logging.info(f"model-selected: {best_ploidy} n={best_n}")
    _log_done("compute-cn")


def solve(
    n: int,
    clonal: dict,
    args: dict,
    ploidy: str,
    input_data: dict,
    purities: dict,
    sol_dir: str,
    solve_mode="ilp",
):
    """Solve for allele-specific integer copy numbers and clone proportions.

    Runs coordinate descent (CD), integer linear programming (ILP), or both
    (CD warm-starting ILP) over a regularization path, selects the best
    instance via Pareto-elbow model selection, and writes BBC/SEG UCN output.

    Returns (pool_instances dict, per-restart objective DataFrame).
    """
    logging.info(f"running {ploidy} with n={n}")
    cn_max = {"diploid": args["diploidcmax"], "tetraploid": args["tetraploidcmax"]}[
        ploidy
    ]
    base = {"diploid": 1, "tetraploid": 2}[ploidy]
    if args["purities"] is not None:
        purities = args["purities"]
        logging.info("user-specified sample purity")
    elif purities is not None:
        logging.info("pre-estimated sample purity")
    else:
        logging.info("infer sample purity directly from deconvolution step")
    if purities is not None:
        for s in sorted(purities):
            logging.info(f"  {s}: {purities[s]:.3f}")

    reg_term = args["reg_term"]
    reg_steps = args["reg_steps"]
    solver_type = args["solver"]
    timelimit = args["timelimit"]
    ampdel = not args["no_ampdel"]

    cd_instances = None
    pool_instances = {}
    u0_tsv_path = const.U0_SEEDS_TSV(sol_dir) if sol_dir is not None else None
    cd_run_kwargs = dict(
        solver_type=solver_type,
        max_iters=args["cd_niters"],
        max_convergence_iters=args["cd_convergence_iters"],
        n_seed=args["cd_nseeds"],
        j=args["cd_njobs"],
        random_seed=args["cd_seed"],
        timelimit=timelimit,
        u0_tsv_path=u0_tsv_path,
    )

    # Build SolverParams shared by both CD and ILP
    weights = input_data["weights"]
    cluster_ids = input_data["cluster_ids"]
    sample_ids = input_data["sample_ids"]
    nbins = input_data["nbins"]

    fixed_rows = set()
    for _m, cid in enumerate(cluster_ids):
        if cid in clonal:
            fixed_rows.add(_m)
    free_rows_list = [_m for _m in range(len(cluster_ids)) if _m not in fixed_rows]

    inputs = SolverInputs(
        f_a=input_data["fa"],
        f_b=input_data["fb"],
        w=weights,
        cluster_ids=cluster_ids,
        sample_ids=sample_ids,
        copy_numbers=clonal,
        free_rows=free_rows_list,
        fixed_rows=fixed_rows,
        purities=purities,
        fa_lo=input_data["fa_lo"],
        fa_hi=input_data["fa_hi"],
        fb_lo=input_data["fb_lo"],
        fb_hi=input_data["fb_hi"],
        nbins=nbins,
    )
    params = SolverParams(
        n=n,
        cn_max=cn_max,
        base=base,
        ampdel=ampdel,
        minprop=args["min_prop"],
        max_ncns_seg=args["num_cnstates"],
        tol=args["tol"],
        zero_cn_thres=args["zero_cn_thres"],
        reg_name=reg_term if reg_term is not None else "RAW",
        obj_type=args["obj_type"],
    )

    if solve_mode in ("cd", "both"):
        cd_instances, obj_df = run_coordinate_descent(
            params=params,
            inputs=inputs,
            reg_steps=reg_steps,
            reg_bound=args["reg_bound"],
            u_init_method=args["u_init"],
            u_dir_alpha=args["u_dir_alpha"],
            solver_threads=args["solver_threads"],
            cd_tol=args["cd_tol"],
            **cd_run_kwargs,
        )
        pool_instances = cd_instances

    if solve_mode in ("ilp", "both"):
        warm_cA = warm_cB = None
        if solve_mode == "both":
            best_cd = min(cd_instances.values(), key=lambda s: s["imf_obj"])
            logging.info(
                f"use CD local opt with obj={best_cd['imf_obj']:.4f} to initialize ILP model"
            )
            warm_cA, warm_cB = best_cd["cA"], best_cd["cB"]

        pool_instances, obj_df = run_full_ilp(
            params=params,
            inputs=inputs,
            reg_steps=reg_steps,
            reg_bound=args["reg_bound"],
            solver_type=solver_type,
            timelimit=timelimit,
            warm_start_cA=warm_cA,
            warm_start_cB=warm_cB,
        )

    store_instance_tofile(pool_instances, input_data, sol_dir, solve_mode)
    return pool_instances, obj_df
